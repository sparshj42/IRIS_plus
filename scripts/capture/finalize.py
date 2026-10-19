"""Turn a fitted capture into a benchmark scene in the same format as the
synthetic generator (scripts/data/gen_layered_scene.py), so eval_layered.py and
the pilot run on it unchanged.

Writes <out>/<scene_id>/: rgb.png (main camera, step 0, undistorted), inst.npy
(GT instance map rendered from the fitted CAD), gt_mesh.ply + face_inst.npy (CAD
at the fitted poses + the table plane), cam.npz, peel/step_XX_{rgb.png,inst.npy}
(the REAL photos after each removal), meta.json (GT occlusion graph and order
from geometry, the operator's actual order and any pair it got backwards, pose
fit quality, disturbance flags), and qa/step_XX.png overlays of the fitted CAD
outlines on the photos: check these before trusting a scene.

Usage: python scripts/capture/finalize.py --rig rig.json --scene captures/scene_001 \
           --library library.json --out data/real --scene_id r001
"""
import argparse, json, os, shutil, sys
import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from common import load_json, load_mesh_m, load_rig, save_json, transform
import eval_layered as E


def write_ply(path, V, F):
    face = np.empty(len(F), dtype=[("n", "u1"), ("i", "<i4", (3,))])
    face["n"], face["i"] = 3, F
    with open(path, "wb") as fh:
        fh.write((f"ply\nformat binary_little_endian 1.0\nelement vertex {len(V)}\n"
                  "property float x\nproperty float y\nproperty float z\n"
                  f"element face {len(F)}\nproperty list uchar int vertex_indices\nend_header\n").encode())
        fh.write(V.astype("<f4").tobytes())
        fh.write(face.tobytes())


def gt_order(edges, nodes, near):
    E_ = dict(edges)
    order, dropped = [], []
    while len(order) < len(nodes):
        rest = [x for x in nodes if x not in order]
        free = [x for x in rest if not any((p, x) in E_ for p in rest if p != x)]
        if not free:
            e = min((e for e in E_ if e[0] in rest and e[1] in rest), key=E_.get)
            dropped.append([int(e[0]), int(e[1]), int(E_.pop(e))])
            continue
        order.append(min(free, key=lambda x: near[x]))
    return order, dropped


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rig", required=True)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--library", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scene_id", required=True)
    ap.add_argument("--main_cam", default="main")
    ap.add_argument("--table_half", type=float, default=1.0, help="half-size (m) of the table plane around the objects")
    a = ap.parse_args()

    rig, lib = load_rig(a.rig), load_json(a.library)
    rep = load_json(os.path.join(a.scene, "session_report.json"))
    poses = load_json(os.path.join(a.scene, "poses.json"))
    removed = rep["removed"]
    cam = rig["cams"][a.main_cam]
    W, H = cam["size"]
    K, c2w = cam["K"], cam["c2w"]
    D = os.path.join(a.out, a.scene_id)
    os.makedirs(os.path.join(D, "peel"), exist_ok=True)
    os.makedirs(os.path.join(D, "qa"), exist_ok=True)

    # ── GT mesh: CAD at fitted poses (instance id = index in removed.txt) + table ──
    parts = []
    for i, name in enumerate(removed):
        V, F = load_mesh_m(lib[name]["mesh"], lib[name].get("units", "mm"))
        parts.append((transform(V, np.array(poses[name]["T_world_cad"])), F, i))
    allV = np.vstack([p[0] for p in parts])
    cx, cy = allV[:, :2].mean(0)
    h = a.table_half
    table_V = np.array([[cx - h, cy - h, 0], [cx + h, cy - h, 0], [cx + h, cy + h, 0], [cx - h, cy + h, 0]])
    parts.append((table_V, np.array([[0, 1, 2], [0, 2, 3]]), -1))

    def build(keep):
        Vs, Fs, Is, off = [], [], [], 0
        for V, F, i in parts:
            if i >= 0 and i not in keep:
                continue
            Vs.append(V); Fs.append(F + off); Is.append(np.full(len(F), i)); off += len(V)
        return np.vstack(Vs), np.vstack(Fs), np.concatenate(Is)

    V, F, inst = build(set(range(len(removed))))
    write_ply(os.path.join(D, "gt_mesh.ply"), V, F)
    np.save(os.path.join(D, "face_inst.npy"), inst)
    np.savez(os.path.join(D, "cam.npz"), K=K, width=W, height=H, c2w=c2w)

    # ── GT occlusion graph + order from geometry, in the main camera ────────────
    w2c = np.linalg.inv(c2w)
    Vc = transform(V, w2c)
    scene = E.make_scene(Vc, F)
    edges, cyclic, _ = E.gt_occlusion_pairs(scene, inst, K, W, H)
    near = {i: float(np.linalg.norm(transform(parts[i][0], w2c), axis=1).min()) for i in range(len(removed))}
    order, dropped = gt_order(edges, list(range(len(removed))), near)
    op = list(range(len(removed)))                       # operator removed them in file order
    backwards = [[int(x), int(y)] for (x, y) in edges if (x, y) not in cyclic and op.index(y) < op.index(x)]

    # ── images: real photos + rendered GT instance maps + QA overlays ───────────
    for k in range(len(removed) + 1):
        src = os.path.join(a.scene, "undist", f"step_{k:02d}", a.main_cam + ".png")
        Vk, Fk, Ik = build(set(range(k, len(removed))))
        imap = E.gt_instance_map(E.make_scene(transform(Vk, w2c), Fk), Ik, K, W, H).astype(np.int16)
        pre = os.path.join(D, "") if k == 0 else os.path.join(D, "peel", f"step_{k:02d}_")
        shutil.copyfile(src, pre + "rgb.png")
        np.save(pre + "inst.npy", imap)
        img = cv2.imread(src)
        for i in np.unique(imap[imap >= 0]):
            cs, _ = cv2.findContours((imap == i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(img, cs, -1, (0, 255, 0), 1)
            ys, xs = np.where(imap == i)
            cv2.putText(img, removed[i], (int(xs.mean()), int(ys.mean())), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        cv2.imwrite(os.path.join(D, "qa", f"step_{k:02d}.png"), img)

    meta = {"scene_id": a.scene_id, "source": "physical capture", "capture_dir": os.path.abspath(a.scene),
            "objects": [{"inst_id": i, "name": n, "peel_step": i + 1, "mean_iou": poses[n]["mean_iou"],
                         "iou": poses[n]["iou"], "refined_6dof": poses[n]["refined_6dof"]}
                        for i, n in enumerate(removed)],
            "gt_edges": [[int(x), int(y), int(c)] for (x, y), c in sorted(edges.items())],
            "gt_cyclic_pairs": sorted([int(x), int(y)] for (x, y) in cyclic),
            "gt_peel_order": [int(x) for x in order], "dropped_for_cycles": dropped,
            "operator_order": op, "operator_pairs_backwards": backwards,
            "disturbance": [{"step": s["step"], "cam": c, **d} for s in rep["steps"]
                            for c, v in s["cams"].items() for d in v["disturbance"]]}
    save_json(os.path.join(D, "meta.json"), meta)
    with open(os.path.join(a.out, "scenes.txt"), "a") as fh:
        ab = lambda x: os.path.abspath(os.path.join(D, x))
        fh.write(f"{a.scene_id} {ab('rgb.png')} {ab('gt_mesh.ply')} {ab('cam.npz')}\n")
    low = [o["name"] for o in meta["objects"] if o["mean_iou"] < 0.85]
    print(f"[{a.scene_id}] {len(removed)} objects, {len(edges)} GT occlusion edges ({len(cyclic)} cyclic), "
          f"operator got {len(backwards)} pair(s) backwards, {len(meta['disturbance'])} disturbance flag(s)"
          + (f", low pose IoU: {low}" if low else "") + f"\n  check overlays in {os.path.join(D, 'qa')}")


if __name__ == "__main__":
    main()
