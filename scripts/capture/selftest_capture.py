"""End-to-end check of the capture toolkit on a simulated capture with known truth.

Simulates: a ChArUco board photographed by 4 cameras (main + 3 "phones"), then
4 objects resting on the table in a cluster, photographed at every peel step
(ray-cast renders with shading and texture). Runs calib.py extrinsics,
session.py, posefit.py and finalize.py exactly as on real data, then checks:
  camera poses      rotation < 0.5 deg, position < 1% of camera-to-scene range
  object poses      ADD-S < 5 mm (symmetric objects make plain ADD ambiguous)
  exported scene    eval_layered.py scores the TRUE geometry against the exported
                    GT mesh at F > 0.97

Usage: python scripts/capture/selftest_capture.py --objects_dir <GSO dir> [--keep DIR]
"""
import argparse, glob, json, os, shutil, subprocess, sys, tempfile
import cv2
import numpy as np
import open3d as o3d
import trimesh
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import W_FROM_B, make_board, save_json, transform

PY = sys.executable
W, H, FOC = 640, 480, 600.0
K = np.array([[FOC, 0, W / 2], [0, FOC, H / 2], [0, 0, 1]])


def look_at(pos, target):
    z = target - pos; z /= np.linalg.norm(z)
    up = np.array([0, 0, 1.0]) if abs(z[2]) < 0.95 else np.array([0, 1.0, 0])
    x = np.cross(z, up); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    T = np.eye(4); T[:3, :3] = np.stack([x, y, z], 1); T[:3, 3] = pos
    return T


def project(Pw, c2w):
    Pc = transform(Pw, np.linalg.inv(c2w))
    return np.stack([K[0, 0] * Pc[:, 0] / Pc[:, 2] + K[0, 2], K[1, 1] * Pc[:, 1] / Pc[:, 2] + K[1, 2]], 1)


def board_shot(board_img, spec, margin, ppm, c2w, rng):
    h, w = board_img.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    bxy = (src - margin) / ppm                                       # board frame metres (y down)
    world = np.c_[bxy, np.zeros(4)] @ W_FROM_B.T
    dst = project(world, c2w).astype(np.float32)
    Hm = cv2.getPerspectiveTransform(src, dst)
    img = cv2.warpPerspective(cv2.cvtColor(board_img, cv2.COLOR_GRAY2BGR), Hm, (W, H),
                              flags=cv2.INTER_AREA, borderValue=(150, 150, 150))
    img = cv2.GaussianBlur(img, (3, 3), 0).astype(np.float32) + rng.normal(0, 2, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def render(objs, present, c2w, colors):
    sc = o3d.t.geometry.RaycastingScene()
    table = np.array([[-1, -1, 0], [1.4, -1, 0], [1.4, 1, 0], [-1, 1, 0]], np.float32)
    gid_table = sc.add_triangles(o3d.core.Tensor(table), o3d.core.Tensor(np.array([[0, 1, 2], [0, 2, 3]], np.uint32)))
    gid = {}
    for i in present:
        V, F = objs[i]
        gid[sc.add_triangles(o3d.core.Tensor(V.astype(np.float32)), o3d.core.Tensor(F.astype(np.uint32)))] = i
    us, vs = np.meshgrid(np.arange(W) + .5, np.arange(H) + .5)
    d = np.stack([(us - K[0, 2]) / K[0, 0], (vs - K[1, 2]) / K[1, 1], np.ones_like(us)], -1).reshape(-1, 3) @ c2w[:3, :3].T
    o = np.broadcast_to(c2w[:3, 3], d.shape)
    r = sc.cast_rays(o3d.core.Tensor(np.hstack([o, d]).astype(np.float32)))
    t, g, n = r["t_hit"].numpy(), r["geometry_ids"].numpy(), r["primitive_normals"].numpy()
    P = o + d * t[:, None]
    L = np.array([0.3, -0.4, 0.85]); L /= np.linalg.norm(L)
    shade = 0.35 + 0.65 * np.abs(n @ L)
    img = np.full((H * W, 3), 40.0)
    hit = np.isfinite(t)
    tb = hit & (g == gid_table)
    chk = (np.floor(P[tb, 0] / 0.05) + np.floor(P[tb, 1] / 0.05)) % 2
    img[tb] = (170 + 25 * chk)[:, None] * np.array([1.0, 0.97, 0.92])
    for gg, i in gid.items():
        m = hit & (g == gg)
        tex = 0.85 + 0.15 * np.sin(P[m] @ np.array([90.0, 70.0, 110.0]))
        img[m] = colors[i] * (shade[m] * tex)[:, None]
    return np.clip(img.reshape(H, W, 3), 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objects_dir", required=True, help="dir of <name>/meshes/model.obj (e.g. downloaded GSO)")
    ap.add_argument("--keep", default=None, help="keep the simulated capture here")
    a = ap.parse_args()
    rng = np.random.default_rng(3)
    T = a.keep or tempfile.mkdtemp()
    if os.path.exists(T):
        shutil.rmtree(T)
    os.makedirs(T)

    # ── board + cameras ──────────────────────────────────────────────────────
    spec = {"cols": 7, "rows": 5, "square_m": 0.07, "marker_m": 0.05, "dict": "4x4_100"}
    save_json(f"{T}/board.json", spec)
    ppm, margin = 3000.0, 80
    board_img = make_board(spec).generateImage((int(7 * .07 * ppm) + 2 * margin, int(5 * .07 * ppm) + 2 * margin),
                                               marginSize=margin)
    center = np.array([0.18, -0.12, 0.04])
    truth_cams = {"main": look_at(center + [0.02, -0.80, 0.38], center),
                  "phoneA": look_at(center + [0.60, 0.45, 0.45], center),
                  "phoneB": look_at(center + [-0.55, 0.50, 0.50], center),
                  "phoneC": look_at(center + [0.05, 0.10, 0.85], center)}
    save_json(f"{T}/rig.json", {"cams": {c: {"K": K.tolist(), "dist": [0] * 5, "size": [W, H]} for c in truth_cams}})
    os.makedirs(f"{T}/board_shot")
    for c, c2w in truth_cams.items():
        cv2.imwrite(f"{T}/board_shot/{c}.png", board_shot(board_img, spec, margin, ppm, c2w, rng))

    # ── objects: 4 small GSO meshes resting in a cluster ─────────────────────
    lib, objs, truth = {}, [], []
    paths = sorted(glob.glob(os.path.join(a.objects_dir, "*", "meshes", "model.obj")))
    rng.shuffle(paths)
    for p in paths:
        m = o3d.io.read_triangle_mesh(p)
        V, F = np.asarray(m.vertices), np.asarray(m.triangles)
        ext = V.max(0) - V.min(0)
        if ext.max() > 0.22 or ext.max() < 0.05:
            continue
        tm = trimesh.Trimesh(V, F, process=False)
        S, prob = tm.compute_stable_poses(n_samples=4, threshold=0.0)
        Vr = transform(V, S[int(np.argmax(prob))])
        rad = np.linalg.norm(Vr[:, :2], axis=1).max()
        for _ in range(500):
            xy = center[:2] + rng.normal(0, [0.035, 0.09])     # spread along the main camera's line of sight
            if all(np.linalg.norm(xy - q) > rad + rq + 0.005 for q, rq in truth):
                break
        else:
            continue
        yaw = rng.uniform(0, 2 * np.pi)
        Tw = np.eye(4); Tw[:3, :3] = [[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]]
        Tw[:2, 3] = xy
        Tw = Tw @ S[int(np.argmax(prob))]
        name = os.path.basename(os.path.dirname(os.path.dirname(p)))
        lib[name] = {"mesh": p, "units": "m"}
        objs.append((transform(V, Tw), F, name, Tw))
        truth.append((xy, rad))
        if len(objs) == 4:
            break
    save_json(f"{T}/library.json", lib)
    save_json(f"{T}/truth_poses.json", {o[2]: o[3].tolist() for o in objs})
    print(f"{len(objs)} objects placed: {[o[2][:30] for o in objs]}")
    colors = rng.uniform(60, 230, (len(objs), 3))

    # operator removes front-to-back from the main camera's point of view
    cpos = truth_cams["main"][:3, 3]
    order = sorted(range(len(objs)), key=lambda i: np.linalg.norm(objs[i][0] - cpos, axis=1).min())
    scene = f"{T}/scene"
    os.makedirs(scene)
    with open(f"{scene}/removed.txt", "w") as f:
        f.write("\n".join(objs[i][2] for i in order) + "\n")
    geo = [(o[0], o[1]) for o in objs]
    for k in range(len(objs) + 1):
        os.makedirs(f"{scene}/steps/step_{k:02d}")
        present = order[k:]
        for c, c2w in truth_cams.items():
            cv2.imwrite(f"{scene}/steps/step_{k:02d}/{c}.png", render(geo, present, c2w, colors))

    # ── run the toolkit ──────────────────────────────────────────────────────
    def run(*args):
        r = subprocess.run([PY, *args], capture_output=True, text=True)
        print(r.stdout.rstrip())
        if r.returncode:
            print(r.stderr[-3000:]); raise SystemExit(f"FAILED: {args[0]}")

    run(f"{HERE}/calib.py", "extrinsics", "--rig", f"{T}/rig.json", "--board", f"{T}/board.json", "--shot", f"{T}/board_shot")
    rig = json.load(open(f"{T}/rig.json"))
    for c, c2w in truth_cams.items():
        est = np.array(rig["cams"][c]["c2w"])
        dr = np.rad2deg(np.arccos(np.clip((np.trace(est[:3, :3].T @ c2w[:3, :3]) - 1) / 2, -1, 1)))
        dt = np.linalg.norm(est[:3, 3] - c2w[:3, 3]) * 1000
        dist = np.linalg.norm(c2w[:3, 3] - center) * 1000
        print(f"  camera {c}: rotation error {dr:.3f} deg, position error {dt:.2f} mm ({dt / dist * 100:.2f}% of range)")
        assert dr < 0.5 and dt < 0.01 * dist, c
    run(f"{HERE}/session.py", "--rig", f"{T}/rig.json", "--scene", scene)
    run(f"{HERE}/posefit.py", "--rig", f"{T}/rig.json", "--scene", scene, "--library", f"{T}/library.json")
    poses = json.load(open(f"{scene}/poses.json"))
    for V, F, name, Tw in objs:
        mesh_V = np.asarray(o3d.io.read_triangle_mesh(lib[name]["mesh"]).vertices)
        est = transform(mesh_V, np.array(poses[name]["T_world_cad"]))
        add = np.linalg.norm(est - V, axis=1).mean() * 1000
        adds = cKDTree(V).query(est)[0].mean() * 1000
        print(f"  {name[:40]:40s} ADD {add:6.2f} mm  ADD-S {adds:5.2f} mm  IoU {poses[name]['mean_iou']:.3f}")
        assert adds < 5, name
    run(f"{HERE}/finalize.py", "--rig", f"{T}/rig.json", "--scene", scene, "--library", f"{T}/library.json",
        "--out", f"{T}/bench", "--scene_id", "r000")

    # the TRUE geometry (objects + table) scored against the exported GT
    D = f"{T}/bench/r000"
    w2c = np.linalg.inv(truth_cams["main"])
    pts = []
    for V, F, _, _ in objs:
        m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V), o3d.utility.Vector3iVector(F))
        pts.append(np.asarray(m.sample_points_uniformly(60000).points))
    ctr = np.vstack([o[0] for o in objs])[:, :2].mean(0)      # same table extent finalize.py exports
    gx, gy = np.meshgrid(np.linspace(ctr[0] - 1, ctr[0] + 1, 700), np.linspace(ctr[1] - 1, ctr[1] + 1, 700))
    pts.append(np.c_[gx.ravel(), gy.ravel(), np.zeros(gx.size)])
    np.save(f"{T}/true_pts.npy", transform(np.vstack(pts), w2c))
    r = subprocess.run([PY, os.path.join(os.path.dirname(HERE), "eval_layered.py"), "--gt_mesh", f"{D}/gt_mesh.ply",
                        "--gt_cam", f"{D}/cam.npz", "--gt_face_inst", f"{D}/face_inst.npy", "--pred",
                        f"{T}/true_pts.npy", "--align", "none", "--tau", "0.02", "--out", f"{T}/eval.json"],
                       capture_output=True, text=True)
    print(r.stdout.rstrip()[-600:])
    ev = json.load(open(f"{T}/eval.json"))
    meta = json.load(open(f"{D}/meta.json"))
    assert len(meta["gt_edges"]) >= 1, "simulated scene has no occlusion; the occlusion-aware fit went untested"
    assert ev["overall"]["F"] > 0.97 and ev["occluded_hard"]["F"] > 0.9, (ev["overall"], ev["occluded_hard"])
    print(f"\nALL CHECKS PASSED  (capture kept in {T})" if a.keep else "\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
