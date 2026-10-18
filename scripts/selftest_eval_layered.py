"""Sanity checks for eval_layered.py on a synthetic scene with known answers.

Scene: floor + back wall (background) and three boxes A, B, C stacked in depth
behind each other along the camera axis, so A hides part of B and both hide part
of C. Checks:
  1. perfect prediction, given at the wrong scale and slightly rotated  -> F ~ 1 at
     every layer and peel depth, recovered scale ~ 1/applied scale
  2. visible-surface-only prediction (what a depth map gives)           -> visible
     recall ~ 1, hard-occluded recall ~ 0 (plain occluded recall is lenient here)
  3. peel order A,B,C has no violations; C,B,A violates every scored pair

Usage: python scripts/selftest_eval_layered.py
"""
import os, sys, tempfile, types
import numpy as np
import open3d as o3d

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_layered as E


def box(x0, x1, y0, y1, z0, z1):
    m = o3d.geometry.TriangleMesh.create_box(x1 - x0, y1 - y0, z1 - z0)
    m.translate((x0, y0, z0))
    return np.asarray(m.vertices), np.asarray(m.triangles)


def build(tmp):
    # world: y up; camera at (0, 1, 0) looking down -z (OpenCV c2w = rot_x(180deg))
    parts = [(box(-3, 3, -0.05, 0.0, -7, 0.5), -1),       # floor
             (box(-3, 3, 0.0, 3.0, -7.05, -7.0), -1),     # back wall
             (box(-0.3, 0.3, 0.0, 0.8, -2.5, -2.0), 0),   # A (front)
             (box(-0.2, 0.6, 0.0, 1.0, -3.5, -3.0), 1),   # B
             (box(-0.4, 0.4, 0.0, 1.2, -4.5, -4.0), 2)]   # C (back)
    V, F, inst, off = [], [], [], 0
    for (v, f), i in parts:
        V.append(v); F.append(f + off); inst += [i] * len(f); off += len(v)
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(np.vstack(V)),
                                     o3d.utility.Vector3iVector(np.vstack(F)))
    o3d.io.write_triangle_mesh(f"{tmp}/gt.ply", mesh)
    np.save(f"{tmp}/face_inst.npy", np.array(inst))
    W, H = 640, 480
    K = np.array([[500.0, 0, W / 2], [0, 500.0, H / 2], [0, 0, 1]])
    c2w = np.array([[1, 0, 0, 0], [0, -1, 0, 1.0], [0, 0, -1, 0], [0, 0, 0, 1]], float)
    np.savez(f"{tmp}/cam.npz", K=K, width=W, height=H, c2w=c2w)
    return mesh, K, W, H, c2w


def args(tmp, pred, **kw):
    a = dict(gt_mesh=f"{tmp}/gt.ply", gt_cam=f"{tmp}/cam.npz", gl_cam=False,
             gt_face_inst=f"{tmp}/face_inst.npy", pred=pred, pred_w2c=None,
             pred_order_masks=None, align="sim3", tau=0.05, taus=[0.02], kmax=3,
             n_gt=120_000, n_pred=120_000, seed=0, out=None)
    a.update(kw)
    return types.SimpleNamespace(**a)


def main():
    tmp = tempfile.mkdtemp()
    mesh, K, W, H, c2w = build(tmp)
    w2c = np.linalg.inv(c2w)
    rng = np.random.default_rng(1)
    full = np.asarray(mesh.sample_points_uniformly(300_000).points)
    full_cam = (w2c[:3, :3] @ full.T).T + w2c[:3, 3]

    # 1. perfect geometry, wrong scale, slightly rotated
    s, ang = 2.7, np.deg2rad(2.0)
    R = np.array([[np.cos(ang), 0, np.sin(ang)], [0, 1, 0], [-np.sin(ang), 0, np.cos(ang)]])
    np.save(f"{tmp}/pred_full.npy", s * (R @ full_cam.T).T)
    r = E.evaluate(args(tmp, f"{tmp}/pred_full.npy"))
    print("perfect   :", {k: round(m["F"], 3) for k, m in r["layer"].items()},
          "| peel", {k: round(m["F"], 3) for k, m in r["peel"].items()},
          "| scale", round(r["align"]["scale"], 4))
    assert abs(r["align"]["scale"] - 1 / s) < 0.01 / s, r["align"]
    assert all(m["F"] > 0.97 for m in r["layer"].values() if m["n_gt"] > 500), r["layer"]
    assert all(m["F"] > 0.97 for m in r["peel"].values() if m["n_gt"] > 500), r["peel"]
    assert r["occluded_hard"]["F"] > 0.97, r["occluded_hard"]
    assert r["layer"]["1"]["n_gt"] > 1000 and r["peel"]["2"]["n_gt"] > 100, "scene lacks deep occlusion"

    # 2. visible surface only
    G, _ = E.sample_mesh(np.asarray(mesh.vertices) @ w2c[:3, :3].T + w2c[:3, 3],
                         np.asarray(mesh.triangles), 300_000, rng)
    G = G[E.in_frustum(G, K, W, H)]
    sc = E.make_scene(np.asarray(mesh.vertices) @ w2c[:3, :3].T + w2c[:3, 3], np.asarray(mesh.triangles))
    ray, t, prim, eps_rel = E.front_hits(sc, G, 1e-3 * np.linalg.norm(G.max(0) - G.min(0)))
    vis = E.layer_depth(ray, t, eps_rel, len(G)) == 0
    np.save(f"{tmp}/pred_vis.npy", 1.9 * G[vis])
    r = E.evaluate(args(tmp, f"{tmp}/pred_vis.npy"))
    print("visible   : vis R", round(r["visible"]["R"], 3), "| occ R (lenient)", round(r["occluded"]["R"], 3),
          "| hard-occ R", round(r["occluded_hard"]["R"], 3), "| ODC-AUC occ", round(r["odc_auc_occluded"], 3))
    assert r["visible"]["R"] > 0.97 and r["occluded_hard"]["R"] < 0.05, (r["visible"], r["occluded_hard"])
    assert r["occluded_hard"]["n_gt"] > 1000, "too few hard-occluded points to test"

    # 3. peel-order scoring
    inst_map = E.gt_instance_map(E.make_scene(np.asarray(mesh.vertices) @ w2c[:3, :3].T + w2c[:3, 3],
                                              np.asarray(mesh.triangles)),
                                 np.load(f"{tmp}/face_inst.npy"), K, W, H)
    for name, order in (("ABC", [0, 1, 2]), ("CBA", [2, 1, 0])):
        np.savez(f"{tmp}/order_{name}.npz", masks=np.stack([inst_map == i for i in order]))
        r = E.evaluate(args(tmp, f"{tmp}/pred_full.npy", pred_order_masks=f"{tmp}/order_{name}.npz"))
        print(f"order {name} :", r["ordering"])
        want = 0.0 if name == "ABC" else 1.0
        assert r["ordering"]["pairs_scored"] >= 2 and abs(r["ordering"]["violation_rate"] - want) < 1e-9
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
