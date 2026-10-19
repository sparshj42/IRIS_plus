"""Fit each removed object's CAD pose to its removal masks in every camera.

Objects are fitted in REVERSE removal order: when the object removed at step k is
fitted, every object still on the table at step k (removed later) already has a
pose, so the predicted silhouette is z-tested against them. The mask from
session.py is exactly the object's visible part, so the score is the IoU between
that mask and the rendered visible silhouette, averaged over cameras.

Search: objects rest on the table, so each stable resting pose of the CAD
(trimesh) is tried at 24 yaws at the position under the mask's centroid, the best
few are refined over (x, y, yaw) by Nelder-Mead, then a full 6-DoF refinement is
kept only if it raises the IoU (stacked or leaning objects need it).

library.json: {"<name>": {"mesh": "path.stl|obj|ply", "units": "mm"}, ...}
Usage: python scripts/capture/posefit.py --rig rig.json --scene captures/scene_001 --library library.json
Writes <scene>/poses.json.
"""
import argparse, os, sys
import cv2
import numpy as np
import open3d as o3d
import trimesh
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_json, load_mesh_m, load_rig, pixel_rays_world, raycast_depth, save_json, transform


def pose_mat(x, y, z, rx, ry, rz):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("xyz", [rx, ry, rz]).as_matrix()
    T[:3, 3] = [x, y, z]
    return T


class Fitter:
    def __init__(self, V, F, rig, cams, masks, known, main, ds):
        self.scene = o3d.t.geometry.RaycastingScene()
        self.scene.add_triangles(o3d.core.Tensor(V.astype(np.float32)), o3d.core.Tensor(F.astype(np.uint32)))
        self.views = []
        for c in cams:
            m = masks[c]
            if m.sum() < 30:                       # object not visible from this camera
                continue
            cam = rig["cams"][c]
            ys, xs = np.where(m)
            W, H = cam["size"]
            pw, ph = (xs.max() - xs.min()) * 0.6 + 20, (ys.max() - ys.min()) * 0.6 + 20
            box = (int(max(0, xs.min() - pw)), int(max(0, ys.min() - ph)),
                   int(min(W, xs.max() + pw)), int(min(H, ys.max() + ph)))
            o, d, us, vs = pixel_rays_world(cam, ds, box)
            t_known = raycast_depth(known, o, d)[0] if known is not None else np.full(len(o), np.inf)
            target = m[vs.astype(int), us.astype(int)]
            self.views.append({"cam": c, "o": o, "d": d, "t_known": t_known, "target": target,
                               "w": 2.0 if c == main else 1.0})
        if not self.views:
            raise RuntimeError("object visible in no camera")

    def ious(self, T):
        Ti = np.linalg.inv(T)
        out = {}
        for v in self.views:
            o = transform(v["o"], Ti)
            d = v["d"] @ Ti[:3, :3].T
            t = raycast_depth(self.scene, o, d)[0]
            vis = np.isfinite(t) & (t < v["t_known"])
            inter = (vis & v["target"]).sum()
            out[v["cam"]] = inter / max((vis | v["target"]).sum(), 1)
        return out

    def cost(self, T):
        iou = self.ious(T)
        return 1 - sum(v["w"] * iou[v["cam"]] for v in self.views) / sum(v["w"] for v in self.views)

    def centroid_offset(self, T, cam_name, rig):
        """World-xy shift that moves the rendered silhouette's centroid onto the mask's
        centroid in one camera (both measured on the table-parallel plane through the
        object). None if either is empty."""
        v = next((v for v in self.views if v["cam"] == cam_name), None)
        if v is None:
            return None
        Ti = np.linalg.inv(T)
        t = raycast_depth(self.scene, transform(v["o"], Ti), v["d"] @ Ti[:3, :3].T)[0]
        vis = np.isfinite(t) & (t < v["t_known"])
        if vis.sum() < 10 or v["target"].sum() < 10:
            return None
        z = T[2, 3] if T[2, 3] > 0 else 0.0
        def hit(sel):
            o, d = v["o"][sel].mean(0), v["d"][sel].mean(0)
            return (o + d * (z - o[2]) / d[2])[:2]
        return hit(v["target"]) - hit(vis)


def fit_object(V, F, fitter, rig, main_cam, mask_main, n_stable):
    tm = trimesh.Trimesh(V, F, process=False)
    S, prob = tm.compute_stable_poses(n_samples=4, threshold=0.0)
    keep = np.argsort(-prob)[:n_stable]
    S, prob = S[keep], prob[keep]

    # position under the main-camera mask centroid, at half the resting height
    cam = rig["cams"][main_cam]
    ys, xs = np.where(mask_main)
    K, c2w = cam["K"], cam["c2w"]
    ray = c2w[:3, :3] @ np.array([(xs.mean() - K[0, 2]) / K[0, 0], (ys.mean() - K[1, 2]) / K[1, 1], 1.0])
    cands = []
    for si, Ss in enumerate(S):
        h = transform(V, Ss)[:, 2].max()
        s = (h / 2 - c2w[2, 3]) / ray[2]
        x0, y0 = (c2w[:3, 3] + s * ray)[:2]
        for yaw in np.deg2rad(np.arange(0, 360, 15)):
            # move each candidate under the mask before scoring it, so orientations
            # are compared at their own best position, not at one shared guess
            x, y = x0, y0
            for _ in range(3):
                off = fitter.centroid_offset(pose_mat(x, y, 0, 0, 0, yaw) @ Ss, main_cam, rig)
                if off is None:
                    break
                x, y = x + off[0], y + off[1]
            cands.append((fitter.cost(pose_mat(x, y, 0, 0, 0, yaw) @ Ss), si, x, y, yaw))
    cands.sort(key=lambda c: c[0])

    best = None
    for c0, si, x0, y0, yaw in cands[:6]:
        f3 = lambda p: fitter.cost(pose_mat(p[0], p[1], 0, 0, 0, p[2]) @ S[si])
        r = minimize(f3, [x0, y0, yaw], method="Nelder-Mead",
                     options={"initial_simplex": np.array([[x0, y0, yaw], [x0 + .01, y0, yaw], [x0, y0 + .01, yaw],
                                                           [x0, y0, yaw + .15]]), "maxiter": 200, "xatol": 2e-4,
                              "fatol": 1e-4})
        if best is None or r.fun < best[0]:
            best = (r.fun, si, r.x)
    c3, si, (x, y, yaw) = best
    T3 = pose_mat(x, y, 0, 0, 0, yaw) @ S[si]
    f6 = lambda p: fitter.cost(pose_mat(*p) @ T3)
    r6 = minimize(f6, np.zeros(6), method="Powell",
                  options={"maxiter": 3000, "xtol": 1e-4, "ftol": 1e-5,
                           "direc": np.diag([.005, .005, .005, .03, .03, .03])})
    T6 = pose_mat(*r6.x) @ T3
    use6 = r6.fun < c3 - 1e-3
    T = T6 if use6 else T3
    return T, {"stable_pose": int(si), "stable_prob": float(prob[si]), "refined_6dof": bool(use6),
               "cost_3dof": float(c3), "cost_6dof": float(r6.fun)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rig", required=True)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--library", required=True)
    ap.add_argument("--main_cam", default="main")
    ap.add_argument("--ds", type=int, default=2, help="pixel stride for silhouette rendering")
    ap.add_argument("--n_stable", type=int, default=4)
    a = ap.parse_args()

    rig = load_rig(a.rig)
    lib = load_json(a.library)
    rep = load_json(os.path.join(a.scene, "session_report.json"))
    cams, removed = rep["cams"], rep["removed"]
    for c in cams:
        rig["cams"][c]["size"] = list(rig["cams"][c]["size"])

    poses, placed = {}, []                     # placed: (V_world, F) of objects already fitted
    for k in reversed(range(len(removed))):
        name = removed[k]
        V, F = load_mesh_m(lib[name]["mesh"], lib[name].get("units", "mm"))
        masks = {c: cv2.imread(os.path.join(a.scene, "masks", f"step_{k:02d}", c + ".png"), 0) > 127 for c in cams}
        known = None
        if placed:
            known = o3d.t.geometry.RaycastingScene()
            for Vw, Fw in placed:
                known.add_triangles(o3d.core.Tensor(Vw.astype(np.float32)), o3d.core.Tensor(Fw.astype(np.uint32)))
        fitter = Fitter(V, F, rig, cams, masks, known, a.main_cam, a.ds)
        T, info = fit_object(V, F, fitter, rig, a.main_cam, masks[a.main_cam], a.n_stable)
        iou = fitter.ious(T)
        poses[name] = {"step": k, "T_world_cad": T.tolist(), "iou": {c: float(v) for c, v in iou.items()},
                       "mean_iou": float(np.mean(list(iou.values()))), **info}
        placed.append((transform(V, T), F))
        flag = "" if poses[name]["mean_iou"] >= 0.85 else "   <-- low IoU: check masks / overlay"
        print(f"step {k} {name}: IoU " + "  ".join(f"{c} {v:.3f}" for c, v in iou.items())
              + f"  ({'6-DoF' if info['refined_6dof'] else 'resting'}){flag}")
    save_json(os.path.join(a.scene, "poses.json"), poses)


if __name__ == "__main__":
    main()
