"""Benchmark an IRIS reconstruction against ScanNet GT — honest split:
  - VISIBLE F1 / accuracy : observed surface vs GT-visible       (reconstruction quality)
  - OCCLUDED recall       : completions vs GT occluded-in-view    (occlusion recovery)

Alignment (recon -> GT world frame) is POSE-ANCHORED: with RGB-D input we already
know the sensor camera pose (c2w) and VGGT's per-frame extrinsics, so the recon is
seated by  T = c2w @ extrinsics0  and refined with point-to-plane ICP. This removes
the FPFH "global registration lottery" that otherwise fails on low-overlap frames.
FPFH is kept only as a fallback when pose-init does not converge. IMPORTANT: the
sensor pose is part of the INPUT; the GT mesh is used ONLY for scoring, never for
alignment. With RGB-D the recon is metric, so the cm numbers are meaningful.

Usage: python scripts/benchmark_scannet.py <iris_out> <scannet_scene_dir> <gt_geom_dir> [tau_m]
"""
import sys, glob, numpy as np, open3d as o3d
from scipy.spatial import cKDTree

iris_dir, scene_dir, gt_dir = sys.argv[1], sys.argv[2], sys.argv[3]
TAU = float(sys.argv[4]) if len(sys.argv) > 4 else 0.05


def pcd(p):
    q = o3d.geometry.PointCloud(); q.points = o3d.utility.Vector3dVector(p); return q


# ── GT: visible (frame depth), full mesh, and occluded-in-view ──────────────
depth = np.load(f"{gt_dir}/depth.npy"); intr = np.load(f"{gt_dir}/intr_depth.npy")
c2w = np.load(f"{gt_dir}/c2w.npy"); w2c = np.linalg.inv(c2w)
H, W = depth.shape; fx, fy, cx, cy = intr[0, 0], intr[1, 1], intr[0, 2], intr[1, 2]
ys, xs = np.where(depth > 0); dd = depth[ys, xs]
gt_vis = (c2w[:3, :3] @ np.stack([(xs-cx)/fx*dd, (ys-cy)/fy*dd, dd], 1).T).T + c2w[:3, 3]
gt_full = np.asarray(o3d.io.read_triangle_mesh(
    glob.glob(f"{scene_dir}/*_vh_clean_2.ply")[0]).sample_points_uniformly(400000).points)

# occluded-in-view = GT points that fall inside the frame but lie BEHIND the
# visible depth at their pixel (i.e. hidden by a foreground surface).
gcam = (w2c[:3, :3] @ gt_full.T).T + w2c[:3, 3]; z = gcam[:, 2]
u = (gcam[:, 0] / np.clip(z, 1e-6, None) * fx + cx)
v = (gcam[:, 1] / np.clip(z, 1e-6, None) * fy + cy)
inv = (z > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
ui = np.clip(u, 0, W-1).astype(int); vi = np.clip(v, 0, H-1).astype(int)
vis_d = depth[vi, ui]
occ = inv & (vis_d > 0) & (z > vis_d + 0.05)        # >5cm behind the visible surface
gt_occ = gt_full[occ]
print(f"GT: visible {len(gt_vis)} | full {len(gt_full)} | occluded-in-view {len(gt_occ)}")


def icp_refine(src_np, T0):
    s, t = pcd(src_np), pcd(gt_vis)
    for p in (s, t):
        p.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.08, max_nn=30))
    return o3d.pipelines.registration.registration_icp(
        s, t, 0.10, T0, o3d.pipelines.registration.TransformationEstimationPointToPlane())


def fpfh_align(src_np):  # fallback: FPFH global -> point-to-plane ICP
    s, t = pcd(src_np), pcd(gt_vis); vv = 0.04
    sd, td = s.voxel_down_sample(vv), t.voxel_down_sample(vv)
    for p in (sd, td, s, t):
        p.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=vv*2, max_nn=30))
    fs, ft = (o3d.pipelines.registration.compute_fpfh_feature(
        p, o3d.geometry.KDTreeSearchParamHybrid(radius=vv*5, max_nn=100)) for p in (sd, td))
    best = None
    for _seed in range(3):
        # DETERMINISM: seed the RANSAC so repeated scoring of the SAME artifacts
        # returns the same transform. This removes the registration lottery as a
        # noise source; it does NOT make the estimate better, only reproducible.
        o3d.utility.random.seed(_seed)
        r = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            sd, td, fs, ft, True, vv*1.5,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 3,
            [o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(vv*1.5)],
            o3d.pipelines.registration.RANSACConvergenceCriteria(1000000, 2000))
        ic = o3d.pipelines.registration.registration_icp(
            s, t, 0.10, r.transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPlane())
        if best is None or ic.fitness > best.fitness:
            best = ic
    return best


# ── observed cloud + POSE-ANCHORED alignment ────────────────────────────────
# Prefer the pre-gravity VGGT scene cloud + known camera pose (robust). Fall back
# to the gravity-aligned observed cloud with FPFH if those artifacts are absent.
have_pose = glob.glob(f"{iris_dir}/scene_pointcloud.npy") and glob.glob(f"{iris_dir}/vggt_pointmaps.npz")
if have_pose:
    obs = np.load(f"{iris_dir}/scene_pointcloud.npy")[:, :3]
    extr0 = np.load(f"{iris_dir}/vggt_pointmaps.npz")["extrinsics"][0]   # (3,4) VGGT world2cam
    T0 = np.eye(4); T0[:3, :4] = extr0
    # Seat via the known camera pose + ICP refine — deterministic, no global-
    # registration lottery. FPFH is a fallback only if pose-init fails to converge.
    a = icp_refine(obs, c2w @ T0); method = "pose-anchored (known cam pose)"
    if a.fitness < 0.4:
        b = fpfh_align(obs)
        if b.fitness > a.fitness:
            a, method = b, "FPFH fallback"
else:
    obs = np.load(f"{iris_dir}/observed_pointcloud.npy")[:, :3]
    a = fpfh_align(obs); method = "FPFH (no pose artifacts)"

T = a.transformation
def xf(P): return (T[:3, :3] @ P.T).T + T[:3, 3]
obs_w = xf(obs)
print(f"alignment ({method}): fitness {a.fitness:.3f}  RMSE {a.inlier_rmse*100:.1f} cm")

# ── VISIBLE region (reconstruction quality) ─────────────────────────────────
d_acc = cKDTree(gt_full).query(obs_w)[0]; d_comp = cKDTree(obs_w).query(gt_vis)[0]
P = (d_acc < TAU).mean(); R = (d_comp < TAU).mean(); F1 = 2*P*R/(P+R+1e-9)
print("\n=== VISIBLE region (observed surface vs GT) ===")
print(f"  Reconstruction accuracy : {d_acc.mean()*100:.1f} cm mean ({np.median(d_acc)*100:.1f} cm median)")
print(f"  Precision@{int(TAU*100)}cm {P:.3f} | Recall@{int(TAU*100)}cm {R:.3f} | F1 {F1:.3f}")

# ── OCCLUDED recall (occlusion recovery) ────────────────────────────────────
# completions are in the gravity-aligned frame (with observed_pointcloud), so this
# block aligns that pair independently (FPFH) — self-contained from the visible F1.
print("\n=== OCCLUDED recovery (completions vs GT occluded-in-view) ===")
comp = np.load(f"{iris_dir}/completion_pointcloud.npy") if glob.glob(f"{iris_dir}/completion_pointcloud.npy") else None
if comp is not None and len(gt_occ):
    obs_g = np.load(f"{iris_dir}/observed_pointcloud.npy")[:, :3]
    ag = fpfh_align(obs_g); Tg = ag.transformation
    def xg(Pp): return (Tg[:3, :3] @ Pp.T).T + Tg[:3, 3]
    comp_w, obs_gw = xg(comp[:, :3]), xg(obs_g)
    rec_occ = (cKDTree(comp_w).query(gt_occ)[0] < TAU).mean()
    rec_occ_obs = (cKDTree(obs_gw).query(gt_occ)[0] < TAU).mean()   # observed-only baseline ~0
    print(f"  Occluded recall @{int(TAU*100)}cm : {rec_occ:.3f}   (observed-only baseline {rec_occ_obs:.3f})")
    print(f"  -> image-to-3D recovers {100*(rec_occ-rec_occ_obs):.1f}% more occluded surface")
else:
    print("  no completions (skip_3d) -> occluded recall not applicable")
