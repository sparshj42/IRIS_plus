"""Chamfer distance + F-score between a predicted mesh and a GT mesh.

Predicted IRIS meshes live in an arbitrary (gravity-aligned VGGT) frame and are
PARTIAL (only the input view's visible surface). The GT GLB is the full room in
its own world frame/scale. We therefore must align pred -> GT before measuring:

  1. sample points on both surfaces
  2. coarse align with scaled FPFH+RANSAC global registration
  3. refine with scaled point-to-point ICP
  4. accuracy  = pred -> nearest GT   (how correct the recon is)
     completeness = GT -> nearest pred (low by design: pred is single-view)
     F-score @ tau on a normalised scale (GT bbox-diagonal = 1)

Usage: python scripts/eval_chamfer_fscore.py <pred.ply> <gt.glb> [tau]
"""
import sys, numpy as np, open3d as o3d
from scipy.spatial import cKDTree

pred_path, gt_path = sys.argv[1], sys.argv[2]
TAU = float(sys.argv[3]) if len(sys.argv) > 3 else 0.05   # frac of GT bbox-diagonal
N = 200_000


def sample(path):
    m = o3d.io.read_triangle_mesh(path)
    if len(m.triangles) == 0:
        raise SystemExit(f"{path}: no triangles")
    m.compute_vertex_normals()
    return m.sample_points_uniformly(N)


def prep(pcd, voxel):
    d = pcd.voxel_down_sample(voxel)
    d.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(voxel * 2, 30))
    f = o3d.pipelines.registration.compute_fpfh_feature(
        d, o3d.geometry.KDTreeSearchParamHybrid(voxel * 5, 100))
    return d, f


pred = sample(pred_path)
gt = sample(gt_path)
gt_np = np.asarray(gt.points)
diag = float(np.linalg.norm(gt_np.max(0) - gt_np.min(0)))   # GT scale unit
tau = TAU * diag
voxel = diag / 50.0

# ── coarse: scaled FPFH + RANSAC (best of a few seeds) ───────────────────────
ps, pf = prep(pred, voxel)
gs, gf = prep(gt, voxel)
best = None
for seed in range(4):
    o3d.utility.random.seed(seed)
    r = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        ps, gs, pf, gf, False, voxel * 1.5,          # RIGID: pred is already metric
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 3,
        [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
         o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(voxel * 1.5)],
        o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    if best is None or r.fitness > best.fitness:
        best = r

# refine with RIGID ICP (no scaling -> cannot collapse pred to a point)
pred.transform(best.transformation)
icp = o3d.pipelines.registration.registration_icp(
    pred, gt, tau * 3, np.eye(4),
    o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200))
pred.transform(icp.transformation)
print(f"alignment: RANSAC fit {best.fitness:.3f} -> rigid-ICP fit "
      f"{icp.fitness:.3f}  inlier-RMSE {icp.inlier_rmse/diag*100:.2f}% of GT-diag")

# ── metrics (normalised so GT bbox-diagonal = 1) ─────────────────────────────
p = np.asarray(pred.points)
g = gt_np
d_acc = cKDTree(g).query(p)[0]      # pred -> GT
d_comp = cKDTree(p).query(g)[0]     # GT  -> pred
acc, comp = d_acc.mean(), d_comp.mean()
chamfer = acc + comp
prec = (d_acc < tau).mean()
rec = (d_comp < tau).mean()
f = 2 * prec * rec / (prec + rec + 1e-9)

print(f"GT bbox-diagonal = {diag:.3f} (alignment units); tau = {TAU:.0%} of diag")
print(f"  Accuracy     (pred->GT)  {acc/diag*100:6.2f}% of diag   ({acc:.4f})")
print(f"  Completeness (GT->pred)  {comp/diag*100:6.2f}% of diag   ({comp:.4f})")
print(f"  Chamfer (acc+comp)       {chamfer/diag*100:6.2f}% of diag   ({chamfer:.4f})")
print(f"  Precision@tau {prec:.3f}   Recall@tau {rec:.3f}   F-score {f:.3f}")
