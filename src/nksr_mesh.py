"""Object-wise NKSR meshing worker (runs in the `nksr` conda env).

Neural Kernel Surface Reconstruction gives smoother, watertight object meshes than
screened Poisson (no density-trim holes, no gaussian dimples). The main pipeline
(iris env, torch 2.5) cannot import nksr (torch 2.7/cu128), so Phase F writes the
per-object clouds to a npz and invokes this script in the nksr env.

  in : npz with arrays obj_0, obj_1, ...   each (N,3) xyz (one placed object cloud)
  out: a single merged triangle mesh ply (geometry only; the parent re-colours it)
  argv[3] mode: "obj" (default) closed objects, outward normals
                "bg"  open background surface, camera-facing normals, keep large comps

Each object gets denoised + globally-consistent normals, then NKSR; the result is
cropped to the input bbox (NKSR can balloon slightly past the points).
"""
import sys
import numpy as np
import open3d as o3d
import torch
import nksr


def mesh_one(pts, rec, dev, mode="obj"):
    p = o3d.geometry.PointCloud()
    p.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    p, _ = p.remove_statistical_outlier(nb_neighbors=16, std_ratio=2.0)
    q = np.asarray(p.points)
    if len(q) < 200:
        return None
    ext = float(np.linalg.norm(q.max(0) - q.min(0))) + 1e-6
    p.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.06 * ext, max_nn=60))
    if mode == "bg":
        # open surface (table+wall): orient toward the camera (≈ origin in gravity frame)
        p.orient_normals_towards_camera_location(np.zeros(3))
        n = np.asarray(p.normals)
    else:
        p.orient_normals_consistent_tangent_plane(30)
        n = np.asarray(p.normals)
        v = q - q.mean(0)
        if (np.einsum('ij,ij->i', n, v) > 0).mean() < 0.5:   # ensure outward
            n = -n
    xyz = torch.from_numpy(q.astype(np.float32)).to(dev)
    nrm = torch.from_numpy(n.astype(np.float32)).to(dev)
    field = rec.reconstruct(xyz, nrm, voxel_size=0.012 * ext, detail_level=0.0)
    mesh = field.extract_dual_mesh(mise_iter=1)
    m = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(mesh.v.cpu().numpy().astype(np.float64)),
        o3d.utility.Vector3iVector(mesh.f.cpu().numpy()))
    m = m.crop(p.get_axis_aligned_bounding_box())
    if mode == "bg" and len(m.triangles) > 0:               # drop floaters, keep table+wall
        tc, nt, _ = m.cluster_connected_triangles(); nt = np.asarray(nt)
        if len(nt):
            m.remove_triangles_by_mask((nt < 0.05 * nt.max())[np.asarray(tc)])
            m.remove_unreferenced_vertices()
    return m


def main():
    inp, outp = sys.argv[1], sys.argv[2]
    mode = sys.argv[3] if len(sys.argv) > 3 else "obj"
    data = np.load(inp)
    keys = sorted([k for k in data.files if k.startswith("obj_")], key=lambda s: int(s.split("_")[1]))
    dev = torch.device("cuda")
    rec = nksr.Reconstructor(dev)
    combined = o3d.geometry.TriangleMesh()
    n_ok = 0
    for k in keys:
        try:
            m = mesh_one(data[k], rec, dev, mode)
            if m is not None and len(m.vertices) > 0:
                combined += m
                n_ok += 1
        except Exception as e:
            print(f"NKSR {mode} {k} failed: {e}", file=sys.stderr)
    combined.compute_vertex_normals()
    o3d.io.write_triangle_mesh(outp, combined)
    print(f"NKSR_MESH_OK mode={mode} parts={n_ok} verts={len(combined.vertices)}")


if __name__ == "__main__":
    main()
