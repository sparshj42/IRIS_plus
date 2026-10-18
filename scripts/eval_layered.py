"""Occlusion-depth evaluation for single-view amodal scene reconstruction.

Every quantity here is defined from the GROUND TRUTH and the input camera alone;
the prediction is only ever scored, never used to decide what is scored.

  1. GT surface is sampled uniformly by area (so means are area-weighted) and put
     in the input camera frame (OpenCV: x right, y down, z forward).
  2. Points outside the image frustum are excluded for every method alike.
  3. Each GT point gets two occlusion depths by ray-casting the GT mesh from the
     camera centre towards it:
       k  (layer depth) : number of GT surfaces in front of it  (k=0 = visible)
       p  (peel depth)  : number of DISTINCT other object instances in front of
                          it, i.e. how many objects must be removed to expose it.
                          Needs per-face instance ids; points hidden behind
                          background structure (walls, floor) are reported
                          separately as "structural", since no peel exposes them.
  4. The prediction (any scale, input-camera frame) is aligned with a similarity
     transform estimated ONLY from visible GT (k=0): a z-buffer median-depth scale,
     then scaled point-to-point ICP between the prediction's front-most points and
     visible GT. Occluded GT never influences alignment.
  5. Metrics at tau = frac * (bbox diagonal of in-view GT):
       overall / visible / occluded  Chamfer (mean of the two directions,
                                     normalised by the diagonal), P, R, F
       hard-occluded                  occluded GT farther than tau from every
                                     visible GT surface. Plain occluded recall
                                     credits a visible-only prediction for floor
                                     strips and thin backs lying near visible
                                     surface; on hard points a depth map scores 0.
       per-layer  R@k, P@k, F@k       (prediction points are assigned to the layer
                                     of their nearest GT point for P@k); layer_hard
                                     restricts k >= 1 to hard points
       ODC-AUC                        mean of R@k over the hard layer buckets, and
                                     its occluded-only variant (k >= 1); the
                                     '_all' variants keep every point
       the same curves over peel depth p, per-instance occluded recall bucketed
       by how occluded the instance is, and (optional) peel-ORDER accuracy against
       GT occlusion pairs.

Usage:
  python scripts/eval_layered.py --gt_mesh room.ply --gt_cam cam.npz \
      --pred recon.ply [--pred_w2c pred_w2c.npy] [--gt_face_inst face_inst.npy] \
      [--pred_order_masks order.npz] [--tau 0.05] [--out scene.json]

  cam.npz : K (3,3), width, height, c2w (4,4, OpenCV convention; pass --gl_cam for
            an OpenGL/Blender camera-to-world)
  --pred  : .ply/.obj/.glb mesh (sampled) or point cloud, or .npy (N,>=3), in the
            input camera frame up to scale. If it lives in another frame, pass that
            frame's world-to-camera (4,4) for the input view as --pred_w2c.
  order.npz : 'masks' (n,H,W) bool, the method's instance masks in peel order.
"""
import argparse, json, os, sys
import numpy as np
import open3d as o3d
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

GL_TO_CV = np.diag([1.0, -1.0, -1.0, 1.0])


# ── I/O ──────────────────────────────────────────────────────────────────────
def load_camera(path, gl):
    c = np.load(path)
    K, W, H, c2w = c["K"].astype(np.float64), int(c["width"]), int(c["height"]), c["c2w"].astype(np.float64)
    if gl:
        c2w = c2w @ GL_TO_CV
    return K, W, H, c2w


def load_pred(path, n_samples, K=None, W=None, H=None, w2c=None):
    """Points of a prediction. Meshes are sampled inside the camera frustum only (the
    frustum test is scale-invariant, so it is valid before alignment), so a mesh that
    includes a whole room still gets dense samples where the camera looks."""
    if path.endswith(".npy"):
        return np.load(path)[:, :3].astype(np.float64)
    m = o3d.io.read_triangle_mesh(path)
    if len(m.triangles) == 0:
        return np.asarray(o3d.io.read_point_cloud(path).points)
    P = np.asarray(m.sample_points_uniformly(n_samples).points)
    if K is not None:
        Pc = P if w2c is None else P @ w2c[:3, :3].T + w2c[:3, 3]
        frac = max(in_frustum(Pc, K, W, H).mean(), 1e-3)
        P = np.asarray(m.sample_points_uniformly(int(min(n_samples / frac, 20 * n_samples))).points)
    return P


def sample_mesh(V, F, n, rng):
    """Area-uniform samples on a triangle mesh, returning the source face of each."""
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    fid = rng.choice(len(F), size=n, p=area / area.sum())
    r1, r2 = rng.random(n), rng.random(n)
    s = np.sqrt(r1)
    P = (1 - s)[:, None] * a[fid] + (s * (1 - r2))[:, None] * b[fid] + (s * r2)[:, None] * c[fid]
    return P, fid


def clip_poly(poly, lo, hi):
    """Sutherland-Hodgman clip of a convex polygon (m,3) to an axis-aligned box."""
    for ax in range(3):
        for bound, keep_ge in ((lo[ax], True), (hi[ax], False)):
            if len(poly) == 0:
                return poly
            out = []
            for i in range(len(poly)):
                p, q = poly[i], poly[(i + 1) % len(poly)]
                pin = p[ax] >= bound if keep_ge else p[ax] <= bound
                qin = q[ax] >= bound if keep_ge else q[ax] <= bound
                if pin:
                    out.append(p)
                if pin != qin:
                    s = (bound - p[ax]) / (q[ax] - p[ax])
                    out.append(p + s * (q - p))
            poly = np.array(out)
    return poly


def clip_mesh_to_box(V, F, lo, hi):
    """Triangles restricted to the box: whole triangles inside are kept, crossing ones
    are clipped and re-fanned. Returns (vertices, faces, source face per triangle) so
    that sampling puts every point in the box (a room's walls would otherwise take
    nearly all area-uniform samples away from the objects)."""
    T = V[F]
    tmin, tmax = T.min(1), T.max(1)
    touch = np.all(tmax >= lo, 1) & np.all(tmin <= hi, 1)
    inside = np.all(tmin >= lo, 1) & np.all(tmax <= hi, 1)
    keep = np.where(inside)[0]
    newT, src = [T[keep]], [keep]
    for f in np.where(touch & ~inside)[0]:
        poly = clip_poly(T[f], lo, hi)
        if len(poly) >= 3:
            fan = np.stack([np.repeat(poly[:1], len(poly) - 2, 0), poly[1:-1], poly[2:]], 1)
            newT.append(fan); src.append(np.full(len(fan), f))
    T2, src = np.concatenate(newT), np.concatenate(src)
    return T2.reshape(-1, 3), np.arange(len(T2) * 3).reshape(-1, 3), src


def roi_box(V, F, face_inst, K, W, H, margin):
    """Camera-frame box around the in-view vertices of object instances (+margin)."""
    P = V[np.unique(F[face_inst >= 0])]
    P = P[in_frustum(P, K, W, H)]
    if len(P) == 0:
        raise SystemExit("no object instance is in view")
    lo, hi = np.percentile(P, 0.5, 0), np.percentile(P, 99.5, 0)
    m = margin * (hi - lo)
    return lo - m, hi + m


def project(P, K):
    z = P[:, 2]
    zs = np.where(z > 1e-9, z, 1e-9)
    return K[0, 0] * P[:, 0] / zs + K[0, 2], K[1, 1] * P[:, 1] / zs + K[1, 2], z


def in_frustum(P, K, W, H):
    u, v, z = project(P, K)
    return (z > 1e-6) & (u >= 0) & (u < W) & (v >= 0) & (v < H)


# ── occlusion depth by ray-casting the GT mesh ───────────────────────────────
def make_scene(V, F):
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor(V.astype(np.float32)), o3d.core.Tensor(F.astype(np.uint32)))
    return scene


def front_hits(scene, P, eps):
    """For rays from the origin through each point P, every GT hit strictly in front
    of the point. Directions are left unnormalised so the point itself sits at t=1.
    Returns (ray index, t, face id) sorted by ray then depth."""
    rays = np.hstack([np.zeros_like(P), P]).astype(np.float32)
    lx = scene.list_intersections(o3d.core.Tensor(rays))
    splits = lx["ray_splits"].numpy().astype(np.int64)
    t = lx["t_hit"].numpy().astype(np.float64)
    prim = lx["primitive_ids"].numpy().astype(np.int64)
    ray = np.repeat(np.arange(len(P)), np.diff(splits))
    eps_rel = eps / np.linalg.norm(P, axis=1)
    keep = t < 1.0 - eps_rel[ray]
    ray, t, prim = ray[keep], t[keep], prim[keep]
    o = np.lexsort((t, ray))
    return ray[o], t[o], prim[o], eps_rel


def layer_depth(ray, t, eps_rel, n):
    """Distinct surfaces in front of each point; hits closer than eps along the same
    ray (shared edges, coplanar duplicates) count once."""
    if len(ray) == 0:
        return np.zeros(n, np.int64)
    new = np.ones(len(ray), bool)
    same = ray[1:] == ray[:-1]
    new[1:] = ~same | (t[1:] - t[:-1] > eps_rel[ray[1:]])
    return np.bincount(ray[new], minlength=n)


def peel_depth(ray, prim, face_inst, own_inst, n):
    """Distinct other instances in front of each point (-1 if any background
    structure is in front: no peel can expose it)."""
    inst = face_inst[prim]
    p = np.zeros(n, np.int64)
    if len(ray) == 0:
        return p
    structural = np.zeros(n, bool)
    structural[ray[inst < 0]] = True
    sel = (inst >= 0) & (inst != own_inst[ray])
    pairs = np.unique(np.stack([ray[sel], inst[sel]], 1), axis=0)
    if len(pairs):
        p = np.bincount(pairs[:, 0], minlength=n)
    p[structural] = -1
    return p


# ── similarity alignment from visible GT only ────────────────────────────────
def zbuffer(P, K, W, H, cell):
    """Front-most point of P in each cell x cell pixel block: (cell id, point index)."""
    u, v, z = project(P, K)
    ok = np.where((z > 1e-6) & (u >= 0) & (u < W) & (v >= 0) & (v < H))[0]
    gw = int(np.ceil(W / cell))
    idx = (v[ok] // cell).astype(np.int64) * gw + (u[ok] // cell).astype(np.int64)
    o = np.lexsort((z[ok], idx))
    idx, ok = idx[o], ok[o]
    first = np.ones(len(idx), bool)
    first[1:] = idx[1:] != idx[:-1]
    return idx[first], ok[first]


def umeyama(X, Y):
    """Similarity (s, R, t) minimising |s R X + t - Y|^2."""
    mx, my = X.mean(0), Y.mean(0)
    Xc, Yc = X - mx, Y - my
    U, D, Vt = np.linalg.svd(Yc.T @ Xc / len(X))
    S = np.eye(3)
    S[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ S @ Vt
    s = float(np.trace(np.diag(D) @ S) / (Xc ** 2).sum(1).mean())
    return s, R, my - s * R @ mx


def align_sim3(pred, gt_vis, K, W, H, tau, cell=4):
    """Pixel-aligned similarity: in each image cell the front-most prediction point
    and the front-most visible GT point form a correspondence; a trimmed Umeyama fit
    on those gives scale/rotation/translation, then rigid ICP (scale held fixed)
    refines it. Scale is never re-estimated by ICP, which drifts on partial
    predictions. Only visible GT is used."""
    cg, ig = zbuffer(gt_vis, K, W, H, cell)
    # A complete (amodal) but sparse prediction can leave a cell without any visible
    # point, and the cell's front-most point is then a hidden surface. Hidden-point
    # removal from the camera centre (Katz et al. 2007, scale-invariant) first keeps
    # only points the camera can see.
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pred))
    span = float(np.linalg.norm(pred.max(0) - pred.min(0)))
    _, seen = pc.hidden_point_removal([0.0, 0.0, 0.0], span * 100)
    seen = np.asarray(seen, np.int64)
    cp, ip = zbuffer(pred[seen], K, W, H, cell)
    ip = seen[ip]
    common, a, b = np.intersect1d(cp, cg, return_indices=True)
    if len(common) < 50:
        raise RuntimeError(f"only {len(common)} shared z-buffer cells; prediction is not in the input camera frame?")
    X, Y = pred[ip[a]], gt_vis[ig[b]]
    keep = np.ones(len(X), bool)
    for _ in range(8):                                       # trimmed refit: drop the worst 20%
        s, R, t = umeyama(X[keep], Y[keep])
        res = np.linalg.norm(s * X @ R.T + t - Y, axis=1)
        keep = res <= np.percentile(res, 80)
    S = np.eye(4)
    S[:3, :3], S[:3, 3] = s * R, t
    src = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pred[ip] @ S[:3, :3].T + S[:3, 3]))
    dst = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(gt_vis))
    T = np.eye(4)
    for d in (2 * tau, tau, tau / 2):
        r = o3d.pipelines.registration.registration_icp(
            src, dst, d, T, o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50))
        T = r.transformation
    S = T @ S
    return S, {"scale": float(s), "n_corr": int(len(X)), "corr_inlier_med": float(np.median(res[keep])),
               "icp_fitness": float(r.fitness), "icp_rmse": float(r.inlier_rmse)}


# ── metrics ──────────────────────────────────────────────────────────────────
def prf(d_g, d_p, tau):
    R = float((d_g < tau).mean()) if len(d_g) else float("nan")
    P = float((d_p < tau).mean()) if len(d_p) else float("nan")
    F = 2 * P * R / (P + R) if (P + R) > 0 else 0.0
    return {"P": P, "R": R, "F": F, "n_gt": int(len(d_g)), "n_pred": int(len(d_p))}


def bucket_curve(depth, d_g2p, nn_gt, d_p2g, tau, kmax, diag, valid=None):
    """R@k for GT points at depth bucket k; P@k for prediction points whose nearest
    GT point is at depth k. Last bucket is '>=kmax'. Points outside `valid` are in
    no bucket (and prediction points nearest to them count for no P@k)."""
    out = {}
    valid = np.ones(len(depth), bool) if valid is None else valid
    pred_depth, pred_valid = depth[nn_gt], valid[nn_gt]
    for k in range(kmax + 1):
        g = valid & ((depth == k) if k < kmax else (depth >= kmax))
        p = pred_valid & ((pred_depth == k) if k < kmax else (pred_depth >= kmax))
        key = str(k) if k < kmax else f">={kmax}"
        m = prf(d_g2p[g], d_p2g[p], tau)
        m["chamfer"] = (0.5 * (d_g2p[g].mean() + d_p2g[p].mean()) / diag) if g.any() and p.any() else float("nan")
        out[key] = m
    recalls = [out[k]["R"] for k in out if out[k]["n_gt"] > 0]
    occ = [out[k]["R"] for k in list(out)[1:] if out[k]["n_gt"] > 0]
    return out, (float(np.mean(recalls)) if recalls else float("nan")), (float(np.mean(occ)) if occ else float("nan"))


def instance_report(inst, pdepth, d_g2p, tau, bins=(0.0, 0.25, 0.5, 0.75, 1.0001)):
    rows = []
    for i in np.unique(inst[inst >= 0]):
        m = inst == i
        hidden = m & (pdepth >= 1)
        rows.append({"instance": int(i), "n": int(m.sum()),
                     "occluded_frac": float(hidden.sum() / m.sum()),
                     "R_all": float((d_g2p[m] < tau).mean()),
                     "R_occluded": float((d_g2p[hidden] < tau).mean()) if hidden.any() else None})
    buckets = {}
    for lo, hi in zip(bins[:-1], bins[1:]):
        sel = [r for r in rows if lo <= r["occluded_frac"] < hi and r["R_occluded"] is not None]
        buckets[f"{lo:.2f}-{min(hi, 1):.2f}"] = {
            "n_instances": len(sel),
            "R_occluded_mean": float(np.mean([r["R_occluded"] for r in sel])) if sel else None}
    return rows, buckets


# ── peel-order accuracy against GT occlusion pairs ───────────────────────────
def pixel_rays(K, W, H, stride):
    us, vs = np.meshgrid(np.arange(0, W, stride) + 0.5, np.arange(0, H, stride) + 0.5)
    d = np.stack([(us - K[0, 2]) / K[0, 0], (vs - K[1, 2]) / K[1, 1], np.ones_like(us)], -1)
    return d.reshape(-1, 3), us.reshape(-1).astype(int), vs.reshape(-1).astype(int)


def gt_instance_map(scene, face_inst, K, W, H):
    d, us, vs = pixel_rays(K, W, H, 1)
    r = scene.cast_rays(o3d.core.Tensor(np.hstack([np.zeros_like(d), d]).astype(np.float32)))
    prim = r["primitive_ids"].numpy().astype(np.int64)
    hit = np.isfinite(r["t_hit"].numpy())
    m = np.full(W * H, -1, np.int64)
    m[hit] = face_inst[prim[hit]]
    return m.reshape(H, W)


def gt_occlusion_pairs(scene, face_inst, K, W, H, stride=2, min_px=20):
    """a -> b when, along some pixel ray, instance a's first hit is in front of
    instance b's first hit. Edges need >= min_px supporting pixels. Pairs inside a
    strongly connected component (mutual / cyclic occlusion) have no correct
    whole-object order and are reported separately, never mixed into the rate."""
    d, _, _ = pixel_rays(K, W, H, stride)
    lx = scene.list_intersections(o3d.core.Tensor(np.hstack([np.zeros_like(d), d]).astype(np.float32)))
    splits = lx["ray_splits"].numpy().astype(np.int64)
    t = lx["t_hit"].numpy()
    inst = face_inst[lx["primitive_ids"].numpy().astype(np.int64)]
    ray = np.repeat(np.arange(len(d)), np.diff(splits))
    sel = inst >= 0
    ray, t, inst = ray[sel], t[sel], inst[sel]
    o = np.lexsort((t, inst, ray))                                  # first hit per (ray, instance)
    ray, t, inst = ray[o], t[o], inst[o]
    first = np.ones(len(ray), bool)
    first[1:] = (ray[1:] != ray[:-1]) | (inst[1:] != inst[:-1])
    ray, t, inst = ray[first], t[first], inst[first]
    o = np.lexsort((t, ray))                                        # front-to-back per ray
    ray, inst = ray[o], inst[o]
    support = {}
    for off in range(1, 16):
        if off >= len(ray):
            break
        same = ray[off:] == ray[:-off]
        for a, b in zip(inst[:-off][same], inst[off:][same]):
            support[(int(a), int(b))] = support.get((int(a), int(b)), 0) + 1
    thr = max(1, int(round(min_px / stride ** 2)))
    edges = {e: c for e, c in support.items() if c >= thr}
    ids = sorted({i for e in edges for i in e})
    if not ids:
        return {}, set(), ids
    ix = {v: k for k, v in enumerate(ids)}
    A = coo_matrix(([1] * len(edges), ([ix[a] for a, _ in edges], [ix[b] for _, b in edges])),
                   shape=(len(ids), len(ids)))
    _, comp = connected_components(A, directed=True, connection="strong")
    sizes = np.bincount(comp)
    cyclic = {e for e in edges if comp[ix[e[0]]] == comp[ix[e[1]]] and sizes[comp[ix[e[0]]]] > 1}
    return edges, cyclic, ids


def score_order(order_masks, inst_map, edges, cyclic, min_iou=0.3):
    """Match the method's masks to GT visible-instance masks (Hungarian on IoU), then
    count GT occlusion pairs placed in the wrong sequence."""
    gt_ids = [i for i in np.unique(inst_map) if i >= 0]
    n = len(order_masks)
    iou = np.zeros((n, len(gt_ids)))
    for a in range(n):
        pm = order_masks[a]
        for b, g in enumerate(gt_ids):
            gm = inst_map == g
            u = (pm | gm).sum()
            iou[a, b] = (pm & gm).sum() / u if u else 0.0
    r, c = linear_sum_assignment(-iou)
    pos = {gt_ids[cc]: int(rr) for rr, cc in zip(r, c) if iou[rr, cc] >= min_iou}
    scored = viol = unmatched = 0
    for (a, b) in edges:
        if (a, b) in cyclic:
            continue
        if a not in pos or b not in pos:
            unmatched += 1
            continue
        scored += 1
        viol += pos[b] < pos[a]
    return {"pairs_acyclic": len(edges) - len(cyclic), "pairs_cyclic": len(cyclic),
            "pairs_scored": scored, "pairs_unmatched": unmatched,
            "violation_rate": viol / scored if scored else None,
            "matched_instances": len(pos), "gt_visible_instances": len(gt_ids)}


# ── main ─────────────────────────────────────────────────────────────────────
def evaluate(a):
    rng = np.random.default_rng(a.seed)
    K, W, H, c2w = load_camera(a.gt_cam, a.gl_cam)
    w2c = np.linalg.inv(c2w)
    mesh = o3d.io.read_triangle_mesh(a.gt_mesh)
    V = (w2c[:3, :3] @ np.asarray(mesh.vertices).T).T + w2c[:3, 3]
    F = np.asarray(mesh.triangles).astype(np.int64)
    face_inst = np.load(a.gt_face_inst).astype(np.int64) if a.gt_face_inst else None
    if face_inst is not None and len(face_inst) != len(F):
        raise SystemExit(f"face_inst has {len(face_inst)} entries for {len(F)} faces")

    roi_lo = roi_hi = None
    if a.roi == "objects":
        # Evaluate where the objects are: a room in view would otherwise set the
        # diagonal (and tau) to metres and take nearly all samples. The mesh is
        # clipped to the box before sampling; occluders outside the box still
        # count, since ray casting below uses the full mesh.
        if face_inst is None:
            raise SystemExit("--roi objects needs --gt_face_inst")
        roi_lo, roi_hi = roi_box(V, F, face_inst, K, W, H, a.roi_margin)
        Vs, Fs, src = clip_mesh_to_box(V, F, roi_lo, roi_hi)
    else:
        Vs, Fs, src = V, F, np.arange(len(F))
    G, sfid = sample_mesh(Vs, Fs, a.n_gt * 2, rng)
    fid = src[sfid]
    inview = in_frustum(G, K, W, H)
    G, fid = G[inview][:a.n_gt], fid[inview][:a.n_gt]
    diag = float(np.linalg.norm(G.max(0) - G.min(0)))
    tau, eps = a.tau * diag, 1e-3 * diag

    scene = make_scene(V, F)
    ray, t, prim, eps_rel = front_hits(scene, G, eps)
    k = layer_depth(ray, t, eps_rel, len(G))
    own = face_inst[fid] if face_inst is not None else None
    p = peel_depth(ray, prim, face_inst, own, len(G)) if face_inst is not None else None

    T = np.load(a.pred_w2c) if a.pred_w2c else None
    pred = load_pred(a.pred, a.n_pred, K, W, H, T)
    if T is not None:
        pred = (T[:3, :3] @ pred.T).T + T[:3, 3]
    pred = pred[in_frustum(pred, K, W, H)]           # scale-invariant, so safe before alignment
    if len(pred) > a.n_pred:
        pred = pred[rng.choice(len(pred), a.n_pred, replace=False)]

    if a.align == "sim3":
        S, info = align_sim3(pred, G[k == 0], K, W, H, tau)
        pred = (S[:3, :3] @ pred.T).T + S[:3, 3]
        info["icp_rmse_over_diag"] = info["icp_rmse"] / diag
    else:
        info = {"mode": "none"}
    pin = in_frustum(pred, K, W, H)
    if roi_lo is not None:                           # same box for the prediction
        pin &= np.all((pred >= roi_lo) & (pred <= roi_hi), 1)
    n_pred_out = int((~pin).sum())
    pred = pred[pin]

    d_g2p = cKDTree(pred).query(G)[0]
    d_p2g, nn_gt = cKDTree(G).query(pred)
    d_vis = cKDTree(G[k == 0]).query(G)[0]          # GT point -> nearest VISIBLE GT surface

    def metrics_at(tf):
        """All metrics at threshold tf * diag. 'hard' occluded points are those farther
        than the threshold from every visible surface: a prediction holding only the
        visible surface (a depth map) scores 0 on them by construction, so they
        measure recovery that visible geometry cannot get credit for by proximity."""
        tt = tf * diag
        hard = (k == 0) | (d_vis >= tt)
        m = {"overall": prf(d_g2p, d_p2g, tt)}
        m["overall"]["chamfer"] = float(0.5 * (d_g2p.mean() + d_p2g.mean()) / diag)
        m["visible"] = prf(d_g2p[k == 0], d_p2g[k[nn_gt] == 0], tt)
        m["occluded"] = prf(d_g2p[k >= 1], d_p2g[k[nn_gt] >= 1], tt)
        m["occluded_hard"] = prf(d_g2p[(k >= 1) & hard], d_p2g[(k[nn_gt] >= 1) & hard[nn_gt]], tt)
        m["layer"], m["odc_auc_all"], m["odc_auc_occluded_all"] = bucket_curve(k, d_g2p, nn_gt, d_p2g, tt, a.kmax, diag)
        m["layer_hard"], m["odc_auc"], m["odc_auc_occluded"] = bucket_curve(
            k, d_g2p, nn_gt, d_p2g, tt, a.kmax, diag, valid=hard)
        if p is not None:
            peelable = p >= 0
            m["peel"], m["peel_auc"], m["peel_auc_occluded"] = bucket_curve(
                p, d_g2p, nn_gt, d_p2g, tt, a.kmax, diag, valid=peelable & hard)
            st = ~peelable
            m["structural"] = {"n_gt": int(st.sum()), "R": float((d_g2p[st] < tt).mean()) if st.any() else None}
            m["instances"], m["instance_buckets"] = instance_report(own, np.where(hard, p, 0), d_g2p, tt)
        return m

    res = {"pred": a.pred, "gt_mesh": a.gt_mesh, "tau_frac": a.tau, "diag": diag,
           "n_gt_inview": int(len(G)), "n_gt_outview": int((~inview).sum()),
           "n_pred_inview": int(len(pred)), "n_pred_outview": n_pred_out, "align": info}
    res.update(metrics_at(a.tau))
    res["by_tau"] = {f"t{tf * 100:g}": {kk: v for kk, v in metrics_at(tf).items() if kk != "instances"}
                     for tf in a.taus}

    if a.pred_order_masks:
        if face_inst is None:
            raise SystemExit("--pred_order_masks needs --gt_face_inst")
        edges, cyclic, _ = gt_occlusion_pairs(scene, face_inst, K, W, H)
        inst_map = gt_instance_map(scene, face_inst, K, W, H)
        masks = np.load(a.pred_order_masks)["masks"].astype(bool)
        res["ordering"] = score_order(masks, inst_map, edges, cyclic)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt_mesh", required=True)
    ap.add_argument("--gt_cam", required=True)
    ap.add_argument("--gl_cam", action="store_true", help="c2w in cam.npz is OpenGL/Blender convention")
    ap.add_argument("--gt_face_inst", default=None)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--pred_w2c", default=None)
    ap.add_argument("--pred_order_masks", default=None)
    ap.add_argument("--align", choices=["sim3", "none"], default="sim3")
    ap.add_argument("--tau", type=float, default=0.05, help="F-score threshold, fraction of in-view GT diagonal")
    ap.add_argument("--taus", type=float, nargs="*", default=[0.01, 0.02, 0.05],
                    help="extra thresholds reported under by_tau.t<percent>")
    ap.add_argument("--roi", choices=["objects", "none"], default=None,
                    help="'objects': score inside the 3D box around in-view object instances "
                         "(+margin), tau relative to that box. Default: objects when "
                         "--gt_face_inst is given, else none")
    ap.add_argument("--roi_margin", type=float, default=0.25)
    ap.add_argument("--kmax", type=int, default=3)
    ap.add_argument("--n_gt", type=int, default=200_000)
    ap.add_argument("--n_pred", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.roi is None:
        a.roi = "objects" if a.gt_face_inst else "none"
    res = evaluate(a)
    txt = json.dumps(res, indent=2, default=float)
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as f:
            f.write(txt)
    o, v, c, h = res["overall"], res["visible"], res["occluded"], res["occluded_hard"]
    print(f"align: {res['align']}")
    print(f"overall  CD {o['chamfer']:.4f}  F {o['F']:.3f}   visible F {v['F']:.3f}   "
          f"occluded R {c['R']:.3f}   hard-occluded R {h['R']:.3f} F {h['F']:.3f}")
    print("R@k (hard)  " + "  ".join(f"{kk}:{m['R']:.3f}" for kk, m in res["layer_hard"].items())
          + f"   ODC-AUC {res['odc_auc']:.3f}  (occluded {res['odc_auc_occluded']:.3f})")
    if "peel" in res:
        print("R@p  " + "  ".join(f"{kk}:{m['R']:.3f}" for kk, m in res["peel"].items()) + f"   peel-AUC {res['peel_auc']:.3f}")
    if "ordering" in res:
        print(f"ordering: {res['ordering']}")


if __name__ == "__main__":
    main()
