"""Order-invariance pilot: does disagreement between peel runs predict error?

Peeling two objects that do not occlude each other in either order must expose the
same hidden surface. So for one scene we run IRIS several times, varying the peel
order among VALID topological orders of the CORe graph (--peel_order_seed) and/or
the inpainting noise (--inpaint_seed), and ask whether points the runs disagree on
are the points that are wrong.

Disagreement is computed WITHOUT ground truth (runs are aligned to run 0 through
pixel correspondences on the shared input view); GT is used only to score it.

Per point of every run:
  u  = mean distance to the nearest point of each other run   (no GT)
  e  = distance to the GT surface                            (GT, after Sim(3))
Reported on all points and on points whose nearest GT is HARD-occluded (see
eval_layered.py), the region the method exists for:
  Spearman(u, e); AUSE (sparsification by u vs. the oracle sort by e); precision of
  the full union vs. the consensus subset; hard-occluded recall of one run vs. the
  union vs. the consensus subset.
Go/no-go: Spearman on hard-occluded points >~ 0.3 and consensus precision above
union precision without losing most of the union's recall gain.

Inputs are run directories made with --skip_3d --stop_after_vggt (they need
vggt_pointmaps.npz; peel_order.json is read if present).

Usage:
  python scripts/pilot_order_invariance.py --runs out/s1_o0 out/s1_o1 out/s1_o2 out/s1_o3 \
      --gt_mesh s1/mesh.ply --gt_cam s1/cam.npz [--gl_cam] [--out s1_pilot.json]
"""
import argparse, json, os, sys
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_layered as E

CONF = 0.5          # same VGGT confidence threshold as the pipeline


def load_run(d, max_pts, rng):
    z = np.load(os.path.join(d, "vggt_pointmaps.npz"), allow_pickle=True)
    wp, conf, ext = z["world_points"], z["conf"], z["extrinsics"]
    R, t = ext[0][:, :3], ext[0][:, 3]
    wp = wp @ R.T + t                                     # VGGT world -> input-camera frame
    view = np.broadcast_to(np.arange(len(wp))[:, None, None], conf.shape)
    m = conf > CONF
    P, V = wp[m], view[m]
    pix0 = np.where(m[0].reshape(-1))[0]                 # pixel ids of view-0 points (for alignment)
    if len(P) > max_pts:
        s = np.sort(rng.choice(len(P), max_pts, replace=False))
        P, V = P[s], V[s]
    order = None
    if os.path.exists(os.path.join(d, "peel_order.json")):
        order = json.load(open(os.path.join(d, "peel_order.json")))["order"]
    return {"P": P, "view": V, "v0": wp[0].reshape(-1, 3), "m0": m[0].reshape(-1), "order": order}


def ause(u, e, steps=20):
    """Area between the sparsification curve (drop highest-u first) and the oracle
    (drop highest-e first), on mean error normalised by the full mean."""
    fr = np.linspace(0, 0.95, steps)
    def curve(key):
        o = np.argsort(-key)
        return np.array([e[o[int(f * len(e)):]].mean() for f in fr]) / e.mean()
    c, orc = curve(u), curve(e)
    trapz = getattr(np, "trapezoid", None) or np.trapz
    return float(trapz(c - orc, fr)), c.tolist(), orc.tolist()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--gt_mesh", required=True)
    ap.add_argument("--gt_cam", required=True)
    ap.add_argument("--gl_cam", action="store_true")
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--consensus", type=float, default=0.01, help="agreement radius, fraction of scene diagonal")
    ap.add_argument("--max_pts", type=int, default=150_000)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if len(a.runs) < 3:
        raise SystemExit("need at least 3 runs")
    rng = np.random.default_rng(0)

    runs = [load_run(d, a.max_pts, rng) for d in a.runs]

    # ── align every run to run 0 on the shared input view (no GT) ────────────
    ref = runs[0]
    for r in runs[1:]:
        both = ref["m0"] & r["m0"]
        s, R, t = E.umeyama(r["v0"][both], ref["v0"][both])
        r["P"] = s * r["P"] @ R.T + t
    diag = float(np.linalg.norm(np.percentile(ref["P"], 99, 0) - np.percentile(ref["P"], 1, 0)))

    # ── disagreement u: mean NN distance to the other runs ───────────────────
    trees = [cKDTree(r["P"]) for r in runs]
    for i, r in enumerate(runs):
        d = np.stack([trees[j].query(r["P"])[0] for j in range(len(runs)) if j != i], 1)
        r["u"] = d.mean(1) / diag
        r["support"] = (d < a.consensus * diag).sum(1)          # other runs that agree

    # ── GT: layers + hard-occluded mask, Sim(3) from visible GT only ─────────
    K, W, H, c2w = E.load_camera(a.gt_cam, a.gl_cam)
    import open3d as o3d
    mesh = o3d.io.read_triangle_mesh(a.gt_mesh)
    w2c = np.linalg.inv(c2w)
    Vg = np.asarray(mesh.vertices) @ w2c[:3, :3].T + w2c[:3, 3]
    Fg = np.asarray(mesh.triangles).astype(np.int64)
    G, _ = E.sample_mesh(Vg, Fg, 200_000, rng)
    G = G[E.in_frustum(G, K, W, H)]
    gdiag = float(np.linalg.norm(G.max(0) - G.min(0)))
    tau = a.tau * gdiag
    ray, t, prim, eps_rel = E.front_hits(E.make_scene(Vg, Fg), G, 1e-3 * gdiag)
    k = E.layer_depth(ray, t, eps_rel, len(G))
    hard_gt = (k >= 1) & (cKDTree(G[k == 0]).query(G)[0] >= tau)

    S, info = E.align_sim3(ref["P"], G[k == 0], K, W, H, tau)
    gtree = cKDTree(G)
    for r in runs:
        r["P"] = r["P"] @ S[:3, :3].T + S[:3, 3]
        r["e"], nn = gtree.query(r["P"])
        r["e"] = r["e"] / gdiag
        r["hard"] = hard_gt[nn]

    # ── statistics ───────────────────────────────────────────────────────────
    u = np.concatenate([r["u"] for r in runs])
    e = np.concatenate([r["e"] for r in runs])
    hard = np.concatenate([r["hard"] for r in runs])
    sup = np.concatenate([r["support"] for r in runs])
    cons = sup >= (len(runs) - 1) // 2 + 1                    # agreed by a majority of the other runs
    out = {"runs": a.runs, "n_runs": len(runs), "align_run0_to_gt": info,
           "distinct_orders": len({tuple(r["order"]) for r in runs if r["order"]}) or None}
    tf = tau / gdiag
    for name, sel in (("all", np.ones_like(hard)), ("hard_occluded", hard)):
        if sel.sum() < 100:
            out[name] = {"n": int(sel.sum())}
            continue
        rho = spearmanr(u[sel], e[sel]).correlation
        A, curve, oracle = ause(u[sel], e[sel])
        out[name] = {"n": int(sel.sum()), "spearman_u_e": float(rho), "ause": A,
                     "sparsification": curve, "oracle": oracle,
                     "precision_union": float((e[sel] < tf).mean()),
                     "precision_consensus": float((e[sel & cons] < tf).mean()) if (sel & cons).any() else None,
                     "frac_kept_consensus": float((sel & cons).sum() / sel.sum())}

    # does using several orders recover more hidden surface than one run?
    Gh = G[hard_gt]
    if len(Gh):
        rec = lambda P: float((cKDTree(P).query(Gh)[0] < tau).mean()) if len(P) else 0.0
        allP = np.concatenate([r["P"] for r in runs])
        consP = allP[cons]
        out["hard_occluded_recall"] = {"single_runs": [rec(r["P"]) for r in runs],
                                       "union": rec(allP), "consensus": rec(consP), "n_gt": int(len(Gh))}

    h = out.get("hard_occluded", {})
    print(f"runs {len(runs)} | distinct orders {out['distinct_orders']} | hard-occluded pred points {h.get('n')}")
    if "spearman_u_e" in h:
        print(f"Spearman(u,e) hard {h['spearman_u_e']:.3f}  all {out['all']['spearman_u_e']:.3f} | AUSE hard {h['ause']:.3f}")
        print(f"precision hard: union {h['precision_union']:.3f}  consensus {h['precision_consensus']}  "
              f"(keeps {h['frac_kept_consensus']:.2f})")
    if "hard_occluded_recall" in out:
        r = out["hard_occluded_recall"]
        print(f"hard-occluded recall: single {np.mean(r['single_runs']):.3f}  union {r['union']:.3f}  consensus {r['consensus']:.3f}")
    if a.out:
        with open(a.out, "w") as f:
            json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
