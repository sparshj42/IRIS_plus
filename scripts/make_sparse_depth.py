"""Sample ~N sparse depth points from a dense depth map -> sparse .npy of
(row, col, metric_depth), simulating the problem's ~500-px sparse depth input.
For evaluation: feed a dataset's GT depth here, then run the pipeline with
--sparse_depth <out>.
"""
import argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--depth", required=True, help="dense depth .npy (H,W), metric")
ap.add_argument("--n", type=int, default=500)
ap.add_argument("--out", default="sparse_depth.npy")
args = ap.parse_args()

depth = np.load(args.depth).astype(np.float32)
valid = np.argwhere(np.isfinite(depth) & (depth > 0))
sel = valid[np.random.choice(len(valid), min(args.n, len(valid)), replace=False)]
sparse = np.stack([sel[:, 0], sel[:, 1], depth[sel[:, 0], sel[:, 1]]], axis=1).astype(np.float32)
np.save(args.out, sparse)
print(f"{len(sparse)} sparse points -> {args.out}  (depth range "
      f"{sparse[:,2].min():.3f}–{sparse[:,2].max():.3f})")
