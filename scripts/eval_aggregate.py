"""Aggregate per-scene eval_layered.py JSONs into a method table with bootstrap CIs.

Each method is a directory of <scene>.json files. Only scenes present for EVERY
method are used, so all rows are paired. With --ref, each row also gets the
paired-bootstrap CI of (method - ref): a claim like "beats Gen3DSR" holds only if
that interval excludes 0.

Usage:
  python scripts/eval_aggregate.py IRIS=runs/iris Gen3DSR=runs/gen3dsr SAM3D=runs/sam3d \
      --ref Gen3DSR [--metrics overall.F occluded.R odc_auc_occluded peel_auc_occluded] [--md table.md]
"""
import argparse, glob, json, os, sys
import numpy as np

DEFAULT = ["overall.chamfer", "overall.F", "visible.F", "occluded_hard.R", "occluded_hard.F",
           "odc_auc_occluded", "peel_auc_occluded", "by_tau.t2.occluded_hard.F", "ordering.violation_rate"]


def get(d, key):
    for part in key.split("."):
        if not isinstance(d, dict) or part not in d:
            return np.nan
        d = d[part]
    return np.nan if d is None else float(d)


def boot(x, n, rng):
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    m = rng.integers(0, len(x), (n, len(x)))
    s = x[m].mean(1)
    return float(x.mean()), float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="NAME=DIR")
    ap.add_argument("--metrics", nargs="+", default=DEFAULT)
    ap.add_argument("--ref", default=None)
    ap.add_argument("--n_boot", type=int, default=10_000)
    ap.add_argument("--md", default=None)
    a = ap.parse_args()
    rng = np.random.default_rng(0)

    data = {}
    for r in a.runs:
        name, d = r.split("=", 1)
        data[name] = {os.path.splitext(os.path.basename(f))[0]: json.load(open(f))
                      for f in glob.glob(os.path.join(d, "*.json"))}
    scenes = sorted(set.intersection(*(set(v) for v in data.values())))
    if not scenes:
        raise SystemExit("no scene is present for every method")
    vals = {m: {k: np.array([get(data[m][s], k) for s in scenes]) for k in a.metrics} for m in data}

    lines = [f"Paired over {len(scenes)} scenes; mean [95% bootstrap CI]"
             + (f"; Δ vs {a.ref} below each cell" if a.ref else ""), "",
             "| Method | " + " | ".join(a.metrics) + " |", "|---" * (len(a.metrics) + 1) + "|"]
    for m in data:
        cells = []
        for k in a.metrics:
            mu, lo, hi = boot(vals[m][k], a.n_boot, rng)
            cell = f"{mu:.3f} [{lo:.3f}, {hi:.3f}]"
            if a.ref and m != a.ref:
                dmu, dlo, dhi = boot(vals[m][k] - vals[a.ref][k], a.n_boot, rng)
                sig = "*" if (dlo > 0 or dhi < 0) else ""
                cell += f"<br>Δ {dmu:+.3f} [{dlo:+.3f}, {dhi:+.3f}]{sig}"
            cells.append(cell)
        lines.append(f"| {m} | " + " | ".join(cells) + " |")
    lines += ["", "`*` = paired CI excludes 0. Lower is better for chamfer and violation_rate."]
    out = "\n".join(lines)
    sys.stdout.reconfigure(encoding="utf-8")
    print(out)
    if a.md:
        open(a.md, "w", encoding="utf-8").write(out + "\n")


if __name__ == "__main__":
    main()
