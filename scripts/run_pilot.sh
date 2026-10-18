#!/usr/bin/env bash
# Order-invariance pilot driver (H100). For every scene:
#   orders  : 4 runs, peel_order_seed 0..3, inpaint_seed fixed at 0
#   inpaint : 4 runs, inpaint_seed 0..3, peel_order_seed fixed at 0  (run 0 shared)
# All runs stop after VGGT (--skip_3d --stop_after_vggt), so each costs only the
# peel loop + VGGT. Object discovery runs once per scene and is reused (--resume
# picks up detected_*.txt), so every run peels the same object list.
# Then pilot_order_invariance.py scores each group and eval_layered.py scores run 0.
#
# scenes.txt, one scene per line:   <scene_id> <image> <gt_mesh> <cam.npz> [--gl_cam]
# Usage: bash scripts/run_pilot.sh scenes.txt pilot_out
set -euo pipefail
SCENES=$1; OUT=$2
PY=${IRIS_PY:-python}
mkdir -p "$OUT"

while read -r SID IMG MESH CAM GL; do
  [[ -z "${SID:-}" || "$SID" == \#* ]] && continue
  echo "=== $SID"
  BASE="$OUT/$SID"
  STEM=$(basename "${IMG%.*}")

  # object discovery once
  if [[ ! -f "$BASE/det/detected_$STEM.txt" ]]; then
    mkdir -p "$BASE/det"
    $PY src/step0_vlm.py --image "$IMG" --out "$BASE/det/detected_$STEM.txt" \
        > "$BASE.det.log" 2>&1 || { echo "  detection failed, see $BASE.det.log"; continue; }
  fi

  run() {  # run <name> <order_seed> <inpaint_seed>
    local D="$BASE/$1"
    [[ -f "$D/vggt_pointmaps.npz" ]] && return 0
    mkdir -p "$D"; cp "$BASE/det/detected_$STEM.txt" "$D/"
    $PY src/pipeline.py --image "$IMG" --output_dir "$D" --resume --skip_3d --stop_after_vggt \
        --peel_order_seed "$2" --inpaint_seed "$3" > "$D/log.txt" 2>&1 \
        || echo "  run $1 failed, see $D/log.txt"
  }
  for s in 0 1 2 3; do run "o$s" "$s" 0; done
  for s in 1 2 3; do run "i$s" 0 "$s"; done

  $PY scripts/pilot_order_invariance.py --runs "$BASE"/o{0,1,2,3} --gt_mesh "$MESH" --gt_cam "$CAM" $GL \
      --out "$OUT/${SID}_orders.json" | sed 's/^/  [orders]  /' || true
  $PY scripts/pilot_order_invariance.py --runs "$BASE"/o0 "$BASE"/i{1,2,3} --gt_mesh "$MESH" --gt_cam "$CAM" $GL \
      --out "$OUT/${SID}_inpaint.json" | sed 's/^/  [inpaint] /' || true
done < "$SCENES"

$PY - "$OUT" <<'EOF'
import glob, json, sys, numpy as np
for kind in ("orders", "inpaint"):
    rows = [json.load(open(f)) for f in sorted(glob.glob(f"{sys.argv[1]}/*_{kind}.json"))]
    rho = [r["hard_occluded"]["spearman_u_e"] for r in rows if "spearman_u_e" in r.get("hard_occluded", {})]
    gain = [r["hard_occluded_recall"]["union"] - np.mean(r["hard_occluded_recall"]["single_runs"])
            for r in rows if "hard_occluded_recall" in r]
    if rho:
        print(f"[{kind}] scenes {len(rho)} | Spearman(u,e) on hard-occluded: median {np.median(rho):.3f} "
              f"(min {min(rho):.3f}, max {max(rho):.3f}) | union recall gain over one run: "
              f"median {np.median(gain):+.3f}")
print("GO if median Spearman >~ 0.3 for either kind.")
EOF
