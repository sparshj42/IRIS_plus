#!/usr/bin/env bash
# Generate N synthetic layered scenes (one BlenderProc process per scene, so a
# failed scene costs only itself). Scenes whose seed yields no valid camera or
# too few objects on the table are skipped and the next seed is tried.
#
# Usage: bash scripts/data/make_dataset.sh <gso_dir> <out_dir> <n_scenes> [extra gen args...]
#   e.g. bash scripts/data/make_dataset.sh data/gso data/layered 300 --samples 128
# Resumable: scenes that already have meta.json are kept.
set -uo pipefail
GSO=$1; OUT=$2; N=$3; shift 3
BP=${BLENDERPROC:-blenderproc}
mkdir -p "$OUT/logs"
made=$(ls "$OUT"/s*/meta.json 2>/dev/null | wc -l)
seed=0
while (( made < N )); do
  SID=$(printf "s%04d" "$seed")
  if [[ ! -f "$OUT/$SID/meta.json" ]]; then
    if $BP run scripts/data/gen_layered_scene.py --models_dir "$GSO" --out_dir "$OUT" \
         --scene_id "$SID" --seed "$seed" "$@" > "$OUT/logs/$SID.log" 2>&1 \
       && [[ -f "$OUT/$SID/meta.json" ]]; then
      made=$((made + 1)); echo "[$made/$N] $SID"
    else
      rm -rf "${OUT:?}/$SID"; echo "skip $SID ($(grep -m1 -E 'SystemExit|Error' "$OUT/logs/$SID.log" | cut -c1-120))"
    fi
  fi
  seed=$((seed + 1))
done
# rebuild the index from what exists (a crashed scene may have left a partial line)
: > "$OUT/scenes.txt"
for d in "$OUT"/s*/; do
  [[ -f "$d/meta.json" ]] || continue
  sid=$(basename "$d"); a=$(cd "$d" && pwd)
  echo "$sid $a/rgb.png $a/gt_mesh.ply $a/cam.npz" >> "$OUT/scenes.txt"
done
echo "$(wc -l < "$OUT/scenes.txt") scenes indexed in $OUT/scenes.txt"
