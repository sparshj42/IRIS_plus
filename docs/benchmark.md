# Layered occlusion benchmark

Tools for evaluating single-view amodal reconstruction by how deep into occlusion
a method recovers geometry, and for generating scenes with complete ground truth.

## Scoring: `scripts/eval_layered.py`

Every region is defined from the GT mesh and the input camera alone, so all
methods are scored on identical point sets.

- **Layer depth k:** GT surfaces in front of a GT point (k = 0 is visible).
- **Peel depth p:** distinct other objects in front of it, i.e. how many removals
  expose it. Needs per-face instance ids. Points behind room structure are reported
  separately as *structural*.
- **Hard-occluded:** occluded GT farther than τ from every visible GT surface. On the
  synthetic self-test, a prediction holding only the visible surface scores 0.80
  "occluded recall" under the usual 5% threshold, because floor strips and thin
  backs sit next to visible surface; on hard points it scores 0.02. The occlusion
  curves use hard points.
- **Region of interest** (default whenever instance ids are given): the 3D box around
  the in-view objects plus a 25% margin. The GT mesh is clipped to the box *before*
  sampling, and τ is relative to the box diagonal. Without this, a room in view sets
  τ to tens of centimetres and the walls take nearly every area-uniform sample, so
  objects get only a few thousand points. Ray casting still uses the full mesh, so
  occluders outside the box count.
- **Alignment:** pixel-aligned trimmed-Umeyama Sim(3) between the prediction's and the
  visible GT's front-most points per image cell, then rigid ICP. Only visible GT is
  used, and scale is never re-fitted by ICP (scaled ICP drifted ~3% on partial
  predictions). The prediction's front-most points are taken after hidden-point
  removal from the camera centre; otherwise a complete but sparse prediction pairs
  hidden surfaces with visible GT (that recovered scale 0.12 instead of 0.59 on a
  test scene). Mesh predictions are sampled densely inside the camera frustum.
- **Reported:** overall / visible / occluded / hard-occluded Chamfer, P, R, F;
  R@k and R@p curves with their AUCs; per-instance occluded recall bucketed by
  occlusion fraction; peel-order violation rate against GT occlusion pairs, with
  cyclic components reported separately. Thresholds of 1%, 2% and 5% of the
  in-view GT diagonal are reported under `by_tau`.

```bash
python scripts/eval_layered.py --gt_mesh S/gt_mesh.ply --gt_cam S/cam.npz \
    --gt_face_inst S/face_inst.npy --pred out/S/observed_pointcloud.npy \
    --pred_order_masks out/S/peel_order_masks.npz --out runs/iris/S.json
python scripts/eval_aggregate.py IRIS=runs/iris Gen3DSR=runs/gen3dsr --ref Gen3DSR --md table.md
python scripts/selftest_eval_layered.py        # synthetic sanity checks
```

The prediction must be in the input camera's frame (OpenCV axes, any scale); pass
`--pred_w2c` for outputs in another frame.

## Synthetic scenes: `scripts/data/`

```bash
python scripts/data/download_gso.py --out data/gso --n 400        # Google Scanned Objects, CC-BY 4.0
bash scripts/data/make_dataset.sh data/gso data/layered 300 --samples 128
```

Each scene is a table in a room with 4–9 GSO objects dropped with physics and a
low camera. Camera poses are resampled until the GT occlusion graph, found by
multi-hit ray casting, has at least 2 occluding pairs. Per scene:
`rgb.png`, `depth.npy`, `inst.npy`, `gt_mesh.ply` + `face_inst.npy`, `cam.npz`
(OpenCV), `meta.json` (GT occlusion edges, GT peel order), and **GT clean plates**
`peel/step_XX_*`: the scene re-rendered after physically removing the first XX
objects of the GT order, for scoring each peel step against the truth.
`scenes.txt` indexes the scenes in the format `scripts/run_pilot.sh` reads.

On Windows, install Blender at a short path (`blenderproc run
--blender-install-path C:\short\dir ...`). Under a long path, Blender's NumPy
extension exceeds the 260-character limit and fails to import.

## Order-invariance pilot: `scripts/run_pilot.sh`

Runs each scene with 4 valid peel orders and 4 inpainting seeds
(`--peel_order_seed`, `--inpaint_seed`, `--stop_after_vggt`), then
`scripts/pilot_order_invariance.py` measures whether cross-run disagreement,
computed without GT, predicts error on hard-occluded points.
