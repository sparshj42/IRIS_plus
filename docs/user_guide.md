# User Guide

## Run the full pipeline

```bash
conda activate iris
cd IRIS

# RGB-only (monocular): geometry is metric-by-proportion, not true scale
python src/pipeline.py --image data/demoImage.png --output_dir output

# RGB-D (recommended when depth is available): pass the sensor depth for true metric scale
python src/pipeline.py --image frame.jpg --depth depth.npy --output_dir output
```

Inputs: **a single RGB image**, optionally with **a depth map** (`--depth`, RGB-D —
see [Input modes](#input-modes-rgb-vs-rgb-d) below). Outputs land in `--output_dir`:
- `synthetic_views/` — the peeled, same-pose views
- `fused_pointcloud.ply` — scene + registered object clouds
- `labeled_pointcloud.ply` — semantically colored cloud
- `final_semantic_mesh.ply` — final watertight semantic mesh

## Flags

The pipeline uses the models chosen by the A/B evaluations in [ax.md](ax.md):
**Qwen3-VL-32B** (discovery), **SAM 3** (segmentation), **Depth-Anything-V2**
(occlusion ordering), **RORem** (removal), **Amodal3R** (occlusion-aware
image-to-3D, default), **VGGT** (multi-view), **Mask2Former** (background labeling).
Image-to-3D is swappable via `--image3d`; the alternatives are documented in
[attribution.md](attribution.md) and [ax.md](ax.md).

| Flag | Default | Purpose |
|------|---------|---------|
| `--image` | `data/demoImage.png` | input RGB image (single-frame, IRIS's main mode) |
| `--depth` | – | **RGB-D mode**: dense metric depth map (`.npy`, aligned to `--image`). Used directly for peel ordering (**skips DepthAnythingV2**) **and** for true metric scale. See [Input modes](#input-modes-rgb-vs-rgb-d). |
| `--scene_dir` | – | folder of images (multi-view); overrides `--image` |
| `--output_dir` | `output` | where artifacts are written |
| `--image3d` | `amodal3r` | per-object 3D backend: `amodal3r` (occlusion-aware, default) or `trellis` (image-only baseline) |
| `--sparse_depth` | – | sparse metric scale: `.npy` of (row,col,metric_depth) ~500 px; scales VGGT via median depth ratio. A lighter alternative to `--depth` when only sparse points are available |
| `--skip_3d` | off | skip per-object image-to-3D; fused recon = VGGT scene only |
| `--resume` | off | skip phases whose outputs already exist (crash recovery) |
| `--vlm` | `Qwen/Qwen3-VL-32B-Instruct` | VLM model ID for object discovery. Use `Qwen/Qwen3-VL-8B-Instruct` on GPUs with <24 GB VRAM (RTX 4080 / 3090 etc.). |
| `--stop_after_peeling` | off | stop after the peel phase |

## Input modes: RGB vs RGB-D

IRIS runs from a single RGB image; a depth map is **optional but recommended when
available** (a depth sensor, an RGB-D dataset, or a stereo rig). Depth enters the
pipeline in two independent ways, and `--depth` supplies both:

| | RGB only | `--depth` (RGB-D) |
|---|---|---|
| **Peel ordering** | DepthAnythingV2 monocular depth | the **sensor depth directly** (no depth model) |
| **Metric scale** | proportional only (arbitrary scale) | **true metric** (sensor depth → scale factor) |
| **Reconstruction accuracy (cm)** | not metric | meaningful in real units |

```bash
# RGB-D: depth.npy is a dense (H×W) float metric depth map, aligned to --image
python src/pipeline.py --image frame.jpg --depth depth.npy --output_dir output
```

Notes:
- `--depth` is the input the **benchmark uses** — sensor depth gives the recon true
  metric scale, so reported centimetre accuracies are in real units (the geometry is
  still colour/VGGT-derived; depth only fixes scale + ordering).
- If you only have **sparse** metric points (not a dense map), use `--sparse_depth`
  instead — it fixes scale but peel ordering still uses DepthAnythingV2.
- The depth map is resized to the RGB resolution internally, so it does not need to
  match `--image` pixel-for-pixel, only be aligned (same camera view).

## Staged / crash-safe execution

On an unstable machine, split the run so each stage is short and resumable:

```bash
# Stage 1 — peel (saves views + object crops), then stop
python src/pipeline.py --image data/demoImage.png --stop_after_peeling \
       --output_dir output --resume

# Stage 2 — image-to-3D → VGGT → fusion → labeling → mesh
python src/pipeline.py --image data/demoImage.png --output_dir output --image3d amodal3r --resume
```

`--resume` reuses any completed phase and continues per-object peeling from the
last checkpoint, so a crash costs at most the object in flight. (Not needed on a
stable/datacenter GPU.)

## Metric reconstruction (sparse depth)

The problem provides ~500 sparse depth points per image. Generate the input from a
dataset's dense depth and run metric:

```bash
python scripts/make_sparse_depth.py --depth gt_depth.npy --n 500 --out sparse.npy
python src/pipeline.py --image rgb.png --sparse_depth sparse.npy --output_dir output
```

Without `--sparse_depth` the reconstruction is up-to-scale (fine for the
occlusion/semantic outputs; metric units require one of the depth inputs).

## Re-running occupancy standalone

`pipeline.py` is the single entry point (all phases run inline). Occupancy can be
recomputed on an existing run's artifacts without re-running the pipeline:

```bash
python src/step10_occupancy.py --output_dir output --grid 160
```
