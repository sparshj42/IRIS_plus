"""
IRIS: Iterative Reconstruction via Incremental Scene-peeling
  Phase A : Object Discovery            (Qwen3-VL)
  Phase B : Depth ordering + iterative occlusion peeling
              - re-evaluate mask on current image state  (SAM3)
              - depth ordering                           (DepthAnything V2)
              - object removal revealing what's behind   (RORem)
  Phase B2: Image-to-3D (deferred)      (Amodal3R / TRELLIS; skip with --skip_3d)
  Phase C : Scene Reconstruction        (VGGT on same-pose synthetic views)
  Phase D : Rigid Registration + Fusion (mask-guided init + ICP; objects = observed
                                         front + Amodal3R back, background = final
                                         peeled view, plane-filled)
  Phase F : Mesh Generation             (per-object NKSR + plane-filled NKSR background,
                                         true colour; Poisson fallback)

Models are loaded/freed in phases. Image-to-3D and NKSR run in pinned-env subprocess workers.
"""

import argparse
import gc
import json
import os
import pickle
import sys

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
# Cap CPU thread fan-out BEFORE numpy/Open3D import: Open3D/MKL/OpenBLAS otherwise
# each spawn ~one thread per core (e.g. 224 on a big box), oversubscribing and
# stalling the CPU-heavy phases (registration, meshing). A modest cap is
# plenty here (the models run on GPU; CPU isn't the bottleneck). Override by
# exporting these vars yourself.
_NT = str(min(16, os.cpu_count() or 16))
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, _NT)

import cv2
import numpy as np
import torch
from PIL import Image

# Determinism: pin RNGs so re-runs are reproducible — e.g. the yaw chosen for a
# rotationally-symmetric object stays stable instead of flipping between runs.
# (Image-to-3D workers run in separate processes and seed themselves too.)
np.random.seed(0)
torch.manual_seed(0)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # src/ on path
import config

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
parser = argparse.ArgumentParser(description="IRIS pipeline")
parser.add_argument("--image", default="data/demoImage.png", help="input RGB image")
parser.add_argument("--scene_dir", default=None,
                    help="folder of images (multi-view); overrides --image")
parser.add_argument("--sparse_depth", default=None,
                    help="optional .npy of (row, col, metric_depth) rows for scale recovery")
parser.add_argument("--depth", default=None,
                    help="optional dense metric depth map (.npy, aligned to --image) — "
                         "RGB-D mode: used directly for peel ordering instead of "
                         "DepthAnythingV2, and for metric scale (skips the depth model)")
parser.add_argument("--output_dir", default="output")
parser.add_argument("--resume", action="store_true",
                    help="skip phases whose outputs already exist (crash recovery)")
parser.add_argument("--skip_3d", action="store_true",
                    help="skip per-object image-to-3D; fused recon = VGGT scene")
parser.add_argument("--image3d", choices=["amodal3r", "trellis"], default="amodal3r",
                    help="per-object 3D backend: occlusion-aware AMODAL3R (default, consumes "
                         "the SAM3 mask) or image-only TRELLIS (baseline). Other backends we "
                         "evaluated (TIGON, Wonder3D, SplAttN, TripoSR...) are documented in "
                         "docs/attribution.md.")
parser.add_argument("--stop_after_peeling", action="store_true",
                    help="exit after Phase B (for fast removal A/B comparison)")
parser.add_argument("--vlm", default=None,
                    help="VLM model ID override (e.g. Qwen/Qwen3-VL-8B-Instruct for GPUs "
                         "with <24 GB VRAM; default: Qwen/Qwen3-VL-32B-Instruct)")
parser.add_argument("--peel_order_seed", type=int, default=None,
                    help="sample a random VALID topological order of the occlusion graph "
                         "(every CORe edge respected) instead of nearest-first tie-breaking; "
                         "different seeds give the alternative orders used for order-invariance")
parser.add_argument("--inpaint_seed", type=int, default=None,
                    help="seed the removal inpainter's noise per peel step (default: global seed)")
parser.add_argument("--stop_after_vggt", action="store_true",
                    help="exit after Phase C (VGGT point maps saved); for cheap pilot runs")
args = parser.parse_args()

# RGB-D mode: derive sparse metric points from the dense --depth so the existing
# metric-scale machinery (Phase C) makes the reconstruction metric for free.
if args.depth and not args.sparse_depth:
    import numpy as _np
    os.makedirs(args.output_dir, exist_ok=True)
    _d = _np.load(args.depth).astype(_np.float32)
    _ys, _xs = _np.where(_d > 1e-3)
    if len(_ys):
        _sel = _np.random.default_rng(0).choice(len(_ys), min(3000, len(_ys)), replace=False)
        _sp = _np.stack([_ys[_sel], _xs[_sel], _d[_ys[_sel], _xs[_sel]]], 1).astype(_np.float32)
        _spp = os.path.join(args.output_dir, "_depth_sparse.npy")
        _np.save(_spp, _sp)
        args.sparse_depth = _spp        # reuse sparse → VGGT metric scaling

# per-output-dir so parallel runs (different --output_dir) don't collide
RECORDS_PATH = os.path.join(args.output_dir, "object_records.pkl")

if args.scene_dir:
    import glob as _glob
    _exts = ('.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG')
    image_list = sorted(p for p in _glob.glob(os.path.join(args.scene_dir, "*"))
                        if p.endswith(_exts))
    if not image_list:
        raise SystemExit(f"No images found in {args.scene_dir}")
    IMAGE_PATH = image_list[0]
else:
    image_list = [args.image]
    IMAGE_PATH = args.image

OUTPUT_DIR = args.output_dir
VIEWS_DIR = os.path.join(OUTPUT_DIR, "synthetic_views")
CROPS_DIR = os.path.join(OUTPUT_DIR, "object_crops")   # for the deferred image-to-3D phase
os.makedirs(VIEWS_DIR, exist_ok=True)
os.makedirs(CROPS_DIR, exist_ok=True)

ROREM_CKPT = config.ROREM_CKPT
SDXL_BASE = config.SDXL_BASE
ROREM_PROMPT = "4K, high quality, masterpiece, Highly detailed, Sharp focus, Professional, photorealistic, realistic"
ROREM_NEG_PROMPT = "low quality, worst, bad proportions, blurry, extra finger, Deformed, disfigured, unclear background"

VGGT_CONF_THRESHOLD = 0.5

def free_cuda(*objs):
    for o in objs:
        del o
    gc.collect()
    torch.cuda.empty_cache()


def metric_scale_from_sparse(world_points_v0, extr_v0, sparse_path, orig_hw):
    """Metric scale for the (arbitrary-scale) VGGT reconstruction from the problem's
    sparse depth (~500 px). sparse_path = .npy of (row, col, metric_depth) in the
    ORIGINAL image. Returns s = median(metric_depth / VGGT_camera_depth) at those
    pixels; multiply world points + camera translations by s to make recon metric."""
    sparse = np.load(sparse_path)
    H0, W0 = orig_hw
    Hv, Wv = world_points_v0.shape[:2]
    rows = np.clip((sparse[:, 0] * Hv / H0).astype(int), 0, Hv - 1)
    cols = np.clip((sparse[:, 1] * Wv / W0).astype(int), 0, Wv - 1)
    z_metric = sparse[:, 2]
    R, t = extr_v0[:, :3], extr_v0[:, 3]
    P = world_points_v0[rows, cols]                       # (N,3) world points
    z_vggt = (P @ R.T + t)[:, 2]                           # VGGT-scale camera depth
    valid = (z_vggt > 1e-6) & np.isfinite(z_vggt) & (z_metric > 0)
    return float(np.median(z_metric[valid] / z_vggt[valid]))


class Image3DWorker:
    """Drives an image->3D model in its own pinned conda env via a persistent
    subprocess speaking a line-based "@@" protocol. Input: a white-bg object crop
    PNG (+ optional text label). Output: a point cloud (gaussian xyz).

    Backends (both built on the TRELLIS gaussian decoder, same xyz+rgb contract):
      - "amodal3r": occlusion-aware Amodal3R (TRELLIS fork), `amodal3r` env. Takes the
                    object's mask in addition to the crop; reconstructs the complete
                    object, tolerating occlusion. Default.
      - "trellis":  image-only TRELLIS-image-large, `trellis` env, cwd = repo root.
    """

    _CFG = {
        "amodal3r": dict(env="amodal3r", script="src/amodal3r_worker.py", cwd=None),
        "trellis":  dict(env="trellis",  script="src/trellis_worker.py",  cwd=None),
    }

    def __init__(self, backend: str = "amodal3r"):
        import subprocess
        import tempfile
        if backend not in self._CFG:
            raise ValueError(f"unknown image3d backend {backend!r}; choose from {list(self._CFG)}")
        cfg = self._CFG[backend]
        self.backend = backend
        pybin = config.conda_env_python(cfg["env"])
        self.tmp = tempfile.mkdtemp(prefix=f"{backend}worker_")
        self.log = open(os.path.join(self.tmp, "worker.log"), "w")
        env = dict(os.environ, ATTN_BACKEND="xformers", SPCONV_ALGO="native")
        # prepend the worker env's bin so its tools (e.g. ninja, for any JIT CUDA
        # builds) are found — the subprocess otherwise inherits the parent (iris)
        # env's PATH, not the worker env's.
        env["PATH"] = os.path.dirname(pybin) + os.pathsep + env.get("PATH", "")
        # absolute worker script path: the worker adds models/<repo> to sys.path
        # relative to the repo root, so it must be launched from there.
        script = os.path.join(config.REPO_ROOT, cfg["script"])
        self.proc = subprocess.Popen(
            [pybin, "-u", script],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log,
            text=True, bufsize=1, env=env, cwd=cfg["cwd"],
        )
        for line in self.proc.stdout:
            if line.strip() == "@@READY":
                break
            if self.proc.poll() is not None:
                raise RuntimeError(f"{backend} worker exited before READY; see " + self.log.name)

    def pointcloud(self, crop_path: str, n: int = 10000, text: str = "",
                   mask_path: str = None) -> np.ndarray:
        import json
        out = os.path.join(self.tmp, "pc.npy")
        # absolute paths so they resolve regardless of the worker's cwd
        req = {"image": os.path.abspath(crop_path), "out": out, "n": n, "text": text}
        if mask_path is not None:                       # amodal3r consumes the mask
            req["mask"] = os.path.abspath(mask_path)
        self.proc.stdin.write(json.dumps(req) + "\n")
        self.proc.stdin.flush()
        for line in self.proc.stdout:
            line = line.strip()
            if line.startswith("@@OK"):
                return np.load(out)
            if line.startswith("@@ERR"):
                raise RuntimeError(f"{self.backend} worker: " + line)
            if self.proc.poll() is not None:
                raise RuntimeError(f"{self.backend} worker died; see " + self.log.name)
        raise RuntimeError(f"{self.backend} worker stdout closed unexpectedly")

    def close(self):
        try:
            self.proc.stdin.close()
            self.proc.wait(timeout=10)
        except Exception:
            self.proc.kill()


# used for sparse-depth metric scaling in Phase C (first image's size)
image = Image.open(IMAGE_PATH).convert("RGB")

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE A: OBJECT DISCOVERY (VLM)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("[Phase A] Object discovery (Qwen3-VL)")
print("=" * 60)

# The VLM (32B by default) is run in its own subprocess so the OS fully reclaims
# its VRAM on exit (device_map="auto" leaves accelerate hooks that don't free
# cleanly in-process, which OOMs the multi-model Phase B that follows).
import subprocess
step0 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "step0_vlm.py")


def _freest_gpu():
    """Pick the GPU with the most free memory, so the big VLM lands on a clear
    card (this box has 8×H100; GPU 0 may be shared with other processes).
    If the caller pinned CUDA_VISIBLE_DEVICES, respect it — otherwise parallel
    runs would all auto-pick the same 'freest' card and collide on the 62 GB VLM."""
    pinned = os.environ.get("CUDA_VISIBLE_DEVICES")
    if pinned not in (None, ""):
        return pinned
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True)
        free = [int(x) for x in out.split()]
        return str(int(np.argmax(free))) if free else "0"
    except Exception:
        return os.environ.get("CUDA_VISIBLE_DEVICES", "0")


_vlm_env = dict(os.environ, CUDA_VISIBLE_DEVICES=_freest_gpu())
if args.vlm:
    _vlm_env["IRIS_VLM_ID"] = args.vlm
_vlm_id = args.vlm or config.VLM_ID
print(f"  VLM ({_vlm_id}) on GPU {_vlm_env['CUDA_VISIBLE_DEVICES']}")
_all_names = []
for _img_path in image_list:
    _stem = os.path.splitext(os.path.basename(_img_path))[0]
    _det_file = os.path.join(OUTPUT_DIR, f"detected_{_stem}.txt")
    if not (args.resume and os.path.exists(_det_file)):
        subprocess.run([sys.executable, step0, "--image", _img_path,
                        "--out", _det_file], check=True, env=_vlm_env)
    with open(_det_file) as _f:
        _all_names.extend(l.strip() for l in _f if l.strip())
object_names = list(dict.fromkeys(_all_names))   # union, preserve order, deduplicate
print(f"Object types: {object_names}")

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE B: DEPTH ORDERING + ITERATIVE OCCLUSION PEELING
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("[Phase B] Depth ordering + iterative occlusion peeling")
print("=" * 60)

RESUME_B = args.resume and os.path.exists(RECORDS_PATH)

if not RESUME_B:
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    # Segmentation: SAM3 (text-promptable, one model)
    from transformers import Sam3Model, Sam3Processor
    sam3_model = Sam3Model.from_pretrained(
        "facebook/sam3", torch_dtype=torch.bfloat16).to("cuda").eval()
    sam3_processor = Sam3Processor.from_pretrained("facebook/sam3")

    depth_processor = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Large-hf")
    depth_model = AutoModelForDepthEstimation.from_pretrained(
        "depth-anything/Depth-Anything-V2-Large-hf", torch_dtype=torch.float16).to("cuda").eval()

    # Removal: RORem (SDXL-inpainting UNet fine-tuned for object removal)
    from diffusers import AutoPipelineForInpainting, UNet2DConditionModel
    rorem_pipe = AutoPipelineForInpainting.from_pretrained(SDXL_BASE, torch_dtype=torch.float16)
    rorem_unet = UNet2DConditionModel.from_pretrained(ROREM_CKPT).to("cuda", dtype=torch.float16)
    rorem_pipe.unet = rorem_unet
    rorem_pipe = rorem_pipe.to("cuda")
    print("Loaded SAM3 + DepthAnything + RORem")

# W, H used later only for sparse-depth (single-image); multi-image uses first image
W, H = image.size


def segment(img: Image.Image, prompt_names: list) -> list:
    """SAM3: one forward per text concept → all instances of that concept, with
    complete masks. Returns a list of per-instance dicts {label, score, mask}."""
    h, w = np.array(img).shape[:2]
    instances = []
    for name in prompt_names:
        inputs = sam3_processor(images=img, text=name, return_tensors="pt")
        inputs = {k: (v.to(device="cuda", dtype=torch.bfloat16) if torch.is_floating_point(v)
                      else v.to("cuda")) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = sam3_model(**inputs)
        res = sam3_processor.post_process_instance_segmentation(
            outputs, threshold=0.4, mask_threshold=0.5, target_sizes=[(h, w)])[0]
        masks, scores = res.get("masks"), res.get("scores")
        if masks is None:
            continue
        for m, s in zip(masks, scores):
            mb = (m.cpu().numpy() if hasattr(m, "cpu") else np.asarray(m)).astype(np.float32)
            instances.append({"label": name, "score": float(s), "mask": mb})
    return instances


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a > 0.5, b > 0.5
    inter = (a & b).sum()
    union = (a | b).sum()
    return inter / union if union > 0 else 0.0


def consolidate_instances(instances: list, img_area: int) -> list:
    """Tame instance explosion from the segmenter:
    - dedupe same-label detections that overlap heavily (IoU > 0.5, keep best score)
    - merge same-label instances into one union mask when they are individually
      small (e.g. 10 toy blocks → one 'blocks' peel); keep large ones separate
    - drop speck instances (< 0.05% of the image)"""
    MERGE_AREA_FRAC = 0.02
    MIN_AREA_FRAC = 0.0005

    by_label = {}
    for inst in instances:
        by_label.setdefault(inst["label"], []).append(inst)

    result = []
    for label, group in by_label.items():
        # dedupe overlapping duplicates
        group = sorted(group, key=lambda x: -x["score"])
        kept = []
        for inst in group:
            if all(mask_iou(inst["mask"], k["mask"]) <= 0.5 for k in kept):
                kept.append(inst)

        areas = [float((i["mask"] > 0.5).sum()) for i in kept]
        small = [i for i, a in zip(kept, areas) if a < MERGE_AREA_FRAC * img_area]
        large = [i for i, a in zip(kept, areas) if a >= MERGE_AREA_FRAC * img_area]

        # Large instances are always kept individually (e.g. the two chairs)
        result.extend(large)

        # Merge small same-label instances only when SPATIALLY CLUSTERED
        # (connected components of their combined, dilated mask). Toys piled on a
        # table merge into one peel; toys across the room on a shelf stay separate.
        if small:
            union = np.zeros_like(kept[0]["mask"], dtype=np.uint8)
            for i in small:
                union |= (i["mask"] > 0.5).astype(np.uint8)
            link = cv2.dilate(union, np.ones((25, 25), np.uint8), iterations=2)
            n_comp, comp = cv2.connectedComponents(link)
            groups = 0
            for c in range(1, n_comp):
                region = comp == c
                grp = [i for i in small if (region & (i["mask"] > 0.5)).any()]
                if not grp:
                    continue
                gmask = np.zeros_like(kept[0]["mask"])
                for i in grp:
                    gmask = np.maximum(gmask, i["mask"])
                if (gmask > 0.5).sum() >= MIN_AREA_FRAC * img_area:
                    result.append({"label": label, "score": max(i["score"] for i in grp),
                                   "mask": gmask, "merged": len(grp) > 1})
                    groups += 1
            if len(small) > 1:
                print(f"  '{label}': {len(small)} small instances -> {groups} clustered group(s)")

    # Cross-label NMS: the VLM names the same object differently across images
    # ("blue bottle" vs "blue water bottle", "black mouse" vs "gray mouse"), and
    # those synonyms are all prompted on every image — so the same physical object
    # gets segmented multiple times under different labels. Geometry settles it:
    # if two detections cover the same pixels, they are one object — keep the
    # highest-scoring, drop the rest. (Mask geometry, so no name matching needed.)
    XLABEL_IOU = 0.5
    result = sorted(result, key=lambda x: -x["score"])
    deduped = []
    for inst in result:
        dup_of = next((k for k in deduped if mask_iou(inst["mask"], k["mask"]) > XLABEL_IOU), None)
        if dup_of is None:
            deduped.append(inst)
        elif inst["label"] != dup_of["label"]:
            print(f"  cross-label dup: '{inst['label']}' ≈ '{dup_of['label']}' "
                  f"(IoU>{XLABEL_IOU}) — dropped")
    return deduped


def get_depth_map(img: Image.Image) -> np.ndarray:
    inputs = depth_processor(images=img, return_tensors="pt")
    inputs = {k: v.to("cuda") for k, v in inputs.items()}
    with torch.no_grad():
        out = depth_model(**inputs)
    d = out.predicted_depth.squeeze().cpu().float().numpy()
    return cv2.resize(d.astype(np.float32), img.size, interpolation=cv2.INTER_LINEAR)


def recover_metric_scale(depth_rel: np.ndarray, sparse_path: str) -> np.ndarray:
    """Least-squares fit of scale/shift aligning relative depth to sparse metric depth.
    sparse file: .npy array of shape (N, 3) with rows (row, col, metric_depth)."""
    sparse = np.load(sparse_path)
    rows = sparse[:, 0].astype(int)
    cols = sparse[:, 1].astype(int)
    z = sparse[:, 2]
    d = depth_rel[rows, cols]
    A = np.stack([d, np.ones_like(d)], axis=1)
    (s, t), *_ = np.linalg.lstsq(A, z, rcond=None)
    print(f"  Scale recovery: scale={s:.4f}, shift={t:.4f} from {len(z)} sparse points")
    return s * depth_rel + t


def build_occlusion_order(instances, depth_map, band_px=6, margin_frac=0.04, rng=None,
                          return_graph=False):
    """Front-to-back peel order via a pairwise occlusion graph + topological sort.

    Peeling must remove an occluder *before* what it hides, so the ordering
    question is fundamentally pairwise (who is in front of whom *where they
    meet*) — not a global depth sort. DepthAnythingV2 outputs DISPARITY
    (higher = nearer), verified empirically.

    For each adjacent mask pair we compare the median disparity in their shared
    contact band (sampled just *inside* each, away from the unreliable border):
    the nearer side there is the occluder → directed edge occluder→occluded.
    Two extra signals: a physical-support edge (a small object resting in
    another's footprint is peeled first), and, when no edge is decisive,
    objects fall back to their global nearest-point disparity. A topological
    sort then yields the peel order; cycles from boundary noise are broken at
    their weakest (smallest-margin) edge.

    With rng, free choices among available nodes are made at random instead of
    nearest-first, so different seeds enumerate different orders that all respect
    every kept edge.

    Returns instances reordered front (peel first) → back; with return_graph, also
    the edge list [(occluder uid, occluded uid, weight)] as built (before cycle
    breaking) and the list of edges dropped to break cycles.
    """
    n = len(instances)
    if n <= 1:
        return (list(instances), [], []) if return_graph else list(instances)

    H, W = depth_map.shape
    spread = float(np.percentile(depth_map, 95) - np.percentile(depth_map, 5)) + 1e-9
    tau = margin_frac * spread                      # min disparity gap to call an edge
    masks = [inst["mask"] > 0.5 for inst in instances]
    k = np.ones((2 * band_px + 1, 2 * band_px + 1), np.uint8)
    dil = [cv2.dilate(m.astype(np.uint8), k) > 0 for m in masks]
    ero = [cv2.erode(m.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0 for m in masks]

    centroid_y = [float(np.where(m)[0].mean()) if m.sum() else 0.0 for m in masks]

    def rests_on(ai, bi):
        """a rests ON TOP of b: a is smaller, makes REAL mask contact with b
        (not mere bbox overlap), and the contact is at/below a's centre
        (gravity — a sits above where it touches b)."""
        am, bm = masks[ai], masks[bi]
        if am.sum() == 0 or bm.sum() == 0 or am.sum() >= bm.sum():
            return False
        contact = dil[ai] & bm
        if contact.sum() < max(12, 0.05 * np.sqrt(am.sum())):
            return False
        return float(np.where(contact)[0].mean()) >= centroid_y[ai]

    # edges[a].add(b)  ==  "a occludes b, peel a first"; weight = decision margin
    edges = {i: set() for i in range(n)}
    weight = {}
    for a in range(n):
        for b in range(a + 1, n):
            # contact band: a's interior touching b, and b's interior touching a
            a_band = ero[a] & dil[b]
            b_band = ero[b] & dil[a]
            occ = None
            if a_band.sum() >= 8 and b_band.sum() >= 8:
                da = float(np.median(depth_map[a_band]))
                db = float(np.median(depth_map[b_band]))
                if abs(da - db) >= tau:                       # decisive depth call
                    occ, hid, w, why = ((a, b, abs(da - db), "depth") if da > db
                                        else (b, a, abs(da - db), "depth"))
            if occ is None:                                   # depth indecisive → support tiebreak
                if rests_on(a, b):
                    occ, hid, w, why = a, b, tau, "support"
                elif rests_on(b, a):
                    occ, hid, w, why = b, a, tau, "support"
            if occ is not None:
                edges[occ].add(hid)
                weight[(occ, hid)] = w
                if os.environ.get("IRIS_DEBUG_OCCLUSION"):
                    print(f"    edge: {instances[occ]['uid']} -> {instances[hid]['uid']} "
                          f"({why}, w={w:.1f})")

    graph = [(instances[a]["uid"], instances[b]["uid"], float(weight[(a, b)]))
             for a in edges for b in edges[a]]
    dropped = []

    # Kahn topological sort; ties + free choices broken by nearest-point disparity
    nearest = {i: inst["nearest"] for i, inst in enumerate(instances)}
    indeg = {i: 0 for i in range(n)}
    for a in edges:
        for b in edges[a]:
            indeg[b] += 1

    order = []
    avail = [i for i in range(n) if indeg[i] == 0]
    placed = set()
    while len(order) < n:
        if not avail:                               # cycle: drop weakest remaining edge
            rem = [(weight[(a, b)], a, b) for a in edges for b in edges[a]
                   if a not in placed and b not in placed]
            if not rem:
                avail = [i for i in range(n) if i not in placed]
            else:
                _, a, b = min(rem)
                edges[a].discard(b)
                dropped.append((instances[a]["uid"], instances[b]["uid"]))
                indeg[b] -= 1
                if indeg[b] == 0:
                    avail.append(b)
                continue
        if rng is None:
            # among available, peel the nearest (highest disparity) first
            avail.sort(key=lambda i: nearest[i], reverse=True)
            cur = avail.pop(0)
        else:
            cur = avail.pop(int(rng.integers(len(avail))))
        if cur in placed:
            continue
        order.append(cur)
        placed.add(cur)
        for b in edges[cur]:
            indeg[b] -= 1
            if indeg[b] == 0 and b not in placed:
                avail.append(b)

    ordered = [instances[i] for i in order]
    return (ordered, graph, dropped) if return_graph else ordered


_inpaint_calls = 0


def remove_object(img: Image.Image, mask: np.ndarray) -> Image.Image:
    """image + mask → image with object removed.

    Inpainting runs on a padded square CROP around the mask (not the whole frame),
    so the model sees the region at high effective resolution. Only the masked
    pixels are feather-composited back, so the rest of the image is untouched —
    no cumulative blur across peels. The crop is filled by RORem (SDXL-inpainting
    UNet fine-tuned for object removal)."""
    img_np = np.array(img)
    h, w = img_np.shape[:2]
    mask_uint8 = (mask > 0.5).astype(np.uint8) * 255
    kernel = np.ones((25, 25), np.uint8)          # generous dilation to catch soft shadows
    mask_uint8 = cv2.dilate(mask_uint8, kernel, iterations=1)

    # Square crop around the mask with 35% context padding, clamped to the image
    ys, xs = np.where(mask_uint8 > 0)
    if len(ys) == 0:
        return img
    cy, cx = (ys.min() + ys.max()) / 2, (xs.min() + xs.max()) / 2
    half = max(ys.max() - ys.min(), xs.max() - xs.min()) / 2
    half = min(max(half * 1.35, 192), max(h, w) / 2)   # ≥384px context window
    y0, y1 = int(max(0, cy - half)), int(min(h, cy + half))
    x0, x1 = int(max(0, cx - half)), int(min(w, cx + half))

    crop = Image.fromarray(img_np[y0:y1, x0:x1])
    crop_mask = Image.fromarray(mask_uint8[y0:y1, x0:x1])
    cw, ch = crop.size

    global _inpaint_calls
    gen = None
    if args.inpaint_seed is not None:                 # per-step seed: (run seed, peel step)
        gen = torch.Generator("cuda").manual_seed(args.inpaint_seed * 1000 + _inpaint_calls)
    _inpaint_calls += 1
    result = rorem_pipe(
        prompt=ROREM_PROMPT, negative_prompt=ROREM_NEG_PROMPT,
        height=512, width=512,
        image=crop.resize((512, 512)),
        mask_image=crop_mask.resize((512, 512)),
        guidance_scale=1.0, num_inference_steps=25, strength=0.99,
        generator=gen,
    ).images[0].resize((cw, ch), Image.LANCZOS)

    # Feathered composite of the inpainted crop region only
    alpha_full = np.zeros((h, w), dtype=np.float32)
    alpha_full[y0:y1, x0:x1] = mask_uint8[y0:y1, x0:x1] / 255.0
    alpha_full = cv2.GaussianBlur(alpha_full, (31, 31), 0)[..., None]
    result_full = img_np.astype(np.float32).copy()
    result_full[y0:y1, x0:x1] = np.array(result).astype(np.float32)
    out = img_np.astype(np.float32) * (1 - alpha_full) + result_full * alpha_full
    return Image.fromarray(out.clip(0, 255).astype(np.uint8))


def build_object_crop(img: Image.Image, mask: np.ndarray):
    """Masked object on a white background, cropped to its bbox (+pad) and
    letterboxed into a square at 512x512 — the input to image-to-3D. Returns
    (crop, mask_crop): the RGB crop and the letterboxed SAM3 mask (L, 0/255) in
    the same frame, so occlusion-aware backends (Amodal3R) get the exact object
    region without re-deriving it from the white background (which would drop
    white object parts, e.g. a white bottle cap).

    The square padding is essential: a direct resize to 512x512 squishes a
    non-square bbox (e.g. a tall bottle, 2.4:1) into a square, and the image-to-3D
    model then reconstructs the distorted proportions. Padding to a square first
    preserves the object's true aspect ratio."""
    mask_binary = (mask > 0.5).astype(np.uint8)
    rows = np.any(mask_binary, axis=1)
    cols = np.any(mask_binary, axis=0)
    if not rows.any():
        return None, None
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    pad = 20
    rmin, rmax = max(0, rmin - pad), min(img.size[1], rmax + pad)
    cmin, cmax = max(0, cmin - pad), min(img.size[0], cmax + pad)
    img_np = np.array(img).copy()
    img_np[mask_binary == 0] = 255
    crop = img_np[rmin:rmax, cmin:cmax]
    mcrop = (mask_binary[rmin:rmax, cmin:cmax] * 255).astype(np.uint8)
    # letterbox both into a square canvas (preserve aspect ratio), then resize
    h, w = crop.shape[:2]
    side = max(h, w)
    canvas = np.full((side, side, 3), 255, dtype=np.uint8)
    mcanvas = np.zeros((side, side), dtype=np.uint8)
    y0, x0 = (side - h) // 2, (side - w) // 2
    canvas[y0:y0 + h, x0:x0 + w] = crop
    mcanvas[y0:y0 + h, x0:x0 + w] = mcrop
    crop_img = Image.fromarray(canvas).resize((512, 512))
    mask_img = Image.fromarray(mcanvas).resize((512, 512), Image.NEAREST)
    return crop_img, mask_img


if RESUME_B:
    with open(RECORDS_PATH, "rb") as f:
        _saved = pickle.load(f)
    if isinstance(_saved, dict):
        object_records = _saved["records"]
        view_paths = _saved["view_paths"]
    else:
        # backward compat: old single-image format (plain list)
        object_records = _saved
        view_paths = [IMAGE_PATH] + sorted(
            os.path.join(VIEWS_DIR, fp) for fp in os.listdir(VIEWS_DIR) if fp.endswith(".png"))
    print(f"[resume] {len(object_records)} object records, {len(view_paths)} views")
else:
    # Clear stale synthetic views from previous runs — but never when resuming
    # from a peel checkpoint, whose saved views ARE the restore state
    PEEL_CKPT = os.path.join(OUTPUT_DIR, "peel_ckpt.pkl")
    if not (args.resume and os.path.exists(PEEL_CKPT)):
        for d in (VIEWS_DIR, CROPS_DIR):
            for fp in os.listdir(d):
                if fp.endswith(".png"):
                    os.remove(os.path.join(d, fp))

    PEEL_CKPT = os.path.join(OUTPUT_DIR, "peel_ckpt.pkl")
    view_paths = []
    object_records = []
    # peel checkpoint tracks progress across images
    _done_img_idx = 0
    _done_uids_by_img = {}   # img_idx -> set of peeled uids
    _ckpt_view_paths = None
    if args.resume and os.path.exists(PEEL_CKPT):
        with open(PEEL_CKPT, "rb") as f:
            _ck = pickle.load(f)
        _done_img_idx = _ck.get("done_img_idx", 0)
        _done_uids_by_img = _ck.get("done_uids_by_img", {})
        view_paths = _ck["view_paths"]
        object_records = _ck["records"]
        _ckpt_view_paths = view_paths
        print(f"[resume] peel checkpoint: img {_done_img_idx}/{len(image_list)}, "
              f"{sum(len(v) for v in _done_uids_by_img.values())} object(s) peeled")

    for img_idx, img_path in enumerate(image_list):
        img_stem = os.path.splitext(os.path.basename(img_path))[0]

        if img_idx < _done_img_idx:
            print(f"\n[img {img_idx+1}/{len(image_list)}] {img_stem} [resume: already done]")
            continue

        print(f"\n{'='*60}")
        print(f"[img {img_idx+1}/{len(image_list)}] Peeling: {img_stem}")
        print(f"{'='*60}")

        cur_img = Image.open(img_path).convert("RGB")
        cur_H, cur_W = np.array(cur_img).shape[:2]

        # ── Initial segmentation + depth ordering ────────────────────────────
        print("\nInitial segmentation...")
        instances = segment(cur_img, object_names)
        instances = consolidate_instances(instances, cur_H * cur_W)
        label_counts = {}
        for inst in instances:
            label_counts[inst["label"]] = label_counts.get(inst["label"], 0) + 1
            n = label_counts[inst["label"]]
            inst["uid"] = inst["label"] if n == 1 else f"{inst['label']} {n}"
        print(f"Created {len(instances)} instance masks")

        if args.depth and img_idx == 0:
            # RGB-D mode: use the provided metric depth directly (no depth model).
            # Peel ordering wants "higher = nearer" (disparity), so invert the
            # metric depth (nearer = smaller metres -> larger inverse depth).
            print("Using provided RGB-D depth (skipping DepthAnythingV2)...")
            dm = np.load(args.depth).astype(np.float32)
            if dm.shape[:2] != (cur_H, cur_W):
                dm = cv2.resize(dm, (cur_W, cur_H), interpolation=cv2.INTER_NEAREST)
            depth_map = np.where(dm > 1e-3, 1.0 / np.clip(dm, 1e-3, None), 0.0)
        else:
            print("Computing depth map...")
            depth_map = get_depth_map(cur_img)
            if args.sparse_depth and img_idx == 0:
                depth_map = recover_metric_scale(depth_map, args.sparse_depth)
        np.save(os.path.join(OUTPUT_DIR, f"depth_map_{img_stem}.npy"), depth_map)

        for inst in instances:
            mb = inst["mask"] > 0.5
            inst["nearest"] = depth_map[mb].max() if mb.sum() > 0 else -np.inf
        if os.environ.get("IRIS_PEEL_ORDER") == "naive":
            # ABLATION (paper): the pre-occlusion-graph baseline. Peel strictly by
            # each object's single nearest pixel (max disparity) first — no pairwise
            # occlusion graph, no topological sort. An object that is globally behind
            # but happens to have one near pixel gets peeled before its occluder, so
            # RORem inpaints over geometry that is still covered → peel artifacts.
            sorted_instances = sorted(instances, key=lambda inst: inst["nearest"], reverse=True)
            print("Peeling order (ABLATION: naive nearest-depth, NO topological sort):")
        else:
            _rng = (np.random.default_rng(args.peel_order_seed)
                    if args.peel_order_seed is not None else None)
            sorted_instances, occ_graph, occ_dropped = build_occlusion_order(
                instances, depth_map, rng=_rng, return_graph=True)
            print("Peeling order (occlusion graph + topological sort, front first"
                  + (f", random valid order seed {args.peel_order_seed}):" if _rng else "):"))
            if img_idx == 0:
                # graph + order for ordering evaluation and order-invariance analysis;
                # masks are the ORIGINAL (pre-peel) instance masks in peel order
                with open(os.path.join(OUTPUT_DIR, "peel_order.json"), "w") as f:
                    json.dump({"order": [x["uid"] for x in sorted_instances],
                               "edges": occ_graph, "dropped_for_cycles": occ_dropped,
                               "peel_order_seed": args.peel_order_seed,
                               "inpaint_seed": args.inpaint_seed}, f, indent=2)
                np.savez_compressed(os.path.join(OUTPUT_DIR, "peel_order_masks.npz"),
                                    masks=np.stack([x["mask"] > 0.5 for x in sorted_instances]),
                                    uids=np.array([x["uid"] for x in sorted_instances]))
        for i, inst in enumerate(sorted_instances, 1):
            print(f"  {i}. {inst['uid']} (nearest-point disparity: {inst['nearest']:.3f})")

        if os.environ.get("IRIS_DUMP_INSTANCES") and img_idx == 0:
            # paper figure: the ORIGINAL pre-peel instance masks (all objects present,
            # properly occluding) + depth — the exact inputs to build_occlusion_order.
            pickle.dump([{"uid": x["uid"], "mask": x["mask"]} for x in instances],
                        open(os.path.join(OUTPUT_DIR, "instances_orig.pkl"), "wb"))
            print(f"  [dump] {len(instances)} original instance masks -> instances_orig.pkl")
            if os.environ.get("IRIS_DUMP_EXIT"):
                sys.exit(0)

        # ── Iterative occlusion peeling ──────────────────────────────────────
        print("\nIterative occlusion peeling...")
        # base view for this image (view_paths may already contain prior images)
        view_paths.append(img_path)
        current_image = cur_img.copy()
        done_uids = set(_done_uids_by_img.get(img_idx, set()))
        # if resuming mid-image, restore current image state from last saved view
        if done_uids and _ckpt_view_paths:
            current_image = Image.open(view_paths[-1]).convert("RGB")

        for idx, inst in enumerate(sorted_instances):
            uid = inst["uid"]
            if uid in done_uids:
                print(f"\n  Peeling [{idx+1}/{len(sorted_instances)}]: {uid} [resume: already done]")
                continue
            print(f"\n  Peeling [{idx+1}/{len(sorted_instances)}]: {uid}")

            fresh = segment(current_image, [inst["label"]])
            mask, best_iou = inst["mask"], 0.0
            if inst.get("merged") and fresh:
                union = np.zeros_like(fresh[0]["mask"])
                for f in fresh:
                    union = np.maximum(union, f["mask"])
                iou = mask_iou(union, inst["mask"])
                if iou > best_iou:
                    mask, best_iou = union, iou
            else:
                for f in fresh:
                    iou = mask_iou(f["mask"], inst["mask"])
                    if iou > best_iou:
                        mask, best_iou = f["mask"], iou
            if best_iou >= 0.3:
                print(f"    Fresh mask from current image state (IoU {best_iou:.2f})")
            else:
                mask = inst["mask"]
                print("    Re-detection mismatch; using initial mask")

            # Always record the instance mask (uid + view + mask) — registration
            # (Phase D) uses it to find each object's observed points. The crop /
            # image-to-3D work stays gated.
            record_mask = (mask > 0.5).astype(np.uint8)
            if not args.skip_3d:
                crop, crop_mask = build_object_crop(current_image, mask)
                if crop is not None:
                    i = len(object_records)
                    crop.save(os.path.join(CROPS_DIR, f"{i:02d}.png"))
                    crop_mask.save(os.path.join(CROPS_DIR, f"{i:02d}_mask.png"))
                    print("    Saved object crop (+SAM3 mask) for deferred image-to-3D")
            object_records.append((uid, len(view_paths) - 1, record_mask, None))

            current_image = remove_object(current_image, mask)
            view_path = os.path.join(
                VIEWS_DIR, f"{img_stem}_view_{idx+1:02d}_{uid.replace(' ', '_')}.png")
            current_image.save(view_path)
            view_paths.append(view_path)
            print(f"    Saved synthetic view: {view_path}")

            done_uids.add(uid)
            _done_uids_by_img[img_idx] = done_uids
            with open(PEEL_CKPT, "wb") as f:
                pickle.dump({"done_img_idx": img_idx, "done_uids_by_img": _done_uids_by_img,
                             "view_paths": view_paths, "records": object_records}, f)

        print(f"\nPeeled {img_stem}: {len(sorted_instances)} objects")
        _done_img_idx = img_idx + 1

    print(f"\nPeeling complete: {len(view_paths)} views, {len(object_records)} object point clouds")
    with open(RECORDS_PATH, "wb") as f:
        pickle.dump({"records": object_records, "view_paths": view_paths}, f)
    if os.path.exists(PEEL_CKPT):
        os.remove(PEEL_CKPT)

    free_cuda(sam3_model, sam3_processor, depth_model, depth_processor, rorem_pipe)

if args.stop_after_peeling:
    print(f"\nStopped after peeling (--stop_after_peeling). "
          f"Synthetic views in {VIEWS_DIR}/")
    sys.exit(0)

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE B2: DEFERRED IMAGE-TO-3D (Amodal3R / TRELLIS)
# Runs after the iris-env models are freed, so the ~8-10 GB worker has the GPU to
# itself. Resumable: each object's point cloud is checkpointed into the records
# file, so a crash only costs the object in flight.
# ═══════════════════════════════════════════════════════════════════════════════
if not args.skip_3d:
    # records without a saved crop (object too small to crop) carry no image-to-3D;
    # they still get an instance mask for registration, just skip them here.
    need = [i for i, rec in enumerate(object_records) if rec[3] is None
            and os.path.exists(os.path.join(CROPS_DIR, f"{i:02d}.png"))]
    if need:
        backend = args.image3d
        print("\n" + "=" * 60)
        print(f"[Phase B2] {backend.upper()} image-to-3D for {len(need)} object(s)")
        print("=" * 60)
        object_records = list(object_records)
        worker = Image3DWorker(backend)
        try:
            for i in need:
                uid, vi, m, _ = object_records[i]
                # uid is the Phase-A object label; TRELLIS ignores it. amodal3r
                # additionally consumes the SAM3 mask (saved alongside the crop).
                mask_path = (os.path.join(CROPS_DIR, f"{i:02d}_mask.png")
                             if backend == "amodal3r" else None)
                pc = worker.pointcloud(os.path.join(CROPS_DIR, f"{i:02d}.png"),
                                       text=uid, mask_path=mask_path)
                object_records[i] = (uid, vi, m, pc)
                with open(RECORDS_PATH, "wb") as f:
                    pickle.dump(object_records, f)   # checkpoint after each object
                print(f"  {uid}: {backend} point cloud {pc.shape}")
        finally:
            worker.close()

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE C: MULTI-VIEW SCENE RECONSTRUCTION (VGGT)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("[Phase C] Multi-view scene reconstruction (VGGT)")
print("=" * 60)

PM_PATH = os.path.join(OUTPUT_DIR, "vggt_pointmaps.npz")
if args.resume and os.path.exists(PM_PATH) and "extrinsics" in np.load(PM_PATH, allow_pickle=True):
    data = np.load(PM_PATH, allow_pickle=True)
    world_points, conf = data["world_points"], data["conf"]
    extrinsics, intrinsics = data["extrinsics"], data["intrinsics"]
    view_paths = [str(p) for p in data["view_paths"]]
    images_rgb = data["images_rgb"] if "images_rgb" in data else None   # (V,Hv,Wv,3) in [0,1]
    print(f"[resume] Loaded VGGT point maps: {world_points.shape}")
else:
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    vggt = VGGT.from_pretrained("facebook/VGGT-1B").to("cuda", dtype=torch.float32).eval()
    images_t = load_and_preprocess_images(view_paths).to("cuda", dtype=torch.float32)
    with torch.no_grad():
        predictions = vggt(images_t)

    world_points = predictions["world_points"].squeeze(0).cpu().float().numpy()   # (V, Hv, Wv, 3)
    conf = predictions["world_points_conf"].squeeze(0).cpu().float().numpy()      # (V, Hv, Wv)
    # true per-pixel RGB, pixel-aligned with world_points (ToTensor → [0,1]) for photometric mesh colour
    images_rgb = images_t.permute(0, 2, 3, 1).cpu().float().numpy()               # (V, Hv, Wv, 3)
    # camera poses (for the background plane-fill ray-cast)
    extri, intri = pose_encoding_to_extri_intri(predictions["pose_enc"], images_t.shape[-2:])
    extrinsics = extri.squeeze(0).cpu().float().numpy()   # (V, 3, 4) world-to-camera
    intrinsics = intri.squeeze(0).cpu().float().numpy()   # (V, 3, 3)

    # Sparse-depth metric scaling: make the reconstruction metric so it can be
    # reported in real units (VGGT alone is up-to-scale).
    metric_scale = 1.0
    if args.sparse_depth:
        metric_scale = metric_scale_from_sparse(
            world_points[0], extrinsics[0], args.sparse_depth, image.size[::-1])
        world_points = (world_points * metric_scale).astype(np.float32)
        extrinsics = extrinsics.copy()
        extrinsics[:, :, 3] *= metric_scale               # keep camera centres consistent
        print(f"Metric scale from sparse depth: x{metric_scale:.4f} (recon now metric)")

    print(f"VGGT point maps: {world_points.shape}")
    np.savez_compressed(PM_PATH, world_points=world_points, conf=conf,
                        extrinsics=extrinsics, intrinsics=intrinsics,
                        metric_scale=metric_scale, view_paths=np.array(view_paths),
                        images_rgb=images_rgb)
    free_cuda(vggt, images_t, predictions)

if args.stop_after_vggt:
    print(f"\nStopped after VGGT (--stop_after_vggt). Point maps in {PM_PATH}")
    sys.exit(0)

V, Hv, Wv = conf.shape
scene_mask = conf > VGGT_CONF_THRESHOLD
scene_pc = world_points[scene_mask]
# true per-view colour, pixel-aligned with world_points (fallback: reload+resize raw
# views for old checkpoints). Used to colour the fused cloud / mesh photometrically.
if images_rgb is None:
    images_rgb = np.stack([
        np.asarray(Image.open(p).convert("RGB").resize((Wv, Hv), Image.BILINEAR), np.float32) / 255.0
        for p in view_paths])
print(f"Scene point cloud: {scene_pc.shape}")
np.save(os.path.join(OUTPUT_DIR, "scene_pointcloud.npy"), scene_pc)

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE D: RIGID REGISTRATION + FUSION
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("[Phase D] Object registration + fusion")
print("=" * 60)

import open3d as o3d

scene_diag = np.linalg.norm(scene_pc.max(0) - scene_pc.min(0))


def _R_axis_angle(axis: np.ndarray, ang: float) -> np.ndarray:
    """Rotation matrix for `ang` radians about unit `axis` (Rodrigues)."""
    a = axis / (np.linalg.norm(axis) + 1e-9)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K)


def _R_align(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rotation aligning unit vector a onto unit vector b."""
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    s = np.linalg.norm(v)
    if s < 1e-9:                                   # parallel or antiparallel
        return np.eye(3) if c > 0 else _R_axis_angle(np.array([1.0, 0, 0]), np.pi)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * ((1 - c) / (s * s))


def scene_up_vector(scene_pc: np.ndarray):
    """Gravity/up direction = the FLOOR/TABLE plane normal, plus the floor height.

    Picking the *dominant* plane is fragile: scenes with a large background wall
    (facing the camera) make the WALL dominant, so the up vector ends up along the
    view axis and the whole reconstruction tilts. Instead we extract several planes
    and keep the most *vertical* one — for a roughly-level camera (VGGT's y-down
    convention) the floor normal aligns with the ±Y axis, whereas a camera-facing
    wall's normal lies along ±Z. Among sufficiently large planes, max |n.y| wins."""
    try:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(scene_pc)
        rem = pcd
        best = None   # (verticality, normal, inlier_pts)
        for _ in range(4):                       # peel off up to 4 dominant planes
            if len(rem.points) < 300:
                break
            plane, inl = rem.segment_plane(0.02 * scene_diag, 3, 200)
            n = np.asarray(plane[:3], float); n /= (np.linalg.norm(n) + 1e-9)
            pts = np.asarray(rem.points)[inl]
            if len(inl) / len(scene_pc) > 0.05:  # only planes of meaningful size
                vert = abs(float(n[1]))          # alignment with the vertical (±Y)
                if best is None or vert > best[0]:
                    best = (vert, n.copy(), pts.copy())
            rem = rem.select_by_index(inl, invert=True)
        if best is None:
            raise RuntimeError("no plane")
        n, fpts = best[1], best[2]
        if np.dot(n, scene_pc.mean(0) - fpts.mean(0)) < 0:
            n = -n
        floor_h = float(np.median(fpts @ n))     # floor level along up
        return n, floor_h
    except Exception:
        return np.array([0.0, 0.0, 1.0]), float(np.percentile(scene_pc[:, 2], 5))


def register_object(obj_pc: np.ndarray, target_pts: np.ndarray,
                    scene_up: np.ndarray, floor_h: float = None) -> np.ndarray:
    """Pose a complete generated object so its visible surface coincides with the
    observed scene points (`target_pts`) — i.e. the object is dropped in as an
    *extension* of the existing segment, completing the occluded geometry.

    The previous version compared the full object's bbox diagonal to the diagonal
    of the partial visible patch (shrinking the object) and ran symmetric ICP from
    the object's canonical pose (wrong orientation). This version instead:
      1. trims observed outliers (mask bleed / depth noise);
      2. fixes scale from the patch's lateral extent (below);
      3. tilts the canonical object so its up-axis matches scene gravity (the floor
         normal), fixing 2 of 3 rotation DoF;
      4. searches the remaining yaw about the up-axis (24 angles), and at each yaw
         slides the object (translation-only ICP) so observed points land on its
         surface; keeps the yaw with the smallest observed→object residual.
    The chosen pose minimizes the observed→object distance (every observed point
    lands on the object surface).

    Two design choices keep it from cheating:
      - Scale is fixed BEFORE refine from the observed patch's lateral extent (its
        two largest principal spreads; the depth spread of the thin front shell is
        unreliable). Letting ICP also solve scale lets it shrink the object to
        nestle inside the patch.
      - The refine is translation + yaw only. Gravity fixes the tilt; a full-SO(3)
        ICP would tip the object over to nestle into the partial segment."""
    from scipy.spatial import cKDTree

    # trim observed outliers (mask bleed onto neighbours, VGGT depth noise) that
    # would inflate the scale estimate and drag the fit — keep points within a
    # robust radius of the median.
    med = np.median(target_pts, axis=0)
    rad = np.linalg.norm(target_pts - med, axis=1)
    keep = rad < (np.median(rad) + 2.5 * (np.median(np.abs(rad - np.median(rad))) + 1e-9))
    if keep.sum() >= 50:
        target_pts = target_pts[keep]

    def _spread(P):                                 # principal std (descending)
        w = np.linalg.eigvalsh(np.cov((P - P.mean(0)).T))
        return np.sqrt(np.clip(w, 0, None))[::-1]
    te, oe = _spread(target_pts), _spread(obj_pc)
    s0 = (te[0] + te[1]) / max(oe[0] + oe[1], 1e-8)  # match the two lateral axes

    obj_c = (obj_pc - obj_pc.mean(0)) * s0           # scaled, centered object
    tgt_mean = target_pts.mean(0)
    R_tilt = _R_align(np.array([0.0, 0.0, 1.0]), scene_up)   # canonical +Z → gravity

    best = None   # (resid, aligned_pts)
    for yaw in np.linspace(0, 2 * np.pi, 24, endpoint=False):
        R0 = _R_axis_angle(scene_up, yaw) @ R_tilt
        src = obj_c @ R0.T + tgt_mean
        # translation-only asymmetric ICP: slide the (upright) object so observed
        # points land on its surface — never re-rotates, so tilt stays gravity-true
        for _ in range(10):
            tree = cKDTree(src)
            _, idx = tree.query(target_pts)
            src = src - (src[idx] - target_pts).mean(0)
        resid = cKDTree(src).query(target_pts)[0].mean()
        if best is None or resid < best[0]:
            best = (resid, src)

    aligned = best[1]

    # Floor contact: tabletop objects rest on the floor, but the visible segment
    # (and a slightly squat generated object) often leaves the base floating just
    # above it — e.g. the bottle's diameter is right but it doesn't reach the
    # ground. Stretch the object along the up-axis, anchored at its observed top,
    # so the base touches the floor — extends height, preserves lateral size.
    if floor_h is not None:
        h = aligned @ scene_up
        top_h, base_h = np.percentile(h, 98), np.percentile(h, 2)
        height = top_h - base_h
        gap = base_h - floor_h
        # Only seat to the support surface if the object was actually OBSERVED resting
        # near it. Objects seen well above the floor (a ceiling chandelier, wall-mounted
        # or shelved items) must NOT be dragged down — the floor-contact stretch is for
        # objects that genuinely sit on the floor/table.
        obs_above = float(np.percentile(target_pts @ scene_up, 2)) - floor_h
        on_floor = obs_above < max(0.5 * height, 0.05 * scene_diag)
        if on_floor and height > 1e-6 and 0 < gap < 0.6 * height:   # floating just above support
            factor = (top_h - floor_h) / (top_h - base_h)
            new_h = top_h - (top_h - h) * factor
            aligned = aligned + (new_h - h)[:, None] * scene_up
            print(f"    floor contact: stretched base to floor (gap {gap/scene_diag:.4f} → 0)")
        elif not on_floor:
            print(f"    floor contact: skipped (observed {obs_above/scene_diag:.3f} above floor — not a floor object)")

    print(f"    fit-to-segment: weld residual {best[0] / scene_diag:.4f} (scene-rel)")
    return aligned


def plane_fill_bg(P, conf_mask, haze, rgb, K, ext, scene_diag):
    """Clean flat background = haze-excluded In view + plane-filled object footprints.

    The inpainted footprint regions in the final peeled view are 'mound' geometry
    (haze) — we drop them, fit the table + wall planes by RANSAC, and fill each
    footprint pixel by casting its camera ray onto the NEARER plane (table for open
    table, wall for behind tall objects). A flat continuation: no inpaint mounds, no
    tent ridges, no holes. Returns (xyz, rgb) in the same (gravity) frame as P."""
    from scipy.spatial import cKDTree as _ck
    valid = conf_mask & ~haze
    real_xyz, real_rgb = P[valid], rgb[valid]
    pc = o3d.geometry.PointCloud(); pc.points = o3d.utility.Vector3dVector(real_xyz)
    try:
        tm, ti = pc.segment_plane(0.006 * scene_diag, 3, 1000)          # table (largest)
        wm, _wi = pc.select_by_index(ti, invert=True).segment_plane(0.006 * scene_diag, 3, 1000)  # wall
    except Exception:
        return real_xyz, real_rgb
    hy, hx = np.where(haze)
    if len(hy) == 0:
        return real_xyz, real_rgb
    Kinv = np.linalg.inv(K); Rc2w = ext[:, :3].T; C = -ext[:, :3].T @ ext[:, 3]
    rw = (Rc2w @ (Kinv @ np.c_[hx + 0.5, hy + 0.5, np.ones(len(hx))].T)).T
    rw /= np.linalg.norm(rw, axis=1, keepdims=True)
    def isect(m):
        n = m[:3]; den = rw @ n
        s = (-m[3] - C @ n) / np.where(np.abs(den) < 1e-9, np.nan, den)
        return np.where(s > 0, s, np.inf)
    S = np.minimum(isect(tm), isect(wm)); good = np.isfinite(S)
    if not good.any():
        return real_xyz, real_rgb
    fill = C + S[good, None] * rw[good]
    fill_rgb = real_rgb[_ck(real_xyz).query(fill)[1]]
    print(f"  plane-fill: {len(real_xyz)} real + {int(good.sum())} footprint-fill pts")
    return np.vstack([real_xyz, fill]), np.vstack([real_rgb, fill_rgb])


FUSED_PATH = os.path.join(OUTPUT_DIR, "fused_pointcloud.npy")
FUSED_RGB_PATH = os.path.join(OUTPUT_DIR, "fused_pointcloud_rgb.npy")
OBJECTS_PATH = os.path.join(OUTPUT_DIR, "registered_objects.pkl")
if args.resume and os.path.exists(FUSED_PATH) and os.path.getmtime(FUSED_PATH) > os.path.getmtime(PM_PATH):
    fused_pc = np.load(FUSED_PATH)
    fused_rgb = np.load(FUSED_RGB_PATH) if os.path.exists(FUSED_RGB_PATH) else np.full_like(fused_pc, 0.6)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(fused_pc)
    registered_objects = pickle.load(open(OBJECTS_PATH, "rb")) if os.path.exists(OBJECTS_PATH) else []
    colored_objects = []   # not cached on resume → Phase F falls back to registered_objects
    bg_inview = None; bg_inview_rgb = None   # not cached → Phase F falls back to fused-minus-objects
    print(f"[resume] Loaded fused point cloud: {fused_pc.shape}")
else:
    registered_objects = []   # per-object registered clouds (observed front + Amodal3R back)
    colored_objects = []      # (placed xyz, rgb) per object → colored objects-only mesh
    scene_up, floor_h = scene_up_vector(scene_pc)
    print(f"  scene up (floor normal): {np.round(scene_up, 3)}, floor_h {floor_h:.3f}")

    # ── Background = the FINAL fully-peeled view, minus object footprints ───────
    # The last synthetic view has every object removed, so its VGGT reconstruction
    # is the background table/wall. BUT behind each object that region was RORem-
    # inpainted (colour ghosts / "haze"), which VGGT turns into floating geometry
    # between objects. Drop those object-footprint pixels (dilated for the halo) —
    # they're occluded by the objects anyway, which sit back on top of the holes.
    haze = np.zeros((Hv, Wv), bool)
    for _on, _vi, _m, _pc in object_records:
        haze |= cv2.resize(_m, (Wv, Hv), interpolation=cv2.INTER_NEAREST) > 0.5
    haze = cv2.dilate(haze.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    bg_mask = scene_mask[-1] & ~haze
    fused = [world_points[-1][bg_mask]]
    fused_colors = [images_rgb[-1][bg_mask]]
    print(f"  background from final peeled view: {int(bg_mask.sum())} pts")

    for obj_name, view_idx, mask, obj_pc in object_records:
        print(f"\n  Registering: {obj_name} (view {view_idx})")
        mask_v = cv2.resize(mask, (Wv, Hv), interpolation=cv2.INTER_NEAREST) > 0.5
        valid = mask_v & scene_mask[view_idx]
        target_pts = world_points[view_idx][valid]
        target_rgb = images_rgb[view_idx][valid]
        if len(target_pts) < 50:
            print(f"    Only {len(target_pts)} mask points in scene, skipping")
            continue
        if obj_pc is None:
            print(f"    No object point cloud (--skip_3d), skipping registration")
            continue
        obj_rgb = obj_pc[:, 3:6] if obj_pc.shape[1] >= 6 else None   # gaussian colour
        obj_pc = obj_pc[:, :3]
        aligned = register_object(obj_pc, target_pts, scene_up, floor_h)
        if obj_rgb is not None:        # full placed object + colour (objects-only render)
            colored_objects.append((aligned.copy(), obj_rgb))
        # 3D inpainting: keep the observed front exactly (real -> true size &
        # position) and graft on ONLY the generated geometry that isn't already
        # observed — the occluded back/sides. Never overwrites the visible surface.
        from scipy.spatial import cKDTree
        d_obs = cKDTree(target_pts).query(aligned)[0]
        # OBJECT-relative threshold: a scene-relative one is larger than a small
        # object when a big floor inflates scene_diag, rejecting the whole completion.
        obj_diag = float(np.linalg.norm(aligned.max(0) - aligned.min(0))) + 1e-9
        m_new = d_obs > 0.03 * obj_diag
        new_pts = aligned[m_new]
        new_rgb = obj_rgb[m_new] if obj_rgb is not None else np.full((int(m_new.sum()), 3), 0.6)
        placed = np.concatenate([target_pts, new_pts], axis=0)
        print(f"    kept {len(target_pts)} observed + grafted {len(new_pts)} generated")
        # object = real observed front (added explicitly; the background no longer
        # carries it) + Amodal3R-generated occluded back
        fused.append(target_pts);  fused_colors.append(target_rgb)
        fused.append(new_pts);     fused_colors.append(np.clip(new_rgb, 0, 1))
        registered_objects.append(placed)

    fused_pc = np.concatenate(fused, axis=0)
    fused_rgb = np.concatenate(fused_colors, axis=0)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(fused_pc)
    pcd.colors = o3d.utility.Vector3dVector(fused_rgb)   # carry RGB through downsample/denoise (o3d averages colours)
    pcd = pcd.voxel_down_sample(voxel_size=0.005 * scene_diag)
    # denoise: drop floaters (VGGT depth noise + per-view background ghosting) that
    # would otherwise become spurious blobs in the mesh.
    n_before = len(pcd.points)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    fused_pc = np.asarray(pcd.points)
    fused_rgb = np.asarray(pcd.colors)                  # stays aligned with fused_pc
    print(f"\nFused point cloud (downsampled + denoised): {fused_pc.shape} "
          f"(removed {n_before - len(fused_pc)} outliers)")

    # ── Gravity-align the whole reconstruction ──────────────────────────────
    # VGGT's world frame is arbitrary (often tilted ~45°), so the output looks
    # tilted / objects look "floating" in a viewer even when registration is
    # correct. Rotate everything so the estimated floor normal (scene_up) points
    # up (+Y). Camera extrinsics rotate too so the background plane-fill ray-cast stays valid.
    def _R_to(a, b):
        a = a / (np.linalg.norm(a) + 1e-9); b = b / (np.linalg.norm(b) + 1e-9)
        v = np.cross(a, b); s = float(np.linalg.norm(v)); c = float(a @ b)
        if s < 1e-9:
            return np.eye(3) if c > 0 else -np.eye(3)
        K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        return np.eye(3) + K + K @ K * ((1 - c) / (s * s))
    R_g = _R_to(np.asarray(scene_up, float), np.array([0.0, 1.0, 0.0]))
    fused_pc = fused_pc @ R_g.T
    registered_objects = [np.asarray(o) @ R_g.T for o in registered_objects]
    # pure per-object Amodal3R clouds (no observed-front graft → no seam) in the
    # gravity frame, for clean per-object meshing in Phase F
    colored_objects = [(np.asarray(a) @ R_g.T, np.asarray(c)) for a, c in colored_objects]
    world_points = world_points @ R_g.T                       # for the background plane-fill
    extrinsics = np.asarray(extrinsics, float).copy()
    extrinsics[:, :, :3] = extrinsics[:, :, :3] @ R_g.T       # world2cam rot
    # clean flat background: haze-excluded In view + plane-filled footprints (table/
    # wall planes), so no inpaint mounds, no holes. Same gravity frame as the objects.
    bg_inview, bg_inview_rgb = plane_fill_bg(
        world_points[-1], scene_mask[-1], haze, images_rgb[-1],
        intrinsics[-1], extrinsics[-1], scene_diag)
    print(f"  gravity-aligned output (scene_up {np.round(scene_up,2)} -> +Y)")

    np.save(FUSED_PATH, fused_pc)
    np.save(FUSED_RGB_PATH, fused_rgb)
    pcd.points = o3d.utility.Vector3dVector(fused_pc)   # keep the .ply in the same gravity-aligned frame as the .npy + mesh
    pcd.colors = o3d.utility.Vector3dVector(fused_rgb)  # true-colour fused cloud
    o3d.io.write_point_cloud(os.path.join(OUTPUT_DIR, "fused_pointcloud.ply"), pcd)
    with open(OBJECTS_PATH, "wb") as f:
        pickle.dump(registered_objects, f)   # saved after gravity rotation so resume loads correct frame

    # Save the OBSERVED surface and the OCCLUDED completions separately, so
    # evaluation can score them honestly: visible-region F1 on the observed cloud,
    # occluded-surface recall on the completions (fused[0] is scene_pc; fused[1:]
    # are the grafted occluded back/sides per object).
    np.save(os.path.join(OUTPUT_DIR, "observed_pointcloud.npy"), fused[0] @ R_g.T)
    if len(fused) > 1:
        np.save(os.path.join(OUTPUT_DIR, "completion_pointcloud.npy"),
                np.concatenate(fused[1:], 0) @ R_g.T)

    # ── Colored objects-only reconstruction (SceneComplete-style) ────────────
    # Compose the placed, COLORED per-object gaussians (no floor/wall) into a
    # clean colored point cloud + Poisson surface mesh — a complete textured
    # object scene, distinct from the semantic mesh.
    if colored_objects:
        co_xyz = np.concatenate([a for a, _ in colored_objects], 0)   # already gravity-aligned
        co_rgb = np.clip(np.concatenate([c for _, c in colored_objects], 0), 0, 1)
        cpcd = o3d.geometry.PointCloud()
        cpcd.points = o3d.utility.Vector3dVector(co_xyz)
        cpcd.colors = o3d.utility.Vector3dVector(co_rgb)
        o3d.io.write_point_cloud(os.path.join(OUTPUT_DIR, "objects_colored.ply"), cpcd)
        try:
            # mesh each object SEPARATELY (cluster → Poisson → merge) so Poisson
            # doesn't web nearby objects together — cleaner than one combined fit.
            # Each cluster is downsampled and gets a cheap O(n) normal orientation
            # (outward from its centre), so this stays ~seconds even under heavy CPU
            # contention — vs. the Riemannian-graph orient that could stall for minutes.
            lbls = np.array(cpcd.cluster_dbscan(eps=0.03 * scene_diag, min_points=30))
            omesh = o3d.geometry.TriangleMesh()
            for k in range(lbls.max() + 1):
                sub = cpcd.select_by_index(np.where(lbls == k)[0])
                if len(sub.points) < 80:
                    continue
                if len(sub.points) > 3000:                  # cap cost; Poisson quality unaffected at this scale
                    sub = sub.voxel_down_sample(0.004 * scene_diag)
                sub.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
                    radius=0.04 * scene_diag, max_nn=30))
                sub.orient_normals_towards_camera_location(sub.get_center())   # O(n) inward...
                sub.normals = o3d.utility.Vector3dVector(-np.asarray(sub.normals))  # ...flip to outward
                m, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(sub, depth=8)
                m.remove_vertices_by_mask(np.asarray(dens) < np.quantile(np.asarray(dens), 0.05))
                omesh += m.crop(sub.get_axis_aligned_bounding_box())
            omesh.compute_vertex_normals()
            o3d.io.write_triangle_mesh(os.path.join(OUTPUT_DIR, "objects_mesh.ply"), omesh)
            print(f"  colored objects mesh: {len(omesh.vertices)} verts, "
                  f"{lbls.max()+1} objects (objects_colored.ply + objects_mesh.ply)")
        except Exception as e:
            print(f"  colored objects mesh skipped: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE F: MESH GENERATION — per-object Poisson + background Poisson
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("[Phase F] Semantic mesh generation (per-object + background Poisson)")
print("=" * 60)

# Meshing the full fused cloud in one pass creates artefacts at every
# object-background boundary: Poisson builds webs across nearby surfaces and
# bleeds geometry between adjacent objects. Instead: mesh each registered object
# independently (outward normals), mesh the remaining background (inward normals
# toward camera origin), then merge. No shared indicator function across
# object/scene boundaries → clean seams.

from scipy.spatial import cKDTree as _cKDTree   # may not be imported in resume path

def _prep_obj_pcd(pts, scene_diag):
    """Denoise + globally-consistent OUTWARD normals for one isolated object cloud.
    Consistent tangent-plane normals (propagated over an MST, not a crude centroid
    flip) are what let Poisson recover a round, seam-free surface instead of a lumpy
    blob — the single biggest lever on object-mesh quality. Returns a pcd or None."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(pts))
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=16, std_ratio=2.0)
    if len(pcd.points) > 9000:
        pcd = pcd.voxel_down_sample(0.004 * scene_diag)
    obj_pts = np.asarray(pcd.points)
    if len(obj_pts) < 80:
        return None
    obj_ext = float(np.linalg.norm(obj_pts.max(0) - obj_pts.min(0))) + 1e-6
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.06 * obj_ext, max_nn=60))
    try:
        pcd.orient_normals_consistent_tangent_plane(30)           # globally consistent
        n = np.asarray(pcd.normals); v = obj_pts - obj_pts.mean(0)
        if (np.einsum('ij,ij->i', n, v) > 0).mean() < 0.5:        # ensure outward
            pcd.normals = o3d.utility.Vector3dVector(-n)
    except Exception:                                             # fallback: cheap centroid flip
        pcd.orient_normals_towards_camera_location(obj_pts.mean(0))
        pcd.normals = o3d.utility.Vector3dVector(-np.asarray(pcd.normals))
    return pcd

def _poisson_trim(pcd, depth):
    """Screened Poisson on a prepared cloud; density-trim + bbox-crop. Returns mesh or None."""
    if len(pcd.points) < 80:
        return None
    m, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth, linear_fit=True)
    dens = np.asarray(dens)
    m.remove_vertices_by_mask(dens < np.quantile(dens, 0.04))
    m = m.crop(pcd.get_axis_aligned_bounding_box())
    return m if len(m.vertices) > 0 else None

combined_mesh = o3d.geometry.TriangleMesh()
col_ref_pts = [fused_pc]; col_ref_rgb = [fused_rgb]   # colour reference for final per-vertex colouring

# Source: the PURE Amodal3R cloud (no observed-front graft → no seam ridge) when
# available; fall back to the grafted registered_objects on resume.
obj_sources = [(np.asarray(o), c) for o, c in
               (colored_objects if colored_objects else [(o, None) for o in registered_objects])
               if len(np.asarray(o)) >= 80]
# colour reference: object verts get their exact Amodal3R gaussian colour
for obj_pts, obj_rgb in obj_sources:
    if obj_rgb is not None:
        col_ref_pts.append(obj_pts); col_ref_rgb.append(np.clip(np.asarray(obj_rgb), 0, 1))

# ── Per-object meshes: NKSR (smooth, watertight) with screened-Poisson fallback ──
NKSR_PY = os.environ.get("IRIS_NKSR_PY", sys.executable.replace("/envs/iris/", "/envs/nksr/"))
nksr_ok = False
if obj_sources and os.environ.get("IRIS_OBJ_MESHER", "nksr") != "poisson" and os.path.exists(NKSR_PY):
    try:
        nin = os.path.join(OUTPUT_DIR, "_nksr_in.npz"); nout = os.path.join(OUTPUT_DIR, "objects_nksr.ply")
        np.savez(nin, **{f"obj_{k}": p for k, (p, _) in enumerate(obj_sources)})
        nscript = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nksr_mesh.py")
        r = subprocess.run([NKSR_PY, nscript, nin, nout], capture_output=True, text=True, timeout=600)
        if "NKSR_MESH_OK" in r.stdout and os.path.exists(nout):
            om = o3d.io.read_triangle_mesh(nout)
            if len(om.vertices) > 0:
                combined_mesh += om; nksr_ok = True
                print(f"  objects (NKSR): {r.stdout.strip().splitlines()[-1]}")
        if not nksr_ok:
            print(f"  NKSR objects empty/failed → Poisson fallback ({r.stderr.strip()[-160:]})")
    except Exception as e:
        print(f"  NKSR objects error → Poisson fallback ({e})")
if not nksr_ok:                                           # screened Poisson per object
    for k, (obj_pts, obj_rgb) in enumerate(obj_sources):
        pcd_obj = _prep_obj_pcd(obj_pts, scene_diag)
        if pcd_obj is None:
            continue
        m = _poisson_trim(pcd_obj, depth=9)
        if m is None:
            print(f"  object {k}: Poisson returned empty mesh, skipped")
            continue
        combined_mesh += m.filter_smooth_taubin(number_of_iterations=10)
        print(f"  object {k}: {len(obj_pts)} pts → {len(m.vertices)} verts, {len(m.triangles)} faces")

# ── Background: plane-filled flat table+wall, meshed with NKSR (Poisson fallback) ──
if bg_inview is not None:
    bg_pts, bg_rgb = bg_inview, bg_inview_rgb
elif registered_objects:                                     # resume fallback: exact subtraction
    keep = _cKDTree(np.concatenate([np.asarray(o) for o in registered_objects], 0)).query(fused_pc, workers=4)[0] > 0.02 * scene_diag
    bg_pts, bg_rgb = fused_pc[keep], fused_rgb[keep]
else:
    bg_pts, bg_rgb = fused_pc, fused_rgb
print(f"  background: {len(bg_pts)} pts")
col_ref_pts.append(bg_pts); col_ref_rgb.append(np.clip(bg_rgb, 0, 1))   # colour bg verts (incl. fill)

bg_done = False
if len(bg_pts) >= 200 and nksr_ok:                            # NKSR (open-surface mode)
    try:
        nbin = os.path.join(OUTPUT_DIR, "_nksr_bg_in.npz"); nbout = os.path.join(OUTPUT_DIR, "background_nksr.ply")
        np.savez(nbin, obj_0=bg_pts.astype(np.float32))
        rb = subprocess.run([NKSR_PY, os.path.join(os.path.dirname(os.path.abspath(__file__)), "nksr_mesh.py"),
                             nbin, nbout, "bg"], capture_output=True, text=True, timeout=600)
        if "NKSR_MESH_OK" in rb.stdout and os.path.exists(nbout):
            bg_m = o3d.io.read_triangle_mesh(nbout)
            if len(bg_m.vertices) > 0:
                combined_mesh += bg_m; bg_done = True
                print(f"  background (NKSR): {rb.stdout.strip().splitlines()[-1]}")
        if not bg_done:
            print(f"  NKSR background empty/failed → Poisson fallback ({rb.stderr.strip()[-160:]})")
    except Exception as e:
        print(f"  NKSR background error → Poisson fallback ({e})")
if not bg_done and len(bg_pts) >= 80:                         # screened-Poisson fallback (hole-filling)
    bg_pcd = o3d.geometry.PointCloud(); bg_pcd.points = o3d.utility.Vector3dVector(bg_pts)
    bg_pcd = bg_pcd.voxel_down_sample(0.005 * scene_diag)
    bg_pcd, _ = bg_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    bg_pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.03 * scene_diag, max_nn=30))
    bg_pcd.orient_normals_towards_camera_location(np.zeros(3))
    bg_m, _bdn = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(bg_pcd, depth=10, linear_fit=True)
    _bdn = np.asarray(_bdn); bg_m.remove_vertices_by_mask(_bdn < np.quantile(_bdn, 0.01))
    bg_m = bg_m.crop(bg_pcd.get_axis_aligned_bounding_box())
    _tc, _nt, _ = bg_m.cluster_connected_triangles(); _nt = np.asarray(_nt)
    if len(_nt):
        bg_m.remove_triangles_by_mask((_nt < 0.05 * _nt.max())[np.asarray(_tc)]); bg_m.remove_unreferenced_vertices()
    if len(bg_m.vertices) > 0:
        combined_mesh += bg_m
        print(f"  background (Poisson): {len(bg_m.vertices)} verts, {len(bg_m.triangles)} faces")

# ── Fragment removal ─────────────────────────────────────────────────────────
tri_clusters, n_tri, _ = combined_mesh.cluster_connected_triangles()
n_tri = np.asarray(n_tri)
n_small = 0
if len(n_tri):
    small = n_tri < 0.01 * n_tri.max()
    combined_mesh.remove_triangles_by_mask(small[np.asarray(tri_clusters)])
    combined_mesh.remove_unreferenced_vertices()
    n_small = int(small.sum())
mesh_o3d = combined_mesh
verts_world = np.asarray(mesh_o3d.vertices)
print(f"Mesh (Poisson): {len(verts_world)} vertices, {len(mesh_o3d.triangles)} faces "
      f"({n_small} fragment component(s) removed)")

# colour each vertex from the nearest reference point: true photometric RGB —
# background from the input/peeled views (fused_pc), object verts from the pure
# Amodal3R gaussian colours (object meshes are built from the pure cloud, which
# isn't in fused_pc, so it must be in the colour reference).
from scipy.spatial import cKDTree as _vertKDTree
col_ref_pts = np.concatenate(col_ref_pts, 0)
col_ref_rgb = np.concatenate(col_ref_rgb, 0)
_, vi = _vertKDTree(col_ref_pts).query(verts_world, workers=4)
mesh_o3d.vertex_colors = o3d.utility.Vector3dVector(np.clip(col_ref_rgb[vi], 0, 1))
mesh_o3d.compute_vertex_normals()
mesh_path = os.path.join(OUTPUT_DIR, "final_mesh.ply")
o3d.io.write_triangle_mesh(mesh_path, mesh_o3d)

print(f"\n{'='*60}")
print("IRIS pipeline complete!")
print(f"  Synthetic views:    {VIEWS_DIR}/")
print(f"  Fused point cloud:  {OUTPUT_DIR}/fused_pointcloud.ply")
print(f"  Mesh (true colour): {mesh_path}")
print(f"  Objects (colored):  {OUTPUT_DIR}/objects_colored.ply (+ objects_mesh.ply)")
print("=" * 60)
