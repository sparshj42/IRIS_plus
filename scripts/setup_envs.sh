#!/bin/bash
# ============================================================================
# IRIS reproducible setup — creates the conda envs and clones the third-party
# model repos. Portable: uses conda env NAMES (no hardcoded home paths).
#
#   bash scripts/setup_envs.sh            # all envs (iris + amodal3r + trellis + nksr)
#   bash scripts/setup_envs.sh iris amodal3r   # a subset (any combination)
#   bash scripts/setup_envs.sh amodal3r   # just the Amodal3R env (default backend, +repo)
#   bash scripts/setup_envs.sh trellis    # just the TRELLIS env (baseline backend, +repo)
#   bash scripts/setup_envs.sh nksr       # just the NKSR meshing env (optional, builds CUDA ext)
#
# After this: python scripts/fetch_weights.py   (downloads RORem)
# ============================================================================
set -e
cd "$(dirname "$0")/.."
REPO="$PWD"
TARGETS="${*:-all}"
want() {   # want <name> -> true if that env was requested (or "all")
  case " $TARGETS " in *" all "*) return 0;; *" $1 "*) return 0;; *) return 1;; esac
}

# Initialize conda if it's not already on PATH (common in fresh WSL shells)
if ! command -v conda &>/dev/null; then
    for _root in "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/miniforge3" "$HOME/mambaforge" "/opt/conda"; do
        if [ -f "$_root/etc/profile.d/conda.sh" ]; then
            source "$_root/etc/profile.d/conda.sh"
            break
        fi
    done
fi
if ! command -v conda &>/dev/null; then
    echo "ERROR: conda not found. Install miniconda/anaconda first." >&2
    exit 1
fi

pipx() { PYTHONNOUSERSITE=1 conda run -n "$1" pip install "${@:2}"; }

# ---------------------------------------------------------------- main: iris
if want iris; then
  echo ">>> [iris] main pipeline env"
  conda env list | grep -q "^iris " || conda create -n iris python=3.10 -y
  pipx iris torch==2.5.1 torchvision --index-url https://download.pytorch.org/whl/cu121
  if [ -f requirements-iris.txt ]; then
    pipx iris -r requirements-iris.txt --extra-index-url https://download.pytorch.org/whl/cu121
  else
    pipx iris transformers==5.9.0 diffusers==0.38.0 accelerate open3d==0.19.0 \
        trimesh==4.12.2 scikit-image==0.24.0 scikit-learn==1.7.2 opencv-python-headless \
        scipy matplotlib qwen-vl-utils omegaconf einops rembg jinja2 psutil gdown
  fi
  # VGGT: install at the pinned commit with --no-deps to avoid its false numpy<2
  # constraint (numpy 2.x works fine at runtime; the bound is overly conservative).
  # All of vggt's actual runtime deps are already in requirements-iris.txt.
  pipx iris --no-deps \
    "git+https://github.com/facebookresearch/vggt.git@a288dd0f14786c93483e45524328726ab7b1b4ce" || \
    echo "  (VGGT install failed — install manually: pip install --no-deps git+https://github.com/facebookresearch/vggt.git@a288dd0f14786c93483e45524328726ab7b1b4ce)"
  echo ">>> [iris] done"
fi

# ------------------------------------------------------------ TRELLIS (image->3D)
if want trellis; then
  echo ">>> [trellis] env + repo (gaussian/point-cloud path; no mesh CUDA builds)"
  [ -d models/TRELLIS ] || git clone --recurse-submodules https://github.com/microsoft/TRELLIS models/TRELLIS
  ( cd models/TRELLIS && git submodule update --init --recursive )
  conda env list | grep -q "^trellis " || conda create -n trellis python=3.10 -y
  if [ -f requirements-trellis.txt ]; then
    pipx trellis torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu118
    pipx trellis -r requirements-trellis.txt --extra-index-url https://download.pytorch.org/whl/cu118
  else
    pipx trellis torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu118
    pipx trellis pillow imageio imageio-ffmpeg tqdm easydict opencv-python-headless \
        scipy ninja rembg onnxruntime trimesh open3d xatlas pyvista pymeshfix igraph \
        transformers safetensors spconv-cu118
    pipx trellis xformers==0.0.28 --index-url https://download.pytorch.org/whl/cu118
    pipx trellis git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8
  fi
  # kaolin stub: flexicubes imports kaolin.utils.testing.check_tensor at load,
  # but the gaussian path never runs it — a no-op stub avoids the heavy build.
  SP=$(conda run -n trellis python -c "import site; print(site.getsitepackages()[0])")
  mkdir -p "$SP/kaolin/utils"
  printf '' > "$SP/kaolin/__init__.py"
  printf '' > "$SP/kaolin/utils/__init__.py"
  printf 'def check_tensor(*a, **k):\n    return True\n' > "$SP/kaolin/utils/testing.py"
  echo ">>> [trellis] done"
fi

# --------------------------------------------- Amodal3R (default, occlusion-aware image->3D)
# Amodal3R is a TRELLIS fork; IRIS uses its gaussian path only, so we install the same
# lightweight stack as the trellis env (spconv/xformers/utils3d, no mesh CUDA builds) and
# clone the repo onto the path. For the full mesh pipeline see Amodal3R's own setup.sh.
if want amodal3r; then
  echo ">>> [amodal3r] env + repo (default backend; occlusion-aware; gaussian path)"
  [ -d models/Amodal3R ] || git clone https://github.com/Sm0kyWu/Amodal3R models/Amodal3R
  conda env list | grep -q "^amodal3r " || conda create -n amodal3r python=3.10 -y
  pipx amodal3r torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu118
  pipx amodal3r pillow imageio imageio-ffmpeg tqdm easydict opencv-python-headless \
      scipy ninja rembg onnxruntime trimesh open3d xatlas pyvista pymeshfix igraph \
      transformers safetensors spconv-cu118
  pipx amodal3r xformers==0.0.27.post2 --index-url https://download.pytorch.org/whl/cu118
  pipx amodal3r git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8
  # same kaolin stub as trellis (flexicubes import at load; gaussian path never runs it)
  SP=$(conda run -n amodal3r python -c "import site; print(site.getsitepackages()[0])")
  mkdir -p "$SP/kaolin/utils"
  printf '' > "$SP/kaolin/__init__.py"
  printf '' > "$SP/kaolin/utils/__init__.py"
  printf 'def check_tensor(*a, **k):\n    return True\n' > "$SP/kaolin/utils/testing.py"
  echo ">>> [amodal3r] done"
fi

# ------------------------------------------------- NKSR (surface reconstruction)
# OPTIONAL. Gives smoother, watertight object meshes than screened Poisson. Needs
# torch 2.7/cu128 and compiles a CUDA extension, so it cannot share the iris env.
# If this env is absent the pipeline falls back to Poisson automatically, so a
# failure here is not fatal to the install.
if want nksr; then
  echo ">>> [nksr] env + repo (optional; builds a CUDA extension)"
  (
    set -e
    : "${CUDA_HOME:=/usr/local/cuda-12.8}"
    export CUDA_HOME
    export PATH="$CUDA_HOME/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
    command -v nvcc >/dev/null || { echo "  nvcc not found under CUDA_HOME=$CUDA_HOME"; exit 1; }
    echo "  nvcc: $(nvcc --version | grep release)"
    [ -d models/NKSR ] || git clone https://github.com/nv-tlabs/NKSR models/NKSR
    conda env list | grep -q "^nksr " || conda create -n nksr python=3.10 -y
    pipx nksr -r requirements-nksr.txt
    # build the CUDA extension from the cloned source (no build isolation so it
    # links against the torch already installed in this env)
    pipx nksr --no-build-isolation ./models/NKSR/package/
    conda run -n nksr python -c "import torch, nksr; print('  NKSR ok — torch', torch.__version__, 'cuda', torch.cuda.is_available())"
  ) || echo "  [nksr] setup FAILED — optional; pipeline will use screened Poisson instead."
  echo ">>> [nksr] done"
fi

echo ""
echo "Envs ready. Next: conda run -n iris python scripts/fetch_weights.py"
