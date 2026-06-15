#!/bin/bash
set -e
set -o pipefail
export PS4='+ [$(date "+%H:%M:%S")] '
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

# Ensure CUDA_HOME is set (sources the conda activate.d script if needed)
source "$SCRIPT_DIR/utils/source_cuda_env.sh"

# Shared pip runner + run_pip_clean helper + build parallelism env vars
# (MAX_JOBS, NVCC_THREADS, TORCH_CUDA_ARCH_LIST, FLASH_ATTN_CUDA_ARCHS)
source "$SCRIPT_DIR/utils/pip_helpers.sh"

PYTHON_VERSION=$(python -c "import sys; print(f'python{sys.version_info.major}.{sys.version_info.minor}')")
NVIDIA_PKG_DIR="$CONDA_PREFIX/lib/$PYTHON_VERSION/site-packages/nvidia"

# Ensure nvidia-nvtx is installed (needed for --no-build-isolation builds)
if [ ! -d "$NVIDIA_PKG_DIR/nvtx/include" ]; then
    echo ">>> NVTX include directory not found, installing nvidia-nvtx..."
    run_pip_clean "$PIP_RUNNER install nvidia-nvtx"
fi

# Install PyTorch with CUDA 12.8 if not already present.
# When called from install.sh, PyTorch is already installed (step 2).
# When called standalone, this ensures the correct variant is used.
PYTORCH_INDEX_URL="https://download.pytorch.org/whl/cu128"
TORCH_VERSION=$(python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "")
if [ -z "$TORCH_VERSION" ]; then
    echo "==> PyTorch not found, installing from $PYTORCH_INDEX_URL..."
    run_pip_clean "$PIP_RUNNER install --index-url '$PYTORCH_INDEX_URL' torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 triton==3.6.0"
    TORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
fi
echo "==> Using PyTorch $TORCH_VERSION (CUDA $(python -c 'import torch; print(torch.version.cuda)'))"

# Add ALL nvidia pip package include paths (cudnn, nccl, nvtx, etc.)
# Needed for source builds like transformer_engine, flash-attn, apex
NVIDIA_INCLUDES=""
if [ -d "$NVIDIA_PKG_DIR" ]; then
    for inc_dir in "$NVIDIA_PKG_DIR"/*/include; do
        [ -d "$inc_dir" ] && NVIDIA_INCLUDES="${NVIDIA_INCLUDES:+$NVIDIA_INCLUDES:}$inc_dir"
    done
fi
if [ -n "$NVIDIA_INCLUDES" ]; then
    export CPLUS_INCLUDE_PATH="$NVIDIA_INCLUDES:${CPLUS_INCLUDE_PATH}"
    export C_INCLUDE_PATH="$NVIDIA_INCLUDES:${C_INCLUDE_PATH}"
    echo ">>> Added NVIDIA include paths: $NVIDIA_INCLUDES"
fi

# 2) Install attention backend based on GPU architecture
#    Blackwell (sm_100+) -> FlashInfer (needed for GDN attention in Qwen3.5 etc.)
#    Hopper   (sm_90)    -> flash-attn
#    Ampere   (sm_80)    -> flash-attn  (e.g. A100)
echo ""
# GPU_COMPUTE_CAP may be pre-set (e.g. via docker build --build-arg) to target
# a specific arch when nvidia-smi isn't available (like during `docker build`).
if [ -z "${GPU_COMPUTE_CAP:-}" ]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
        GPU_COMPUTE_CAP=$(nvidia-smi --id=0 --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | tr -d '.' || true)
    else
        echo ">>> nvidia-smi not available (typical during docker build) — skipping GPU arch auto-detect"
        GPU_COMPUTE_CAP=""
    fi
fi
echo ">>> Detected GPU compute capability: ${GPU_COMPUTE_CAP:-unknown}"

if [ -n "$GPU_COMPUTE_CAP" ] && [ "$GPU_COMPUTE_CAP" -ge 90 ] 2>/dev/null; then
    echo ">>> Hopper+ GPU detected (sm_90+) — installing FlashInfer..."
    if [ "$GPU_COMPUTE_CAP" -ge 100 ] 2>/dev/null; then
        run_pip_clean "$PIP_RUNNER install nvidia-nvshmem-cu12==3.4.5"
    fi
    run_pip_clean "$PIP_RUNNER install flashinfer-python==0.6.6 flashinfer-cubin==0.6.6"
    run_pip_clean "$PIP_RUNNER install flashinfer-jit-cache==0.6.6 --index-url https://flashinfer.ai/whl/cu128"
    echo "✓ FlashInfer installed (core + kernels + JIT cache)"
elif [ -n "$GPU_COMPUTE_CAP" ] && [ "$GPU_COMPUTE_CAP" -ge 80 ] 2>/dev/null; then
    echo ">>> Ampere GPU detected (sm_80, e.g. A100) — using flash-attn backend"
else
    echo ">>> WARNING: Unknown or unsupported GPU compute capability: ${GPU_COMPUTE_CAP:-not detected}"
    echo ">>> Proceeding with flash-attn backend — build may fail if GPU arch is too old"
fi

run_pip_clean "$PIP_RUNNER install psutil ninja setuptools_scm cmake packaging wheel"

# flash-attention source for the Hopper/FA3 build below, pinned to a known-good
# commit so the build stays reproducible as upstream main advances.
FLASH_ATTN_DIR="$ROOT_DIR/flash-attention"
FLASH_ATTN_REF="fc8cbad6b6b90220cf6ef8121c29e299a3ba7d9a"  # fa4 v4.0.0-beta17 line; builds against torch 2.10 / CUDA 12.8
if [ ! -d "$FLASH_ATTN_DIR" ]; then
    echo ">>> Cloning flash-attention repository..."
    git clone https://github.com/Dao-AILab/flash-attention.git "$FLASH_ATTN_DIR"
else
    echo ">>> flash-attention directory already exists at $FLASH_ATTN_DIR"
fi
( cd "$FLASH_ATTN_DIR" && git fetch --quiet --all && git checkout --quiet "$FLASH_ATTN_REF" && git submodule update --init --recursive )

# Always install flash-attn 2; if Hopper+ (sm_90+) also install flash-attn 3
FA_BUILD_DIR="$FLASH_ATTN_DIR"
echo ">>> Building flash-attn 2 from $FA_BUILD_DIR..."
run_pip_clean "$PIP_RUNNER install flash-attn==2.8.3.post1 --no-build-isolation --force-reinstall --no-deps"
verify_import flash_attn
echo "✓ flash-attn 2 installed"
if [ -n "$GPU_COMPUTE_CAP" ] && [ "$GPU_COMPUTE_CAP" -ge 90 ] 2>/dev/null; then
    FA_BUILD_DIR="$FLASH_ATTN_DIR/hopper"
    echo ">>> Hopper+ GPU detected (sm_90+) — installing flash-attn 3 (Hopper kernels) from $FA_BUILD_DIR..."
    run_pip_clean "cd '$FA_BUILD_DIR' && $PIP_RUNNER install . --no-build-isolation --force-reinstall --no-deps"
    verify_import flash_attn_3
    echo "✓ flash-attn 3 installed"
fi

# Fix FA3 wheel layout: the flash_attn_3 3.x wheel installs
# flash_attn_interface.py and flash_attn_config.py at the top of site-packages
# instead of inside the flash_attn_3/ package. TE 2.13+ imports them as
# `flash_attn_3.flash_attn_interface` / `flash_attn_3.flash_attn_config`, so
# symlink them into the package dir. Idempotent; only runs on the Hopper path.
if [ -n "$GPU_COMPUTE_CAP" ] && [ "$GPU_COMPUTE_CAP" -ge 90 ] 2>/dev/null; then
    SITE_PACKAGES=$(python -c "from sysconfig import get_path; print(get_path('purelib'))")
    FA3_PKG_DIR="$SITE_PACKAGES/flash_attn_3"
    if [ -d "$FA3_PKG_DIR" ]; then
        for f in flash_attn_interface.py flash_attn_config.py; do
            if [ -f "$SITE_PACKAGES/$f" ] && [ ! -e "$FA3_PKG_DIR/$f" ]; then
                ln -s "../$f" "$FA3_PKG_DIR/$f"
                echo ">>> Symlinked $f into flash_attn_3/ for TE compatibility"
            fi
        done
    fi
fi

# 2b) Install GDN attention dependencies (causal-conv1d + flash-linear-attention)
#     Required for Qwen3.5 linear_attention layers. causal-conv1d needs CUDA to compile.
echo ""
echo ">>> Installing GDN attention dependencies (causal-conv1d, flash-linear-attention)..."
run_pip_clean "$PIP_RUNNER install causal-conv1d==1.6.2.post1 --no-build-isolation --no-cache-dir"
run_pip_clean "$PIP_RUNNER install flash-linear-attention==0.5.0 --no-build-isolation --no-cache-dir"
verify_import causal_conv1d
verify_import fla
echo "✓ causal-conv1d + flash-linear-attention installed"

# 3) Install transformer_engine (rebuild from source against current PyTorch;
#    --no-cache-dir prevents reuse of a stale cached wheel built against a different torch)
#    NOTE: Do NOT use --force-reinstall here — it forces reinstallation of ALL
#    resolved deps (including torch), which pulls torch from PyPI and replaces
#    the cu128 variant with cu130. Instead, uninstall first to force a rebuild.
echo ""
echo ">>> Installing transformer_engine..."
# transformer_engine (pinned). Install the cu12 backend explicitly so it matches
# the CUDA 12.x toolchain — TE's PyTorch frontend (transformer_engine_torch)
# otherwise resolves its CUDA backend to the newest wheel, transformer_engine_cu13.
TE_VERSION="2.16.0"
run_pip_clean "$UNINSTALL transformer-engine transformer-engine-torch transformer-engine-cu12 2>/dev/null || true"
run_pip_clean "$PIP_RUNNER install --no-build-isolation --no-cache-dir \"transformer_engine[pytorch]==${TE_VERSION}\" \"transformer_engine_cu12==${TE_VERSION}\""

# The PyTorch frontend pulls the cu13 backend in alongside cu12. Installed last,
# the cu13 wheel owns the shared transformer_engine/{__init__,common,pytorch}
# modules, so uninstalling it for this cu12-only environment deletes them
# (leaving only transformer_engine/wheel_lib and breaking
# `import transformer_engine.pytorch`). Reinstall the cu12 packages with
# --no-deps to restore the shared modules without re-pulling cu13.
if $PIP_RUNNER show transformer-engine-cu13 >/dev/null 2>&1; then
    echo ">>> Removing cu13 backend and restoring cu12 transformer_engine modules..."
    run_pip_clean "$UNINSTALL transformer-engine-cu13"
    run_pip_clean "$PIP_RUNNER install --no-build-isolation --no-cache-dir --force-reinstall --no-deps \"transformer_engine==${TE_VERSION}\" \"transformer_engine_torch==${TE_VERSION}\" \"transformer_engine_cu12==${TE_VERSION}\""
fi

# Catch the classic silent-broken-install failure: pip leaves a dist-info even
# when the source build errors out, so `pip list` shows TE but the .py files
# are missing and `import transformer_engine.pytorch` fails at training time.
verify_import transformer_engine.pytorch

echo "✓ transformer_engine installed"

# 4) Install apex from source
echo ""
echo ">>> Installing apex from source..."

APEX_DIR="$ROOT_DIR/apex"
APEX_REF="becbb77cea4cb54f2929f7c938a0a6f7dd1fdc39"  # pinned for a reproducible CUDA-extension build

# Clone if not exists
if [ ! -d "$APEX_DIR" ]; then
    echo ">>> Cloning Apex repository..."
    git clone https://github.com/NVIDIA/apex "$APEX_DIR"
else
    echo ">>> Apex directory already exists at $APEX_DIR"
fi
( cd "$APEX_DIR" && git fetch --quiet --all && git checkout --quiet "$APEX_REF" )

# Build and install with CUDA extensions
echo ">>> Building and installing Apex..."
run_pip_clean "cd '$APEX_DIR' && APEX_CPP_EXT=1 APEX_CUDA_EXT=1 $PIP_RUNNER install --no-build-isolation ."
verify_import apex

echo "✓ Apex installed successfully"

# 6) Install megatron-core without dependencies to allow numpy>=2
MEGATRON_SPEC=$(grep -E '^megatron-core' "$ROOT_DIR/install/dependencies/requirements-mcore.txt" | head -n 1)
if [ -n "$MEGATRON_SPEC" ]; then
    echo ""
    echo ">>> Installing megatron-core without dependencies..."
    run_pip_clean "$PIP_RUNNER install --no-deps $MEGATRON_SPEC"
    verify_import megatron.core
    echo "✓ megatron-core installed (no-deps)"
fi
# Note: the rest of requirements-mcore.txt (mbridge, tilelang + its pinned
# apache-tvm-ffi) is already installed by `pip install -e .` in step 3 — see
# setup.py:get_requirements(), which folds requirements-mcore.txt (minus
# megatron-core) into install_requires. Only megatron-core is special-cased
# here (installed --no-deps to allow numpy>=2).

# 7) Install vllm (pinned release + axon patch, or dev mode)
echo ""
echo ">>> Installing vllm..."
bash "$ROOT_DIR/install/vllm/install_vllm.sh"
verify_import vllm
echo "✓ vllm installed"
