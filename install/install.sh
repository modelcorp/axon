#!/bin/bash
# =============================================================================
# install.sh - Full Axon installation (CUDA + Python deps + PyTorch + vllm)
# =============================================================================
# This script runs all install steps in a single process so that environment
# variables (CUDA_HOME, etc.) set by preinstall are available for later steps.
#
# Usage:
#   conda activate <your-env>
#   bash install/install.sh
# =============================================================================
set -e
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

echo "============================================="
echo " Axon Installation"
echo "============================================="
echo ""

# ---- Step 1: CUDA toolkit + system deps ----
echo "==> [1/5] Installing CUDA toolkit and system dependencies..."
bash "$SCRIPT_DIR/preinstall_script.sh"

# Source CUDA env vars into this process (preinstall ran in a subprocess so
# its exports didn't propagate here)
source "$SCRIPT_DIR/utils/source_cuda_env.sh"

# ---- Step 2: PyTorch (must be installed before Python deps so transitive deps resolve correctly) ----
echo ""
echo "==> [2/5] Installing PyTorch with CUDA 12.8 support..."

PIP_RUNNER="pip install"
if command -v uv >/dev/null 2>&1; then
    PIP_RUNNER="uv pip install"
fi

PYTORCH_INDEX_URL="https://download.pytorch.org/whl/cu128"
$PIP_RUNNER --index-url "$PYTORCH_INDEX_URL" torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 triton==3.6.0

# ---- Step 3: axon + Python dependencies ----
echo ""
echo "==> [3/5] Installing axon and Python dependencies..."
cd "$ROOT_DIR"
$PIP_RUNNER -e .

# ---- Step 4: vllm, flash-attn, apex, megatron-core, and other compiled extensions ----
echo ""
echo "==> [4/5] Installing vllm and compiled extensions..."
bash "$SCRIPT_DIR/postinstall_script.sh"

# ---- Step 5: Agent-specific packages (browsergym, swebench, etc.) ----
# Installed separately because browsergym's transitive deps conflict with
# transformers>=5.3.0 during resolution. Installing after vllm avoids this.
echo ""
echo "==> [5/5] Installing agent packages..."
$PIP_RUNNER -e ".[agents]" --no-deps 2>/dev/null || \
    $PIP_RUNNER -r "$SCRIPT_DIR/dependencies/requirements-agents.txt" || \
    echo "WARNING: Agent packages failed to install. Install manually with: pip install -e .[agents]"

echo ""
echo "============================================="
echo " Axon installation complete!"
echo "============================================="
