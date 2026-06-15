#!/bin/bash
# Install vllm for axon — two modes controlled by AXON_VLLM_DEV.
#
# AXON_VLLM_DEV=0 (default):
#   Installs the pinned vllm release from PyPI and applies axon.patch.
#
# AXON_VLLM_DEV=1:
#   Dev mode. If vllm is already installed, does nothing. Otherwise uses
#   VLLM_DEV_PATH, or clones VLLM_FORK_URL into axon/vllm, checks out the
#   pinned branch, and installs it editable.
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
PATCH_FILE="$SCRIPT_DIR/axon.patch"

# Source CUDA env vars (CUDA_HOME, CPATH, LD_LIBRARY_PATH) — needed when
# running this script standalone (postinstall_script.sh sources it too).
source "$ROOT_DIR/install/utils/source_cuda_env.sh"
# Shared pip runner + run_pip_clean helper (PIP_RUNNER, run_pip_clean)
source "$ROOT_DIR/install/utils/pip_helpers.sh"

# ---------------------------------------------------------------------------
# Pinned vllm version + fork commit — update when rebasing.
# ---------------------------------------------------------------------------
VLLM_VERSION="0.19.0"
VLLM_WHEEL_URL="vllm==${VLLM_VERSION}"
VLLM_COMMIT="2a69949bdadf0e8942b7a1619b229cb475beef20"
VLLM_COMMIT_SHORT="2a69949bd"
VLLM_FORK_URL="${VLLM_FORK_URL:-https://github.com/modelcorp/vllm.git}"
VLLM_FORK_BRANCH="${VLLM_FORK_BRANCH:-modelcorp/v${VLLM_VERSION}}"
VLLM_DEV_PATH="${VLLM_DEV_PATH:-$ROOT_DIR/vllm}"
AXON_VLLM_DEV="${AXON_VLLM_DEV:-0}"

# ---------------------------------------------------------------------------
# Dev mode (AXON_VLLM_DEV=1): editable install from fork
# ---------------------------------------------------------------------------
if [ "$AXON_VLLM_DEV" = "1" ]; then
    # Check for a *real* vllm install (not a broken namespace package from a
    # failed uninstall). A real install has a resolvable __file__.
    VLLM_CHECK=$(python -c "
import vllm
f = getattr(vllm, '__file__', None)
if not f:
    raise SystemExit(1)
print(f)
print(getattr(vllm, '__version__', 'unknown'))
" 2>/dev/null || true)
    if [ -n "$VLLM_CHECK" ]; then
        VLLM_FILE=$(echo "$VLLM_CHECK" | head -1)
        INSTALLED_VERSION=$(echo "$VLLM_CHECK" | tail -1)
        echo "==> Dev mode: vllm already installed ($INSTALLED_VERSION at $VLLM_FILE) — skipping."
        exit 0
    fi

    # Clone the fork if missing
    if [ ! -d "$VLLM_DEV_PATH" ]; then
        echo "==> Cloning vllm fork to $VLLM_DEV_PATH..."
        git clone "$VLLM_FORK_URL" "$VLLM_DEV_PATH"
    fi

    # Checkout the pinned branch/commit
    echo "==> Checking out $VLLM_FORK_BRANCH in $VLLM_DEV_PATH..."
    (cd "$VLLM_DEV_PATH" && git fetch origin && git checkout "$VLLM_FORK_BRANCH")

    # MAX_JOBS, NVCC_THREADS, TORCH_CUDA_ARCH_LIST come from pip_helpers.sh.
    # --no-deps because vllm pins compressed-tensors==0.14.0.1 which caps
    # transformers<5.0, conflicting with axon's transformers>=5.3.0.
    echo "==> Installing vllm editable from $VLLM_DEV_PATH..."
    echo ">>> MAX_JOBS=$MAX_JOBS NVCC_THREADS=$NVCC_THREADS TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-auto}"
    run_pip_clean "$PIP_RUNNER install --editable '$VLLM_DEV_PATH' --no-deps --no-build-isolation"

    # Install vllm's CUDA runtime deps from cuda.txt (NOT just common.txt). cuda.txt
    # does `-r common.txt` and adds the GPU/Blackwell extras the user-mode wheel bundles:
    # flashinfer-cubin, nvidia-cutlass-dsl and quack-kernels (the FA4 cute-DSL path on
    # sm_100/B200) plus numba. Installing only common.txt here left dev-mode without FA4,
    # so a dev install on Blackwell diverged from (and underperformed) the user-mode wheel.
    echo "==> Installing vllm CUDA runtime dependencies (cuda.txt)..."
    run_pip_clean "$PIP_RUNNER install -r '$VLLM_DEV_PATH/requirements/cuda.txt'"

    # cuda.txt -> common.txt pins compressed-tensors==0.14.0.1, which caps transformers<5.
    # Override it (then restore the exact transformers pin) the same way the user-mode path
    # does — sequentially, so the resolver doesn't see the two constraints at once.
    echo "==> Overriding compressed-tensors / transformers to axon-compatible versions..."
    run_pip_clean "$PIP_RUNNER install 'compressed-tensors>=0.15.0'"
    # Must match install/dependencies/requirements-base.txt (transformers==5.5.3) and the
    # user-mode path; '>=' would float past the pin on a fresh install.
    run_pip_clean "$PIP_RUNNER install 'transformers==5.5.3'"

    echo "==> vllm dev install complete."
    exit 0
fi

# ---------------------------------------------------------------------------
# Default mode (AXON_VLLM_DEV=0): PyPI install + apply committed axon.patch
# ---------------------------------------------------------------------------

# axon.patch should already exist in the repo (committed to git). If it's
# missing (e.g. someone deleted it), regenerate from the fork as a fallback.
if [ ! -s "$PATCH_FILE" ]; then
    echo "==> axon.patch missing — regenerating from fork..."
    if [ ! -d "$VLLM_DEV_PATH" ]; then
        echo "==> Cloning vllm fork to $VLLM_DEV_PATH..."
        git clone "$VLLM_FORK_URL" "$VLLM_DEV_PATH"
    fi
    echo "==> Checking out $VLLM_FORK_BRANCH..."
    (cd "$VLLM_DEV_PATH" && git fetch origin && git checkout "$VLLM_FORK_BRANCH")
    bash "$SCRIPT_DIR/generate_patch.sh" "$VLLM_DEV_PATH"
fi

echo "==> Installing vllm ${VLLM_VERSION} from PyPI..."
run_pip_clean "$PIP_RUNNER install '$VLLM_WHEEL_URL'"

# vllm 0.19.0 pulls compressed-tensors==0.14.0.1 as a transitive dep, and
# that version caps transformers<5. Bumping to >=0.15.0 lifts the cap so the
# subsequent transformers upgrade doesn't leave a broken resolver state.
# This mirrors the same fix in the dev-mode branch above.
echo "==> Bumping compressed-tensors to a transformers>=5-compatible version..."
run_pip_clean "$PIP_RUNNER install 'compressed-tensors>=0.15.0'"

# The PyPI vllm wheel caps transformers<5, but axon requires the exact pin.
# Must match install/dependencies/requirements-base.txt (transformers==5.5.3) and the
# dev-mode branch above; a floating '>=5.3.0' here silently drifts user-mode installs to
# the latest 5.x (e.g. 5.12.0) while dev-mode stays at 5.5.3, so the two modes diverge.
echo "==> Restoring transformers to axon-compatible version..."
run_pip_clean "$PIP_RUNNER install 'transformers==5.5.3'"

# Locate the installed vllm package
VLLM_INSTALL_DIR=$(python -c "import vllm, pathlib; print(pathlib.Path(vllm.__file__).parent)")
if [ -z "$VLLM_INSTALL_DIR" ] || [ ! -d "$VLLM_INSTALL_DIR" ]; then
    echo "ERROR: Could not locate installed vllm package."
    exit 1
fi
VLLM_SITE_PKG_DIR=$(dirname "$VLLM_INSTALL_DIR")
INSTALLED_VERSION=$(python -c "import vllm; print(vllm.__version__)")
echo "==> vllm ${INSTALLED_VERSION} installed at: $VLLM_INSTALL_DIR"

# Apply patch if it exists and is non-empty
# NOTE: axon.patch contains Python-only hunks. C++ changes (csrc/) are in
# axon-csrc.patch and only apply to source builds (wheels ship precompiled .so).
if [ -s "$PATCH_FILE" ]; then
    echo "==> Applying axon patch to vllm..."

    # Apply with --forward --batch:
    #   --forward: don't reverse already-applied hunks
    #   --batch:   non-interactive (skip prompts for missing files)
    PATCH_OUTPUT=$(patch --forward --batch -d "$VLLM_SITE_PKG_DIR" -p1 < "$PATCH_FILE" 2>&1) || true

    if echo "$PATCH_OUTPUT" | grep -q "FAILED"; then
        echo "$PATCH_OUTPUT"
        echo ""
        echo "ERROR: Some patch hunks FAILED to apply."
        echo "       Expected commit: ${VLLM_COMMIT_SHORT}"
        echo "       Installed version: $INSTALLED_VERSION"
        exit 1
    elif echo "$PATCH_OUTPUT" | grep -q "Reversed"; then
        echo "==> Patch already applied, skipping."
    else
        echo "$PATCH_OUTPUT"
        echo "==> Patch applied successfully."
    fi
else
    echo "==> No patch file at $PATCH_FILE — skipping patch step."
fi

echo "==> vllm installation complete."
