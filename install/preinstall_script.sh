#!/bin/bash
# =============================================================================
# install_cuda.sh - Install CUDA 12.8 dependencies for Axon
# =============================================================================
set -e

# Locate a package manager. `conda` is commonly a shell function defined by
# `conda init`, which is NOT inherited by sub-scripts. So PATH lookup alone is
# insufficient — also check $CONDA_EXE (set by conda init) and common install
# locations before giving up.
PKG_MGR=""
if command -v micromamba >/dev/null 2>&1; then
    PKG_MGR="$(command -v micromamba)"
elif command -v mamba >/dev/null 2>&1; then
    PKG_MGR="$(command -v mamba)"
elif command -v conda >/dev/null 2>&1; then
    PKG_MGR="$(command -v conda)"
elif [ -n "${CONDA_EXE:-}" ] && [ -x "$CONDA_EXE" ]; then
    PKG_MGR="$CONDA_EXE"
elif [ -n "${MAMBA_EXE:-}" ] && [ -x "$MAMBA_EXE" ]; then
    PKG_MGR="$MAMBA_EXE"
else
    for _candidate in \
        "$HOME/miniconda3/condabin/conda" \
        "$HOME/miniconda3/bin/conda" \
        "$HOME/anaconda3/condabin/conda" \
        "$HOME/anaconda3/bin/conda" \
        "$HOME/miniforge3/condabin/conda" \
        "$HOME/mambaforge/condabin/conda" \
        "/opt/conda/condabin/conda" \
        "/opt/conda/bin/conda"
    do
        if [ -x "$_candidate" ]; then
            PKG_MGR="$_candidate"
            break
        fi
    done
    unset _candidate
fi

if [ -z "$PKG_MGR" ]; then
    echo "ERROR: conda or micromamba is required but was not found." >&2
    echo "" >&2
    echo "Checked:" >&2
    echo "  - PATH lookup for: micromamba, mamba, conda" >&2
    echo "  - Env vars: \$CONDA_EXE, \$MAMBA_EXE" >&2
    echo "  - Common install paths under \$HOME and /opt/conda" >&2
    echo "" >&2
    echo "Current PATH: $PATH" >&2
    echo "" >&2
    echo "Hint: 'conda' is typically a shell function defined by 'conda init'," >&2
    echo "      and scripts run in subshells do NOT inherit shell functions." >&2
    echo "      Fix by running one of:" >&2
    echo "        source \"\$HOME/miniconda3/etc/profile.d/conda.sh\" && conda activate <env>" >&2
    echo "        export CONDA_EXE=\"\$(type -p conda 2>/dev/null)\"" >&2
    exit 1
fi

# Normalize a display name for later branches (conda vs micromamba vs mamba)
PKG_MGR_NAME="$(basename "$PKG_MGR")"

# Auto-detect environment from current activation
ENV_NAME=""
ENV_PREFIX="${CONDA_PREFIX:-}"
if [ -z "$ENV_PREFIX" ] && [ -n "${CONDA_DEFAULT_ENV:-}" ]; then
    if [[ "$CONDA_DEFAULT_ENV" == *"/"* ]]; then
        ENV_PREFIX="$CONDA_DEFAULT_ENV"
    else
        ENV_NAME="$CONDA_DEFAULT_ENV"
    fi
fi

if [ -z "$ENV_PREFIX" ] && [ -z "$ENV_NAME" ]; then
    echo "ERROR: No conda environment is activated."
    echo "Please activate your conda environment first."
    exit 1
fi

if [ -z "$ENV_PREFIX" ]; then
    ENV_ARGS=(-n "$ENV_NAME")
    ENV_DESC="$ENV_NAME"
else
    ENV_ARGS=(-p "$ENV_PREFIX")
    ENV_DESC="$ENV_PREFIX"
fi

if [ -z "${CONDA_PREFIX:-}" ]; then
    echo "ERROR: CONDA_PREFIX is not set. Please activate your environment."
    exit 1
fi

echo "==> Detected environment: $ENV_DESC (using $PKG_MGR_NAME at $PKG_MGR)"

if [ "$PKG_MGR_NAME" = "conda" ]; then
    echo "==> Updating conda..."
    "$PKG_MGR" update -n base -c defaults conda -y
fi

# Cache package list output for efficiency
echo "==> Checking installed packages..."
PKG_LIST_OUTPUT=$("$PKG_MGR" list "${ENV_ARGS[@]}")

# Check if CUDA toolkit is already installed
echo "==> Checking for CUDA toolkit installation..."
if echo "$PKG_LIST_OUTPUT" | grep -q "cuda-toolkit.*12\.8"; then
    echo "    ✓ CUDA toolkit 12.8 is already installed"
else
    echo "==> Installing CUDA toolkit 12.8..."
    "$PKG_MGR" install "${ENV_ARGS[@]}" -c nvidia/label/cuda-12.8.0 -c conda-forge cuda-toolkit=12.8.0 cuda-nvcc=12.8 -y
    # Refresh package list after installation
    PKG_LIST_OUTPUT=$("$PKG_MGR" list "${ENV_ARGS[@]}")
fi

# Check if cuDNN is already installed
echo "==> Checking for cuDNN installation..."
if echo "$PKG_LIST_OUTPUT" | grep -q "cudnn"; then
    echo "    ✓ cuDNN is already installed"
else
    echo "==> Installing cuDNN..."
    "$PKG_MGR" install "${ENV_ARGS[@]}" -c nvidia -c conda-forge "cudnn>=9,<10" "cuda-version>=12,<13" -y
fi

# Check if NCCL is already installed
echo "==> Checking for NCCL installation..."
if echo "$PKG_LIST_OUTPUT" | grep -q "nccl"; then
    echo "    ✓ NCCL is already installed"
else
    echo "==> Installing NCCL..."
    "$PKG_MGR" install "${ENV_ARGS[@]}" -c conda-forge nccl -y
fi

# Check if cuda-nvtx is already installed (needed by transformer-engine)
echo "==> Checking for cuda-nvtx installation..."
if echo "$PKG_LIST_OUTPUT" | grep -q "cuda-nvtx"; then
    echo "    ✓ cuda-nvtx is already installed"
else
    echo "==> Installing cuda-nvtx..."
    "$PKG_MGR" install "${ENV_ARGS[@]}" -c nvidia/label/cuda-12.8.0 cuda-nvtx -y
fi

pip install nvidia-nvtx-cu12

# This creates an invalid libcudart.so.13 file that causes dlopen to fail,
# preventing system CUDA 13 from being detected alongside conda's CUDA 12.
BLOCKER_DIR="$CONDA_PREFIX/lib/cuda_blocker"
if [ ! -f "$BLOCKER_DIR/libcudart.so.13" ]; then
    echo "==> Creating CUDA 13 blocker file..."
    mkdir -p "$BLOCKER_DIR"
    echo "invalid" > "$BLOCKER_DIR/libcudart.so.13"
    echo "    ✓ Created $BLOCKER_DIR/libcudart.so.13"
else
    echo "==> CUDA 13 blocker file already exists, skipping..."
fi

# =============================================================================
# Persist environment variables to conda activate.d
# =============================================================================
ACTIVATE_DIR="$CONDA_PREFIX/etc/conda/activate.d"
DEACTIVATE_DIR="$CONDA_PREFIX/etc/conda/deactivate.d"
ACTIVATE_SCRIPT="$ACTIVATE_DIR/axon_cuda_env.sh"
DEACTIVATE_SCRIPT="$DEACTIVATE_DIR/axon_cuda_env.sh"

# Create directories if they don't exist
mkdir -p "$ACTIVATE_DIR"
mkdir -p "$DEACTIVATE_DIR"

# Check if already added
if [ -f "$ACTIVATE_SCRIPT" ]; then
    echo "==> CUDA environment variables already configured in conda activate.d, skipping..."
else
    echo "==> Adding CUDA environment variables to conda activate.d..."

    CUDA_HOME_VAL="$CONDA_PREFIX"
    CUDA_PATH_VAL="$CONDA_PREFIX"
    CPATH_PREFIX="$CONDA_PREFIX/targets/x86_64-linux/include:$CONDA_PREFIX/include"
    LD_LIBRARY_PREFIX="$CONDA_PREFIX/lib"

    escape_sed() {
        printf '%s' "$1" | sed -e 's/[\\/&]/\\&/g'
    }

    # Create activate script
    cat > "$ACTIVATE_SCRIPT" << 'EOF'
#!/bin/bash
# Axon CUDA 12.8 environment - auto-generated by install_cuda.sh

# Backup single-value variables (only if not already backed up)
[ -z "${_AXON_OLD_NCCL_ROOT+x}" ] && export _AXON_OLD_NCCL_ROOT="${NCCL_ROOT-}"
[ -z "${_AXON_OLD_CUDA_HOME+x}" ] && export _AXON_OLD_CUDA_HOME="${CUDA_HOME-}"
[ -z "${_AXON_OLD_CUDA_PATH+x}" ] && export _AXON_OLD_CUDA_PATH="${CUDA_PATH-}"

export NCCL_ROOT="$CONDA_PREFIX"
export CUDA_HOME="__CUDA_HOME_VAL__"
export CUDA_PATH="__CUDA_PATH_VAL__"
export CPATH="__CPATH_PREFIX__"${CPATH:+:$CPATH}

# Fix for "Multiple libcudart libraries found" - blocker directory must be first
if [ -d "$CONDA_PREFIX/lib/cuda_blocker" ]; then
    [ -z "${_AXON_OLD_LD_LIBRARY_PATH+x}" ] && export _AXON_OLD_LD_LIBRARY_PATH="${LD_LIBRARY_PATH-}"
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/cuda_blocker:__LD_LIBRARY_PREFIX__"${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
else
    [ -z "${_AXON_OLD_LD_LIBRARY_PATH+x}" ] && export _AXON_OLD_LD_LIBRARY_PATH="${LD_LIBRARY_PATH-}"
    export LD_LIBRARY_PATH="__LD_LIBRARY_PREFIX__"${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
fi
EOF

    # Create deactivate script
    cat > "$DEACTIVATE_SCRIPT" << 'EOF'
#!/bin/bash
# Axon CUDA 12.8 environment - auto-generated by install_cuda.sh

# Helper function to remove all occurrences of a value from a colon-separated variable
_remove_from_path() {
    local var_name="$1"
    local value_to_remove="$2"
    local current_value="${!var_name}"
    local new_value=""
    IFS=':' read -ra _parts <<< "$current_value"
    for _part in "${_parts[@]}"; do
        [[ "$_part" != "$value_to_remove" ]] && new_value="${new_value:+$new_value:}$_part"
    done
    export "$var_name"="$new_value"
}

# Remove all occurrences of CONDA_PREFIX paths from path-like variables
_remove_from_path CPATH "$CONDA_PREFIX/targets/x86_64-linux/include"
_remove_from_path CPATH "$CONDA_PREFIX/include"

# Restore single-value variables from backup
if [ -n "${_AXON_OLD_NCCL_ROOT+x}" ]; then
    [ -n "$_AXON_OLD_NCCL_ROOT" ] && export NCCL_ROOT="$_AXON_OLD_NCCL_ROOT" || unset NCCL_ROOT
    unset _AXON_OLD_NCCL_ROOT
fi
if [ -n "${_AXON_OLD_CUDA_HOME+x}" ]; then
    [ -n "$_AXON_OLD_CUDA_HOME" ] && export CUDA_HOME="$_AXON_OLD_CUDA_HOME" || unset CUDA_HOME
    unset _AXON_OLD_CUDA_HOME
fi
if [ -n "${_AXON_OLD_CUDA_PATH+x}" ]; then
    [ -n "$_AXON_OLD_CUDA_PATH" ] && export CUDA_PATH="$_AXON_OLD_CUDA_PATH" || unset CUDA_PATH
    unset _AXON_OLD_CUDA_PATH
fi
if [ -n "${_AXON_OLD_LD_LIBRARY_PATH+x}" ]; then
    [ -n "$_AXON_OLD_LD_LIBRARY_PATH" ] && export LD_LIBRARY_PATH="$_AXON_OLD_LD_LIBRARY_PATH" || unset LD_LIBRARY_PATH
    unset _AXON_OLD_LD_LIBRARY_PATH
fi

# Cleanup
unset -f _remove_from_path
EOF

    sed -i \
        -e "s|__CUDA_HOME_VAL__|$(escape_sed "$CUDA_HOME_VAL")|g" \
        -e "s|__CUDA_PATH_VAL__|$(escape_sed "$CUDA_PATH_VAL")|g" \
        -e "s|__CPATH_PREFIX__|$(escape_sed "$CPATH_PREFIX")|g" \
        -e "s|__LD_LIBRARY_PREFIX__|$(escape_sed "$LD_LIBRARY_PREFIX")|g" \
        "$ACTIVATE_SCRIPT" "$DEACTIVATE_SCRIPT"

    chmod +x "$ACTIVATE_SCRIPT"
    chmod +x "$DEACTIVATE_SCRIPT"
    echo "==> Environment variables will be set automatically when activating '$ENV_DESC'"
fi

# =============================================================================
# Verify installation
# =============================================================================
echo "==> Verifying NCCL installation..."
if [ -f "$CONDA_PREFIX/include/nccl.h" ]; then
    echo "    ✓ nccl.h found at $CONDA_PREFIX/include/nccl.h"
else
    echo "    ✗ WARNING: nccl.h not found at $CONDA_PREFIX/include/nccl.h"
fi

# Install uv.
pip install uv

echo "==> Installation complete!"
