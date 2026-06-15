#!/bin/bash
# =============================================================================
# pip_helpers.sh - Shared pip/build helpers sourced by postinstall/install_vllm
# =============================================================================
# Sourcing this file:
#   - Detects PIP_RUNNER (pip or uv pip)
#   - Auto-discovers CUDA_HOME (conda env, nvcc on PATH, or /usr/local/cuda)
#   - Makes pip-installed CUDA headers (nvtx, cudnn, cublas, ...) findable
#     by gcc/nvcc — fixes the classic "fatal error: nvtx3/nvToolsExt.h"
#     that hits TE, flash-attn, apex etc. when CUDA toolkit ships without
#     a header that lives in a separate nvidia-*-cu12 pip package
#   - Sets build parallelism + auto-detects GPU arch
#   - Provides run_pip_clean (sandboxed pip) and verify_import (sanity check)
#
# Meant to be *sourced*, not executed.
# =============================================================================

# Idempotency — safe to source repeatedly
if [ "${_PIP_HELPERS_SOURCED:-}" = "1" ]; then
    return 0 2>/dev/null || exit 0
fi
_PIP_HELPERS_SOURCED=1

# -----------------------------------------------------------------------------
# Pip runner detection (prefer uv for speed)
# -----------------------------------------------------------------------------
PIP_RUNNER="pip"
UNINSTALL="pip uninstall -y"
if command -v uv >/dev/null 2>&1; then
    PIP_RUNNER="uv pip"
    UNINSTALL="uv pip uninstall"
fi

# -----------------------------------------------------------------------------
# Python site-packages discovery (no hardcoded python3.10)
# -----------------------------------------------------------------------------
PY_SITE_PACKAGES="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])' 2>/dev/null || echo "")"

# -----------------------------------------------------------------------------
# CUDA_HOME auto-detection
# -----------------------------------------------------------------------------
if [ -z "${CUDA_HOME:-}" ]; then
    if [ -n "${CONDA_PREFIX:-}" ] && [ -x "${CONDA_PREFIX}/bin/nvcc" ]; then
        CUDA_HOME="$CONDA_PREFIX"
    elif command -v nvcc >/dev/null 2>&1; then
        CUDA_HOME="$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")"
    elif [ -d /usr/local/cuda ]; then
        CUDA_HOME=/usr/local/cuda
    fi
fi
export CUDA_HOME
export CUDA_PATH="${CUDA_PATH:-$CUDA_HOME}"
export CUDAToolkit_ROOT="${CUDAToolkit_ROOT:-$CUDA_HOME}"
export CUDA_TOOLKIT_ROOT_DIR="${CUDA_TOOLKIT_ROOT_DIR:-$CUDA_HOME}"
[ -n "$CUDA_HOME" ] && echo ">>> CUDA_HOME=$CUDA_HOME"

# -----------------------------------------------------------------------------
# Make pip-installed CUDA headers (nvtx, cudnn, cublas, ...) visible to gcc
# -----------------------------------------------------------------------------
# nvidia-*-cu12 pip wheels drop headers under site-packages/nvidia/<lib>/include
# but never touch CUDA_HOME, so source builds that #include <nvtx3/...> fail
# unless we add those paths to CPATH explicitly.
_added_dirs=0
if [ -n "$PY_SITE_PACKAGES" ] && [ -d "$PY_SITE_PACKAGES/nvidia" ]; then
    for inc in "$PY_SITE_PACKAGES/nvidia"/*/include; do
        if [ -d "$inc" ]; then
            CPATH="${inc}:${CPATH:-}"
            _added_dirs=$((_added_dirs + 1))
        fi
    done
    for libd in "$PY_SITE_PACKAGES/nvidia"/*/lib; do
        [ -d "$libd" ] && LIBRARY_PATH="${libd}:${LIBRARY_PATH:-}"
    done
fi
if [ -n "$CUDA_HOME" ] && [ -d "$CUDA_HOME/include" ]; then
    CPATH="${CUDA_HOME}/include:${CPATH:-}"
fi
CPATH="${CPATH%:}"
LIBRARY_PATH="${LIBRARY_PATH%:}"
export CPATH
export CPLUS_INCLUDE_PATH="${CPLUS_INCLUDE_PATH:-$CPATH}"
export C_INCLUDE_PATH="${C_INCLUDE_PATH:-$CPATH}"
export LIBRARY_PATH
[ "$_added_dirs" -gt 0 ] && echo ">>> Added $_added_dirs pip-installed NVIDIA include dirs to CPATH"

# -----------------------------------------------------------------------------
# Build parallelism (shared across flash-attn, apex, TE, vllm, ...)
# -----------------------------------------------------------------------------
export MAX_JOBS="${MAX_JOBS:-$(( $(nproc) / 2 ))}"
export NVCC_THREADS="${NVCC_THREADS:-1}"

# Build only for the detected GPU arch.
# Override with TORCH_CUDA_ARCH_LIST="8.0;9.0" to build for multiple.
if [ -z "${TORCH_CUDA_ARCH_LIST:-}" ]; then
    GPU_ARCH_DOT=$(nvidia-smi --id=0 --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 || echo "")
    if [ -n "$GPU_ARCH_DOT" ]; then
        export TORCH_CUDA_ARCH_LIST="$GPU_ARCH_DOT"
        # flash-attn uses its own env var (no dot)
        export FLASH_ATTN_CUDA_ARCHS="$(echo "$GPU_ARCH_DOT" | tr -d '.')"
        echo ">>> Auto-detected GPU arch: $GPU_ARCH_DOT — building for this arch only"
    fi
fi

# TE specific: force the pytorch frontend instead of relying on autodetect.
# Autodetection silently produces empty wheels when torch import fails for any
# reason during the build subprocess. Explicit is safer; harmless for non-TE.
export NVTE_FRAMEWORK="${NVTE_FRAMEWORK:-pytorch}"

# -----------------------------------------------------------------------------
# run_pip_clean: sandboxed pip invocation
# -----------------------------------------------------------------------------
# env -i wipes the caller's shell state, then we re-export only what builds need.
run_pip_clean() {
    local cmd="$1"
    echo ">>> Running: $cmd"
    env -i \
        PATH="$PATH" \
        HOME="$HOME" \
        CUDA_HOME="${CUDA_HOME:-}" \
        CUDA_PATH="${CUDA_PATH:-}" \
        CUDAToolkit_ROOT="${CUDAToolkit_ROOT:-}" \
        CUDA_TOOLKIT_ROOT_DIR="${CUDA_TOOLKIT_ROOT_DIR:-}" \
        CPATH="${CPATH:-}" \
        CPLUS_INCLUDE_PATH="${CPLUS_INCLUDE_PATH:-}" \
        C_INCLUDE_PATH="${C_INCLUDE_PATH:-}" \
        LIBRARY_PATH="${LIBRARY_PATH:-}" \
        CONDA_PREFIX="${CONDA_PREFIX:-}" \
        CONDA_DEFAULT_ENV="${CONDA_DEFAULT_ENV:-}" \
        ${CMAKE_ARGS:+CMAKE_ARGS="$CMAKE_ARGS"} \
        ${MAX_JOBS:+MAX_JOBS="$MAX_JOBS"} \
        ${NVCC_THREADS:+NVCC_THREADS="$NVCC_THREADS"} \
        ${TORCH_CUDA_ARCH_LIST:+TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST"} \
        ${FLASH_ATTN_CUDA_ARCHS:+FLASH_ATTN_CUDA_ARCHS="$FLASH_ATTN_CUDA_ARCHS"} \
        ${NVTE_FRAMEWORK:+NVTE_FRAMEWORK="$NVTE_FRAMEWORK"} \
        bash -c "$cmd"
}

# -----------------------------------------------------------------------------
# verify_import: sanity-check a module imports after install
# -----------------------------------------------------------------------------
# Usage: verify_import transformer_engine.pytorch
# Returns non-zero if the import fails — turns silently-broken installs
# (empty wheels, ABI mismatches) into loud immediate failures instead of
# letting them propagate into your training run.
verify_import() {
    local module="$1"
    echo ">>> Verifying: import $module"
    if python -c "import $module" 2>&1; then
        echo ">>> OK: $module imports"
    else
        echo ">>> FAIL: $module did not import after install"
        return 1
    fi
}