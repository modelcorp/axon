#!/bin/bash
# =============================================================================
# source_cuda_env.sh - Ensure CUDA_HOME is set from the conda environment
# =============================================================================
# This is meant to be *sourced* (not executed) so that CUDA env vars are
# exported into the calling process.  It is safe to source multiple times.
#
# Usage:  source install/utils/source_cuda_env.sh
# =============================================================================

if [ -n "${CONDA_PREFIX:-}" ]; then
    if [ -z "${CUDA_HOME:-}" ] || [ ! -x "${CUDA_HOME}/bin/nvcc" ]; then
        _AXON_ACTIVATE="$CONDA_PREFIX/etc/conda/activate.d/axon_cuda_env.sh"
        if [ -f "$_AXON_ACTIVATE" ]; then
            echo ">>> CUDA_HOME=${CUDA_HOME:-<unset>} has no nvcc, sourcing $_AXON_ACTIVATE..."
            source "$_AXON_ACTIVATE"
        elif [ -x "$CONDA_PREFIX/bin/nvcc" ]; then
            export CUDA_HOME="$CONDA_PREFIX"
            export CUDA_PATH="$CONDA_PREFIX"
        fi
        unset _AXON_ACTIVATE
        echo ">>> Using CUDA_HOME=${CUDA_HOME:-<unset>}"
    fi

    # FlashInfer JIT expects $CUDA_HOME/lib64 for -lcudart, but conda puts
    # libraries in $CONDA_PREFIX/lib.  Create symlink if missing.
    if [ -d "$CONDA_PREFIX/lib" ] && [ ! -e "$CONDA_PREFIX/lib64" ]; then
        ln -s "$CONDA_PREFIX/lib" "$CONDA_PREFIX/lib64"
        echo ">>> Created lib64 -> lib symlink for FlashInfer JIT"
    fi

    # Conda's cuda-toolkit puts headers under $CUDA_HOME/targets/<arch>/include,
    # but PyTorch's cuda.cmake calls the old find_package(CUDA) which only looks
    # in $CUDA_HOME/include — which on conda holds cudnn/X11/etc. but NOT
    # cuda_runtime.h. Symlink the toolkit headers in so vllm / TE / any
    # CMake-based build finds them. Idempotent.
    if [ -n "${CUDA_HOME:-}" ] && [ ! -e "$CUDA_HOME/include/cuda_runtime.h" ]; then
        for _arch in x86_64-linux sbsa-linux aarch64-linux; do
            _AXON_CUDA_TARGET="$CUDA_HOME/targets/$_arch"
            if [ -f "$_AXON_CUDA_TARGET/include/cuda_runtime.h" ]; then
                mkdir -p "$CUDA_HOME/include"
                for _entry in "$_AXON_CUDA_TARGET/include"/*; do
                    _name="$(basename "$_entry")"
                    [ -e "$CUDA_HOME/include/$_name" ] || ln -s "$_entry" "$CUDA_HOME/include/$_name"
                done
                echo ">>> Symlinked CUDA toolkit headers from $_AXON_CUDA_TARGET/include into \$CUDA_HOME/include"
                export CUDA_TOOLKIT_ROOT_DIR="$CUDA_HOME"
                export CUDAToolkit_ROOT="$CUDA_HOME"
                break
            fi
        done
        unset _AXON_CUDA_TARGET _arch _entry _name
    fi
fi
