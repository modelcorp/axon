# =============================================================================
# Axon Docker Image
# =============================================================================
# This Dockerfile creates a complete Axon environment with CUDA 12.8 support.
# 
# Build command:
#   docker build -f ~/axon/install/docker/Dockerfile -t axon-v0 ~/axon
# 
# Run command:
#   docker run --gpus all --shm-size=16g -it axon-v0:latest
# 
# The image includes:
#   - CUDA 12.8 development environment
#   - Micromamba with Python 3.10
#   - All Axon dependencies (PyTorch, flash-attn, transformer_engine, apex)
#   - Proper CUDA environment configuration
# =============================================================================

# Base image with CUDA 12.8 development tools and Ubuntu 24.04
FROM nvidia/cuda:12.8.0-devel-ubuntu24.04

ARG CACHE_SRC=.
ARG CONDA_ENV=axon
ARG PYTHON_VERSION=3.10
ARG APT_UPGRADE=0
ARG HF_MODEL_ID=TinyLlama/TinyLlama-1.1B-Chat-v1.0
# Target GPU compute capability for attention-backend selection in postinstall_script.sh.
# Docker build has no GPU access, so nvidia-smi can't auto-detect. Set explicitly:
#   80 = Ampere (A100), 90 = Hopper (H100), 100 = Blackwell (B100/B200).
# Empty = skip arch-specific install (flash-attn still installed unconditionally).
ARG GPU_COMPUTE_CAP=

ENV MAMBA_ROOT_PREFIX=/opt/micromamba
ENV CONDA_DIR=/opt/conda
ENV CONDA_ENVS_PATH=/opt/micromamba/envs
ENV PATH="/opt/conda/bin:/opt/conda/condabin:/usr/local/bin:$PATH"
ENV HF_HOME=/root/.cache/huggingface
ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV UV_NO_CACHE=1

# Install essential system packages
# - htop: system monitoring
# - tmux: terminal multiplexer for session management
# - wget: download files
# - git: version control (needed for cloning dependencies)
RUN apt-get update && \
    if [ "$APT_UPGRADE" = "1" ]; then apt-get upgrade -y; fi && \
    apt-get install -y --no-install-recommends \
    htop \
    tmux \
    wget \
    git \
    ca-certificates \
    bzip2 \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# Copy the Axon source code into the container
# This will be used for installation and contains all necessary scripts
COPY ${CACHE_SRC} /root/.cache/axon

# =============================================================================
# Install Micromamba
# =============================================================================
# Download and install micromamba for Python environment management
RUN wget -qO- https://micro.mamba.pm/api/micromamba/linux-64/latest \
    | tar -xvj -C /usr/local/bin --strip-components=1 bin/micromamba

# =============================================================================
# Install Miniconda (optional interactive use)
# =============================================================================
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p "$CONDA_DIR" && \
    rm /tmp/miniconda.sh && \
    "$CONDA_DIR/bin/python" -m pip install --upgrade pip uv && \
    "$CONDA_DIR/bin/conda" clean -a -y

# Configure conda to find micromamba envs and enable conda activate in shells.
RUN mkdir -p /etc/conda && \
    conda config --system --prepend envs_dirs /opt/micromamba/envs && \
    echo ". $CONDA_DIR/etc/profile.d/conda.sh" >> /etc/bash.bashrc && \
    printf '%s\n' 'eval "$(micromamba shell hook --shell bash)"' >> /etc/bash.bashrc


# Set bash as the default shell for subsequent RUN commands
# This ensures shell features work properly in multi-line RUN commands
SHELL ["/bin/bash", "-c"]


# Create dedicated micromamba environment for Axon with Python 3.10
# Python 3.10 is required for compatibility with Axon dependencies
RUN micromamba create -y -n "$CONDA_ENV" -c defaults python="$PYTHON_VERSION" pip && \
    micromamba clean -a -y

# =============================================================================
# Install CUDA Dependencies
# =============================================================================
RUN micromamba run -n "$CONDA_ENV" bash /root/.cache/axon/install/preinstall_script.sh && \
    micromamba clean -a -y

# =============================================================================
# Install Axon and Dependencies
# =============================================================================
RUN micromamba run -n "$CONDA_ENV" python -m pip install --upgrade uv && \
    micromamba run -n "$CONDA_ENV" uv pip install \
      -r /root/.cache/axon/install/dependencies/requirements-base.txt && \
    micromamba run -n "$CONDA_ENV" bash -lc \
      "grep -v '^megatron-core' /root/.cache/axon/install/dependencies/requirements-mcore.txt > /tmp/requirements-mcore-no-megatron.txt" && \
    micromamba run -n "$CONDA_ENV" uv pip install -r /tmp/requirements-mcore-no-megatron.txt && \
    GPU_COMPUTE_CAP="$GPU_COMPUTE_CAP" micromamba run -n "$CONDA_ENV" bash /root/.cache/axon/install/postinstall_script.sh && \
    micromamba run -n "$CONDA_ENV" bash -lc \
      "PIP_USE_PEP517=0 python -m pip install --no-deps -e /root/.cache/axon" && \
    micromamba run -n "$CONDA_ENV" uv pip install --upgrade --force-reinstall pandas

# Generate FrozenLake dataset for offline smoke tests.
RUN micromamba run -n "$CONDA_ENV" python /root/.cache/axon/recipes/frozenlake/data.py

# Preload TinyLlama so smoke tests can run in offline mode.
RUN HF_MODEL_ID="$HF_MODEL_ID" micromamba run -n "$CONDA_ENV" python - <<'PY'
import os
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id=os.environ["HF_MODEL_ID"],
    local_dir=None,
    local_dir_use_symlinks=False,
)
PY

# Move examples directory to home for easy access
RUN if [ -d /root/.cache/axon/recipes ]; then \
      cp -r /root/.cache/axon/recipes /root/; \
    fi

# Activate the axon environment for container commands by default.
RUN printf '%s\n' \
      '#!/usr/bin/env bash' \
      'set -e' \
      'if [ -f /opt/conda/etc/profile.d/conda.sh ]; then' \
      '  . /opt/conda/etc/profile.d/conda.sh' \
      '  conda activate "${CONDA_ENV:-axon}" >/dev/null 2>&1 || true' \
      'fi' \
      'exec "$@"' \
    > /usr/local/bin/axon-entrypoint && chmod +x /usr/local/bin/axon-entrypoint

# Default working directory for interactive use.
WORKDIR /root

# # Default command when container starts
# # Launches an interactive bash shell with the axon environment ready to use
ENTRYPOINT ["/usr/local/bin/axon-entrypoint"]
CMD ["/bin/bash"]
