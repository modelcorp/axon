# Copyright 2025 Model AI Corp
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0. See the LICENSE file for terms.
"""
Dynamic dependency loading for Axon

This file reads requirements from install/*.txt files and provides them
to setuptools. GPU/CUDA setup is orchestrated by install/install.sh rather
than by pip package metadata.

INSTALLATION ORDER:
1. Pre-install:  bash install/preinstall_script.sh   (CUDA 12.8, conda env vars)
2. PyTorch:      pip install torch==2.10.0 ...       (from cu128 index, before deps)
3. Dependencies: uv pip install -e .                 (Python packages from requirements*.txt)
4. Post-install: bash install/postinstall_script.sh  (flash-attn, vllm, apex, etc.)
5. Agents:       uv pip install -e ".[agents]"       (browsergym, SWE, tool deps)

vllm is installed by install/vllm/install_vllm.sh (called from postinstall), NOT via pip.
Set AXON_VLLM_DEV=1 to use an editable vllm checkout; optionally set VLLM_DEV_PATH.
"""

from pathlib import Path

from setuptools import setup

ROOT_DIR = Path(__file__).parent
INSTALL_DIR = ROOT_DIR / "install"


def _read_requirements(filename: str) -> list[str]:
    """
    Read requirements from a file, handling:
    - Comments (#)
    - Empty lines
    - -r includes (recursive)
    - Skips --index-url and other pip flags
    """
    filepath = INSTALL_DIR / filename
    if not filepath.exists():
        print(f"Warning: {filepath} not found, skipping...")
        return []

    with open(filepath) as f:
        lines = f.read().strip().split("\n")

    requirements = []
    for line in lines:
        line = line.strip()
        # Skip empty lines and comments
        if not line or line.startswith("#"):
            continue
        # Skip pip flags like --index-url
        if line.startswith("--"):
            continue
        # Handle -r includes
        if line.startswith("-r "):
            included_file = line.split()[1]
            requirements.extend(_read_requirements(included_file))
        else:
            requirements.append(line)

    return requirements


def _filter_requirements(requirements: list[str]) -> list[str]:
    """
    Drop megatron-core from dependency resolution to avoid NumPy<2 conflicts.
    megatron-core is installed post-install with --no-deps.
    """
    return [req for req in requirements if not req.startswith("megatron-core")]


def get_requirements() -> list[str]:
    """Get all base requirements (vllm is installed separately by postinstall)."""
    requirements_list = [
        "dependencies/requirements-base.txt",
        "dependencies/requirements-mcore.txt",
    ]
    requirements = []
    for requirement in requirements_list:
        requirements.extend(_read_requirements(requirement))
    return _filter_requirements(requirements)


def get_extras_require() -> dict[str, list[str]]:
    """Get optional dependency groups."""
    extras = {
        "agents": _read_requirements("dependencies/requirements-agents.txt"),
        "sglang": _read_requirements("dependencies/requirements-sglang.txt"),
        "mcore": _read_requirements("dependencies/requirements-mcore.txt"),
        "dev": _read_requirements("dependencies/requirements-dev.txt"),
        "docs": _read_requirements("dependencies/requirements-docs.txt"),
    }
    extras["mcore"] = _filter_requirements(extras["mcore"])
    return extras


# =============================================================================
# Setup
# =============================================================================
setup(
    install_requires=get_requirements(),
    extras_require=get_extras_require(),
)
