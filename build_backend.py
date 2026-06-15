# Custom build backend that wraps setuptools.
#
# The install hooks below are intentionally disabled. CUDA/toolchain setup is
# handled by install/install.sh, because pip metadata hooks are too early and
# too opaque for a GPU build that needs an activated conda environment.

import os
import shutil
import subprocess
import sys
from pathlib import Path

# Import all standard setuptools.build_meta functions
from setuptools.build_meta import *
from setuptools.build_meta import (
    build_editable as _build_editable,
)
from setuptools.build_meta import (
    build_wheel as _build_wheel,
)
from setuptools.build_meta import (
    get_requires_for_build_editable as _get_requires_for_build_editable,
)
from setuptools.build_meta import (
    get_requires_for_build_wheel as _get_requires_for_build_wheel,
)

ROOT_DIR = Path(__file__).parent
INSTALL_DIR = ROOT_DIR / "install"


def _run_script(script_name: str) -> bool:
    """Run a bash script in a clean login shell with conda activated."""
    script_path = INSTALL_DIR / script_name
    if not script_path.exists():
        print(f"Warning: {script_path} not found, skipping...")
        return False

    conda_env = os.environ.get("CONDA_DEFAULT_ENV", "")
    conda_prefix = os.environ.get("CONDA_PREFIX", "")
    conda_exe = shutil.which("conda")
    micromamba_exe = shutil.which("micromamba")

    cmd = f"bash {script_path}"
    if conda_env or conda_prefix:
        if conda_exe:
            env_ref = conda_prefix or conda_env
            cmd = (
                f'source "$(conda info --base)/etc/profile.d/conda.sh" '
                f"&& conda activate {env_ref} && bash {script_path}"
            )
        elif micromamba_exe:
            if conda_prefix:
                cmd = f"micromamba run -p {conda_prefix} bash {script_path}"
            else:
                cmd = f"micromamba run -n {conda_env} bash {script_path}"

    # Inherit environment but remove pip's build isolation variables.
    # These cause nested pip calls to look for build backends in the wrong place.
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("_PIP", "PIP_", "PYTHON", "PEP517", "_PEP517", "BUILD_"))
    }

    # Write to stderr so Docker/pip always captures the logs.
    def _log(message: str) -> None:
        sys.stderr.write(message + "\n")
        sys.stderr.flush()

    _log(f"\n>>> Running {script_path}...")
    _log(f">>> Command: {cmd}")

    # Don't run from ROOT_DIR - pip would pick up our pyproject.toml
    process = subprocess.Popen(
        ["bash", "-l", "-c", cmd],
        cwd=os.environ.get("HOME", "/tmp"),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    for line in process.stdout:
        sys.stderr.write(line)
        sys.stderr.flush()

    returncode = process.wait()

    if returncode != 0:
        _log(f"✗ Script {script_name} failed with exit code {returncode}")
        raise subprocess.CalledProcessError(returncode, cmd)

    _log(f"✓ {script_name} completed")


def _run_pre_install():
    """Run pre-install scripts if hooks are re-enabled."""
    _run_script("preinstall_script.sh")


def _run_post_install():
    """Run post-install scripts if hooks are re-enabled."""
    _run_script("postinstall_script.sh")


# =============================================================================
# Hook into METADATA generation (runs BEFORE pip installs dependencies)
# =============================================================================
def get_requires_for_build_editable(config_settings=None):
    """Called by pip before resolving dependencies."""
    # _run_pre_install()
    return _get_requires_for_build_editable(config_settings)


def get_requires_for_build_wheel(config_settings=None):
    """Called by pip before resolving dependencies."""
    # _run_pre_install()
    return _get_requires_for_build_wheel(config_settings)


# =============================================================================
# Hook into BUILD (runs AFTER pip installs dependencies)
# =============================================================================
def build_editable(wheel_directory, config_settings=None, metadata_directory=None):
    """Build editable wheel."""
    result = _build_editable(wheel_directory, config_settings, metadata_directory)
    # _run_post_install()
    return result


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    """Build wheel."""
    result = _build_wheel(wheel_directory, config_settings, metadata_directory)
    # _run_post_install()
    return result
