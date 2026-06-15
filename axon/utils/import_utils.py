# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Utilities to check if packages are available.
We assume package availability won't change during runtime.
"""

import importlib
import importlib.util
from functools import cache

from axon.utils.py_utils import deprecated


@cache
def is_megatron_core_available():
    try:
        mcore_spec = importlib.util.find_spec("megatron.core")
    except ModuleNotFoundError:
        mcore_spec = None
    return mcore_spec is not None


@cache
def is_vllm_available():
    try:
        vllm_spec = importlib.util.find_spec("vllm")
    except ModuleNotFoundError:
        vllm_spec = None
    return vllm_spec is not None


@cache
def is_sglang_available():
    try:
        sglang_spec = importlib.util.find_spec("sglang")
    except ModuleNotFoundError:
        sglang_spec = None
    return sglang_spec is not None


@cache
def is_nvtx_available():
    try:
        nvtx_spec = importlib.util.find_spec("nvtx")
    except ModuleNotFoundError:
        nvtx_spec = None
    return nvtx_spec is not None


@cache
def is_trl_available():
    try:
        trl_spec = importlib.util.find_spec("trl")
    except ModuleNotFoundError:
        trl_spec = None
    return trl_spec is not None


def import_external_libs(external_libs=None):
    if external_libs is None:
        return
    if not isinstance(external_libs, list):
        external_libs = [external_libs]
    import importlib

    for external_lib in external_libs:
        importlib.import_module(external_lib)


PKG_PATH_PREFIX = "pkg://"
FILE_PATH_PREFIX = "file://"


def load_module(module_path: str, module_name: str | None = None) -> object:
    """Re-export of :func:`axon.utils.module_loader.load_module`."""
    from axon.utils.module_loader import load_module as _load_module

    return _load_module(module_path, module_name)


def load_extern_object(module_path: str, object_name: str) -> object:
    """Load an object from a module path.

    Args:
        module_path (str): See :func:`load_module`.
        object_name (str):
            The name of the object to load with ``getattr(module, object_name)``.
    """
    module = load_module(module_path)

    if not hasattr(module, object_name):
        raise AttributeError(f"Object not found in module: {object_name=}, {module_path=}.")

    return getattr(module, object_name)


def load_class_from_fqn(fqn: str, description: str = "class") -> type:
    """Load a class from its fully qualified name.

    Args:
        fqn: Fully qualified class name (e.g., 'mypackage.module.ClassName').
        description: Description for error messages (e.g., 'AgentLoopManager').

    Returns:
        The loaded class.

    Raises:
        ValueError: If fqn format is invalid (missing dot separator).
        ImportError: If the module cannot be imported.
        AttributeError: If the class is not found in the module.

    Example:
        >>> cls = load_class_from_fqn("axon.programs.external.nemo_gym_program.NemoGymProgram")
        >>> instance = cls(config=config, ...)
    """
    if "." not in fqn:
        raise ValueError(
            f"Invalid {description} '{fqn}'. Expected fully qualified class name (e.g., 'mypackage.module.ClassName')."
        )
    try:
        module_path, class_name = fqn.rsplit(".", 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    except ImportError as e:
        raise ImportError(f"Failed to import module '{module_path}' for {description}: {e}") from e
    except AttributeError as e:
        raise AttributeError(f"Class '{class_name}' not found in module '{module_path}': {e}") from e


@deprecated(replacement="load_module(file_path); getattr(module, type_name)")
def load_extern_type(file_path: str, type_name: str) -> type:
    """DEPRECATED. Directly use `load_extern_object` instead."""
    return load_extern_object(file_path, type_name)
