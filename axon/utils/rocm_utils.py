# Copyright 2025 Model AI Corp.
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
"""ROCm/HIP compatibility utilities.

Provides detection helpers and workarounds for running on AMD GPUs with ROCm.
Based on the approach used by the miles framework.
"""

import logging

import torch

logger = logging.getLogger(__name__)


def is_rocm() -> bool:
    """Return True when running on AMD ROCm/HIP (not NVIDIA CUDA)."""
    return hasattr(torch.version, "hip") and torch.version.hip is not None


def get_visible_devices_env_key() -> str:
    """Return the environment variable name used to control GPU visibility.

    On ROCm this is HIP_VISIBLE_DEVICES; on CUDA it is CUDA_VISIBLE_DEVICES.
    """
    return "HIP_VISIBLE_DEVICES" if is_rocm() else "CUDA_VISIBLE_DEVICES"


# ---------------------------------------------------------------------------
# ROCm checkpoint writer fix
# ---------------------------------------------------------------------------
# On ROCm/HIP, using non_blocking=True in preload_tensors causes tensors to
# be stored in pinned memory.  Forking a subprocess afterward (which
# FileSystemWriterAsync does) triggers a segmentation fault.
#
# This is the same fix used by the miles framework
# (miles/utils/rocm_checkpoint_writer.py): subclass FileSystemWriterAsync
# and override preload_tensors to force non_blocking=False on HIP.
# ---------------------------------------------------------------------------

_ROCM_CKPT_PATCH_APPLIED = False


def apply_rocm_checkpoint_writer_patch() -> None:
    """Replace Megatron's FileSystemWriterAsync with a ROCm-safe subclass.

    Uses class replacement (not method monkey-patching) to match the
    upstream miles approach.  Safe to call multiple times; only patches
    once.  No-op on CUDA.
    """
    global _ROCM_CKPT_PATCH_APPLIED
    if _ROCM_CKPT_PATCH_APPLIED or not is_rocm():
        return

    try:
        import megatron.core.dist_checkpointing.strategies.filesystem_async as fs_mod
        from megatron.core.dist_checkpointing.strategies.filesystem_async import FileSystemWriterAsync

        class ROCmFileSystemWriterAsync(FileSystemWriterAsync):
            """FileSystemWriterAsync wrapper for ROCm compatibility.

            On ROCm/HIP, using non_blocking=True causes tensors to be stored
            in pinned memory, which triggers segmentation faults when forking
            subprocesses afterward.
            """

            @staticmethod
            def preload_tensors(*args, **kwargs):
                if torch.version.hip:
                    # Force synchronous copy to avoid pinned-memory + fork segfault
                    if "non_blocking" in kwargs:
                        kwargs["non_blocking"] = False
                    elif len(args) > 1 and isinstance(args[-1], bool):
                        # non_blocking passed as positional argument
                        args = args[:-1] + (False,)
                return FileSystemWriterAsync.preload_tensors(*args, **kwargs)

        fs_mod.FileSystemWriterAsync = ROCmFileSystemWriterAsync
        _ROCM_CKPT_PATCH_APPLIED = True
        logger.info("[ROCm] Applied FileSystemWriterAsync patch for HIP compatibility")
    except (ImportError, AttributeError):
        # Megatron not installed or API changed – skip silently
        pass


def get_rocm_env_vars() -> dict[str, str]:
    """Return recommended environment variables for ROCm training.

    These mirror the tuning knobs used by production ROCm training setups
    (MI300/MI350).  Returns an empty dict on CUDA.
    """
    if not is_rocm():
        return {}
    return {
        "HIP_FORCE_DEV_KERNARG": "1",
        "HSA_NO_SCRATCH_RECLAIM": "1",
        "NCCL_MIN_NCHANNELS": "112",
        "TORCHINDUCTOR_MAX_AUTOTUNE": "1",
        "TORCHINDUCTOR_MAX_AUTOTUNE_POINTWISE": "1",
        # vLLM / SGLang FP8 padding for ROCm
        "VLLM_FP8_PADDING": "1",
        "VLLM_FP8_ACT_PADDING": "1",
        "VLLM_FP8_WEIGHT_PADDING": "1",
        "VLLM_FP8_REDUCE_CONV": "1",
        # SGLang ROCm-specific
        "SGLANG_USE_AITER": "1",
        "SGLANG_MOE_PADDING": "1",
        "SGLANG_SET_CPU_AFFINITY": "1",
        "SGLANG_ROCM_FUSED_DECODE_MLA": "1",
        "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN": "1",
    }
