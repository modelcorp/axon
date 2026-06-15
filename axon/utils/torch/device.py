# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# This code is inspired by the torchtune.
# https://github.com/pytorch/torchtune/blob/main/torchtune/utils/_device.py
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license in https://github.com/pytorch/torchtune/blob/main/LICENSE

"""
Device utilities for hardware-agnostic code.

This module provides utilities for detecting and working with different compute
devices (CUDA, NPU, CPU). It abstracts device-specific logic to allow the rest
of the codebase to work seamlessly across different hardware backends.

Supported devices (in priority order):
    1. CUDA (NVIDIA GPUs)
    2. NPU (Huawei Ascend)
    3. CPU (fallback)

Example:
    >>> from axon.utils.torch import get_device_name, get_torch_device
    >>> device_name = get_device_name()  # e.g., "cuda"
    >>> torch_device = get_torch_device()  # e.g., torch.cuda
    >>> current_id = torch_device.current_device()
"""

from __future__ import annotations

from types import ModuleType

import torch

# Cache availability checks at module load time for performance
is_cuda_available: bool = torch.cuda.is_available()
is_npu_available: bool = hasattr(torch, "npu") and torch.npu.is_available()
is_rocm: bool = hasattr(torch.version, "hip") and torch.version.hip is not None


def get_device_name() -> str:
    """Get the name of the best available compute device.

    Checks for available hardware in priority order: CUDA > NPU > CPU.

    Returns:
        Device name string: "cuda", "npu", or "cpu".

    Example:
        >>> device = torch.device(get_device_name())
    """
    if is_cuda_available:
        return "cuda"
    if is_npu_available:
        return "npu"
    return "cpu"


def get_torch_device() -> ModuleType:
    """Get the torch device module for the current hardware.

    Returns the appropriate torch namespace (e.g., `torch.cuda`, `torch.npu`)
    for the detected device, allowing device-agnostic operations like
    `get_torch_device().current_device()`.

    Returns:
        The torch device module (torch.cuda, torch.npu, or torch.cpu).
        Falls back to torch.cuda with a warning if the device namespace is not found.

    Example:
        >>> device_module = get_torch_device()
        >>> device_module.synchronize()  # Works for cuda/npu
    """
    device_name = get_device_name()
    device_module = getattr(torch, device_name, None)
    return device_module if device_module is not None else torch.cuda


def get_device_id() -> int:
    """Get the current device index.

    Returns:
        The index of the currently selected device (e.g., 0 for "cuda:0").

    Example:
        >>> device_id = get_device_id()
        >>> tensor = tensor.to(f"cuda:{device_id}")
    """
    return get_torch_device().current_device()


def get_nccl_backend() -> str:
    """Get the appropriate collective communication backend name.

    Returns the backend name for distributed operations:
    - "hccl" for Huawei NPU (Huawei Collective Communication Library)
    - "nccl" for CUDA/CPU (NVIDIA Collective Communications Library)

    Returns:
        Backend name string: "hccl" or "nccl".

    Example:
        >>> import torch.distributed as dist
        >>> dist.init_process_group(backend=get_nccl_backend())
    """
    return "hccl" if is_npu_available else "nccl"


def set_expandable_segments(enable: bool) -> None:
    """Configure CUDA memory allocator expandable segments.

    Expandable segments allow the CUDA allocator to expand memory blocks
    instead of allocating new ones, which can help reduce memory fragmentation
    and avoid OOM errors in some scenarios.

    Args:
        enable: Whether to enable expandable segments.

    Note:
        This is a no-op on non-CUDA devices.

    Example:
        >>> set_expandable_segments(True)  # Enable to reduce fragmentation
    """
    if is_cuda_available:
        torch.cuda.memory._set_allocator_settings(f"expandable_segments:{enable}")
