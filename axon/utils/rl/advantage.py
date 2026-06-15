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
"""
Group-wise helpers for RL training utilities.

Public API:
    - as_torch_index(index, device=None) -> torch.LongTensor
    - group_mean_std(scores, gidx, eps=1e-6, device=None) -> (mean_g, std_g, count_g)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

__all__ = ["as_torch_index", "group_mean_std", "masked_mean", "masked_var", "masked_whiten"]


def _to_1d_numpy_object_array(x: Any) -> np.ndarray:
    """Convert arbitrary input into a 1-D numpy array"""
    try:
        arr = np.asarray(x)
    except Exception:
        try:
            arr = np.array(list(x), dtype=object)
        except Exception:
            arr = np.array([x], dtype=object)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    return arr


def as_torch_index(index: Any, device: torch.device | str | None = None) -> torch.Tensor:
    """Convert arbitrary group labels to a contiguous 1-D torch.long tensor (0..G-1)"""
    # Use provided device, otherwise default to CUDA if available
    if device is not None:
        target = device
    else:
        target = "cuda" if torch.cuda.is_available() else "cpu"

    if isinstance(index, torch.Tensor):
        t = index.reshape(-1)
        if t.dtype in (
            torch.int64,
            torch.int32,
            torch.int16,
            torch.int8,
            getattr(torch, "uint8", torch.uint8),
            torch.bool,
        ):
            return t.to(device=target, dtype=torch.long)

        if t.dtype in (torch.float16, torch.float32, torch.float64, torch.bfloat16):
            t64 = t.to(dtype=torch.float64)
            rounded = torch.round(t64)
            if torch.allclose(t64, rounded, rtol=0.0, atol=1e-6):
                return rounded.to(device=target, dtype=torch.long)
            arr = np.array([str(x.item()) for x in t], dtype=object)
        else:
            arr = np.array([str(x.item()) if hasattr(x, "item") else str(x) for x in t], dtype=object)

    else:
        arr = _to_1d_numpy_object_array(index)

        if arr.dtype != object and np.issubdtype(arr.dtype, np.integer):
            return torch.from_numpy(arr.astype(np.int64, copy=False)).to(device=target)

        if arr.dtype != object and np.issubdtype(arr.dtype, np.floating):
            arr64 = arr.astype(np.float64, copy=False)
            rounded = np.rint(arr64)
            if np.allclose(arr64, rounded, rtol=0.0, atol=1e-6):
                return torch.from_numpy(rounded.astype(np.int64)).to(device=target)

        try:
            coerced = arr.astype(np.int64)
            return torch.from_numpy(coerced).to(device=target)
        except Exception:
            pass

        if arr.dtype != object:
            arr = arr.astype(object)

    try:
        _, inv = np.unique(arr, return_inverse=True)
    except Exception:
        sarr = np.array([str(x) for x in arr], dtype=object)
        _, inv = np.unique(sarr, return_inverse=True)

    inv = inv.astype(np.int64, copy=False)
    return torch.from_numpy(inv).to(device=target)


@torch.no_grad()
def group_mean_std(
    scores: torch.Tensor,
    gidx: torch.Tensor,
    eps: float = 1e-6,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute per-group mean/std/count with Bessel correction. Singleton groups get mean=0, std=1."""
    # Use provided device, or infer from input tensor
    if device is not None:
        target = device
    else:
        target = scores.device

    scores = scores.reshape(-1).to(device=target, dtype=torch.float32)
    gidx = gidx.reshape(-1).to(device=target, dtype=torch.long)

    if scores.numel() != gidx.numel():
        raise ValueError(f"scores and gidx length mismatch: {scores.numel()} vs {gidx.numel()}")

    G = int(torch.max(gidx).item()) + 1 if gidx.numel() > 0 else 0
    if G == 0:
        empty = torch.empty(0, device=target, dtype=torch.float32)
        return empty, empty, empty

    ones = torch.ones_like(scores, dtype=torch.float32)

    count = torch.zeros(G, device=target, dtype=torch.float32).index_add_(0, gidx, ones)
    s1 = torch.zeros(G, device=target, dtype=torch.float32).index_add_(0, gidx, scores)
    s2 = torch.zeros(G, device=target, dtype=torch.float32).index_add_(0, gidx, scores * scores)

    mean = s1 / count.clamp_min(1.0)
    var_num = s2 - (s1 * s1) / count.clamp_min(1.0)
    denom = (count - 1.0).clamp_min(1.0)
    var = var_num / denom
    std = torch.sqrt(torch.clamp(var, min=eps))

    single = count <= 1.0
    if torch.any(single):
        mean = mean.clone()
        std = std.clone()
        mean[single] = 0.0
        std[single] = 1.0

    return mean, std, count


@torch.no_grad()
def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Compute mean of values where mask is True."""
    mask = mask.to(values.dtype)
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


@torch.no_grad()
def masked_var(values: torch.Tensor, mask: torch.Tensor, bessel_correction: bool = True) -> torch.Tensor:
    """Compute variance of values where mask is True."""
    mask = mask.to(values.dtype)
    mask_sum = mask.sum()
    mean = (values * mask).sum() / mask_sum.clamp_min(1.0)
    variance = ((values - mean) ** 2 * mask).sum() / mask_sum.clamp_min(1.0)
    if bessel_correction and mask_sum > 1:
        variance = variance * mask_sum / (mask_sum - 1)
    return variance


@torch.no_grad()
def masked_whiten(values: torch.Tensor, mask: torch.Tensor, shift_mean: bool = True) -> torch.Tensor:
    """
    Whiten values by normalizing with mean and variance computed over mask.

    Args:
        values: Input tensor
        mask: Boolean tensor of same shape, selects elements for stats
        shift_mean: If True (default), output is zero-mean;
                    if False, the original mean is re-added after scaling

    Returns:
        Whitened tensor of same shape as values
    """
    mean, var = masked_mean(values, mask), masked_var(values, mask)
    whitened = (values - mean) * torch.rsqrt(var + 1e-8)
    if not shift_mean:
        whitened = whitened + mean
    return whitened
