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
#
# Ported from miles quantizer_mxfp8.py (github.com/radixark/miles), Apache-2.0.

"""MxFP8 (Mixed-Format FP8) quantization utilities for weight compression.

MxFP8 is a group-wise FP8 quantization format where:
- Weights are quantized to ``float8_e4m3fn`` (E4M3 format)
- Groups of 32 elements share a single scale factor
- Scale factors are stored in UE8M0 format (uint8) for 4x memory savings
  over standard float32 scales

This format is designed for Blackwell-class GPUs and requires SGLang's
``mxfp8_group_quantize`` utility.

Ported from miles' ``quantizer_mxfp8.py``.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable

import torch

try:
    from sglang.srt.layers.quantization.fp8_utils import mxfp8_group_quantize
except ImportError:
    mxfp8_group_quantize = None

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("AXON_LOGGING_LEVEL", "INFO"))

# MxFP8 always quantizes in groups of 32 along the last dimension
_MXFP8_GROUP_SIZE = 32

# Layer filtering: matches miles' SKIP_WEIGHT_SUBSTRINGS exactly
_SKIP_WEIGHT_SUBSTRINGS = (
    "layernorm",
    "embed",
    "router",
    "mlp.gate.",
    "norm",
    "lm_head",
    "eh_proj",
    "weights_proj",
)


def should_quantize_mxfp8(name: str, param: torch.Tensor) -> bool:
    """Determine whether to quantize a weight tensor to MxFP8.

    Uses the same layer filtering logic as miles' convert_hf_to_mxfp8.py.
    """
    if not name.endswith(".weight"):
        return False
    if any(substr in name for substr in _SKIP_WEIGHT_SUBSTRINGS):
        return False
    if param.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return False
    if param.dim() < 2:
        return False
    if param.shape[-1] % _MXFP8_GROUP_SIZE != 0:
        return False
    return True


def quantize_weight_mxfp8(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a weight tensor to MxFP8 format.

    Args:
        weight: 2-D weight tensor [M, N] where N must be divisible by 32.

    Returns:
        (qweight, scale) where:
        - qweight: float8_e4m3fn, same shape as input
        - scale: uint8 (UE8M0 format), shape [M, N // 32]

    Raises:
        RuntimeError: If SGLang's mxfp8_group_quantize is not available.
        ValueError: If last dimension is not divisible by 32.
    """
    if mxfp8_group_quantize is None:
        raise RuntimeError(
            "MxFP8 quantization requires sglang with fp8_utils.mxfp8_group_quantize. "
            "Please install a compatible version of sglang."
        )

    weight = weight.contiguous()
    k = weight.shape[-1]
    if k % _MXFP8_GROUP_SIZE != 0:
        raise ValueError(f"Last dimension {k} must be divisible by {_MXFP8_GROUP_SIZE} for MxFP8 quantization.")

    # Flatten to 2D for the quantization call
    orig_shape = weight.shape
    weight_flat = weight.view(-1, k).contiguous()

    qweight, scale = mxfp8_group_quantize(weight_flat)

    # Reshape back to original shape (+ compressed scale shape)
    qweight = qweight.view(orig_shape)
    scale = scale.view(*orig_shape[:-1], k // _MXFP8_GROUP_SIZE).contiguous()

    return qweight, scale


def mxfp8_quantize_weight_generator(
    weights: Iterable[tuple[str, torch.Tensor]],
    quant_config: dict,
    dtype: torch.dtype = torch.bfloat16,
) -> Iterable[tuple[str, torch.Tensor]]:
    """Lazily quantize weights to MxFP8, yielding ``(name, tensor)`` pairs.

    For each quantizable weight the generator yields two pairs:
    ``(name, fp8_weight)`` and ``(name + "_scale_inv", scale)``.
    Non-quantizable parameters are yielded unchanged.

    Config keys:
    - ``weight_block_size``: should be [1, 32] for MxFP8 (default)
    """
    for name, param in weights:
        # Skip pre-existing scale tensors
        if name.endswith("_scale") or name.endswith("_scale_inv"):
            continue

        if not should_quantize_mxfp8(name, param):
            yield (name, param)
            continue

        try:
            logger.debug(f"  Quantizing to MxFP8: {name}")
            # Convert to target dtype first (trainer may send fp32)
            w = param.to(dtype) if param.dtype != dtype else param
            qweight, scale = quantize_weight_mxfp8(w)

            scale_name = name.replace(".weight", ".weight_scale_inv")
            yield (name, qweight)
            yield (scale_name, scale)
        except Exception as e:
            logger.error(f"Failed to quantize MxFP8 {name}: {e}")
            yield (name, param)
