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
# Portions copied from miles quantizer_compressed_tensors.py (github.com/radixark/miles), Apache-2.0.

"""INT4 quantization utilities for weight compression during rollout.

Training stays in BF16/FP32; only weights sent to inference engines are
quantized to INT4 and packed into the compressed-tensors format consumed
by SGLang / vLLM.

The packing logic (``pack_to_int32``, ``quantize``, ``pack_layer``) is
copied directly from miles' ``quantizer_compressed_tensors.py`` to
guarantee bit-exact compatibility.  The CUDA fake-quantization kernel
(``fake_int4_quant_cuda``) is replaced by a portable Triton kernel with
a pure-PyTorch fallback.
"""

from __future__ import annotations

import logging
import math
import os
import re
from collections.abc import Iterable

import torch

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ImportError:
    _TRITON_AVAILABLE = False

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("AXON_LOGGING_LEVEL", "INFO"))

# ---------------------------------------------------------------------------
# Triton kernel – drop-in replacement for miles' fake_int4_quant_cuda
#
# The CUDA kernel computes, per group of GROUP_SIZE elements:
#   scale = max(max(|group|) / 7, 1e-5)          [symmetric]
#   q     = rintf(val / scale)                     [rounded integer]
# and returns (q_float, scale, zero_point) all as the same float dtype.
# ---------------------------------------------------------------------------
if _TRITON_AVAILABLE:

    @triton.jit
    def _int4_fake_quant_kernel(
        X,
        Q,  # output: quantised integer values as float
        S,  # output: per-group scales
        stride_xm,
        stride_xn,
        stride_qm,
        stride_qn,
        stride_sm,
        stride_sn,
        M,
        N,
        GROUP_SIZE: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_g = tl.program_id(1)

        row = pid_m
        col_start = pid_g * GROUP_SIZE
        offs_n = col_start + tl.arange(0, GROUP_SIZE)

        mask = (row < M) & (offs_n < N)

        x = tl.load(X + row * stride_xm + offs_n * stride_xn, mask=mask, other=0.0).to(tl.float32)

        # scale = max(max(|x|) / 7, 1e-5)  — matches miles CUDA kernel
        abs_max = tl.max(tl.abs(x))
        scale = tl.maximum(abs_max * (1.0 / 7.0), 1e-5)

        # q = rintf(x / scale)  — miles uses C rintf ≡ round-half-to-even
        q = tl.extra.cuda.libdevice.nearbyint(x / scale)

        tl.store(Q + row * stride_qm + offs_n * stride_qn, q.to(Q.dtype.element_ty), mask=mask)
        tl.store(S + pid_m * stride_sm + pid_g * stride_sn, scale.to(S.dtype.element_ty))


def _fake_int4_quant_triton(
    weight: torch.Tensor, group_size: int, sym: bool = True
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Triton replacement for ``fake_int4_quant_cuda(weight, (1, group_size), sym)``."""
    assert sym, "Triton INT4 kernel only supports symmetric mode"
    M, N = weight.shape
    n_groups = N // group_size

    q = torch.empty_like(weight)
    s = torch.empty(M, n_groups, dtype=weight.dtype, device=weight.device)
    zp = torch.zeros_like(s)

    _int4_fake_quant_kernel[(M, n_groups)](
        weight,
        q,
        s,
        *weight.stride(),
        *q.stride(),
        *s.stride(),
        M,
        N,
        GROUP_SIZE=group_size,
        num_warps=1,
        num_stages=1,
    )
    return q, s, zp


def _fake_int4_quant_torch(
    weight: torch.Tensor, group_size: int, sym: bool = True
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pure-PyTorch replacement for ``fake_int4_quant_cuda``."""
    M, N = weight.shape
    w = weight.float().reshape(M, N // group_size, group_size)

    if sym:
        abs_max = w.abs().amax(dim=-1, keepdim=True)
        scale = (abs_max / 7.0).clamp(min=1e-5)
        q = (w / scale).round()  # integers in roughly [-7, 7]
    else:
        vmin = w.amin(dim=-1, keepdim=True)
        vmax = w.amax(dim=-1, keepdim=True)
        scale = ((vmax - vmin) / 15.0).clamp(min=1e-5)
        zp_float = (-vmin / scale).round().clamp(0, 15)
        q = (w / scale).round() + zp_float

    q_flat = q.reshape(M, N).to(weight.dtype)
    s_flat = scale.squeeze(-1).to(weight.dtype)
    if sym:
        zp_flat = torch.zeros(M, N // group_size, dtype=weight.dtype, device=weight.device)
    else:
        zp_flat = zp_float.squeeze(-1).to(weight.dtype)
    return q_flat, s_flat, zp_flat


def fake_int4_quant(
    weight: torch.Tensor, block_size: tuple[int, int], sym: bool = True
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Drop-in replacement for miles' ``fake_int4_quant_cuda.fake_int4_quant_cuda``.

    Args:
        weight: 2-D tensor [M, N].
        block_size: (block_m, block_n).  Only (1, group_size) is supported for
                    the Triton path; PyTorch fallback handles all sizes.
        sym: symmetric quantization.

    Returns:
        (q, scale, zero_point) — all same dtype as ``weight``.
        ``q`` contains the quantised integer values as floats.
    """
    assert weight.ndim == 2
    block_m, block_n = block_size
    assert block_m == 1, "Only block_m=1 is supported (matches miles default)"
    group_size = block_n
    assert weight.shape[1] % group_size == 0

    # Triton's tl.arange requires power-of-2 size; fall back to PyTorch otherwise
    _is_pow2 = group_size > 0 and (group_size & (group_size - 1)) == 0
    if _TRITON_AVAILABLE and weight.is_cuda and sym and _is_pow2:
        return _fake_int4_quant_triton(weight, group_size, sym)
    return _fake_int4_quant_torch(weight, group_size, sym)


# ---------------------------------------------------------------------------
# Packing helpers — copied from miles' quantizer_compressed_tensors.py
# to guarantee bit-exact format compatibility with SGLang/vLLM.
# ---------------------------------------------------------------------------


def round_to_quantized_type_dtype(tensor, dtype, cast_to_original_dtype=False):
    """Round and clamp a tensor to fit within the range of ``dtype``."""
    original_dtype = tensor.dtype
    iinfo = torch.iinfo(dtype)
    rounded = torch.round(torch.clamp(tensor, iinfo.min, iinfo.max)).to(dtype)
    if cast_to_original_dtype:
        return rounded.to(original_dtype)
    return rounded


@torch.no_grad()
def quantize(x, scale, zero_point, dtype=torch.int8):
    """Re-quantize dequantized values back to integer type.

    Copied from miles' ``quantizer_compressed_tensors.quantize``.
    """
    group_size = x.shape[-1] // scale.shape[-1]
    reshaped_dims = (math.ceil(x.shape[-1] / group_size), group_size)
    x = x.unflatten(-1, reshaped_dims)

    scaled = x / scale.unsqueeze(-1)
    if zero_point is not None:
        scaled += zero_point.unsqueeze(-1).to(x.dtype)

    output = round_to_quantized_type_dtype(tensor=scaled, dtype=dtype)
    output = output.flatten(start_dim=-2).to(dtype)
    return output


def pack_to_int32(value, num_bits, packed_dim=1, sym=False):
    """Pack low-bit integer values into int32.

    Copied from miles' ``quantizer_compressed_tensors.pack_to_int32``.
    """
    if num_bits > 8:
        raise ValueError("Packing is only supported for less than 8 bits")
    if num_bits < 1:
        raise ValueError(f"num_bits must be at least 1, got {num_bits}")

    # Convert to unsigned range for packing, matching quantization offset
    if sym:
        offset = 1 << (num_bits - 1)
        value = (value + offset).to(torch.uint8)
    device = value.device

    pack_factor = 32 // num_bits

    if packed_dim == 0:
        value = value.transpose(0, 1)

    rows, cols = value.shape
    padded_cols = math.ceil(cols / pack_factor) * pack_factor
    pad_len = padded_cols - cols

    if pad_len > 0:
        value = torch.nn.functional.pad(value, (0, pad_len))

    num_groups = padded_cols // pack_factor

    reshaped = value.view(rows, num_groups, pack_factor).to(torch.int32)
    bit_shifts = torch.arange(pack_factor, device=device, dtype=torch.int32) * num_bits
    packed = (reshaped << bit_shifts).sum(dim=2, dtype=torch.int32)

    if packed_dim == 0:
        packed = packed.transpose(0, 1)

    return packed


def pack_layer(weight, group_size, sym=True):
    """Quantize and pack a weight tensor to compressed-tensors format.

    Copied from miles' ``quantizer_compressed_tensors.pack_layer``,
    substituting our ``fake_int4_quant`` for ``fake_int4_quant_cuda``.

    Returns:
        (packed_weight, scales, packed_zero_point)
        - packed_weight: int32 [M, N // 8]
        - scales: same dtype as weight, [M, N // group_size]
        - packed_zero_point: int32 (asymmetric) or None (symmetric)
    """
    w, scale, zp = fake_int4_quant(weight, (1, group_size), sym)

    w = w.view(weight.shape[0], 1, weight.shape[1] // group_size, group_size)
    scale = scale.view(weight.shape[0], 1, weight.shape[1] // group_size, 1)
    zp = zp.view(weight.shape[0], 1, weight.shape[1] // group_size, 1)

    if sym:
        w = w * scale
    else:
        w = (w - zp) * scale

    w = w.view(weight.shape)
    scale = scale.view(weight.shape[0], -1).contiguous()

    if not sym:
        zp = zp.view(weight.shape[0], -1)
        zeros = zp.t().contiguous().to(torch.float32)
        zeros = zeros.to(dtype=torch.int32, device=w.device)
        zeros = zeros.reshape(-1, zeros.shape[1] // 8, 8)
        # Interleaved bit ordering for zero-points (matches miles)
        new_order_map = torch.tensor([0, 4, 1, 5, 2, 6, 3, 7], device=zeros.device) * 4
        zeros = zeros << new_order_map
        packed_zp = torch.sum(zeros, dim=-1).to(torch.int32)
    else:
        zp = None
        packed_zp = None

    quantized_weight = quantize(
        x=w,
        scale=scale,
        zero_point=zp,
        dtype=torch.int8 if sym else torch.uint8,
    )
    packed_weight = pack_to_int32(quantized_weight, 4, sym=sym)
    return packed_weight, scale, packed_zp


# ---------------------------------------------------------------------------
# Generator-based quantization for the weight-sync pipeline
# ---------------------------------------------------------------------------


def _matches_ignore(name: str, ignore_rules: list[str]) -> bool:
    """Check if a parameter name matches any ignore rule.

    Matches miles' convention: rules prefixed with ``re:`` are regex,
    otherwise prefix/exact match.
    """
    for rule in ignore_rules:
        if rule.startswith("re:"):
            if re.match(rule[3:], name):
                return True
        elif rule == name or name.startswith(rule):
            return True
    return False


def int4_quantize_weight_generator(
    weights: Iterable[tuple[str, torch.Tensor]],
    quant_config: dict,
    dtype: torch.dtype = torch.bfloat16,
) -> Iterable[tuple[str, torch.Tensor]]:
    """Lazily quantize weights to INT4 compressed-tensors format.

    For each quantizable weight the generator yields (matching miles' naming):
    - ``(name.replace('.weight', '.weight_packed'), packed_int32)``
    - ``(name.replace('.weight', '.weight_scale'), scales)``
    - ``(name.replace('.weight', '.weight_shape'), shape_tensor)``
    - ``(name.replace('.weight', '.weight_zero_point'), zp)``  [asymmetric only]

    Non-quantizable parameters (embeddings, norms, etc.) are yielded unchanged.

    Config keys:
    - ``group_size``: INT4 quantization group size (default 128)
    - ``symmetric``: whether to use symmetric quantization (default True)
    - ``ignore``: list of name / regex patterns to skip
    """
    group_size = quant_config.get("group_size", 128)
    symmetric = quant_config.get("symmetric", True)
    ignore_rules = quant_config.get("ignore", [])

    from axon.utils.sglang.fp8 import should_quantize_param

    for name, param in weights:
        # Skip pre-existing auxiliary tensors
        if name.endswith("_scale") or name.endswith("_scale_inv") or name.endswith("_packed"):
            continue

        is_ignored = _matches_ignore(name, ignore_rules)

        # Two-layer filtering:
        # 1. should_quantize_param (include-list shared with FP8): only quantize
        #    known linear layers (q/k/v/o_proj, gate/up/down_proj, experts, etc.)
        #    This is a safety net — INT4 is aggressive (16 levels) so we avoid
        #    quantizing layers that may be sensitive to precision loss.
        # 2. ignore rules from model's quantization_config: honour any additional
        #    per-model exclusions set during offline conversion.
        if is_ignored or not should_quantize_param(name) or param.ndim < 2:
            yield (name, param)
            continue

        w = param.to(dtype)
        orig_shape = w.shape
        w2d = w.reshape(-1, w.shape[-1])

        if w2d.shape[-1] % group_size != 0:
            logger.warning(
                f"Skipping INT4 for {name}: last dim {w2d.shape[-1]} not divisible by group_size {group_size}"
            )
            yield (name, param)
            continue

        try:
            logger.debug(f"  Quantizing to INT4: {name}")
            qw, s, zp = pack_layer(w2d, group_size, symmetric)

            # Naming convention: matches miles' quantize_params_compressed_tensors
            qweight_name = name.replace(".weight", ".weight_packed")
            scale_name = name.replace(".weight", ".weight_scale")
            weight_shape = torch.tensor(list(orig_shape), dtype=torch.int64, device=qw.device)
            weight_shape_name = name.replace(".weight", ".weight_shape")

            if zp is not None:
                zp_name = name.replace(".weight", ".weight_zero_point")
                yield (zp_name, zp)
            yield (qweight_name, qw)
            yield (scale_name, s)
            yield (weight_shape_name, weight_shape)
        except Exception as e:
            logger.error(f"Failed to quantize INT4 {name}: {e}")
            yield (name, param)
