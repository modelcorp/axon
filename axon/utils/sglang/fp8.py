# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# Copyright 2025 z.ai
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
# Ported from slime backends/megatron_utils/kernels/fp8_kernel.py (github.com/THUDM/slime), Apache-2.0.

import logging
import os
from collections.abc import Iterable

import torch

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ImportError:
    _TRITON_AVAILABLE = False

# DeepGEMM UE8M0 scale format support (needed for newer SGLang with DeepGEMM kernels)
try:
    from sglang.srt.layers.quantization.fp8_utils import quant_weight_ue8m0, transform_scale_ue8m0
    from sglang.srt.model_loader.utils import should_deepgemm_weight_requant_ue8m0
except ImportError:
    quant_weight_ue8m0 = None
    transform_scale_ue8m0 = None
    should_deepgemm_weight_requant_ue8m0 = None

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("AXON_LOGGING_LEVEL", "INFO"))

# ---------------------------------------------------------------------------
# Triton FP8 blockwise quantization kernel
# Ported from slime/backends/megatron_utils/kernels/fp8_kernel.py
# ---------------------------------------------------------------------------
_FP8_DTYPE = torch.float8_e4m3fn
_FP8_MAX = torch.finfo(_FP8_DTYPE).max
_FP8_MIN = -_FP8_MAX


def _ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


if _TRITON_AVAILABLE:

    @triton.jit
    def _blockwise_cast_to_fp8_triton(
        X,
        Y,
        S,
        stride_xm,
        stride_xn,
        stride_ym,
        stride_yn,
        stride_sm,
        stride_sn,
        M,
        N,
        eps,
        fp8_min,
        fp8_max,
        BLOCK_M: tl.constexpr = 32,
        BLOCK_N: tl.constexpr = 128,
    ):
        pid_m = tl.cast(tl.program_id(axis=0), tl.int64)
        pid_n = tl.cast(tl.program_id(axis=1), tl.int64)
        off_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        off_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_m = off_m < M
        mask_n = off_n < N
        mask = mask_m[:, None] & mask_n[None, :]

        x = tl.load(X + off_m[:, None] * stride_xm + off_n[None, :] * stride_xn, mask=mask, other=0.0).to(tl.float32)
        _absmax = tl.maximum(tl.max(tl.abs(x)), eps)
        x_s = _absmax / fp8_max
        s_inv = 1.0 / x_s
        y_q = tl.clamp(x * s_inv, fp8_min, fp8_max).to(Y.dtype.element_ty)

        tl.store(Y + off_m[:, None] * stride_ym + off_n[None, :] * stride_yn, y_q, mask=mask)
        tl.store(S + pid_m * stride_sm + pid_n * stride_sn, x_s)

    def blockwise_cast_to_fp8_triton(x: torch.Tensor, block_size=None) -> tuple[torch.Tensor, torch.Tensor]:
        BLOCK_M, BLOCK_N = 128, 128
        if block_size:
            BLOCK_M, BLOCK_N = block_size[0], block_size[1]
        M, N = x.shape
        y = torch.empty(M, N, device=x.device, dtype=_FP8_DTYPE)
        s = torch.empty(_ceil_div(M, BLOCK_M), _ceil_div(N, BLOCK_N), dtype=torch.float32, device=x.device)

        def grid(meta):
            return (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(N, meta["BLOCK_N"]))

        if x.is_contiguous():
            kwargs = {"BLOCK_M": BLOCK_M, "BLOCK_N": BLOCK_N, "num_warps": 8, "num_stages": 2}
        else:
            kwargs = {"BLOCK_M": BLOCK_M, "BLOCK_N": BLOCK_N, "num_warps": 1, "num_stages": 4}
        _blockwise_cast_to_fp8_triton[grid](
            x, y, s, *x.stride(), *y.stride(), *s.stride(), M, N, 1e-10, _FP8_MIN, _FP8_MAX, **kwargs
        )
        return y, s


def should_quantize_param(param_name: str) -> bool:
    """Determine whether to quantize to FP8 based on parameter name.

    Quantization rules:
    - Must end with .weight (exclude bias)
    - Exclude embedding layers, normalization layers, output layer, router/gate layers
    - Include all linear projection layers across standard, MLA, MoE, indexer,
      and linear-attention architectures
    """
    # Must be a weight parameter
    if not param_name.endswith(".weight"):
        return False

    # Layer types to exclude
    exclude_patterns = [
        "embed_tokens",  # Embedding layer
        "lm_head",  # Output layer
        "layernorm",  # LayerNorm
        "norm",  # Various Norm layers (rmsnorm, etc.)
        "ln_",  # LayerNorm variants
        "embeddings",  # Embeddings
        "router",  # MoE router (kept in high precision)
        "mlp.gate.",  # MoE gate (distinct from gate_proj MLP)
        "eh_proj",  # GLM5 eh_proj (kept in high precision)
        "weights_proj",  # GLM5 DSA weights_proj (kept in high precision)
    ]

    # Check if matches exclude patterns
    param_lower = param_name.lower()
    for pattern in exclude_patterns:
        if pattern in param_lower:
            return False

    # Layer types to include (Linear layers)
    include_patterns = [
        # Standard transformer projections
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        # MLP projections
        "gate_proj",
        "up_proj",
        "down_proj",
        # Generic FC layers
        "fc1",
        "fc2",
        # MoE expert layers (matched via .experts. in name)
        "experts",
        # Shared experts
        "shared_experts",
        # MLA (Multi-head Latent Attention) projections
        "q_a_proj",
        "q_b_proj",
        "kv_a_proj",
        "kv_b_proj",
        # Indexer attention layers
        "wq_b",
        "wk",
        # Linear attention layers
        "in_proj_qkv",
        "in_proj_z",
        "out_proj",
    ]

    # Check if matches include patterns
    for pattern in include_patterns:
        if pattern in param_lower:
            logger.debug(f"Will quantize FP8: {param_name}")
            return True

    # Do not quantize by default
    logger.debug(f"Skip quantization: {param_name}")
    return False


def scaled_fp8_blockwise(
    data_hp,
    weight_block_size,
):
    """Cast tensor from high precision to FP8 with blockwise quantization.

    When the DeepGEMM UE8M0 scale format is available and applicable, uses
    sglang's native ``quant_weight_ue8m0`` for kernel-compatible output.
    Otherwise uses a Triton kernel when available, falling back to PyTorch.

    Returns:
        (fp8_data, scale_inv) where scale_inv has shape (blk_m, blk_n, 1).
    """
    assert len(data_hp.shape) == 2, "Only 2d input tensor is supported"

    # Prefer DeepGEMM UE8M0 quantization when available and compatible
    if should_deepgemm_weight_requant_ue8m0 is not None and should_deepgemm_weight_requant_ue8m0(
        weight_block_size=weight_block_size
    ):
        qweight, scale = quant_weight_ue8m0(data_hp, weight_block_size=weight_block_size)
        scale = transform_scale_ue8m0(scale, mn=qweight.shape[-2])
        return qweight, scale.unsqueeze(-1)

    if _TRITON_AVAILABLE and data_hp.is_cuda:
        fp_data, scale_inv = blockwise_cast_to_fp8_triton(data_hp, block_size=weight_block_size)
        # Add trailing dim to match downstream squeeze(-1) expectation
        return fp_data, scale_inv.unsqueeze(-1)

    # Pure PyTorch fallback
    block_size0 = weight_block_size[0]
    block_size1 = weight_block_size[1]
    assert data_hp.shape[0] % block_size0 == 0, (
        f"data_hp.shape[0] {data_hp.shape[0]} must be a multiple of block_size0: {block_size0}."
    )
    assert data_hp.shape[1] % block_size1 == 0, (
        f"data_hp.shape[1] {data_hp.shape[1]} must be a multiple of block_size1: {block_size1}."
    )

    max_dtype = torch.finfo(torch.float8_e4m3fn).max
    original_shape = data_hp.shape
    blk_m, blk_n = data_hp.shape[0] // block_size0, data_hp.shape[1] // block_size1

    assert block_size0 == block_size1
    data_hp = data_hp.reshape(blk_m, block_size0, blk_n, block_size1)
    data_hp = data_hp.permute(0, 2, 1, 3)
    data_hp = data_hp.to(torch.float32).contiguous().flatten(start_dim=2)

    max_abs = torch.amax(torch.abs(data_hp), dim=-1, keepdim=True)
    scale_fp = max_dtype / max_abs
    scale_fp = torch.where(max_abs == 0, 1.0, scale_fp)
    scale_fp = torch.where(max_abs == torch.inf, 1.0, scale_fp)
    descale_fp = torch.reciprocal(scale_fp)

    data_lp = torch.clamp(data_hp * scale_fp, min=-1 * max_dtype, max=max_dtype)
    fp_data = data_lp.to(torch.float8_e4m3fn)
    fp_data = fp_data.reshape(blk_m, blk_n, block_size0, block_size1).permute(0, 2, 1, 3).reshape(original_shape)

    return fp_data, descale_fp


def fp8_quantize_weight_generator(
    weights: Iterable[tuple[str, torch.Tensor]],
    quant_config: dict,
    dtype: torch.dtype = torch.bfloat16,
):
    """Lazily quantize weights to FP8, yielding ``(name, tensor)`` pairs.

    For each quantizable weight the generator yields two pairs:
    ``(name, fp8_weight)`` and ``(name + "_scale_inv", scale)``.
    Non-quantizable parameters are yielded unchanged.

    This is the preferred entry-point for the sync-mixin path because it
    avoids materialising the full BF16 weight set — each tensor is quantised
    and the BF16 original can be freed immediately by the caller.
    """
    if isinstance(quant_config, dict):
        weight_block_size = quant_config.get("weight_block_size")
    else:
        weight_block_size = getattr(quant_config, "weight_block_size", None)

    if weight_block_size is None:
        raise ValueError("weight_block_size not found in quant_config")

    for k, v in weights:
        # Skip pre-existing scale tensors from upstream (e.g. --fp8-param-gather)
        if k.endswith("_scale") or k.endswith("_scale_inv"):
            continue

        if not should_quantize_param(k):
            yield (k, v)
            continue

        try:
            logger.debug(f"  Quantizing to FP8 blockwise: {k}")
            param_lp, param_scale = scaled_fp8_blockwise(
                v.to(dtype),
                weight_block_size=weight_block_size,
            )
            param_scale = param_scale.squeeze(-1)
            yield (k, param_lp)
            yield (k + "_scale_inv", param_scale)
        except Exception as e:
            logger.error(f"Failed to quantize {k}: {e}")
            yield (k, v)


def quant_weights_by_name(weights, quant_config, dtype=torch.bfloat16):
    """FP8 quantization based on parameter name (list version).

    Thin wrapper around :func:`fp8_quantize_weight_generator` that
    materialises results into a list for callers that need random access.
    """
    return list(fp8_quantize_weight_generator(weights, quant_config, dtype))
