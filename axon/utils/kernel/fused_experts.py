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
# Forward path builds on the vLLM fused-MoE kernel (github.com/vllm-project/vllm), Apache-2.0.
"""Fused MoE expert computation with Triton-accelerated forward and backward passes.

Provides differentiable ``torch.autograd.Function`` wrappers around the
vLLM fused-MoE forward kernel and custom Triton backward kernels, enabling
efficient MoE training under FSDP.

Components:
  - ``GateUpProjFunction``  – gate+up projection  (input @ w1.T)
  - ``SiluAndMulFunction``  – SiLU activation + element-wise multiply
  - ``DownProjFunction``    – down projection      (intermediate @ w2.T, weighted by topk)
  - ``MoeSumReduceFunction``– sum-reduce across top-k experts
  - ``fused_experts_forward``– chains the four stages into a single call
"""

from __future__ import annotations

import torch
import triton.language as tl
from vllm.model_executor.layers.fused_moe.fused_moe import (
    dispatch_fused_moe_kernel,
)
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)

from .fused_moe_backward import invoke_fused_moe_backward_kernel

# Default Triton block config for deterministic behaviour.
_DEFAULT_CONFIG = {
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 32,
    "GROUP_SIZE_M": 8,
}

# Process tokens in chunks to avoid https://github.com/vllm-project/vllm/issues/5938
_CHUNK_SIZE = 64 * 1024


def _silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    """SiLU-gated linear unit: silu(x1) * x2 where x = [x1, x2]."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.nn.functional.silu(x1) * x2


def _moe_sum_reduce(src: torch.Tensor, out: torch.Tensor) -> None:
    """Sum-reduce over the top-k dimension: out = src.sum(dim=1)."""
    torch.sum(src, dim=1, out=out)


class GateUpProjFunction(torch.autograd.Function):
    """Fused gate+up projection: output[token] = hidden_states[token] @ w1[expert].T"""

    @staticmethod
    def forward(
        ctx,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ):
        num_tokens, _ = hidden_states.shape
        E, N, _ = w1.shape
        config = _DEFAULT_CONFIG
        topk = topk_ids.shape[1]

        # vLLM kernel expects C to be 3D: (num_tokens, topk, N)
        intermediate_cache1_3d = torch.empty(
            (num_tokens, topk, N),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )

        for chunk in range((num_tokens // _CHUNK_SIZE) + 1):
            begin_idx = chunk * _CHUNK_SIZE
            end_idx = min((chunk + 1) * _CHUNK_SIZE, num_tokens)
            curr_hidden = hidden_states[begin_idx:end_idx]
            curr_out = intermediate_cache1_3d[begin_idx:end_idx]
            curr_topk_ids = topk_ids[begin_idx:end_idx]
            curr_topk_weights = topk_weights[begin_idx:end_idx]

            sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
                curr_topk_ids, config["BLOCK_SIZE_M"], E
            )

            dispatch_fused_moe_kernel(
                A=curr_hidden,
                B=w1,
                C=curr_out,
                A_scale=None,
                B_scale=None,
                B_zp=None,
                topk_weights=curr_topk_weights,
                sorted_token_ids=sorted_token_ids,
                expert_ids=expert_ids,
                num_tokens_post_padded=num_tokens_post_padded,
                mul_routed_weight=False,
                top_k=topk,
                config=config,
                compute_type=tl.bfloat16,
                use_fp8_w8a8=False,
                use_int8_w8a8=False,
                use_int8_w8a16=False,
                use_int4_w4a16=False,
                per_channel_quant=False,
                block_shape=None,
            )

        # Reshape to 2D for downstream SiluAndMul
        intermediate_cache1 = intermediate_cache1_3d.view(num_tokens * topk, N)

        ctx.save_for_backward(hidden_states, w1, topk_weights, topk_ids)
        ctx.config = config
        ctx.num_tokens = num_tokens
        ctx.topk = topk
        return intermediate_cache1

    @staticmethod
    def backward(ctx, grad_output):
        hidden_states, w1, topk_weights, topk_ids = ctx.saved_tensors
        config = ctx.config
        num_tokens = ctx.num_tokens
        topk = ctx.topk

        E, N, D_in = w1.shape

        grad_hidden_states = torch.zeros_like(hidden_states)
        grad_w1 = torch.zeros_like(w1)
        grad_topk_weights = torch.zeros_like(topk_weights)

        for chunk in range((num_tokens // _CHUNK_SIZE) + 1):
            begin_idx = chunk * _CHUNK_SIZE
            end_idx = min((chunk + 1) * _CHUNK_SIZE, num_tokens)
            curr_num = end_idx - begin_idx
            if curr_num == 0:
                continue

            curr_hidden = hidden_states[begin_idx:end_idx]
            curr_topk_ids = topk_ids[begin_idx:end_idx]
            curr_topk_weights = topk_weights[begin_idx:end_idx]
            curr_grad_out = grad_output[begin_idx * topk : end_idx * topk]

            sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
                curr_topk_ids, config["BLOCK_SIZE_M"], E
            )

            curr_grad_hidden = torch.zeros_like(curr_hidden)
            curr_grad_w1 = torch.zeros_like(w1)

            invoke_fused_moe_backward_kernel(
                grad_output=curr_grad_out,
                input=curr_hidden,
                weight=w1,
                grad_input=curr_grad_hidden,
                grad_weight=curr_grad_w1,
                grad_topk_weights=None,
                topk_weights=curr_topk_weights,
                topk_ids=curr_topk_ids,
                sorted_token_ids=sorted_token_ids,
                expert_ids=expert_ids,
                num_tokens_post_padded=num_tokens_post_padded,
                mul_routed_weight=False,
                top_k=topk,
                config=config,
                compute_type=tl.bfloat16,
            )

            grad_hidden_states[begin_idx:end_idx] += curr_grad_hidden
            grad_w1 += curr_grad_w1

        return grad_hidden_states, grad_w1, grad_topk_weights, None


class SiluAndMulFunction(torch.autograd.Function):
    """SiLU-gated linear unit with explicit backward."""

    @staticmethod
    def forward(ctx, intermediate_cache1: torch.Tensor):
        num_tokens, N = intermediate_cache1.shape
        intermediate_cache2 = _silu_and_mul(intermediate_cache1.view(-1, N))
        ctx.save_for_backward(intermediate_cache1)
        return intermediate_cache2

    @staticmethod
    def backward(ctx, grad_output):
        (intermediate_cache1,) = ctx.saved_tensors
        N = intermediate_cache1.shape[-1]
        x1, x2 = intermediate_cache1.view(-1, N).chunk(2, dim=-1)
        silu_x1 = torch.nn.functional.silu(x1)

        sig = torch.sigmoid(x1)
        dsilu_dx1 = sig + x1 * sig * (1 - sig)
        grad_x1 = grad_output * x2 * dsilu_dx1
        grad_x2 = grad_output * silu_x1
        grad_input = torch.cat([grad_x1, grad_x2], dim=-1)

        return grad_input.view_as(intermediate_cache1)


class DownProjFunction(torch.autograd.Function):
    """Down projection: output[token] = intermediate[token] @ w2[expert].T * topk_weight."""

    @staticmethod
    def forward(
        ctx,
        intermediate_cache2: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ):
        num_tokens, _ = intermediate_cache2.shape
        topk = topk_ids.shape[1]
        num_tokens //= topk
        E, _, _ = w2.shape
        config = _DEFAULT_CONFIG

        intermediate_cache3 = torch.empty(
            (num_tokens, topk, w2.shape[1]),
            device=intermediate_cache2.device,
            dtype=intermediate_cache2.dtype,
        )

        for chunk in range((num_tokens // _CHUNK_SIZE) + 1):
            begin_idx = chunk * _CHUNK_SIZE
            end_idx = min((chunk + 1) * _CHUNK_SIZE, num_tokens)
            curr_in = intermediate_cache2[begin_idx * topk : end_idx * topk]
            curr_out = intermediate_cache3[begin_idx:end_idx]
            curr_topk_ids = topk_ids[begin_idx:end_idx]
            curr_topk_weights = topk_weights[begin_idx:end_idx]

            sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
                curr_topk_ids, config["BLOCK_SIZE_M"], E
            )
            dispatch_fused_moe_kernel(
                A=curr_in,
                B=w2,
                C=curr_out,
                A_scale=None,
                B_scale=None,
                B_zp=None,
                topk_weights=curr_topk_weights,
                sorted_token_ids=sorted_token_ids,
                expert_ids=expert_ids,
                num_tokens_post_padded=num_tokens_post_padded,
                mul_routed_weight=True,
                top_k=1,
                config=config,
                compute_type=tl.bfloat16,
                use_fp8_w8a8=False,
                use_int8_w8a8=False,
                use_int8_w8a16=False,
                use_int4_w4a16=False,
                per_channel_quant=False,
                block_shape=None,
            )

        ctx.save_for_backward(intermediate_cache2, w2, topk_weights, topk_ids)
        ctx.config = config
        ctx.num_tokens = num_tokens
        ctx.topk = topk
        return intermediate_cache3

    @staticmethod
    def backward(ctx, grad_output):
        intermediate_cache2, w2, topk_weights, topk_ids = ctx.saved_tensors
        config = ctx.config
        num_tokens = ctx.num_tokens
        topk = ctx.topk

        E, hidden_size, intermediate_size = w2.shape

        grad_intermediate_cache2 = torch.zeros_like(intermediate_cache2)
        grad_w2 = torch.zeros_like(w2)
        grad_topk_weights = torch.zeros_like(topk_weights)

        for chunk in range((num_tokens // _CHUNK_SIZE) + 1):
            begin_idx = chunk * _CHUNK_SIZE
            end_idx = min((chunk + 1) * _CHUNK_SIZE, num_tokens)
            curr_num = end_idx - begin_idx
            if curr_num == 0:
                continue

            curr_in = intermediate_cache2[begin_idx * topk : end_idx * topk]
            curr_topk_ids = topk_ids[begin_idx:end_idx]
            curr_topk_weights = topk_weights[begin_idx:end_idx]
            curr_grad_out = grad_output[begin_idx:end_idx]

            sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
                curr_topk_ids, config["BLOCK_SIZE_M"], E
            )

            curr_grad_in = torch.zeros_like(curr_in)
            curr_grad_w2 = torch.zeros_like(w2)
            curr_grad_topk_weights = torch.zeros_like(curr_topk_weights)

            invoke_fused_moe_backward_kernel(
                grad_output=curr_grad_out,
                input=curr_in,
                weight=w2,
                grad_input=curr_grad_in,
                grad_weight=curr_grad_w2,
                grad_topk_weights=curr_grad_topk_weights,
                topk_weights=curr_topk_weights,
                topk_ids=curr_topk_ids,
                sorted_token_ids=sorted_token_ids,
                expert_ids=expert_ids,
                num_tokens_post_padded=num_tokens_post_padded,
                mul_routed_weight=True,
                top_k=1,
                config=config,
                compute_type=tl.bfloat16,
            )

            grad_intermediate_cache2[begin_idx * topk : end_idx * topk] = curr_grad_in
            grad_w2 += curr_grad_w2
            grad_topk_weights[begin_idx:end_idx] = curr_grad_topk_weights

        return grad_intermediate_cache2, grad_w2, grad_topk_weights, None


class MoeSumReduceFunction(torch.autograd.Function):
    """Sum-reduce over the top-k expert dimension."""

    @staticmethod
    def forward(ctx, intermediate_cache3: torch.Tensor, hidden_states_shape):
        out = torch.empty(hidden_states_shape, device=intermediate_cache3.device, dtype=intermediate_cache3.dtype)
        _moe_sum_reduce(intermediate_cache3, out)
        ctx.save_for_backward(intermediate_cache3)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (intermediate_cache3,) = ctx.saved_tensors
        return grad_output.unsqueeze(1).expand_as(intermediate_cache3), None


def fused_experts_forward(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
) -> torch.Tensor:
    """Run the full fused MoE expert computation with Triton-accelerated forward + backward.

    Args:
        hidden_states: (num_tokens, hidden_size) – flattened input.
        w1: (num_experts, 2*intermediate_size, hidden_size) – gate+up weights.
        w2: (num_experts, hidden_size, intermediate_size) – down weights.
        topk_weights: (num_tokens, top_k) – routing probabilities.
        topk_ids: (num_tokens, top_k) – selected expert indices.

    Returns:
        Tensor of shape (num_tokens, hidden_size).
    """
    assert hidden_states.shape[1] == w1.shape[2], "Hidden size mismatch"
    assert topk_weights.shape == topk_ids.shape, "topk shape mismatch"
    assert hidden_states.is_contiguous(), "hidden_states must be contiguous"
    assert w1.is_contiguous(), "w1 must be contiguous"
    assert w2.is_contiguous(), "w2 must be contiguous"
    assert hidden_states.dtype in (torch.bfloat16,), f"Unsupported dtype {hidden_states.dtype}; expected bfloat16"

    intermediate_cache1 = GateUpProjFunction.apply(hidden_states, w1, topk_weights, topk_ids)
    intermediate_cache2 = SiluAndMulFunction.apply(intermediate_cache1)
    intermediate_cache3 = DownProjFunction.apply(intermediate_cache2, w2, topk_weights, topk_ids)
    return MoeSumReduceFunction.apply(intermediate_cache3, hidden_states.shape)
