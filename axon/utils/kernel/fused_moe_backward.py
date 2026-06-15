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
"""Triton backward kernels for fused Mixture-of-Experts (MoE) layers.

Provides backward-pass kernels for computing gradients of:
  - grad_input:        via ``fused_moe_backward_input_kernel``
  - grad_weight:       via ``fused_moe_backward_weight_kernel``
  - grad_topk_weights: via ``fused_moe_backward_topk_weights_kernel``

These kernels are launched by ``invoke_fused_moe_backward_kernel``.
"""

from __future__ import annotations

from typing import Any

import torch
import triton
import triton.language as tl


@triton.jit
def fused_moe_backward_input_kernel(
    # Pointers to matrices
    grad_output_ptr,
    weight_ptr,
    grad_input_ptr,
    grad_topk_weights_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    # Matrix dimensions
    N,
    K,
    EM,
    num_valid_tokens,
    # Strides
    stride_gom,
    stride_gon,
    stride_we,
    stride_wn,
    stride_wk,
    stride_gim,
    stride_gik,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
):
    """Backward kernel for grad_input.

    Forward:  output = input @ weight.T  (optionally * topk_weights)
    Backward: grad_input = grad_output @ weight  (optionally * topk_weights)
    """
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)

    if pid_m * BLOCK_SIZE_M < num_tokens_post_padded:
        offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
        offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
        offs_token = offs_token.to(tl.int64)
        token_mask = offs_token < num_valid_tokens

        off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)

        if off_experts != -1:
            offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
            offs_k = tl.arange(0, BLOCK_SIZE_K)

            grad_output_ptrs = grad_output_ptr + (offs_token[:, None] * stride_gom + offs_n[None, :] * stride_gon)
            grad_out = tl.load(
                grad_output_ptrs,
                mask=token_mask[:, None] & (offs_n[None, :] < N),
                other=0.0,
            )

            if MUL_ROUTED_WEIGHT:
                moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0)
                grad_out = grad_out * moe_weight[:, None]

            for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
                curr_offs_k = k * BLOCK_SIZE_K + offs_k

                weight_ptrs = (
                    weight_ptr
                    + off_experts * stride_we
                    + offs_n[:, None] * stride_wn
                    + curr_offs_k[None, :] * stride_wk
                )
                w = tl.load(
                    weight_ptrs,
                    mask=(offs_n[:, None] < N) & (curr_offs_k[None, :] < K),
                    other=0.0,
                )

                contribution = tl.dot(grad_out, w)

                grad_input_ptrs = grad_input_ptr + (
                    (offs_token[:, None] // top_k) * stride_gim + curr_offs_k[None, :] * stride_gik
                )
                grad_input_mask = token_mask[:, None] & (curr_offs_k[None, :] < K)
                tl.atomic_add(grad_input_ptrs, contribution.to(compute_type), mask=grad_input_mask)


@triton.jit
def fused_moe_backward_weight_kernel(
    # Pointers to matrices
    grad_output_ptr,
    input_ptr,
    grad_weight_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    # Matrix dimensions
    N,
    K,
    EM,
    num_valid_tokens,
    # Strides
    stride_gom,
    stride_gon,
    stride_im,
    stride_ik,
    stride_gwe,
    stride_gwn,
    stride_gwk,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
):
    """Backward kernel for grad_weight.

    Forward:  output = input @ weight.T  (optionally * topk_weights)
    Backward: grad_weight = input.T @ grad_output  (optionally * topk_weights)
    """
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)

    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return

    expert_id = tl.load(expert_ids_ptr + pid_m).to(tl.int64)

    if expert_id == -1:
        return

    offs_m = tl.arange(0, BLOCK_SIZE_M)
    offs_token_id = pid_m * BLOCK_SIZE_M + offs_m.to(tl.int64)
    offs_token = tl.load(
        sorted_token_ids_ptr + offs_token_id, mask=offs_token_id < num_tokens_post_padded, other=num_valid_tokens
    )
    offs_token = offs_token.to(tl.int64)
    token_mask = (offs_token_id < num_tokens_post_padded) & (offs_token < num_valid_tokens)

    offs_token_clamped = tl.where(token_mask, offs_token, 0)

    if MUL_ROUTED_WEIGHT:
        input_token_idx = offs_token_clamped
        input_mask = token_mask
    else:
        input_token_idx = offs_token_clamped // top_k
        num_input_tokens = num_valid_tokens // top_k
        input_mask = token_mask & (input_token_idx < num_input_tokens)

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token_clamped, mask=token_mask, other=0.0)

    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)

    grad_output_ptrs = grad_output_ptr + (offs_token_clamped[:, None] * stride_gom + offs_n[None, :] * stride_gon)
    grad_out = tl.load(
        grad_output_ptrs,
        mask=token_mask[:, None] & (offs_n[None, :] < N),
        other=0.0,
    )

    if MUL_ROUTED_WEIGHT:
        grad_out = grad_out * moe_weight[:, None]

    token_mask_col = token_mask[:, None]
    grad_out = grad_out * token_mask_col

    for k_block in range(tl.cdiv(K, BLOCK_SIZE_K)):
        offs_k = k_block * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K).to(tl.int64)

        input_ptrs = input_ptr + (input_token_idx[:, None] * stride_im + offs_k[None, :] * stride_ik)
        inp = tl.load(
            input_ptrs,
            mask=input_mask[:, None] & (offs_k[None, :] < K),
            other=0.0,
        )

        input_mask_col = input_mask[:, None]
        inp = inp * input_mask_col

        grad_w_contribution = tl.dot(grad_out.T, inp)

        grad_weight_ptrs = (
            grad_weight_ptr + expert_id * stride_gwe + offs_n[:, None] * stride_gwn + offs_k[None, :] * stride_gwk
        )
        grad_weight_mask = (offs_n[:, None] < N) & (offs_k[None, :] < K)
        tl.atomic_add(grad_weight_ptrs, grad_w_contribution.to(compute_type), mask=grad_weight_mask)


@triton.jit
def fused_moe_backward_topk_weights_kernel(
    # Pointers to matrices
    grad_output_ptr,
    input_ptr,
    weight_ptr,
    grad_topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    # Matrix dimensions
    N,
    K,
    EM,
    num_valid_tokens,
    # Strides
    stride_gom,
    stride_gon,
    stride_im,
    stride_ik,
    stride_we,
    stride_wn,
    stride_wk,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
):
    """Backward kernel for grad_topk_weights.

    Forward:  output = topk_weights * (input @ weight.T)
    Backward: grad_topk_weights = sum(grad_output * (input @ weight.T))
    """
    pid = tl.program_id(axis=0)

    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)

    if pid * BLOCK_SIZE_M < num_tokens_post_padded:
        offs_token_id = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
        offs_token = tl.load(
            sorted_token_ids_ptr + offs_token_id, mask=offs_token_id < num_tokens_post_padded, other=num_valid_tokens
        )
        offs_token = offs_token.to(tl.int64)
        token_mask = (offs_token_id < num_tokens_post_padded) & (offs_token < num_valid_tokens)

        offs_token_clamped = tl.where(token_mask, offs_token, 0)

        off_experts = tl.load(expert_ids_ptr + pid).to(tl.int64)

        if off_experts != -1:
            offs_n = tl.arange(0, BLOCK_SIZE_N)
            offs_k = tl.arange(0, BLOCK_SIZE_K)

            accumulator = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)

            for n in range(0, tl.cdiv(N, BLOCK_SIZE_N)):
                curr_offs_n = n * BLOCK_SIZE_N + offs_n

                grad_output_ptrs = grad_output_ptr + (
                    offs_token_clamped[:, None] * stride_gom + curr_offs_n[None, :] * stride_gon
                )
                grad_out = tl.load(
                    grad_output_ptrs,
                    mask=token_mask[:, None] & (curr_offs_n[None, :] < N),
                    other=0.0,
                )

                forward_output_n = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

                for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
                    curr_offs_k = k * BLOCK_SIZE_K + offs_k

                    input_ptrs = input_ptr + (
                        (offs_token_clamped[:, None] // top_k) * stride_im + curr_offs_k[None, :] * stride_ik
                    )
                    inp = tl.load(
                        input_ptrs,
                        mask=token_mask[:, None] & (curr_offs_k[None, :] < K),
                        other=0.0,
                    )

                    weight_ptrs = (
                        weight_ptr
                        + off_experts * stride_we
                        + curr_offs_n[:, None] * stride_wn
                        + curr_offs_k[None, :] * stride_wk
                    )
                    w = tl.load(
                        weight_ptrs,
                        mask=(curr_offs_n[:, None] < N) & (curr_offs_k[None, :] < K),
                        other=0.0,
                    )

                    forward_output_n += tl.dot(inp, w.T)

                accumulator += tl.sum(grad_out * forward_output_n, axis=1)

            tl.atomic_add(grad_topk_weights_ptr + offs_token_clamped, accumulator.to(compute_type), mask=token_mask)


def invoke_fused_moe_backward_kernel(
    grad_output: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    grad_input: torch.Tensor,
    grad_weight: torch.Tensor,
    grad_topk_weights: torch.Tensor | None,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    mul_routed_weight: bool,
    top_k: int,
    config: dict[str, Any],
    compute_type: tl.dtype,
) -> None:
    """Launch the fused MoE backward Triton kernels."""
    assert topk_weights.stride(1) == 1
    assert sorted_token_ids.stride(0) == 1

    if grad_output.ndim == 3:
        grad_output = grad_output.reshape(-1, grad_output.shape[-1])

    E, N, K = weight.shape

    # ---- grad_input ----
    def grid_input(META):
        return (triton.cdiv(sorted_token_ids.shape[0], META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),)

    fused_moe_backward_input_kernel[grid_input](
        grad_output,
        weight,
        grad_input,
        grad_topk_weights if grad_topk_weights is not None else grad_input,  # dummy pointer
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        N,
        K,
        sorted_token_ids.shape[0],
        grad_output.shape[0],
        grad_output.stride(0),
        grad_output.stride(1),
        weight.stride(0),
        weight.stride(1),
        weight.stride(2),
        grad_input.stride(0),
        grad_input.stride(1),
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        top_k=top_k,
        compute_type=compute_type,
        **config,
    )

    # ---- grad_weight ----
    grad_weight.zero_()

    def grid_weight(META):
        return (triton.cdiv(sorted_token_ids.shape[0], META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),)

    fused_moe_backward_weight_kernel[grid_weight](
        grad_output,
        input,
        grad_weight,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        N,
        K,
        sorted_token_ids.shape[0],
        grad_output.shape[0],
        grad_output.stride(0),
        grad_output.stride(1),
        input.stride(0),
        input.stride(1),
        grad_weight.stride(0),
        grad_weight.stride(1),
        grad_weight.stride(2),
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        top_k=top_k,
        compute_type=compute_type,
        **config,
    )

    # ---- grad_topk_weights (only when mul_routed_weight) ----
    if mul_routed_weight and grad_topk_weights is not None:

        def grid_topk(META):
            return (triton.cdiv(sorted_token_ids.shape[0], META["BLOCK_SIZE_M"]),)

        fused_moe_backward_topk_weights_kernel[grid_topk](
            grad_output,
            input,
            weight,
            grad_topk_weights.view(-1),
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            N,
            K,
            sorted_token_ids.shape[0],
            grad_output.shape[0],
            grad_output.stride(0),
            grad_output.stride(1),
            input.stride(0),
            input.stride(1),
            weight.stride(0),
            weight.stride(1),
            weight.stride(2),
            top_k=top_k,
            compute_type=compute_type,
            BLOCK_SIZE_M=config["BLOCK_SIZE_M"],
            BLOCK_SIZE_N=config["BLOCK_SIZE_N"],
            BLOCK_SIZE_K=config["BLOCK_SIZE_K"],
        )
