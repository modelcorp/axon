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

"""TP-aware (logprob, entropy) with Gemma-style ``tanh(logits/cap)*cap`` softcap.

The fused ``linear_cross_entropy`` kernel computes logprobs from ``hidden @ W.T``
in a tiled fashion — it never materializes the full ``[tokens, vocab]`` tensor.
Gemma4 requires applying a tanh softcap to the raw logits before cross-entropy
(``final_logit_softcapping=30.0``); vLLM and HF both apply it, so skipping it in
Megatron's logprob recomputation causes systematic drift.

This module provides a drop-in replacement that **chunks along the token dim** to
keep peak memory at ``chunk_size × V/tp`` instead of ``N × V/tp``.

The log-normalizer is computed with fp32 precision using a straight-through
estimator: ``torch.logsumexp(logits.float())`` under ``no_grad`` provides the
precise fp32 value for the forward pass, while the bf16 computation graph is
kept intact for backward. This matches CUDA's ``torch.log_softmax`` which
accumulates in fp32 internally, closing a ~0.025 nat gap vs vLLM with zero
extra memory (the fp32 tensor is transient and not saved by autograd).
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F
from megatron.core.tensor_parallel.mappings import reduce_from_tensor_model_parallel_region

_DEFAULT_CHUNK_SIZE = 1024


def softcap_linear_cross_entropy(
    hidden_states: torch.Tensor,  # [N, H]
    weight: torch.Tensor,  # [V_local, H]  — TP-sharded along V
    labels: torch.Tensor,  # [N]  — global vocab indices
    temperature: float,
    softcap: float,
    tp_group: dist.ProcessGroup | None,
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-token ``(logprob_at_label, entropy)`` with a logit softcap.

    Chunks along the token dimension so peak memory is
    ``O(chunk_size × V_local)`` instead of ``O(N × V_local)``.
    """
    original_shape = hidden_states.shape
    if hidden_states.dim() != 2:
        hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
    if labels.dim() != 1:
        labels = labels.reshape(-1)

    N = hidden_states.shape[0]
    V_local = weight.shape[0]
    tp_size = dist.get_world_size(tp_group) if tp_group is not None else 1
    tp_rank = dist.get_rank(tp_group) if tp_group is not None else 0
    rank_start = tp_rank * V_local
    w = weight.to(hidden_states.dtype)

    all_logprobs = []
    all_entropy = []

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        h_chunk = hidden_states[start:end]  # [C, H]
        lab_chunk = labels[start:end]  # [C]

        # 1. Compute logits for this chunk, apply softcap + temperature.
        logits = F.linear(h_chunk, w)  # [C, V_local] bf16
        logits = torch.tanh(logits / softcap) * softcap
        if temperature != 1.0:
            logits = logits / temperature

        # 2. TP-distributed log-normalizer with fp32 precision.
        #
        # CUDA's torch.log_softmax uses fp32 accumulation internally; bf16
        # accumulation drifts ~0.025 nats over V/tp=32768 values. We match
        # fp32 precision via a straight-through estimator:
        #
        #   log_norm_bf16: computed normally in bf16, stays in autograd graph.
        #   log_norm_fp32: computed under no_grad from logits.float(),
        #                  freed immediately (not saved by autograd).
        #   log_norm = log_norm_bf16 + (log_norm_fp32 - log_norm_bf16).detach()
        #
        # Forward uses the fp32-precise value; backward flows through bf16.
        # Extra memory: only [C]-sized fp32 scalars, not [C, V_local].

        # bf16 path (for backward)
        local_max = logits.detach().max(dim=-1, keepdim=True).values
        if tp_size > 1:
            global_max = local_max.contiguous()
            dist.all_reduce(global_max, op=dist.ReduceOp.MAX, group=tp_group)
        else:
            global_max = local_max

        shifted = logits - global_max
        local_sumexp = torch.exp(shifted).sum(dim=-1, keepdim=True)
        if tp_size > 1:
            global_sumexp = reduce_from_tensor_model_parallel_region(local_sumexp)
        else:
            global_sumexp = local_sumexp
        log_norm_bf16 = torch.log(global_sumexp).squeeze(-1) + global_max.squeeze(-1)

        # fp32 path (for forward precision) — no autograd, no saved tensors
        with torch.no_grad():
            local_lse_fp32 = torch.logsumexp(logits.float(), dim=-1)  # [C] fp32
            if tp_size > 1:
                # Combine per-rank logsumexp: log(sum_r exp(lse_r))
                lse_max = local_lse_fp32.detach().clone()
                dist.all_reduce(lse_max, op=dist.ReduceOp.MAX, group=tp_group)
                local_exp = torch.exp(local_lse_fp32 - lse_max)
                dist.all_reduce(local_exp, op=dist.ReduceOp.SUM, group=tp_group)
                log_norm_fp32 = torch.log(local_exp) + lse_max
            else:
                log_norm_fp32 = local_lse_fp32

        # Straight-through: forward = fp32 value, backward = bf16 graph
        log_norm = log_norm_bf16 + (log_norm_fp32 - log_norm_bf16).detach()

        # 3. Logprob at label.
        local_labels = lab_chunk - rank_start
        in_range = (local_labels >= 0) & (local_labels < V_local)
        safe_labels = local_labels.clamp(min=0, max=V_local - 1)
        logit_at_label_local = logits.gather(1, safe_labels.unsqueeze(-1)).squeeze(-1)
        logit_at_label_local = torch.where(in_range, logit_at_label_local, logits.new_zeros(()))
        if tp_size > 1:
            logit_at_label = reduce_from_tensor_model_parallel_region(logit_at_label_local)
        else:
            logit_at_label = logit_at_label_local
        all_logprobs.append(logit_at_label - log_norm)

        # 4. Entropy in bf16 (precision not critical for entropy bonus).
        log_p = logits - log_norm.to(logits.dtype).unsqueeze(-1)
        local_ent = -(torch.exp(log_p) * log_p).sum(dim=-1)
        if tp_size > 1:
            ent = reduce_from_tensor_model_parallel_region(local_ent)
        else:
            ent = local_ent
        all_entropy.append(ent)

        del logits, shifted, log_p

    logprobs = torch.cat(all_logprobs, dim=0)
    entropy = torch.cat(all_entropy, dim=0)

    if len(original_shape) > 2:
        logprobs = logprobs.view(*original_shape[:-1])
        entropy = entropy.view(*original_shape[:-1])
    return logprobs, entropy
