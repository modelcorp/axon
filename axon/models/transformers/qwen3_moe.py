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
"""Qwen3-MoE FSDP support with fused expert kernels.

Replaces the HuggingFace ``Qwen3MoeSparseMoeBlock.forward`` with a version
that uses Triton-fused expert kernels for both forward and backward passes,
giving significant speedups during FSDP training of MoE models.

Also provides ``forward_with_triton_backend`` and ``forward_with_torch_backend``
for the fused lm_head (log-probs + entropy) computation, following the same
pattern as ``dense_common.py``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers.cache_utils import Cache

from axon.models.transformers.dense_common import CausalLMOutputForPPO, forward_base_model
from axon.utils.kernel.fused_experts import fused_experts_forward


class Qwen3MoeFusedSparseMoeBlock(nn.Module):
    """Drop-in replacement for ``Qwen3MoeSparseMoeBlock`` that uses fused Triton kernels."""

    def __init__(self, original_block: nn.Module):
        super().__init__()
        self.num_experts = original_block.gate.num_experts
        self.top_k = original_block.gate.top_k
        self.norm_topk_prob = original_block.gate.norm_topk_prob
        self.gate = original_block.gate
        self.experts = original_block.experts

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_2d = hidden_states.view(-1, hidden_dim)

        # In transformers 5.x, Qwen3MoeTopKRouter.forward returns
        # (router_logits, routing_weights, selected_experts) — routing is fully
        # handled by the gate module (softmax + topk + optional normalization).
        _router_logits, routing_weights, selected_experts = self.gate(hidden_states_2d)
        routing_weights = routing_weights.to(hidden_states_2d.dtype)

        # In transformers 5.x, Qwen3MoeExperts holds stacked parameters directly:
        #   gate_up_proj: (num_experts, 2*intermediate_size, hidden_size)
        #   down_proj:    (num_experts, hidden_size, intermediate_size)
        w1 = self.experts.gate_up_proj.contiguous()
        w2 = self.experts.down_proj.contiguous()

        final_hidden_states = fused_experts_forward(
            hidden_states_2d.to(torch.bfloat16),
            w1,
            w2,
            routing_weights,
            selected_experts,
        )

        return final_hidden_states.view(batch_size, sequence_length, hidden_dim)


def patch_qwen3_moe_with_fused_experts(model: nn.Module) -> None:
    """Replace all ``Qwen3MoeSparseMoeBlock`` instances in *model* with fused versions."""
    for name, module in model.named_modules():
        if type(module).__name__ == "Qwen3MoeSparseMoeBlock":
            _replace_module(model, name, Qwen3MoeFusedSparseMoeBlock(module))
    print(f"Patched {type(model).__name__} with fused MoE expert kernels")


def _replace_module(root: nn.Module, dotted_name: str, new_module: nn.Module) -> None:
    """Replace a nested sub-module identified by a dotted name."""
    parts = dotted_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


# ---------------------------------------------------------------------------
# Fused lm_head backends (same pattern as dense_common.py)
# ---------------------------------------------------------------------------


def forward_with_torch_backend(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_values: Cache | list[torch.FloatTensor] | None = None,
    inputs_embeds: torch.FloatTensor | None = None,
    labels: torch.LongTensor | None = None,
    use_cache: bool | None = None,
    output_attentions: bool | None = None,
    output_hidden_states: bool | None = None,
    return_dict: bool | None = None,
    cache_position: torch.LongTensor | None = None,
    logits_to_keep: int | torch.Tensor = 0,
    temperature: float = 1.0,
    **loss_kwargs,
) -> tuple | CausalLMOutputForPPO:
    from axon.utils.torch.fused_linear import FusedLinearForPPO

    outputs = forward_base_model(
        self,
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        cache_position=cache_position,
    )

    hidden_states = outputs[0]

    if not return_dict:
        raise NotImplementedError("forward_with_torch_backend has to return_dict")

    if labels is not None:
        rolled_labels = torch.roll(labels, shifts=-1, dims=-1)
    elif input_ids is not None:
        rolled_labels = torch.roll(input_ids, shifts=-1, dims=-1)
    else:
        raise RuntimeError("Either labels or input_ids must be provided.")

    fused_linear_for_ppo = FusedLinearForPPO()
    log_probs, entropy = fused_linear_for_ppo.forward(
        hidden_states=hidden_states,
        vocab_weights=self.lm_head.weight,
        input_ids=rolled_labels,
        temperature=temperature,
    )

    return CausalLMOutputForPPO(
        log_probs=log_probs,
        entropy=entropy,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )


def forward_with_triton_backend(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_values: Cache | list[torch.FloatTensor] | None = None,
    inputs_embeds: torch.FloatTensor | None = None,
    labels: torch.LongTensor | None = None,
    use_cache: bool | None = None,
    output_attentions: bool | None = None,
    output_hidden_states: bool | None = None,
    return_dict: bool | None = None,
    cache_position: torch.LongTensor | None = None,
    logits_to_keep: int | torch.Tensor = 0,
    temperature: float = 1.0,
    **loss_kwargs,
) -> tuple | CausalLMOutputForPPO:
    from axon.utils.kernel.linear_cross_entropy import linear_cross_entropy

    outputs = forward_base_model(
        self,
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
    )

    hidden_states = outputs[0]

    if not return_dict:
        raise NotImplementedError("forward_with_triton_backend has to return_dict")

    if labels is not None:
        rolled_labels = torch.roll(labels, shifts=-1, dims=-1)
    elif input_ids is not None:
        rolled_labels = torch.roll(input_ids, shifts=-1, dims=-1)
    else:
        raise RuntimeError("Either labels or input_ids must be provided.")

    log_probs, entropy = linear_cross_entropy(
        hidden_states,
        self.lm_head.weight,
        rolled_labels,
        temperature,
        "none",
    )

    return CausalLMOutputForPPO(
        log_probs=log_probs,
        entropy=entropy,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )
