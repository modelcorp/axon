# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

from dataclasses import dataclass
from typing import Union

import torch
from transformers.cache_utils import Cache
from transformers.modeling_outputs import CausalLMOutputWithPast


@dataclass
class CausalLMOutputForPPO(CausalLMOutputWithPast):
    log_probs: torch.FloatTensor | None = None
    entropy: torch.FloatTensor | None = None


def forward_base_model(
    self,
    input_ids: torch.LongTensor | None = None,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_values: Cache | None = None,
    inputs_embeds: torch.FloatTensor | None = None,
    use_cache: bool | None = None,
    output_attentions: bool | None = None,
    output_hidden_states: bool | None = None,
    return_dict: bool | None = None,
    cache_position: torch.LongTensor | None = None,
) -> CausalLMOutputWithPast:
    r"""
    Copy paste LLaMa's forward
    https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/transformers/model/llama.py

    This function should be generic enough for all pure text models.
    ```"""

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )

    # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
    outputs = self.model(
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

    return outputs


def forward_with_torch_backend(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_values: Union["Cache", list[torch.FloatTensor]] | None = None,
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

    # Loss calculations.
    # ``pre_rolled_labels`` is the rolled-then-sliced label tensor prepared by
    # the caller — required under Ulysses SP because ``torch.roll`` on the
    # per-rank slice wraps to that rank's own first token instead of the next
    # rank's first token, producing wrong labels at slice boundaries.
    pre_rolled_labels = loss_kwargs.pop("pre_rolled_labels", None)
    if pre_rolled_labels is not None:
        rolled_labels = pre_rolled_labels
    elif labels is not None:
        rolled_labels = torch.roll(labels, shifts=-1, dims=-1)
    elif input_ids is not None:
        rolled_labels = torch.roll(input_ids, shifts=-1, dims=-1)
    else:
        raise RuntimeError("To use forward_with_torch_backend, either labels or input_ids must be provided.")

    # Models with final_logit_softcapping (e.g. Gemma4: tanh(logits/30)*30)
    softcap = (
        getattr(self.config.get_text_config(), "final_logit_softcapping", None)
        if hasattr(self.config, "get_text_config")
        else getattr(self.config, "final_logit_softcapping", None)
    )

    fused_linear_for_ppo = FusedLinearForPPO()
    log_probs, entropy = fused_linear_for_ppo.forward(
        hidden_states=hidden_states,
        vocab_weights=self.lm_head.weight,
        input_ids=rolled_labels,
        temperature=temperature,
        logit_softcapping=softcap,
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
    past_key_values: Union["Cache", list[torch.FloatTensor]] | None = None,
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

    # Loss calculations.
    # ``pre_rolled_labels`` is the rolled-then-sliced label tensor prepared by
    # the caller — required under Ulysses SP because ``torch.roll`` on the
    # per-rank slice wraps to that rank's own first token instead of the next
    # rank's first token, producing wrong labels at slice boundaries.
    pre_rolled_labels = loss_kwargs.pop("pre_rolled_labels", None)
    if pre_rolled_labels is not None:
        rolled_labels = pre_rolled_labels
    elif labels is not None:
        rolled_labels = torch.roll(labels, shifts=-1, dims=-1)
    elif input_ids is not None:
        rolled_labels = torch.roll(input_ids, shifts=-1, dims=-1)
    else:
        raise RuntimeError("To use forward_with_triton_backend, either labels or input_ids must be provided.")

    # Models with final_logit_softcapping (e.g. Gemma4) can't use the triton
    # linear_cross_entropy kernel (no softcap support).  Fall back to the
    # chunked torch path which does support it.
    softcap = (
        getattr(self.config.get_text_config(), "final_logit_softcapping", None)
        if hasattr(self.config, "get_text_config")
        else getattr(self.config, "final_logit_softcapping", None)
    )
    if softcap is not None:
        from axon.utils.torch.fused_linear import FusedLinearForPPO

        fused_linear_for_ppo = FusedLinearForPPO()
        log_probs, entropy = fused_linear_for_ppo.forward(
            hidden_states=hidden_states,
            vocab_weights=self.lm_head.weight,
            input_ids=rolled_labels,
            temperature=temperature,
            logit_softcapping=softcap,
        )
    else:
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
