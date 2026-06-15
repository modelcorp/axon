# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

from collections import OrderedDict

import megatron.core as mcore
import torch
from megatron.core import parallel_state
from megatron.core.config_logger import has_config_logger_enabled, log_config_to_disk
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region
from megatron.core.utils import deprecate_inference_params
from torch import Tensor

from axon.models.mcore.forward.softcap_cross_entropy import softcap_linear_cross_entropy
from axon.models.mcore.forward.util import postprocess_packed_seqs_for_dict_output, preprocess_packed_seqs
from axon.utils.hf_model import CausalLMOutputForPPO
from axon.utils.kernel.linear_cross_entropy import linear_cross_entropy
from axon.utils.megatron.utils import unwrap_model


def _get_final_logit_softcapping(model: torch.nn.Module) -> float | None:
    """Return the model's ``final_logit_softcapping`` if set, else None.

    The attribute is installed by bridges for models that require a Gemma-style
    ``tanh(logits/cap)*cap`` on the output logits (e.g. Gemma4 at 30.0). The
    triton fused cross-entropy kernel has no hook for this, so forward paths
    that go through ``linear_cross_entropy`` skip the cap — producing a
    systematic drift vs. vLLM/HF, which both apply it. We detect the cap here
    and route to a softcap-aware replacement.
    """
    for m in (model, unwrap_model(model)):
        cap = getattr(m, "_final_logit_softcapping", None)
        if cap is not None:
            return float(cap)
    return None


def _get_patching_model(model: torch.nn.Module):
    model = unwrap_model(model)
    if isinstance(model, GPTModel):
        return model

    if not (hasattr(model, "language_model") and isinstance(model.language_model, GPTModel)):
        print(f"Model {model.__class__.__name__} is not a supported for fused forward")
        return None

    return model.language_model


def patch_fused_forward(model: torch.nn.Module):
    from packaging.version import Version

    assert Version(mcore.__version__) >= Version("0.13.0"), "Fused forward patching requires mecore >= 0.13.0"
    model = _get_patching_model(model)
    if model is not None:
        model.forward_backup = model.forward
        model.forward = _fused_GPTModel_forward.__get__(model, model.__class__)


def unpatch_fused_forward(model: torch.nn.Module):
    model = _get_patching_model(model)
    if model is not None:
        model.forward = model.forward_backup


def fused_forward_model_gen(vision_model: bool = False):
    def fused_forward_model(
        model,
        input_ids: Tensor,
        position_ids: Tensor,
        attention_mask: Tensor,
        labels: Tensor,
        labels_mask: Tensor,
        temperature: float,
        multi_modal_inputs: dict,
        skip_cp_gather: bool = False,
    ):
        pre_process: bool = (
            unwrap_model(model).pre_process if not vision_model else False
        )  # vision model does not need pre_process, because we pack the input_ids to thd in the forward function
        post_process: bool = unwrap_model(model).post_process

        model_kwargs = {}
        if "pixel_values" in multi_modal_inputs:
            model_kwargs["pixel_values"] = multi_modal_inputs["pixel_values"].to(input_ids.device)
        if "image_grid_thw" in multi_modal_inputs:
            model_kwargs["image_grid_thw"] = multi_modal_inputs["image_grid_thw"].to(input_ids.device)
        if "pixel_values_videos" in multi_modal_inputs:
            model_kwargs["pixel_values_videos"] = multi_modal_inputs["pixel_values_videos"].to(input_ids.device)
        if "video_grid_thw" in multi_modal_inputs:
            model_kwargs["video_grid_thw"] = multi_modal_inputs["video_grid_thw"].to(input_ids.device)

        batch_size, seq_len = attention_mask.shape[:2]
        fp8 = unwrap_model(model).config.fp8
        use_fp8_padding = fp8 in ["e4m3", "hybrid"]
        input_ids_rmpad, packed_seq_params = preprocess_packed_seqs(
            input_ids,
            attention_mask,
            pre_process=pre_process,
            use_fp8_padding=use_fp8_padding,
        )
        input_ids_rmpad = input_ids_rmpad.contiguous()
        labels_rmpad, _ = preprocess_packed_seqs(
            labels, attention_mask, pre_process=True, use_fp8_padding=use_fp8_padding
        )
        labels_mask_rmpad, _ = preprocess_packed_seqs(
            labels_mask, attention_mask, pre_process=True, use_fp8_padding=use_fp8_padding
        )
        labels_rmpad = labels_rmpad.contiguous()
        labels_mask_rmpad = labels_mask_rmpad.contiguous()

        input_args = dict(
            input_ids=input_ids_rmpad,
            attention_mask=None,
            position_ids=position_ids if not vision_model else None,  # vision models will calculate position_ids
            packed_seq_params=packed_seq_params,
            labels=labels_rmpad,
            temperature=temperature,
            **model_kwargs,
        )

        if vision_model:
            # workaround for supporting sequence packing with context parallelism
            # cp split with sequence packing will make model lose vision token information, so we need to keep
            # the original input_ids and pack them after vision embedding is calculated,
            # cooporate with mbridge
            input_args["input_ids"] = input_ids
            input_args["attention_mask"] = attention_mask

        output_orig: CausalLMOutputForPPO = model(**input_args)

        if post_process:
            # output_orig is in type of CausalLMOutputForPPO
            output = postprocess_packed_seqs_for_dict_output(
                labels_mask_rmpad,
                output_orig,
                packed_seq_params,
                attention_mask,
                batch_size,
                seq_len,
                post_process=post_process,
                skip_cp_gather=skip_cp_gather,
            )
        else:
            output = output_orig
        return output

    return fused_forward_model


def _fused_GPTModel_forward(
    model,
    input_ids: Tensor,
    position_ids: Tensor,
    attention_mask: Tensor,
    decoder_input: Tensor = None,
    labels: Tensor = None,
    inference_context: BaseInferenceContext = None,
    packed_seq_params: PackedSeqParams = None,
    extra_block_kwargs: dict = None,
    runtime_gather_output: bool | None = None,
    *,
    inference_params: BaseInferenceContext | None = None,
    loss_mask: Tensor | None = None,
    temperature: float = 1.0,
    **kwargs,
) -> CausalLMOutputForPPO:
    """
    Patch self._postprocess in forward for GPT models to enable fused kernel support.
    https://github.com/NVIDIA/Megatron-LM/blob/core_v0.13.0/megatron/core/models/gpt/gpt_model.py

    This patch keeps temperature explicit when calling ``self._postprocess``.
    Upstream Megatron does not currently expose that through the public call
    surface used here.
    """

    inference_context = deprecate_inference_params(inference_context, inference_params)

    preproc_output = model._preprocess(
        input_ids=input_ids,
        position_ids=position_ids,
        decoder_input=decoder_input,
        inference_context=inference_context,
        packed_seq_params=packed_seq_params,
    )

    (decoder_input, rotary_pos_emb, rotary_pos_cos, rotary_pos_sin, sequence_len_offset) = preproc_output[:5]

    # Run decoder.
    hidden_states = model.decoder(
        hidden_states=decoder_input,
        attention_mask=attention_mask,
        inference_context=inference_context,
        rotary_pos_emb=rotary_pos_emb,
        rotary_pos_cos=rotary_pos_cos,
        rotary_pos_sin=rotary_pos_sin,
        packed_seq_params=packed_seq_params,
        sequence_len_offset=sequence_len_offset,
        **(extra_block_kwargs or {}),
        **kwargs,
    )

    # Process inference output.
    if inference_context and not inference_context.is_static_batching():
        hidden_states = inference_context.last_token_logits(hidden_states.squeeze(1).unsqueeze(0)).unsqueeze(1)

    if model.mtp_process:
        hidden_states = model.mtp(
            input_ids=input_ids,
            position_ids=position_ids,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            inference_params=inference_context,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            packed_seq_params=packed_seq_params,
            sequence_len_offset=sequence_len_offset,
            embedding=model.embedding,
            **(extra_block_kwargs or {}),
        )

    if not model.post_process:
        return hidden_states

    # Process MTP loss if enabled
    if model.mtp_process and labels is not None:
        mtp_labels = labels.clone()
        hidden_states_list = torch.chunk(hidden_states, 1 + model.config.mtp_num_layers, dim=0)
        hidden_states = hidden_states_list[0]

        if loss_mask is None:
            # if loss_mask is not provided, use all ones as loss_mask
            loss_mask = torch.ones_like(mtp_labels)

        for mtp_layer_number in range(model.config.mtp_num_layers):
            # Output logits for current MTP layer
            if model.config.sequence_parallel:
                mtp_hidden_states = gather_from_sequence_parallel_region(hidden_states_list[mtp_layer_number + 1])
            else:
                mtp_hidden_states = hidden_states_list[mtp_layer_number + 1]

            # Force FP32 for MTP hidden states to reduce numerical precision differences
            # between vLLM (which uses FP32 for logits) and Megatron
            mtp_hidden_states = mtp_hidden_states.to(torch.float32)

            from megatron.core.transformer.multi_token_prediction import (
                MTPLossAutoScaler,
                MTPLossLoggingHelper,
                roll_tensor,
            )

            # Roll labels and loss_mask for next prediction
            mtp_labels, _ = roll_tensor(
                mtp_labels,
                shifts=-1,
                dims=-1,
                cp_group=model.cp_group,
                packed_seq_params=packed_seq_params,
            )

            mtp_logprobs, _ = linear_cross_entropy(
                mtp_hidden_states,
                model.output_layer.weight.to(mtp_hidden_states.dtype),
                mtp_labels,
                temperature,
                "none",
                parallel_state.get_tensor_model_parallel_group(),
            )
            loss_mask, num_tokens = roll_tensor(
                loss_mask, shifts=-1, dims=-1, cp_group=model.cp_group, packed_seq_params=packed_seq_params
            )

            # Compute MTP loss: convert logprobs to loss
            mtp_loss = -mtp_logprobs
            mtp_loss = loss_mask * mtp_loss

            if model.training:
                # Keep Megatron's current MTP loss logging path until loss logging
                # moves into the loss function.
                MTPLossLoggingHelper.save_loss_to_tracker(
                    torch.sum(mtp_loss) / num_tokens,
                    mtp_layer_number,
                    model.config.mtp_num_layers,
                    avg_group=parallel_state.get_data_parallel_group(with_context_parallel=True),
                )
            mtp_loss_scale = model.config.mtp_loss_scaling_factor / model.config.mtp_num_layers
            if model.config.calculate_per_token_loss:
                hidden_states = MTPLossAutoScaler.apply(hidden_states, mtp_loss_scale * mtp_loss)
            else:
                hidden_states = MTPLossAutoScaler.apply(hidden_states, mtp_loss_scale * mtp_loss / num_tokens)

    if model.config.sequence_parallel:
        hidden_states = gather_from_sequence_parallel_region(hidden_states)

    output = CausalLMOutputForPPO(
        loss=None,
        logits=None,
        past_key_values=None,
        hidden_states=hidden_states,
        attentions=None,
    )

    # Get the output weight - use embedding weight if output_layer is None or weight is shared
    if hasattr(model, "output_layer") and model.output_layer is not None and model.output_layer.weight is not None:
        output_weight = model.output_layer.weight
    elif model.share_embeddings_and_output_weights:
        output_weight = model.shared_embedding_or_output_weight()
    else:
        # When embeddings are tied, use the embedding weight
        output_weight = model.embedding.word_embeddings.weight

    _softcap = _get_final_logit_softcapping(model)
    if _softcap is not None:
        # Softcap-aware manual path (needed for Gemma4). Trades the fused
        # kernel for correctness: logits are materialized TP-sharded, softcap
        # applied elementwise, then TP-distributed log-softmax + label-gather.
        logprobs, entropy = softcap_linear_cross_entropy(
            hidden_states,
            output_weight,
            labels,
            temperature,
            _softcap,
            parallel_state.get_tensor_model_parallel_group(),
        )
    else:
        logprobs, entropy = linear_cross_entropy(
            hidden_states,
            output_weight,
            labels,
            temperature,
            "none",
            parallel_state.get_tensor_model_parallel_group(),
        )

    if has_config_logger_enabled(model.config):
        payload = OrderedDict(
            {
                "input_ids": input_ids,
                "position_ids": position_ids,
                "attention_mask": attention_mask,
                "decoder_input": decoder_input,
                "logprobs": logprobs,
                "entropy": entropy,
            }
        )
        log_config_to_disk(model.config, payload, prefix="input_and_logits")

    output.entropy = entropy
    output.log_probs = logprobs

    return output
