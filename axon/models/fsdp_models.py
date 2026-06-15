# Copyright 2025 Model AI Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Model-type classes that define model-specific behavior for TrainerWorker.

``CausalLM``    -- causal-LM actor (policy) behavior.
``ValueModel``  -- value-head critic behavior.
``RewardModel`` -- reward model (inference-only) behavior.

Each class exposes static methods that TrainerWorker delegates to:

* ``create_model(model_config, **kwargs)``
    — instantiate the model from an already-loaded ``AutoConfig``.
      ``TrainerWorker.init_model`` handles config parsing, ``AutoConfig`` loading,
      init-weight context, monkey patching, dtype casting, and gradient
      checkpointing.  ``create_model`` only needs to do model-specific config
      tweaks and call ``from_pretrained`` / ``load_valuehead_model``.
      Common kwargs: ``model_path``, ``torch_dtype``, ``trust_remote_code``.
* ``forward_fn(module, data, meta_info)``
* ``forward_keys(data)``
* ``forward_backward_keys(data)``
* ``forward_backward_fn(module, data, meta_info)``

Usage::

    actor  = TrainerWorker(config, model=CausalLM)
    critic = TrainerWorker(config, model=ValueModel, name="critic")
"""

import logging
import os

import numpy as np
import torch
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoModelForTokenClassification,
)

try:
    from transformers import AutoModelForVision2Seq
except ImportError:
    # Removed in transformers 5.x, replaced by AutoModelForImageTextToText
    AutoModelForVision2Seq = AutoModelForImageTextToText

from axon.trainer.algos.loss import agg_loss, compute_loss_fn
from axon.utils.fsdp.forward_utils import fsdp_model_forward, rmpad_postprocess, unpad_inputs
from axon.utils.hf_model import load_valuehead_model
from axon.utils.rl.kl import kl_penalty
from axon.utils.torch import get_device_name
from axon.utils.torch.attention import index_first_axis, pad_input, rearrange, unpad_input
from axon.utils.ulysses import gather_outputs_and_unpad, ulysses_pad_and_slice_inputs

# transformers 5.x removed AutoModelForVision2Seq; use AutoModelForImageTextToText
AutoModelForVision2Seq = AutoModelForImageTextToText

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("AXON_LOGGING_LEVEL", "WARN"))


class CausalLM:
    """Model-type class for causal-LM (actor / policy) models."""

    @staticmethod
    def create_model(model_config: AutoConfig, **kwargs):
        """Instantiate the causal-LM module.

        Called inside ``init_context`` by ``TrainerWorker.init_model``.
        ``model_config`` is already overridden (token IDs + user overrides).
        """
        model_path = kwargs["model_path"]
        torch_dtype = kwargs["torch_dtype"]
        trust_remote_code = kwargs["trust_remote_code"]
        attn_implementation = kwargs.get("attn_implementation", "flash_attention_2")

        has_remote_code = hasattr(model_config, "auto_map") and any(
            model_config.architectures[0] in val for val in model_config.auto_map.values()
        )
        if has_remote_code:
            auto_class = next(k for k, v in model_config.auto_map.items() if model_config.architectures[0] in v)
            match auto_class:
                case "AutoModelForVision2Seq":
                    module_class = AutoModelForVision2Seq
                case "AutoModelForCausalLM":
                    module_class = AutoModelForCausalLM
                case "AutoModelForImageTextToText":
                    module_class = AutoModelForImageTextToText
                case _:
                    module_class = AutoModel
        else:
            if type(model_config) in AutoModelForVision2Seq._model_mapping.keys():
                module_class = AutoModelForVision2Seq
            elif type(model_config) in AutoModelForCausalLM._model_mapping.keys():
                module_class = AutoModelForCausalLM
            elif type(model_config) in AutoModelForImageTextToText._model_mapping.keys():
                module_class = AutoModelForImageTextToText
            else:
                module_class = AutoModel

        return module_class.from_pretrained(
            pretrained_model_name_or_path=model_path,
            torch_dtype=torch_dtype,
            config=model_config,
            trust_remote_code=trust_remote_code,
            attn_implementation=attn_implementation,
        )

    _FWD_KWARGS = (
        "use_remove_padding",
        "use_fused_kernels",
        "ulysses_sp_size",
        "device_name",
        "param_dtype",
        "entropy_checkpointing",
        "entropy_from_logits_with_chunking",
        "use_torch_compile",
    )

    @staticmethod
    def forward_fn(module, data, meta_info):
        """Run a single micro-batch forward pass.

        Returns ``{"log_probs": ..., "entropys": ...}``.
        """
        temperature = meta_info.get("temperature", 1.0)
        calculate_entropy = meta_info.get("calculate_entropy", True)
        fwd_kwargs = {k: meta_info[k] for k in CausalLM._FWD_KWARGS if k in meta_info}

        entropy, log_probs = fsdp_model_forward(
            module,
            data,
            temperature=temperature,
            calculate_entropy=calculate_entropy,
            **fwd_kwargs,
        )
        return {"log_probs": log_probs, "entropys": entropy}

    @staticmethod
    def forward_keys(data):
        """Return (batch_keys, non_tensor_batch_keys) for forward key selection."""
        batch_keys = ["input_ids", "attention_mask", "position_ids"]
        if "response_mask" in data.batch:
            batch_keys.append("response_mask")
        non_tensor_keys = ["multi_modal_inputs"] if "multi_modal_inputs" in data.non_tensor_batch else []
        return batch_keys, non_tensor_keys

    @staticmethod
    def forward_backward_keys(data):
        """Return (batch_keys, non_tensor_batch_keys) for forward_backward key selection."""
        _optional_batch = (
            "num_program_tokens",
            "sampler_log_probs",
            "ref_log_prob",
            "valid_batch_size",
            "valid_token_count",
            "valid_program_count",
            "sampler_is_weights",
        )
        batch_keys = [
            "input_ids",
            "attention_mask",
            "position_ids",
            "response_mask",
            "old_log_probs",
            "advantages",
        ] + [k for k in _optional_batch if k in data.batch]
        non_tensor_keys = [k for k in ("multi_modal_inputs", "num_program_steps") if k in data.non_tensor_batch] or None
        return batch_keys, non_tensor_keys

    @staticmethod
    def forward_backward_fn(module, data, meta_info):
        """Per micro-batch forward + loss computation.

        ``data`` is a DataProto (micro-batch).

        Returns ``(scaled_loss, metrics_dict)``.
        """
        loss_fn = meta_info["loss_fn"]
        loss_fn_args = meta_info["loss_fn_args"]
        temperature = meta_info["temperature"]
        dp_replicas = meta_info["dp_replicas"]

        model_inputs = {**data.batch, **data.non_tensor_batch}
        response_mask = model_inputs["response_mask"]

        nps = model_inputs.get("num_program_steps", None)
        agg_kwargs = dict(
            loss_mask=response_mask,
            num_program_tokens=model_inputs.get("num_program_tokens", None),
            num_program_steps=torch.from_numpy(nps.astype(np.int64)).to(response_mask.device)
            if nps is not None
            else None,
            token_reduce=loss_fn_args.get("token_reduce", "sum"),
            batch_reduce=loss_fn_args.get("batch_reduce", "token-mean"),
            valid_token_count=model_inputs.get("valid_token_count", None),
            valid_batch_size=model_inputs.get("valid_batch_size", None),
            valid_program_count=model_inputs.get("valid_program_count", None),
        )

        result = CausalLM.forward_fn(
            module,
            model_inputs,
            {
                **meta_info,
                "temperature": temperature,
                "calculate_entropy": True,
            },
        )
        log_prob = result["log_probs"]
        entropy = result["entropys"]
        data.batch["log_probs"] = log_prob

        pg_loss, pg_metrics = compute_loss_fn(data, loss_fn=loss_fn, loss_fn_args=loss_fn_args)
        mb_metrics = dict(pg_metrics)

        # agg_loss with global valid_batch_size returns partial_sum/global_count per micro-batch.
        # Rescale logged metrics by n_micro_batches (same approach as megatron_models.py).
        n_micro_batches = meta_info.get("n_micro_batches", 1)
        mb_metrics["pg_loss"] = (pg_loss.detach() * n_micro_batches).item()
        loss = pg_loss

        sampler_log_prob = model_inputs.get("sampler_log_probs", None)
        if loss_fn != "bypass_mode" and sampler_log_prob is not None:
            from axon.utils.rl.sampler import compute_sampler_corr_metrics_from_logprobs

            mb_metrics.update(
                compute_sampler_corr_metrics_from_logprobs(
                    log_prob=log_prob,
                    sampler_log_prob=sampler_log_prob,
                    response_mask=response_mask,
                )
            )

        entropy_coef = loss_fn_args.get("entropy_coef", 0)
        if entropy_coef != 0:
            entropy_agg = agg_loss(loss_mat=entropy, **agg_kwargs)
            loss -= entropy_agg * entropy_coef
            mb_metrics["entropy"] = (entropy_agg.detach() * n_micro_batches).item()
        else:
            with torch.no_grad():
                mb_metrics["entropy"] = (agg_loss(loss_mat=entropy, **agg_kwargs) * n_micro_batches).item()
        with torch.no_grad():
            mb_metrics["entropy_token_mean"] = (
                agg_loss(loss_mat=entropy, loss_mask=response_mask, token_reduce="sum", batch_reduce="token-mean")
                .detach()
                .item()
            )

        kl_coef = loss_fn_args.get("kl_coef", 0)
        if kl_coef != 0:
            assert "ref_log_prob" in model_inputs, "ref_log_prob is required for KL loss"
            kld = kl_penalty(
                logprob=log_prob,
                ref_logprob=model_inputs["ref_log_prob"],
                kl_penalty_type=loss_fn_args.get("kl_type", "low_var_kl"),
            )
            kl_loss = agg_loss(loss_mat=kld, **agg_kwargs)
            loss += kl_loss * kl_coef
            mb_metrics["kl_loss"] = (kl_loss.detach() * n_micro_batches).item()
            mb_metrics["kl_coef"] = kl_coef

        # Compensate for framework gradient reduction. dp_replicas is set by
        # fsdp_workers based on backend: D for FSDP2 (AVG ÷D), 1 for FSDP1 (SUM).
        loss = loss * dp_replicas

        return loss, mb_metrics


class ValueModel:
    """Model-type class for critic (value-head) models."""

    @staticmethod
    def create_model(model_config, **kwargs):
        """Instantiate the value-head critic module.

        Called inside ``init_context`` by ``TrainerWorker.init_model``.
        """
        model_config.num_labels = 1
        model_config.classifier_dropout = 0.0
        model_config.hidden_dropout = 0.0
        model_config.summary_dropout_prob = 0.0

        return load_valuehead_model(
            kwargs["model_path"], kwargs["torch_dtype"], model_config, kwargs["trust_remote_code"]
        )

    @staticmethod
    def forward_fn(module, data, meta_info):
        """Run a single micro-batch through the critic model.

        Returns ``{"values": ...}``.
        """
        use_remove_padding = meta_info.get("use_remove_padding", False)
        ulysses_sp_size = meta_info.get("ulysses_sp_size", 1)
        device_name = get_device_name()

        multi_modal_inputs = {}
        if "multi_modal_inputs" in data.keys():
            from axon.utils.hf_model import extract_multi_modal_inputs

            multi_modal_inputs = extract_multi_modal_inputs(data["multi_modal_inputs"])

        with torch.autocast(device_type=device_name, dtype=torch.bfloat16):
            input_ids = data["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = data["attention_mask"]
            position_ids = data["position_ids"]
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)

            if use_remove_padding:
                input_ids_rmpad, position_ids_rmpad, indices, _ = unpad_inputs(input_ids, attention_mask, position_ids)

                pad_size = 0
                if ulysses_sp_size > 1:
                    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad, position_ids_rmpad, sp_size=ulysses_sp_size
                    )

                output = module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                )

                if hasattr(module, "v_head"):
                    values_rmpad = output[2].squeeze(0).unsqueeze(-1)
                else:
                    values_rmpad = output.logits
                    values_rmpad = values_rmpad.squeeze(0)

                values = rmpad_postprocess(
                    values_rmpad,
                    use_ulysses_sp=ulysses_sp_size > 1,
                    pad_size=pad_size,
                    is_mask_all_zero=attention_mask.sum() == 0,
                    indices=indices,
                    batch_size=batch_size,
                    seqlen=seqlen,
                )
            else:
                output = module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                )
                if hasattr(module, "v_head"):
                    values = output[2]
                else:
                    values = output.logits
                # Shift: model output at position t is V(state before seeing token t+1).
                values = torch.nn.functional.pad(values[:, :-1], (1, 0), value=0.0).squeeze(-1)
            if "response_mask" in data:
                values = values * data["response_mask"].to(values.device)
            return {"values": values}

    @staticmethod
    def forward_keys(data):
        """Return (batch_keys, non_tensor_batch_keys) for forward key selection."""
        batch_keys = ["input_ids", "attention_mask", "position_ids"]
        if "response_mask" in data.batch:
            batch_keys.append("response_mask")
        non_tensor_keys = ["multi_modal_inputs"] if "multi_modal_inputs" in data.non_tensor_batch else []
        return batch_keys, non_tensor_keys

    @staticmethod
    def forward_backward_keys(data):
        """Return (batch_keys, non_tensor_batch_keys) for forward_backward key selection."""
        batch_keys = ["input_ids", "response_mask", "attention_mask", "position_ids", "values", "returns"]
        non_tensor_keys = ["multi_modal_inputs"] if "multi_modal_inputs" in data.non_tensor_batch else None
        return batch_keys, non_tensor_keys

    @staticmethod
    def forward_backward_fn(module, data, meta_info):
        """Per micro-batch critic loss computation.

        ``data`` is a DataProto (micro-batch).

        Returns ``(scaled_loss, metrics_dict)``.
        """
        use_remove_padding = meta_info.get("use_remove_padding", False)
        ulysses_sp_size = meta_info.get("ulysses_sp_size", 1)
        loss_fn = meta_info["loss_fn"]
        loss_fn_args = meta_info["loss_fn_args"]

        model_inputs = {**data.batch, **data.non_tensor_batch}
        vpreds = ValueModel.forward_fn(
            module,
            model_inputs,
            {
                "use_remove_padding": use_remove_padding,
                "ulysses_sp_size": ulysses_sp_size,
            },
        )["values"]
        data.batch["vpreds"] = vpreds

        vf_loss, vf_metrics = compute_loss_fn(data, loss_fn=loss_fn, loss_fn_args=loss_fn_args)

        # Fix logged metric: agg_loss with global valid_batch_size returns partial_sum/global_count
        n_micro_batches = meta_info.get("n_micro_batches", 1)
        vf_metrics["vf_loss"] = (vf_loss.detach() * n_micro_batches).item()

        loss = vf_loss * meta_info["dp_replicas"]

        return loss, vf_metrics


class RewardModel:
    """Model-type class for reward models (AutoModelForTokenClassification)."""

    @staticmethod
    def create_model(model_config, **kwargs):
        """Instantiate the reward model module.

        Called inside ``init_context`` by ``TrainerWorker.init_model``.
        Always uses bfloat16 regardless of ``torch_dtype``.
        """
        model_config.num_labels = 1
        model_config.classifier_dropout = 0.0
        return AutoModelForTokenClassification.from_pretrained(  # nosec B615
            pretrained_model_name_or_path=kwargs["model_path"],
            config=model_config,
            torch_dtype=torch.bfloat16,
            trust_remote_code=kwargs["trust_remote_code"],
        )

    @staticmethod
    def forward_fn(module, data, meta_info):
        """Run a single micro-batch through the reward model.

        Returns ``{"rm_scores": ...}`` with shape ``(batch_size,)``.
        """
        use_remove_padding = meta_info.get("use_remove_padding", False)
        ulysses_sp_size = meta_info.get("ulysses_sp_size", 1)
        device_name = get_device_name()

        with torch.autocast(device_type=device_name, dtype=torch.bfloat16):
            input_ids = data["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = data["attention_mask"]
            position_ids = data["position_ids"]
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)

            if use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                if ulysses_sp_size > 1:
                    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad, position_ids_rmpad, sp_size=ulysses_sp_size
                    )

                output = module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    use_cache=False,
                )
                scores_rmpad = output.logits.squeeze(0)

                if ulysses_sp_size > 1:
                    scores_rmpad = gather_outputs_and_unpad(
                        scores_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size
                    )

                rm_score = pad_input(scores_rmpad, indices=indices, batch=batch_size, seqlen=seqlen).squeeze(-1)
            else:
                output = module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    use_cache=False,
                )
                rm_score = output.logits.squeeze(-1)

            # Extract score at the last valid token (EOS position)
            position_ids_orig = data["position_ids"]
            if position_ids_orig.dim() == 3:  # qwen2vl mrope
                position_ids_orig = position_ids_orig[:, 0, :]
            eos_mask_idx = torch.argmax(position_ids_orig * attention_mask, dim=-1)
            rm_score_per_sample = rm_score[torch.arange(batch_size), eos_mask_idx]

            # Expand to token-level scores aligned with the full sequence (no prompt prefix to skip).
            token_level_scores = torch.zeros_like(attention_mask, dtype=rm_score.dtype)
            token_level_scores[torch.arange(batch_size), eos_mask_idx] = rm_score_per_sample
            return {"rm_scores": token_level_scores}

    @staticmethod
    def forward_keys(data):
        """Return (batch_keys, non_tensor_batch_keys) for forward key selection."""
        batch_keys = ["input_ids", "attention_mask", "position_ids"]
        if "response_mask" in data.batch:
            batch_keys.append("response_mask")
        non_tensor_keys = ["multi_modal_inputs"] if "multi_modal_inputs" in data.non_tensor_batch else []
        return batch_keys, non_tensor_keys

    @staticmethod
    def forward_backward_keys(data):
        raise NotImplementedError("RewardModel is inference-only")

    @staticmethod
    def forward_backward_fn(module, data, meta_info):
        raise NotImplementedError("RewardModel is inference-only")
