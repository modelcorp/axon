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
"""Model-type classes that define model-specific behavior for Megatron TrainerWorker.

``CausalLM``    -- causal-LM actor (policy) behavior.
``ValueModel``  -- value-head critic behavior.
``RewardModel`` -- reward model (inference-only) behavior.

Each class exposes static methods that are passed to ``megatron_forward_backward``:

* ``forward_step(batch_iter, model, **kwargs)``
    — per micro-batch forward computation.
      Returns ``(output, partial(loss_func, ...))``.
* ``loss_func(output, **kwargs)``
    — loss computation given model output.
      Returns ``(loss, metrics)``.

Usage::

    actor  = TrainerWorker(config, model=CausalLM)
    critic = TrainerWorker(config, model=ValueModel, name="critic")
"""

import logging
import os
from functools import partial

import torch
from megatron.core import parallel_state as mpu

from axon.protocol import DataProto
from axon.trainer.algos.loss import agg_loss, compute_loss_fn
from axon.utils.megatron.tensor_parallel import vocab_parallel_entropy, vocab_parallel_log_probs_from_logits
from axon.utils.print_utils import append_to_dict
from axon.utils.rl.kl import kl_penalty
from axon.utils.rl.sampler import compute_sampler_corr_metrics_from_logprobs
from axon.utils.seqlen_balancing import get_reverse_idx
from axon.utils.torch import get_device_id

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("AXON_LOGGING_LEVEL", "WARN"))


def _resolve_vpp_rank(model, vpp_rank):
    """Resolve VPP rank by unwrapping DDP -> Float16Module -> GPTModel."""
    if vpp_rank is not None:
        return vpp_rank
    for obj in (
        getattr(getattr(model, "module", None), "module", None),
        getattr(model, "module", None),
        model,
    ):
        if obj is not None and (rank := getattr(obj, "vp_stage", None)) is not None:
            return rank
    return None


class CausalLM:
    """Model-type class for causal-LM (actor / policy) models."""

    @staticmethod
    def create_model(config, **kwargs):
        """Create a Megatron CausalLM module.

        Called by ``TrainerWorker._build_model_optimizer``.
        ``config`` is the worker config.

        Required kwargs: ``share_embeddings_and_output_weights``,
        ``tf_config``, ``hf_config``, ``bridge``, ``provider``.
        Optional kwargs: ``override_ddp_config``, ``peft_cls``.
        """
        from axon.utils.megatron.utils import McoreModuleWrapperConfig, make_megatron_module

        wrap_with_ddp = not config.get("forward_only", False)
        wrap_config = McoreModuleWrapperConfig(
            is_value_model=False,
            share_embeddings_and_output_weights=kwargs["share_embeddings_and_output_weights"],
            wrap_with_ddp=wrap_with_ddp,
            use_distributed_optimizer=config.megatron.use_distributed_optimizer,
        )
        peft_config = config.get("lora", None) if wrap_with_ddp else None
        return make_megatron_module(
            wrap_config=wrap_config,
            tf_config=kwargs["tf_config"],
            hf_config=kwargs["hf_config"],
            bridge=kwargs["bridge"],
            provider=kwargs["provider"],
            override_ddp_config=kwargs.get("override_ddp_config"),
            peft_cls=kwargs.get("peft_cls"),
            peft_config=peft_config,
            freeze_moe_router=config.megatron.get("freeze_moe_router", False),
        )

    @staticmethod
    def forward_output_fn(output, data, **kwargs):
        """Extract log_probs (and optionally entropy) from pipeline output on the last PP stage.

        Called by TrainerWorker.forward after megatron_forward_backward.
        ``output`` is the dict returned by megatron_forward_backward.
        Returns dict with ``log_probs`` and optionally ``entropys``.

        kwargs:
            use_dynamic_bsz: Whether dynamic batch sizing was used.
            calculate_entropy: Must be passed explicitly so that ALL pipeline
                stages participate in the entropy broadcast (NCCL collective).
        """
        use_dynamic_bsz = kwargs.get("use_dynamic_bsz", False)
        calculate_entropy = kwargs.get("calculate_entropy", False)
        import itertools

        seq_length = data.batch["input_ids"].size(1)
        batch_size = data.batch["input_ids"].size(0)
        device = data.batch["input_ids"].device

        is_last_stage = mpu.is_pipeline_last_stage(ignore_virtual=True)
        raw_outputs = output["output"]

        def _gather_and_broadcast(values):
            if is_last_stage:
                tensor = torch.cat(values, dim=0).to(torch.float32)
                if use_dynamic_bsz:
                    flat_indices = list(itertools.chain.from_iterable(output["indices"]))
                    assert len(flat_indices) == tensor.size(0), f"{len(flat_indices)} vs. {tensor.size()}"
                    tensor = tensor[torch.tensor(get_reverse_idx(flat_indices), dtype=torch.long)]
            else:
                tensor = torch.empty(batch_size, seq_length, dtype=torch.float32, device=device)
            tensor = tensor.to(get_device_id())
            from torch import distributed as dist

            dist.broadcast(
                tensor=tensor,
                src=mpu.get_pipeline_model_parallel_last_rank(),
                group=mpu.get_pipeline_model_parallel_group(),
                async_op=False,
            )
            return tensor.to("cpu")

        log_probs = _gather_and_broadcast(
            [(o[0] if calculate_entropy else o)["log_probs"] for o in raw_outputs] if is_last_stage else None
        )
        result = {"log_probs": log_probs}
        if calculate_entropy:
            entropys = _gather_and_broadcast([o[1] for o in raw_outputs] if is_last_stage else None)
            result["entropys"] = entropys

        # When skip_cp_gather is active, each CP rank's log_probs have zeros at
        # non-local positions.  The driver collects from CP rank 0 only
        # (_is_collect_rank checks cp_rank==0), so we must reconstruct the full
        # log_probs before returning.  all_reduce(SUM) works because each
        # position has a real value on exactly one rank and zero on all others.
        cp_size = mpu.get_context_parallel_world_size()
        if cp_size > 1:
            from torch import distributed as dist

            for key in result:
                t = result[key].to(get_device_id())
                dist.all_reduce(t, group=mpu.get_context_parallel_group())
                result[key] = t.to("cpu")

        return result

    @staticmethod
    def forward_step(batch_iter, model, **kwargs):
        """Per micro-batch forward computation for causal LM.

        Returns ``(output, partial(loss_func, ...))``.

        Expected kwargs: ``hf_config``, ``tf_config``, ``config``, ``temperature``,
        ``calculate_entropy``, ``use_fused_kernels``.
        Megatron may also pass ``return_schedule_plan``, ``vpp_rank``.
        """
        tf_config = kwargs["tf_config"]
        calculate_entropy = kwargs.get("calculate_entropy", False)
        use_fused_kernels = kwargs.get("use_fused_kernels", False)
        hf_config = kwargs["hf_config"]
        temperature = kwargs["temperature"]
        config = kwargs["config"]
        return_schedule_plan = kwargs.get("return_schedule_plan", False)
        vpp_rank = kwargs.get("vpp_rank", None)

        if return_schedule_plan:
            assert tf_config.overlap_moe_expert_parallel_comm, (
                "overlap_moe_expert_parallel_comm must be enabled to return the schedule plan"
            )
            assert not calculate_entropy, "calculate_entropy must be disabled to return the schedule plan"
            from megatron.core.models.gpt.gpt_model import GPTModel

            assert isinstance(model, GPTModel), "model must be a GPTModel"
            assert use_fused_kernels, "use_fused_kernels must be enabled to return the schedule plan"
            from axon.models.mcore.forward.model_forward_1f1b_overlap import gptmodel_forward_1f1b_overlap

        batch = next(batch_iter)
        # Separate moe routermap from batch before GPU transfer.
        # IMPORTANT: do NOT mutate the micro-batch TensorDict (it may be a view
        # from split/rearrange). Use exclude() to create a new TensorDict.
        micro_moe_routermap = batch.get("local_moe_routermap", None)
        micro_vpp_counts = batch.get("local_vpp_counts", None)
        _exclude_keys = [k for k in ("local_moe_routermap", "local_vpp_counts") if k in batch]
        if _exclude_keys:
            batch = batch.exclude(*_exclude_keys)
        batch = batch.to(get_device_id()).contiguous()
        vpp_rank = _resolve_vpp_rank(model, vpp_rank)

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"].to(bool)
        position_ids = batch["position_ids"]

        multi_modal_inputs = {}
        if "multi_modal_inputs" in batch:
            from axon.utils.hf_model import extract_multi_modal_inputs

            # multi_modal_inputs rides through the dynamic micro-batch split as a TensorDict
            # NonTensorData scalar, which index_select_tensor_dict does NOT slice — so every
            # micro-batch carries the FULL (all-rows) mm array. The sliced multi_modal_inputs_idx
            # tensor holds THIS micro-batch's original row positions; use it to select the matching
            # grids so they align 1:1 with input_ids. (mbridge get_rope_index rebuilds mRoPE with a
            # single global image_index and raises a shape mismatch on any row<->grid desync — which
            # is exactly what happens at >1 micro-batch if we concat the full grid set.) Guard: if the
            # array is already sliced to this micro-batch (len == n_rows), use it as-is — the idx
            # would be out of range and drop every row.
            _mm = batch["multi_modal_inputs"]
            _mm_idx = batch.get("multi_modal_inputs_idx", None)
            if _mm_idx is not None and len(_mm) > batch["input_ids"].shape[0]:
                multi_modal_inputs = extract_multi_modal_inputs(_mm, _mm_idx.tolist())
            else:
                multi_modal_inputs = extract_multi_modal_inputs(_mm)

        # Labels are input_ids shifted left by 1: label[t] = input_ids[t+1]
        label = torch.zeros_like(input_ids)
        label[:, :-1] = input_ids[:, 1:]
        # Label mask: train on response tokens only, shifted to align with logits.
        # The logit at position t predicts label[t] = input_ids[t+1].
        # We want to train where input_ids[t+1] is a response token, so shift response_mask left by 1.
        label_mask = torch.zeros_like(attention_mask, dtype=torch.bool)
        response_mask = batch["response_mask"].bool() if "response_mask" in batch else attention_mask.clone().bool()
        label_mask[:, :-1] = response_mask[:, 1:]
        if "response_mask" in batch:
            label_mask[:, :-1] = batch["response_mask"][:, 1:].clone().to(bool)

        from axon.utils.megatron.forward_utils import setup_moe_routing

        setup_moe_routing(
            model, batch, vpp_rank, attention_mask, moe_routermap=micro_moe_routermap, moe_vpp_counts=micro_vpp_counts
        )

        from axon.models.mcore import get_mcore_forward_fn, get_mcore_forward_fused_fn

        # CP-aware optimization: skip the all-gather in postprocessing so each
        # rank only has log-probs for its own zigzag chunk (non-local = zero).
        _CP_SEQ_LEVEL_LOSSES = {"gspo", "geo_mean", "clip_cov", "kl_cov"}
        cp_size = mpu.get_context_parallel_world_size()
        loss_fn = kwargs.get("loss_fn", "ppo")
        forward_only = kwargs.get("forward_only", False)
        use_skip_cp = cp_size > 1 and (forward_only or loss_fn not in _CP_SEQ_LEVEL_LOSSES)

        if use_fused_kernels:
            forward_fn = get_mcore_forward_fused_fn(hf_config)
            if return_schedule_plan:
                forward_fn = gptmodel_forward_1f1b_overlap
            fused_kwargs = dict(
                model=model,
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                labels=label,
                labels_mask=label_mask,
                temperature=temperature,
                multi_modal_inputs=multi_modal_inputs,
            )
            # 1f1b overlap path does not support skip_cp_gather yet.
            if use_skip_cp and not return_schedule_plan:
                fused_kwargs["skip_cp_gather"] = True
            output = forward_fn(**fused_kwargs)
        else:
            forward_fn = get_mcore_forward_fn(hf_config)

            def logits_processor(logits, label, label_mask):
                assert logits.shape[:2] == label.shape[:2]
                assert label.shape == label_mask.shape
                logits.div_(temperature)
                ret = {}
                logits_bak = logits.clone() if calculate_entropy else logits
                if calculate_entropy:
                    ret["entropy"] = vocab_parallel_entropy(logits)
                ret["log_probs"] = vocab_parallel_log_probs_from_logits(logits_bak, label).masked_fill(~label_mask, 0.0)
                return ret

            output = forward_fn(
                model=model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                multi_modal_inputs=multi_modal_inputs,
                logits_processor=logits_processor,
                logits_processor_args={"label": label, "label_mask": label_mask},
                data_format="thd" if config.megatron.use_remove_padding else "bshd",
                skip_cp_gather=use_skip_cp,
            )

        return output, partial(
            CausalLM.loss_func,
            data=batch,
            use_cp_local_loss=use_skip_cp,
            **kwargs,
        )

    @staticmethod
    def loss_func(output, **kwargs):
        """Loss computation for causal LM.

        Expected kwargs: ``data``, ``forward_only``, ``calculate_entropy``,
        ``config``, ``n_micro_batches``, ``loss_fn``, ``loss_fn_args``.
        """
        data = kwargs["data"]
        forward_only = kwargs.get("forward_only", False)
        calculate_entropy = kwargs.get("calculate_entropy", False)
        loss_fn = kwargs.get("loss_fn", "ppo")
        loss_fn_args = kwargs.get("loss_fn_args", None)
        n_micro_batches = kwargs["n_micro_batches"]
        config = kwargs["config"]

        log_probs = (output["log_probs"] if isinstance(output, dict) else output).to(torch.float32)
        device = log_probs.device
        metrics = {}

        if forward_only:
            # Shift log_probs to align with token positions: log_prob[t] = log_probs[t-1]
            seq_length = data["input_ids"].size(1)
            log_prob = torch.zeros(log_probs.size(0), seq_length, dtype=log_probs.dtype, device=device)
            log_prob[:, 1:] = log_probs[:, :-1]
            metrics["log_probs"] = log_prob.contiguous()
            if not calculate_entropy:
                return torch.tensor(1.0, device=device), metrics
            ent = torch.zeros(log_probs.size(0), seq_length, dtype=log_probs.dtype, device=device)
            ent[:, 1:] = output["entropy"][:, :-1]
            return torch.tensor(1.0, device=device), [metrics, ent.contiguous()]

        # Training path
        seq_length = data["input_ids"].size(1)
        response_mask = data["response_mask"].to(bool)

        # Shift log_probs to align with token positions: log_prob[t] = log_probs[t-1]
        log_prob = torch.zeros(log_probs.size(0), seq_length, dtype=log_probs.dtype, device=device)
        log_prob[:, 1:] = log_probs[:, :-1]
        log_prob = log_prob.contiguous()
        data["log_probs"] = log_prob

        use_cp_local = kwargs.get("use_cp_local_loss", False)
        if use_cp_local:
            cp_local_response_mask = log_prob != 0
            response_mask = response_mask & cp_local_response_mask
            data["response_mask"] = response_mask
            data["response_mask"] = response_mask

        _lfn = loss_fn_args or {}
        agg_kw = dict(
            token_reduce=_lfn.get("token_reduce", "sum"),
            batch_reduce=_lfn.get("batch_reduce", "token-mean"),
            num_program_tokens=data.get("num_program_tokens", None),
            num_program_steps=data.get("num_program_steps", None),
            valid_token_count=data.get("valid_token_count", None),
            valid_batch_size=data.get("valid_batch_size", None),
            valid_program_count=data.get("valid_program_count", None),
            per_row_token_count=data.get("per_row_token_count", None),
        )

        pg_loss, pg_metrics = compute_loss_fn(DataProto.from_dict(data), loss_fn=loss_fn, loss_fn_args=loss_fn_args)
        stats = {}
        stats.update(pg_metrics)

        # Log-prob distribution diagnostics
        with torch.no_grad():
            valid_lp = log_prob[response_mask]
            if valid_lp.numel() > 0:
                stats["logprob_mean"] = valid_lp.mean().item()
                stats["logprob_min"] = valid_lp.min().item()
                stats["logprob_std"] = valid_lp.std().item()

        sampler_log_prob = data.get("sampler_log_probs", None)
        if loss_fn != "bypass_mode" and sampler_log_prob is not None and response_mask.any():
            stats.update(
                compute_sampler_corr_metrics_from_logprobs(
                    log_prob=log_prob, sampler_log_prob=sampler_log_prob, response_mask=response_mask
                )
            )
        # pg_loss from agg_loss is partial_sum/global_count (tiny per micro-batch).
        # Log the rescaled value so metrics reflect the true per-sample loss.
        stats["pg_loss"] = (pg_loss.detach() * n_micro_batches).item()
        policy_loss = pg_loss

        if calculate_entropy:
            entropy = torch.zeros(log_probs.size(0), seq_length, dtype=log_probs.dtype, device=device)
            entropy[:, 1:] = output["entropy"][:, :-1]
            entropy = entropy.contiguous()
            entropy_coef = (loss_fn_args or {}).get("entropy_coef", 0)
            if entropy_coef:
                entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, **agg_kw)
                policy_loss = pg_loss - entropy_coef * entropy_loss
            # Log entropy metrics without connecting to autograd graph
            stats["entropy"] = agg_loss(
                loss_mat=entropy.detach(),
                loss_mask=response_mask,
                token_reduce="sum",
                batch_reduce="token-mean",
                valid_token_count=agg_kw.get("valid_token_count"),
            ).item()

        kl_coef = (loss_fn_args or {}).get("kl_coef", 0.0)
        if kl_coef != 0:
            assert "ref_log_prob" in data, "ref_log_prob is required for KL loss"
            kl_type = (loss_fn_args or {}).get("kl_type", "low_var_kl")
            kld = kl_penalty(logprob=log_prob, ref_logprob=data["ref_log_prob"], kl_penalty_type=kl_type)
            kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, **agg_kw)
            policy_loss = policy_loss + kl_loss * kl_coef
            # kl_loss is partial_sum/global_count — rescale for logging (same as pg_loss fix)
            metrics["kl_loss"] = (kl_loss.detach() * n_micro_batches).item()
            metrics["kl_coef"] = kl_coef

        # Megatron DDP flat-averages gradients across dp_cp_world = D*K ranks.
        # CP-local: K partial grads sum to full. Need ×K to undo DDP's ÷K.
        # All-gather: K identical grads. DDP's ÷K already cancels them.
        dp_cp_world = mpu.get_data_parallel_world_size(with_context_parallel=True)
        cp_size = config.megatron.context_parallel_size
        if use_cp_local:
            world_factor = n_micro_batches * dp_cp_world
        else:
            world_factor = n_micro_batches * dp_cp_world / cp_size
        policy_loss = policy_loss * world_factor

        append_to_dict(metrics, stats)
        return policy_loss, metrics


class ValueModel:
    """Model-type class for critic (value-head) models."""

    @staticmethod
    def create_model(config, **kwargs):
        """Create a Megatron ValueModel (value-head) module.

        Called by ``TrainerWorker._build_model_optimizer``.
        ``config`` is the worker config.

        Required kwargs: ``tf_config``, ``hf_config``, ``bridge``, ``provider``.
        Optional kwargs: ``override_ddp_config``, ``peft_cls``.
        """
        from axon.utils.megatron.utils import McoreModuleWrapperConfig, make_megatron_module

        wrap_with_ddp = not config.get("forward_only", False)
        wrap_config = McoreModuleWrapperConfig(
            is_value_model=True,
            share_embeddings_and_output_weights=False,
            wrap_with_ddp=wrap_with_ddp,
            use_distributed_optimizer=config.megatron.use_distributed_optimizer,
        )
        peft_config = config.get("lora", None) if wrap_with_ddp else None
        return make_megatron_module(
            wrap_config=wrap_config,
            tf_config=kwargs["tf_config"],
            hf_config=kwargs["hf_config"],
            bridge=kwargs["bridge"],
            provider=kwargs["provider"],
            override_ddp_config=kwargs.get("override_ddp_config"),
            peft_cls=kwargs.get("peft_cls"),
            peft_config=peft_config,
            freeze_moe_router=config.megatron.get("freeze_moe_router", False),
        )

    @staticmethod
    def forward_output_fn(output, data, **kwargs):
        """Extract value predictions from pipeline output on the last PP stage.

        Called by TrainerWorker.forward after megatron_forward_backward.
        ``output`` is the dict returned by megatron_forward_backward.
        Returns dict with ``values``.
        """
        import itertools

        use_dynamic_bsz = kwargs.get("use_dynamic_bsz", False)
        attention_mask = data.batch["attention_mask"]
        seq_length = data.batch["input_ids"].size(1)

        if mpu.is_pipeline_last_stage(ignore_virtual=True):
            values = [o["vpreds"] for o in output["output"]]
            values = torch.cat(values, dim=0).to(torch.float32)
            if use_dynamic_bsz:
                indices = output["indices"]
                indices = list(itertools.chain.from_iterable(indices))
                assert len(indices) == values.size(0), f"{len(indices)} vs. {values.size()}"
                revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
                values = values[revert_indices]
        else:
            values = torch.empty_like(attention_mask, dtype=torch.float32)

        # Shift values to align with token positions: value[t] = vpreds[t-1]
        shifted_values = torch.zeros(values.size(0), seq_length, dtype=values.dtype, device=values.device)
        shifted_values[:, 1:] = values[:, :-1]
        response_mask = data.batch["response_mask"] if "response_mask" in data.batch else attention_mask
        shifted_values = shifted_values * response_mask
        values = shifted_values.contiguous()

        values = values.to(get_device_id())
        from torch import distributed as dist

        dist.broadcast(
            tensor=values,
            src=mpu.get_pipeline_model_parallel_last_rank(),
            group=mpu.get_pipeline_model_parallel_group(),
        )
        values = values.to("cpu")
        return {"values": values}

    @staticmethod
    def forward_step(batch_iter, model, **kwargs):
        """Per micro-batch forward computation for value model.

        Expected kwargs: ``hf_config``.
        """
        hf_config = kwargs["hf_config"]

        batch = next(batch_iter).to(get_device_id()).contiguous()

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        position_ids = batch["position_ids"]

        multi_modal_inputs = {}
        if "multi_modal_inputs" in batch:
            from axon.utils.hf_model import extract_multi_modal_inputs

            # See CausalLM.forward_step: multi_modal_inputs is the FULL (unsliced) mm array on
            # every micro-batch; select this micro-batch's rows via the sliced idx so grids align
            # 1:1 with input_ids (mbridge get_rope_index global-image_index desyncs otherwise).
            _mm = batch["multi_modal_inputs"]
            _mm_idx = batch.get("multi_modal_inputs_idx", None)
            if _mm_idx is not None and len(_mm) > batch["input_ids"].shape[0]:
                multi_modal_inputs = extract_multi_modal_inputs(_mm, _mm_idx.tolist())
            else:
                multi_modal_inputs = extract_multi_modal_inputs(_mm)

        from axon.models.mcore import get_mcore_forward_fn

        forward_fn = get_mcore_forward_fn(hf_config)
        output = forward_fn(
            model,
            input_ids,
            attention_mask,
            position_ids,
            multi_modal_inputs,
            value_model=True,
        )

        return output, partial(ValueModel.loss_func, data=batch, **kwargs)

    @staticmethod
    def loss_func(output, **kwargs):
        """Loss computation for value model.

        Expected kwargs: ``data``, ``forward_only``, ``loss_fn`` (optional),
        ``loss_fn_args`` (optional), ``n_micro_batches``, ``config``.
        """
        data = kwargs["data"]
        forward_only = kwargs.get("forward_only", False)
        loss_fn = kwargs.get("loss_fn", "value")
        loss_fn_args = kwargs.get("loss_fn_args", None)
        n_micro_batches = kwargs["n_micro_batches"]
        config = kwargs["config"]

        if forward_only:
            return torch.tensor(1.0, device=output.device), {"vpreds": output}

        seq_length = data["input_ids"].size(1)
        # Shift vpreds to align with token positions: vpreds[t] = output[t-1]
        shifted_vpreds = torch.zeros(output.size(0), seq_length, dtype=output.dtype, device=output.device)
        shifted_vpreds[:, 1:] = output[:, :-1]
        data["vpreds"] = shifted_vpreds
        if "response_mask" not in data:
            data["response_mask"] = data["attention_mask"]

        vf_loss, vf_metrics = compute_loss_fn(DataProto.from_dict(data), loss_fn=loss_fn, loss_fn_args=loss_fn_args)

        # Fix logged metric: agg_loss with global valid_batch_size returns partial_sum/global_count
        vf_metrics["vf_loss"] = (vf_loss.detach() * n_micro_batches).item()

        # Scale loss for Megatron pipeline (same as CausalLM)
        num_replicas = (
            mpu.get_data_parallel_world_size(with_context_parallel=True) / config.megatron.context_parallel_size
        )
        vf_loss = vf_loss * (n_micro_batches * num_replicas)

        return vf_loss, vf_metrics


class RewardModel:
    """Model-type class for reward models (inference-only)."""

    @staticmethod
    def create_model(config, **kwargs):
        """Create a Megatron RewardModel (value-head) module.

        Called by ``TrainerWorker._build_model_optimizer``.
        Same as ``ValueModel.create_model`` — reward models use the same
        value-head architecture (``is_value_model=True``, no tied embeddings).
        """
        from axon.utils.megatron.utils import McoreModuleWrapperConfig, make_megatron_module

        wrap_with_ddp = not config.get("forward_only", False)
        wrap_config = McoreModuleWrapperConfig(
            is_value_model=True,
            share_embeddings_and_output_weights=False,
            wrap_with_ddp=wrap_with_ddp,
            use_distributed_optimizer=config.megatron.use_distributed_optimizer,
        )
        peft_config = config.get("lora", None) if wrap_with_ddp else None
        return make_megatron_module(
            wrap_config=wrap_config,
            tf_config=kwargs["tf_config"],
            hf_config=kwargs["hf_config"],
            bridge=kwargs["bridge"],
            provider=kwargs["provider"],
            override_ddp_config=kwargs.get("override_ddp_config"),
            peft_cls=kwargs.get("peft_cls"),
            peft_config=peft_config,
            freeze_moe_router=config.megatron.get("freeze_moe_router", False),
        )

    @staticmethod
    def forward_output_fn(output, data, **kwargs):
        """Extract reward scores from pipeline output on the last PP stage.

        Called by TrainerWorker.forward after megatron_forward_backward.
        ``output`` is the dict returned by megatron_forward_backward.
        Returns dict with ``rm_scores`` shaped ``(batch_size, response_length)``.
        """
        import itertools

        use_dynamic_bsz = kwargs.get("use_dynamic_bsz", False)
        input_ids = data.batch["input_ids"]
        attention_mask = data.batch["attention_mask"]
        position_ids = data.batch["position_ids"]
        batch_size = input_ids.size(0)
        seq_len = input_ids.size(1)

        # --- Gather logits on last PP stage, broadcast to all PP ranks ---
        if mpu.is_pipeline_last_stage(ignore_virtual=True):
            logits = torch.cat(output["output"], dim=0).to(torch.float32)
            if use_dynamic_bsz:
                indices = list(itertools.chain.from_iterable(output["indices"]))
                assert len(indices) == logits.size(0), f"{len(indices)} vs. {logits.size()}"
                logits = logits[torch.tensor(get_reverse_idx(indices), dtype=torch.long)]
        else:
            logits = torch.empty((batch_size, seq_len), device=input_ids.device, dtype=torch.float32)

        logits = logits.to(get_device_id())
        from torch import distributed as dist

        dist.broadcast(
            tensor=logits,
            src=mpu.get_pipeline_model_parallel_last_rank(),
            group=mpu.get_pipeline_model_parallel_group(),
            async_op=False,
        )

        # --- Extract last-token score and expand to token-level at EOS ---
        pos_ids = position_ids[:, 0, :] if position_ids.dim() == 3 else position_ids
        eos_mask_idx = torch.argmax(pos_ids * attention_mask, dim=-1)  # (bs,)
        rm_score_per_sample = logits[torch.arange(batch_size, device=logits.device), eos_mask_idx]

        token_level_scores = torch.zeros_like(attention_mask, dtype=torch.float32)
        token_level_scores[torch.arange(batch_size, device=logits.device), eos_mask_idx] = rm_score_per_sample

        return {"rm_scores": token_level_scores.to("cpu")}

    @staticmethod
    def forward_step(batch_iter, model, **kwargs):
        """Per micro-batch forward computation for reward model.

        Expected kwargs: ``hf_config``.
        """
        hf_config = kwargs["hf_config"]

        batch = next(batch_iter).to(get_device_id()).contiguous()

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        position_ids = batch["position_ids"]

        multi_modal_inputs = {}
        if "multi_modal_inputs" in batch:
            from axon.utils.hf_model import extract_multi_modal_inputs

            # See CausalLM.forward_step: full (unsliced) mm array per micro-batch; select this
            # micro-batch's rows via the sliced idx so grids align 1:1 with input_ids.
            _mm = batch["multi_modal_inputs"]
            _mm_idx = batch.get("multi_modal_inputs_idx", None)
            if _mm_idx is not None and len(_mm) > batch["input_ids"].shape[0]:
                multi_modal_inputs = extract_multi_modal_inputs(_mm, _mm_idx.tolist())
            else:
                multi_modal_inputs = extract_multi_modal_inputs(_mm)

        from axon.models.mcore import get_mcore_forward_fn

        forward_fn = get_mcore_forward_fn(hf_config)
        output = forward_fn(
            model,
            input_ids,
            attention_mask,
            position_ids,
            multi_modal_inputs,
            value_model=True,
        )

        return output, RewardModel.loss_func

    @staticmethod
    def loss_func(output, **kwargs):
        """No-op loss for reward model (inference-only)."""
        return torch.tensor(1.0, device=output.device), output
