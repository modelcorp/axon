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

"""Sync sampler mixins for hybrid-engine weight transfer.

FSDPSyncSamplerMixin and MegatronSyncSamplerMixin provide the
framework-specific ``sampler_mode()`` method that exports actor weights
to the inference engine when co-located (fused) with a TrainerWorker.
"""

import logging
import os

import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import FullStateDictConfig, ShardedStateDictConfig, StateDictType

try:
    from torch.distributed._tensor import DTensor
except ImportError:
    from torch.distributed.tensor import DTensor

from axon.utils.fsdp.utils import (
    collect_lora_params,
    fsdp_version,
    load_fsdp_model_to_gpu,
    offload_fsdp_model_to_cpu,
    replace_lora_wrapper,
)
from axon.utils.hf_model import convert_weight_keys
from axon.utils.memory_utils import aggressive_empty_cache
from axon.utils.profiler import log_gpu_memory_usage
from axon.utils.torch import get_device_id, get_torch_device, set_expandable_segments

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("AXON_LOGGING_LEVEL", "WARN"))


class SyncSamplerMixin:
    """Base class for hybrid-engine sampler sync mixins."""

    async def sampler_mode(self):
        raise NotImplementedError


class FSDPSyncSamplerMixin(SyncSamplerMixin):
    """FSDP-specific hybrid-engine weight export to sampler."""

    def _build_sampler(self):
        """Override to add FSDP-specific post-build setup."""
        super()._build_sampler()
        self._fsdp_post_build_sampler()

    def _fsdp_post_build_sampler(self):
        """Called after _build_sampler() to set FSDP state dict type on the actor module."""
        actor = getattr(self, "_colocated", {}).get("actor")
        if actor is not None:
            module_fsdp = actor.module_fsdp
            if torch.distributed.get_world_size() == 1 and fsdp_version(module_fsdp) == 1:
                FSDP.set_state_dict_type(
                    module_fsdp,
                    state_dict_type=StateDictType.FULL_STATE_DICT,
                    state_dict_config=FullStateDictConfig(),
                )
            elif fsdp_version(module_fsdp) == 1:
                FSDP.set_state_dict_type(
                    module_fsdp,
                    state_dict_type=StateDictType.SHARDED_STATE_DICT,
                    state_dict_config=ShardedStateDictConfig(),
                )

        self.base_sync_done: bool = "dummy" not in self.config.load_format
        self.layered_summon = self.config.get("layered_summon", False)

    async def sampler_mode(self):
        """Export actor weights to sampler in hybrid-engine (fused) mode."""
        actor = self._colocated["actor"]
        module_fsdp = actor.module_fsdp

        aggressive_empty_cache(force_sync=True)

        log_gpu_memory_usage("Before load_fsdp_model_to_gpu", logger=logger)
        if actor._is_offload_param:
            load_fsdp_model_to_gpu(module_fsdp)
        log_gpu_memory_usage("After load_fsdp_model_to_gpu", logger=logger)

        peft_config = None
        peft_model = getattr(module_fsdp, "_fsdp_wrapped_module", module_fsdp)
        if hasattr(peft_model, "peft_config"):  # LoRA
            peft_config = peft_model.peft_config.get("default", None)
            params = collect_lora_params(
                module=module_fsdp,
                layered_summon=self.layered_summon,
                base_sync_done=self.base_sync_done,
            )
            if not self.base_sync_done:
                params = {replace_lora_wrapper(k, peft_config): v for k, v in params.items()}
        else:
            params = module_fsdp.state_dict()

        params = convert_weight_keys(params, getattr(module_fsdp, "_fsdp_wrapped_module", module_fsdp))

        if peft_config is not None and getattr(self.sampler, "sleep_level", None) == 2:
            base_model_params = collect_lora_params(
                module=module_fsdp,
                layered_summon=self.layered_summon,
                base_sync_done=False,
            )
            base_model_params = {replace_lora_wrapper(k, peft_config): v for k, v in base_model_params.items()}
            base_model_params = convert_weight_keys(
                base_model_params, getattr(module_fsdp, "_fsdp_wrapped_module", module_fsdp)
            )

        log_gpu_memory_usage("Before offload_fsdp_model_to_cpu", logger=logger)
        if actor._is_offload_param:
            offload_fsdp_model_to_cpu(module_fsdp)
        log_gpu_memory_usage("After offload_fsdp_model_to_cpu", logger=logger)

        set_expandable_segments(False)

        device = get_device_id()
        if peft_config is not None and self.base_sync_done:
            per_tensor_param = params.items() if isinstance(params, dict) else params
        else:
            per_tensor_param = (
                (name, param.to(device, non_blocking=True).full_tensor() if isinstance(param, DTensor) else param)
                for name, param in params.items()
            )

        if self.config.offload_sampler:
            await self.sampler.resume(tags=["weights"])
        log_gpu_memory_usage("After resume weights", logger=logger)

        if peft_config is not None and getattr(self.sampler, "sleep_level", None) == 2:
            per_tensor_base_params = (
                (name, param.to(device, non_blocking=True).full_tensor() if isinstance(param, DTensor) else param)
                for name, param in base_model_params.items()
            )
            await self.sampler.update_weights(per_tensor_base_params, base_sync_done=False)
            del base_model_params, per_tensor_base_params

        await self.sampler.update_weights(per_tensor_param, peft_config=peft_config, base_sync_done=self.base_sync_done)
        log_gpu_memory_usage("After update_weights", logger=logger)
        del params, per_tensor_param
        aggressive_empty_cache(force_sync=True)
        if self.config.offload_sampler:
            await self.sampler.resume(tags=["kv_cache"])
            # For Mamba/GDN models (e.g. Qwen3.5), zero conv_state and
            # ssm_state after remap to ensure clean initial state.
            await self.sampler.zero_mamba_cache()

        log_gpu_memory_usage("After resume kv_cache", logger=logger)

        self.base_sync_done = True
        self._colocated["actor"].torch_random_states = get_torch_device().get_rng_state()
        get_torch_device().set_rng_state(self.gen_random_states)


class MegatronSyncSamplerMixin(SyncSamplerMixin):
    """Megatron-specific hybrid-engine weight export to sampler."""

    async def sampler_mode(self):
        """Export actor weights to sampler in hybrid-engine (fused) mode."""
        from axon.utils.megatron.utils import (
            load_megatron_model_to_gpu,
            offload_megatron_model_to_cpu,
            per_tensor_generator,
        )

        assert "actor" in self._colocated, "Actor is not collocated with sampler (hybrid_engine=False)"
        actor = self._colocated["actor"]

        aggressive_empty_cache(force_sync=True)
        set_expandable_segments(False)

        if actor._is_offload_param:
            load_megatron_model_to_gpu(actor.module, load_grad=False)
            log_gpu_memory_usage("After load actor params during sampler_mode", logger=logger)

        if actor.bridge is not None:
            if actor.vanilla_bridge:
                per_tensor_param = actor.bridge.export_weights(actor.module)
            else:
                per_tensor_param = actor.bridge.export_hf_weights(actor.module)
        else:
            per_tensor_param = per_tensor_generator(
                actor.module,
                actor.model_config,
                actor.weight_converter,
                actor.tf_config,
                actor.layer_name_mapping,
            )

        if self.config.offload_sampler:
            await self.sampler.resume(tags=["weights"])
        await self.sampler.update_weights(per_tensor_param)

        self.model_runner = self.sampler.inference_engine.worker.model_runner
        drafter = getattr(self.model_runner, "drafter", None)
        if drafter is not None and hasattr(drafter, "model"):
            try:
                if actor.bridge is not None:
                    per_tensor_param_draft = actor.bridge.export_weights(actor.module, export_mtp=True)
                else:
                    per_tensor_param_draft = per_tensor_generator(
                        actor.module,
                        actor.model_config,
                        actor.weight_converter,
                        actor.tf_config,
                        actor.layer_name_mapping,
                    )

                draft_model = drafter.model
                loaded_params_draft = draft_model.load_weights(per_tensor_param_draft)
                logger.info(
                    "vLLM load drafter weights (%s), loaded_params: %d",
                    draft_model.__class__.__name__,
                    len(loaded_params_draft),
                )
            except Exception as e:
                logger.warning(f"Failed to load drafter weights: {e}")
                import traceback

                logger.warning(traceback.format_exc())

        if actor._is_offload_param:
            offload_megatron_model_to_cpu(actor.module)
        aggressive_empty_cache(force_sync=True)
        if self.config.offload_sampler:
            await self.sampler.resume(tags=["kv_cache"])
            # For Mamba/GDN models (e.g. Qwen3.5), zero conv_state and
            # ssm_state after remap to ensure clean initial state.
            await self.sampler.zero_mamba_cache()

        self._colocated["actor"].torch_random_states = get_torch_device().get_rng_state()
        get_torch_device().set_rng_state(self.gen_random_states)
