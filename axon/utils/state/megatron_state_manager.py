# Copyright 2025 Model AI Corp.
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
#
# Adapted from verl Megatron checkpoint manager (github.com/volcengine/verl), Apache-2.0.
import json
import logging
import os
import random
from collections.abc import Callable
from dataclasses import asdict

import numpy as np
import torch
import torch.distributed
from megatron.core import mpu, tensor_parallel
from megatron.core.dist_checkpointing.mapping import ShardedObject
from megatron.core.transformer.enums import AttnBackend
from transformers import GenerationConfig

from axon.utils.fs import local_mkdir_safe
from axon.utils.logger import log_with_rank
from axon.utils.megatron.dist_checkpointing import load_dist_checkpointing, save_dist_checkpointing
from axon.utils.megatron.utils import (
    get_hf_model_checkpoint_path,
    get_transformer_config_checkpoint_path,
)
from axon.utils.torch import get_device_name, get_torch_device

from .state_manager import BaseStateManager
from .utils import delete_oldest_checkpoints

# Setup logging
logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("AXON_LOGGING_LEVEL", "INFO"))


class MegatronStateManager(BaseStateManager):
    """
    Checkpoint manager for Megatron-LM distributed training.

    Save modes:
        - sharded: Save distributed checkpoint (model + optimizer + extra states)
        - hf: Save optimizer + extra states (distributed) + HuggingFace model weights
        - both: Save distributed checkpoint (model + optimizer + extra) + HuggingFace model weights

    Configuration:
        checkpoint_config:
          save_mode: "sharded" | "hf" | "both"
          async_save: false

    Key features:
    - Distributed checkpoint saving/loading using Megatron's dist_checkpointing
    - Support for tensor parallel, pipeline parallel, and data parallel configurations
    - Automatic handling of model state dictionaries across multiple pipeline stages
    - Integration with HuggingFace model configurations and tokenizers
    - Random number generator state management for reproducibility
    - Support for both synchronous and asynchronous checkpoint operations
    - PEFT/LoRA adapter checkpoint support

    Directory structure:
        checkpoints/step_N/
        ├── data.pt                    # Dataloader state
        ├── actor/
        │   ├── *.distcp, common.pt    # Megatron distributed checkpoint files
        │   └── huggingface/           # HF config + tokenizer (+ weights if save_mode=hf/both)
        └── critic/ (if using critic)
            └── ...
    """

    def __init__(
        self,
        config,
        model_config,
        transformer_config,
        role,
        model: torch.nn.ModuleList,
        arch: str,
        hf_config,
        param_dtype: torch.dtype,
        share_embeddings_and_output_weights: bool,
        processing_class,
        optimizer,
        optimizer_scheduler,
        use_distributed_optimizer: bool,
        use_dist_checkpointing: bool = True,
        bridge=None,
        provider=None,
        peft_cls=None,
        **kwargs,
    ):
        super().__init__(
            model,
            optimizer=optimizer,
            lr_scheduler=optimizer_scheduler,
            processing_class=processing_class,
        )
        self.arch = arch
        self.config = config
        self.transformer_config = transformer_config
        self.role = role
        self.is_value_model = False
        if self.role in ["reward", "critic"]:
            self.is_value_model = True
        self.model_config = model_config
        self.hf_config = hf_config
        self.param_dtype = param_dtype
        self.share_embeddings_and_output_weights = share_embeddings_and_output_weights
        self.model_path = self.config.model_path
        self.use_distributed_optimizer = use_distributed_optimizer
        self.bridge = bridge
        self.provider = provider
        self.vanilla_bridge = self.provider is None
        self.peft_cls = peft_cls
        self.rank = torch.distributed.get_rank()
        # Megatron-Bridge is Okay to load/save HF checkpoint for value model as well
        self.use_dist_checkpointing = (
            use_dist_checkpointing or not self.bridge or (self.vanilla_bridge and self.is_value_model)
        )
        self.use_hf_checkpoint = not self.use_dist_checkpointing

        assert self.bridge is not None, "mbridge is required. Non-bridge weight saving is no longer supported."

        # Initialize async queue for dynamic async save support
        # Always initialize when using dist checkpointing to allow dynamic control
        if self.use_dist_checkpointing:
            from megatron.core.dist_checkpointing.strategies.async_utils import AsyncCallsQueue

            self.async_calls_queue = AsyncCallsQueue(persistent=True)
            self.pending_request = None

    def get_rng_state(self, use_dist_ckpt: bool = True, data_parallel_random_init: bool = False):
        """collect rng state across data parallel ranks"""
        rng_state = {
            "random_rng_state": random.getstate(),
            "np_rng_state": np.random.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "rng_tracker_states": tensor_parallel.get_cuda_rng_tracker().get_states(),
        }

        if get_device_name() != "cpu":
            rng_state[f"{get_device_name()}_rng_state"] = get_torch_device().get_rng_state()

        rng_state_list = None
        if torch.distributed.is_initialized() and mpu.get_data_parallel_world_size() > 1 and data_parallel_random_init:
            rng_state_list = [None for i in range(mpu.get_data_parallel_world_size())]
            torch.distributed.all_gather_object(rng_state_list, rng_state, group=mpu.get_data_parallel_group())
        else:
            rng_state_list = [rng_state]

        if use_dist_ckpt:
            pp_rank = mpu.get_pipeline_model_parallel_rank()
            pp_size = mpu.get_pipeline_model_parallel_world_size()
            tp_rank = mpu.get_tensor_model_parallel_rank()
            tp_size = mpu.get_tensor_model_parallel_world_size()
            rng_state_list = ShardedObject(
                "rng_state",
                rng_state_list,
                (pp_size, tp_size),
                (pp_rank, tp_rank),
                replica_id=mpu.get_data_parallel_rank(with_context_parallel=True),
            )

        return rng_state_list

    def generate_state_dict(self, include_model: bool = True, is_loading: bool = False):
        """Generate sharded state dict for distributed checkpointing.

        Args:
            include_model: If True, include model state dict. Set to False for 'hf' save mode
                          where model weights are saved in HuggingFace format instead.
            is_loading: If True, generating state dict structure for loading.
        """
        state_dict = {}

        # Build model sharded state dict once — used for both model saving and optimizer structure.
        # model.sharded_state_dict() can be expensive for MoE models (ShardedTensorFactory
        # materializations), so we must avoid calling it twice.
        model_sharded_state = {}
        for vpp_rank, model in enumerate(self.model):
            if len(self.model) > 1:
                mpu.set_virtual_pipeline_model_parallel_rank(vpp_rank)
                key = f"model{vpp_rank}"
            else:
                key = "model"
            if hasattr(model, "module"):
                model = model.module
            model_sharded_state[key] = model.sharded_state_dict()

        if include_model:
            state_dict.update(model_sharded_state)

        # Optimizer state dict (uses model sharded state as structural template)
        torch.distributed.barrier()
        optimizer_sharded_states = self.optimizer.sharded_state_dict(model_sharded_state, is_loading=is_loading)
        state_dict["optimizer"] = optimizer_sharded_states

        if self.lr_scheduler is not None:
            state_dict["lr_scheduler"] = self.lr_scheduler.state_dict()

        # RNG states
        torch.distributed.barrier()
        state_dict["rng_state"] = self.get_rng_state()

        return state_dict

    def load_rng_states(self, rng_states, data_parallel_random_init=False):
        # access rng_state for data parallel rank
        if data_parallel_random_init:
            rng_states = rng_states[mpu.get_data_parallel_rank()]
        else:
            rng_states = rng_states[0]
        random.setstate(rng_states["random_rng_state"])
        np.random.set_state(rng_states["np_rng_state"])
        torch.set_rng_state(rng_states["torch_rng_state"])

        if get_device_name() != "cpu":
            get_torch_device().set_rng_state(rng_states[f"{get_device_name()}_rng_state"])

        # Check for empty states array
        if not rng_states["rng_tracker_states"]:
            raise KeyError
        tensor_parallel.get_cuda_rng_tracker().set_states(rng_states["rng_tracker_states"])

    def load_state(self, local_path: str, save_mode: str = None):
        """Load state (optimizer, extra states, and model from either distributed or HF format).

        Save modes determine where model weights are loaded from:
        - sharded/both: Load model from distributed state
        - hf: Load model from HuggingFace format

        Args:
            local_path: Directory with state files.
            save_mode: Save mode to use ("sharded", "hf", "both"). Defaults to "sharded".
        """
        if local_path is not None:
            assert os.path.exists(local_path), f"State path {local_path} does not exist."

        # For load optimizer dist_ckpt
        import transformer_engine

        torch.serialization.add_safe_globals([torch.optim.AdamW])
        torch.serialization.add_safe_globals([transformer_engine.pytorch.optimizers.fused_adam.FusedAdam])

        local_mkdir_safe(local_path)

        # Default to "sharded" if not specified
        from .state_manager import StateSaveMode

        effective_save_mode = StateSaveMode(save_mode) if save_mode else StateSaveMode.SHARDED
        should_save_sharded = effective_save_mode in (StateSaveMode.SHARDED, StateSaveMode.BOTH)

        # Generate sharded state dict structure for loading
        # include_model=True only if we saved model in distributed format (sharded/both modes)
        sharded_state_dict = self.generate_state_dict(include_model=should_save_sharded, is_loading=True)
        log_with_rank(f"Generated state dict for loading: {sharded_state_dict.keys()}", rank=self.rank, logger=logger)

        # Load distributed checkpoint (optimizer + extra, and optionally model)
        state_dict = load_dist_checkpointing(
            sharded_state_dict=sharded_state_dict,
            ckpt_dir=local_path,
        )
        del sharded_state_dict

        # Load model state
        if should_save_sharded:
            # Load model from distributed checkpoint (sharded/both modes)
            assert "model" in state_dict or any(
                f"model{vpp_rank}" in state_dict for vpp_rank in range(len(self.model))
            ), f"Model state dict not found in {state_dict.keys()}. Please check the checkpoint file {local_path}."
            for vpp_rank, model in enumerate(self.model):
                if len(self.model) == 1:
                    key = "model"
                else:
                    key = f"model{vpp_rank}"
                    assert key in state_dict, f"{key} not found in state_dict"
                mpu.set_virtual_pipeline_model_parallel_rank(vpp_rank)
                self.model[vpp_rank].load_state_dict(state_dict.pop(key))
            log_with_rank(f"Loaded sharded model checkpoint from {local_path}", rank=self.rank, logger=logger)
        elif self.peft_cls is None:
            # Load model from HuggingFace format (hf mode, skip if PEFT is used)
            hf_model_path = get_hf_model_checkpoint_path(local_path)
            if self.bridge is not None:
                if self.vanilla_bridge:
                    self.bridge.load_weights(self.model, hf_model_path)
                else:
                    self.bridge.load_hf_weights(self.model, hf_model_path)
                log_with_rank(f"Loaded HF model checkpoint from {hf_model_path}", rank=self.rank, logger=logger)
            else:
                raise ValueError("Bridge is required to load HF model checkpoint in 'hf' save mode")

        # Load PEFT adapter checkpoint if available
        if self.peft_cls is not None:
            adapter_ckpt_path = os.path.join(local_path, "adapter_checkpoint")
            if os.path.exists(adapter_ckpt_path):
                from axon.utils.megatron.peft_utils import load_adapter_checkpoint

                load_adapter_checkpoint(self.model, adapter_ckpt_path)
                log_with_rank(f"Loaded adapter checkpoint from {adapter_ckpt_path}", rank=self.rank, logger=logger)
            else:
                log_with_rank(
                    f"PEFT config is set but no adapter checkpoint found at {adapter_ckpt_path}",
                    rank=self.rank,
                    logger=logger,
                )

        # Load optimizer state — pop to free the loaded copy after load_state_dict
        assert "optimizer" in state_dict, (
            f"Optimizer state dict not found in {state_dict.keys()}. Please check the checkpoint file {local_path}."
        )
        self.optimizer.load_state_dict(state_dict.pop("optimizer"))
        log_with_rank(f"Loaded optimizer checkpoint from {local_path}", rank=self.rank, logger=logger)

        # Load LR scheduler state (optional - may not exist in older checkpoints)
        if self.lr_scheduler is not None and "lr_scheduler" in state_dict:
            self.lr_scheduler.load_state_dict(state_dict.pop("lr_scheduler"))
            log_with_rank(f"Loaded LR scheduler checkpoint from {local_path}", rank=self.rank, logger=logger)

        # Load RNG states
        assert "rng_state" in state_dict, (
            f"RNG state dict not found in {state_dict.keys()}. Please check the checkpoint file {local_path}."
        )
        self.load_rng_states(state_dict.pop("rng_state"))
        log_with_rank(f"Loaded RNG states from {local_path}", rank=self.rank, logger=logger)

        # Ensure loaded state dict is fully freed
        del state_dict

    def save_state(
        self,
        local_path: str,
        global_step: int = 0,
        max_ckpt_to_keep=None,
        save_mode: str = None,
        async_save: bool = None,
    ):
        """Save state based on save_mode.

        Save modes:
        - sharded: Save distributed state (model + optimizer + extra states)
        - hf: Save optimizer + extra in distributed format, model in HuggingFace format
        - both: Save distributed state (model + optimizer + extra) + HuggingFace model

        Args:
            local_path: Target directory for state files.
            global_step: Current training step (used for bookkeeping).
            max_ckpt_to_keep: Number of recent states to retain.
            save_mode: Save mode to use ("sharded", "hf", "both"). Defaults to "sharded".
            async_save: Whether to save asynchronously. Defaults to False.
        """
        # Use parameters if provided, else use defaults
        from .state_manager import StateSaveMode

        effective_save_mode = StateSaveMode(save_mode) if save_mode else StateSaveMode.SHARDED
        effective_async_save = async_save if async_save is not None else False
        should_save_sharded = effective_save_mode in (StateSaveMode.SHARDED, StateSaveMode.BOTH)
        should_save_hf = effective_save_mode in (StateSaveMode.HF, StateSaveMode.BOTH)

        # Finish pending async requests
        if effective_async_save and hasattr(self, "pending_request") and self.pending_request:
            self.async_calls_queue.maybe_finalize_async_calls(blocking=True, no_dist=False)
            self.pending_request = None
            torch.distributed.barrier()

        local_path = local_mkdir_safe(local_path)

        # Generate and save distributed checkpoint
        # For 'hf' mode: save optimizer + extra only (model saved in HF format)
        # For 'sharded'/'both' modes: save model + optimizer + extra
        state_dict = self.generate_state_dict(include_model=should_save_sharded)
        log_with_rank(f"Generated state dict for saving: {state_dict.keys()}", rank=self.rank, logger=logger)

        async_save_request = save_dist_checkpointing(
            sharded_state_dict=state_dict,
            ckpt_path=local_path,
            async_save=effective_async_save,
        )

        if not effective_async_save:
            torch.distributed.barrier()

        # Save adapter-only checkpoint if PEFT is enabled
        if self.peft_cls is not None:
            from axon.utils.megatron.peft_utils import save_adapter_checkpoint

            adapter_ckpt_path = os.path.join(local_path, "adapter_checkpoint")
            save_adapter_checkpoint(self.model, adapter_ckpt_path, self.rank)
            log_with_rank(
                f"Saved adapter-only checkpoint to {adapter_ckpt_path}",
                rank=self.rank,
                logger=logger,
                log_only_rank_0=True,
            )

        # Save HuggingFace model weights if save_mode is 'hf' or 'both'
        if should_save_hf:
            hf_ckpt_path = get_hf_model_checkpoint_path(local_path)
            if self.vanilla_bridge:
                self.bridge.save_weights(self.model, hf_ckpt_path, distributed_filesystem=True, memory_efficient=True)
            else:
                self.bridge.save_hf_weights(self.model, hf_ckpt_path)
            log_with_rank(f"Saved HF model checkpoint to {hf_ckpt_path}", rank=self.rank, logger=logger)

        # Save HF config, tokenizer, and generation config (always, for checkpoint validation)
        if self.rank == 0:
            hf_config_path = get_hf_model_checkpoint_path(local_path)
            if self.processing_class is not None:
                self.processing_class.save_pretrained(hf_config_path)
            self.hf_config.save_pretrained(hf_config_path)
            if hasattr(self.hf_config, "name_or_path") and self.hf_config.name_or_path:
                try:
                    generation_config = GenerationConfig.from_pretrained(self.hf_config.name_or_path)  # nosec B615
                    generation_config.save_pretrained(hf_config_path)
                except Exception:
                    pass
            log_with_rank(
                f"Saved HF config and tokenizer to {hf_config_path}",
                rank=self.rank,
                logger=logger,
                log_only_rank_0=True,
            )

            # Save transformer config
            bypass_keys = [
                "finalize_model_grads_func",
                "grad_scale_func",
                "no_sync_func",
                "grad_sync_func",
                "param_sync_func",
                "generation_config",
            ]
            backup = {}
            for k in bypass_keys:
                if hasattr(self.transformer_config, k):
                    backup[k] = getattr(self.transformer_config, k, None)
                    delattr(self.transformer_config, k)
            transformer_config_dict = asdict(self.transformer_config)
            for k in backup:
                setattr(self.transformer_config, k, backup[k])
            to_convert_types = {torch.dtype: str, AttnBackend: str}
            pop_keys = []
            for key, value in transformer_config_dict.items():
                if type(value) in to_convert_types:
                    transformer_config_dict[key] = to_convert_types[type(value)](value)
                if type(value) in [Callable] or callable(value):
                    pop_keys.append(key)
            for key in pop_keys:
                transformer_config_dict.pop(key)
            transformer_config_path = get_transformer_config_checkpoint_path(local_path)
            with open(transformer_config_path, "w") as f:
                json.dump(transformer_config_dict, f, indent=2)

        # Handle async save
        if effective_async_save and hasattr(self, "async_calls_queue"):
            self.pending_request = async_save_request
            self.async_calls_queue.schedule_async_request(self.pending_request)

        # Delete old checkpoints if max_ckpt_to_keep is set
        if max_ckpt_to_keep and isinstance(max_ckpt_to_keep, int) and max_ckpt_to_keep > 0:
            checkpoint_root_dir = os.path.dirname(os.path.dirname(local_path))
            if checkpoint_root_dir and os.path.exists(checkpoint_root_dir):
                delete_oldest_checkpoints(checkpoint_root_dir, max_ckpt_to_keep)
