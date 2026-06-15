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
# Adapted from verl FSDP checkpoint manager (github.com/volcengine/verl), Apache-2.0.
import json
import logging
import os
import warnings
from dataclasses import asdict, dataclass

import torch
import torch.distributed
from accelerate import init_empty_weights
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardedOptimStateDictConfig, ShardedStateDictConfig, StateDictType
from transformers import GenerationConfig, PreTrainedTokenizer, ProcessorMixin
from transformers.dynamic_module_utils import custom_object_save

from axon.utils.fs import local_mkdir_safe
from axon.utils.fsdp.utils import fsdp2_load_full_state_dict, fsdp_version, get_fsdp_full_state_dict, get_fsdp_state_ctx
from axon.utils.logger import log_with_rank
from axon.utils.torch import is_cuda_available

from .state_manager import BaseStateManager
from .utils import delete_oldest_checkpoints

# Setup logging
logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("AXON_LOGGING_LEVEL", "INFO"))


@dataclass
class FSDPConfig:
    """Configuration for FSDP checkpointing.

    Args:
        FSDP_version (int): Version of FSDP being used.
        world_size (int): Number of processes in the distributed training setup.
    """

    FSDP_version: int
    world_size: int


class FSDPStateManager(BaseStateManager):
    """
    Manage FSDP state in SPMD training.

    Save modes:
        - sharded: Save sharded model + optimizer + extra state per rank (for resuming training)
        - hf: Save sharded optimizer + extra state, model in HuggingFace format
        - both: Save sharded model + optimizer + extra state + HuggingFace model

    Args:
        model (FSDP): Wrapped model instance.
        optimizer (Optimizer): Training optimizer.
        lr_scheduler (LRScheduler): Learning-rate scheduler.
        processing_class (PreTrainedTokenizer or ProcessorMixin, optional):
            Pre-/post-processing artifact handler.
        state_config DictConfig: Configuration for state.
            - 'save_mode': One of 'sharded', 'hf', or 'both'. Defaults to 'sharded'.
    """

    def __init__(
        self,
        model: FSDP,
        optimizer: torch.optim.Optimizer | None = None,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        processing_class: PreTrainedTokenizer | ProcessorMixin = None,
        **kwargs,
    ):
        if processing_class is None and "tokenizer" in kwargs:
            warnings.warn(
                "`tokenizer` is deprecated. use `processing_class` instead.", DeprecationWarning, stacklevel=2
            )
            processing_class = kwargs.pop("tokenizer")

        super().__init__(
            model,
            optimizer,
            lr_scheduler=lr_scheduler,
            processing_class=processing_class,
        )

    def load_state(self, local_path: str, save_mode: str = None):
        """
        Load state based on save_mode.

        Save modes determine where model weights are loaded from:
        - sharded/both: Load model from sharded state
        - hf: Load model from HuggingFace format

        Optimizer and extra state are always loaded from sharded format.

        Args:
            local_path: Directory with state files.
            save_mode: Save mode to use ("sharded", "hf", "both"). Defaults to "sharded".
        """
        if local_path is None:
            return

        assert self.model is not None, "model must be provided to load state"
        assert self.optimizer is not None, "optimizer must be provided to load state"

        # Default to "sharded" if not specified
        from .state_manager import StateSaveMode

        effective_save_mode = StateSaveMode(save_mode) if save_mode else StateSaveMode.SHARDED
        should_save_sharded = effective_save_mode in (StateSaveMode.SHARDED, StateSaveMode.BOTH)

        state_dict_cfg = ShardedStateDictConfig(offload_to_cpu=True if is_cuda_available else False)
        optim_cfg = ShardedOptimStateDictConfig(offload_to_cpu=True if is_cuda_available else False)

        # Load model - from sharded or HF format depending on save_mode
        if should_save_sharded:
            # Load model from sharded checkpoint (sharded/both modes)
            with get_fsdp_state_ctx(self.model, StateDictType.SHARDED_STATE_DICT, state_dict_cfg, optim_cfg):
                remote_model_path = os.path.join(local_path, f"model_world_size_{self.world_size}_rank_{self.rank}.pt")
                model_state_dict = torch.load(remote_model_path, weights_only=False)  # nosec B614
                self.model.load_state_dict(model_state_dict)
                log_with_rank(f"Loaded model from {remote_model_path}", rank=self.rank, logger=logger)
        else:
            # Load model from HuggingFace format (hf mode)
            from transformers import AutoModelForCausalLM

            hf_model_path = os.path.join(local_path, "huggingface")

            # Load full state dict on rank 0
            if self.rank == 0:
                hf_model = AutoModelForCausalLM.from_pretrained(hf_model_path, torch_dtype="auto")  # nosec B615
                full_state_dict = hf_model.state_dict()
                del hf_model
            else:
                full_state_dict = None

            # Load into FSDP model - handle both FSDP1 and FSDP2
            model_fsdp_version = fsdp_version(self.model)
            if model_fsdp_version == 1:
                # FSDP1: Use FullStateDictConfig with rank0_only=True to broadcast from rank 0
                from torch.distributed.fsdp import FullStateDictConfig

                full_state_dict_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
                with get_fsdp_state_ctx(self.model, StateDictType.FULL_STATE_DICT, full_state_dict_cfg, None):
                    self.model.load_state_dict(full_state_dict if self.rank == 0 else {})
            elif model_fsdp_version == 2:
                # FSDP2: Use fsdp2_load_full_state_dict which broadcasts from rank 0 to all ranks
                fsdp2_load_full_state_dict(self.model, full_state_dict if self.rank == 0 else {})
            else:
                raise ValueError(f"Unknown FSDP version: {model_fsdp_version}")

            log_with_rank(f"Loaded model from HF checkpoint {hf_model_path}", rank=self.rank, logger=logger)

        # Load optimizer shard (always from sharded format)
        with get_fsdp_state_ctx(self.model, StateDictType.SHARDED_STATE_DICT, state_dict_cfg, optim_cfg):
            remote_optim_path = os.path.join(local_path, f"optim_world_size_{self.world_size}_rank_{self.rank}.pt")
            optimizer_state_dict = torch.load(remote_optim_path, weights_only=False)  # nosec B614
            self.optimizer.load_state_dict(optimizer_state_dict)
            log_with_rank(f"Loaded optimizer from {remote_optim_path}", rank=self.rank, logger=logger)

        # Load extra state (lr_scheduler + RNG)
        remote_extra_state_path = os.path.join(
            local_path, f"extra_state_world_size_{self.world_size}_rank_{self.rank}.pt"
        )
        extra_state_dict = torch.load(remote_extra_state_path, weights_only=False)  # nosec B614

        # Recover random state
        if "rng" in extra_state_dict:
            self.load_rng_state(extra_state_dict["rng"])
            log_with_rank(f"Loaded rng from {remote_extra_state_path}", rank=self.rank, logger=logger)

        lr_scheduler_state_dict = extra_state_dict["lr_scheduler"]
        if lr_scheduler_state_dict is not None and self.lr_scheduler is not None:
            self.lr_scheduler.load_state_dict(lr_scheduler_state_dict)
            log_with_rank(f"Loaded lr_scheduler from {remote_extra_state_path}", rank=self.rank, logger=logger)

        # wait for everyone to load checkpoints
        torch.distributed.barrier()

        del extra_state_dict
        torch.cuda.empty_cache()

    def save_state(
        self,
        local_path: str,
        global_step: int = 0,
        max_ckpt_to_keep=None,
        save_mode: str = None,
        async_save: bool = None,
    ):
        """
        Save state based on save_mode.

        Save modes:
          - sharded: Save sharded model + optimizer + extra state per rank
          - hf: Save sharded optimizer + extra state, model in HuggingFace format
          - both: Save sharded model + optimizer + extra + HuggingFace model

        Rotates old states, keeping at most `max_ckpt_to_keep`.

        Args:
            local_path: Target directory for state files.
            global_step: Current training step (used for bookkeeping).
            max_ckpt_to_keep: Number of recent states to retain.
            save_mode: Save mode to use ("sharded", "hf", "both"). Defaults to "sharded".
            async_save: Ignored for FSDP (async save not supported).
        """
        if local_path is None:
            return

        assert self.model is not None, "model must be provided to save state"
        assert self.optimizer is not None, "optimizer must be provided to save state"

        # Default to "sharded" if not specified
        from .state_manager import StateSaveMode

        effective_save_mode = StateSaveMode(save_mode) if save_mode else StateSaveMode.SHARDED
        should_save_sharded = effective_save_mode in (StateSaveMode.SHARDED, StateSaveMode.BOTH)
        should_save_hf = effective_save_mode in (StateSaveMode.HF, StateSaveMode.BOTH)

        local_path = local_mkdir_safe(local_path)
        torch.distributed.barrier()

        # Get model config and generation config early
        if fsdp_version(self.model) == 1:
            unwrap_model = self.model._fsdp_wrapped_module
        else:
            unwrap_model = self.model

        model_config = unwrap_model.config
        generation_config = None
        if unwrap_model.can_generate() and hasattr(model_config, "name_or_path") and model_config.name_or_path:
            try:
                generation_config = GenerationConfig.from_pretrained(model_config.name_or_path)  # nosec B615
            except Exception:
                pass

        state_dict_cfg = ShardedStateDictConfig(offload_to_cpu=True if is_cuda_available else False)
        optim_cfg = ShardedOptimStateDictConfig(offload_to_cpu=True if is_cuda_available else False)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with get_fsdp_state_ctx(self.model, StateDictType.SHARDED_STATE_DICT, state_dict_cfg, optim_cfg):
                # Save model shard only for sharded/both modes
                if should_save_sharded:
                    model_path = os.path.join(local_path, f"model_world_size_{self.world_size}_rank_{self.rank}.pt")
                    model_state_dict = self.model.state_dict()
                    torch.save(model_state_dict, model_path)
                    log_with_rank(f"Saved model to {os.path.abspath(model_path)}", rank=self.rank, logger=logger)

                # Save optimizer shard (always)
                optim_path = os.path.join(local_path, f"optim_world_size_{self.world_size}_rank_{self.rank}.pt")
                optimizer_state_dict = self.optimizer.state_dict()
                torch.save(optimizer_state_dict, optim_path)
                log_with_rank(f"Saved optim to {os.path.abspath(optim_path)}", rank=self.rank, logger=logger)

                # Save extra state (always)
                extra_path = os.path.join(local_path, f"extra_state_world_size_{self.world_size}_rank_{self.rank}.pt")
                lr_scheduler_state_dict = self.lr_scheduler.state_dict() if self.lr_scheduler is not None else None
                extra_state_dict = {
                    "lr_scheduler": lr_scheduler_state_dict,
                    "rng": self.get_rng_state(),
                }
                torch.save(extra_state_dict, extra_path)
                log_with_rank(f"Saved extra_state to {os.path.abspath(extra_path)}", rank=self.rank, logger=logger)

        # Save FSDP config on rank 0
        if self.rank == 0:
            fsdp_config_path = os.path.join(local_path, "fsdp_config.json")
            fsdp_config = FSDPConfig(
                FSDP_version=fsdp_version(self.model),
                world_size=self.world_size,
            )
            with open(fsdp_config_path, "w") as f:
                json.dump(asdict(fsdp_config), f, indent=4)

        # Ensure all sharded saves complete before proceeding
        torch.distributed.barrier()

        hf_local_path = os.path.join(local_path, "huggingface")
        local_mkdir_safe(hf_local_path)
        if self.rank == 0:
            if hasattr(model_config, "auto_map") and None in model_config.auto_map:
                model_config.auto_map = {k: v for k, v in model_config.auto_map.items() if k is not None}

        if should_save_hf:
            # Save full HuggingFace model (weights gathered on rank 0)
            state_dict = get_fsdp_full_state_dict(self.model, offload_to_cpu=True, rank0_only=True)

            if self.rank == 0:
                if "ForTokenClassification" in model_config.architectures[0]:
                    from transformers import AutoModelForTokenClassification

                    auto_model_cls = AutoModelForTokenClassification
                elif "ForCausalLM" in model_config.architectures[0]:
                    from transformers import AutoModelForCausalLM

                    auto_model_cls = AutoModelForCausalLM
                elif "ForConditionalGeneration" in model_config.architectures[0]:
                    from transformers import AutoModelForImageTextToText

                    auto_model_cls = AutoModelForImageTextToText
                else:
                    raise NotImplementedError(f"Unknown architecture {model_config['architectures']}")

                with init_empty_weights():
                    save_model = auto_model_cls.from_config(model_config, torch_dtype=torch.bfloat16)
                save_model.to_empty(device="cpu")
                if save_model.can_generate() and generation_config is not None:
                    save_model.generation_config = generation_config
                # save_pretrained saves weights + config.json + generation_config.json
                save_model.save_pretrained(hf_local_path, state_dict=state_dict)
                log_with_rank(
                    f"Saved hf_model to {os.path.abspath(hf_local_path)}",
                    rank=self.rank,
                    logger=logger,
                    log_only_rank_0=True,
                )
                del state_dict
                del save_model
        else:
            # Sharded-only: save config for later checkpoint merging
            if self.rank == 0:
                model_config.save_pretrained(hf_local_path)
                if generation_config is not None:
                    generation_config.save_pretrained(hf_local_path)

        # Always save tokenizer/processor (save_pretrained doesn't include these)
        if self.rank == 0:
            if self.processing_class is not None:
                self.processing_class.save_pretrained(hf_local_path)
            # Copy custom model definition files if needed
            if hasattr(model_config, "auto_map"):
                custom_object_save(unwrap_model, hf_local_path, config=model_config)
            log_with_rank(
                f"Saved checkpoint to {os.path.abspath(hf_local_path)}",
                rank=self.rank,
                logger=logger,
                log_only_rank_0=True,
            )

        torch.distributed.barrier()

        # Delete old checkpoints if max_ckpt_to_keep is set (in background)
        if max_ckpt_to_keep and isinstance(max_ckpt_to_keep, int) and max_ckpt_to_keep > 0:
            checkpoint_root_dir = os.path.dirname(os.path.dirname(local_path))
            if checkpoint_root_dir and os.path.exists(checkpoint_root_dir):
                import threading

                cleanup_thread = threading.Thread(
                    target=delete_oldest_checkpoints, args=(checkpoint_root_dir, max_ckpt_to_keep), daemon=True
                )
                cleanup_thread.start()
