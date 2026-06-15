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
"""Megatron-based TrainerWorker and SamplerWorker for PPO training."""

import datetime
import logging
import os
import resource
import time

import psutil
import torch
import torch.distributed as dist
from megatron.core import parallel_state as mpu
from omegaconf import DictConfig, OmegaConf
from transformers import AutoConfig

from axon.controller.decorator import Dispatch, make_nd_compute_dataproto_dispatch_fn, register
from axon.core.worker import Worker
from axon.models.megatron_models import CausalLM
from axon.protocol import DataProto
from axon.utils import hf_tokenizer
from axon.utils.config import get_profiler_tool_config, omega_conf_to_dataclass
from axon.utils.hf_model import get_hf_model_path, load_mcore_dist_weights
from axon.utils.megatron.forward_utils import megatron_forward_backward
from axon.utils.megatron.utils import (
    load_megatron_model_to_gpu,
    load_megatron_optimizer,
    offload_megatron_model_to_cpu,
    offload_megatron_optimizer,
    register_megatron_training_hooks,
)
from axon.utils.memory_utils import aggressive_empty_cache
from axon.utils.profiler import (
    DistProfiler,
    DistProfilerExtension,
    GPUMemoryLogger,
    ProfilerConfig,
    log_gpu_memory_usage,
)
from axon.utils.profiler.flops_counter import FlopsCounter
from axon.utils.state.megatron_state_manager import MegatronStateManager
from axon.utils.torch import (
    get_nccl_backend,
    get_torch_device,
    set_expandable_segments,
)
from axon.utils.torch.distributed import set_numa_affinity

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("AXON_LOGGING_LEVEL", "WARN"))

DEBUG_P2P_MISMATCH = False


def set_random_seed(seed, only_sampler=False):
    import random

    import numpy as np
    import torch

    seed = seed + (100 * mpu.get_pipeline_model_parallel_rank())

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if not only_sampler and get_torch_device().device_count() > 0:
        from megatron.core import tensor_parallel

        tensor_parallel.model_parallel_cuda_manual_seed(seed)
    # Deterministic algorithms are intentionally not forced here because
    # torch.cumsum is still unsupported in deterministic mode and is used by the sampler.
    # https://github.com/pytorch/pytorch/issues/89492
    # torch.use_deterministic_algorithms(True, warn_only=True)
    # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


class TrainerWorker(Worker, DistProfilerExtension):
    """Megatron-Core-backed trainer worker — runs forward + backward + optimiser step.

    The Megatron counterpart to the FSDP trainer in
    ``axon/trainer/fsdp_trainer.py``. Use ``strategy: megatron`` in the recipe
    to select this backend. Megatron supports the full 6D parallelism surface
    (TP / PP / EP / ETP / CP / DP) and is the default for large-model recipes.

    Mode-specific behaviour is layered on at runtime via mixins —
    ``MegatronSyncTrainerMixin`` / ``MegatronTrainerP2PMixin`` /
    ``AsyncTrainerMixin`` are composed by ``axon/driver/train_agent_ppo.py``.

    Parameterized by a *model-type* class (``CausalLM``, ``ValueModel``, …)
    that defines model-specific forward, loss, and output-extraction logic.

    Usage::

        actor  = TrainerWorker(config, model=CausalLM)
        critic = TrainerWorker(config, model=ValueModel, name="critic")
        ref    = TrainerWorker(config, model=CausalLM, name="ref")

    With ``forward_only=False`` (default) this is a full training worker.
    With ``forward_only=True`` it becomes a forward-only reference worker.

    For hybrid-engine mode, compose with ``SamplerWorker`` via
    ``fuse_worker_cls({"actor": TrainerWorker, "sampler": SamplerWorker})``.

    This class is designed to be composed with other workers via
    ``fuse_worker_cls()``, which discovers ``@register``-decorated methods
    automatically.
    """

    def __init__(self, config: DictConfig, name: str = "actor", model=None, **kwargs):
        Worker.__init__(self)
        self.config = config
        self._mesh_name = "trainer"
        self.name = name
        self.model = model or CausalLM

        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        if soft < 65536:
            new_limit = min(65536, hard)
            resource.setrlimit(resource.RLIMIT_NOFILE, (new_limit, hard))

        from axon.monkey_patches.megatron.moe_replay import apply_router_replay_patches

        apply_router_replay_patches()
        # Align Megatron's MoE routing probability placement with the sampler path
        # before old-logprob recompute.
        from axon.monkey_patches.megatron.moe_prob_placement import apply_moe_post_fc2_prob_patch

        apply_moe_post_fc2_prob_patch()

        from axon.monkey_patches.megatron.mtp_training import patch_mtp_for_packed_sequences

        patch_mtp_for_packed_sequences()

        # MoE checkpoint fixes: EP replica_id + master_param fallback
        from axon.monkey_patches.megatron.checkpoint_moe_fix import apply_moe_checkpoint_patches

        apply_moe_checkpoint_patches()

        # ROCm: patch checkpoint writer to avoid pinned-memory + fork segfaults
        from axon.utils.rocm_utils import apply_rocm_checkpoint_writer_patch

        apply_rocm_checkpoint_writer_patch()

        if not self._ensure_distributed_initialized(600):
            rank = int(os.environ["LOCAL_RANK"])
            get_torch_device().set_device(rank)
            if self.config.megatron.sequence_parallel:
                os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"

        # In fused mode, TrainerWorker already called mpu.initialize_model_parallel().
        if not mpu.model_parallel_is_initialized():
            mpu.initialize_model_parallel(
                tensor_model_parallel_size=self.config.megatron.tensor_model_parallel_size,
                pipeline_model_parallel_size=self.config.megatron.pipeline_model_parallel_size,
                virtual_pipeline_model_parallel_size=self.config.megatron.virtual_pipeline_model_parallel_size,
                use_sharp=False,
                context_parallel_size=self.config.megatron.context_parallel_size,
                expert_model_parallel_size=self.config.megatron.expert_model_parallel_size,
                expert_tensor_parallel_size=self.config.megatron.expert_tensor_parallel_size,
                nccl_communicator_config_path=None,
            )

            # Initialize single-rank groups for vision model TP bypass
            # Must be called collectively by all ranks after dist init and mpu init
            from axon.models.mbridge.qwen2_5_vl import ensure_single_rank_groups

            ensure_single_rank_groups()

        self._register_dispatch_collect_info(
            mesh_name=self._mesh_name, dp_rank=mpu.get_data_parallel_rank(), is_collect=self._is_collect_rank()
        )

        set_random_seed(seed=self.config.megatron.seed, only_sampler=False)
        self.torch_random_states = get_torch_device().get_rng_state()
        get_torch_device().set_rng_state(self.torch_random_states)

        # Profiler
        omega_profiler_config = config.get("profiler", {})
        profiler_config = omega_conf_to_dataclass(omega_profiler_config, dataclass_type=ProfilerConfig)
        tool_config = get_profiler_tool_config(omega_profiler_config)
        DistProfilerExtension.__init__(
            self, DistProfiler(rank=self.rank, config=profiler_config, tool_config=tool_config)
        )

        # forward_only: skip optimizer/training-only init (used by ref workers)
        self._forward_only = self.config.get("forward_only", False)

        # Defaults for training-only attributes (set properly in _init_config when training)
        self.use_torch_profiler = False
        self.prof = None
        self.use_fused_kernels = False
        self._profiler_started = False

        # Offload flags
        self._is_offload_param = self.config.get("param_offload", False)
        self._is_offload_grad = False if self._forward_only else self.config.megatron.get("grad_offload", False)
        self._is_offload_optimizer = False if self._forward_only else self.config.get("optimizer_offload", False)

        # Normalize micro_batch_size (training batches)
        if not self._forward_only and self.config.get("micro_batch_size", None):
            self.config.micro_batch_size //= mpu.get_data_parallel_world_size()
            self.config.micro_batch_size_per_gpu = self.config.micro_batch_size

        # Normalize forward_micro_batch_size (inference/forward-only batches)
        if self.config.get("forward_micro_batch_size", None):
            self.config.forward_micro_batch_size //= mpu.get_data_parallel_world_size()
            self.config.forward_micro_batch_size_per_gpu = self.config.forward_micro_batch_size

        # P2P state
        self.debug_p2p_mismatch = DEBUG_P2P_MISMATCH
        self.offload_p2p_buffer = self.config.get("offload_p2p_buffer", False)
        self.ops = []
        self.buffers = []
        self.routing_table = None

    # ---- Static helpers ----

    @staticmethod
    def _ensure_distributed_initialized(nccl_timeout=600):
        """Initialize torch.distributed if not already done. Returns True if newly initialized."""
        if dist.is_initialized():
            return False
        set_numa_affinity()
        rank = int(os.environ["LOCAL_RANK"])
        dist.init_process_group(
            backend=get_nccl_backend(),
            timeout=datetime.timedelta(seconds=nccl_timeout),
            init_method=os.environ.get("DIST_INIT_METHOD", None),
        )
        get_torch_device().set_device(rank)
        return True

    @staticmethod
    def _is_collect_rank():
        """Check if this rank should collect dispatch results."""
        return (
            mpu.get_tensor_model_parallel_rank() == 0
            and mpu.get_pipeline_model_parallel_rank() == mpu.get_pipeline_model_parallel_world_size() - 1
            and mpu.get_context_parallel_rank() == 0
        )

    # ---- Config and weight loading ----

    def _load_module_weights(self, module, config_section, is_value_model=False):
        """Load model weights from checkpoint or HF format."""
        if config_section.megatron.use_dist_checkpointing:
            load_mcore_dist_weights(
                module,
                config_section.megatron.dist_checkpointing_path,
                is_value_model=is_value_model,
                prefix=config_section.megatron.dist_checkpointing_prefix,
            )
        else:
            assert self.bridge is not None, "mbridge is required for weight loading"
            local_model_path = get_hf_model_path(self.config)
            if self.vanilla_bridge:
                self.bridge.load_weights(module, local_model_path)
            else:
                kwargs = {}
                if is_value_model:
                    kwargs["allowed_mismatched_params"] = ["output_layer.weight"]
                self.bridge.load_hf_weights(module, local_model_path, **kwargs)

    def _init_hf_config_and_tf_config(
        self,
        model_path,
        tokenizer_or_path,
        dtype,
        override_hf_config,
        override_transformer_config,
        trust_remote_code=False,
        megatron_config=None,
    ):
        from axon.utils import hf_processor
        from axon.utils.hf_model import update_model_config

        # Step 1: initialize the tokenizer
        self.local_path = model_path
        if tokenizer_or_path is None:
            self.tokenizer = hf_tokenizer(self.local_path, trust_remote_code=trust_remote_code)
            self.processor = hf_processor(self.local_path, trust_remote_code=trust_remote_code)
        elif isinstance(tokenizer_or_path, str):
            self.tokenizer = hf_tokenizer(tokenizer_or_path, trust_remote_code=trust_remote_code)
            self.processor = hf_processor(tokenizer_or_path, trust_remote_code=trust_remote_code)
        else:
            self.tokenizer = tokenizer_or_path
            self.processor = tokenizer_or_path

        # Step 2: get the hf
        hf_config = AutoConfig.from_pretrained(self.local_path, trust_remote_code=trust_remote_code)  # nosec B615

        # Step 3: override the hf config
        override_config_kwargs = {
            "bos_token_id": self.tokenizer.bos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        override_config_kwargs.update(override_hf_config)
        self.share_embeddings_and_output_weights = getattr(hf_config, "tie_word_embeddings", False)
        update_model_config(hf_config, override_config_kwargs=override_config_kwargs)
        self.architectures = getattr(hf_config, "architectures", None)
        if self.rank == 0:
            print(f"Model config after override: {hf_config}")

        if megatron_config.use_mbridge:
            # Import monkey patches for custom bridge implementations
            # then patch whatever is currently in the registry
            from axon.models.mbridge import (
                GLM5Bridge,
                Qwen3_5Bridge,
                Qwen3_5MoEBridge,
                Qwen3NextBridge,
            )
            from axon.models.mbridge.export_weights_patch import apply_export_weights_patch

            apply_export_weights_patch(Qwen3NextBridge)
            apply_export_weights_patch(GLM5Bridge)
            apply_export_weights_patch(Qwen3_5Bridge)
            apply_export_weights_patch(Qwen3_5MoEBridge)

            # Fix mbridge for transformers 5.x where rope_theta moved into rope_parameters dict
            from axon.models.mbridge.mbridge_compat import (
                apply_mbridge_qwen25vl_mrope_patch,
                apply_mbridge_rope_theta_patch,
            )

            apply_mbridge_rope_theta_patch()
            # Fix mbridge for transformers 5.x where rope_scaling is on text_config only
            apply_mbridge_qwen25vl_mrope_patch()

            from axon.models.mbridge.qwen2_5_vl import apply_all_qwen2_5_vl_patches

            apply_all_qwen2_5_vl_patches()

            from axon.models.mbridge.deepseek_v3 import apply_deepseek_v3_patch

            apply_deepseek_v3_patch()

            from axon.models.mbridge import AutoBridge
        # todo: remove this line after mcore adopt mbridge 0.15, now for compatibility
        if "attention_backend" in override_transformer_config and isinstance(
            override_transformer_config["attention_backend"], str
        ):
            from megatron.core.transformer.enums import AttnBackend

            override_transformer_config["attention_backend"] = AttnBackend[
                override_transformer_config["attention_backend"]
            ]
        fp16 = dtype == torch.float16
        bf16 = dtype == torch.bfloat16
        if fp16:
            assert megatron_config.use_mbridge, "fp16 mode requires use_mbridge to be True"

        self.provider = None
        self.vanilla_bridge = megatron_config.get("vanilla_mbridge", True)
        if megatron_config.use_mbridge:
            if self.vanilla_bridge:
                from axon.models.mbridge import AutoBridge

                bridge = AutoBridge.from_config(hf_config, dtype=dtype)
                bridge.set_extra_args(**override_transformer_config)
                tf_config = bridge.config
                tf_config.fp16 = fp16
                tf_config.bf16 = bf16
            else:
                from axon.models.megatron_bridge import AutoBridge

                # Use Megatron-Bridge to convert HF config to Megatron config
                bridge = AutoBridge.from_hf_pretrained(self.local_path, trust_remote_code=trust_remote_code)
                # Get Megatron provider and configure it
                provider = bridge.to_megatron_provider(load_weights=False)

                # In case of invalid overrides, we need to make sure some critical params are set correctly
                provider.params_dtype = dtype

                # Pass distributed info
                provider.tensor_model_parallel_size = megatron_config.tensor_model_parallel_size
                provider.pipeline_model_parallel_size = megatron_config.pipeline_model_parallel_size
                provider.expert_model_parallel_size = megatron_config.expert_model_parallel_size
                provider.expert_tensor_parallel_size = megatron_config.expert_tensor_parallel_size
                provider.virtual_pipeline_model_parallel_size = megatron_config.virtual_pipeline_model_parallel_size
                provider.context_parallel_size = megatron_config.context_parallel_size
                provider.sequence_parallel = megatron_config.sequence_parallel

                # Need variable_seq_lengths for sequence packing
                from megatron.core.transformer.enums import AttnBackend

                provider.attention_backend = AttnBackend.flash
                provider.variable_seq_lengths = True
                provider.moe_token_dispatcher_type = "alltoall"
                provider.moe_router_load_balancing_type = "none"

                # Apply transformer config overrides
                for key, value in override_transformer_config.items():
                    setattr(provider, key, value)

                provider.finalize()
                self.provider = provider
                tf_config = None  # Will be set after model creation
            self.bridge = bridge
        else:
            raise AssertionError("mbridge is required. Set megatron_config.use_mbridge=True.")

        if dist.get_rank() == 0:
            if tf_config is not None:
                print(f"TF config: {tf_config}")
        self.hf_config = hf_config
        self.tf_config = tf_config

        # Get PEFT config from lora: sub-config
        from axon.utils.megatron.peft_utils import get_peft_cls

        self.peft_cls = get_peft_cls(
            lora_cfg=self.config.get("lora", None), bridge=self.bridge, provider=self.provider, dtype=dtype
        )

    def _init_config(self, config):
        """Validate config and set up profiler / fused-kernel state."""
        from axon.utils.profiler.profile import Profiler

        if config.megatron.tensor_model_parallel_size == 1:
            config.megatron.sequence_parallel = False

        self.use_torch_profiler = config.profiler.get("tool") == "torch"
        if self.use_torch_profiler:
            self.prof = Profiler(config.profiler, tool_config=config.profiler.get("tool_config", {}).get("torch", {}))
        else:
            self.prof = None

        self.use_fused_kernels = config.get("use_fused_kernels", False)
        if self.use_fused_kernels and not getattr(config, "overlap_moe_expert_parallel_comm", False):
            from axon.utils.rocm_utils import is_rocm

            if is_rocm():
                logger.warning("[ROCm] Skipping fused kernel patching – not supported on HIP")
                self.use_fused_kernels = False
            else:
                from axon.models.mcore.forward.model_forward_fused import patch_fused_forward

                for model in self.module:
                    patch_fused_forward(model)

    def _build_model_optimizer(
        self,
        model_path,
        optimizer_name,
        optimizer_args,
        grad_clip,
        lr_scheduler_type,
        lr_scheduler_args,
        override_hf_config,
        override_transformer_config,
        override_ddp_config=None,
        role="actor",
    ):
        from axon.utils.hf_model import print_model_size

        self._init_hf_config_and_tf_config(
            model_path,
            model_path,
            self.dtype,
            override_hf_config,
            override_transformer_config,
            self.config.get("trust_remote_code", False),
            self.config.megatron,
        )

        is_value_model = self.model is not CausalLM
        if is_value_model:
            self.share_embeddings_and_output_weights = False

        peft_cls = self.peft_cls if not self._forward_only else None
        module, updated_tf_config = self.model.create_model(
            self.config,
            share_embeddings_and_output_weights=self.share_embeddings_and_output_weights,
            tf_config=self.tf_config,
            hf_config=self.hf_config,
            bridge=self.bridge,
            provider=self.provider,
            override_ddp_config=override_ddp_config,
            peft_cls=peft_cls,
        )
        self.tf_config = updated_tf_config
        print(f"{role}_module: {len(module)}")
        load_weight = self.config.megatron.get("load_weight", False)
        if load_weight:
            self._load_module_weights(module, self.config, is_value_model=is_value_model)

        if self.rank == 0:
            print_model_size(module[0])
        log_gpu_memory_usage(f"After {role} module init", logger=logger)

        if self._forward_only:
            return module, None, None, self.hf_config

        from axon.utils.megatron.optimizer import (
            get_megatron_optimizer,
            get_megatron_optimizer_param_scheduler,
            init_megatron_optim_config,
        )

        optim_config_megatron = init_megatron_optim_config(
            optimizer_name=optimizer_name,
            optimizer_args_config=optimizer_args,
            grad_clip=grad_clip,
            lr_scheduler_args=lr_scheduler_args,
            use_distributed_optimizer=self.config.megatron.use_distributed_optimizer,
            fp16=self.dtype == torch.float16,
        )
        optimizer = get_megatron_optimizer(model=module, config=optim_config_megatron)
        optimizer_scheduler = get_megatron_optimizer_param_scheduler(
            optimizer=optimizer,
            lr_scheduler_type=lr_scheduler_type,
            lr_scheduler_args=lr_scheduler_args,
            optimizer_args_config=optimizer_args,
        )

        log_gpu_memory_usage(f"After {role} optimizer init", logger=logger)

        register_megatron_training_hooks(module, optimizer)

        return module, optimizer, optimizer_scheduler, self.hf_config

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        if hasattr(self, "_colocated"):
            set_expandable_segments(False)
        else:
            set_expandable_segments(True)

        if self.config.get("external_lib", None) is not None:
            import importlib

            importlib.import_module(self.config.external_lib)

        from axon.utils.torch.dtypes import PrecisionType

        override_hf_config = OmegaConf.to_container(OmegaConf.create(self.config.get("override_hf_config", {})))
        override_transformer_config = OmegaConf.to_container(
            OmegaConf.create(self.config.megatron.get("override_transformer_config", {}))
        )
        override_ddp_config = OmegaConf.to_container(
            OmegaConf.create(self.config.megatron.get("override_ddp_config", {}))
        )
        self.param_dtype = PrecisionType.to_dtype(self.config.dtype)
        log_gpu_memory_usage(f"Before init {self._mesh_name} model", logger=logger)
        self.dtype = PrecisionType.to_dtype(self.param_dtype)

        # Resolve model path (ref workers may override via model_override)
        model_path = self.config.model_path
        model_override = self.config.get("model_override", None)
        if model_override is not None:
            model_path = model_override.get("path", model_path)
        if self.rank == 0 and self._forward_only:
            print("reference model:", model_path)

        # When forward_only, skip optimizer/scheduler args
        if self._forward_only:
            optimizer_name = None
            optimizer_args = None
            grad_clip = None
            lr_scheduler_type = None
            lr_scheduler_args = None
        else:
            optimizer_name = self.config.optimizer
            optimizer_args = self.config.optimizer_args
            grad_clip = self.config.grad_clip
            lr_scheduler_type = self.config.lr_scheduler
            lr_scheduler_args = self.config.lr_scheduler_args

        (
            module,
            self.optimizer,
            self.optimizer_scheduler,
            self.model_config,
        ) = self._build_model_optimizer(
            model_path=model_path,
            optimizer_name=optimizer_name,
            optimizer_args=optimizer_args,
            grad_clip=grad_clip,
            lr_scheduler_type=lr_scheduler_type,
            lr_scheduler_args=lr_scheduler_args,
            override_hf_config=override_hf_config,
            override_transformer_config=override_transformer_config,
            override_ddp_config=override_ddp_config,
            role=self._mesh_name,
        )
        self.module = module

        if self._is_offload_param:
            offload_megatron_model_to_cpu(self.module)
            log_gpu_memory_usage(f"After offload {self._mesh_name} params during init", logger=logger)
        if self._is_offload_optimizer and self.optimizer is not None:
            offload_megatron_optimizer(self.optimizer)
            log_gpu_memory_usage(f"After offload {self._mesh_name} optimizer during init", logger=logger)

        if self.config.megatron.tensor_model_parallel_size == 1:
            self.config.megatron.sequence_parallel = False

        # Actor-only post-init: fused kernels, flops counter, state manager, weight converter
        if not self._forward_only:
            self._init_config(self.config)
            log_gpu_memory_usage(f"After {self._mesh_name} init", logger=logger)

            self.flops_counter = FlopsCounter(self.model_config, dtype=self.param_dtype)
            self._fb_time_acc = 0.0
            self._fb_seqlens_acc = []
            self._profiler_started = False
            self.state_manager = MegatronStateManager(
                config=self.config,
                model_config=self.model_config,
                transformer_config=self.tf_config,
                role=self.name,
                model=self.module,
                arch=self.architectures[0],
                hf_config=self.hf_config,
                param_dtype=self.param_dtype,
                share_embeddings_and_output_weights=self.share_embeddings_and_output_weights,
                processing_class=self.processor if self.processor is not None else self.tokenizer,
                optimizer=self.optimizer,
                optimizer_scheduler=self.optimizer_scheduler,
                use_distributed_optimizer=self.config.megatron.use_distributed_optimizer,
                bridge=self.bridge,
                provider=self.provider,
                use_dist_checkpointing=self.config.megatron.use_dist_checkpointing,
                peft_cls=self.peft_cls,
            )

            self.layer_name_mapping = {
                "qkv_layer_name": "self_attention.linear_qkv.",
                "gate_proj_layer_name": "linear_fc1.",
            }
            self.weight_converter = None
            assert self.config.megatron.use_mbridge, (
                "get_mcore_weight_converter is no longer supported. Please use mbridge instead."
            )

        get_torch_device().empty_cache()
        log_gpu_memory_usage("After init_model finish", logger=logger)

    # ---- Core training methods ----

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_model(self, include_model=True, include_optimizer=True):
        """Load actor model parameters to GPU if offloaded.

        Optimizer state is NOT loaded here even when include_optimizer=True.
        It is loaded lazily inside optim_step() to keep GPU memory free
        during forward-backward.
        """
        if include_model and self._is_offload_param:
            load_megatron_model_to_gpu(self.module)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def offload_model(self, include_model=True, include_optimizer=True):
        """Offload actor model parameters to CPU and clear cache.

        Optimizer state offloading is handled inside optim_step().
        """
        if include_model and self._is_offload_param:
            offload_megatron_model_to_cpu(self.module)
        aggressive_empty_cache(force_sync=True)

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="trainer"))
    @GPUMemoryLogger(role="forward_backward", logger=logger)
    @DistProfiler.annotate(color="red")
    def forward_backward(self, data: DataProto, loss_fn: str, loss_fn_args: dict):
        """Forward + backward on one mini-batch. No optimizer step."""
        if self._forward_only:
            raise RuntimeError("forward_backward must not be called on forward-only workers")
        if not self._profiler_started and self.use_torch_profiler and self.prof and self.prof.enable:
            self.prof.start()
            self._profiler_started = True

        for chunk in self.module:
            chunk.zero_grad_buffer()

        max_token_len = None
        if self.config.use_dynamic_bsz:
            max_token_len = self.config.max_token_len_per_gpu * self.config.megatron.context_parallel_size

        # Reset peak stats so we measure the per-step peak, not cumulative.
        get_torch_device().reset_peak_memory_stats()

        _fb_start = time.monotonic()
        metric_micro_batch = megatron_forward_backward(
            module=self.module,
            data=data,
            forward_step_fn=self.model.forward_step,
            use_dynamic_bsz=self.config.use_dynamic_bsz,
            micro_batch_size=self.config.micro_batch_size_per_gpu,
            max_token_len=max_token_len,
            tf_config=self.tf_config,
            hf_config=self.hf_config,
            config=self.config,
            temperature=data.meta_info.get("temperature", 1.0),
            use_fused_kernels=self.use_fused_kernels,
            calculate_entropy=True,
            loss_fn=loss_fn,
            loss_fn_args=loss_fn_args,
        )
        self._fb_time_acc += time.monotonic() - _fb_start
        self._fb_seqlens_acc.extend(data.batch["attention_mask"].sum(dim=-1).tolist())

        tp_size = mpu.get_tensor_model_parallel_world_size()
        if tp_size > 1:
            from axon.utils.megatron.utils import unwrap_model

            tp_group = mpu.get_tensor_model_parallel_group()
            for model in self.module:
                model_unwrapped = unwrap_model(model)
                for name, param in model_unwrapped.named_parameters():
                    if param.grad is not None and getattr(param, "is_replicated_vision_weight", False):
                        dist.all_reduce(param.grad, op=dist.ReduceOp.AVG, group=tp_group)

        from axon.utils.print_utils import append_to_dict

        metrics = {}
        metric_micro_batch = metric_micro_batch["output"]
        for metric in metric_micro_batch:
            append_to_dict(metrics, metric)

        metrics = {f"{self.name}/{k}": v for k, v in metrics.items()}
        output = DataProto(meta_info={"metrics": metrics})
        output = output.to("cpu")
        return output

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="trainer"))
    @DistProfiler.annotate(color="red", role="optim_step")
    def optim_step(self, step_lr: bool = False):
        """Optimizer step. Optionally steps the LR scheduler.

        Optimizer state is loaded from CPU just-in-time and offloaded back
        immediately after the step, keeping it off GPU during forward-backward.
        """
        if self._forward_only:
            raise RuntimeError("optim_step must not be called on forward-only workers")

        # Load optimizer state from CPU just before the step.
        if self._is_offload_optimizer and self.optimizer is not None:
            load_megatron_optimizer(self.optimizer)

        _optim_start = time.monotonic()
        update_successful, grad_norm, num_zeros_in_grad = self.optimizer.step()
        self.optimizer.zero_grad()

        # Offload optimizer state back to CPU immediately after the step.
        if self._is_offload_optimizer and self.optimizer is not None:
            offload_megatron_optimizer(self.optimizer)
        self._fb_time_acc += time.monotonic() - _optim_start

        device = get_torch_device()
        # Driver-level memory: captures everything including cuBLAS workspaces,
        # CUDA graphs, and non-PyTorch allocations (e.g., vLLM cumem pools).
        gpu_free, gpu_total = device.mem_get_info()
        gpu_used_gb = (gpu_total - gpu_free) / (1024**3)
        mem_stats = device.memory_stats()
        largest_free_gb = mem_stats.get("largest_free_block", gpu_free) / (1024**3)
        metrics = {
            "grad_norm": grad_norm,
            "perf/max_memory_allocated_gb": device.max_memory_allocated() / (1024**3),
            "perf/max_memory_reserved_gb": device.max_memory_reserved() / (1024**3),
            # Current allocation (not cumulative max) — tracks baseline creep
            "perf/memory_allocated_gb": device.memory_allocated() / (1024**3),
            "perf/memory_reserved_gb": device.memory_reserved() / (1024**3),
            # Driver-level GPU usage — the real number, includes cuBLAS/CUDA graphs/cumem
            "perf/gpu_used_gb": gpu_used_gb,
            "perf/gpu_total_gb": gpu_total / (1024**3),
            # Largest contiguous free block — tracks fragmentation
            "perf/largest_free_block_gb": largest_free_gb,
            "perf/cpu_memory_used_gb": psutil.virtual_memory().used / (1024**3),
        }

        if not update_successful:
            import warnings

            warnings.warn(
                f"[Rank {dist.get_rank()}] Optimizer step failed, skipping update.",
                stacklevel=2,
            )
            metrics["update_skipped"] = 1.0
        else:
            metrics["update_skipped"] = 0.0

        if step_lr:
            from axon.utils.megatron.optimizer import get_megatron_last_lr

            metrics["lr"] = get_megatron_last_lr(self.optimizer)
            self.optimizer_scheduler.step(1)

            if self._fb_time_acc > 0 and self._fb_seqlens_acc:
                estimated_flops, promised_flops = self.flops_counter.estimate_flops(
                    self._fb_seqlens_acc, self._fb_time_acc
                )
                metrics["perf/mfu"] = estimated_flops / promised_flops
            self._fb_time_acc = 0.0
            self._fb_seqlens_acc = []

        if self._profiler_started and self.use_torch_profiler and self.prof and self.prof.enable:
            self.prof.step()
            if step_lr:
                self.prof.stop_and_save()
                self.prof.stop_trace()
                self._profiler_started = False

        get_torch_device().empty_cache()
        metrics = {f"{self.name}/{k}": v for k, v in metrics.items()}
        return DataProto(meta_info={"metrics": metrics}).to("cpu")

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="trainer"))
    @GPUMemoryLogger(role="forward", logger=logger)
    @DistProfiler.annotate(color="blue", role="forward")
    def forward(self, data: DataProto):
        """Forward-only pass to compute log-probs, entropy, or value predictions.

        Note: load/offload of model params is handled by the caller
        (TrainingClient.forward wraps this in load_model_context).
        """
        config = self.config
        use_dynamic_bsz = config.forward_use_dynamic_bsz
        micro_batch_size = config.forward_micro_batch_size_per_gpu
        max_token_len = None
        if use_dynamic_bsz:
            max_token_len = config.forward_max_token_len_per_gpu * config.megatron.context_parallel_size
        calculate_entropy = not self._forward_only

        # Do NOT move data to GPU here — _prepare_mini_batch (inside
        # megatron_forward_backward) handles the GPU transfer, broadcast,
        # and micro-batch splitting. Moving the full batch to GPU here
        # wastes memory (all rows on GPU at once instead of per micro-batch).
        with torch.no_grad():
            output = megatron_forward_backward(
                module=self.module,
                data=data,
                forward_step_fn=self.model.forward_step,
                forward_only=True,
                use_dynamic_bsz=use_dynamic_bsz,
                micro_batch_size=micro_batch_size,
                max_token_len=max_token_len,
                tf_config=self.tf_config,
                hf_config=self.hf_config,
                config=config,
                temperature=data.meta_info.get("temperature", 1.0),
                use_fused_kernels=config.get("use_fused_kernels", False),
                calculate_entropy=calculate_entropy,
            )
            tensors = self.model.forward_output_fn(
                output, data, use_dynamic_bsz=use_dynamic_bsz, calculate_entropy=calculate_entropy
            )

        output = DataProto.from_dict(tensors=tensors)
        output = output.to("cpu")

        get_torch_device().empty_cache()
        return output

    # ---- State management ----

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_state(self, checkpoint_path, save_mode=None):
        if self._forward_only:
            return

        if checkpoint_path is None:
            if self._is_offload_param:
                offload_megatron_model_to_cpu(self.module)
            if self._is_offload_optimizer and self.optimizer is not None:
                offload_megatron_optimizer(self.optimizer)
            log_gpu_memory_usage(
                f"After offload {self._mesh_name} params and optimizer during load_state", logger=logger
            )
            return

        if self._is_offload_param:
            load_megatron_model_to_gpu(self.module)
        self.state_manager.load_state(local_path=checkpoint_path, save_mode=save_mode)
        if self._is_offload_param:
            offload_megatron_model_to_cpu(self.module)
        if self._is_offload_optimizer and self.optimizer is not None:
            offload_megatron_optimizer(self.optimizer)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_state(self, checkpoint_path, global_step=0, max_ckpt_to_keep=None, save_mode=None, async_save=None):
        if self._forward_only:
            return

        if self._is_offload_param:
            load_megatron_model_to_gpu(self.module)

        self.state_manager.save_state(
            local_path=checkpoint_path,
            global_step=global_step,
            max_ckpt_to_keep=max_ckpt_to_keep,
            save_mode=save_mode,
            async_save=async_save,
        )
        dist.barrier()

        if self._is_offload_param:
            offload_megatron_model_to_cpu(self.module)

    # ---- Profiling ----

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def start_profile(self, **kwargs) -> None:
        self.profiler.start(**kwargs)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def stop_profile(self) -> None:
        self.profiler.stop()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def dump_memory_snapshot(self, tag: str = "manual", sub_dir: str = None) -> None:
        if hasattr(self, "profiler") and hasattr(self.profiler, "_impl"):
            try:
                if hasattr(self.profiler._impl, "sampler"):
                    out_dir = OmegaConf.select(self.config, "profiler.save_path") or "."
                    self.profiler._impl.sampler.dump_memory_snapshot(out_dir=out_dir, tag=tag, sub_dir=sub_dir)
            except Exception as e:
                logger.warning(f"Failed to dump memory snapshot: {e}")
