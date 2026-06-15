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
"""FSDP-based decoupled TrainerWorker."""

import datetime
import json
import logging
import os
import time
import warnings
from contextlib import nullcontext
from dataclasses import asdict

import numpy as np
import psutil
import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf
from safetensors.torch import save_file
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

try:
    # for torch 2.5+
    from torch.distributed.tensor import DTensor
except ImportError:
    from torch.distributed._tensor import DTensor

from axon.controller.decorator import Dispatch, make_nd_compute_dataproto_dispatch_fn, register
from axon.core.worker import Worker

# Apply FSDP monkey patch to enable rank0_only state dict without cpu_offload
# This must be done before any FSDP state dict operations
from axon.monkey_patches.fsdp.base import apply_fsdp_monkey_patch
from axon.monkey_patches.transformers.monkey_patch import apply_monkey_patch
from axon.protocol import DataProto
from axon.utils.fsdp.activation_offload import enable_activation_offloading
from axon.utils.fsdp.optimizer import build_optimizer
from axon.utils.fsdp.utils import (
    CPUOffloadPolicy,
    FSDPModule,
    MixedPrecisionPolicy,
    apply_fsdp2,
    create_device_mesh,
    fsdp2_clip_grad_norm_,
    fsdp2_load_full_state_dict,
    fsdp_version,
    get_fsdp_wrap_policy,
    get_init_weight_context_manager,
    get_shard_placement_fn,
    get_sharding_strategy,
    init_fn,
    layered_summon_lora_params,
    load_fsdp_model_to_gpu,
    load_fsdp_optimizer,
    offload_fsdp_model_to_cpu,
    offload_fsdp_optimizer,
    parse_mixed_precision,
)
from axon.utils.hf_model import (
    apply_lora_to_module,
    get_vl_model_vision_tower,
    update_model_config,
)
from axon.utils.import_utils import import_external_libs
from axon.utils.memory_utils import aggressive_empty_cache
from axon.utils.print_utils import append_to_dict
from axon.utils.profiler import DistProfiler, DistProfilerExtension, init_profiler_on_worker, log_gpu_memory_usage
from axon.utils.profiler.flops_counter import FlopsCounter
from axon.utils.scheduler_utils import build_lr_scheduler
from axon.utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from axon.utils.state.fsdp_state_manager import FSDPStateManager
from axon.utils.torch import (
    get_device_id,
    get_device_name,
    get_nccl_backend,
    get_torch_device,
    set_expandable_segments,
)
from axon.utils.torch.dtypes import PrecisionType
from axon.utils.ulysses import FSDPUlyssesShardingManager, build_ulysses_device_mesh, register_ulysses_dispatch

device_name = get_device_name()

apply_fsdp_monkey_patch()


logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("AXON_LOGGING_LEVEL", "WARN"))


class TrainerWorker(Worker, DistProfilerExtension):
    """FSDP-backed trainer worker — runs forward + backward + optimiser step.

    One of the two trainer backends in Axon (the other is the Megatron worker
    in ``axon/trainer/megatron_trainer.py``). Use ``strategy: fsdp`` or
    ``strategy: fsdp2`` in the recipe to select this backend.

    Mode-specific behaviour is layered on at runtime via mixins —
    ``FSDPSyncTrainerMixin`` / ``FSDPTrainerP2PMixin`` / ``AsyncTrainerMixin``
    are composed by ``axon/driver/train_agent_ppo.py`` so the
    ``{Sync, Async} × {Hybrid, Disaggregated}`` matrix is one set of classes.

    With ``forward_only=False`` (default) this is a full training worker.
    With ``forward_only=True`` it becomes a forward-only worker with no
    optimizer or gradient state — used for ``ref`` and ``reward_model`` roles.

    For hybrid-engine mode, compose with ``SamplerWorker`` via
    ``fuse_worker_cls({"actor": TrainerWorker, "sampler": SamplerWorker})``.

    This class is designed to be composed with other workers via
    ``fuse_worker_cls()``, which discovers ``@register``-decorated methods
    automatically.

    Model-specific behavior (``create_model``, ``forward_fn``,
    ``forward_keys``, ``forward_backward_keys``,
    ``forward_backward_fn``) is supplied via a *model* class
    (default: ``CausalLM``) whose static methods are called by the
    worker.  Pass ``model=ValueModel`` to get critic behavior.
    """

    def __init__(self, config: DictConfig, model=None, name: str = "trainer", **kwargs):
        Worker.__init__(self)

        from axon.models.fsdp_models import CausalLM

        self.model = model if model is not None else CausalLM
        self.config = config
        self.name = name
        self._forward_only = config.get("forward_only", False)

        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                backend=f"cpu:gloo,{get_device_name()}:{get_nccl_backend()}",
                rank=int(os.environ.get("RANK", 0)),
                world_size=int(os.environ.get("WORLD_SIZE", 1)),
                timeout=datetime.timedelta(seconds=600),
                init_method=os.environ.get("DIST_INIT_METHOD", None),
            )

        world_size = torch.distributed.get_world_size()
        self.device_mesh = create_device_mesh(world_size=world_size, fsdp_size=config.fsdp.fsdp_size)
        self.ulysses_sequence_parallel_size = config.fsdp.ulysses_sequence_parallel_size
        self.ulysses_device_mesh = build_ulysses_device_mesh(world_size, self.ulysses_sequence_parallel_size)
        register_ulysses_dispatch(self, "trainer", self.ulysses_device_mesh)
        self.ulysses_sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)

        # LoRA
        self._lora_rank = 0 if self._forward_only else config.get("lora", {}).get("rank", 0)
        self._is_lora = not self._forward_only and (
            config.get("lora", {}).get("adapter_path") is not None or self._lora_rank > 0
        )

        init_profiler_on_worker(self, config.get("profiler", {}))

        self._is_offload_param = config.get("param_offload", False)
        self._is_offload_optimizer = not self._forward_only and config.get("optimizer_offload", False)

        # Normalize micro_batch_size from global to per-GPU
        dp_size = self.device_mesh.size() // self.ulysses_sequence_parallel_size
        for key in ("micro_batch_size", "forward_micro_batch_size"):
            if config.get(key, None) is not None:
                config[key] //= dp_size
                config[f"{key}_per_gpu"] = config[key]

        # Save the training RNG state so sampler_mode/sleep can restore it
        self.torch_random_states = get_torch_device().get_rng_state()

        # P2P state
        self.offload_p2p_buffer = config.get("offload_p2p_buffer", False)
        self.ops = []
        self.buffers = []
        self.routing_table = None

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        from torch.distributed.fsdp import CPUOffload, MixedPrecision

        from axon.utils.hf_model import print_model_size

        import_external_libs(self.config.get("external_lib", None))

        set_expandable_segments(not hasattr(self, "_colocated"))

        fsdp_config = self.config.fsdp
        self.use_remove_padding = fsdp_config.get("use_remove_padding", False)
        role = self.name

        # ---- Tokenizer / processor ----
        from axon.utils import hf_processor, hf_tokenizer

        model_path = self.config.model_path
        model_override = self.config.get("model_override", None)
        if model_override is not None:
            model_path = model_override.get("path", model_path)
        trust_remote_code = self.config.get("trust_remote_code", False)
        self.tokenizer = hf_tokenizer(model_path, trust_remote_code=trust_remote_code)
        self.processor = hf_processor(model_path, trust_remote_code=trust_remote_code)

        # ---- Model config ----
        from transformers import AutoConfig

        override_hf_config = OmegaConf.to_container(OmegaConf.create(self.config.get("override_hf_config", {})))
        torch_dtype = PrecisionType.to_dtype(fsdp_config.get("model_dtype", "fp32"))
        attn_implementation = override_hf_config.pop("attn_implementation", None)
        ulysses_sp_size = self.ulysses_sequence_parallel_size

        # The Ulysses SP all-to-all is patched onto ``_flash_attention_forward``
        # (the FA-family integration in transformers).  When ``attn_implementation``
        # falls back to anything else (sdpa / flex_attention / eager), the patch
        # is silently skipped and per-rank attention is computed only over each
        # rank's slice — producing wrong logits for every layer.  Fail loudly.
        if ulysses_sp_size > 1:
            if attn_implementation is None:
                attn_implementation = "flash_attention_2"
            elif not attn_implementation.startswith("flash_attention_"):
                raise ValueError(
                    f"ulysses_sequence_parallel_size>1 requires attn_implementation in "
                    f"{{flash_attention_2, flash_attention_3, flash_attention_4}}; got "
                    f"{attn_implementation!r}. The SP all-to-all is only patched on the "
                    f"FA family. For models with head_dim>256 (e.g. Gemma4 full-attention "
                    f"layers at 512), install FA3 or set ulysses_sequence_parallel_size=1."
                )

        model_config = AutoConfig.from_pretrained(  # nosec B615
            model_path,
            trust_remote_code=trust_remote_code,
            attn_implementation=attn_implementation,
        )

        # Patch VisionAttention for Ulysses SP compatibility
        if ulysses_sp_size > 1 and hasattr(model_config, "vision_config"):
            model_config.vision_config._attn_implementation = "eager"

        # Patch for kimi-vl
        if getattr(model_config, "model_type", None) == "kimi_vl":
            model_config.text_config.topk_method = "greedy"

        # ---- Apply config overrides (token IDs + user overrides) ----
        override_config_kwargs = {
            "bos_token_id": self.tokenizer.bos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        override_config_kwargs.update(override_hf_config)
        update_model_config(model_config, override_config_kwargs=override_config_kwargs)

        # ---- Create model inside init context ----
        init_context = get_init_weight_context_manager(
            use_meta_tensor=not model_config.tie_word_embeddings, mesh=self.device_mesh
        )

        with init_context(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            module = self.model.create_model(
                model_config,
                model_path=model_path,
                torch_dtype=torch_dtype,
                trust_remote_code=trust_remote_code,
                attn_implementation=attn_implementation,
            )

            if fsdp_config.get("use_liger", False):
                from liger_kernel.transformers.monkey_patch import _apply_liger_kernel_to_instance

                _apply_liger_kernel_to_instance(model=module)

            apply_monkey_patch(
                model=module,
                use_remove_padding=self.use_remove_padding,
                ulysses_sp_size=ulysses_sp_size,
                use_fused_kernels=fsdp_config.get("use_fused_kernels", False),
                fused_kernels_backend=(
                    fsdp_config.fused_kernel_options.get("impl_backend", None)
                    if fsdp_config.get("fused_kernel_options", None) is not None
                    else None
                ),
            )

            module.to(torch_dtype)

            # When activation offloading is enabled with gradient checkpointing,
            # the ActivationHandler applies per-layer checkpointing itself.
            # Skip model-level gradient_checkpointing_enable to avoid double
            # checkpointing which corrupts the offload handler's bookkeeping.
            _enable_gc = fsdp_config.get("enable_gradient_checkpointing", False)
            _enable_act_offload = not self._forward_only and fsdp_config.get("enable_activation_offload", False)
            if _enable_act_offload:
                # The activation-offload outer checkpoint must use ``use_reentrant=True``
                # so save_for_backward fires under the offload hook.  Reentrant ckpt
                # captures non-tensor kwargs (like Gemma4's ``_SharedKVStatesCarrier``)
                # opaquely, so the K/V tensors held inside the carrier never enter
                # the autograd graph.  Donor layers therefore miss the gradient
                # contribution from receiver layers, the optimizer applies a partial
                # update, and step 2+ forward overflows to NaN.  Until we either move
                # the carrier to a tensor-typed kwarg or selectively skip offload for
                # KV-shared receiver layers, refuse the combo.
                _text_config = (
                    model_config.get_text_config() if hasattr(model_config, "get_text_config") else model_config
                )
                if getattr(_text_config, "num_kv_shared_layers", 0) > 0:
                    raise ValueError(
                        "enable_activation_offload=True is incompatible with KV-shared "
                        "models (e.g. Gemma4-E2B/E4B with num_kv_shared_layers>0). The "
                        "reentrant checkpoint required by activation offload does not "
                        "preserve the autograd graph through the shared-K/V carrier, "
                        "leading to silently-bad gradients on donor layers and NaN at "
                        "step 2+.  Set enable_activation_offload=False, or use a "
                        "Gemma4 variant with num_kv_shared_layers=0 (31B / 26B-A4B)."
                    )
            if _enable_gc and not _enable_act_offload:
                module.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        self.model_config = model_config

        if self._is_lora:
            module = apply_lora_to_module(module, self.config.lora, role)

        self.use_orig_params = fsdp_config.get("use_orig_params", False)
        if fsdp_config.get("freeze_vision_tower", False):
            vision_tower = get_vl_model_vision_tower(module)
            if vision_tower is not None:
                vision_tower.requires_grad_(False)
                self.use_orig_params = True
                if self.rank == 0:
                    print(f"[{role} model] Vision tower is set to not trainable.")
            elif self.rank == 0:
                print(f"[{role} model] No vision tower found.")

        torch.distributed.barrier()
        if self.rank == 0:
            print_model_size(module)
        log_gpu_memory_usage(f"After init {role} from HF AutoModel", logger=logger)

        # ---- FSDP wrapping ----
        param_dtype, reduce_dtype, buffer_dtype = parse_mixed_precision(
            fsdp_config, default_param_dtype=PrecisionType.to_dtype(self.config.get("dtype", "bfloat16"))
        )
        auto_wrap_policy = get_fsdp_wrap_policy(
            module=module,
            config=fsdp_config.get("wrap_policy", None),
            is_lora=self._is_lora,
        )
        if self.rank == 0:
            print(f"wrap_policy: {auto_wrap_policy}")

        fsdp_strategy = self.config.strategy
        self._fsdp_strategy = fsdp_strategy  # stored for forward_backward dp_replicas logic
        sharding_strategy = get_sharding_strategy(self.device_mesh)

        if fsdp_strategy == "fsdp":
            self.module_fsdp = FSDP(
                module,
                cpu_offload=CPUOffload(offload_params=True) if self._forward_only else None,
                param_init_fn=init_fn,
                auto_wrap_policy=auto_wrap_policy,
                device_id=get_device_id(),
                sharding_strategy=sharding_strategy,
                mixed_precision=MixedPrecision(
                    param_dtype=param_dtype,
                    reduce_dtype=reduce_dtype,
                    buffer_dtype=buffer_dtype,
                ),
                sync_module_states=True,
                device_mesh=self.device_mesh,
                use_orig_params=self.use_orig_params,
                forward_prefetch=fsdp_config.get("forward_prefetch", False),
            )
        elif fsdp_strategy == "fsdp2":
            assert CPUOffloadPolicy is not None, "PyTorch >= 2.4 required for FSDP2"
            mp_policy = MixedPrecisionPolicy(
                param_dtype=param_dtype,
                reduce_dtype=reduce_dtype,
                cast_forward_inputs=True,
            )
            if not self._forward_only and fsdp_config.offload_policy:
                cpu_offload = CPUOffloadPolicy(pin_memory=True)
                self._is_offload_param = False
                self._is_offload_optimizer = False
            else:
                cpu_offload = CPUOffloadPolicy(pin_memory=True) if self._forward_only else None

            # When use_meta_tensor=True (tie_word_embeddings=False), only rank 0
            # has the full model weights.  We must capture state_dict before
            # sharding and broadcast from rank 0 after.
            # When use_meta_tensor=False (tie_word_embeddings=True, e.g. Gemma4),
            # ALL ranks already loaded identical weights from the checkpoint.
            # fully_shard() shards correctly in-place — no broadcast or reload
            # needed.  Skipping saves ~62GB CPU per rank and avoids OOM from
            # set_model_state_dict temporarily materializing full tensors on GPU.
            use_meta_tensor = not getattr(model_config, "tie_word_embeddings", False)
            if use_meta_tensor:
                full_state = module.state_dict()

            apply_fsdp2(
                module,
                {
                    "mesh": self.device_mesh,
                    "mp_policy": mp_policy,
                    "offload_policy": cpu_offload,
                    "reshard_after_forward": fsdp_config.reshard_after_forward,
                    "shard_placement_fn": get_shard_placement_fn(fsdp_size=self.device_mesh.shape[-1]),
                },
                fsdp_config,
            )

            if use_meta_tensor:
                fsdp2_load_full_state_dict(module, full_state, self.device_mesh, cpu_offload)
                del full_state
            # else: all ranks loaded identical weights; fully_shard() already
            # sharded and moved to GPU in-place — nothing more to do.
            self.module_fsdp = module
        else:
            raise NotImplementedError(f"Unsupported FSDP strategy: {fsdp_strategy!r}")

        if not self._forward_only and fsdp_config.get("enable_activation_offload", False):
            enable_gc = fsdp_config.get("enable_gradient_checkpointing", False)
            enable_activation_offloading(self.module_fsdp, fsdp_strategy, enable_gc)

        log_gpu_memory_usage(f"After {role} FSDP init", logger=logger)

        # ---- Optimizer ----
        if self._forward_only:
            self.optimizer = self.lr_scheduler = None
        else:
            optimizer_args = self.config.optimizer_args
            self.optimizer = build_optimizer(self.module_fsdp.parameters(), self.config.optimizer, optimizer_args)
            lr = optimizer_args.get("lr", 1.0) if optimizer_args is not None else 1.0
            self.lr_scheduler = build_lr_scheduler(
                self.optimizer,
                self.config.lr_scheduler,
                self.config.lr_scheduler_args,
                lr,
                self.rank,
            )
            log_gpu_memory_usage(f"After {role} optimizer init", logger=logger)

        self.param_dtype = PrecisionType.to_dtype(self.config.get("dtype", "bfloat16"))
        if self.param_dtype == torch.float16:
            from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

            self.scaler = ShardedGradScaler(growth_interval=400)
        else:
            self.scaler = None

        # ---- Post-init ----
        self.module = self.module_fsdp
        if fsdp_version(self.module_fsdp) == 1:
            self.module = self.module_fsdp._fsdp_wrapped_module

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.module_fsdp)
            log_gpu_memory_usage(f"After offload {role} model during init", logger=logger)
        if self._is_offload_optimizer and self.optimizer is not None:
            offload_fsdp_optimizer(optimizer=self.optimizer)
            log_gpu_memory_usage(f"After offload {role} optimizer during init", logger=logger)

        self._forward_kwargs = dict(
            use_remove_padding=self.use_remove_padding,
            use_fused_kernels=fsdp_config.get("use_fused_kernels", False),
            ulysses_sp_size=self.ulysses_sequence_parallel_size,
            device_name=get_device_name(),
            param_dtype=self.param_dtype,
            entropy_checkpointing=fsdp_config.get("entropy_checkpointing", False),
            entropy_from_logits_with_chunking=fsdp_config.get("entropy_from_logits_with_chunking", False),
            use_torch_compile=fsdp_config.get("use_torch_compile", True),
        )

        if not self._forward_only:
            self.flops_counter = FlopsCounter(self.model_config, dtype=self.param_dtype)
            self._fb_time_acc = 0.0
            self._fb_seqlens_acc = []
            self.state_manager = FSDPStateManager(
                model=self.module_fsdp,
                optimizer=self.optimizer,
                lr_scheduler=self.lr_scheduler,
                processing_class=self.processor or self.tokenizer,
            )

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_model(self, include_model=True, include_optimizer=True):
        """Load actor model parameters to GPU if offloaded.

        Optimizer state is NOT loaded here even when include_optimizer=True.
        Loading optimizer state (Adam m/v tensors, 2x model size) alongside
        model params would double the GPU memory footprint during the
        forward-backward pass, which doesn't need optimizer state at all.
        Optimizer state is instead loaded lazily inside optim_step().
        """
        if include_model and self._is_offload_param:
            load_fsdp_model_to_gpu(self.module_fsdp)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def offload_model(self, include_model=True, include_optimizer=True):
        """Offload actor model parameters to CPU and clear cache.

        Optimizer state offloading is handled inside optim_step() immediately
        after the optimizer step, so it is never on GPU during backward.
        The include_optimizer parameter is accepted for API compatibility but
        has no effect here (optimizer offload is managed by optim_step).
        """
        if include_model and self._is_offload_param:
            offload_fsdp_model_to_cpu(self.module_fsdp)
            log_gpu_memory_usage(f"After offload {self.name} model", logger=logger)
        aggressive_empty_cache(force_sync=True)

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="trainer"))
    @DistProfiler.annotate(color="blue", role="forward")
    def forward(self, data: DataProto):
        """Forward-only pass on a batch. No loss computation or backward.

        Note: load/offload of model params is handled by the caller
        (TrainingClient.forward wraps this in load_model_context).
        """
        self.module_fsdp.eval()

        is_lora = data.meta_info.pop("is_lora", False)
        adapter_ctx = self.module_fsdp.disable_adapter() if is_lora and self._is_lora else nullcontext()

        assert "temperature" in data.meta_info, "temperature must be provided in meta_info"
        temperature = data.meta_info["temperature"]
        use_dynamic_bsz = data.meta_info.get("use_dynamic_bsz", self.config.forward_use_dynamic_bsz)
        sp = self.ulysses_sequence_parallel_size

        batch_keys, non_tensor_keys = self.model.forward_keys(data)
        data = data.select(batch_keys=batch_keys, non_tensor_batch_keys=non_tensor_keys)

        meta_info = {
            **self._forward_kwargs,
            "temperature": temperature,
            "calculate_entropy": not self._forward_only,
        }

        with self.ulysses_sharding_manager, adapter_ctx:
            data = data.to("cpu")

            if use_dynamic_bsz:
                max_token_len = data.meta_info.get("max_token_len", self.config.forward_max_token_len_per_gpu) * sp
                micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
            else:
                fwd_mbs = data.meta_info.get("micro_batch_size", self.config.forward_micro_batch_size_per_gpu)
                micro_batches = data.split(fwd_mbs)

            results = {}
            for micro_batch in micro_batches:
                micro_batch = micro_batch.to(get_device_id())
                with torch.no_grad():
                    mb_result = self.model.forward_fn(
                        self.module_fsdp,
                        {**micro_batch.batch, **micro_batch.non_tensor_batch},
                        meta_info,
                    )
                for k, v in mb_result.items():
                    results.setdefault(k, []).append(v)

            tensors = {k: torch.concat(v, dim=0) for k, v in results.items() if v[0] is not None}
            if use_dynamic_bsz:
                tensors = {k: restore_dynamic_batch(v, batch_idx_list) for k, v in tensors.items()}
            output = DataProto.from_dict(tensors=tensors)

        output = output.to("cpu")
        self._reshard_if_needed()

        get_torch_device().empty_cache()
        return output

    def _reshard_if_needed(self):
        if self.world_size > 1:
            v = fsdp_version(self.module_fsdp)
            if v == 1:
                self.module_fsdp._handle.reshard(True)
            elif v == 2:
                self.module_fsdp.reshard()

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="trainer"))
    @DistProfiler.annotate(color="red", role="forward_backward")
    def forward_backward(self, data: DataProto, loss_fn: str, loss_fn_args: dict):
        """Forward + backward on one mini-batch with gradient accumulation. No optimizer step."""
        if self._forward_only:
            raise RuntimeError("forward_backward must not be called on forward-only workers")
        self.module_fsdp.train()

        # NOTE: padding rows are NOT filtered here. balance_batch() ensures all
        # DP ranks receive the same number of rows (real + padding). Padding rows
        # have zeroed response_mask, so they contribute zero loss.
        # Filtering them would cause unequal micro-batch counts across ranks,
        # deadlocking FSDP2 gradient all-reduce.

        batch_keys, non_tensor_keys = self.model.forward_backward_keys(data)
        data = data.select(batch_keys=batch_keys, non_tensor_batch_keys=non_tensor_keys)

        sp = self.ulysses_sequence_parallel_size
        # dp_replicas cancels the framework's automatic gradient reduction:
        # FSDP2 (fully_shard): ReduceOp.AVG (÷D) → loss × D to cancel.
        # FSDP1 (FSDP): ReduceOp.SUM → loss × 1 (no compensation needed).
        fsdp_is_avg = getattr(self, "_fsdp_strategy", "fsdp2") == "fsdp2"
        dp_replicas = (torch.distributed.get_world_size() // sp) if fsdp_is_avg else 1
        meta_info = {
            **self._forward_kwargs,
            "loss_fn": loss_fn,
            "loss_fn_args": loss_fn_args,
            "temperature": data.meta_info["temperature"],
            "data_size": len(data),
            "dp_replicas": dp_replicas,
        }

        with self.ulysses_sharding_manager:
            data = data.to("cpu")

            if self.config.use_dynamic_bsz:
                micro_batches, _ = prepare_dynamic_batch(
                    data,
                    max_token_len=self.config.max_token_len_per_gpu * sp,
                )
            else:
                micro_batches = data.split(self.config.micro_batch_size_per_gpu)

            meta_info["n_micro_batches"] = len(micro_batches)

            # Defragment before the micro-batch loop: return cached blocks to
            # the CUDA driver so the FSDP all-gather can find contiguous memory.
            # This is safe (only releases already-freed blocks) and prevents
            # fragmentation from the old_log_prob forward pass causing OOM here.
            get_torch_device().empty_cache()

            # Reset peak stats so we measure the per-step peak, not cumulative.
            get_torch_device().reset_peak_memory_stats()

            metrics = {}
            _fb_start = time.monotonic()
            for micro_batch in micro_batches:
                micro_batch = micro_batch.to(get_device_id())
                loss, mb_metrics = self.model.forward_backward_fn(self.module_fsdp, micro_batch, meta_info)
                (self.scaler.scale(loss) if self.scaler else loss).backward()
                append_to_dict(metrics, mb_metrics)

            # Weighted mean by micro-batch size
            sizes = [len(mb) for mb in micro_batches]
            total = sum(sizes) or 1
            aggregated_metrics = {}
            for k, vals in metrics.items():
                if len(vals) == len(sizes):
                    aggregated_metrics[k] = sum(v * s for v, s in zip(vals, sizes, strict=False)) / total
                else:
                    aggregated_metrics[k] = float(np.mean(vals)) if vals else 0.0

            self._fb_time_acc += time.monotonic() - _fb_start
            self._fb_seqlens_acc.extend(data.batch["attention_mask"].sum(dim=-1).tolist())
            aggregated_metrics = {f"{self.name}/{k}": v for k, v in aggregated_metrics.items()}
            return DataProto(meta_info={"metrics": aggregated_metrics}).to("cpu")

    def _clip_grad_norm(self, max_norm):
        """Clip gradients and return the total norm (handles FSDP1, FSDP2, plain)."""
        if isinstance(self.module_fsdp, FSDP):
            grad_norm = self.module_fsdp.clip_grad_norm_(max_norm=max_norm)
        elif isinstance(self.module_fsdp, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.module_fsdp.parameters(), max_norm=max_norm)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.module_fsdp.parameters(), max_norm=max_norm)
        return grad_norm.full_tensor() if isinstance(grad_norm, DTensor) else grad_norm

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="trainer"))
    @DistProfiler.annotate(color="red", role="optim_step")
    def optim_step(self, step_lr: bool = False):
        """Gradient clip + optimizer step + zero grad. Optionally steps the LR scheduler.

        Optimizer state (Adam m/v) is loaded from CPU to GPU here (just-in-time)
        and offloaded back immediately after the step.  This keeps optimizer state
        off GPU during forward/backward, halving peak GPU memory for fp32 training.
        """
        if self._forward_only:
            raise RuntimeError("optim_step must not be called on forward-only workers")

        _optim_start = time.monotonic()

        grad_clip = self.config.grad_clip
        assert grad_clip is not None

        # Load optimizer state from CPU just before the step.
        # At step 1 the state dict is empty and this is a no-op; after the
        # step() call below Adam initialises m/v on GPU, then we offload them.
        if self._is_offload_optimizer and self.optimizer is not None:
            load_fsdp_optimizer(optimizer=self.optimizer, device_id=get_device_id())

        if self.scaler:
            self.scaler.unscale_(self.optimizer)

        grad_norm = self._clip_grad_norm(grad_clip)

        if not torch.isfinite(grad_norm) or grad_norm >= self.config.fsdp.grad_norm_threshold:
            warnings.warn(
                f"[Rank {self.rank}] grad_norm not finite or exceeds threshold, skipping: {grad_norm}",
                stacklevel=2,
            )
            self.optimizer.zero_grad()
            if self.scaler:
                self.scaler.update()
            if self._is_offload_optimizer and self.optimizer is not None:
                offload_fsdp_optimizer(self.optimizer)
            self._fb_time_acc += time.monotonic() - _optim_start
            return DataProto(meta_info={"metrics": {}}).to("cpu")

        if self.scaler:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        self.optimizer.zero_grad()

        # Offload optimizer state (Adam m/v) back to CPU immediately after
        # the step.  This keeps the ~2x-model-size optimizer buffers off GPU
        # while the next forward/backward is running.
        if self._is_offload_optimizer and self.optimizer is not None:
            offload_fsdp_optimizer(self.optimizer)

        self._fb_time_acc += time.monotonic() - _optim_start

        device = get_torch_device()
        # Driver-level memory: captures everything including cuBLAS workspaces,
        # CUDA graphs, and non-PyTorch allocations (e.g., vLLM cumem pools).
        gpu_free, gpu_total = device.mem_get_info()
        gpu_used_gb = (gpu_total - gpu_free) / (1024**3)
        # Fragmentation metric: largest contiguous free block.
        # If this shrinks across steps while total free stays the same,
        # the allocator is fragmenting and large allocations will fail.
        mem_stats = device.memory_stats()
        largest_free_gb = mem_stats.get("largest_free_block", gpu_free) / (1024**3)
        metrics = {
            "grad_norm": grad_norm.detach().item(),
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

        if step_lr:
            lr = self.lr_scheduler.get_last_lr()[0]
            metrics["lr"] = lr.item() if torch.is_tensor(lr) else lr
            self.lr_scheduler.step()

            if self._fb_time_acc > 0 and self._fb_seqlens_acc:
                estimated, promised = self.flops_counter.estimate_flops(self._fb_seqlens_acc, self._fb_time_acc)
                metrics["perf/mfu"] = estimated / promised
            self._fb_time_acc = 0.0
            self._fb_seqlens_acc = []

        metrics = {f"{self.name}/{k}": v for k, v in metrics.items()}
        return DataProto(meta_info={"metrics": metrics}).to("cpu")

    # ---- P2P methods (get_parameter_mapping, send_trainer_to_sampler_weights) ----
    # Provided by FSDPTrainerP2PMixin.

    # ---- State management ----

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_state(self, local_path, global_step=0, max_ckpt_to_keep=None, save_mode=None, async_save=None):
        from axon.utils.logger import log_with_rank

        if self._forward_only:
            logger.info("save_state called on a non-training worker (forward_only=True); skipping.")
            return

        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.module_fsdp)

        self.state_manager.save_state(
            local_path=local_path,
            global_step=global_step,
            max_ckpt_to_keep=max_ckpt_to_keep,
            save_mode=save_mode,
            async_save=async_save,
        )
        dist.barrier()

        if self._is_lora and hasattr(getattr(self, "module", self.module_fsdp), "peft_config"):
            lora_save_path = os.path.join(local_path, "lora_adapter")
            peft_model = getattr(self, "module", self.module_fsdp)
            peft_config = {}
            if dist.get_rank() == 0:
                os.makedirs(lora_save_path, exist_ok=True)
                peft_config = asdict(peft_model.peft_config.get("default", {}))
                peft_config["task_type"] = peft_config["task_type"].value
                peft_config["peft_type"] = peft_config["peft_type"].value
                peft_config["target_modules"] = list(peft_config["target_modules"])
            try:
                if fsdp_version(self.module_fsdp) > 0:
                    self.module_fsdp = self.module_fsdp.to(get_device_name())
                    lora_params = layered_summon_lora_params(self.module_fsdp)
                    if dist.get_rank() == 0:
                        save_file(lora_params, os.path.join(lora_save_path, "adapter_model.safetensors"))
                        with open(os.path.join(lora_save_path, "adapter_config.json"), "w", encoding="utf-8") as f:
                            json.dump(peft_config, f, ensure_ascii=False, indent=4)
            except Exception as e:
                log_with_rank(
                    f"Save LoRA Adapter Error ({e})", rank=dist.get_rank(), logger=logger, log_only_rank_0=True
                )

            dist.barrier()
            log_with_rank(
                f"[rank-{self.rank}]: Saved LoRA adapter to: {lora_save_path}",
                rank=dist.get_rank(),
                logger=logger,
                log_only_rank_0=True,
            )
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.module_fsdp)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_state(self, local_path, save_mode=None):
        if self._forward_only:
            logger.info("load_state called on a non-training worker (forward_only=True); skipping.")
            return

        if local_path is None:
            if self._is_offload_param:
                offload_fsdp_model_to_cpu(self.module_fsdp)
            if self._is_offload_optimizer:
                offload_fsdp_optimizer(self.optimizer)
            return

        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.module_fsdp)

        self.state_manager.load_state(local_path=local_path, save_mode=save_mode)

        dist.barrier()

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.module_fsdp)

        if self._is_offload_optimizer:
            offload_fsdp_optimizer(self.optimizer)

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
            except Exception:
                pass
