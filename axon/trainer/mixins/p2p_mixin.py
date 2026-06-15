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

"""P2P weight transfer mixins for trainer workers.

TrainerP2PMixin provides the shared sender-side methods
(get_node_ip_and_free_port, connect_trainer_to_sampler).

FSDPTrainerP2PMixin and MegatronTrainerP2PMixin provide the
framework-specific methods (get_parameter_mapping,
send_trainer_to_sampler_weights, construct_send_ops_and_buffers).
"""

import json
import logging
import os
import socket
from collections import defaultdict
from pathlib import Path

import ray
import torch
import torch.distributed as dist
from torch.distributed import P2POp

from axon.controller.decorator import Dispatch, register
from axon.controller.ray import RayWorkerGroup

# Use patched batch_isend_irecv to avoid _coalescing_manager state corruption
# when multiple process groups have concurrent P2P ops (see p2p_fix.py).
from axon.monkey_patches.torch.p2p_fix import patched_batch_isend_irecv
from axon.monkey_patches.torch.p2p_fix import patched_batch_isend_irecv as batch_isend_irecv
from axon.utils.fsdp.utils import get_fsdp_full_state_dict, load_fsdp_model_to_gpu, offload_fsdp_model_to_cpu
from axon.utils.hf_model import convert_param_name, convert_weight_keys
from axon.utils.megatron.utils import load_megatron_model_to_gpu, offload_megatron_model_to_cpu
from axon.utils.memory_utils import aggressive_empty_cache
from axon.utils.p2p.distributed import init_trainer_sampler_process_group
from axon.utils.p2p.param_mapper import convert_hf_to_vllm_param_name, get_megatron_param_metadata
from axon.utils.p2p.routing_table import FusedParameterMetadata, ParameterMetadata, RankMapping
from axon.utils.profiler import log_gpu_memory_usage
from axon.utils.torch import get_device_id, get_device_name

logger = logging.getLogger(__name__)


# Helper function to slice a tensor for TP sharding
def slice_for_tp(tensor, shard_dim, shard_idx, tp_size):
    """
    Slice or replicate tensor for the given TP configuration.

    Args:
        tensor: Input tensor (already sharded in Megatron)
        shard_dim: Dimension along which to slice/replicate
        shard_idx: Target shard index for sampler
        tp_size_ratio: sampler_tp / actor_tp
    """
    if shard_dim == -1:
        return tensor  # No sharding needed (replicated parameter)
    # One actor shard -> multiple sampler shards
    # E.g., actor has full shard, need to split it further
    shard_size = tensor.shape[shard_dim] // tp_size
    start = shard_idx * shard_size
    end = start + shard_size

    if shard_dim == 0:
        return tensor[start:end, ...]  # Column-parallel
    elif shard_dim == 1:
        return tensor[:, start:end, ...]  # Row-parallel, makes a new copy.
    else:
        raise ValueError(f"Unsupported shard dimension: {shard_dim}")


def _compute_tp_degree(meta):
    """Compute the tensor-parallel degree from parameter metadata."""
    if meta.split_dim == -1:
        return 1
    return meta.full_param_shape[meta.split_dim] // meta.param_shape[meta.split_dim]


def _write_p2p_debug_log(entries, log_filename, rank=None, print_prefix=None):
    """Write P2P operation log to file for debugging."""
    home_dir = str(Path.home())
    log_dir = os.environ.get("P2P_LOG_DIR", os.path.join(home_dir, "p2p_logs"))
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, log_filename)
    with open(log_file, "w") as f:
        json.dump(entries, f, indent=2)
    if rank == 0:
        prefix = print_prefix or "[P2P Check] Saved log to"
        print(f"{prefix} {log_file}")


class TrainerP2PMixin:
    """P2P weight transfer mixin for TrainerWorker (sender side).

    Provides shared methods: get_node_ip_and_free_port, connect_trainer_to_sampler.

    Subclasses must implement:
    - get_parameter_mapping()
    - send_trainer_to_sampler_weights(routing_table=None)
    """

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def get_node_ip_and_free_port(self):
        if self._forward_only:
            raise RuntimeError("P2P methods must not be called on forward-only workers")
        if self.rank != 0:
            return None, None

        def get_local_ip():
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                s.connect(("8.8.8.8", 80))
                local_ip = s.getsockname()[0]
                s.close()
                return local_ip
            except Exception:
                return "127.0.0.1"

        ip = get_local_ip()
        with socket.socket() as sock:
            sock.bind(("", 0))
            port = sock.getsockname()[1]
        return ip, port

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def connect_trainer_to_sampler(
        self,
        sampler_wg: "RayWorkerGroup",
        init_method: str,
        group_name: str = "bridge_pg",
        backend: str = "nccl",
        group_attribute_name: str = "bridge_pg",
    ) -> None:
        """Connect sampler samplers and actors, e.g. initialize the process group between them."""
        if self._forward_only:
            raise RuntimeError("P2P methods must not be called on forward-only workers")
        self.sampler_workers = sampler_wg._workers

        device_name = get_device_name()
        total_world_size = self.world_size + len(self.sampler_workers)

        if self.rank == 0:
            refs = []
            for worker in self.sampler_workers:
                if hasattr(worker, "sampler_connect_sampler_to_trainer"):
                    method = worker.sampler_connect_sampler_to_trainer
                else:
                    method = worker.connect_sampler_to_trainer
                refs.append(
                    method.remote(
                        init_method=init_method,
                        rank_offset=self.world_size,
                        world_size=total_world_size,
                        backend=backend,
                        group_name=group_name,
                        group_attribute_name=group_attribute_name,
                    )
                )

        setattr(
            self,
            group_attribute_name,
            init_trainer_sampler_process_group(
                backend=backend,
                init_method=init_method,
                world_size=total_world_size,
                rank=self.rank,
                group_name=group_name,
                device_id=torch.device(device_name, get_device_id()),
            ),
        )

        torch.cuda.set_device(get_device_id())
        group = getattr(self, group_attribute_name, None)
        assert group is not None
        dist.barrier(group=group)
        _warmup = torch.ones(1, device=get_device_id())
        dist.all_reduce(_warmup, op=dist.ReduceOp.SUM, group=group)
        del _warmup
        if self.rank == 0:
            ray.get(refs)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def get_parameter_mapping(self):
        raise NotImplementedError

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def send_trainer_to_sampler_weights(self, routing_table=None):
        raise NotImplementedError


class FSDPTrainerP2PMixin(TrainerP2PMixin):
    """FSDP-specific P2P weight transfer methods."""

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def get_parameter_mapping(self):
        """Get the parameter keys (not tensors) residing on this rank."""
        if self._forward_only:
            raise RuntimeError("P2P methods must not be called on forward-only workers")
        rank_mapping = RankMapping(rank=self.rank, params=[])

        is_fsdp_rank_0 = self.device_mesh["fsdp"].get_local_rank() == 0
        if not is_fsdp_rank_0:
            return rank_mapping

        param_keys = []
        base_model = getattr(self.module_fsdp, "_fsdp_wrapped_module", self.module_fsdp)

        # VL models: HF wraps all submodules under a top-level "model." (e.g.
        # "model.visual.X", "model.language_model.X"). _checkpoint_conversion_mapping
        # handles language model renaming but not the vision encoder. Detect VL wrapper
        # by checking if any param starts with "model.visual" or "model.language_model",
        # then strip "model." from unconverted params to match vLLM's naming convention.
        all_names = [n for n, _ in self.module_fsdp.named_parameters()]
        is_vl_model = any(n.startswith("model.visual") or n.startswith("model.language_model") for n in all_names)

        for name, tensor in self.module_fsdp.named_parameters():
            local_shape, global_shape = tensor._local_tensor.shape, tensor.shape
            ratio_list = [global_shape[i] / local_shape[i] for i in range(len(global_shape))]
            max_ratio = max(ratio_list)
            split_dim = -1 if max_ratio == 1 else ratio_list.index(max_ratio)

            # Convert internal name to checkpoint format (handles _checkpoint_conversion_mapping)
            checkpoint_name = convert_param_name(name, base_model)
            # VL models: HF wraps submodules under "model." (e.g. "model.visual.X",
            # "model.language_model.X"). Normalize to match vLLM's convention:
            #   "model.language_model.X" → "model.X" (vLLM nests LLM under language_model.model.X,
            #                                          then strips language_model. → model.X)
            #   "model.visual.X"         → "visual.X" (vLLM uses bare visual.X)
            if is_vl_model:
                if checkpoint_name.startswith("model.language_model."):
                    checkpoint_name = "model." + checkpoint_name[len("model.language_model.") :]
                elif checkpoint_name.startswith("model.visual"):
                    checkpoint_name = checkpoint_name[len("model.") :]
            sampler_param_name = convert_hf_to_vllm_param_name(checkpoint_name)
            param_metadata = ParameterMetadata(
                param_name=sampler_param_name,
                original_param_name=name,
                param_shape=global_shape,
                full_param_shape=global_shape,
                param_dtype=tensor.dtype,
                split_dim=-1,
                split_idx=self.device_mesh["fsdp"].get_local_rank(),
            )
            param_keys.append(param_metadata)

        param_groups = defaultdict(list)
        for pm in param_keys:
            key = (pm.param_name, pm.split_dim, pm.split_idx)
            param_groups[key].append(pm)

        fused_param_keys = []
        for key, params in param_groups.items():
            if len(params) > 1:
                param_name, split_dim, split_idx = key
                param_dtype = params[0].param_dtype
                param_shape_list = list(params[0].param_shape)
                full_param_shape_list = list(params[0].full_param_shape)

                for param in params[1:]:
                    param_shape_list[0] += param.param_shape[0]
                    full_param_shape_list[0] += param.full_param_shape[0]

                fused_param = FusedParameterMetadata(
                    param_name=param_name,
                    param_shape=tuple(param_shape_list),
                    param_dtype=param_dtype,
                    full_param_shape=tuple(full_param_shape_list),
                    split_dim=split_dim,
                    split_idx=split_idx,
                    params=params,
                )
                fused_param_keys.append(fused_param)
            else:
                fused_param_keys.append(params[0])

        rank_mapping.params = fused_param_keys
        return rank_mapping

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def send_trainer_to_sampler_weights(self, routing_table=None):
        """Asynchronously push actor weights to sampler workers."""
        if self._forward_only:
            raise RuntimeError("P2P methods must not be called on forward-only workers")
        assert hasattr(self, "bridge_pg"), "Bridge process group is not initialized"

        aggressive_empty_cache(force_sync=True)

        if routing_table:
            self.routing_table = routing_table

        transfers_for_actor_rank = self.routing_table.get_transfers_for_actor_rank(self.rank)

        actor_offset = self.world_size

        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.module_fsdp)

        full_actor_params = get_fsdp_full_state_dict(
            self.module_fsdp,
            offload_to_cpu=False,
            rank0_only=True,
        )

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.module_fsdp)

        if not transfers_for_actor_rank:
            del full_actor_params
            return

        base_model = getattr(self.module_fsdp, "_fsdp_wrapped_module", self.module_fsdp)
        full_actor_params = convert_weight_keys(full_actor_params, base_model)

        tensor_cache = {}
        ops = []

        all_send_tensors = []
        for sampler_rank, param_dicts in sorted(transfers_for_actor_rank.items()):
            for param_dict in sorted(param_dicts, key=lambda x: (x["actor"].param_name, x["actor"].split_idx)):
                actor_meta = param_dict["actor"]
                sampler_meta = param_dict["sampler"]

                # Compute sampler TP degree from metadata (no TP in FSDP actor, so actor_tp=1)
                if sampler_meta.split_dim != -1:
                    sampler_tp = (
                        sampler_meta.full_param_shape[sampler_meta.split_dim]
                        // sampler_meta.param_shape[sampler_meta.split_dim]
                    )
                else:
                    sampler_tp = 1

                cache_key = (actor_meta.param_name, sampler_meta.split_dim, sampler_meta.split_idx)
                if actor_meta.split_dim != -1 and sampler_meta.split_dim != -1:
                    assert actor_meta.split_dim == sampler_meta.split_dim, (
                        f"Actor and sampler split dimensions must match. Actor: {actor_meta.split_dim}, Sampler: {sampler_meta.split_dim}"
                    )

                if cache_key in tensor_cache:
                    send_tensor = tensor_cache[cache_key]
                else:
                    if isinstance(actor_meta, FusedParameterMetadata):
                        individual_parts = []
                        for param in actor_meta.params:
                            # original_param_name is in HF internal format; full_actor_params
                            # has been converted to checkpoint format via convert_weight_keys.
                            ckpt_key = convert_param_name(param.original_param_name, model=base_model)
                            full_tensor = full_actor_params[ckpt_key]
                            sliced = slice_for_tp(
                                full_tensor, sampler_meta.split_dim, sampler_meta.split_idx, sampler_tp
                            )
                            individual_parts.append(sliced)
                        send_tensor = torch.cat(individual_parts, dim=0)
                    else:
                        ckpt_key = convert_param_name(actor_meta.original_param_name, model=base_model)
                        send_tensor = full_actor_params[ckpt_key]
                        send_tensor = slice_for_tp(
                            send_tensor, sampler_meta.split_dim, sampler_meta.split_idx, sampler_tp
                        )

                    actor_dtype = (
                        actor_meta.params[0].param_dtype
                        if isinstance(actor_meta, FusedParameterMetadata)
                        else actor_meta.param_dtype
                    )
                    if actor_dtype != sampler_meta.param_dtype:
                        send_tensor = send_tensor.to(dtype=sampler_meta.param_dtype, non_blocking=True)

                    tensor_cache[cache_key] = send_tensor

                all_send_tensors.append((sampler_rank, send_tensor))

        for sampler_rank, send_tensor in all_send_tensors:
            ops.append(
                P2POp(
                    op=torch.distributed.isend,
                    tensor=send_tensor,
                    group_peer=(sampler_rank + actor_offset),
                    group=self.bridge_pg,
                )
            )

        if ops:
            torch.cuda.synchronize()
            reqs = batch_isend_irecv(ops)
            for req in reqs:
                req.wait()

        del full_actor_params
        del tensor_cache
        del all_send_tensors
        aggressive_empty_cache(force_sync=True)


class MegatronTrainerP2PMixin(TrainerP2PMixin):
    """Megatron-specific P2P weight transfer methods."""

    def _get_actor_parameter_mapping(self):
        """Build parameter metadata list for Megatron actor parameters on this rank.

        The default flow is: mcore name → mbridge HF name →
        convert_hf_to_vllm_param_name. Bridges that need to deviate from
        this (e.g. models whose HF layout and vLLM layout aren't simple
        fusion inverses) can expose two optional hooks:

          * ``p2p_mcore_name_to_vllm(mcore_name, hf_name, meta) -> str``
            — replaces ``convert_hf_to_vllm_param_name`` for this bridge.
          * ``p2p_extra_params(metadata_list, actor_parameters)``
            — returns the (possibly extended) metadata list, used to
            register additional entries that alias existing tensors
            (e.g. Gemma4's k_eq_v v_proj aliasing k_proj).
        """
        assert self.bridge, "get_mcore_weight_converter is no longer supported. Please use mbridge instead."
        convert_name = getattr(self.bridge, "p2p_mcore_name_to_vllm", None)
        extra_params = getattr(self.bridge, "p2p_extra_params", None)

        # Get metadata for all parameters on this rank (includes sharding info)
        self.actor_parameters = {}
        metadata_list = get_megatron_param_metadata(
            model=self.module, transformer_config=self.tf_config, model_config=self.model_config
        )
        skipped = []
        converted_metadata_list = []
        for meta in metadata_list:
            meta_name = meta["name"]
            # Mbridge weight converter. Extra state_dict entries (registered buffers)
            # may not have mbridge mappings — skip them since they don't need P2P sync.
            try:
                hf_names = self.bridge._weight_name_mapping_mcore_to_hf(meta_name)
            except (NotImplementedError, KeyError):
                skipped.append(meta_name)
                continue
            hf_name = hf_names[0]

            if convert_name is not None:
                vllm_name = convert_name(meta_name, hf_name, meta)
            else:
                # Default: vLLM's fused parameters = mcore's fused parameters.
                vllm_name = convert_hf_to_vllm_param_name(hf_name)

            self.actor_parameters[vllm_name] = meta["tensor"]

            meta["sampler_name"] = vllm_name
            converted_metadata_list.append(meta)
        if skipped:
            logger.info(f"[P2P] Skipped {len(skipped)} unmappable state_dict entries: {skipped[:5]}")
        metadata_list = converted_metadata_list

        if extra_params is not None:
            metadata_list = extra_params(metadata_list, self.actor_parameters)

        param_keys = []
        for meta in metadata_list:
            metadata_dict = {}
            if "linear_qkv.weight" in meta["name"] or "linear_qkv.bias" in meta["name"]:
                # Use vision head counts for vision encoder params, language model counts otherwise
                vision_config = getattr(self.model_config, "vision_config", None)
                if vision_config and "visual" in meta.get("sampler_name", ""):
                    num_heads = getattr(vision_config, "num_heads", getattr(vision_config, "num_attention_heads", 1))
                    metadata_dict["num_attention_heads"] = num_heads
                    metadata_dict["num_key_value_heads"] = num_heads  # Vision uses MHA
                else:
                    # Transformers 5.x VL models use composite configs where text
                    # attributes live on model_config.text_config, not directly.
                    text_cfg = getattr(self.model_config, "text_config", self.model_config)
                    metadata_dict["num_attention_heads"] = text_cfg.num_attention_heads
                    metadata_dict["num_key_value_heads"] = text_cfg.num_key_value_heads

            param_metadata = ParameterMetadata(
                param_name=meta["sampler_name"],
                original_param_name=meta["name"],
                param_shape=meta["local_shape"],
                full_param_shape=meta["full_shape"],
                param_dtype=meta["dtype"],
                split_dim=meta["split_dim"],
                split_idx=meta["split_idx"],
                expert_idx=meta["expert_idx"],
                metadata=metadata_dict,
            )
            param_keys.append(param_metadata)
        return param_keys

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def get_parameter_mapping(self):
        """Get the parameter keys (not tensors) residing on this rank."""
        if self._forward_only:
            raise RuntimeError("P2P methods must not be called on forward-only workers")
        rank_mapping = RankMapping(rank=self.rank, params=[])
        rank_mapping.params = self._get_actor_parameter_mapping()
        return rank_mapping

    def construct_send_ops_and_buffers(self, routing_table=None):
        """Construct send operations and weight buffers for weight transfer."""
        actor_offset = self.world_size
        transfers_for_actor_rank = routing_table.get_transfers_for_actor_rank(self.rank)

        if not transfers_for_actor_rank:
            return [], []

        tensor_cache, all_send_tensors, p2p_log = {}, [], []
        ops, buffers = [], []

        for sampler_rank, param_dicts in sorted(transfers_for_actor_rank.items()):
            for param_dict in sorted(param_dicts, key=lambda x: (x["actor"].param_name, x["actor"].split_idx)):
                actor_meta = param_dict["actor"]
                sampler_meta = param_dict["sampler"]

                actor_tp = _compute_tp_degree(actor_meta)
                sampler_tp = _compute_tp_degree(sampler_meta)
                if "qkv_proj" in sampler_meta.param_name or ".attn.qkv." in sampler_meta.param_name:
                    sampler_tp = sampler_meta.metadata["tp_size"]

                if sampler_meta.split_dim != -1:
                    assert actor_tp % sampler_tp == 0 or sampler_tp % actor_tp == 0, (
                        f"Actor TP {actor_tp} and sampler TP {sampler_tp} must be multiples of each other."
                    )

                cache_key = (actor_meta.param_name, sampler_meta.split_dim, sampler_meta.split_idx)

                is_gate_up_fused = (
                    "gate_up_proj" in actor_meta.param_name or "w13_weight" in actor_meta.param_name
                ) and actor_tp != sampler_tp
                is_qkv_fused = "qkv_proj" in actor_meta.param_name or ".attn.qkv." in actor_meta.param_name

                if cache_key in tensor_cache:
                    send_tensor = tensor_cache[cache_key]
                else:
                    actor_tensor = self.actor_parameters[actor_meta.param_name]

                    send_tensor = actor_tensor
                    if not is_gate_up_fused and not is_qkv_fused:
                        if actor_tp < sampler_tp:
                            shards_per_actor = sampler_tp // actor_tp
                            local_idx = sampler_meta.split_idx % shards_per_actor
                            send_tensor = slice_for_tp(
                                actor_tensor, sampler_meta.split_dim, local_idx, shards_per_actor
                            )
                            if not send_tensor.is_contiguous():
                                contiguous_send_tensor = send_tensor.contiguous()
                                buffers.append((send_tensor, contiguous_send_tensor))
                                send_tensor = contiguous_send_tensor

                    actor_dtype = actor_meta.param_dtype
                    if actor_dtype != sampler_meta.param_dtype:
                        send_tensor = send_tensor.to(dtype=sampler_meta.param_dtype, non_blocking=True)

                    assert send_tensor.is_contiguous(), "Actor tensor must be contiguous for efficient transfer"

                    tensor_cache[cache_key] = send_tensor

                if is_qkv_fused:
                    num_q_heads = actor_meta.metadata["num_attention_heads"]
                    num_kv_heads = actor_meta.metadata["num_key_value_heads"]
                    head_dim = actor_meta.full_param_shape[0] // (num_q_heads + 2 * num_kv_heads)

                    assert sampler_meta.split_dim < 1 and actor_meta.split_dim < 1, (
                        "Gate and up must be split by 0-dim."
                    )

                    actor_q_heads = num_q_heads // actor_tp
                    actor_kv_heads = num_kv_heads // actor_tp

                    if actor_tp < sampler_tp and actor_kv_heads * actor_tp < sampler_tp:
                        replicas_per_kv = sampler_tp // (actor_kv_heads * actor_tp)
                        which_kv_global = sampler_meta.split_idx // replicas_per_kv
                        group_offset = which_kv_global % actor_kv_heads
                        local_idx = sampler_meta.split_idx % replicas_per_kv
                        kv_heads_to_send = 1
                        q_per_kv_send = actor_q_heads // actor_kv_heads // replicas_per_kv
                        q_per_kv_megatron = actor_q_heads // actor_kv_heads
                        group_size = (q_per_kv_megatron + 2) * head_dim
                    elif actor_tp < sampler_tp:
                        shards_per_actor = sampler_tp // actor_tp
                        local_idx_kv = sampler_meta.split_idx % shards_per_actor
                        local_idx = 0
                        kv_heads_to_send = actor_kv_heads // shards_per_actor
                        q_per_kv_send = actor_q_heads // actor_kv_heads
                        group_offset = local_idx_kv * kv_heads_to_send
                        q_per_kv_megatron = q_per_kv_send
                        group_size = (q_per_kv_send + 2) * head_dim
                    else:
                        local_idx = 0
                        q_per_kv_send = actor_q_heads // actor_kv_heads
                        kv_heads_to_send = actor_kv_heads
                        group_offset = 0
                        q_per_kv_megatron = q_per_kv_send
                        group_size = (q_per_kv_send + 2) * head_dim

                    for group_idx in range(kv_heads_to_send):
                        group_start = (group_offset + group_idx) * group_size
                        q_start = group_start + local_idx * q_per_kv_send * head_dim
                        k_start = group_start + q_per_kv_megatron * head_dim

                        q_subtensor = send_tensor[q_start : q_start + q_per_kv_send * head_dim]
                        k_subtensor = send_tensor[k_start : k_start + head_dim]
                        v_subtensor = send_tensor[k_start + head_dim : k_start + 2 * head_dim]

                        all_send_tensors.append((sampler_rank, q_subtensor))
                        all_send_tensors.append((sampler_rank, k_subtensor))
                        all_send_tensors.append((sampler_rank, v_subtensor))

                elif (
                    "gate_up_proj" in actor_meta.param_name or "w13_weight" in actor_meta.param_name
                ) and actor_tp != sampler_tp:
                    gate_up_size = send_tensor.shape[0] // 2
                    gate_tensor = send_tensor[:gate_up_size]
                    up_tensor = send_tensor[gate_up_size:]
                    assert sampler_meta.split_dim < 1 and actor_meta.split_dim < 1, (
                        "Gate and up must be split by 0-dim."
                    )

                    if actor_tp < sampler_tp:
                        shards_per_actor = sampler_tp // actor_tp
                        local_idx = sampler_meta.split_idx % shards_per_actor
                        gate_tensor = slice_for_tp(gate_tensor, sampler_meta.split_dim, local_idx, shards_per_actor)
                        up_tensor = slice_for_tp(up_tensor, sampler_meta.split_dim, local_idx, shards_per_actor)

                    all_send_tensors.append((sampler_rank, gate_tensor))
                    all_send_tensors.append((sampler_rank, up_tensor))

                else:
                    all_send_tensors.append((sampler_rank, send_tensor))

        for sampler_rank, send_tensor in all_send_tensors:
            if self.debug_p2p_mismatch:
                p2p_log.append(
                    {
                        "op_type": "send",
                        "peer_rank": sampler_rank + actor_offset,
                        "shape": list(send_tensor.shape),
                        "dtype": str(send_tensor.dtype),
                        "numel": send_tensor.numel(),
                    }
                )
            ops.append(
                P2POp(
                    op=dist.isend,
                    tensor=send_tensor,
                    group_peer=(sampler_rank + actor_offset),
                    group=self.bridge_pg,
                )
            )

        if p2p_log:
            _write_p2p_debug_log(
                p2p_log,
                f"actor_rank_{self.rank}_send.json",
                rank=self.rank,
                print_prefix="[P2P Check] Saved send log to",
            )

        return ops, buffers

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def send_trainer_to_sampler_weights(self, routing_table=None):
        """Asynchronously push actor weights to sampler workers."""
        if self._forward_only:
            raise RuntimeError("P2P methods must not be called on forward-only workers")
        assert hasattr(self, "bridge_pg"), "Bridge process group is not initialized"

        aggressive_empty_cache(force_sync=True)

        if self._is_offload_param:
            load_megatron_model_to_gpu(self.module)
            log_gpu_memory_usage("After load actor params for weight transfer", logger=logger)

        if not self.ops:
            if routing_table:
                self.routing_table = routing_table
            self.ops, self.buffers = self.construct_send_ops_and_buffers(routing_table=self.routing_table)
        else:
            if self.offload_p2p_buffer:
                device_id = get_device_id()
                for _, dest_tensor in self.buffers:
                    dest_tensor.data = dest_tensor.data.to(device_id, non_blocking=True)

            for source_tensor, dest_tensor in self.buffers:
                dest_tensor.copy_(source_tensor, non_blocking=True)

        if self.ops:
            reqs = patched_batch_isend_irecv(self.ops)
            for req in reqs:
                req.wait()

        if self.offload_p2p_buffer:
            for _, dest_tensor in self.buffers:
                dest_tensor.data = dest_tensor.data.to("cpu", non_blocking=True)

        if self._is_offload_param:
            offload_megatron_model_to_cpu(self.module)
            log_gpu_memory_usage("After offload actor params after weight transfer", logger=logger)
