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

"""P2P weight transfer mixins for sampler workers (receiver side).

SamplerP2PMixin provides shared receiver-side methods
(connect_sampler_to_trainer, set_actor_wg, receive_sampler_to_trainer_weights).

FSDPSamplerP2PMixin and MegatronSamplerP2PMixin provide the
framework-specific methods (get_parameter_mapping, construct_recv_ops_and_buffers).
"""

import logging
import os

import torch
import torch.distributed as dist
from torch.distributed import P2POp

from axon.controller.decorator import Dispatch, register
from axon.controller.ray import RayWorkerGroup

# Use patched batch_isend_irecv to avoid _coalescing_manager state corruption
# when multiple process groups have concurrent P2P ops (see p2p_fix.py).
from axon.monkey_patches.torch.p2p_fix import patched_batch_isend_irecv
from axon.trainer.mixins.p2p_mixin import _compute_tp_degree, _write_p2p_debug_log, slice_for_tp
from axon.utils.memory_utils import aggressive_empty_cache
from axon.utils.p2p.distributed import init_trainer_sampler_process_group
from axon.utils.p2p.routing_table import ParameterMetadata, RankMapping
from axon.utils.torch import get_device_id, get_device_name

logger = logging.getLogger(__name__)

DEBUG_P2P_MISMATCH = os.environ.get("AXON_DEBUG_P2P_MISMATCH", "0") == "1"


class SamplerP2PMixin:
    """P2P weight transfer mixin for SamplerWorker (receiver side).

    Provides shared methods: connect_sampler_to_trainer, set_actor_wg,
    receive_sampler_to_trainer_weights.

    Subclasses must implement:
    - get_parameter_mapping()
    - construct_recv_ops_and_buffers(routing_table)
    """

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def connect_sampler_to_trainer(
        self,
        init_method: str,
        rank_offset: int,
        world_size: int,
        backend: str = "nccl",
        group_name: str = "bridge_pg",
        group_attribute_name: str = "bridge_pg",
    ) -> torch.distributed.ProcessGroup:
        """Initialize the bridge process group between this sampler and the actor workers."""
        device_name = get_device_name()
        setattr(
            self,
            group_attribute_name,
            init_trainer_sampler_process_group(
                backend=backend,
                init_method=init_method,
                world_size=world_size,
                rank=self.rank + rank_offset,
                group_name=group_name,
                device_id=torch.device(device_name, get_device_id()),
            ),
        )
        group = getattr(self, group_attribute_name, None)
        assert group
        torch.cuda.set_device(get_device_id())
        torch.distributed.barrier(group=group)
        _warmup = torch.ones(1, device=get_device_id())
        torch.distributed.all_reduce(_warmup, op=dist.ReduceOp.SUM, group=group)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def set_actor_wg(self, actor_wg: "RayWorkerGroup"):
        """Save all actor workers. Used for sampler to trainer data sending."""
        self.actor_workers = actor_wg._workers
        self.actor_world_size = actor_wg.world_size

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def get_parameter_mapping(self):
        raise NotImplementedError

    def construct_recv_ops_and_buffers(self, routing_table=None):
        raise NotImplementedError

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def receive_sampler_to_trainer_weights(self, routing_table=None):
        raise NotImplementedError


class FSDPSamplerP2PMixin(SamplerP2PMixin):
    """FSDP-specific P2P weight transfer methods for SamplerWorker."""

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def get_parameter_mapping(self):
        """Get the parameter keys (not tensors) residing on this rank."""
        from vllm.model_executor.layers.linear import (
            ColumnParallelLinear,
            MergedColumnParallelLinear,
            QKVParallelLinear,
            RowParallelLinear,
        )
        from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding

        from axon.utils.hf_model import (
            convert_param_name,
            get_checkpoint_conversion_mapping,
        )

        rank_mapping = RankMapping(rank=self.rank, params=[])
        sampler_name = self.config.name
        param_keys = []
        if sampler_name == "vllm":
            vllm_model = self.sampler.inference_engine.worker.model_runner.model
            tp_size = self.sampler_device_mesh["infer_tp"].size()

            _vllm_to_orig_prefix = {}
            mapper = getattr(type(vllm_model), "hf_to_vllm_mapper", None)
            if mapper is not None and hasattr(mapper, "orig_to_new_prefix"):
                for orig, new in mapper.orig_to_new_prefix.items():
                    if new not in _vllm_to_orig_prefix or len(orig) < len(_vllm_to_orig_prefix[new]):
                        _vllm_to_orig_prefix[new] = orig
            ckpt_mapping = get_checkpoint_conversion_mapping(
                self.config.model_path, trust_remote_code=self.config.get("trust_remote_code", False)
            )

            def _vllm_name_to_checkpoint(name: str) -> str:
                for prefix in sorted(_vllm_to_orig_prefix, key=len, reverse=True):
                    if name.startswith(prefix):
                        name = _vllm_to_orig_prefix[prefix] + name[len(prefix) :]
                        break
                name = convert_param_name(name, mapping=ckpt_mapping)
                # VL models: vLLM wraps language model under "language_model." prefix
                # (e.g. "language_model.model.layers.0..."). FSDP actor uses HF names
                # without this prefix ("model.layers.0..."). Strip to match.
                if name.startswith("language_model."):
                    name = name[len("language_model.") :]
                return name

            self.sampler_parameters = {}
            for module_name, module in vllm_model.named_modules():
                for param_name, param in module.named_parameters(recurse=False):
                    full_param_name = f"{module_name}.{param_name}" if module_name else param_name
                    checkpoint_name = _vllm_name_to_checkpoint(full_param_name)

                    split_dim = -1
                    full_param_shape = param.shape

                    input_dim = getattr(param, "input_dim", None)
                    output_dim = getattr(param, "output_dim", None)

                    if isinstance(module, ColumnParallelLinear | QKVParallelLinear | MergedColumnParallelLinear):
                        if output_dim is not None:
                            split_dim = output_dim
                            full_param_shape = list(param.shape)
                            full_param_shape[split_dim] = full_param_shape[split_dim] * tp_size
                            full_param_shape = tuple(full_param_shape)

                    elif isinstance(module, RowParallelLinear):
                        if input_dim is not None:
                            split_dim = input_dim
                            full_param_shape = list(param.shape)
                            full_param_shape[split_dim] = full_param_shape[split_dim] * tp_size
                            full_param_shape = tuple(full_param_shape)

                    elif isinstance(module, VocabParallelEmbedding):
                        if output_dim is not None:
                            split_dim = output_dim
                            full_param_shape = list(param.shape)
                            full_param_shape[split_dim] = full_param_shape[split_dim] * tp_size
                            full_param_shape = tuple(full_param_shape)

                    split_idx = self.sampler_device_mesh["infer_tp"].get_local_rank()
                    param_metadata = ParameterMetadata(
                        param_name=checkpoint_name,
                        original_param_name=full_param_name,
                        param_shape=param.shape,
                        full_param_shape=full_param_shape,
                        param_dtype=param.dtype,
                        split_dim=split_dim,
                        split_idx=split_idx,
                    )
                    self.sampler_parameters[(full_param_name, split_idx)] = param
                    param_keys.append(param_metadata)
        elif sampler_name == "sglang":
            raise NotImplementedError("SGLang parameter keys are not supported yet.")
        else:
            raise ValueError(f"Invalid sampler name: {sampler_name}")
        rank_mapping.params = param_keys
        return rank_mapping

    def construct_recv_ops_and_buffers(self, routing_table=None):
        """Construct receive operations and weight buffers for weight transfer."""
        transfers_for_sampler_rank = routing_table.get_transfers_for_sampler_rank(self.rank)

        ops = []
        buffers = []

        if transfers_for_sampler_rank:
            for src_rank in sorted(transfers_for_sampler_rank.keys()):
                param_dicts = sorted(
                    transfers_for_sampler_rank[src_rank], key=lambda x: (x["actor"].param_name, x["actor"].split_idx)
                )

                for param_dict in param_dicts:
                    sampler_meta = param_dict["sampler"]
                    sampler_split_idx = sampler_meta.split_idx
                    param_key = (sampler_meta.original_param_name, sampler_split_idx)

                    if param_key not in self.sampler_parameters:
                        raise ValueError(
                            f"Parameter {sampler_meta.original_param_name} with split_idx {sampler_split_idx} not found in sampler model"
                        )

                    recv_buffer = self.sampler_parameters[param_key]

                    recv_op = P2POp(
                        op=torch.distributed.irecv,
                        tensor=recv_buffer,
                        group_peer=src_rank,
                        group=self.bridge_pg,
                    )
                    ops.append(recv_op)
        return ops, buffers

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def receive_sampler_to_trainer_weights(self, routing_table=None):
        """Receive tensors from actor via bridge PG and update vLLM model."""
        assert hasattr(self, "bridge_pg"), "Bridge process group is not initialized"

        aggressive_empty_cache(force_sync=True)

        if not self.ops:
            if routing_table:
                self.routing_table = routing_table
            self.ops, self.buffers = self.construct_recv_ops_and_buffers(routing_table=self.routing_table)
        else:
            if self.offload_p2p_buffer:
                device_id = get_device_id()
                for _, source_tensor in self.buffers:
                    source_tensor.data = source_tensor.data.to(device_id, non_blocking=True)

        if self.ops:
            torch.cuda.synchronize()
            reqs = patched_batch_isend_irecv(self.ops)
            for req in reqs:
                req.wait()

        for source_tensor, dest_tensor in self.buffers:
            source_tensor.copy_(dest_tensor, non_blocking=True)

        if self.offload_p2p_buffer:
            for _, dest_tensor in self.buffers:
                dest_tensor.data = dest_tensor.data.to("cpu", non_blocking=True)


class MegatronSamplerP2PMixin(SamplerP2PMixin):
    """Megatron-specific P2P weight transfer methods for SamplerWorker."""

    def _get_sampler_vllm_parameter_mapping(self, hf_config):
        """Build parameter metadata list for vLLM sampler parameters on this rank."""
        from vllm.model_executor.layers.fused_moe import FusedMoE
        from vllm.model_executor.layers.linear import (
            ColumnParallelLinear,
            MergedColumnParallelLinear,
            QKVParallelLinear,
            ReplicatedLinear,
            RowParallelLinear,
        )
        from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding

        from axon.sampler.p2p import get_hooks

        vllm_model = self.sampler.inference_engine.worker.model_runner.model
        tp_size = self.sampler_device_mesh["infer_tp"].size()
        split_idx = self.sampler_device_mesh["infer_tp"].get_local_rank()

        # Model-specific hooks (optional). A model registers here when
        # its parameter layout doesn't match what the generic enumeration
        # below produces — see axon/sampler/p2p/__init__.py.
        hooks = get_hooks(vllm_model) or {}
        override_param = hooks.get("override_param")
        extra_buffers = hooks.get("extra_buffers")

        param_keys = []
        self.sampler_parameters = {}
        self.sampler_parameters_copy = {}
        for module_name, module in vllm_model.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                full_param_name = f"{module_name}.{param_name}" if module_name else param_name
                if full_param_name.startswith("language_model."):
                    full_param_name = full_param_name[len("language_model.") :]

                if override_param is not None:
                    overrides = override_param(
                        full_param_name=full_param_name,
                        module=module,
                        param=param,
                        tp_size=tp_size,
                        split_idx=split_idx,
                        sampler_parameters=self.sampler_parameters,
                    )
                    if overrides is not None:
                        param_keys.extend(overrides)
                        continue

                split_dim = -1
                full_param_shape = param.shape

                input_dim = getattr(param, "input_dim", None)
                output_dim = getattr(param, "output_dim", None)
                if isinstance(module, ColumnParallelLinear | QKVParallelLinear | MergedColumnParallelLinear):
                    if output_dim is not None:
                        split_dim = output_dim
                        full_param_shape = list(param.shape)
                        full_param_shape[split_dim] = full_param_shape[split_dim] * tp_size
                        full_param_shape = tuple(full_param_shape)
                        if isinstance(module, QKVParallelLinear):
                            full_param_shape = list(full_param_shape)
                            full_param_shape[0] = (
                                module.total_num_heads + 2 * module.total_num_kv_heads
                            ) * module.head_size
                            full_param_shape = tuple(full_param_shape)

                elif isinstance(module, RowParallelLinear):
                    if input_dim is not None:
                        split_dim = input_dim
                        full_param_shape = list(param.shape)
                        full_param_shape[split_dim] = full_param_shape[split_dim] * tp_size
                        full_param_shape = tuple(full_param_shape)

                elif isinstance(module, VocabParallelEmbedding):
                    if output_dim is not None:
                        split_dim = output_dim
                        full_param_shape = list(param.shape)
                        full_param_shape[split_dim] = full_param_shape[split_dim] * tp_size
                        full_param_shape = tuple(full_param_shape)

                elif isinstance(module, FusedMoE):
                    assert ".experts." in full_param_name
                    num_experts = param.shape[0]
                    intermediate_size_per_partition = module.intermediate_size_per_partition

                    expert_param_shape = param.shape[1:]
                    expert_split_dim = -1
                    full_expert_shape = list(expert_param_shape)

                    if len(full_expert_shape) == 0:
                        continue

                    for dim_idx, dim_size in enumerate(expert_param_shape):
                        if dim_size == 2 * intermediate_size_per_partition:
                            expert_split_dim = dim_idx
                            full_expert_shape[dim_idx] = dim_size * tp_size
                            break
                        elif dim_size == intermediate_size_per_partition:
                            expert_split_dim = dim_idx
                            full_expert_shape[dim_idx] = dim_size * tp_size
                            break

                    full_expert_shape = tuple(full_expert_shape)

                    prefix, suffix = full_param_name.split(".experts.", 1)
                    base_name = f"{prefix}.experts"
                    weight_name = suffix
                    if DEBUG_P2P_MISMATCH:
                        param_copy = param.detach().clone()
                        with torch.no_grad():
                            param.zero_()
                    for expert_idx in range(num_experts):
                        expert_param_name = f"{base_name}.{expert_idx}.{weight_name}"
                        expert_tensor = param[expert_idx, ...]
                        assert expert_tensor.is_contiguous(), "Expert tensor must be contiguous for efficient transfer"
                        param_metadata = ParameterMetadata(
                            param_name=expert_param_name,
                            original_param_name=expert_param_name,
                            param_shape=expert_param_shape,
                            full_param_shape=full_expert_shape,
                            param_dtype=expert_tensor.dtype,
                            split_dim=expert_split_dim,
                            split_idx=split_idx,
                            expert_idx=expert_idx,
                        )
                        self.sampler_parameters[expert_param_name] = expert_tensor
                        if DEBUG_P2P_MISMATCH:
                            self.sampler_parameters_copy[expert_param_name] = param_copy[expert_idx, ...]
                        param_keys.append(param_metadata)
                    continue

                elif "fused_qkv_a_proj" in full_param_name and isinstance(module, ReplicatedLinear):
                    output_sizes = module.output_sizes
                    split_names = ["q_a_proj", "kv_a_proj_with_mqa"]

                    if DEBUG_P2P_MISMATCH:
                        param_copy = param.detach().clone()
                        with torch.no_grad():
                            param.zero_()

                    current_offset = 0
                    for shard_id, (split_name, size) in enumerate(zip(split_names, output_sizes, strict=True)):
                        unfused_param_name = full_param_name.replace("fused_qkv_a_proj", split_name)
                        param_slice = param[current_offset : current_offset + size]
                        param_slice_shape = tuple(param_slice.shape)
                        param_metadata = ParameterMetadata(
                            param_name=unfused_param_name,
                            original_param_name=full_param_name,
                            param_shape=param_slice_shape,
                            full_param_shape=param_slice_shape,
                            param_dtype=param.dtype,
                            split_dim=-1,
                            split_idx=0,
                            metadata={"fused_shard_id": shard_id, "fused_offset": current_offset},
                        )
                        self.sampler_parameters[unfused_param_name] = param_slice
                        if DEBUG_P2P_MISMATCH:
                            self.sampler_parameters_copy[unfused_param_name] = param_copy[
                                current_offset : current_offset + size, ...
                            ]
                        param_keys.append(param_metadata)
                        current_offset += size
                    continue

                metadata_dict = {}
                if "qkv_proj" in full_param_name:
                    # Transformers 5.x VL models use composite configs where text
                    # attributes live on hf_config.text_config, not hf_config directly.
                    text_cfg = getattr(hf_config, "text_config", hf_config)
                    metadata_dict["num_attention_heads"] = text_cfg.num_attention_heads
                    metadata_dict["num_key_value_heads"] = text_cfg.num_key_value_heads
                    metadata_dict["tp_size"] = tp_size
                elif ".attn.qkv." in full_param_name:
                    vision_config = getattr(hf_config, "vision_config", None)
                    if vision_config:
                        num_heads = getattr(
                            vision_config, "num_heads", getattr(vision_config, "num_attention_heads", 1)
                        )
                        metadata_dict["num_attention_heads"] = num_heads
                        metadata_dict["num_key_value_heads"] = num_heads
                        metadata_dict["tp_size"] = tp_size

                if len(param.shape) == 0:
                    continue

                param_metadata = ParameterMetadata(
                    param_name=full_param_name,
                    original_param_name=full_param_name,
                    param_shape=param.shape,
                    full_param_shape=full_param_shape,
                    param_dtype=param.dtype,
                    split_dim=split_dim,
                    split_idx=split_idx,
                    metadata=metadata_dict,
                )
                self.sampler_parameters[full_param_name] = param
                param_keys.append(param_metadata)
                if DEBUG_P2P_MISMATCH:
                    param_copy = param.detach().clone()
                    with torch.no_grad():
                        param.zero_()
                    self.sampler_parameters_copy[full_param_name] = param_copy

        # Model-specific persistent buffers live outside `named_parameters`
        # so the actor ships them through its state_dict extra-keys path
        # and we mirror them here to keep the routing table balanced.
        if extra_buffers is not None:
            for full_buf_name, buf in extra_buffers(vllm_model):
                if full_buf_name.startswith("language_model."):
                    full_buf_name = full_buf_name[len("language_model.") :]
                param_keys.append(
                    ParameterMetadata(
                        param_name=full_buf_name,
                        original_param_name=full_buf_name,
                        param_shape=tuple(buf.shape),
                        full_param_shape=tuple(buf.shape),
                        param_dtype=buf.dtype,
                        split_dim=-1,
                        split_idx=0,
                    )
                )
                self.sampler_parameters[full_buf_name] = buf

        return param_keys

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def get_parameter_mapping(self):
        """Get the parameter keys (not tensors) residing on this rank."""
        from transformers import AutoConfig

        rank_mapping = RankMapping(rank=self.rank, params=[])
        hf_config = AutoConfig.from_pretrained(self.config.model_path)  # nosec B615
        sampler_name = self.config.name
        if sampler_name == "vllm":
            rank_mapping.params = self._get_sampler_vllm_parameter_mapping(hf_config)
        elif sampler_name == "sglang":
            raise NotImplementedError("SGLang parameter keys are not supported yet.")
        else:
            raise ValueError(f"Invalid sampler name: {sampler_name}")
        return rank_mapping

    def construct_recv_ops_and_buffers(self, routing_table=None):
        """Construct receive operations and weight buffers for weight transfer."""
        transfers_for_sampler_rank = routing_table.get_transfers_for_sampler_rank(self.rank)
        if not transfers_for_sampler_rank:
            return [], []

        tensor_cache, all_recv_tensors, p2p_log = {}, [], []
        ops, buffers = [], []
        for src_rank in sorted(transfers_for_sampler_rank.keys()):
            for param_dict in sorted(
                transfers_for_sampler_rank[src_rank], key=lambda x: (x["actor"].param_name, x["actor"].split_idx)
            ):
                actor_meta = param_dict["actor"]
                sampler_meta = param_dict["sampler"]
                sampler_param_name = sampler_meta.param_name
                cache_key = (sampler_param_name, actor_meta.split_dim, actor_meta.split_idx)

                actor_tp = _compute_tp_degree(actor_meta)
                sampler_tp = _compute_tp_degree(sampler_meta)
                if "qkv_proj" in sampler_meta.param_name or ".attn.qkv." in sampler_meta.param_name:
                    sampler_tp = sampler_meta.metadata["tp_size"]

                is_qkv_fused = "qkv_proj" in sampler_param_name or ".attn.qkv." in sampler_param_name

                if actor_meta.split_dim != -1:
                    assert actor_tp % sampler_tp == 0 or sampler_tp % actor_tp == 0, (
                        f"Actor TP {actor_tp} and sampler TP {sampler_tp} must be multiples of each other."
                    )

                if sampler_param_name not in self.sampler_parameters:
                    raise ValueError(f"Parameter {sampler_param_name} not found in sampler model")

                if is_qkv_fused:
                    assert sampler_meta.split_dim < 1 and actor_meta.split_dim < 1, "QKV must be split by 0-dim."
                    num_q_heads = sampler_meta.metadata["num_attention_heads"]
                    num_kv_heads = sampler_meta.metadata["num_key_value_heads"]
                    head_dim = sampler_meta.full_param_shape[0] // (num_q_heads + 2 * num_kv_heads)

                    target_param = self.sampler_parameters[sampler_param_name]
                    sampler_q_heads = num_q_heads // sampler_tp
                    sampler_kv_heads = max(1, num_kv_heads // sampler_tp)

                    actor_q_heads = num_q_heads // actor_tp
                    actor_kv_heads = max(1, num_kv_heads // actor_tp)

                    if actor_tp > sampler_tp:
                        actor_idx = actor_meta.split_idx % (actor_tp // sampler_tp)
                        q_heads_per_actor = sampler_q_heads // (actor_tp // sampler_tp)
                        kv_heads_per_actor = sampler_kv_heads // (actor_tp // sampler_tp)
                        q_offset = actor_idx * q_heads_per_actor * head_dim
                        kv_offset = actor_idx * kv_heads_per_actor * head_dim
                        num_groups_recv = kv_heads_per_actor
                    elif actor_tp < sampler_tp and actor_kv_heads * actor_tp < sampler_tp:
                        replicas_per_kv = sampler_tp // (actor_kv_heads * actor_tp)
                        num_groups_recv = 1
                        q_heads_per_actor = actor_q_heads // actor_kv_heads // replicas_per_kv
                        q_offset = kv_offset = 0
                    else:
                        q_offset = kv_offset = 0
                        q_heads_per_actor = sampler_q_heads
                        num_groups_recv = sampler_kv_heads

                    all_q_size = sampler_q_heads * head_dim
                    all_k_size = sampler_kv_heads * head_dim
                    q_per_kv = q_heads_per_actor // num_groups_recv

                    for group_idx in range(num_groups_recv):
                        q_start = q_offset + group_idx * q_per_kv * head_dim
                        k_start = all_q_size + kv_offset + group_idx * head_dim
                        v_start = all_q_size + all_k_size + kv_offset + group_idx * head_dim

                        q_recv = target_param[q_start : q_start + q_per_kv * head_dim]
                        k_recv = target_param[k_start : k_start + head_dim]
                        v_recv = target_param[v_start : v_start + head_dim]

                        assert q_recv.is_contiguous() and k_recv.is_contiguous() and v_recv.is_contiguous()
                        all_recv_tensors.append((src_rank, q_recv))
                        all_recv_tensors.append((src_rank, k_recv))
                        all_recv_tensors.append((src_rank, v_recv))

                elif (
                    "gate_up_proj" in sampler_param_name or "w13_weight" in sampler_param_name
                ) and actor_tp != sampler_tp:
                    assert sampler_meta.split_dim < 1 and actor_meta.split_dim < 1, (
                        "Gate and up must be split by 0-dim."
                    )
                    target_param = self.sampler_parameters[sampler_param_name]
                    gate_size = target_param.shape[0] // 2
                    if actor_tp < sampler_tp:
                        gate_recv_slice = target_param[0:gate_size]
                        up_recv_slice = target_param[gate_size : 2 * gate_size]
                    else:
                        shards_per_sampler = actor_tp // sampler_tp
                        local_idx = actor_meta.split_idx % shards_per_sampler
                        chunk_size = gate_size // shards_per_sampler
                        gate_start = local_idx * chunk_size
                        gate_end = gate_start + chunk_size
                        up_start = gate_size + local_idx * chunk_size
                        up_end = up_start + chunk_size
                        gate_recv_slice = target_param[gate_start:gate_end]
                        up_recv_slice = target_param[up_start:up_end]

                    all_recv_tensors.append((src_rank, gate_recv_slice))
                    all_recv_tensors.append((src_rank, up_recv_slice))

                elif cache_key in tensor_cache:
                    recv_tensor = tensor_cache[cache_key]
                    all_recv_tensors.append((src_rank, recv_tensor))
                else:
                    target_param = self.sampler_parameters[sampler_param_name]

                    if actor_tp > sampler_tp:
                        shards_per_sampler = actor_tp // sampler_tp
                        local_idx = actor_meta.split_idx % shards_per_sampler
                        recv_tensor = slice_for_tp(target_param, actor_meta.split_dim, local_idx, shards_per_sampler)
                        if not recv_tensor.is_contiguous():
                            contiguous_recv_tensor = recv_tensor.contiguous()
                            buffers.append((recv_tensor, contiguous_recv_tensor))
                            recv_tensor = contiguous_recv_tensor
                    else:
                        recv_tensor = target_param

                    assert recv_tensor.is_contiguous(), f"Receive tensor for {sampler_param_name} must be contiguous"
                    tensor_cache[cache_key] = recv_tensor
                    all_recv_tensors.append((src_rank, recv_tensor))

        for src_rank, recv_tensor in all_recv_tensors:
            if DEBUG_P2P_MISMATCH:
                p2p_log.append(
                    {
                        "op_type": "recv",
                        "peer_rank": src_rank,
                        "shape": list(recv_tensor.shape),
                        "dtype": str(recv_tensor.dtype),
                        "numel": recv_tensor.numel(),
                    }
                )
            ops.append(
                P2POp(
                    op=dist.irecv,
                    tensor=recv_tensor,
                    group_peer=src_rank,
                    group=self.bridge_pg,
                )
            )

        if p2p_log:
            _write_p2p_debug_log(
                p2p_log,
                f"sampler_rank_{self.rank}_recv.json",
                rank=0,
                print_prefix=f"[P2P Check] Sampler rank {self.rank} saved receive log to",
            )

        if DEBUG_P2P_MISMATCH:
            self._print_p2p_mismatch()

        return ops, buffers

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def receive_sampler_to_trainer_weights(self, routing_table=None):
        """Receive tensors from actor via bridge PG and update vLLM model."""
        assert hasattr(self, "bridge_pg"), "Bridge process group is not initialized"

        aggressive_empty_cache(force_sync=True)

        if not self.ops:
            if routing_table:
                self.routing_table = routing_table
            self.ops, self.buffers = self.construct_recv_ops_and_buffers(routing_table=self.routing_table)
        else:
            if self.offload_p2p_buffer:
                device_id = get_device_id()
                for _, source_tensor in self.buffers:
                    source_tensor.data = source_tensor.data.to(device_id, non_blocking=True)

        if self.ops:
            reqs = patched_batch_isend_irecv(self.ops)
            for req in reqs:
                req.wait()

        if DEBUG_P2P_MISMATCH:
            self._print_p2p_mismatch()

        for source_tensor, dest_tensor in self.buffers:
            source_tensor.copy_(dest_tensor, non_blocking=True)

        if self.offload_p2p_buffer:
            for _, dest_tensor in self.buffers:
                dest_tensor.data = dest_tensor.data.to("cpu", non_blocking=True)

    def _print_p2p_mismatch(self):
        param_diffs = []
        for param_name, received_tensor in self.sampler_parameters.items():
            if param_name in self.sampler_parameters_copy:
                original_tensor = self.sampler_parameters_copy[param_name]
                abs_diff = torch.abs(received_tensor - original_tensor)
                max_abs_diff = abs_diff.max().item()
                mean_abs_diff = abs_diff.mean().item()

                original_abs = torch.abs(original_tensor)
                relative_diff = torch.where(original_abs > 1e-10, abs_diff / (original_abs + 1e-10), abs_diff)
                max_relative_diff = relative_diff.max().item()
                mean_relative_diff = relative_diff.mean().item()

                param_diffs.append(
                    {
                        "name": param_name,
                        "max_abs_diff": max_abs_diff,
                        "mean_abs_diff": mean_abs_diff,
                        "max_relative_diff": max_relative_diff,
                        "mean_relative_diff": mean_relative_diff,
                        "shape": tuple(received_tensor.shape),
                        "dtype": received_tensor.dtype,
                        "numel": received_tensor.numel(),
                    }
                )

        param_diffs.sort(key=lambda x: x["mean_abs_diff"], reverse=True)

        if param_diffs:
            print(f"\n{'=' * 100}")
            print(f"[Sampler Rank {self.rank}] Top 10 Tensors with Largest P2P Transfer Differences:")
            print(f"{'=' * 100}")
            for i, diff_info in enumerate(param_diffs[:10]):
                print(f"\n{i + 1}. Parameter: {diff_info['name']}")
                print(f"   Shape: {diff_info['shape']}, Dtype: {diff_info['dtype']}, Numel: {diff_info['numel']}")
                print(f"   Max Abs Diff: {diff_info['max_abs_diff']:.6e}")
                print(f"   Mean Abs Diff: {diff_info['mean_abs_diff']:.6e}")
                print(f"   Max Relative Diff: {diff_info['max_relative_diff']:.6e}")
                print(f"   Mean Relative Diff: {diff_info['mean_relative_diff']:.6e}")
            print(f"{'=' * 100}\n")

            total_params = len(param_diffs)
            params_with_diff = sum(1 for d in param_diffs if d["mean_abs_diff"] > 1e-10)
            print(
                f"[Sampler Rank {self.rank}] Summary: {params_with_diff}/{total_params} parameters have differences > 1e-10"
            )
            if params_with_diff > 0:
                avg_max_diff = sum(d["max_abs_diff"] for d in param_diffs) / total_params
                print(f"[Sampler Rank {self.rank}] Average max abs diff across all params: {avg_max_diff:.6e}\n")
