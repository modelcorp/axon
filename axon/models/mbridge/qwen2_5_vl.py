# Copyright 2025 Model AI Corp.
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
# Adapted from the mbridge Qwen2.5-VL bridge (github.com/ISEEKYAN/mbridge), BSD-3-Clause.
from collections.abc import Generator

import torch
import torch.distributed as dist
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig

# Global cache for single-rank groups
_SINGLE_RANK_GROUP_CACHE = {}
_GROUPS_INITIALIZED = False


def ensure_single_rank_groups():
    """
    Initialize single-rank groups for all ranks exactly once.

    Must be called collectively by all ranks on first invocation.
    Subsequent calls are no-ops (safe from any rank).
    """
    global _SINGLE_RANK_GROUP_CACHE, _GROUPS_INITIALIZED

    if _GROUPS_INITIALIZED:
        return

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    for r in range(world_size):
        group = dist.new_group([r])
        if r == rank:
            _SINGLE_RANK_GROUP_CACHE[rank] = group

    _GROUPS_INITIALIZED = True


def get_single_rank_group():
    """Get the single-rank group for this rank. Returns None if not initialized."""
    if not _GROUPS_INITIALIZED:
        raise RuntimeError(
            "Single-rank groups not initialized. "
            "Call ensure_single_rank_groups() during worker init before model creation."
        )
    return _SINGLE_RANK_GROUP_CACHE.get(dist.get_rank())


def monkey_patch_qwen2_5_vl_vision_model_init():
    """Patch Qwen2_5VisionModel.__init__ to disable TP for vision encoder and projection."""

    from mbridge.models.qwen2_5_vl.vision_model import Qwen2_5VisionModel

    def patched_init(
        self,
        transformer_config: TransformerConfig,
        transformer_layer_spec: ModuleSpec,
        projection_config: TransformerConfig,
        projection_layer_spec: ModuleSpec,
        projection_type: str = "mlp",
        pre_process: bool = True,
        post_process: bool = False,
    ) -> None:
        import copy

        import megatron.core as mcore
        from mbridge.models.qwen2_5_vl.vision_model import PatchEmbed, VisionRotaryEmbedding
        from mbridge.models.qwen2_5_vl.vision_transformer_block import Qwen2_5VisionTransformerBlock as TransformerBlock
        from megatron.core import parallel_state
        from megatron.core.models.common.vision_module.vision_module import VisionModule
        from megatron.core.models.vision.multimodal_projector import MultimodalProjector
        from megatron.core.transformer.enums import ModelType

        VisionModule.__init__(self, config=transformer_config)

        self.spatial_merge_size = transformer_config.spatial_merge_size

        embed_dim = transformer_config.hidden_size
        num_heads = transformer_config.num_attention_heads
        temporal_patch_size = transformer_config.temporal_patch_size
        patch_size = transformer_config.patch_size
        in_channels = transformer_config.in_channels

        self.patch_size = transformer_config.patch_size
        self.fullatt_block_indexes = transformer_config.fullatt_block_indexes
        self.window_size = transformer_config._qwen2_5_vl_window_size
        self.spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size

        self.max_sequence_length = transformer_config.seq_length
        self.patch_embed = PatchEmbed(
            patch_size=patch_size,
            temporal_patch_size=temporal_patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )

        head_dim = embed_dim // num_heads
        self.rotary_pos_emb = VisionRotaryEmbedding(head_dim // 2)

        self.model_type = ModelType.encoder_or_decoder
        self.pre_process = pre_process
        self.post_process = post_process

        # Transformer layers. pre_process/post_process are fixed for this bridge;
        # pipeline-parallel variants should make them configurable.
        # NOTE: a final layer norm and/or linear layer present in some implementations are omitted here.
        args = {}
        from packaging.version import Version

        if Version(mcore.__version__) >= Version("0.13.0"):
            args["vp_stage"] = 0
        self.decoder = TransformerBlock(
            config=transformer_config,
            spec=transformer_layer_spec,
            pre_process=self.pre_process,
            post_process=self.post_process,
            post_layer_norm=True,
            **args,
        )

        self.merge_hidden_size = projection_config.ffn_hidden_size
        self.square_merge_size = self.merge_hidden_size // embed_dim

        ########################
        # The MultimodalProjector uses MLP which internally uses TP-parallel linear layers (ColumnParallelLinear and RowParallelLinear).
        # The tp_group parameter defaults to None, which means it uses the global TP group from parallel_state.
        # The vision projection should NOT use tensor parallelism to minimize probs diff with vLLM.
        # Need to explicitly set to 1 to ensure no TP.
        tp_size = parallel_state.get_tensor_model_parallel_world_size()
        if tp_size > 1:
            no_tp_group = get_single_rank_group()
        else:
            no_tp_group = None

        if self.post_process:
            projection_config_no_tp = copy.deepcopy(projection_config)
            projection_config_no_tp.tensor_model_parallel_size = 1

            self.projection = MultimodalProjector(
                projection_config_no_tp,
                projection_layer_spec,
                projection_type,
                projection_config.ffn_hidden_size,
                tp_group=no_tp_group,
            )
            for name, param in self.projection.named_parameters():
                param.is_replicated_vision_weight = True  # Custom marker: weights are replicated, not TP split
        else:
            self.projection = None
        ############################
        self.input_tensor = None

    Qwen2_5VisionModel.__init__ = patched_init
    print("Patched Qwen2_5VisionModel.__init__ to disable TP for vision components")


def monkey_patch_bridge_load_weights():
    """Patch Bridge.load_weights to correctly handle vision model weights (broadcast, not scatter)."""

    from mbridge.core.bridge import Bridge

    def patched_load_weights(
        self,
        models: list[torch.nn.Module],
        weights_path: str,
        memory_efficient: bool = False,
    ) -> None:
        """
        Load weights from a Hugging Face model into a Megatron-Core model.

        Args:
            models: List of model instances, supporting VPP (Virtual Pipeline Parallelism)
            weights_path: Path to the weights file or Hugging Face model identifier
        """
        self.safetensor_io = self._get_safetensor_io(weights_path)

        for i, model in enumerate(models):
            # map local weight names to global weight names
            local_to_global_map = self._weight_name_mapping_mcore_local_to_global(model)
            # map local weight names to huggingface weight names
            local_to_hf_map = {
                k: self._weight_name_mapping_mcore_to_hf(v)
                for k, v in local_to_global_map.items()
                if "_extra_state" not in k
            }
            # only tp_rank0/etp_rank0 load from disk, others load from tp_rank0/etp_rank0
            to_load_from_disk = []
            for local_name, hf_names in local_to_hf_map.items():
                if ".mlp.experts.linear_fc" in local_name:
                    if self.mpu.etp_rank == 0:
                        to_load_from_disk.extend(hf_names)
                else:
                    if self.mpu.tp_rank == 0:
                        to_load_from_disk.extend(hf_names)
                    else:
                        # special case for lm_head.weight
                        # if make value model, every tp rank will load lm_head.weight
                        if "lm_head.weight" in hf_names:
                            to_load_from_disk.extend(hf_names)

            # load huggingface weights
            if not memory_efficient:
                hf_weights_map = self.safetensor_io.load_some_hf_weight(to_load_from_disk)

            # import mcore weights
            for local_name, hf_names in local_to_hf_map.items():
                param = model.state_dict()[local_name]
                # hf format to mcore format
                if set(to_load_from_disk) & set(hf_names):
                    if not memory_efficient:
                        hf_weights = [hf_weights_map[x] for x in hf_names]
                    else:
                        hf_weights = [self.safetensor_io.load_one_hf_weight(x) for x in hf_names]
                    mcore_weight = self._weight_to_mcore_format(local_name, hf_weights)
                else:
                    mcore_weight = None
                if hf_names[0] in {"lm_head.weight", "model.embed_tokens.weight"}:
                    if param.shape[0] == 1 and (mcore_weight is None or mcore_weight.shape[0] != 1):
                        # skip lm_head.weight when the model is a value model
                        continue

                param_to_load = torch.empty_like(param)
                if ".mlp.experts.linear_fc" in local_name:
                    # split mcore weights across etp
                    if self.mpu.etp_rank == 0:
                        mcore_weights_tp_split = self._weight_split_across_tp(
                            local_name, mcore_weight, param, self.mpu.etp_size
                        )
                        mcore_weights_tp_split = list(mcore_weights_tp_split)
                        mcore_weights_tp_split = [
                            t.to(param.device, dtype=param.dtype).contiguous() for t in mcore_weights_tp_split
                        ]
                    else:
                        mcore_weights_tp_split = None
                    torch.distributed.scatter(
                        param_to_load,
                        mcore_weights_tp_split,
                        src=torch.distributed.get_global_rank(self.mpu.etp_group, 0),
                        group=self.mpu.etp_group,
                    )
                else:
                    #################
                    # Patch to automatically replicate those from vision model.
                    # Only tp_rank 0 has mcore_weight, so broadcast the decision
                    if self.mpu.tp_rank == 0:
                        needs_tp_split = mcore_weight.shape != param.shape and self.mpu.tp_size > 1
                        needs_tp_split_tensor = torch.tensor(
                            [1 if needs_tp_split else 0], device=param.device, dtype=torch.int32
                        )
                    else:
                        needs_tp_split_tensor = torch.tensor([0], device=param.device, dtype=torch.int32)

                    torch.distributed.broadcast(
                        needs_tp_split_tensor,
                        src=torch.distributed.get_global_rank(self.mpu.tp_group, 0),
                        group=self.mpu.tp_group,
                    )
                    needs_tp_split = needs_tp_split_tensor.item() == 1

                    if needs_tp_split:
                        # scatter
                        if self.mpu.tp_rank == 0:
                            mcore_weights_tp_split = self._weight_split_across_tp(
                                local_name, mcore_weight, param, self.mpu.tp_size
                            )
                            mcore_weights_tp_split = [
                                t.to(param.device, dtype=param.dtype).contiguous() for t in mcore_weights_tp_split
                            ]
                        else:
                            mcore_weights_tp_split = None
                        torch.distributed.scatter(
                            param_to_load,
                            mcore_weights_tp_split,
                            src=torch.distributed.get_global_rank(self.mpu.tp_group, 0),
                            group=self.mpu.tp_group,
                        )
                    else:
                        # broadcast full weight
                        if self.mpu.tp_rank == 0:
                            param_to_load.copy_(mcore_weight.to(param.device, dtype=param.dtype))
                        torch.distributed.broadcast(
                            param_to_load,
                            src=torch.distributed.get_global_rank(self.mpu.tp_group, 0),
                            group=self.mpu.tp_group,
                        )
                    ############################
                # load
                param.copy_(param_to_load)

    Bridge.load_weights = patched_load_weights
    print("Patched Bridge.load_weights to correctly handle vision model weights")


def monkey_patch_bridge_export_weights():
    """Patch Bridge.export_weights to correctly handle vision model weights (Projection is tp1)."""

    from mbridge.core.bridge import Bridge
    from mbridge.core.util import (
        broadcast_from_megatron_pp,
        broadcast_str_from_megatron_pp,
        unwrap_model,
    )

    @torch.no_grad()
    def patched_export_weights(
        self,
        models: list[torch.nn.Module],
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        assert len(self.export_weights_buff) == 0, f"should be empty {self.export_weights_buff=}"
        models = [unwrap_model(model) for model in models]

        def get_model_chunk_generator():
            for model in models:
                existing_keys = set()
                for name, param in model.named_parameters():
                    existing_keys.add(name)
                    yield name, param

                # note
                # there is a bug in megatron GPTModel
                # decoder.layers[n].mlp.router.expert_bias" in GPTModel is not registered in named_parameter, but in state_dict().
                # for now we patch it by adding those keys to extra_keys.
                extra_keys = [
                    x
                    for x in model.state_dict().keys()
                    if "_extra_state" not in x and "expert_bias" in x and x not in existing_keys
                ]
                for name in extra_keys:
                    yield name, model.state_dict()[name].to(torch.cuda.current_device())

        weights_names = []
        for vpp_rank, model in enumerate(models):
            existing_keys = set()
            for name, param in model.named_parameters():
                existing_keys.add(name)
                weights_names.append((self.mpu.pp_rank, vpp_rank, name))
            extra_keys = [
                x
                for x in model.state_dict().keys()
                if "_extra_state" not in x and "expert_bias" in x and x not in existing_keys
            ]
            for name in extra_keys:
                weights_names.append((self.mpu.pp_rank, vpp_rank, name))

        weights_names_all_pp = [None] * self.mpu.pp_size
        torch.distributed.all_gather_object(
            object_list=weights_names_all_pp, obj=weights_names, group=self.mpu.pp_group
        )
        weights_names_all_pp = sum(weights_names_all_pp, [])
        model_chunk_generator = get_model_chunk_generator()
        local_to_global_maps = [
            self._weight_name_mapping_mcore_local_to_global(model, consider_ep=False) for model in models
        ]
        for iter_pp_rank, iter_vpp_rank, iter_name in weights_names_all_pp:
            local_to_global_map = local_to_global_maps[iter_vpp_rank]
            if iter_pp_rank == self.mpu.pp_rank:
                try:
                    name, param = next(model_chunk_generator)
                except StopIteration:
                    name, param = None, None
                name = local_to_global_map[iter_name]
            else:
                name, param = None, None

            name = broadcast_str_from_megatron_pp(name)
            broad_pp_param = broadcast_from_megatron_pp(param)

            # EP
            if ".mlp.experts.linear_fc" in name and self.mpu.ep_size >= 1:
                num_experts = self.config.num_moe_experts
                num_experts_per_rank = num_experts // self.mpu.ep_size
                infer_params = [torch.empty_like(broad_pp_param) for _ in range(self.mpu.ep_size)]
                torch.distributed.all_gather(infer_params, broad_pp_param, group=self.mpu.ep_group)

                name_prefix, local_expert_id = name.split(".weight")
                local_expert_id = int(local_expert_id)
                global_expert_ids = [
                    num_experts_per_rank * ep_rank + local_expert_id for ep_rank in range(self.mpu.ep_size)
                ]
                global_expert_names = [f"{name_prefix}.weight{expert_id}" for expert_id in global_expert_ids]

                for name, param in zip(global_expert_names, infer_params, strict=False):
                    if self.mpu.etp_size > 1:
                        # gather etp
                        etp_params = [torch.empty_like(param) for _ in range(self.mpu.etp_size)]
                        torch.distributed.all_gather(etp_params, param, group=self.mpu.etp_group)
                        params = etp_params
                    else:
                        params = [param]

                    merge_params = self._weight_merge_across_tp(name, params, broad_pp_param)
                    converted_names, converted_params = self._weight_to_hf_format(name, merge_params)
                    # Some moe models require multiple weights to be merge into one, such as qwen3vl
                    if len(converted_names) == 0:
                        continue

                    yield from zip(converted_names, [p.detach() for p in converted_params], strict=False)
                continue

            # TP
            if (
                hasattr(broad_pp_param, "tensor_model_parallel")
                and broad_pp_param.tensor_model_parallel
                and not getattr(broad_pp_param, "is_replicated_vision_weight", False)  # Check custom marker
            ):
                # allocate a new tensor with proper size
                if self.mpu.tp_size <= 1:
                    infer_params = [broad_pp_param]
                else:
                    infer_params = [torch.empty_like(broad_pp_param) for _ in range(self.mpu.tp_size)]
                    torch.distributed.all_gather(infer_params, broad_pp_param, group=self.mpu.tp_group)
                infer_params = self._weight_merge_across_tp(name, infer_params, broad_pp_param)
            else:
                infer_params = broad_pp_param

            converted_names, converted_params = self._weight_to_hf_format(name, infer_params)
            # Some moe models require multiple weights to be merge into one, such as qwen3vl
            if len(converted_names) == 0:
                continue

            yield from zip(converted_names, [p.detach() for p in converted_params], strict=False)

    Bridge.export_weights = patched_export_weights
    print("Patched Bridge.export_weights to correctly handle vision model weights")


def apply_all_qwen2_5_vl_patches():
    """Apply all Qwen2.5-VL patches in the correct order."""

    monkey_patch_qwen2_5_vl_vision_model_init()
    monkey_patch_bridge_load_weights()
    monkey_patch_bridge_export_weights()
    print("Applied all Qwen2.5-VL patches")
