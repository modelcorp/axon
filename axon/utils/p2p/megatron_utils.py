# Copyright 2025 Model AI Corp.
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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
# get_transformer_layer_offset adapted from Megatron-LM transformer_layer.py (github.com/NVIDIA/Megatron-LM), BSD-3-Clause.
import inspect

from megatron.core import parallel_state
from megatron.core.distributed import DistributedDataParallel as DDP
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.module import Float16Module


def extract_expert_idx(param_name: str) -> int:
    """
    Extract expert index from parameter name.

    Expert parameters have names like:
    - 'decoder.layers.X.mlp.experts.linear_fc1.weight{expert_id}'     (GroupedMLP)
    - 'model.layers.X.mlp.experts.{expert_id}.w13_weight'             (vLLM FusedMoE)
    - 'decoder.layers.X.moe_block.moe_layer.experts.local_experts.{expert_id}.linear_fc1.weight'
      (Megatron SequentialMLP — Gemma4 style)

    Args:
        param_name: Parameter name

    Returns:
        Expert index if this is an expert parameter, -1 otherwise
    """
    # Method 3: SequentialMLP `local_experts.{id}.` — used by legacy Gemma4 and others.
    # Covers any path that contains `.local_experts.{digit}.` since some models
    # wrap experts under `.moe_block.moe_layer.experts.local_experts.` rather
    # than `.mlp.experts.`.
    if ".local_experts." in param_name:
        parts = param_name.split(".local_experts.")
        if len(parts) == 2:
            remaining = parts[1].split(".")[0]
            if remaining.isdigit():
                return int(remaining)

    # Method 4: TEGroupedMLP under a non-`.mlp.experts.` parent (e.g. Gemma4's
    # `.moe_block.moe_layer.experts.linear_fc{1,2}.weight{i}`). Match on the
    # generic `.experts.linear_fc` pattern with a digit-suffixed weight/bias.
    if ".mlp.experts." not in param_name and ".experts.linear_fc" in param_name:
        for kw in (".weight", ".bias"):
            if kw in param_name:
                parts = param_name.rsplit(kw, 1)
                if len(parts) == 2 and parts[1].isdigit():
                    return int(parts[1])

    if ".mlp.experts." not in param_name:
        return -1

    # Method 1: Check for pattern .weight{digit} or .bias{digit} (Megatron format)
    if ".weight" in param_name:
        parts = param_name.split(".weight")
        if len(parts) == 2 and parts[1].isdigit():
            return int(parts[1])

    if ".bias" in param_name:
        parts = param_name.split(".bias")
        if len(parts) == 2 and parts[1].isdigit():
            return int(parts[1])

    # Method 2: Check for pattern .experts.{digit}. (HF/vLLM format after conversion)
    # e.g., model.layers.X.mlp.experts.5.w13_weight
    parts = param_name.split(".mlp.experts.")
    if len(parts) == 2:
        remaining = parts[1].split(".")[0]  # Get first segment after .mlp.experts.
        if remaining.isdigit():
            return int(remaining)

    return -1


def unwrap_model(model, module_instances=(DDP, Float16Module)):
    return_list = True
    if not isinstance(model, list):
        model = [model]
        return_list = False
    unwrapped_model = []
    for model_module in model:
        while isinstance(model_module, module_instances):
            model_module = model_module.module
        unwrapped_model.append(model_module)
    if not return_list:
        return unwrapped_model[0]
    return unwrapped_model


def get_transformer_layer_offset(pipeline_rank: int, vp_stage: int, config: TransformerConfig) -> int:
    """
    Get the index offset of any pipeline stage, given the level of pipelining.

    Make pipeline_rank and vp_stage as two arguments to make it more flexible,
    which is able to fetch layer offset for any pipeline stage.
    The original function only returns the layer offset for current pipeline stage.

    Extension to https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/transformer_layer.py::get_transformer_layer_offset
    """

    has_vp_stage = (
        inspect.signature(parallel_state.is_pipeline_first_stage).parameters.get("vp_stage", None) is not None
    )
    extra_kwargs = {} if not has_vp_stage else {"ignore_virtual": False, "vp_stage": vp_stage}

    # Check if is_inside_encoder method exists before calling it (not available in all Megatron versions)
    if hasattr(parallel_state, "is_inside_encoder") and not parallel_state.is_inside_encoder():
        pp_decoder_start = parallel_state.get_pipeline_model_parallel_decoder_start()
        if pp_decoder_start is not None:
            pipeline_rank = pipeline_rank - pp_decoder_start

    if config.pipeline_model_parallel_size > 1:
        if hasattr(config, "pipeline_model_parallel_layout") and config.pipeline_model_parallel_layout:
            from megatron.core.transformer.enums import LayerType

            offset = config.pipeline_model_parallel_layout.get_layer_offset(
                layer_type=LayerType.decoder, vp_stage=vp_stage
            )
        elif (
            config.num_layers_in_first_pipeline_stage is not None
            or config.num_layers_in_last_pipeline_stage is not None
        ):
            # Calculate number of pipeline stages to distribute the remaining Transformer
            # layers after deducting the Transformer layers in the first or the last stages
            middle_pipeline_stages = config.pipeline_model_parallel_size
            middle_pipeline_stages -= sum(
                [
                    1 if x is not None else 0
                    for x in (
                        config.num_layers_in_first_pipeline_stage,
                        config.num_layers_in_last_pipeline_stage,
                    )
                ]
            )

            # Calculate layers to distribute in each pipeline stage. If the
            # num_layers_in_first_pipeline_stage and num_layers_in_last_pipeline_stage
            # are not set, we will not enable uneven pipeline. All layers will be treated
            # as middle layers.
            num_layers_in_first_pipeline_stage = (
                0 if config.num_layers_in_first_pipeline_stage is None else config.num_layers_in_first_pipeline_stage
            )
            num_layers_in_last_pipeline_stage = (
                0 if config.num_layers_in_last_pipeline_stage is None else config.num_layers_in_last_pipeline_stage
            )

            middle_num_layers = (
                config.num_layers - num_layers_in_first_pipeline_stage - num_layers_in_last_pipeline_stage
            )

            if (vp_size := config.virtual_pipeline_model_parallel_size) is not None:
                assert vp_stage is not None, "vp_stage must be provided if virtual pipeline model parallel size is set"

                # Calculate number of layers in each virtual model chunk
                # If the num_layers_in_first_pipeline_stage and
                # num_layers_in_last_pipeline_stage are not set, all pipeline stages
                # will be treated as middle pipeline stages in the calculation
                num_layers_per_virtual_model_chunk_in_first_pipeline_stage = (
                    0
                    if config.num_layers_in_first_pipeline_stage is None
                    else config.num_layers_in_first_pipeline_stage // vp_size
                )

                num_layers_per_virtual_model_chunk_in_last_pipeline_stage = (
                    0
                    if config.num_layers_in_last_pipeline_stage is None
                    else config.num_layers_in_last_pipeline_stage // vp_size
                )

                num_layers_per_vritual_model_chunk_in_middle_pipeline_stage = middle_num_layers // vp_size

                # First stage + middle stage + last stage
                total_virtual_chunks = (
                    num_layers_per_virtual_model_chunk_in_first_pipeline_stage
                    + num_layers_per_vritual_model_chunk_in_middle_pipeline_stage
                    + num_layers_per_virtual_model_chunk_in_last_pipeline_stage
                )

                # Calculate the layer offset with interleaved uneven pipeline parallelism
                if pipeline_rank == 0:
                    offset = vp_stage * total_virtual_chunks
                else:
                    offset = (
                        vp_stage * total_virtual_chunks
                        + num_layers_per_virtual_model_chunk_in_first_pipeline_stage
                        + (pipeline_rank - 1)
                        * (num_layers_per_vritual_model_chunk_in_middle_pipeline_stage // middle_pipeline_stages)
                    )
            else:
                if middle_pipeline_stages > 0:
                    num_layers_per_pipeline_rank = middle_num_layers // middle_pipeline_stages
                else:
                    num_layers_per_pipeline_rank = 0

                middle_pipeline_rank = (
                    pipeline_rank if config.num_layers_in_first_pipeline_stage is None else pipeline_rank - 1
                )

                if pipeline_rank == 0:
                    offset = 0
                else:
                    offset = (middle_pipeline_rank * num_layers_per_pipeline_rank) + num_layers_in_first_pipeline_stage
        else:
            num_layers = config.num_layers

            # Increase the number of layers by one if we include the embedding (loss)
            # layer into pipeline parallelism partition and placement
            if config.account_for_embedding_in_pipeline_split:
                num_layers += 1

            if config.account_for_loss_in_pipeline_split:
                num_layers += 1

            num_layers_per_pipeline_rank = num_layers // config.pipeline_model_parallel_size

            if (vp_size := config.virtual_pipeline_model_parallel_size) is not None:
                assert vp_stage is not None, "vp_stage must be provided if virtual pipeline model parallel size is set"

                num_layers_per_virtual_rank = num_layers_per_pipeline_rank // vp_size
                total_virtual_chunks = num_layers // vp_size
                offset = vp_stage * total_virtual_chunks + (pipeline_rank * num_layers_per_virtual_rank)

                # Reduce the offset of embedding layer from the total layer number
                if config.account_for_embedding_in_pipeline_split and not parallel_state.is_pipeline_first_stage(
                    **extra_kwargs
                ):
                    offset -= 1
            else:
                offset = pipeline_rank * num_layers_per_pipeline_rank

                # Reduce the offset of embedding layer from the total layer number
                if config.account_for_embedding_in_pipeline_split and not parallel_state.is_pipeline_first_stage(
                    **extra_kwargs
                ):
                    offset -= 1
    else:
        offset = 0
    return offset


def adjust_param_pp_vpp(name, pp_rank, vpp_rank, transformer_config, layer_name="layers"):
    """
    Adjust parameter names to account for Pipeline Parallelism (PP) and Virtual Pipeline Parallelism (VPP).

    Args:
        name (str): Original parameter name with local layer index (e.g., "decoder.layers.2.self_attention.linear_qkv.weight")
        pp_rank (int): Pipeline parallel rank (which pipeline stage this parameter belongs to)
        vpp_rank (int): Virtual pipeline parallel rank within the pipeline stage
        transformer_config: Megatron transformer configuration containing layer distribution info
        layer_name (str, optional): Name of the layer container in the parameter path. Defaults to "layers".

    Returns:
        str: Adjusted parameter name with global layer index (e.g., "decoder.layers.14.self_attention.linear_qkv.weight")

    Example:
        >>> # Parameter from PP rank 2, VPP rank 0, local layer 2
        >>> adjust_param_pp_vpp("decoder.layers.2.self_attention.linear_qkv.weight", 2, 0, config)
        "decoder.layers.14.self_attention.linear_qkv.weight"  # Global layer 14
    """
    # Calculate the global layer offset for this PP/VPP combination
    layer_offset = get_transformer_layer_offset(pp_rank, vpp_rank, transformer_config)

    # Only adjust parameters that belong to transformer layers (contain layer_name in path)
    if layer_name in name:
        split_name = name.split(".")

        # Find the index of the layer container name (e.g., "layers")
        layer_container_idx = None
        for i, name_part in enumerate(split_name):
            if name_part == layer_name:
                layer_container_idx = i
                break

        # The layer number should be immediately after the layer container name
        layer_num_idx = layer_container_idx + 1

        # Validate that we found the layer container and the layer number exists
        assert len(split_name) >= layer_num_idx + 1, f"Invalid parameter name structure: {split_name}"
        assert split_name[layer_num_idx].isdigit(), f"Expected layer number at index {layer_num_idx}, got: {split_name}"

        # Convert local layer index to global layer index
        local_layer_idx = int(split_name[layer_num_idx])
        global_layer_idx = local_layer_idx + layer_offset
        split_name[layer_num_idx] = str(global_layer_idx)

        # Reconstruct the parameter name with global layer index
        name = ".".join(split_name)

    return name


def adjust_param_ep(param_name, expert_offset):
    """
    Adjust local expert index to global expert index in parameter name.
    Expert parameters in Megatron have names like:
    - 'decoder.layers.X.mlp.experts.linear_fc1.weight0' (GroupedMLP)
    - 'decoder.layers.X.moe_block.moe_layer.experts.local_experts.0.linear_fc1.weight'
      (SequentialMLP — Gemma4 style)

    Args:
        param_name: Parameter name with local expert index
        expert_offset: Offset to add to local expert index (ep_rank * experts_per_rank)

    Returns:
        Parameter name with global expert index
    """
    if expert_offset == 0:
        return param_name

    # SequentialMLP path: `.local_experts.{id}.` — legacy naming.
    if ".local_experts." in param_name:
        prefix, suffix = param_name.split(".local_experts.", 1)
        first_dot = suffix.find(".")
        if first_dot < 0:
            return param_name
        local_id_str = suffix[:first_dot]
        if not local_id_str.isdigit():
            return param_name
        rest = suffix[first_dot:]
        global_id = int(local_id_str) + expert_offset
        return f"{prefix}.local_experts.{global_id}{rest}"

    # TEGroupedMLP under a non-`.mlp.experts.` parent (e.g. Gemma4's
    # `.moe_block.moe_layer.experts.linear_fc{1,2}.weight{i}`).
    if ".mlp.experts." not in param_name and ".experts.linear_fc" in param_name:
        for kw in (".weight", ".bias"):
            if kw in param_name:
                parts = param_name.rsplit(kw, 1)
                if len(parts) == 2 and parts[1].isdigit():
                    local_expert_id = int(parts[1])
                    global_expert_id = local_expert_id + expert_offset
                    return f"{parts[0]}{kw}{global_expert_id}"

    if ".mlp.experts." not in param_name:
        return param_name

    # Expert parameters end with '.weight{expert_id}' or '.bias{expert_id}'
    # Find the position where we split between name and expert ID
    if ".weight" in param_name:
        # Split on '.weight' and check if what follows is a digit
        parts = param_name.split(".weight")
        if len(parts) == 2 and parts[1].isdigit():
            local_expert_id = int(parts[1])
            global_expert_id = local_expert_id + expert_offset
            return f"{parts[0]}.weight{global_expert_id}"

    if ".bias" in param_name:
        parts = param_name.split(".bias")
        if len(parts) == 2 and parts[1].isdigit():
            local_expert_id = int(parts[1])
            global_expert_id = local_expert_id + expert_offset
            return f"{parts[0]}.bias{global_expert_id}"

    return param_name
