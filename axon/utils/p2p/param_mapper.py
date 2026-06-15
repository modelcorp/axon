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
from typing import Any

import torch
from megatron.core import mpu
from megatron.core.transformer import TransformerConfig

from .megatron_utils import adjust_param_ep, adjust_param_pp_vpp, extract_expert_idx, unwrap_model


def get_megatron_param_metadata(
    model: list[torch.nn.Module], transformer_config: TransformerConfig, model_config: dict[str, Any] = None
) -> list[dict[str, Any]]:
    """
    Get parameter metadata for all parameters on this rank, including sharding information.

    Returns metadata about each parameter including:
    - Parameter name (global, Megatron format)
    - HF/vLLM parameter name (after conversion)
    - Local shape (shape on this rank)
    - Full shape (global shape before sharding)
    - Split dimension (which dimension is sharded, -1 if not sharded)
    - Split index (which shard this rank holds)
    - Tensor reference

    Args:
        model: List of model modules (may contain multiple for VPP)
        transformer_config: TransformerConfig needed for layer offset calculation
        model_config: Model config (needed for num_moe_experts if EP > 1)

    Returns:
        List of dicts with keys: 'megatron_name', 'hf_vllm_name', 'local_shape', 'full_shape',
                                  'split_dim', 'split_idx', 'tensor', 'dtype'
    """
    from megatron.core.tensor_parallel.layers import (
        ColumnParallelLinear,
        RowParallelLinear,
        VocabParallelEmbedding,
    )

    # WORKAROUND(megatron-core/TE): TE's GroupedLinear per-expert weight tensors
    # (weight0, weight1, ...) are ETP-sharded but lack tensor_model_parallel=True.
    # This causes Method 1 to miss them. Used by Method 3 below.
    #
    # TO REMOVE: When upgrading Megatron-Core / TransformerEngine, check if
    # TEColumnParallelGroupedLinear.weight0 now has tensor_model_parallel=True.
    # If yes, Method 1 handles it and this import + Method 3 can be deleted.
    # Test: run P2P with ETP>1 and verify the [ETP Debug] log below shows
    # tensor_model_parallel=True for expert params.
    try:
        from megatron.core.extensions.transformer_engine import (
            TEColumnParallelGroupedLinear,
            TERowParallelGroupedLinear,
        )
    except ImportError:
        TEColumnParallelGroupedLinear = None
        TERowParallelGroupedLinear = None

    metadata_list = []

    # Get current parallel ranks
    pp_rank = mpu.get_pipeline_model_parallel_rank()
    ep_rank = mpu.get_expert_model_parallel_rank()
    ep_size = mpu.get_expert_model_parallel_world_size()
    tp_rank = mpu.get_tensor_model_parallel_rank()
    tp_size = mpu.get_tensor_model_parallel_world_size()
    etp_rank = mpu.get_expert_tensor_parallel_rank()
    etp_size = mpu.get_expert_tensor_parallel_world_size()

    # Calculate expert offset if using EP
    expert_offset = 0
    num_experts_per_rank = 0
    if ep_size > 1:
        # Try multiple ways to get num_moe_experts (depends on config type)
        # Priority: transformer_config (Megatron-Core config) > model_config (HF config)
        num_experts = getattr(transformer_config, "num_moe_experts", None)

        if not num_experts and model_config is not None:
            # Fallback to model_config with various attribute names
            num_experts = (
                getattr(model_config, "num_moe_experts", None)
                or getattr(model_config, "num_local_experts", None)
                or getattr(model_config, "num_experts", None)
                or getattr(model_config, "n_routed_experts", None)
                or (
                    model_config.get("moe_config", {}).get("num_experts", None)
                    if isinstance(model_config, dict)
                    else None
                )
            )

        if num_experts:
            num_experts_per_rank = num_experts // ep_size
            expert_offset = ep_rank * num_experts_per_rank

    # Handle VPP: megatron_model is a list of sub-models
    for vpp_idx, sub_model in enumerate(model):
        # Unwrap DDP/Float16Module wrappers to get to actual model
        unwrapped = unwrap_model(sub_model)

        # Track existing keys to find extra_keys later
        existing_keys = set()

        # Iterate through named parameters with their parent modules
        for module_name, module in unwrapped.named_modules():
            for param_name, param_tensor in module.named_parameters(recurse=False):
                # Construct full parameter name
                full_param_name = f"{module_name}.{param_name}" if module_name else param_name
                existing_keys.add(full_param_name)

                # Skip internal state
                if "_extra_state" in full_param_name:
                    continue

                # Get global name with PP/VPP/EP adjustments
                global_name = adjust_param_pp_vpp(full_param_name, pp_rank, vpp_idx, transformer_config)

                # Strip "module." prefix if present (DDP wrapper artifact)
                while global_name.startswith("module."):
                    global_name = global_name[len("module.") :]

                # Promote EP-local expert ids to global ids in the param name, across the
                # three expert layouts: `.mlp.experts.` (GroupedMLP), `.local_experts.`
                # (SequentialMLP), and `.experts.linear_fc{i}.weight{j}` (TEGroupedMLP,
                # e.g. Gemma4). Without the last, every EP rank emits colliding local ids
                # 0..(local-1), so the actor names only a subset of vLLM's per-expert
                # params and the routing-table name check fails.
                if expert_offset > 0 and (
                    ".mlp.experts." in global_name
                    or ".local_experts." in global_name
                    or ".experts.linear_fc" in global_name
                ):
                    global_name = adjust_param_ep(global_name, expert_offset)

                # Determine sharding information based on module type and tensor attributes
                local_shape = tuple(param_tensor.shape)
                full_shape = local_shape  # Default: not sharded
                split_dim = -1
                split_idx = 0

                # Method 1: Check tensor's built-in parallel attributes (most reliable)
                # Megatron sets these attributes on parameters during initialization.
                # Skip tensors marked as replicated vision weights (set by the
                # Qwen2.5-VL mbridge patch for projection params that use a single-rank
                # TP group and are NOT actually sharded).
                is_replicated_vision = getattr(param_tensor, "is_replicated_vision_weight", False)
                if (
                    not is_replicated_vision
                    and hasattr(param_tensor, "tensor_model_parallel")
                    and param_tensor.tensor_model_parallel
                ):
                    partition_dim = getattr(param_tensor, "partition_dim", -1)

                    # Special case: DeepSeek V3 MLA down projections (q_a_proj, kv_a_proj_with_mqa)
                    # TE's Linear has a bug where it always sets tensor_model_parallel=True even for
                    # duplicated mode. These params are actually REPLICATED in vLLM, so override.
                    is_mla_down_proj = (
                        "linear_q_down_proj" in full_param_name or "linear_kv_down_proj" in full_param_name
                    )

                    if is_mla_down_proj:
                        # Treat as replicated (not TP-sharded) for vLLM compatibility
                        split_dim = -1
                        split_idx = 0
                        full_shape = local_shape
                    elif partition_dim != -1:
                        # Determine which parallel group this parameter belongs to.
                        # Gemma4-style SequentialMLP exposes experts under
                        # `.local_experts.`; either path uses the ETP group.
                        is_expert_param = (
                            ".mlp.experts." in global_name or ".local_experts." in global_name
                        )

                        if is_expert_param:
                            # Expert parameters use ETP group
                            split_idx = etp_rank
                            parallel_size = etp_size

                            # When ETP=1, expert parameters are NOT sharded (replicated)
                            if etp_size == 1:
                                split_dim = -1
                                split_idx = 0
                                full_shape = local_shape
                            else:
                                # Hardcode split dim for 'experts.linear_fc2' due to bug with Megatron.
                                split_dim = partition_dim if ".experts.linear_fc2." not in global_name else 1
                                full_shape = list(local_shape)
                                full_shape[split_dim] = full_shape[split_dim] * parallel_size
                                full_shape = tuple(full_shape)
                            # Debug removed (moved below)
                        else:
                            # Regular TP parameters use TP group
                            split_idx = tp_rank
                            parallel_size = tp_size
                            split_dim = partition_dim

                            # Calculate full shape
                            full_shape = list(local_shape)
                            full_shape[partition_dim] = full_shape[partition_dim] * parallel_size
                            full_shape = tuple(full_shape)

                # Method 2: Fallback to module type checking (for parameters without attributes)
                elif isinstance(module, VocabParallelEmbedding):
                    # VocabParallelEmbedding shards vocabulary dimension (dim 0)
                    if "weight" in param_name:
                        split_dim = 0
                        split_idx = tp_rank
                        full_shape = (local_shape[0] * tp_size, *local_shape[1:])

                elif isinstance(module, ColumnParallelLinear):
                    # ColumnParallelLinear shards output dimension (dim 0 for both weight and bias)
                    is_expert_param = ".mlp.experts." in global_name or ".local_experts." in global_name

                    if "weight" in param_name:
                        if is_expert_param:
                            split_dim = -1
                            split_idx = 0
                            full_shape = local_shape
                            if etp_size > 1:
                                split_dim = 0
                                split_idx = etp_rank
                                parallel_size = etp_size
                                full_shape = (local_shape[0] * parallel_size, *local_shape[1:])
                        else:
                            # Non-expert parameters use TP group
                            split_dim = 0
                            split_idx = tp_rank
                            parallel_size = tp_size
                            full_shape = (local_shape[0] * parallel_size, *local_shape[1:])
                    elif "bias" in param_name:
                        if is_expert_param:
                            if etp_size == 1:
                                split_dim = -1
                                split_idx = 0
                                full_shape = local_shape
                            else:
                                split_dim = 0
                                split_idx = etp_rank
                                parallel_size = etp_size
                                full_shape = (local_shape[0] * parallel_size,)
                        else:
                            split_dim = 0
                            split_idx = tp_rank
                            parallel_size = tp_size
                            full_shape = (local_shape[0] * parallel_size,)

                elif isinstance(module, RowParallelLinear):
                    # RowParallelLinear shards input dimension (dim 1 for weight)
                    # Bias in RowParallelLinear is NOT sharded (replicated)
                    is_expert_param = ".mlp.experts." in global_name or ".local_experts." in global_name

                    if "weight" in param_name:
                        if is_expert_param:
                            # Expert parameters use ETP group
                            if etp_size == 1:
                                # Not sharded when ETP=1
                                split_dim = -1
                                split_idx = 0
                                full_shape = local_shape
                            else:
                                split_dim = 1
                                split_idx = etp_rank
                                parallel_size = etp_size
                                full_shape = (local_shape[0], local_shape[1] * parallel_size)
                        else:
                            # Non-expert parameters use TP group
                            split_dim = 1
                            split_idx = tp_rank
                            parallel_size = tp_size
                            full_shape = (local_shape[0], local_shape[1] * parallel_size)

                # Method 3: Detect ETP-sharded TE GroupedLinear expert params.
                # These are ETP-sharded but tensor_model_parallel=False, so Methods 1 & 2
                # miss them. P2P does not yet support ETP > 1 — raise a clear error.
                if split_dim == -1 and etp_size > 1 and ".mlp.experts." in global_name and "weight" in param_name:
                    detected = False

                    # 3a: isinstance on known TE grouped linear types
                    if TEColumnParallelGroupedLinear is not None and isinstance(module, TEColumnParallelGroupedLinear):
                        detected = True
                    elif TERowParallelGroupedLinear is not None and isinstance(module, TERowParallelGroupedLinear):
                        detected = True

                    # 3b: Config-based fallback — match local dim against moe_ffn_hidden_size / etp
                    if not detected:
                        ffn_hidden = getattr(transformer_config, "moe_ffn_hidden_size", None) or getattr(
                            transformer_config, "ffn_hidden_size", None
                        )
                        if ffn_hidden:
                            for dim_idx, dim_size in enumerate(local_shape):
                                if dim_size == ffn_hidden // etp_size or dim_size == 2 * ffn_hidden // etp_size:
                                    detected = True
                                    break

                    if detected:
                        raise AssertionError(
                            f"P2P disaggregated engine does not support expert_tensor_parallel_size > 1 "
                            f"(detected etp_size={etp_size} for '{global_name}'). "
                            f"TE GroupedLinear expert weights require all-gather reconstruction that "
                            f"exceeds GPU memory. Workaround: set expert_tensor_parallel_size=1, or "
                            f"use hybrid_engine=true. Future: streaming P2P reconstruction needed to "
                            f"support ETP > 1."
                        )

                # Extract expert index from parameter name
                expert_idx = extract_expert_idx(global_name)

                metadata = {
                    "name": global_name,
                    "local_shape": local_shape,
                    "full_shape": full_shape,
                    "split_dim": split_dim,
                    "split_idx": split_idx,
                    "tensor": param_tensor,
                    "dtype": param_tensor.dtype,
                    "expert_idx": expert_idx,
                }
                metadata_list.append(metadata)

        # Handle extra_keys from state_dict that aren't in named_parameters
        state_dict = unwrapped.state_dict()
        extra_keys = [k for k in state_dict.keys() if "_extra_state" not in k and k not in existing_keys]

        for param_name in extra_keys:
            param_tensor = state_dict[param_name]

            # Apply same transformations as above
            global_name = adjust_param_pp_vpp(param_name, pp_rank, vpp_idx, transformer_config)

            while global_name.startswith("module."):
                global_name = global_name[len("module.") :]

            if expert_offset > 0 and ".mlp.experts." in global_name:
                global_name = adjust_param_ep(global_name, expert_offset)

            # These extra keys are typically not sharded (like expert router bias)
            local_shape = tuple(param_tensor.shape)

            # Extract expert index from parameter name
            expert_idx = extract_expert_idx(global_name)

            metadata = {
                "name": global_name,
                "local_shape": local_shape,
                "full_shape": local_shape,
                "split_dim": -1,
                "split_idx": 0,
                "tensor": param_tensor,
                "dtype": param_tensor.dtype,
                "expert_idx": expert_idx,
            }
            metadata_list.append(metadata)

    return metadata_list


def convert_hf_to_vllm_param_name(name: str) -> str:
    """Convert HF unfused parameter names to vLLM fused parameter names.

    HF uses separate projections (q_proj, k_proj, v_proj, gate_proj, up_proj)
    while vLLM uses fused projections (qkv_proj, gate_up_proj) for efficiency.

    Args:
        name: Parameter name from HF checkpoint

    Returns:
        Converted name for vLLM
    """

    if ".experts." in name:
        # Handle expert params with or without .weight suffix.
        # Megatron/mbridge includes .weight (e.g. "experts.0.gate_up_proj.weight"),
        # FSDP/HF grouped experts may omit it (e.g. "experts.gate_up_proj").
        for src, dst in [
            (".gate_proj", ".w13_weight"),
            (".up_proj", ".w13_weight"),
            (".gate_up_proj", ".w13_weight"),
            (".down_proj", ".w2_weight"),
        ]:
            # Match "src.weight" or "src" at end of name
            if f"{src}.weight" in name:
                return name.replace(f"{src}.weight", dst)
            elif name.endswith(src):
                return name.replace(src, dst)
        # Pass through other expert params (router, norms, biases) unchanged
        return name
    # Handle QKV fusion: q_proj, k_proj, v_proj → qkv_proj
    if ".q_proj." in name:
        return name.replace(".q_proj.", ".qkv_proj.")
    elif ".k_proj." in name:
        return name.replace(".k_proj.", ".qkv_proj.")
    elif ".v_proj." in name:
        return name.replace(".v_proj.", ".qkv_proj.")

    # Handle gate_up fusion: gate_proj, up_proj → gate_up_proj
    elif ".gate_proj." in name:
        return name.replace(".gate_proj.", ".gate_up_proj.")
    elif ".up_proj." in name:
        return name.replace(".up_proj.", ".gate_up_proj.")

    # No conversion needed for other parameters
    else:
        return name
