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
# Adapted from mbridge Bridge.export_weights, BSD-3-Clause (github.com/ISEEKYAN/mbridge).
"""
Monkey patch for export_weights method to be applied to general mbridge bridges.

This patch adds support for:
- MTP weight filtering via export_mtp parameter
- Expert bias keys that aren't in named_parameters
- Expert parallel (EP) and tensor parallel (TP) weight gathering
- Proper weight name mapping and conversion to HuggingFace format

Usage:
    # Option 1: Apply to a specific bridge class
    from mbridge.core.llm_bridge import LLMBridge
    from axon.models.mbridge.export_weights_patch import apply_export_weights_patch

    apply_export_weights_patch(LLMBridge)

    # Option 2: Auto-apply to common bridge classes
    from axon.models.mbridge.export_weights_patch import auto_apply_export_weights_patch

    auto_apply_export_weights_patch()

    # Option 3: Import the bridges module (auto-applies if configured)
    from axon.models.mbridge import auto_apply_export_weights_patch
    auto_apply_export_weights_patch()
"""

from collections.abc import Generator

import torch


def export_weights_patch(
    self,
    models: list[torch.nn.Module],
    export_mtp: bool = False,
) -> Generator[tuple[str, torch.Tensor], None, None]:
    """
    Export weights from Megatron-Core models to HuggingFace format.

    This is a monkey-patched version that handles:
    - MTP weight filtering
    - Expert bias keys missing from named_parameters
    - EP and TP weight gathering

    Args:
        models: List of model instances (one per pipeline stage)
        export_mtp: If True, only export MTP weights. If False, exclude MTP weights.

    Yields:
        Tuple of (weight_name, weight_tensor) in HuggingFace format
    """
    from mbridge.core.util import (
        broadcast_from_megatron_pp,
        broadcast_str_from_megatron_pp,
        unwrap_model,
    )

    models = [unwrap_model(model) for model in models]

    def get_model_chunk_generator():
        for model in models:
            existing_keys = set()
            for name, param in model.named_parameters():
                if (export_mtp and "mtp" not in name) or (not export_mtp and "mtp" in name):
                    continue
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
            if (export_mtp and "mtp" not in name) or (not export_mtp and "mtp" in name):
                continue
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
    torch.distributed.all_gather_object(object_list=weights_names_all_pp, obj=weights_names, group=self.mpu.pp_group)
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
        if ".mlp.experts.linear_fc" in name and self.mpu.ep_size > 1:
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
                yield from zip(converted_names, converted_params, strict=False)
            continue

        # TP
        if hasattr(broad_pp_param, "tensor_model_parallel") and broad_pp_param.tensor_model_parallel:
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

        yield from zip(converted_names, converted_params, strict=False)


def apply_export_weights_patch(bridge_class):
    """
    Apply the export_weights monkey patch to a bridge class.

    Args:
        bridge_class: The bridge class to patch (e.g., LLMBridge, Qwen2MoEBridge, etc.)

    Example:
        from mbridge.core.llm_bridge import LLMBridge
        from axon.models.mbridge.export_weights_patch import apply_export_weights_patch

        apply_export_weights_patch(LLMBridge)
    """
    bridge_class.export_weights = export_weights_patch
    return bridge_class
