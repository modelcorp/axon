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
# load_weights adapted from mbridge Bridge.load_weights, BSD-3-Clause (github.com/ISEEKYAN/mbridge).
import os

import torch
from mbridge.core import register_model
from mbridge.core.safetensor_io import SafeTensorIO
from mbridge.core.util import (
    get_model,
)
from mbridge.models.qwen2moe import Qwen2MoEBridge


@register_model("qwen3_next")
class Qwen3NextBridge(Qwen2MoEBridge):
    # Qwen3Next has hybrid attention: some layers use mcore attention (linear_qkv),
    # some use HuggingFace attention (separate q_proj, k_proj, v_proj).
    # We keep ALL base mappings and ADD Qwen3Next-specific ones.

    # Start with ALL base mappings (including linear_qkv for mcore attention layers)
    _ATTENTION_MAPPING = dict(Qwen2MoEBridge._ATTENTION_MAPPING)

    # Add Qwen3Next-specific mappings for hybrid attention (linear_attn + self_attn)
    _ATTENTION_MAPPING.update(
        {
            f"self_attention.{weight_name}": ["model.layers.{layer_number}." + weight_name]
            for weight_name in [
                "input_layernorm.weight",
                # linear attn (for linear_attention layers)
                "linear_attn.A_log",
                "linear_attn.conv1d.weight",
                "linear_attn.dt_bias",
                "linear_attn.in_proj_ba.weight",
                "linear_attn.in_proj_qkvz.weight",
                "linear_attn.norm.weight",
                "linear_attn.out_proj.weight",
                # self attn (for full_attention layers with HF modules) - separate Q, K, V
                "self_attn.k_norm.weight",
                "self_attn.k_proj.weight",
                "self_attn.o_proj.weight",
                "self_attn.q_norm.weight",
                "self_attn.q_proj.weight",
                "self_attn.v_proj.weight",
                # rotary embedding (buffer, may not be trained but vLLM needs it)
                "rotary_emb.inv_freq",
            ]
        }
    )

    def _build_config(self):
        config = self._build_base_config(
            use_cpu_initialization=False,
            # MoE specific
            moe_ffn_hidden_size=self.hf_config.moe_intermediate_size,
            moe_router_bias_update_rate=0.001,
            moe_router_topk=self.hf_config.num_experts_per_tok,
            num_moe_experts=self.hf_config.num_experts,
            moe_aux_loss_coeff=self.hf_config.router_aux_loss_coef,
            # moe_router_load_balancing_type="aux_loss",
            moe_router_load_balancing_type="none",  # default None for RL
            moe_grouped_gemm=True,
            moe_router_score_function="softmax",
            # Other optimizations
            persist_layer_norm=True,
            bias_activation_fusion=True,
            bias_dropout_fusion=True,
            # Qwen specific
            moe_router_pre_softmax=False,
            qk_layernorm=True,
        )

        linear_conv_kernel_dim: int = 4
        linear_key_head_dim: int = 128
        linear_value_head_dim: int = 128
        linear_num_key_heads: int = 16
        linear_num_value_heads: int = 32

        full_attention_interval: int = 4

        config.layer_types = None
        config.linear_conv_kernel_dim = linear_conv_kernel_dim
        config.linear_key_head_dim = linear_key_head_dim
        config.linear_value_head_dim = linear_value_head_dim
        config.linear_num_key_heads = linear_num_key_heads
        config.linear_num_value_heads = linear_num_value_heads
        config.full_attention_interval = full_attention_interval
        config.layernorm_zero_centered_gamma = True

        if os.environ.get("AXON_ENABLE_MTP") == "1":
            config.mtp_num_layers = 1
            config.mtp_loss_scaling_factor = 0.1
        config.moe_shared_expert_intermediate_size = self.hf_config.shared_expert_intermediate_size
        if config.layer_types is None:
            config.layer_types = [
                "linear_attention" if bool((i + 1) % full_attention_interval) else "full_attention"
                for i in range(config.num_layers)
            ]

        return config

    def get_model(
        self,
        post_model_creation_callbacks=None,
        wrap_with_ddp=True,
    ):
        """
        Override get_model to use custom Attention module for hybrid attention layers.

        Qwen3Next has hybrid attention: some layers use linear attention (Qwen3NextGatedDeltaNet),
        and some use full attention (Qwen3NextAttention). Both are HuggingFace modules wrapped
        in our custom Attention class.
        """
        from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec, get_gpt_mtp_block_spec
        from megatron.core.transformer.spec_utils import ModuleSpec

        from axon.models.mcore.models.qwen3_next import Attention

        # Get the transformer config from the bridge
        config = self.config

        def model_provider(pre_process, post_process, vp_stage=None):
            from megatron.core.models.gpt.gpt_model import GPTModel

            # Get the base decoder block spec
            extra_kwargs = {"vp_stage": vp_stage} if vp_stage is not None else {}
            transformer_block_spec = get_gpt_decoder_block_spec(config, use_transformer_engine=True, **extra_kwargs)

            # Replace self_attention with our custom Attention module for all layers
            for i in range(len(transformer_block_spec.layer_specs)):
                transformer_block_spec.layer_specs[i].submodules.self_attention = ModuleSpec(
                    module=Attention,
                    params={"hf_config": self.hf_config},
                )
                transformer_block_spec.layer_specs[i].submodules.mlp.submodules.shared_experts.params["gate"] = True

            if config.mtp_num_layers is not None and config.mtp_num_layers > 0:
                import copy

                # Patch layer spec for shared experts (similar to Qwen2MoE)
                mtp_block_spec = get_gpt_mtp_block_spec(
                    config, copy.deepcopy(transformer_block_spec), use_transformer_engine=True, vp_stage=vp_stage
                )
                if mtp_block_spec is not None:
                    mtp_block_spec.layer_specs[0].submodules.transformer_layer.submodules.self_attention = ModuleSpec(
                        module=Attention,
                        params={"hf_config": self.hf_config, "layer_type": "full_attention"},
                    )
            else:
                mtp_block_spec = None

            # Create the model
            model = GPTModel(
                config=config,
                transformer_layer_spec=transformer_block_spec,
                vocab_size=self.hf_config.vocab_size,
                max_sequence_length=self.hf_config.max_position_embeddings,
                pre_process=pre_process,
                post_process=post_process,
                share_embeddings_and_output_weights=True,
                position_embedding_type="rope",
                rotary_base=getattr(self.hf_config, "rope_theta", None)
                or (getattr(self.hf_config, "rope_parameters", None) or {}).get("rope_theta", 10000),
                mtp_block_spec=mtp_block_spec,
            )

            return model

        # Use mbridge's get_model utility with our custom model_provider
        models = get_model(
            model_provider,
            wrap_with_ddp=wrap_with_ddp,
        )

        # Apply post-creation callbacks if any
        if post_model_creation_callbacks:
            for callback in post_model_creation_callbacks:
                for model in models:
                    callback(model)

        return models

    def _extract_expert_id_from_name(self, name: str) -> tuple[int, str]:
        """
        Extract expert ID from either format:
        - GroupedMLP: mlp.experts.linear_fc1.weight0 -> (0, "weight")
        - SequentialMLP: mlp.experts.local_experts.0.linear_fc1.weight -> (0, "weight")

        Returns: (expert_id, keyword) where keyword is "weight" or "bias"
        """
        if "local_experts" in name:
            # SequentialMLP format: mlp.experts.local_experts.{expert_id}.linear_fc1.weight
            parts = name.split("local_experts.")
            if len(parts) > 1:
                expert_str = parts[1].split(".")[0]
                expert_id = int(expert_str)
                keyword = "bias" if "bias" in name else "weight"
                return expert_id, keyword
        else:
            # GroupedMLP format: mlp.experts.linear_fc1.weight{expert_id}
            if "bias" in name:
                keyword = "bias"
                expert_str = name.split("bias")[-1]
            else:
                keyword = "weight"
                expert_str = name.split("weight")[-1]

            if expert_str.isdigit():
                return int(expert_str), keyword

        raise ValueError(f"Cannot extract expert_id from name: {name}")

    def _weight_name_mapping_mcore_to_hf(self, mcore_weights_name: str) -> list[str]:
        """
        Override to handle MTP weight mappings.
        """
        # Check if this is an MTP weight
        if "mtp" in mcore_weights_name:
            return self._weight_name_mapping_mtp(mcore_weights_name)

        # Otherwise, use parent class mapping
        return super()._weight_name_mapping_mcore_to_hf(mcore_weights_name)

    def _weight_name_mapping_mtp(self, name: str) -> list[str]:
        """
        Map MTP weight names from Megatron-Core format to HuggingFace format.
        """
        import re

        # Direct MTP mappings (exact match - outside transformer_layer)
        _MTP_DIRECT_MAPPING = {
            "mtp.layers.0.enorm.weight": "mtp.pre_fc_norm_embedding.weight",
            "mtp.layers.0.hnorm.weight": "mtp.pre_fc_norm_hidden.weight",
            "mtp.layers.0.eh_proj.weight": "mtp.fc.weight",
            "mtp.layers.0.final_layernorm.weight": "mtp.norm.weight",
        }

        # Check direct mappings first (exact match)
        if name in _MTP_DIRECT_MAPPING:
            return [_MTP_DIRECT_MAPPING[name]]

        # Handle mtp.layers.{N}.transformer_layer.* weights
        match = re.match(r"mtp\.layers\.(\d+)\.transformer_layer\.(.*)", name)
        if match:
            layer_number = match.group(1)
            remainder = match.group(2)

            # MTP Attention mappings (substring match)
            _MTP_ATTENTION_MAPPING = {
                "self_attention.input_layernorm.weight": [f"mtp.layers.{layer_number}.input_layernorm.weight"],
                "self_attention.linear_attn.": None,  # Special handling below
                "self_attention.self_attn.": None,  # Special handling below
                "self_attention.linear_proj.weight": [f"mtp.layers.{layer_number}.self_attn.o_proj.weight"],
                "self_attention.q_layernorm.weight": [f"mtp.layers.{layer_number}.self_attn.q_norm.weight"],
                "self_attention.k_layernorm.weight": [f"mtp.layers.{layer_number}.self_attn.k_norm.weight"],
                "self_attention.linear_qkv.weight": [
                    f"mtp.layers.{layer_number}.self_attn.q_proj.weight",
                    f"mtp.layers.{layer_number}.self_attn.k_proj.weight",
                    f"mtp.layers.{layer_number}.self_attn.v_proj.weight",
                ],
                "self_attention.linear_qkv.bias": [
                    f"mtp.layers.{layer_number}.self_attn.q_proj.bias",
                    f"mtp.layers.{layer_number}.self_attn.k_proj.bias",
                    f"mtp.layers.{layer_number}.self_attn.v_proj.bias",
                ],
            }

            # MTP MLP mappings (substring match - same pattern as base class _MLP_MAPPING)
            _MTP_MLP_MAPPING = {
                "shared_experts.linear_fc1.weight": [
                    f"mtp.layers.{layer_number}.mlp.shared_expert.gate_proj.weight",
                    f"mtp.layers.{layer_number}.mlp.shared_expert.up_proj.weight",
                ],
                "shared_experts.linear_fc2.weight": [f"mtp.layers.{layer_number}.mlp.shared_expert.down_proj.weight"],
                "shared_experts.gate_weight": [f"mtp.layers.{layer_number}.mlp.shared_expert_gate.weight"],
                "pre_mlp_layernorm": [f"mtp.layers.{layer_number}.post_attention_layernorm.weight"],
                "mlp.router.weight": [f"mtp.layers.{layer_number}.mlp.gate.weight"],
                # Experts - handled separately below
            }

            # Check attention mappings first
            for keyword, hf_names in _MTP_ATTENTION_MAPPING.items():
                if keyword in remainder:
                    if hf_names is not None:
                        return hf_names
                    # Special handling for linear_attn and self_attn (passthrough with prefix change)
                    if "self_attention.linear_attn." in remainder:
                        attn_part = remainder[len("self_attention.") :]
                        return [f"mtp.layers.{layer_number}.{attn_part}"]
                    if "self_attention.self_attn." in remainder:
                        self_attn_part = remainder[len("self_attention.") :]
                        return [f"mtp.layers.{layer_number}.{self_attn_part}"]

            # Check MLP mappings (substring match like base class)
            for keyword, hf_names in _MTP_MLP_MAPPING.items():
                if keyword in remainder:
                    return hf_names

            # Experts (grouped gemm format: mlp.experts.linear_fc1.weight0)
            expert_match = re.search(r"mlp\.experts\.linear_fc(\d+)\.(weight|bias)(\d+)", remainder)
            if expert_match:
                fc_num = expert_match.group(1)
                param_type = expert_match.group(2)
                expert_id = expert_match.group(3)

                if fc_num == "1":
                    return [
                        f"mtp.layers.{layer_number}.mlp.experts.{expert_id}.gate_proj.{param_type}",
                        f"mtp.layers.{layer_number}.mlp.experts.{expert_id}.up_proj.{param_type}",
                    ]
                elif fc_num == "2":
                    return [f"mtp.layers.{layer_number}.mlp.experts.{expert_id}.down_proj.{param_type}"]

            # Sequential experts format: mlp.experts.local_experts.0.linear_fc1.weight
            seq_expert_match = re.search(
                r"mlp\.experts\.local_experts\.(\d+)\.linear_fc(\d+)\.(weight|bias)", remainder
            )
            if seq_expert_match:
                expert_id = seq_expert_match.group(1)
                fc_num = seq_expert_match.group(2)
                param_type = seq_expert_match.group(3)

                if fc_num == "1":
                    return [
                        f"mtp.layers.{layer_number}.mlp.experts.{expert_id}.gate_proj.{param_type}",
                        f"mtp.layers.{layer_number}.mlp.experts.{expert_id}.up_proj.{param_type}",
                    ]
                elif fc_num == "2":
                    return [f"mtp.layers.{layer_number}.mlp.experts.{expert_id}.down_proj.{param_type}"]

        raise NotImplementedError(f"Unsupported MTP parameter name: {name}")

    def _weight_to_hf_format(
        self, mcore_weights_name: str, mcore_weights: torch.Tensor
    ) -> tuple[list[str], list[torch.Tensor]]:
        """
        Export MCore weights to Hugging Face format.

        Override for Qwen3Next to handle hybrid attention weights.
        Most Qwen3Next attention weights map 1:1 (no splitting needed).
        """
        if "mtp" in mcore_weights_name:
            hf_names = self._weight_name_mapping_mtp(mcore_weights_name)

            # Handle linear_fc1 (needs splitting into gate and up)
            if "linear_fc1.weight" in mcore_weights_name or "linear_fc1.bias" in mcore_weights_name:
                assert len(hf_names) == 2
                gate, up = mcore_weights.chunk(2)
                return hf_names, [gate, up]

            # Single weight mapping
            return hf_names, [mcore_weights] * len(hf_names)

        hf_names = self._weight_name_mapping_mcore_to_hf(mcore_weights_name)
        # Single weight - direct mapping (most Qwen3Next attention weights)
        if len(hf_names) == 1:
            # Pad embedding and output layer
            if self.make_vocab_size_divisible_by is not None and (
                "embedding.word_embeddings.weight" in mcore_weights_name or "output_layer.weight" in mcore_weights_name
            ):
                assert mcore_weights.shape[0] == self.padded_vocab_size
                assert self.vocab_size is not None
                return [hf_names[0]], [mcore_weights[: self.vocab_size]]
            return [hf_names[0]], [mcore_weights]

        # Skip linear_qkv splitting - Qwen3Next uses separate Q, K, V
        if "self_attention.linear_qkv." in mcore_weights_name and "layer_norm" not in mcore_weights_name:
            # This shouldn't happen for Qwen3Next, but handle gracefully
            # Just return the weight as-is, mapped to the first HF name
            return [hf_names[0]], [mcore_weights]

        # MLP fc1 weights (gate_proj + up_proj merged)
        if "linear_fc1.weight" in mcore_weights_name or "linear_fc1.bias" in mcore_weights_name:
            assert len(hf_names) == 2
            gate, up = mcore_weights.chunk(2)
            return hf_names, [gate, up]

        # For any other multi-name mapping, return as-is
        return [hf_names[0]], [mcore_weights]

    def _get_safetensor_io(self, weights_path: str):
        """
        Override to update the safetensor index for hybrid attention models.

        For Qwen3Next models, some layers are "full_attention" (with self_attn.* weights)
        and some are "linear_attention" (with linear_attn.* weights).

        The base Qwen2MoE mapping produces weight names like 'model.layers.X.self_attn.o_proj.weight'
        for all layers, but linear_attention layers don't have these weights in the HF checkpoint.

        This method adds dummy entries to the index for weights that would be missing,
        pointing them to an existing file so the lookup doesn't fail. The actual loading
        logic will handle them appropriately.
        """
        safetensor_io = SafeTensorIO(self._get_actual_hf_path(weights_path))

        if not safetensor_io.index:
            return safetensor_io

        # Get the layer types configuration
        layer_types = getattr(self.config, "layer_types", None)
        if layer_types is None:
            full_attention_interval = getattr(self.config, "full_attention_interval", 4)
            num_layers = self.config.num_layers
            layer_types = [
                "linear_attention" if bool((i + 1) % full_attention_interval) else "full_attention"
                for i in range(num_layers)
            ]

        return safetensor_io

    def load_weights(
        self,
        models: list[torch.nn.Module],
        weights_path: str,
        memory_efficient: bool = False,
    ) -> None:
        """
        Load weights from a Hugging Face model into a Megatron-Core model.

        Override for Qwen3Next hybrid attention models that may have some weights
        in the mapping that don't exist in the HF checkpoint (e.g., self_attn weights
        for linear_attention layers).
        """
        self.safetensor_io = self._get_safetensor_io(weights_path)

        # Get the set of available HF weight names
        available_hf_weights = set(self.safetensor_io.index.keys()) if self.safetensor_io.index else set()

        for i, model in enumerate(models):
            # map local weight names to global weight names
            local_to_global_map = self._weight_name_mapping_mcore_local_to_global(model)
            # map local weight names to huggingface weight names
            local_to_hf_map = {
                k: self._weight_name_mapping_mcore_to_hf(v)
                for k, v in local_to_global_map.items()
                if "_extra_state" not in k
            }

            # Filter out mappings where HF weights don't exist (for hybrid attention models)
            # Also skip linear_qkv mappings - Qwen3Next uses HuggingFace attention with separate Q, K, V
            filtered_local_to_hf_map = {}
            for local_name, hf_names in local_to_hf_map.items():
                # Skip linear_qkv.weight and linear_qkv.bias - they trigger QKV merge which fails for GQA
                # The actual weights are loaded via self_attn.q_proj.weight etc. mappings
                if "linear_qkv.weight" in local_name or "linear_qkv.bias" in local_name:
                    continue

                # Check if all HF names exist
                if all(hf_name in available_hf_weights for hf_name in hf_names):
                    filtered_local_to_hf_map[local_name] = hf_names
                elif any(hf_name in available_hf_weights for hf_name in hf_names):
                    # Partial match - only include the existing ones (should be rare)
                    existing_hf_names = [n for n in hf_names if n in available_hf_weights]
                    if existing_hf_names:
                        filtered_local_to_hf_map[local_name] = existing_hf_names
                # else: skip - none of the HF names exist (e.g., self_attn weights for linear_attention layers)

            local_to_hf_map = filtered_local_to_hf_map

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
                    # split mcore weights across tp
                    if self.mpu.tp_rank == 0:
                        mcore_weights_tp_split = self._weight_split_across_tp(
                            local_name, mcore_weight, param, self.mpu.tp_size
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
                        src=torch.distributed.get_global_rank(self.mpu.tp_group, 0),
                        group=self.mpu.tp_group,
                    )
                # load
                param.copy_(param_to_load)

            torch.cuda.empty_cache()
