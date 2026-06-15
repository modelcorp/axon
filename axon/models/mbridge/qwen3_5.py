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
"""
Qwen3.5 bridge for Megatron-Core (text backbone only, dense and MoE variants).

Qwen3.5 is architecturally always a VL model — even "Qwen3.5-9B" uses
Qwen3_5ForConditionalGeneration with text_config + vision_config. This bridge
unwraps text_config and builds only the text backbone (GPTModel), which is
sufficient for RL post-training on text. For full VL training (image+text),
a VL bridge handling the vision encoder + projector would be needed; mbridge
>= 4389fcc has native VL support but requires mcore with GDN attention
(experimental_attention_variant), which mcore 0.15.0rc7 does not have.

Uses HuggingFace attention modules wrapped in Megatron SP/CP/TP framework,
avoiding the need for mcore's native GDN support.

Supports both:
- qwen3_5 / qwen3_5_text (dense, e.g. Qwen3.5-9B)
- qwen3_5_moe / qwen3_5_moe_text (MoE, e.g. Qwen3.5-35B-A3B)

Based on the Qwen3-Next bridge pattern.
"""

import logging
import os
import re

import torch
from mbridge.core import register_model
from mbridge.core.safetensor_io import SafeTensorIO
from mbridge.core.util import get_model
from mbridge.models.qwen2moe import Qwen2MoEBridge

logger = logging.getLogger(__name__)


class Qwen3_5BridgeBase(Qwen2MoEBridge):
    """Base bridge for Qwen3.5 models (dense and MoE)."""

    def __init__(self, hf_config, **kwargs):
        # Qwen3.5 HF configs are VL wrappers — unwrap to text_config so that
        # _build_base_config can read num_hidden_layers, hidden_size, etc.
        # Store the original VL config for the attention wrapper (it needs text_config path).
        self._original_hf_config = hf_config
        text_config = hf_config.text_config if hasattr(hf_config, "text_config") else hf_config

        # MoE configs don't have intermediate_size (they use moe_intermediate_size).
        # _build_base_config expects it for ffn_hidden_size mapping, so set a fallback.
        if not hasattr(text_config, "intermediate_size"):
            text_config.intermediate_size = getattr(text_config, "moe_intermediate_size", text_config.hidden_size * 4)

        super().__init__(text_config, **kwargs)

    # Qwen3.5 has hybrid attention: some layers use mcore attention (linear_qkv),
    # some use HuggingFace attention (separate q_proj, k_proj, v_proj).
    # We keep ALL base mappings and ADD Qwen3.5-specific ones.
    _ATTENTION_MAPPING = dict(Qwen2MoEBridge._ATTENTION_MAPPING)

    # Dense Qwen3.5 uses standard MLP with linear_fc1 (not MoE), so we need
    # the base MLP mapping for linear_fc1.layer_norm_weight -> post_attention_layernorm.
    # The Qwen2MoE base only has "pre_mlp_layernorm" which is for MoE layers.
    _MLP_MAPPING = dict(Qwen2MoEBridge._MLP_MAPPING)
    _MLP_MAPPING.update(
        {
            "mlp.linear_fc1.layer_norm_weight": ["model.layers.{layer_number}.post_attention_layernorm.weight"],
            "mlp.linear_fc1.weight": [
                "model.layers.{layer_number}.mlp.gate_proj.weight",
                "model.layers.{layer_number}.mlp.up_proj.weight",
            ],
            "mlp.linear_fc2.weight": ["model.layers.{layer_number}.mlp.down_proj.weight"],
        }
    )

    _ATTENTION_MAPPING.update(
        {
            f"self_attention.{weight_name}": ["model.layers.{layer_number}." + weight_name]
            for weight_name in [
                "input_layernorm.weight",
                # GDN linear attention weights
                "linear_attn.A_log",
                "linear_attn.conv1d.weight",
                "linear_attn.conv1d.bias",
                "linear_attn.dt_bias",
                "linear_attn.in_proj_a.weight",
                "linear_attn.in_proj_b.weight",
                "linear_attn.in_proj_qkv.weight",
                "linear_attn.in_proj_z.weight",
                "linear_attn.norm.weight",
                "linear_attn.out_proj.weight",
                # Full attention weights (with gated output)
                "self_attn.k_norm.weight",
                "self_attn.k_proj.weight",
                "self_attn.o_proj.weight",
                "self_attn.q_norm.weight",
                "self_attn.q_proj.weight",
                "self_attn.v_proj.weight",
                # Rotary embedding buffer
                "rotary_emb.inv_freq",
            ]
        }
    )

    def _build_config(self):
        tc = self.hf_config

        build_kwargs = dict(
            use_cpu_initialization=False,
            persist_layer_norm=True,
            bias_activation_fusion=True,
            bias_dropout_fusion=True,
            qk_layernorm=True,
            moe_router_load_balancing_type="none",  # default None for RL
        )

        # MoE-specific config (only for MoE variants)
        if hasattr(tc, "num_experts") and tc.num_experts > 1:
            build_kwargs.update(
                moe_ffn_hidden_size=tc.moe_intermediate_size,
                moe_router_bias_update_rate=0.001,
                moe_router_topk=tc.num_experts_per_tok,
                num_moe_experts=tc.num_experts,
                moe_aux_loss_coeff=tc.router_aux_loss_coef,
                moe_grouped_gemm=True,
                moe_router_score_function="softmax",
                moe_router_pre_softmax=False,
            )
            if hasattr(tc, "shared_expert_intermediate_size"):
                build_kwargs["moe_shared_expert_intermediate_size"] = tc.shared_expert_intermediate_size

        # Mcore requires num_query_groups % TP == 0, but we use HF attention modules
        # (not mcore's native attention), so this constraint is artificial. Override
        # num_key_value_heads to num_attention_heads to satisfy the validator — the
        # actual GQA head count is handled by the HF modules directly.
        if tc.num_key_value_heads < tc.num_attention_heads:
            build_kwargs["num_query_groups"] = tc.num_attention_heads

        config = self._build_base_config(**build_kwargs)

        # Linear attention config from HF
        config.linear_conv_kernel_dim = tc.linear_conv_kernel_dim
        config.linear_key_head_dim = tc.linear_key_head_dim
        config.linear_value_head_dim = tc.linear_value_head_dim
        config.linear_num_key_heads = tc.linear_num_key_heads
        config.linear_num_value_heads = tc.linear_num_value_heads
        config.full_attention_interval = tc.full_attention_interval
        config.layernorm_zero_centered_gamma = True

        # Layer types — use HF config directly if available
        config.layer_types = getattr(tc, "layer_types", None)
        if config.layer_types is None:
            config.layer_types = [
                "linear_attention" if bool((i + 1) % tc.full_attention_interval) else "full_attention"
                for i in range(config.num_layers)
            ]

        # MTP
        if os.environ.get("AXON_ENABLE_MTP") == "1":
            mtp_layers = getattr(tc, "mtp_num_hidden_layers", 0)
            if mtp_layers > 0:
                config.mtp_num_layers = mtp_layers
                config.mtp_loss_scaling_factor = 0.1

        return config

    def _get_gptmodel_args(self):
        tc = self.hf_config
        # Handle rope_theta in both transformers 4.x and 5.x formats
        rope_params = getattr(tc, "rope_parameters", None) or {}
        rope_theta = rope_params.get("rope_theta", getattr(tc, "rope_theta", 10000000))
        return dict(
            vocab_size=tc.vocab_size,
            max_sequence_length=tc.max_position_embeddings,
            position_embedding_type="rope",
            rotary_base=rope_theta,
        )

    def get_model(self, post_model_creation_callbacks=None, wrap_with_ddp=True, **kwargs):
        from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec, get_gpt_mtp_block_spec
        from megatron.core.transformer.spec_utils import ModuleSpec

        from axon.models.mcore.models.qwen3_5 import Attention

        config = self.config

        def model_provider(pre_process, post_process, vp_stage=None):
            from megatron.core.models.gpt.gpt_model import GPTModel

            extra_kwargs = {"vp_stage": vp_stage} if vp_stage is not None else {}
            transformer_block_spec = get_gpt_decoder_block_spec(config, use_transformer_engine=True, **extra_kwargs)

            # Replace self_attention with our custom Attention module for all layers.
            # Also replace input_layernorm with IdentityOp — our HuggingfaceAttention
            # has its own input_layernorm (HF Qwen3_5RMSNorm), so we disable the mcore
            # TransformerLayer's layernorm to avoid double normalization.
            from megatron.core.transformer.identity_op import IdentityOp

            for i in range(len(transformer_block_spec.layer_specs)):
                transformer_block_spec.layer_specs[i].submodules.self_attention = ModuleSpec(
                    module=Attention,
                    params={"hf_config": self.hf_config},
                )
                transformer_block_spec.layer_specs[i].submodules.input_layernorm = IdentityOp
                # Enable shared expert gate if MoE
                if hasattr(transformer_block_spec.layer_specs[i].submodules, "mlp"):
                    mlp_spec = transformer_block_spec.layer_specs[i].submodules.mlp
                    if hasattr(mlp_spec, "submodules") and hasattr(mlp_spec.submodules, "shared_experts"):
                        mlp_spec.submodules.shared_experts.params["gate"] = True

            # MTP
            if config.mtp_num_layers is not None and config.mtp_num_layers > 0:
                import copy

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

            gptmodel_args = self._get_gptmodel_args()
            model = GPTModel(
                config=config,
                transformer_layer_spec=transformer_block_spec,
                pre_process=pre_process,
                post_process=post_process,
                share_embeddings_and_output_weights=getattr(self.hf_config, "tie_word_embeddings", False),
                mtp_block_spec=mtp_block_spec,
                **gptmodel_args,
            )
            return model

        models = get_model(model_provider, wrap_with_ddp=wrap_with_ddp)

        if post_model_creation_callbacks:
            for callback in post_model_creation_callbacks:
                for model in models:
                    callback(model)

        return models

    def _weight_name_mapping_mcore_to_hf(self, mcore_weights_name: str) -> list[str]:
        if "mtp" in mcore_weights_name:
            return self._weight_name_mapping_mtp(mcore_weights_name)
        return super()._weight_name_mapping_mcore_to_hf(mcore_weights_name)

    def _weight_name_mapping_mtp(self, name: str) -> list[str]:
        """Map MTP weight names from Megatron-Core to HuggingFace format."""
        _MTP_DIRECT_MAPPING = {
            "mtp.layers.0.enorm.weight": "mtp.pre_fc_norm_embedding.weight",
            "mtp.layers.0.hnorm.weight": "mtp.pre_fc_norm_hidden.weight",
            "mtp.layers.0.eh_proj.weight": "mtp.fc.weight",
            "mtp.layers.0.final_layernorm.weight": "mtp.norm.weight",
        }

        if name in _MTP_DIRECT_MAPPING:
            return [_MTP_DIRECT_MAPPING[name]]

        match = re.match(r"mtp\.layers\.(\d+)\.transformer_layer\.(.*)", name)
        if match:
            layer_number = match.group(1)
            remainder = match.group(2)

            _MTP_ATTENTION_MAPPING = {
                "self_attention.input_layernorm.weight": [f"mtp.layers.{layer_number}.input_layernorm.weight"],
                "self_attention.linear_attn.": None,
                "self_attention.self_attn.": None,
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

            _MTP_MLP_MAPPING = {
                "shared_experts.linear_fc1.weight": [
                    f"mtp.layers.{layer_number}.mlp.shared_expert.gate_proj.weight",
                    f"mtp.layers.{layer_number}.mlp.shared_expert.up_proj.weight",
                ],
                "shared_experts.linear_fc2.weight": [f"mtp.layers.{layer_number}.mlp.shared_expert.down_proj.weight"],
                "shared_experts.gate_weight": [f"mtp.layers.{layer_number}.mlp.shared_expert_gate.weight"],
                "pre_mlp_layernorm": [f"mtp.layers.{layer_number}.post_attention_layernorm.weight"],
                "mlp.router.weight": [f"mtp.layers.{layer_number}.mlp.gate.weight"],
            }

            for keyword, hf_names in _MTP_ATTENTION_MAPPING.items():
                if keyword in remainder:
                    if hf_names is not None:
                        return hf_names
                    if "self_attention.linear_attn." in remainder:
                        attn_part = remainder[len("self_attention.") :]
                        return [f"mtp.layers.{layer_number}.{attn_part}"]
                    if "self_attention.self_attn." in remainder:
                        self_attn_part = remainder[len("self_attention.") :]
                        return [f"mtp.layers.{layer_number}.{self_attn_part}"]

            for keyword, hf_names in _MTP_MLP_MAPPING.items():
                if keyword in remainder:
                    return hf_names

            # Experts (grouped gemm format)
            expert_match = re.search(r"mlp\.experts\.linear_fc(\d+)\.(weight|bias)(\d+)", remainder)
            if expert_match:
                fc_num, param_type, expert_id = expert_match.groups()
                if fc_num == "1":
                    return [
                        f"mtp.layers.{layer_number}.mlp.experts.{expert_id}.gate_proj.{param_type}",
                        f"mtp.layers.{layer_number}.mlp.experts.{expert_id}.up_proj.{param_type}",
                    ]
                elif fc_num == "2":
                    return [f"mtp.layers.{layer_number}.mlp.experts.{expert_id}.down_proj.{param_type}"]

            # Sequential experts format
            seq_match = re.search(r"mlp\.experts\.local_experts\.(\d+)\.linear_fc(\d+)\.(weight|bias)", remainder)
            if seq_match:
                expert_id, fc_num, param_type = seq_match.groups()
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
        if "mtp" in mcore_weights_name:
            hf_names = self._weight_name_mapping_mtp(mcore_weights_name)
            if "linear_fc1.weight" in mcore_weights_name or "linear_fc1.bias" in mcore_weights_name:
                assert len(hf_names) == 2
                gate, up = mcore_weights.chunk(2)
                return hf_names, [gate, up]
            return hf_names, [mcore_weights] * len(hf_names)

        hf_names = self._weight_name_mapping_mcore_to_hf(mcore_weights_name)

        if len(hf_names) == 1:
            if self.make_vocab_size_divisible_by is not None and (
                "embedding.word_embeddings.weight" in mcore_weights_name or "output_layer.weight" in mcore_weights_name
            ):
                assert mcore_weights.shape[0] == self.padded_vocab_size
                assert self.vocab_size is not None
                return [hf_names[0]], [mcore_weights[: self.vocab_size]]
            return [hf_names[0]], [mcore_weights]

        # Skip linear_qkv splitting — Qwen3.5 uses separate Q, K, V via HF modules
        if "self_attention.linear_qkv." in mcore_weights_name and "layer_norm" not in mcore_weights_name:
            return [hf_names[0]], [mcore_weights]

        # MLP fc1 (gate_proj + up_proj merged)
        if "linear_fc1.weight" in mcore_weights_name or "linear_fc1.bias" in mcore_weights_name:
            assert len(hf_names) == 2
            gate, up = mcore_weights.chunk(2)
            return hf_names, [gate, up]

        return [hf_names[0]], [mcore_weights]

    def _get_safetensor_io(self, weights_path: str):
        return SafeTensorIO(self._get_actual_hf_path(weights_path))

    def _build_fused_expert_map(self, available_hf_weights, hf_name_remap):
        """Build a map from individual expert HF names to (fused_key, expert_id, proj_type).

        Qwen3.5 stores expert weights in fused format:
          model.language_model.layers.X.mlp.experts.gate_up_proj  [E, gate+up, hidden]
          model.language_model.layers.X.mlp.experts.down_proj     [E, hidden, intermediate]

        But the bridge mapping expects individual expert weights:
          model.layers.X.mlp.experts.{id}.gate_proj.weight  [intermediate, hidden]
          model.layers.X.mlp.experts.{id}.up_proj.weight    [intermediate, hidden]
          model.layers.X.mlp.experts.{id}.down_proj.weight  [hidden, intermediate]
        """
        fused_expert_map = {}

        # Find all fused expert keys in the safetensor
        fused_keys = {}
        for avail_key in available_hf_weights:
            if ".mlp.experts.gate_up_proj" in avail_key:
                # Extract layer prefix: model.language_model.layers.X.mlp.experts
                prefix = avail_key.rsplit(".gate_up_proj", 1)[0]
                fused_keys.setdefault(prefix, {})["gate_up"] = avail_key
            elif ".mlp.experts.down_proj" in avail_key and ".shared_expert" not in avail_key:
                prefix = avail_key.rsplit(".down_proj", 1)[0]
                fused_keys.setdefault(prefix, {})["down"] = avail_key

        if not fused_keys:
            return fused_expert_map

        # For each fused key, build mappings for individual expert names
        tc = self.hf_config
        num_experts = getattr(tc, "num_experts", 0)

        for prefix, keys in fused_keys.items():
            # Derive the text-only layer prefix for matching bridge HF names
            # prefix: model.language_model.layers.X.mlp.experts
            # text_prefix: model.layers.X.mlp.experts
            _vl_prefix = "model.language_model."
            _text_prefix = "model."
            if prefix.startswith(_vl_prefix):
                text_prefix = _text_prefix + prefix[len(_vl_prefix) :]
            else:
                text_prefix = prefix

            for expert_id in range(num_experts):
                if "gate_up" in keys:
                    # gate_proj: first half of gate_up_proj[expert_id]
                    gate_name = f"{text_prefix}.{expert_id}.gate_proj.weight"
                    fused_expert_map[gate_name] = (keys["gate_up"], expert_id, "gate")
                    # up_proj: second half of gate_up_proj[expert_id]
                    up_name = f"{text_prefix}.{expert_id}.up_proj.weight"
                    fused_expert_map[up_name] = (keys["gate_up"], expert_id, "up")
                if "down" in keys:
                    down_name = f"{text_prefix}.{expert_id}.down_proj.weight"
                    fused_expert_map[down_name] = (keys["down"], expert_id, "down")

        return fused_expert_map

    def load_weights(
        self,
        models: list[torch.nn.Module],
        weights_path: str,
        memory_efficient: bool = False,
    ) -> None:
        """Load weights with hybrid attention filtering (skip missing HF weights)."""
        self.safetensor_io = self._get_safetensor_io(weights_path)
        available_hf_weights = set(self.safetensor_io.index.keys()) if self.safetensor_io.index else set()

        # Qwen3.5 is a VL model — safetensor keys use "model.language_model.layers.X..."
        # but _ATTENTION_MAPPING produces "model.layers.X...". Build a remap to handle this.
        hf_name_remap = {}
        _vl_prefix = "model.language_model."
        _text_prefix = "model."
        for avail_key in available_hf_weights:
            if avail_key.startswith(_vl_prefix):
                text_key = _text_prefix + avail_key[len(_vl_prefix) :]
                hf_name_remap[text_key] = avail_key

        # Build fused expert map for Qwen3.5's packed expert format
        fused_expert_map = self._build_fused_expert_map(available_hf_weights, hf_name_remap)
        # Cache loaded fused tensors to avoid re-reading from disk
        _fused_tensor_cache = {}

        def _resolve_hf_name(name):
            """Resolve HF name, trying: direct match, VL prefix, fused expert."""
            if name in available_hf_weights:
                return name
            if name in hf_name_remap:
                return hf_name_remap[name]
            # For fused experts, return the original name — we handle loading separately
            if name in fused_expert_map:
                return name  # mark as resolvable
            return None

        def _load_expert_weight_from_fused(name):
            """Load an individual expert weight by slicing from the fused tensor."""
            fused_key, expert_id, proj_type = fused_expert_map[name]
            if fused_key not in _fused_tensor_cache:
                _fused_tensor_cache[fused_key] = self.safetensor_io.load_one_hf_weight(fused_key)
            fused = _fused_tensor_cache[fused_key]
            # fused shape: [num_experts, dim1, dim2]
            expert_weight = fused[expert_id]  # [dim1, dim2]
            if proj_type == "gate":
                # gate_up_proj[expert_id] = [gate_size + up_size, hidden] -> first half
                half = expert_weight.shape[0] // 2
                return expert_weight[:half]
            elif proj_type == "up":
                half = expert_weight.shape[0] // 2
                return expert_weight[half:]
            else:  # down
                return expert_weight

        for i, model in enumerate(models):
            local_to_global_map = self._weight_name_mapping_mcore_local_to_global(model)
            local_to_hf_map = {
                k: self._weight_name_mapping_mcore_to_hf(v)
                for k, v in local_to_global_map.items()
                if "_extra_state" not in k
            }

            # Filter out mappings where HF weights don't exist (hybrid attention)
            # Also skip linear_qkv — Qwen3.5 uses HF attention with separate Q, K, V
            filtered_local_to_hf_map = {}
            skipped_params = []
            for local_name, hf_names in local_to_hf_map.items():
                if "linear_qkv.weight" in local_name or "linear_qkv.bias" in local_name:
                    skipped_params.append((local_name, hf_names, "linear_qkv skip"))
                    continue

                # Resolve HF names (handle VL model prefix + fused experts)
                resolved = [_resolve_hf_name(n) for n in hf_names]
                if all(r is not None for r in resolved):
                    filtered_local_to_hf_map[local_name] = resolved
                elif any(r is not None for r in resolved):
                    existing = [r for r in resolved if r is not None]
                    if existing:
                        filtered_local_to_hf_map[local_name] = existing
                else:
                    skipped_params.append((local_name, hf_names, "not in safetensor"))

            if self.mpu.tp_rank == 0 and self.mpu.pp_rank == 0:
                non_qkv_skipped = [s for s in skipped_params if "linear_qkv" not in s[0]]
                logger.info(
                    f"[bridge-load] Model {i}: {len(filtered_local_to_hf_map)} params to load, "
                    f"{len(skipped_params)} skipped ({len(non_qkv_skipped)} non-trivial)"
                )
                for local_name, hf_names, reason in non_qkv_skipped[:10]:
                    logger.warning(f"[bridge-load] SKIPPED: {local_name} -> {hf_names} ({reason})")
                if len(non_qkv_skipped) > 10:
                    logger.warning(f"[bridge-load] ... and {len(non_qkv_skipped) - 10} more")

            local_to_hf_map = filtered_local_to_hf_map

            to_load_from_disk = []
            for local_name, hf_names in local_to_hf_map.items():
                # Skip fused expert weights — they're loaded separately via _load_expert_weight_from_fused
                if any(n in fused_expert_map for n in hf_names):
                    continue
                if ".mlp.experts.linear_fc" in local_name:
                    if self.mpu.etp_rank == 0:
                        to_load_from_disk.extend(hf_names)
                else:
                    if self.mpu.tp_rank == 0:
                        to_load_from_disk.extend(hf_names)
                    elif "lm_head.weight" in hf_names:
                        to_load_from_disk.extend(hf_names)

            if not memory_efficient:
                hf_weights_map = self.safetensor_io.load_some_hf_weight(to_load_from_disk)

            for local_name, hf_names in local_to_hf_map.items():
                param = model.state_dict()[local_name]

                # Handle fused expert weights (Qwen3.5 packed format)
                if any(n in fused_expert_map for n in hf_names):
                    if self.mpu.etp_rank == 0:
                        hf_weights = [_load_expert_weight_from_fused(n) for n in hf_names]
                        mcore_weight = self._weight_to_mcore_format(local_name, hf_weights)
                    else:
                        mcore_weight = None
                elif set(to_load_from_disk) & set(hf_names):
                    if not memory_efficient:
                        hf_weights = [hf_weights_map[x] for x in hf_names]
                    else:
                        hf_weights = [self.safetensor_io.load_one_hf_weight(x) for x in hf_names]
                    mcore_weight = self._weight_to_mcore_format(local_name, hf_weights)
                else:
                    mcore_weight = None

                if hf_names[0] in {"lm_head.weight", "model.embed_tokens.weight"}:
                    if param.shape[0] == 1 and (mcore_weight is None or mcore_weight.shape[0] != 1):
                        continue

                param_to_load = torch.empty_like(param)
                if ".mlp.experts.linear_fc" in local_name:
                    if self.mpu.etp_rank == 0:
                        splits = self._weight_split_across_tp(local_name, mcore_weight, param, self.mpu.etp_size)
                        splits = [t.to(param.device, dtype=param.dtype).contiguous() for t in splits]
                    else:
                        splits = None
                    torch.distributed.scatter(
                        param_to_load,
                        splits,
                        src=torch.distributed.get_global_rank(self.mpu.etp_group, 0),
                        group=self.mpu.etp_group,
                    )
                else:
                    if self.mpu.tp_rank == 0:
                        splits = self._weight_split_across_tp(local_name, mcore_weight, param, self.mpu.tp_size)
                        splits = [t.to(param.device, dtype=param.dtype).contiguous() for t in splits]
                    else:
                        splits = None
                    torch.distributed.scatter(
                        param_to_load,
                        splits,
                        src=torch.distributed.get_global_rank(self.mpu.tp_group, 0),
                        group=self.mpu.tp_group,
                    )
                param.copy_(param_to_load)

            # Clear fused tensor cache to free memory
            _fused_tensor_cache.clear()
            torch.cuda.empty_cache()


@register_model("qwen3_5")
@register_model("qwen3_5_text")
class Qwen3_5Bridge(Qwen3_5BridgeBase):
    """Dense Qwen3.5 bridge (e.g. Qwen3.5-9B)."""

    def _build_config(self):
        config = super()._build_config()
        # Dense model has no MoE — ensure these aren't set
        if not hasattr(self.hf_config, "num_experts") or self.hf_config.num_experts <= 1:
            config.num_moe_experts = None
        return config


@register_model("qwen3_5_moe")
@register_model("qwen3_5_moe_text")
class Qwen3_5MoEBridge(Qwen3_5BridgeBase):
    """MoE Qwen3.5 bridge (e.g. Qwen3.5-35B-A3B)."""

    pass
