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
# Extends mbridge DeepseekV3Bridge, BSD-3-Clause (github.com/ISEEKYAN/mbridge).
"""
MBridge bridge for GLM-5 (model_type="glm_moe_dsa").

GLM-5 is a Mixture-of-Experts model with Dynamic Sparse Attention (DSA):
- Multi-Latent Attention with Q/KV LoRA compression (same as DeepSeek V3)
- Sparse token selection via learned indexer (wq_b, wk, k_norm, weights_proj)
- Sigmoid router with expert bias, shared experts, MTP support

Extends DeepseekV3Bridge (which handles MLA weight mapping, MLATransformerConfig,
MoE config, and MTP) with:
- DSAMLASelfAttention injection for sparse MLA with learned token indexing
- Indexer weight name mappings (wq_b, wk, k_norm, weights_proj)
- Dynamic MTP layer indexing (DeepseekV3Bridge hardcodes layer 61)
- Standard bf16 safetensor loading (DeepseekV3Bridge uses FP8 dequantization)
- Rope theta patching from rope_parameters dict
"""

import copy
from collections.abc import Callable

import torch
from mbridge.core import register_model
from mbridge.core.safetensor_io import SafeTensorIO
from mbridge.models import DeepseekV3Bridge
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_spec,
    get_gpt_mtp_block_spec,
)
from megatron.core.models.gpt.gpt_model import GPTModel


@register_model("glm_moe_dsa")
class GLM5Bridge(DeepseekV3Bridge):
    """
    Bridge for GLM-5 (GlmMoeDsaForCausalLM).

    Extends DeepseekV3Bridge with DSA (Dynamic Sparse Attention) support:
    indexer weight mappings, custom transformer layer spec, and dynamic
    MTP layer indexing.
    """

    # Inherit all DeepseekV3 MLA attention mappings, add indexer weights.
    # HF names: model.layers.{layer}.self_attn.indexer.{wq_b,wk,k_norm,weights_proj}
    _ATTENTION_MAPPING = {
        **DeepseekV3Bridge._ATTENTION_MAPPING,
        "self_attention.wq_b.weight": ["model.layers.{layer_number}.self_attn.indexer.wq_b.weight"],
        "self_attention.wk.weight": ["model.layers.{layer_number}.self_attn.indexer.wk.weight"],
        "self_attention.k_norm.weight": ["model.layers.{layer_number}.self_attn.indexer.k_norm.weight"],
        "self_attention.k_norm.bias": ["model.layers.{layer_number}.self_attn.indexer.k_norm.bias"],
        "self_attention.weights_proj.weight": ["model.layers.{layer_number}.self_attn.indexer.weights_proj.weight"],
    }

    def __init__(self, hf_config, **kwargs):
        # Patch rope_theta: GLM-5 stores it in rope_parameters dict,
        # but DeepseekV3Bridge._build_config() expects hf_config.rope_theta directly.
        if not hasattr(hf_config, "rope_theta"):
            rope_params = getattr(hf_config, "rope_parameters", None) or {}
            hf_config.rope_theta = rope_params.get("rope_theta", 1000000)

        num_mtp_layers = getattr(hf_config, "num_nextn_predict_layers", 0)

        super().__init__(hf_config, **kwargs)

        # The DeepSeek-V3 monkey patch zeros hf_config.num_nextn_predict_layers
        # before _build_config() runs, so self.config.mtp_num_layers is never set.
        # Restore it on both hf_config and the already-built Megatron config.
        if num_mtp_layers:
            hf_config.num_nextn_predict_layers = num_mtp_layers
            self.config.mtp_num_layers = num_mtp_layers
            self.config.mtp_loss_scaling_factor = 0.1

        # Override the shared state dict mapping with dynamic layer index.
        # DeepseekV3Bridge hardcodes layer 61; GLM-5 has 78 layers.
        n = hf_config.num_hidden_layers
        if num_mtp_layers and n:
            self._SHARED_STATE_DICT_MAPPING = {
                "embedding.word_embeddings.weight": [
                    "model.embed_tokens.weight",
                    f"model.layers.{n}.embed_tokens.weight",
                ],
                "output_layer.weight": [
                    "lm_head.weight",
                    f"model.layers.{n}.shared_head.head.weight",
                ],
            }

    def _get_safetensor_io(self, weights_path: str):
        """Use standard SafeTensorIO — GLM-5 ships bf16 safetensors, not FP8."""
        return SafeTensorIO(self._get_actual_hf_path(weights_path))

    def _get_transformer_layer_spec(self, vp_stage=None):
        """Override to inject DSAMLASelfAttention for sparse MLA."""
        from axon.models.mcore.models.glm5 import get_glm5_spec

        return get_glm5_spec(self.config, self.hf_config, vp_stage=vp_stage)

    def _model_provider(self, post_model_creation_callbacks: list[Callable[[torch.nn.Module], None]]):
        """Override to ensure MTP block spec uses DSA attention, not standard MLA.

        DeepseekV3Bridge._model_provider falls back to standard MLA for the MTP
        layer spec when the last PP stage has no decoder layers. GLM-5 needs DSA
        (with indexer) for MTP too.
        """
        if not (self.config.mtp_num_layers and self.config.mtp_num_layers > 0):
            return super()._model_provider(post_model_creation_callbacks)

        from axon.models.mcore.models.glm5 import build_dsa_self_attn_spec

        share_embeddings_and_output_weights = getattr(self.hf_config, "tie_word_embeddings", False)

        def provider(pre_process, post_process, vp_stage: int | None = None):
            transformer_layer_spec = self._get_transformer_layer_spec(vp_stage)
            gptmodel_args = self._get_gptmodel_args()
            if vp_stage is not None and self.has_vp_stage:
                gptmodel_args["vp_stage"] = vp_stage

            if hasattr(transformer_layer_spec, "layer_specs") and len(transformer_layer_spec.layer_specs) == 0:
                # Last PP stage has no decoder layers — build a single-layer
                # spec and inject DSA attention for the MTP layer.
                transformer_layer_spec_for_mtp = get_gpt_layer_with_transformer_engine_spec(
                    num_experts=self.config.num_moe_experts,
                    moe_grouped_gemm=self.config.moe_grouped_gemm,
                    qk_layernorm=self.config.qk_layernorm,
                    multi_latent_attention=True,
                )
                transformer_layer_spec_for_mtp = copy.deepcopy(transformer_layer_spec_for_mtp)
                transformer_layer_spec_for_mtp.submodules.self_attention = build_dsa_self_attn_spec()
            else:
                transformer_layer_spec_for_mtp = transformer_layer_spec

            mtp_block_spec = get_gpt_mtp_block_spec(
                self.config,
                transformer_layer_spec_for_mtp,
                use_transformer_engine=True,
                vp_stage=vp_stage,
            )
            gptmodel_args["mtp_block_spec"] = mtp_block_spec

            model = GPTModel(
                config=self.config,
                transformer_layer_spec=transformer_layer_spec,
                pre_process=pre_process,
                post_process=post_process,
                share_embeddings_and_output_weights=share_embeddings_and_output_weights,
                **gptmodel_args,
            )
            for callback in post_model_creation_callbacks:
                callback(
                    model,
                    pre_process=pre_process,
                    post_process=post_process,
                    config=self.config,
                    hf_config=self.hf_config,
                )
            return model

        return provider

    def _weight_to_hf_format(
        self, mcore_weights_name: str, mcore_weights: torch.Tensor
    ) -> tuple[list[str], list[torch.Tensor]]:
        """Apply rope reordering for DSA indexer weights and handle MTP shared weights.

        Training uses last half for rope while HF uses first half,
        so we swap the two halves when exporting.
        """
        index_head_dim = self.hf_config.index_head_dim
        rope_split = index_head_dim - self.config.qk_pos_emb_head_dim

        if "self_attention.wq_b.weight" in mcore_weights_name:
            hf_names = self._weight_name_mapping_mcore_to_hf(mcore_weights_name)
            wq_b = mcore_weights
            wq_b = wq_b.view(-1, index_head_dim, wq_b.shape[-1])
            wq_b = torch.cat([wq_b[:, rope_split:], wq_b[:, :rope_split]], dim=1).view(-1, wq_b.shape[-1])
            return hf_names, [wq_b]
        elif "self_attention.wk.weight" in mcore_weights_name:
            hf_names = self._weight_name_mapping_mcore_to_hf(mcore_weights_name)
            wk = mcore_weights
            wk = torch.cat([wk[rope_split:], wk[:rope_split]], dim=0)
            return hf_names, [wk]
        elif "self_attention.k_norm.weight" in mcore_weights_name:
            hf_names = self._weight_name_mapping_mcore_to_hf(mcore_weights_name)
            knorm_weight = mcore_weights
            knorm_weight = torch.cat([knorm_weight[rope_split:], knorm_weight[:rope_split]], dim=0)
            return hf_names, [knorm_weight]
        elif "self_attention.k_norm.bias" in mcore_weights_name:
            hf_names = self._weight_name_mapping_mcore_to_hf(mcore_weights_name)
            knorm_bias = mcore_weights
            knorm_bias = torch.cat([knorm_bias[rope_split:], knorm_bias[:rope_split]], dim=0)
            return hf_names, [knorm_bias]

        if (
            self.config.mtp_num_layers is not None
            and self.config.mtp_num_layers >= 1
            and mcore_weights_name in self._SHARED_STATE_DICT_MAPPING
        ):
            hf_names = self._SHARED_STATE_DICT_MAPPING[mcore_weights_name]
            return hf_names, [mcore_weights] * len(hf_names)
        # Skip DeepseekV3Bridge's _weight_to_hf_format (hardcoded 61) and go to Bridge base
        return super(DeepseekV3Bridge, self)._weight_to_hf_format(mcore_weights_name, mcore_weights)

    def _weight_to_mcore_format(self, mcore_weights_name: str, hf_weights: list[torch.Tensor]) -> torch.Tensor:
        """Apply inverse rope reordering when importing DSA indexer weights from HF format.

        The swap operation is its own inverse: swap the two halves back.
        """
        index_head_dim = self.hf_config.index_head_dim
        rope_split = index_head_dim - self.config.qk_pos_emb_head_dim

        if "self_attention.wq_b.weight" in mcore_weights_name:
            wq_b = hf_weights[0]
            wq_b = wq_b.view(-1, index_head_dim, wq_b.shape[-1])
            wq_b = torch.cat([wq_b[:, rope_split:], wq_b[:, :rope_split]], dim=1).view(-1, wq_b.shape[-1])
            return wq_b
        elif "self_attention.wk.weight" in mcore_weights_name:
            wk = hf_weights[0]
            wk = torch.cat([wk[rope_split:], wk[:rope_split]], dim=0)
            return wk
        elif "self_attention.k_norm.weight" in mcore_weights_name:
            knorm_weight = hf_weights[0]
            knorm_weight = torch.cat([knorm_weight[rope_split:], knorm_weight[:rope_split]], dim=0)
            return knorm_weight
        elif "self_attention.k_norm.bias" in mcore_weights_name:
            knorm_bias = hf_weights[0]
            knorm_bias = torch.cat([knorm_bias[rope_split:], knorm_bias[:rope_split]], dim=0)
            return knorm_bias
        return super()._weight_to_mcore_format(mcore_weights_name, hf_weights)

    def _convert_mtp_param(self, name: str) -> list[str]:
        """Convert MTP parameter names with dynamic layer count (not hardcoded 61)."""
        assert self.config.mtp_num_layers == 1, "only support one mtp layer for now"
        n = self.config.num_layers

        direct_name_mapping = {
            "mtp.layers.0.enorm.weight": f"model.layers.{n}.enorm.weight",
            "mtp.layers.0.hnorm.weight": f"model.layers.{n}.hnorm.weight",
            "mtp.layers.0.eh_proj.weight": f"model.layers.{n}.eh_proj.weight",
            "mtp.layers.0.final_layernorm.weight": f"model.layers.{n}.shared_head.norm.weight",
        }
        if name in direct_name_mapping:
            return [direct_name_mapping[name]]

        assert "mtp.layers.0.transformer_layer" in name, f"mtp not found in {name}"
        proxy_name = name.replace("mtp.layers.0.transformer_layer", f"decoder.layers.{n}")

        if "self_attention" in proxy_name or "input_layernorm.weight" in proxy_name:
            return self._weight_name_mapping_attention(proxy_name)
        elif "mlp" in proxy_name:
            return self._weight_name_mapping_mlp(proxy_name)
        else:
            raise NotImplementedError(f"Unsupported MTP parameter name: {name}")
