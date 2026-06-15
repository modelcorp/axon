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
MBridge bridge for GLM-4.7 Flash (glm4_moe_lite) models.

GLM-4.7 Flash extends DeepseekV3Bridge with:
- Dynamic MTP layer indexing (DeepseekV3 hardcodes layer 61)
- Standard bf16 safetensor loading (DeepseekV3 uses FP8 dequantization)
- Rope theta patching from rope_parameters dict
"""

import torch
from mbridge.core import register_model
from mbridge.core.safetensor_io import SafeTensorIO
from mbridge.models import DeepseekV3Bridge


@register_model("glm4_moe_lite")
class GLM4MoELiteBridge(DeepseekV3Bridge):
    """
    Bridge for GLM-4.7-Flash (glm4_moe_lite) models.

    Extends DeepseekV3Bridge with dynamic MTP layer indexing and standard
    bf16 safetensor loading (no FP8 dequantization).
    """

    def __init__(self, hf_config, **kwargs):
        # Patch rope_theta: GLM-4.7-Flash stores it in rope_parameters dict,
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
        # DeepseekV3Bridge hardcodes layer 61; GLM-4.7-Flash uses num_hidden_layers.
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
        """Use standard SafeTensorIO — GLM-4.7-Flash ships bf16 safetensors, not FP8."""
        return SafeTensorIO(self._get_actual_hf_path(weights_path))

    def _weight_to_hf_format(
        self, mcore_weights_name: str, mcore_weights: torch.Tensor
    ) -> tuple[list[str], list[torch.Tensor]]:
        """Handle shared embedding/output weights for MTP with dynamic layer count."""
        if (
            self.config.mtp_num_layers is not None
            and self.config.mtp_num_layers >= 1
            and mcore_weights_name in self._SHARED_STATE_DICT_MAPPING
        ):
            hf_names = self._SHARED_STATE_DICT_MAPPING[mcore_weights_name]
            return hf_names, [mcore_weights] * len(hf_names)
        # Skip DeepseekV3Bridge's _weight_to_hf_format (hardcoded 61) and go to Bridge base
        return super(DeepseekV3Bridge, self)._weight_to_hf_format(mcore_weights_name, mcore_weights)

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
