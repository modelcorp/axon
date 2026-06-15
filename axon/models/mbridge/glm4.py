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
# Uses / extends mbridge LLMBridge and Glm4TransformerLayer, BSD-3-Clause (github.com/ISEEKYAN/mbridge).
"""
MBridge bridge for GLM-4 dense models (model_type="glm4").

GLM-4 is a dense transformer (e.g., GLM-4-9B-0414, GLM-Z1-9B/32B) with:
- QKV bias (attention_bias=True in HF config)
- Post-attention and post-MLP layer norms (extra layernorms after attn/MLP outputs)
- Partial rotary position embeddings (partial_rotary_factor=0.5)
- RMSNorm normalization
- Fused gate+up MLP projections (gate_up_proj)
- GQA (e.g., 32 heads / 2 KV heads)

The post-layernorm architecture requires a custom TransformerLayer (Glm4TransformerLayer
from mbridge) that applies layernorm to the attention and MLP outputs before the
residual addition. Standard Megatron-Core TransformerLayer does not support this.
"""

from mbridge.core import LLMBridge, register_model
from mbridge.models.glm4_vl.transformer_layer import Glm4TransformerLayer

# ---------------------------------------------------------------------------
# Patch Glm4TransformerLayer for compatibility with newer megatron-core.
#
# megatron-core added pg_collection/vp_stage to __init__ and changed the
# forward() contract: _forward_attention must return (hidden_states, context)
# and _forward_mlp takes (hidden_states, inference_context).  The mbridge
# Glm4TransformerLayer was written for an older API where _forward_attention
# returned (pre_mlp_output, residual, context) and _forward_mlp took both.
# We override forward() to bridge the gap.
# ---------------------------------------------------------------------------
_original_glm4_layer_init = Glm4TransformerLayer.__init__
_original_glm4_forward_attention = Glm4TransformerLayer._forward_attention
_original_glm4_forward_mlp = Glm4TransformerLayer._forward_mlp


def _patched_glm4_layer_init(self, config, submodules, layer_number=1, hidden_dropout=None, **kwargs):
    # Forward `vp_stage` (and any other newer kwargs Megatron added) through to
    # the underlying TransformerLayer.__init__. Required for VPP — otherwise
    # `get_transformer_layer_offset` asserts inside Megatron when
    # `virtual_pipeline_model_parallel_size` is set.
    import inspect as _inspect

    base_init = Glm4TransformerLayer.__mro__[1].__init__  # Megatron TransformerLayer.__init__
    base_params = _inspect.signature(base_init).parameters
    fwd_kwargs = {k: v for k, v in kwargs.items() if k in base_params}

    if "vp_stage" in base_params and "vp_stage" in fwd_kwargs:
        # Re-implement Glm4TransformerLayer.__init__ inline so we can pass vp_stage.
        # (The mbridge layer's __init__ is just `super().__init__(...) + create
        # post-norm/post-self-attn layernorms`. We reproduce that.)
        from megatron.core.transformer.spec_utils import build_module

        base_init(
            self,
            config,
            submodules,
            layer_number=layer_number,
            hidden_dropout=hidden_dropout,
            **fwd_kwargs,
        )
        self.post_self_attn_layernorm = build_module(
            submodules.post_mlp_layernorm,
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )
        self.post_mlp_layernorm = build_module(
            submodules.post_mlp_layernorm,
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )
    else:
        _original_glm4_layer_init(self, config, submodules, layer_number=layer_number, hidden_dropout=hidden_dropout)


def _patched_glm4_forward(self, *args, **kwargs):
    """Override forward to use the original 3-return _forward_attention + 2-arg _forward_mlp."""
    import inspect

    # Filter kwargs for _forward_attention (strip rotary_pos_cos_sin etc.)
    attn_params = inspect.signature(_original_glm4_forward_attention).parameters
    attn_kwargs = {k: v for k, v in kwargs.items() if k in attn_params}

    pre_mlp_output, residual, context = _original_glm4_forward_attention(self, *args, **attn_kwargs)
    output = _original_glm4_forward_mlp(self, pre_mlp_output, residual)
    return output, context


Glm4TransformerLayer.__init__ = _patched_glm4_layer_init
Glm4TransformerLayer.forward = _patched_glm4_forward


@register_model("glm4")
class GLM4Bridge(LLMBridge):
    """
    Bridge implementation for GLM-4 dense models.

    Handles conversion between HuggingFace GLM-4 format and Megatron-Core,
    with support for post-attention/post-MLP layer norms, QKV bias, and
    partial rotary embeddings.
    """

    _DIRECT_MAPPING = {
        "embedding.word_embeddings.weight": "model.embed_tokens.weight",
        "decoder.final_layernorm.weight": "model.norm.weight",
        "output_layer.weight": "lm_head.weight",
    }

    _ATTENTION_MAPPING = {
        "self_attention.linear_proj.weight": ["model.layers.{layer_number}.self_attn.o_proj.weight"],
        "self_attention.linear_qkv.layer_norm_weight": ["model.layers.{layer_number}.input_layernorm.weight"],
        "self_attention.q_layernorm.weight": ["model.layers.{layer_number}.self_attn.q_norm.weight"],
        "self_attention.k_layernorm.weight": ["model.layers.{layer_number}.self_attn.k_norm.weight"],
        "self_attention.linear_qkv.weight": [
            "model.layers.{layer_number}.self_attn.q_proj.weight",
            "model.layers.{layer_number}.self_attn.k_proj.weight",
            "model.layers.{layer_number}.self_attn.v_proj.weight",
        ],
        "self_attention.linear_qkv.bias": [
            "model.layers.{layer_number}.self_attn.q_proj.bias",
            "model.layers.{layer_number}.self_attn.k_proj.bias",
            "model.layers.{layer_number}.self_attn.v_proj.bias",
        ],
    }

    _MLP_MAPPING = {
        "mlp.linear_fc1.weight": [
            "model.layers.{layer_number}.mlp.gate_up_proj.weight",
        ],
        "mlp.linear_fc1.layer_norm_weight": ["model.layers.{layer_number}.post_attention_layernorm.weight"],
        "mlp.linear_fc2.weight": ["model.layers.{layer_number}.mlp.down_proj.weight"],
    }

    def _build_config(self):
        # attention_bias varies by model size: True for 9B, False for 32B
        add_qkv_bias = getattr(self.hf_config, "attention_bias", True)
        return self._build_base_config(
            add_qkv_bias=add_qkv_bias,
            add_bias_linear=False,
            qk_layernorm=False,
            rotary_interleaved=True,
            persist_layer_norm=True,
            bias_activation_fusion=True,
            bias_dropout_fusion=True,
        )

    def _get_gptmodel_args(self) -> dict:
        return dict(
            vocab_size=self.hf_config.vocab_size,
            max_sequence_length=self.hf_config.max_position_embeddings,
            position_embedding_type="rope",
            rotary_base=self.hf_config.rope_theta,
            rotary_percent=self.hf_config.partial_rotary_factor,
        )

    def _get_transformer_layer_spec(self, vp_stage: int | None = None):
        """
        Override to inject Glm4TransformerLayer for post-attention/post-MLP layernorms.

        GLM-4 applies extra layernorms after attention and MLP outputs (before residual
        addition). Standard Megatron TransformerLayer doesn't support this, so we swap
        in Glm4TransformerLayer and wire up the post-layernorm submodules.
        """
        from megatron.core.transformer.transformer_block import TransformerBlockSubmodules

        block_spec = super()._get_transformer_layer_spec(vp_stage=vp_stage)
        assert isinstance(block_spec, TransformerBlockSubmodules)
        for layer_spec in block_spec.layer_specs:
            layer_spec.module = Glm4TransformerLayer
            layer_spec.submodules.post_self_attn_layernorm = block_spec.layer_norm
            layer_spec.submodules.post_mlp_layernorm = block_spec.layer_norm
        return block_spec

    def _weight_name_mapping_mcore_to_hf(self, mcore_weights_name: str) -> list[str]:
        assert "_extra_state" not in mcore_weights_name, "extra_state should not be loaded"

        if mcore_weights_name in self._DIRECT_MAPPING:
            return [self._DIRECT_MAPPING[mcore_weights_name]]

        if "post_self_attn_layernorm" in mcore_weights_name:
            layer_number = mcore_weights_name.split(".")[2]
            return [f"model.layers.{layer_number}.post_self_attn_layernorm.weight"]
        elif "post_mlp_layernorm" in mcore_weights_name:
            layer_number = mcore_weights_name.split(".")[2]
            return [f"model.layers.{layer_number}.post_mlp_layernorm.weight"]
        elif "self_attention" in mcore_weights_name:
            return self._weight_name_mapping_attention(mcore_weights_name)
        elif "mlp" in mcore_weights_name:
            return self._weight_name_mapping_mlp(mcore_weights_name)
        else:
            raise NotImplementedError(f"Unsupported parameter name: {mcore_weights_name}")
