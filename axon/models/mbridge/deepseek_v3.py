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
"""
Monkey patches for mbridge DeepseekV3Bridge:

1. __init__: sanitize hf_config before bridge construction — remove
   quantization_config and disable MTP (num_nextn_predict_layers=0).

2. _build_config: rename 'max_position_embeddings' -> 'original_max_position_embeddings'
   for MLATransformerConfig compatibility.

3. _get_gptmodel_args: handle missing 'max_position_embeddings' on DeepseekV3Config
   by falling back to rope_scaling.original_max_position_embeddings.
"""

from mbridge.models.deepseek_v3 import DeepseekV3Bridge

_original_init = DeepseekV3Bridge.__init__
_original_build_config = DeepseekV3Bridge._build_config
_original_get_gptmodel_args = DeepseekV3Bridge._get_gptmodel_args


def _patched_init(self, hf_config, *args, **kwargs):
    # Remove quantization_config (FP8 quant not supported in training)
    if hasattr(hf_config, "quantization_config"):
        delattr(hf_config, "quantization_config")

    # Disable MTP (not currently supported for DeepSeek-V3)
    if hasattr(hf_config, "num_nextn_predict_layers"):
        hf_config.num_nextn_predict_layers = 0

    _original_init(self, hf_config, *args, **kwargs)


def _patched_build_config(self):
    # Patch _build_base_config temporarily to rename the kwarg
    original_build_base_config = self._build_base_config

    def patched_build_base_config(**kwargs):
        if "max_position_embeddings" in kwargs:
            kwargs["original_max_position_embeddings"] = kwargs.pop("max_position_embeddings")
        return original_build_base_config(**kwargs)

    self._build_base_config = patched_build_base_config
    try:
        return _original_build_config(self)
    finally:
        self._build_base_config = original_build_base_config


def _patched_get_gptmodel_args(self):
    hf_config = self.hf_config
    max_pos = getattr(hf_config, "max_position_embeddings", None)
    if max_pos is None:
        # Fall back to rope_scaling.original_max_position_embeddings
        rope_scaling = getattr(hf_config, "rope_scaling", None) or {}
        max_pos = rope_scaling.get("original_max_position_embeddings", 163840)
    return dict(
        vocab_size=hf_config.vocab_size,
        max_sequence_length=max_pos,
        position_embedding_type="rope",
        rotary_base=hf_config.rope_theta,
    )


def apply_deepseek_v3_patch():
    DeepseekV3Bridge.__init__ = _patched_init
    DeepseekV3Bridge._build_config = _patched_build_config
    DeepseekV3Bridge._get_gptmodel_args = _patched_get_gptmodel_args
