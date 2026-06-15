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
"""Compatibility patches for the mbridge library with transformers 5.x.

In transformers 5.x, ``rope_theta`` moved from a top-level config attribute
to a nested ``rope_parameters`` dict for several model families.  The mbridge
library's ``LLMBridge._get_gptmodel_args()`` reads ``hf_config.rope_theta``
directly, which fails for these models.

This patch is applied once on the base ``LLMBridge`` class so ALL bridge
subclasses benefit automatically.
"""

import logging

logger = logging.getLogger(__name__)


def apply_mbridge_rope_theta_patch():
    """Patch LLMBridge._get_gptmodel_args to handle nested rope_theta."""
    from mbridge.core.llm_bridge import LLMBridge

    _orig_get_gptmodel_args = LLMBridge._get_gptmodel_args

    def _patched_get_gptmodel_args(self):
        if not hasattr(self.hf_config, "rope_theta") or self.hf_config.rope_theta is None:
            rope_params = getattr(self.hf_config, "rope_parameters", {}) or {}
            if isinstance(rope_params, dict) and "rope_theta" in rope_params:
                self.hf_config.rope_theta = rope_params["rope_theta"]
        return _orig_get_gptmodel_args(self)

    LLMBridge._get_gptmodel_args = _patched_get_gptmodel_args
    logger.info("Patched LLMBridge._get_gptmodel_args for transformers 5.x rope_theta compatibility")


def apply_mbridge_qwen25vl_mrope_patch():
    """Patch Qwen2_5VLBridge._build_config to handle transformers 5.x rope_scaling layout
    and missing top-level config attributes.

    In transformers 5.x, ``Qwen2_5_VLConfig.rope_scaling`` is a property that
    returns ``self.rope_parameters``, but ``rope_parameters`` only exists on
    ``text_config`` (Qwen2_5_VLTextConfig), not on the top-level config.
    Additionally, attributes such as ``num_hidden_layers``, ``hidden_size``, etc.
    moved to ``text_config`` and are no longer accessible directly on the top-level
    VL config.  mbridge reads all of these from ``self.hf_config`` directly, which
    raises ``AttributeError``.

    Fix: before delegating to the original ``_build_config``, inject the missing
    attributes from ``text_config`` onto the top-level config so mbridge can read
    them without modification.
    """
    try:
        from mbridge.models.qwen2_5_vl import Qwen2_5VLBridge
    except ImportError:
        logger.debug("mbridge.models.qwen2_5_vl not available, skipping mrope patch")
        return

    _orig_build_config = Qwen2_5VLBridge._build_config

    # Attributes that live on text_config in transformers 5.x but that mbridge's
    # _build_base_config / _get_gptmodel_args read from the top-level VL config.
    _TEXT_CONFIG_ATTRS = (
        "num_hidden_layers",
        "hidden_size",
        "num_attention_heads",
        "num_key_value_heads",
        "intermediate_size",
        "attention_dropout",
        "rms_norm_eps",
        "head_dim",
        "vocab_size",
        "max_position_embeddings",
    )

    def _patched_build_config(self):
        hf_cfg = self.hf_config
        # ---- rope_scaling / mrope_section fix ----
        try:
            _ = hf_cfg.rope_scaling
        except AttributeError:
            text_cfg = getattr(hf_cfg, "text_config", hf_cfg)
            rope_params = getattr(text_cfg, "rope_parameters", None) or getattr(text_cfg, "rope_scaling", None) or {}
            if isinstance(rope_params, dict) and rope_params:
                # Inject rope_scaling as an instance attribute so mbridge can read it
                object.__setattr__(hf_cfg, "rope_scaling", rope_params)
                logger.debug("Injected rope_scaling=%r onto hf_config from text_config.rope_parameters", rope_params)
        # ---- promote missing text_config attrs to top-level config ----
        text_cfg = getattr(hf_cfg, "text_config", None)
        if text_cfg is not None:
            for attr in _TEXT_CONFIG_ATTRS:
                if not hasattr(hf_cfg, attr) and hasattr(text_cfg, attr):
                    object.__setattr__(hf_cfg, attr, getattr(text_cfg, attr))
                    logger.debug("Promoted text_config.%s onto hf_config", attr)
            # rope_theta lives in text_config.rope_parameters in transformers 5.x,
            # not directly on text_config or the top-level VL config.
            if not hasattr(hf_cfg, "rope_theta"):
                rope_params = getattr(text_cfg, "rope_parameters", None) or {}
                if isinstance(rope_params, dict) and "rope_theta" in rope_params:
                    object.__setattr__(hf_cfg, "rope_theta", rope_params["rope_theta"])
                    logger.debug(
                        "Promoted text_config.rope_parameters['rope_theta']=%r onto hf_config",
                        rope_params["rope_theta"],
                    )
        return _orig_build_config(self)

    Qwen2_5VLBridge._build_config = _patched_build_config
    logger.info(
        "Patched Qwen2_5VLBridge._build_config for transformers 5.x mrope_section and text_config compatibility"
    )
