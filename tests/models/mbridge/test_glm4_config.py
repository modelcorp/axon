"""Tests for GLM-4 bridge weight name mappings against the real HuggingFace model.

Validates GLM4Bridge._DIRECT_MAPPING, _ATTENTION_MAPPING, and _MLP_MAPPING
against the actual weight names in THUDM/GLM-4-9B-0414 by downloading only
config.json and model.safetensors.index.json (no model weights).

Usage:
    pytest tests/models/mbridge/test_glm4_config.py -v
"""

import json
import re
from types import SimpleNamespace

import pytest
from mbridge.core.bridge import Bridge as LLMBridge

from axon.models.mbridge.glm4 import GLM4Bridge

HF_MODEL_ID = "THUDM/GLM-4-9B-0414"


# Attention mapping keys that are only present when qk_layernorm is enabled.
# GLM-4-9B-0414 does NOT use QK layernorm, so these produce HF weight names
# (q_norm.weight, k_norm.weight) that are absent from this checkpoint.
_QK_LAYERNORM_KEYS = {
    "self_attention.q_layernorm.weight",
    "self_attention.k_layernorm.weight",
}


# ---------------------------------------------------------------------------
# Fixtures -- download HF metadata once per module
# ---------------------------------------------------------------------------


def _download(filename):
    """Download a file from HF Hub, skipping the test on network/auth failure."""
    hf_hub = pytest.importorskip("huggingface_hub")
    try:
        return hf_hub.hf_hub_download(HF_MODEL_ID, filename)
    except Exception as exc:
        pytest.skip(f"Cannot download {HF_MODEL_ID}/{filename}: {exc}")


@pytest.fixture(scope="module")
def hf_config():
    path = _download("config.json")
    with open(path) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def hf_weight_names():
    path = _download("model.safetensors.index.json")
    with open(path) as f:
        return set(json.load(f)["weight_map"].keys())


# ---------------------------------------------------------------------------
# Helpers -- mock bridge for calling _weight_name_mapping_mcore_to_hf
# ---------------------------------------------------------------------------


def _make_bridge():
    """Create a mock that uses GLM4Bridge class attributes + LLMBridge parent methods."""
    bridge = SimpleNamespace(
        _DIRECT_MAPPING=GLM4Bridge._DIRECT_MAPPING,
        _ATTENTION_MAPPING=GLM4Bridge._ATTENTION_MAPPING,
        _MLP_MAPPING=GLM4Bridge._MLP_MAPPING,
    )
    # Bind the parent class methods (they use self._ATTENTION_MAPPING etc.)
    bridge._weight_name_mapping_attention = lambda name: LLMBridge._weight_name_mapping_attention(bridge, name)
    bridge._weight_name_mapping_mlp = lambda name: LLMBridge._weight_name_mapping_mlp(bridge, name)
    return bridge


def _map_name(name):
    """Call GLM4Bridge._weight_name_mapping_mcore_to_hf on a mock instance."""
    bridge = _make_bridge()
    return GLM4Bridge._weight_name_mapping_mcore_to_hf(bridge, name)


def _layer0_hf_weights(hf_weight_names):
    """Extract the set of HF weight names belonging to layer 0."""
    return {n for n in hf_weight_names if n.startswith("model.layers.0.")}


def _non_layer_hf_weights(hf_weight_names):
    """Extract HF weight names that are NOT per-layer (embedding, norm, lm_head)."""
    return {n for n in hf_weight_names if not n.startswith("model.layers.")}


def _has_qk_layernorm(hf_config):
    """Return True if the HF config enables QK layernorm."""
    return hf_config.get("qk_layernorm", False) is True


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestConfigMatchesBridgeModelType:
    def test_config_matches_bridge_model_type(self, hf_config):
        assert hf_config["model_type"] == "glm4"


class TestDirectMappingsExistInCheckpoint:
    def test_direct_mappings_exist_in_checkpoint(self, hf_weight_names):
        for mcore_name, hf_name in GLM4Bridge._DIRECT_MAPPING.items():
            assert hf_name in hf_weight_names, (
                f"Direct mapping value '{hf_name}' (from '{mcore_name}') not found in checkpoint"
            )


class TestAllAttentionMappingsProduceValidHfNames:
    @pytest.mark.parametrize("attn_key", list(GLM4Bridge._ATTENTION_MAPPING.keys()))
    def test_attention_mapping_produces_valid_hf_names(self, attn_key, hf_config, hf_weight_names):
        # q_layernorm / k_layernorm mappings are config-conditional: they only
        # produce weights when qk_layernorm is enabled in the HF config.
        if attn_key in _QK_LAYERNORM_KEYS and not _has_qk_layernorm(hf_config):
            pytest.skip(f"'{attn_key}' requires qk_layernorm=True, which is not set for {HF_MODEL_ID}")

        mcore_name = f"decoder.layers.0.{attn_key}"
        hf_names = _map_name(mcore_name)
        assert len(hf_names) > 0, f"No HF names produced for '{mcore_name}'"
        for hf_name in hf_names:
            assert hf_name in hf_weight_names, (
                f"Attention mapping '{attn_key}' produced '{hf_name}' which is not in the checkpoint"
            )


class TestAllMlpMappingsProduceValidHfNames:
    @pytest.mark.parametrize("mlp_key", list(GLM4Bridge._MLP_MAPPING.keys()))
    def test_mlp_mapping_produces_valid_hf_names(self, mlp_key, hf_weight_names):
        mcore_name = f"decoder.layers.0.{mlp_key}"
        hf_names = _map_name(mcore_name)
        assert len(hf_names) > 0, f"No HF names produced for '{mcore_name}'"
        for hf_name in hf_names:
            assert hf_name in hf_weight_names, (
                f"MLP mapping '{mlp_key}' produced '{hf_name}' which is not in the checkpoint"
            )


class TestPostLayernormMappingsProduceValidHfNames:
    @pytest.mark.parametrize(
        "mcore_suffix, expected_hf_suffix",
        [
            ("post_self_attn_layernorm.weight", "post_self_attn_layernorm.weight"),
            ("post_mlp_layernorm.weight", "post_mlp_layernorm.weight"),
        ],
    )
    def test_post_layernorm_mapping(self, mcore_suffix, expected_hf_suffix, hf_weight_names):
        mcore_name = f"decoder.layers.0.{mcore_suffix}"
        hf_names = _map_name(mcore_name)
        assert len(hf_names) == 1
        expected = f"model.layers.0.{expected_hf_suffix}"
        assert hf_names[0] == expected
        assert expected in hf_weight_names, (
            f"Post-layernorm mapping produced '{expected}' which is not in the checkpoint"
        )


class TestAllLayer0HfWeightsAreReachable:
    """The most important test: verify there is no mapping gap for layer 0.

    Builds the COMPLETE set of HF weight names the bridge can produce for
    layer 0 (from attention, MLP, and post-layernorm mappings) plus non-layer
    weights from the direct mapping, then checks:
      1. Every layer-0 weight in the checkpoint is reachable.
      2. Every name the bridge produces actually exists in the checkpoint
         (excluding config-conditional QK layernorm names when not enabled).
    """

    def test_all_layer0_hf_weights_are_reachable(self, hf_config, hf_weight_names):
        bridge_produced = set()
        has_qk_ln = _has_qk_layernorm(hf_config)

        # Direct mappings (non-layer weights)
        for hf_name in GLM4Bridge._DIRECT_MAPPING.values():
            bridge_produced.add(hf_name)

        # Attention mappings for layer 0
        for attn_key in GLM4Bridge._ATTENTION_MAPPING:
            # Skip QK layernorm mappings when the model doesn't use them
            if attn_key in _QK_LAYERNORM_KEYS and not has_qk_ln:
                continue
            mcore_name = f"decoder.layers.0.{attn_key}"
            for hf_name in _map_name(mcore_name):
                bridge_produced.add(hf_name)

        # MLP mappings for layer 0
        for mlp_key in GLM4Bridge._MLP_MAPPING:
            mcore_name = f"decoder.layers.0.{mlp_key}"
            for hf_name in _map_name(mcore_name):
                bridge_produced.add(hf_name)

        # Post-layernorm mappings for layer 0
        for ln_suffix in ("post_self_attn_layernorm.weight", "post_mlp_layernorm.weight"):
            mcore_name = f"decoder.layers.0.{ln_suffix}"
            for hf_name in _map_name(mcore_name):
                bridge_produced.add(hf_name)

        # Split into layer-0 produced and non-layer produced
        bridge_layer0 = {n for n in bridge_produced if n.startswith("model.layers.0.")}
        bridge_non_layer = {n for n in bridge_produced if not n.startswith("model.layers.")}

        checkpoint_layer0 = _layer0_hf_weights(hf_weight_names)
        checkpoint_non_layer = _non_layer_hf_weights(hf_weight_names)

        # 1. Every layer-0 checkpoint weight must be reachable from the bridge
        missing_from_bridge = checkpoint_layer0 - bridge_layer0
        assert not missing_from_bridge, (
            f"Layer-0 checkpoint weights NOT reachable from bridge:\n  {sorted(missing_from_bridge)}"
        )

        # 2. Every name the bridge produces for layer 0 must exist in checkpoint
        extra_in_bridge = bridge_layer0 - checkpoint_layer0
        assert not extra_in_bridge, f"Bridge produces layer-0 names NOT in checkpoint:\n  {sorted(extra_in_bridge)}"

        # 3. Non-layer weights: bridge direct mappings must be subset of checkpoint
        extra_non_layer = bridge_non_layer - checkpoint_non_layer
        assert not extra_non_layer, f"Bridge produces non-layer names NOT in checkpoint:\n  {sorted(extra_non_layer)}"

        # 4. Non-layer weights: checkpoint non-layer must be subset of bridge
        missing_non_layer = checkpoint_non_layer - bridge_non_layer
        assert not missing_non_layer, (
            f"Non-layer checkpoint weights NOT reachable from bridge:\n  {sorted(missing_non_layer)}"
        )


class TestNumHiddenLayersMatches:
    def test_num_hidden_layers_matches(self, hf_config, hf_weight_names):
        num_hidden_layers = hf_config["num_hidden_layers"]
        # Extract unique layer indices from weight names like model.layers.X.foo
        layer_indices = set()
        pattern = re.compile(r"^model\.layers\.(\d+)\.")
        for name in hf_weight_names:
            m = pattern.match(name)
            if m:
                layer_indices.add(int(m.group(1)))
        assert len(layer_indices) == num_hidden_layers, (
            f"config num_hidden_layers={num_hidden_layers} but found "
            f"{len(layer_indices)} unique layer indices in checkpoint"
        )


class TestAttentionBiasMatchesQkvBiasInCheckpoint:
    def test_attention_bias_matches_qkv_bias_in_checkpoint(self, hf_config, hf_weight_names):
        attention_bias = hf_config.get("attention_bias", False)
        q_bias_name = "model.layers.0.self_attn.q_proj.bias"
        if attention_bias:
            assert q_bias_name in hf_weight_names, "attention_bias is True in config but q_proj.bias not in checkpoint"
        else:
            assert q_bias_name not in hf_weight_names, (
                "attention_bias is False in config but q_proj.bias found in checkpoint"
            )
