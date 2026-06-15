"""Tests for GPTOSSBridge weight name mappings against the real HuggingFace model.

Validates that GPTOSSBridge._ATTENTION_MAPPING, _MLP_MAPPING, _DIRECT_MAPPING,
and _SKIP_LOADING_WEIGHTS are consistent with the actual weight names in the
unsloth/gpt-oss-20b-BF16 checkpoint.

Tests:
- Config model_type is "gpt_oss"
- GPT-OSS-specific HF attention keys exist in the checkpoint
- MLP router/layernorm weights exist in the checkpoint
- MLP expert weights (gate_up_proj, down_proj, biases) exist in the checkpoint
- Weights listed in _SKIP_LOADING_WEIGHTS do NOT exist in the checkpoint
- All layer-0 HF weights are reachable via bridge mappings
- layer_types length matches num_hidden_layers
- All layers have the same weight suffix structure

Usage:
    pytest tests/models/mbridge/test_gpt_oss_config.py -v
"""

import json
import re
from types import SimpleNamespace

import pytest
from mbridge.core.bridge import Bridge
from mbridge.models.qwen2moe import Qwen2MoEBridge

from axon.models.mbridge.gpt_oss import GPTOSSBridge

HF_MODEL_ID = "unsloth/gpt-oss-20b-BF16"


# ---------------------------------------------------------------------------
# Fixtures
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
# Helpers -- bridge mock
# ---------------------------------------------------------------------------


def _make_bridge(num_moe_experts=32):
    """Build a minimal mock that can call the real bridge mapping methods.

    GPTOSSBridge extends Qwen2MoEBridge extends LLMBridge extends Bridge.
    We need:
      - _ATTENTION_MAPPING (from GPTOSSBridge)
      - _MLP_MAPPING (from GPTOSSBridge)
      - _DIRECT_MAPPING (inherited from Qwen2MoEBridge)
      - _OTHER_MAPPING (inherited from Bridge)
      - config.num_moe_experts (used by _weight_name_mapping_mlp)
      - _weight_name_mapping_attention (from Bridge base)
      - _weight_name_mapping_mlp (overridden in GPTOSSBridge)
      - _weight_name_mapping_mcore_to_hf (from Bridge base)
      - _weight_name_mapping_other (from Bridge base)
    """
    bridge = SimpleNamespace(
        _ATTENTION_MAPPING=GPTOSSBridge._ATTENTION_MAPPING,
        _MLP_MAPPING=GPTOSSBridge._MLP_MAPPING,
        _DIRECT_MAPPING=Qwen2MoEBridge._DIRECT_MAPPING,
        _OTHER_MAPPING=getattr(GPTOSSBridge, "_OTHER_MAPPING", Bridge._OTHER_MAPPING),
        config=SimpleNamespace(num_moe_experts=num_moe_experts),
    )
    bridge._weight_name_mapping_attention = lambda name: Bridge._weight_name_mapping_attention(bridge, name)
    bridge._weight_name_mapping_mlp = lambda name: GPTOSSBridge._weight_name_mapping_mlp(bridge, name)
    bridge._weight_name_mapping_other = lambda name: Bridge._weight_name_mapping_other(bridge, name)
    bridge._weight_name_mapping_mcore_to_hf = lambda name: Bridge._weight_name_mapping_mcore_to_hf(bridge, name)
    return bridge


def _collect_layer0_hf_names_from_attention(bridge):
    """Produce all HF weight names for layer 0 from _ATTENTION_MAPPING."""
    names = set()
    for _mcore_key, hf_templates in bridge._ATTENTION_MAPPING.items():
        for tmpl in hf_templates:
            names.add(tmpl.format(layer_number=0))
    return names


def _collect_layer0_hf_names_from_mlp(bridge):
    """Produce all HF weight names for layer 0 from _weight_name_mapping_mlp.

    We call the mapping method with representative MCore names to extract
    all possible HF names for layer 0.
    """
    names = set()

    # Static MLP mapping keys (router, layernorm, grouped experts)
    static_mcore_names = [
        "decoder.layers.0.mlp.router.weight",
        "decoder.layers.0.mlp.router.bias",
        "decoder.layers.0.pre_mlp_layernorm",
        "decoder.layers.0.mlp.experts.linear_fc1.weight",
        "decoder.layers.0.mlp.experts.linear_fc2.weight",
        "decoder.layers.0.mlp.experts.linear_fc1.bias",
        "decoder.layers.0.mlp.experts.linear_fc2.bias",
    ]
    for mcore_name in static_mcore_names:
        try:
            hf_names = bridge._weight_name_mapping_mlp(mcore_name)
            names.update(hf_names)
        except NotImplementedError:
            pass

    # Per-expert MLP mapping (local_experts.{i}.*)
    for i in range(bridge.config.num_moe_experts):
        per_expert_names = [
            f"decoder.layers.0.mlp.experts.local_experts.{i}.linear_fc1.weight",
            f"decoder.layers.0.mlp.experts.local_experts.{i}.linear_fc2.weight",
            f"decoder.layers.0.mlp.experts.local_experts.{i}.linear_fc1.bias",
            f"decoder.layers.0.mlp.experts.local_experts.{i}.linear_fc2.bias",
        ]
        for mcore_name in per_expert_names:
            try:
                hf_names = bridge._weight_name_mapping_mlp(mcore_name)
                names.update(hf_names)
            except NotImplementedError:
                pass

    return names


def _collect_direct_hf_names(bridge):
    """Produce all HF weight names from _DIRECT_MAPPING."""
    return set(bridge._DIRECT_MAPPING.values())


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestConfigModelType:
    def test_config_model_type(self, hf_config):
        assert hf_config["model_type"] == "gpt_oss"


class TestGptOssSpecificAttentionKeysInCheckpoint:
    """Verify that GPT-OSS-specific HF attention keys exist in the checkpoint."""

    # These are the GPT-OSS-specific entries added on top of Qwen2MoE
    GPT_OSS_HF_ATTN_KEYS = [
        "model.layers.{layer_number}.input_layernorm.weight",
        "model.layers.{layer_number}.self_attn.k_proj.weight",
        "model.layers.{layer_number}.self_attn.k_proj.bias",
        "model.layers.{layer_number}.self_attn.o_proj.weight",
        "model.layers.{layer_number}.self_attn.o_proj.bias",
        "model.layers.{layer_number}.self_attn.q_proj.weight",
        "model.layers.{layer_number}.self_attn.q_proj.bias",
        "model.layers.{layer_number}.self_attn.v_proj.weight",
        "model.layers.{layer_number}.self_attn.v_proj.bias",
        "model.layers.{layer_number}.self_attn.sinks",
    ]

    def test_gpt_oss_specific_attention_keys_in_checkpoint(self, hf_weight_names):
        missing = []
        for tmpl in self.GPT_OSS_HF_ATTN_KEYS:
            hf_name = tmpl.format(layer_number=0)
            if hf_name not in hf_weight_names:
                missing.append(hf_name)
        assert not missing, f"GPT-OSS attention keys missing from checkpoint: {missing}"


class TestMlpRouterAndLayernormInCheckpoint:
    """Verify router and post_attention_layernorm weights exist for layer 0."""

    EXPECTED = [
        "model.layers.0.mlp.router.weight",
        "model.layers.0.mlp.router.bias",
        "model.layers.0.post_attention_layernorm.weight",
    ]

    def test_mlp_router_and_layernorm_in_checkpoint(self, hf_weight_names):
        missing = [n for n in self.EXPECTED if n not in hf_weight_names]
        assert not missing, f"MLP router/layernorm weights missing: {missing}"


class TestMlpExpertWeightsInCheckpoint:
    """Verify expert weight tensors exist for layer 0."""

    EXPECTED = [
        "model.layers.0.mlp.experts.gate_up_proj",
        "model.layers.0.mlp.experts.down_proj",
        "model.layers.0.mlp.experts.gate_up_proj_bias",
        "model.layers.0.mlp.experts.down_proj_bias",
    ]

    def test_mlp_expert_weights_in_checkpoint(self, hf_weight_names):
        missing = [n for n in self.EXPECTED if n not in hf_weight_names]
        assert not missing, f"MLP expert weights missing: {missing}"


class TestSkipLoadingWeightsNotInCheckpoint:
    """Verify weights in _SKIP_LOADING_WEIGHTS do NOT exist in the checkpoint."""

    def test_skip_loading_weights_not_in_checkpoint(self, hf_weight_names):
        for skip_suffix in GPTOSSBridge._SKIP_LOADING_WEIGHTS:
            # Check layer 0 specifically
            # The skip patterns are suffixes like "self_attn.q_norm.weight"
            hf_name = f"model.layers.0.{skip_suffix}"
            assert hf_name not in hf_weight_names, (
                f"Weight '{hf_name}' is in _SKIP_LOADING_WEIGHTS but exists in "
                f"the checkpoint -- it should not be skipped"
            )


class TestAllLayer0HfWeightsAreReachable:
    """Verify that every layer-0 weight in the checkpoint is producible by the bridge."""

    def test_all_layer0_hf_weights_are_reachable(self, hf_weight_names):
        bridge = _make_bridge(num_moe_experts=32)

        # Collect all HF names producible from the bridge mappings
        reachable = set()
        reachable.update(_collect_layer0_hf_names_from_attention(bridge))
        reachable.update(_collect_layer0_hf_names_from_mlp(bridge))
        reachable.update(_collect_direct_hf_names(bridge))

        # Filter checkpoint weights to only layer-0 weights
        layer0_checkpoint_weights = {name for name in hf_weight_names if name.startswith("model.layers.0.")}

        # Also include non-layer weights (embed_tokens, norm, lm_head)
        non_layer_weights = {name for name in hf_weight_names if not name.startswith("model.layers.")}

        all_checkpoint_weights = layer0_checkpoint_weights | non_layer_weights

        unreachable = all_checkpoint_weights - reachable
        assert not unreachable, "Checkpoint weights not reachable from bridge mappings:\n" + "\n".join(
            sorted(unreachable)
        )


class TestLayerTypesLengthMatchesNumHiddenLayers:
    def test_layer_types_length_matches_num_hidden_layers(self, hf_config):
        layer_types = hf_config.get("layer_types")
        num_hidden_layers = hf_config["num_hidden_layers"]
        assert layer_types is not None, "layer_types not found in config"
        assert len(layer_types) == num_hidden_layers, (
            f"len(layer_types)={len(layer_types)} != num_hidden_layers={num_hidden_layers}"
        )


class TestAllLayersHaveSameStructure:
    """Verify every layer has the same set of weight suffixes.

    GPT-OSS uses alternating sliding/full attention but the HF checkpoint
    has the same weight structure for all layers.
    """

    _LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.(.+)$")

    def test_all_layers_have_same_structure(self, hf_weight_names, hf_config):
        # Group weight suffixes by layer index
        layer_suffixes: dict[int, set[str]] = {}
        for name in hf_weight_names:
            m = self._LAYER_RE.match(name)
            if m:
                layer_idx = int(m.group(1))
                suffix = m.group(2)
                layer_suffixes.setdefault(layer_idx, set()).add(suffix)

        num_layers = hf_config["num_hidden_layers"]
        assert len(layer_suffixes) == num_layers, (
            f"Expected {num_layers} layers in checkpoint, found {len(layer_suffixes)}"
        )

        # All layers should have identical suffix sets
        reference = layer_suffixes[0]
        for layer_idx in range(1, num_layers):
            diff = reference.symmetric_difference(layer_suffixes[layer_idx])
            assert not diff, f"Layer {layer_idx} has different weight suffixes than layer 0: {diff}"
