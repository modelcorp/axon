"""Tests for Qwen3NextBridge weight name mappings validated against real HuggingFace checkpoint.

Downloads config.json and model.safetensors.index.json from
Qwen/Qwen3-Next-80B-A3B-Instruct to verify that bridge mappings produce weight
names that actually exist in the checkpoint.

Validates:
- Config model_type matches expected value
- MTP direct, attention, MLP, and expert mappings produce valid checkpoint names
- Linear attention layers (linear_attn.*) and full attention layers (self_attn.*)
  are correctly separated
- The full_attention_interval=4 layer type pattern matches checkpoint contents
- All MTP non-expert weights are reachable through _weight_name_mapping_mtp

Usage:
    pytest tests/models/mbridge/test_qwen3_next_config.py -v
"""

import json
from types import SimpleNamespace

import pytest

from axon.models.mbridge.qwen3_next import Qwen3NextBridge

HF_MODEL_ID = "Qwen/Qwen3-Next-80B-A3B-Instruct"


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
# Helpers
# ---------------------------------------------------------------------------


def _make_bridge():
    """Minimal mock for calling _weight_name_mapping_mtp."""
    return SimpleNamespace(
        config=SimpleNamespace(mtp_num_layers=1, num_layers=48),
    )


def _map_mtp(name):
    """Call Qwen3NextBridge._weight_name_mapping_mtp on a mock bridge."""
    bridge = _make_bridge()
    return Qwen3NextBridge._weight_name_mapping_mtp(bridge, name)


def _layer_weights(hf_weight_names, layer_idx):
    """Return the set of weight names for a specific layer."""
    prefix = f"model.layers.{layer_idx}."
    return {n for n in hf_weight_names if n.startswith(prefix)}


def _mtp_weights(hf_weight_names):
    """Return the set of weight names in the MTP block."""
    return {n for n in hf_weight_names if n.startswith("mtp.")}


def _mtp_non_expert_weights(hf_weight_names):
    """Return MTP weights excluding per-expert weights."""
    return {n for n in _mtp_weights(hf_weight_names) if "mlp.experts." not in n}


# ---------------------------------------------------------------------------
# 1. Config validation
# ---------------------------------------------------------------------------


class TestConfigModelType:
    def test_config_model_type(self, hf_config):
        assert hf_config["model_type"] == "qwen3_next"


# ---------------------------------------------------------------------------
# 2. MTP direct mappings exist in checkpoint
# ---------------------------------------------------------------------------


class TestMTPDirectMappingsExistInCheckpoint:
    """Verify all 4 MTP direct mappings produce names that exist in the checkpoint."""

    @pytest.mark.parametrize(
        "mcore_name,expected_hf_name",
        [
            ("mtp.layers.0.enorm.weight", "mtp.pre_fc_norm_embedding.weight"),
            ("mtp.layers.0.hnorm.weight", "mtp.pre_fc_norm_hidden.weight"),
            ("mtp.layers.0.eh_proj.weight", "mtp.fc.weight"),
            ("mtp.layers.0.final_layernorm.weight", "mtp.norm.weight"),
        ],
    )
    def test_mtp_direct_mappings_exist_in_checkpoint(self, hf_weight_names, mcore_name, expected_hf_name):
        result = _map_mtp(mcore_name)
        assert result == [expected_hf_name]
        assert expected_hf_name in hf_weight_names, (
            f"MTP direct mapping '{mcore_name}' -> '{expected_hf_name}' not found in checkpoint"
        )


# ---------------------------------------------------------------------------
# 3. MTP attention mappings exist in checkpoint
# ---------------------------------------------------------------------------


class TestMTPAttentionMappingsExistInCheckpoint:
    """Verify MTP attention mappings produce names in the checkpoint."""

    def test_input_layernorm(self, hf_weight_names):
        result = _map_mtp("mtp.layers.0.transformer_layer.self_attention.input_layernorm.weight")
        assert result == ["mtp.layers.0.input_layernorm.weight"]
        assert result[0] in hf_weight_names

    def test_linear_proj_to_o_proj(self, hf_weight_names):
        result = _map_mtp("mtp.layers.0.transformer_layer.self_attention.linear_proj.weight")
        assert result == ["mtp.layers.0.self_attn.o_proj.weight"]
        assert result[0] in hf_weight_names

    def test_q_layernorm_to_q_norm(self, hf_weight_names):
        result = _map_mtp("mtp.layers.0.transformer_layer.self_attention.q_layernorm.weight")
        assert result == ["mtp.layers.0.self_attn.q_norm.weight"]
        assert result[0] in hf_weight_names

    def test_k_layernorm_to_k_norm(self, hf_weight_names):
        result = _map_mtp("mtp.layers.0.transformer_layer.self_attention.k_layernorm.weight")
        assert result == ["mtp.layers.0.self_attn.k_norm.weight"]
        assert result[0] in hf_weight_names

    def test_linear_qkv_to_q_k_v_proj(self, hf_weight_names):
        result = _map_mtp("mtp.layers.0.transformer_layer.self_attention.linear_qkv.weight")
        assert len(result) == 3
        expected = {
            "mtp.layers.0.self_attn.q_proj.weight",
            "mtp.layers.0.self_attn.k_proj.weight",
            "mtp.layers.0.self_attn.v_proj.weight",
        }
        assert set(result) == expected
        for name in result:
            assert name in hf_weight_names, f"MTP QKV mapping '{name}' not found in checkpoint"


# ---------------------------------------------------------------------------
# 4. MTP MLP mappings exist in checkpoint
# ---------------------------------------------------------------------------


class TestMTPMLPMappingsExistInCheckpoint:
    """Verify MTP MLP mappings produce names in the checkpoint."""

    def test_shared_expert_fc1(self, hf_weight_names):
        result = _map_mtp("mtp.layers.0.transformer_layer.shared_experts.linear_fc1.weight")
        expected = [
            "mtp.layers.0.mlp.shared_expert.gate_proj.weight",
            "mtp.layers.0.mlp.shared_expert.up_proj.weight",
        ]
        assert result == expected
        for name in result:
            assert name in hf_weight_names

    def test_shared_expert_fc2(self, hf_weight_names):
        result = _map_mtp("mtp.layers.0.transformer_layer.shared_experts.linear_fc2.weight")
        assert result == ["mtp.layers.0.mlp.shared_expert.down_proj.weight"]
        assert result[0] in hf_weight_names

    def test_shared_expert_gate(self, hf_weight_names):
        result = _map_mtp("mtp.layers.0.transformer_layer.shared_experts.gate_weight")
        assert result == ["mtp.layers.0.mlp.shared_expert_gate.weight"]
        assert result[0] in hf_weight_names

    def test_pre_mlp_layernorm_to_post_attention_layernorm(self, hf_weight_names):
        result = _map_mtp("mtp.layers.0.transformer_layer.pre_mlp_layernorm")
        assert result == ["mtp.layers.0.post_attention_layernorm.weight"]
        assert result[0] in hf_weight_names

    def test_router_to_gate(self, hf_weight_names):
        result = _map_mtp("mtp.layers.0.transformer_layer.mlp.router.weight")
        assert result == ["mtp.layers.0.mlp.gate.weight"]
        assert result[0] in hf_weight_names


# ---------------------------------------------------------------------------
# 5. MTP expert mappings exist in checkpoint
# ---------------------------------------------------------------------------


class TestMTPExpertMappingsExistInCheckpoint:
    """Verify grouped gemm expert mapping for expert 0 fc1 produces checkpoint names."""

    def test_expert_0_fc1_gate_and_up(self, hf_weight_names):
        result = _map_mtp("mtp.layers.0.transformer_layer.mlp.experts.linear_fc1.weight0")
        expected = [
            "mtp.layers.0.mlp.experts.0.gate_proj.weight",
            "mtp.layers.0.mlp.experts.0.up_proj.weight",
        ]
        assert result == expected
        for name in result:
            assert name in hf_weight_names, f"MTP expert mapping '{name}' not found in checkpoint"

    def test_expert_0_fc2_down(self, hf_weight_names):
        result = _map_mtp("mtp.layers.0.transformer_layer.mlp.experts.linear_fc2.weight0")
        expected = ["mtp.layers.0.mlp.experts.0.down_proj.weight"]
        assert result == expected
        assert result[0] in hf_weight_names


# ---------------------------------------------------------------------------
# 6. Linear attention layer weights covered
# ---------------------------------------------------------------------------


class TestLinearAttentionLayerWeightsCovered:
    """For layer 0 (linear_attention), verify bridge attention mappings produce
    all linear_attn HF names in the checkpoint."""

    def test_linear_attention_layer_weights_covered(self, hf_weight_names):
        layer_idx = 0
        layer_wts = _layer_weights(hf_weight_names, layer_idx)

        # Expected linear_attn weights from the checkpoint
        expected_linear_attn = {n for n in layer_wts if ".linear_attn." in n}
        assert len(expected_linear_attn) > 0, "Layer 0 should have linear_attn weights"

        # Build the set of HF names the bridge's _ATTENTION_MAPPING can produce for this layer
        produced_linear_attn = set()
        for mcore_key, hf_templates in Qwen3NextBridge._ATTENTION_MAPPING.items():
            for template in hf_templates:
                hf_name = template.format(layer_number=layer_idx)
                if ".linear_attn." in hf_name:
                    produced_linear_attn.add(hf_name)

        missing = expected_linear_attn - produced_linear_attn
        assert not missing, (
            f"Bridge _ATTENTION_MAPPING fails to produce these linear_attn weights for layer {layer_idx}: {missing}"
        )


# ---------------------------------------------------------------------------
# 7. Full attention layer weights covered
# ---------------------------------------------------------------------------


class TestFullAttentionLayerWeightsCovered:
    """For layer 3 (full_attention), verify bridge attention mappings produce
    all self_attn HF names in the checkpoint."""

    def test_full_attention_layer_weights_covered(self, hf_weight_names):
        layer_idx = 3
        layer_wts = _layer_weights(hf_weight_names, layer_idx)

        # Expected self_attn weights from the checkpoint
        expected_self_attn = {n for n in layer_wts if ".self_attn." in n}
        assert len(expected_self_attn) > 0, "Layer 3 should have self_attn weights"

        # Build the set of HF names the bridge's _ATTENTION_MAPPING can produce for this layer
        produced_self_attn = set()
        for mcore_key, hf_templates in Qwen3NextBridge._ATTENTION_MAPPING.items():
            for template in hf_templates:
                hf_name = template.format(layer_number=layer_idx)
                if ".self_attn." in hf_name:
                    produced_self_attn.add(hf_name)

        missing = expected_self_attn - produced_self_attn
        assert not missing, (
            f"Bridge _ATTENTION_MAPPING fails to produce these self_attn weights for layer {layer_idx}: {missing}"
        )


# ---------------------------------------------------------------------------
# 8. Linear attention layers have no self_attn
# ---------------------------------------------------------------------------


class TestLinearAttentionLayersHaveNoSelfAttn:
    def test_linear_attention_layers_have_no_self_attn(self, hf_weight_names):
        layer_wts = _layer_weights(hf_weight_names, 0)
        self_attn_wts = {n for n in layer_wts if ".self_attn." in n}
        assert not self_attn_wts, (
            f"Layer 0 (linear_attention) should have no self_attn weights, but found: {self_attn_wts}"
        )


# ---------------------------------------------------------------------------
# 9. Full attention layers have no linear_attn
# ---------------------------------------------------------------------------


class TestFullAttentionLayersHaveNoLinearAttn:
    def test_full_attention_layers_have_no_linear_attn(self, hf_weight_names):
        layer_wts = _layer_weights(hf_weight_names, 3)
        linear_attn_wts = {n for n in layer_wts if ".linear_attn." in n}
        assert not linear_attn_wts, (
            f"Layer 3 (full_attention) should have no linear_attn weights, but found: {linear_attn_wts}"
        )


# ---------------------------------------------------------------------------
# 10. Layer type pattern
# ---------------------------------------------------------------------------


class TestLayerTypePattern:
    """Verify full_attention_interval=4 pattern matches checkpoint weight names."""

    def test_layer_type_pattern(self, hf_config, hf_weight_names):
        num_layers = hf_config["num_hidden_layers"]
        assert num_layers == 48

        for i in range(num_layers):
            layer_wts = _layer_weights(hf_weight_names, i)
            has_linear_attn = any(".linear_attn." in n for n in layer_wts)
            has_self_attn = any(".self_attn." in n for n in layer_wts)

            is_full_attention = (i + 1) % 4 == 0

            if is_full_attention:
                assert has_self_attn, f"Layer {i} should be full_attention (self_attn) but has none"
                assert not has_linear_attn, f"Layer {i} should be full_attention but has linear_attn weights"
            else:
                assert has_linear_attn, f"Layer {i} should be linear_attention (linear_attn) but has none"
                assert not has_self_attn, f"Layer {i} should be linear_attention but has self_attn weights"


# ---------------------------------------------------------------------------
# 11. All MTP non-expert weights reachable
# ---------------------------------------------------------------------------


class TestAllMTPNonExpertWeightsReachable:
    """Build the complete set of MTP non-expert HF names that _weight_name_mapping_mtp
    can produce and verify it covers all MTP non-expert weights in the checkpoint."""

    def test_all_mtp_non_expert_weights_reachable(self, hf_weight_names):
        checkpoint_mtp_non_expert = _mtp_non_expert_weights(hf_weight_names)
        assert len(checkpoint_mtp_non_expert) > 0, "Should have MTP non-expert weights"

        # Build the set of all HF names producible by _weight_name_mapping_mtp
        produced = set()

        # 1. Direct mappings
        direct_mcore_names = [
            "mtp.layers.0.enorm.weight",
            "mtp.layers.0.hnorm.weight",
            "mtp.layers.0.eh_proj.weight",
            "mtp.layers.0.final_layernorm.weight",
        ]
        for name in direct_mcore_names:
            produced.update(_map_mtp(name))

        # 2. Attention mappings (transformer_layer.self_attention.*)
        attn_mcore_names = [
            "mtp.layers.0.transformer_layer.self_attention.input_layernorm.weight",
            "mtp.layers.0.transformer_layer.self_attention.linear_proj.weight",
            "mtp.layers.0.transformer_layer.self_attention.q_layernorm.weight",
            "mtp.layers.0.transformer_layer.self_attention.k_layernorm.weight",
            "mtp.layers.0.transformer_layer.self_attention.linear_qkv.weight",
        ]
        for name in attn_mcore_names:
            produced.update(_map_mtp(name))

        # 3. MLP mappings
        mlp_mcore_names = [
            "mtp.layers.0.transformer_layer.shared_experts.linear_fc1.weight",
            "mtp.layers.0.transformer_layer.shared_experts.linear_fc2.weight",
            "mtp.layers.0.transformer_layer.shared_experts.gate_weight",
            "mtp.layers.0.transformer_layer.pre_mlp_layernorm",
            "mtp.layers.0.transformer_layer.mlp.router.weight",
        ]
        for name in mlp_mcore_names:
            produced.update(_map_mtp(name))

        missing = checkpoint_mtp_non_expert - produced
        assert not missing, (
            f"The following MTP non-expert weights in the checkpoint are not reachable "
            f"through _weight_name_mapping_mtp:\n{sorted(missing)}"
        )
