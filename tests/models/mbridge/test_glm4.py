"""Tests for GLM-4 bridge weight name mapping and class attributes.

Verifies:
- _DIRECT_MAPPING, _ATTENTION_MAPPING, _MLP_MAPPING structure and values
- _weight_name_mapping_mcore_to_hf handles direct, layernorm, attention,
  MLP, extra_state (AssertionError), and unknown (NotImplementedError)
- Branch ordering in _weight_name_mapping_mcore_to_hf matters

Usage:
    pytest tests/models/mbridge/test_glm4.py -v
"""

from types import SimpleNamespace

import pytest

from axon.models.mbridge.glm4 import GLM4Bridge

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_glm4_mock():
    """Minimal mock with attributes needed by _weight_name_mapping_mcore_to_hf.

    GLM4Bridge._weight_name_mapping_mcore_to_hf accesses:
      - self._DIRECT_MAPPING
      - self._weight_name_mapping_attention(name)
      - self._weight_name_mapping_mlp(name)
    """
    return SimpleNamespace(
        _DIRECT_MAPPING=GLM4Bridge._DIRECT_MAPPING,
        _weight_name_mapping_attention=lambda name: [name + "::attn"],
        _weight_name_mapping_mlp=lambda name: [name + "::mlp"],
    )


def _map_name(bridge, name):
    """Call the real GLM4Bridge method on a mock instance."""
    return GLM4Bridge._weight_name_mapping_mcore_to_hf(bridge, name)


# ---------------------------------------------------------------------------
# _DIRECT_MAPPING -- one comprehensive test
# ---------------------------------------------------------------------------


class TestDirectMapping:
    def test_all_direct_mappings_present_with_correct_values(self):
        expected = {
            "embedding.word_embeddings.weight": "model.embed_tokens.weight",
            "decoder.final_layernorm.weight": "model.norm.weight",
            "output_layer.weight": "lm_head.weight",
        }
        assert GLM4Bridge._DIRECT_MAPPING == expected


# ---------------------------------------------------------------------------
# _ATTENTION_MAPPING -- comprehensive tests
# ---------------------------------------------------------------------------


class TestAttentionMapping:
    EXPECTED_KEYS = {
        "self_attention.linear_proj.weight",
        "self_attention.linear_qkv.layer_norm_weight",
        "self_attention.q_layernorm.weight",
        "self_attention.k_layernorm.weight",
        "self_attention.linear_qkv.weight",
        "self_attention.linear_qkv.bias",
    }

    def test_all_expected_keys_present(self):
        assert set(GLM4Bridge._ATTENTION_MAPPING.keys()) == self.EXPECTED_KEYS

    def test_all_hf_names_contain_layer_number_template(self):
        for key, hf_names in GLM4Bridge._ATTENTION_MAPPING.items():
            for hf_name in hf_names:
                assert "{layer_number}" in hf_name, f"HF name for '{key}' missing {{layer_number}}: {hf_name}"

    def test_qkv_splits_to_three_projections(self):
        """Both linear_qkv.weight and linear_qkv.bias must split to q/k/v."""
        for suffix in ("weight", "bias"):
            hf_names = GLM4Bridge._ATTENTION_MAPPING[f"self_attention.linear_qkv.{suffix}"]
            assert len(hf_names) == 3
            assert any(f"q_proj.{suffix}" in n for n in hf_names)
            assert any(f"k_proj.{suffix}" in n for n in hf_names)
            assert any(f"v_proj.{suffix}" in n for n in hf_names)


# ---------------------------------------------------------------------------
# _MLP_MAPPING -- one comprehensive test
# ---------------------------------------------------------------------------


class TestMLPMapping:
    def test_all_mlp_mappings_structure_and_values(self):
        m = GLM4Bridge._MLP_MAPPING
        expected_keys = {
            "mlp.linear_fc1.weight",
            "mlp.linear_fc1.layer_norm_weight",
            "mlp.linear_fc2.weight",
        }
        assert set(m.keys()) == expected_keys
        # fc1 -> gate_up_proj (single name)
        assert len(m["mlp.linear_fc1.weight"]) == 1
        assert "gate_up_proj" in m["mlp.linear_fc1.weight"][0]
        # fc2 -> down_proj (single name)
        assert len(m["mlp.linear_fc2.weight"]) == 1
        assert "down_proj" in m["mlp.linear_fc2.weight"][0]
        # layer_norm_weight -> post_attention_layernorm
        assert "post_attention_layernorm" in m["mlp.linear_fc1.layer_norm_weight"][0]


# ---------------------------------------------------------------------------
# _weight_name_mapping_mcore_to_hf -- hardened
# ---------------------------------------------------------------------------


class TestWeightNameMappingMcoreToHf:
    @pytest.fixture
    def bridge(self):
        return _make_glm4_mock()

    # --- Direct mappings: parametrized ---
    @pytest.mark.parametrize(
        "mcore_name, expected_hf_name",
        [
            ("embedding.word_embeddings.weight", "model.embed_tokens.weight"),
            ("decoder.final_layernorm.weight", "model.norm.weight"),
            ("output_layer.weight", "lm_head.weight"),
        ],
    )
    def test_direct_mapping(self, bridge, mcore_name, expected_hf_name):
        assert _map_name(bridge, mcore_name) == [expected_hf_name]

    # --- Layernorm extraction: includes 3-digit layer number ---
    @pytest.mark.parametrize("layer", [0, 5, 31, 100])
    def test_post_self_attn_layernorm(self, bridge, layer):
        result = _map_name(bridge, f"decoder.layers.{layer}.post_self_attn_layernorm.weight")
        assert result == [f"model.layers.{layer}.post_self_attn_layernorm.weight"]

    @pytest.mark.parametrize("layer", [0, 5, 100])
    def test_post_mlp_layernorm(self, bridge, layer):
        result = _map_name(bridge, f"decoder.layers.{layer}.post_mlp_layernorm.weight")
        assert result == [f"model.layers.{layer}.post_mlp_layernorm.weight"]

    # --- Branch ordering: layernorm branches win over self_attention/mlp ---
    def test_post_self_attn_layernorm_not_routed_to_attention(self, bridge):
        """post_self_attn_layernorm contains 'self_attention' is not in name,
        but it does NOT go through the attention delegation branch.
        Verify it returns the layernorm path, not the ::attn mock path."""
        result = _map_name(bridge, "decoder.layers.7.post_self_attn_layernorm.weight")
        assert result == ["model.layers.7.post_self_attn_layernorm.weight"]
        assert not any("::attn" in r for r in result)

    def test_post_mlp_layernorm_not_routed_to_mlp(self, bridge):
        """post_mlp_layernorm contains 'mlp' is not in name before the branch,
        but verify it does NOT go through the mlp delegation branch."""
        result = _map_name(bridge, "decoder.layers.7.post_mlp_layernorm.weight")
        assert result == ["model.layers.7.post_mlp_layernorm.weight"]
        assert not any("::mlp" in r for r in result)

    # --- All _ATTENTION_MAPPING keys go through attention delegation ---
    @pytest.mark.parametrize("attn_key", list(GLM4Bridge._ATTENTION_MAPPING.keys()))
    def test_attention_mapping_keys_delegate_to_attention(self, bridge, attn_key):
        name = f"decoder.layers.3.{attn_key}"
        result = _map_name(bridge, name)
        assert isinstance(result, list)
        assert all("::attn" in r for r in result), f"Expected attention delegation for '{attn_key}', got {result}"

    # --- All _MLP_MAPPING keys go through mlp delegation ---
    @pytest.mark.parametrize("mlp_key", list(GLM4Bridge._MLP_MAPPING.keys()))
    def test_mlp_mapping_keys_delegate_to_mlp(self, bridge, mlp_key):
        name = f"decoder.layers.3.{mlp_key}"
        result = _map_name(bridge, name)
        assert isinstance(result, list)
        assert all("::mlp" in r for r in result), f"Expected MLP delegation for '{mlp_key}', got {result}"

    # --- Extra state raises AssertionError ---
    def test_extra_state_raises_assertion(self, bridge):
        with pytest.raises(AssertionError):
            _map_name(bridge, "decoder.layers.0.self_attention._extra_state")

    # --- Unknown name raises NotImplementedError ---
    @pytest.mark.parametrize(
        "unknown_name",
        [
            "decoder.final_layernorm.bias",
            "some.unknown.parameter.weight",
        ],
    )
    def test_unknown_name_raises(self, bridge, unknown_name):
        with pytest.raises(NotImplementedError, match="Unsupported parameter name"):
            _map_name(bridge, unknown_name)
