"""Tests for GPT-OSS bridge weight name mapping and class attributes.

Verifies:
- _extract_expert_id_from_name parses GroupedMLP and SequentialMLP formats
  including multi-digit IDs, boundary IDs, and error cases
- _is_expert_weight identifies expert vs non-expert weights
- _ATTENTION_MAPPING extends Qwen2MoEBridge with GPT-OSS-specific keys
- _MLP_MAPPING contains router, pre_mlp_layernorm, and expert mappings
- _SKIP_LOADING_WEIGHTS and _NON_TP_WEIGHT_PATTERNS structure
- _weight_name_mapping_mlp handles router, layernorm, experts, per-expert,
  different layer numbers, and larger expert counts
- _weight_merge_across_tp handles expert biases, fc2 weights, non-expert
  fallback, fc1 weight fallback, and empty weights edge case

Usage:
    pytest tests/models/mbridge/test_gpt_oss.py -v
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from axon.models.mbridge.gpt_oss import GPTOSSBridge

# ---------------------------------------------------------------------------
# _extract_expert_id_from_name
# ---------------------------------------------------------------------------


class TestExtractExpertIdFromName:
    def _extract(self, name):
        return GPTOSSBridge._extract_expert_id_from_name(SimpleNamespace(), name)

    def test_grouped_mlp_weight(self):
        assert self._extract("decoder.layers.0.mlp.experts.linear_fc1.weight3") == (3, "weight")

    def test_grouped_mlp_bias(self):
        assert self._extract("decoder.layers.0.mlp.experts.linear_fc1.bias5") == (5, "bias")

    def test_sequential_mlp_weight(self):
        assert self._extract("decoder.layers.0.mlp.experts.local_experts.7.linear_fc1.weight") == (7, "weight")

    def test_sequential_mlp_bias(self):
        assert self._extract("decoder.layers.0.mlp.experts.local_experts.2.linear_fc2.bias") == (2, "bias")

    def test_multi_digit_expert_id(self):
        """127 is a valid multi-digit expert ID."""
        assert self._extract("decoder.layers.5.mlp.experts.linear_fc2.weight127") == (127, "weight")

    def test_expert_id_zero_boundary(self):
        """Expert ID 0 is a valid boundary case."""
        assert self._extract("decoder.layers.0.mlp.experts.linear_fc1.weight0") == (0, "weight")

    def test_sequential_expert_large_id(self):
        """Sequential expert with large ID (63)."""
        assert self._extract("decoder.layers.0.mlp.experts.local_experts.63.linear_fc2.weight") == (63, "weight")

    def test_name_with_weight_substring_but_no_trailing_digits_raises(self):
        """A grouped name where the 'weight' suffix has no trailing digits should raise."""
        with pytest.raises(ValueError, match="Cannot extract expert_id"):
            self._extract("decoder.layers.0.mlp.experts.linear_fc1.weight")

    def test_invalid_name_raises_value_error(self):
        with pytest.raises(ValueError, match="Cannot extract expert_id"):
            self._extract("decoder.layers.0.mlp.router.weight")


# ---------------------------------------------------------------------------
# _is_expert_weight
# ---------------------------------------------------------------------------


class TestIsExpertWeight:
    def _check(self, name):
        return GPTOSSBridge._is_expert_weight(SimpleNamespace(), name)

    def test_expert_fc1_weight_is_true(self):
        assert self._check("decoder.layers.0.mlp.experts.linear_fc1.weight0") is True

    def test_expert_fc2_bias_is_true(self):
        assert self._check("decoder.layers.0.mlp.experts.linear_fc2.bias3") is True

    def test_sequential_expert_is_true(self):
        assert self._check("decoder.layers.0.mlp.experts.local_experts.0.linear_fc1.weight") is True

    def test_router_weight_is_false(self):
        assert self._check("decoder.layers.0.mlp.experts.router.weight") is False

    def test_attention_weight_is_false(self):
        assert self._check("decoder.layers.0.self_attention.linear_proj.weight") is False

    def test_non_expert_mlp_is_false(self):
        assert self._check("decoder.layers.0.mlp.linear_fc1.weight") is False


# ---------------------------------------------------------------------------
# _ATTENTION_MAPPING -- consolidated
# ---------------------------------------------------------------------------


class TestAttentionMapping:
    def test_preserves_all_qwen2moe_base_keys(self):
        from mbridge.models.qwen2moe import Qwen2MoEBridge

        base_keys = set(Qwen2MoEBridge._ATTENTION_MAPPING.keys())
        assert base_keys.issubset(set(GPTOSSBridge._ATTENTION_MAPPING.keys()))

    def test_all_gpt_oss_specific_keys_present(self):
        expected_gpt_oss_keys = {
            "self_attention.input_layernorm.weight",
            "self_attention.self_attn.k_proj.weight",
            "self_attention.self_attn.k_proj.bias",
            "self_attention.self_attn.o_proj.weight",
            "self_attention.self_attn.o_proj.bias",
            "self_attention.self_attn.q_proj.weight",
            "self_attention.self_attn.q_proj.bias",
            "self_attention.self_attn.v_proj.weight",
            "self_attention.self_attn.v_proj.bias",
            "self_attention.self_attn.sinks",
            "self_attention.self_attn.q_norm.weight",
            "self_attention.self_attn.k_norm.weight",
        }
        assert expected_gpt_oss_keys.issubset(set(GPTOSSBridge._ATTENTION_MAPPING.keys()))

    def test_all_hf_names_contain_layer_number_template(self):
        for key, hf_names in GPTOSSBridge._ATTENTION_MAPPING.items():
            for hf_name in hf_names:
                assert "{layer_number}" in hf_name, f"HF name for '{key}' missing {{layer_number}}: {hf_name}"


# ---------------------------------------------------------------------------
# _MLP_MAPPING -- one comprehensive test
# ---------------------------------------------------------------------------


class TestMLPMapping:
    def test_all_mlp_mappings_structure(self):
        m = GPTOSSBridge._MLP_MAPPING
        expected_keys = {
            "mlp.router.weight",
            "mlp.router.bias",
            "pre_mlp_layernorm",
            "mlp.experts.linear_fc1.weight",
            "mlp.experts.linear_fc2.weight",
            "mlp.experts.linear_fc1.bias",
            "mlp.experts.linear_fc2.bias",
        }
        assert set(m.keys()) == expected_keys
        # pre_mlp_layernorm -> post_attention_layernorm
        assert "post_attention_layernorm" in m["pre_mlp_layernorm"][0]
        # fc1 -> gate_up_proj
        assert any("gate_up_proj" in n for n in m["mlp.experts.linear_fc1.weight"])
        # fc2 -> down_proj
        assert any("down_proj" in n for n in m["mlp.experts.linear_fc2.weight"])


# ---------------------------------------------------------------------------
# _SKIP_LOADING_WEIGHTS and _NON_TP_WEIGHT_PATTERNS -- merged
# ---------------------------------------------------------------------------


class TestClassAttributes:
    def test_skip_loading_weights(self):
        assert isinstance(GPTOSSBridge._SKIP_LOADING_WEIGHTS, set)
        assert GPTOSSBridge._SKIP_LOADING_WEIGHTS == {
            "self_attn.q_norm.weight",
            "self_attn.k_norm.weight",
        }

    def test_non_tp_weight_patterns(self):
        assert isinstance(GPTOSSBridge._NON_TP_WEIGHT_PATTERNS, list)
        patterns = GPTOSSBridge._NON_TP_WEIGHT_PATTERNS
        assert any("self_attention.self_attn." in p for p in patterns)
        assert any("self_attention.linear_attn." in p for p in patterns)
        assert any("self_attention.rotary_emb." in p for p in patterns)


# ---------------------------------------------------------------------------
# _weight_name_mapping_mlp (requires mock with config)
# ---------------------------------------------------------------------------


def _make_gpt_oss_mlp_mock(num_moe_experts=4):
    return SimpleNamespace(
        config=SimpleNamespace(num_moe_experts=num_moe_experts),
        _MLP_MAPPING=GPTOSSBridge._MLP_MAPPING,
    )


def _map_mlp(bridge, name):
    return GPTOSSBridge._weight_name_mapping_mlp(bridge, name)


class TestWeightNameMappingMlp:
    @pytest.fixture
    def bridge(self):
        return _make_gpt_oss_mlp_mock(num_moe_experts=4)

    def test_router_weight(self, bridge):
        result = _map_mlp(bridge, "decoder.layers.3.mlp.router.weight")
        assert result == ["model.layers.3.mlp.router.weight"]

    def test_router_bias(self, bridge):
        result = _map_mlp(bridge, "decoder.layers.3.mlp.router.bias")
        assert result == ["model.layers.3.mlp.router.bias"]

    def test_pre_mlp_layernorm(self, bridge):
        result = _map_mlp(bridge, "decoder.layers.3.pre_mlp_layernorm")
        assert result == ["model.layers.3.post_attention_layernorm.weight"]

    def test_expert_fc1_weight(self, bridge):
        result = _map_mlp(bridge, "decoder.layers.3.mlp.experts.linear_fc1.weight")
        assert any("gate_up_proj" in n for n in result)

    def test_expert_fc2_weight(self, bridge):
        result = _map_mlp(bridge, "decoder.layers.3.mlp.experts.linear_fc2.weight")
        assert any("down_proj" in n for n in result)

    def test_per_expert_fc2_weight(self, bridge):
        """Per-expert mapping: local_experts.0.linear_fc2 -> down_proj."""
        result = _map_mlp(bridge, "decoder.layers.3.mlp.experts.local_experts.0.linear_fc2.weight")
        assert any("down_proj" in n for n in result)

    def test_per_expert_fc1_weight(self, bridge):
        """Per-expert mapping: local_experts.2.linear_fc1 -> gate_up_proj."""
        result = _map_mlp(bridge, "decoder.layers.3.mlp.experts.local_experts.2.linear_fc1.weight")
        assert any("gate_up_proj" in n for n in result)

    def test_layer_number_correctly_extracted(self, bridge):
        """Different layer numbers appear in the output."""
        for layer_num in [0, 7, 31]:
            result = _map_mlp(bridge, f"decoder.layers.{layer_num}.mlp.router.weight")
            assert result == [f"model.layers.{layer_num}.mlp.router.weight"]

    def test_with_num_moe_experts_64(self):
        """Larger expert count: 64 experts should work for sequential format."""
        bridge = _make_gpt_oss_mlp_mock(num_moe_experts=64)
        result = _map_mlp(bridge, "decoder.layers.3.mlp.experts.local_experts.63.linear_fc2.weight")
        assert any("down_proj" in n for n in result)

    def test_unknown_name_raises(self, bridge):
        with pytest.raises(NotImplementedError, match="Unsupported parameter name"):
            _map_mlp(bridge, "decoder.layers.3.unknown.parameter")


# ---------------------------------------------------------------------------
# _weight_merge_across_tp (CPU tensor tests)
# ---------------------------------------------------------------------------


def _make_gpt_oss_merge_mock():
    """Mock for _weight_merge_across_tp.

    GPTOSSBridge._weight_merge_across_tp calls:
      - self._is_expert_weight(name)
      - super()._weight_merge_across_tp(name, weights, param)  [fallback]

    We mock _is_expert_weight to use the real method and provide a parent fallback.
    """
    mock = MagicMock(spec=GPTOSSBridge)
    mock._is_expert_weight = lambda name: GPTOSSBridge._is_expert_weight(mock, name)
    return mock


class TestWeightMergeAcrossTp:
    def test_expert_bias_returns_single_tensor(self):
        mock = _make_gpt_oss_merge_mock()
        bias = torch.randn(16)
        result = GPTOSSBridge._weight_merge_across_tp(
            mock,
            "decoder.layers.0.mlp.experts.linear_fc1.bias3",
            [bias, torch.randn(16)],
            bias,
        )
        # Expert biases are not TP-sharded, so the first tensor is returned
        torch.testing.assert_close(result, bias)

    def test_expert_fc2_weight_concat_dim1(self):
        mock = _make_gpt_oss_merge_mock()
        w1 = torch.randn(32, 8)
        w2 = torch.randn(32, 8)
        result = GPTOSSBridge._weight_merge_across_tp(
            mock,
            "decoder.layers.0.mlp.experts.linear_fc2.weight0",
            [w1, w2],
            w1,
        )
        assert result.shape == (32, 16)
        torch.testing.assert_close(result, torch.cat([w1, w2], dim=1))

    def test_expert_fc2_single_weight(self):
        mock = _make_gpt_oss_merge_mock()
        w = torch.randn(32, 16)
        result = GPTOSSBridge._weight_merge_across_tp(
            mock,
            "decoder.layers.0.mlp.experts.linear_fc2.weight0",
            [w],
            w,
        )
        torch.testing.assert_close(result, w)

    def test_sequential_expert_fc2_weight_concat_dim1(self):
        mock = _make_gpt_oss_merge_mock()
        w1 = torch.randn(32, 8)
        w2 = torch.randn(32, 8)
        result = GPTOSSBridge._weight_merge_across_tp(
            mock,
            "decoder.layers.0.mlp.experts.local_experts.0.linear_fc2.weight",
            [w1, w2],
            w1,
        )
        assert result.shape == (32, 16)
        torch.testing.assert_close(result, torch.cat([w1, w2], dim=1))

    def test_non_expert_weight_falls_to_parent(self):
        """Non-expert weight should call the parent (super) _weight_merge_across_tp."""
        from unittest.mock import patch

        from mbridge.core import Bridge

        parent_result = torch.randn(32, 16)
        mock = _make_gpt_oss_merge_mock()

        with patch.object(Bridge, "_weight_merge_across_tp", return_value=parent_result) as patched:
            result = GPTOSSBridge._weight_merge_across_tp(
                mock,
                "decoder.layers.0.self_attention.linear_proj.weight",
                [torch.randn(32, 16)],
                torch.randn(32, 16),
            )
            patched.assert_called_once()
        torch.testing.assert_close(result, parent_result)

    def test_expert_fc1_weight_falls_to_parent(self):
        """Expert fc1 weight is NOT special-cased for fc2, so it goes to parent fallback."""
        from unittest.mock import patch

        from mbridge.core import Bridge

        parent_result = torch.randn(32, 16)
        mock = _make_gpt_oss_merge_mock()

        with patch.object(Bridge, "_weight_merge_across_tp", return_value=parent_result) as patched:
            result = GPTOSSBridge._weight_merge_across_tp(
                mock,
                "decoder.layers.0.mlp.experts.linear_fc1.weight0",
                [torch.randn(32, 16)],
                torch.randn(32, 16),
            )
            patched.assert_called_once()
        torch.testing.assert_close(result, parent_result)

    def test_expert_fc2_empty_weights_list(self):
        """Edge case: empty weights list for fc2. len==0 is neither 1 nor >1, so cat is called."""
        mock = _make_gpt_oss_merge_mock()
        # torch.cat on an empty list raises ValueError (post-torch-2.0 behaviour).
        with pytest.raises((ValueError, RuntimeError)):
            GPTOSSBridge._weight_merge_across_tp(
                mock,
                "decoder.layers.0.mlp.experts.linear_fc2.weight0",
                [],
                torch.randn(32, 16),
            )
