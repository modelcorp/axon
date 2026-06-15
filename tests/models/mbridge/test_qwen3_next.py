"""Tests for Qwen3Next bridge MTP mapping, attention mapping, and weight conversion.

Verifies:
- _weight_name_mapping_mtp handles direct, attention, MLP, expert, and unknown MTP names
- _ATTENTION_MAPPING extends Qwen2MoE with Qwen3Next-specific hybrid attention keys
- _weight_to_hf_format handles MTP fc1 splitting, vocab padding, qkv passthrough
- _weight_name_mapping_mcore_to_hf dispatches MTP vs non-MTP names

Usage:
    pytest tests/models/mbridge/test_qwen3_next.py -v
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from axon.models.mbridge.qwen3_next import Qwen3NextBridge

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_qwen3_mock():
    """Minimal mock for _weight_name_mapping_mtp.

    The method accesses self.config.mtp_num_layers and self.config.num_layers.
    """
    return SimpleNamespace(
        config=SimpleNamespace(mtp_num_layers=1, num_layers=28),
    )


def _map_mtp(bridge, name):
    """Call the real Qwen3NextBridge._weight_name_mapping_mtp on a mock."""
    return Qwen3NextBridge._weight_name_mapping_mtp(bridge, name)


# ---------------------------------------------------------------------------
# _ATTENTION_MAPPING
# ---------------------------------------------------------------------------


class TestAttentionMapping:
    def test_all_qwen2moe_base_keys_preserved(self):
        from mbridge.models.qwen2moe import Qwen2MoEBridge

        base_keys = set(Qwen2MoEBridge._ATTENTION_MAPPING.keys())
        qwen3_keys = set(Qwen3NextBridge._ATTENTION_MAPPING.keys())
        assert base_keys <= qwen3_keys, f"Missing base keys: {base_keys - qwen3_keys}"

    def test_all_qwen3next_specific_keys_present(self):
        expected_specific = {
            "self_attention.input_layernorm.weight",
            "self_attention.linear_attn.A_log",
            "self_attention.linear_attn.conv1d.weight",
            "self_attention.linear_attn.dt_bias",
            "self_attention.linear_attn.in_proj_ba.weight",
            "self_attention.linear_attn.in_proj_qkvz.weight",
            "self_attention.linear_attn.norm.weight",
            "self_attention.linear_attn.out_proj.weight",
            "self_attention.self_attn.k_norm.weight",
            "self_attention.self_attn.k_proj.weight",
            "self_attention.self_attn.o_proj.weight",
            "self_attention.self_attn.q_norm.weight",
            "self_attention.self_attn.q_proj.weight",
            "self_attention.self_attn.v_proj.weight",
            "self_attention.rotary_emb.inv_freq",
        }
        actual_keys = set(Qwen3NextBridge._ATTENTION_MAPPING.keys())
        assert expected_specific <= actual_keys, f"Missing Qwen3Next-specific keys: {expected_specific - actual_keys}"

    def test_all_hf_names_contain_layer_number_template(self):
        for key, hf_names in Qwen3NextBridge._ATTENTION_MAPPING.items():
            for hf_name in hf_names:
                assert "{layer_number}" in hf_name, f"HF name for '{key}' missing {{layer_number}}: {hf_name}"


# ---------------------------------------------------------------------------
# _weight_name_mapping_mtp -- direct mappings
# ---------------------------------------------------------------------------


class TestMTPDirectMappings:
    @pytest.fixture
    def bridge(self):
        return _make_qwen3_mock()

    def test_enorm(self, bridge):
        result = _map_mtp(bridge, "mtp.layers.0.enorm.weight")
        assert result == ["mtp.pre_fc_norm_embedding.weight"]

    def test_hnorm(self, bridge):
        result = _map_mtp(bridge, "mtp.layers.0.hnorm.weight")
        assert result == ["mtp.pre_fc_norm_hidden.weight"]

    def test_eh_proj(self, bridge):
        result = _map_mtp(bridge, "mtp.layers.0.eh_proj.weight")
        assert result == ["mtp.fc.weight"]

    def test_final_layernorm(self, bridge):
        result = _map_mtp(bridge, "mtp.layers.0.final_layernorm.weight")
        assert result == ["mtp.norm.weight"]


# ---------------------------------------------------------------------------
# _weight_name_mapping_mtp -- attention mappings
# ---------------------------------------------------------------------------


class TestMTPAttentionMappings:
    @pytest.fixture
    def bridge(self):
        return _make_qwen3_mock()

    def test_input_layernorm(self, bridge):
        name = "mtp.layers.0.transformer_layer.self_attention.input_layernorm.weight"
        result = _map_mtp(bridge, name)
        assert result == ["mtp.layers.0.input_layernorm.weight"]

    def test_linear_proj(self, bridge):
        name = "mtp.layers.0.transformer_layer.self_attention.linear_proj.weight"
        result = _map_mtp(bridge, name)
        assert result == ["mtp.layers.0.self_attn.o_proj.weight"]

    def test_q_layernorm(self, bridge):
        name = "mtp.layers.0.transformer_layer.self_attention.q_layernorm.weight"
        result = _map_mtp(bridge, name)
        assert result == ["mtp.layers.0.self_attn.q_norm.weight"]

    def test_k_layernorm(self, bridge):
        name = "mtp.layers.0.transformer_layer.self_attention.k_layernorm.weight"
        result = _map_mtp(bridge, name)
        assert result == ["mtp.layers.0.self_attn.k_norm.weight"]

    def test_linear_qkv_weight(self, bridge):
        name = "mtp.layers.0.transformer_layer.self_attention.linear_qkv.weight"
        result = _map_mtp(bridge, name)
        assert len(result) == 3
        assert "mtp.layers.0.self_attn.q_proj.weight" in result
        assert "mtp.layers.0.self_attn.k_proj.weight" in result
        assert "mtp.layers.0.self_attn.v_proj.weight" in result

    def test_linear_qkv_bias(self, bridge):
        name = "mtp.layers.0.transformer_layer.self_attention.linear_qkv.bias"
        result = _map_mtp(bridge, name)
        assert len(result) == 3
        assert "mtp.layers.0.self_attn.q_proj.bias" in result
        assert "mtp.layers.0.self_attn.k_proj.bias" in result
        assert "mtp.layers.0.self_attn.v_proj.bias" in result


# ---------------------------------------------------------------------------
# _weight_name_mapping_mtp -- linear_attn and self_attn passthrough
# ---------------------------------------------------------------------------


class TestMTPPassthroughMappings:
    @pytest.fixture
    def bridge(self):
        return _make_qwen3_mock()

    def test_linear_attn_a_log(self, bridge):
        name = "mtp.layers.0.transformer_layer.self_attention.linear_attn.A_log"
        result = _map_mtp(bridge, name)
        assert result == ["mtp.layers.0.linear_attn.A_log"]

    def test_linear_attn_conv1d_weight(self, bridge):
        name = "mtp.layers.0.transformer_layer.self_attention.linear_attn.conv1d.weight"
        result = _map_mtp(bridge, name)
        assert result == ["mtp.layers.0.linear_attn.conv1d.weight"]

    def test_linear_attn_dt_bias(self, bridge):
        name = "mtp.layers.0.transformer_layer.self_attention.linear_attn.dt_bias"
        result = _map_mtp(bridge, name)
        assert result == ["mtp.layers.0.linear_attn.dt_bias"]

    def test_self_attn_q_proj_weight(self, bridge):
        name = "mtp.layers.0.transformer_layer.self_attention.self_attn.q_proj.weight"
        result = _map_mtp(bridge, name)
        assert result == ["mtp.layers.0.self_attn.q_proj.weight"]

    def test_self_attn_k_proj_weight(self, bridge):
        name = "mtp.layers.0.transformer_layer.self_attention.self_attn.k_proj.weight"
        result = _map_mtp(bridge, name)
        assert result == ["mtp.layers.0.self_attn.k_proj.weight"]

    def test_self_attn_v_proj_weight(self, bridge):
        name = "mtp.layers.0.transformer_layer.self_attention.self_attn.v_proj.weight"
        result = _map_mtp(bridge, name)
        assert result == ["mtp.layers.0.self_attn.v_proj.weight"]

    def test_self_attn_o_proj_weight(self, bridge):
        name = "mtp.layers.0.transformer_layer.self_attention.self_attn.o_proj.weight"
        result = _map_mtp(bridge, name)
        assert result == ["mtp.layers.0.self_attn.o_proj.weight"]


# ---------------------------------------------------------------------------
# _weight_name_mapping_mtp -- MLP mappings
# ---------------------------------------------------------------------------


class TestMTPMLPMappings:
    @pytest.fixture
    def bridge(self):
        return _make_qwen3_mock()

    def test_shared_experts_fc1_weight(self, bridge):
        name = "mtp.layers.0.transformer_layer.shared_experts.linear_fc1.weight"
        result = _map_mtp(bridge, name)
        assert len(result) == 2
        assert "mtp.layers.0.mlp.shared_expert.gate_proj.weight" in result
        assert "mtp.layers.0.mlp.shared_expert.up_proj.weight" in result

    def test_shared_experts_fc2_weight(self, bridge):
        name = "mtp.layers.0.transformer_layer.shared_experts.linear_fc2.weight"
        result = _map_mtp(bridge, name)
        assert result == ["mtp.layers.0.mlp.shared_expert.down_proj.weight"]

    def test_shared_experts_gate_weight(self, bridge):
        name = "mtp.layers.0.transformer_layer.shared_experts.gate_weight"
        result = _map_mtp(bridge, name)
        assert result == ["mtp.layers.0.mlp.shared_expert_gate.weight"]

    def test_pre_mlp_layernorm(self, bridge):
        name = "mtp.layers.0.transformer_layer.pre_mlp_layernorm"
        result = _map_mtp(bridge, name)
        assert result == ["mtp.layers.0.post_attention_layernorm.weight"]

    def test_router_weight(self, bridge):
        name = "mtp.layers.0.transformer_layer.mlp.router.weight"
        result = _map_mtp(bridge, name)
        assert result == ["mtp.layers.0.mlp.gate.weight"]


# ---------------------------------------------------------------------------
# _weight_name_mapping_mtp -- expert mappings (grouped gemm format)
# ---------------------------------------------------------------------------


class TestMTPExpertsGroupedGemm:
    @pytest.fixture
    def bridge(self):
        return _make_qwen3_mock()

    def test_fc1_weight_expert3(self, bridge):
        name = "mtp.layers.0.transformer_layer.mlp.experts.linear_fc1.weight3"
        result = _map_mtp(bridge, name)
        assert len(result) == 2
        assert "mtp.layers.0.mlp.experts.3.gate_proj.weight" in result
        assert "mtp.layers.0.mlp.experts.3.up_proj.weight" in result

    def test_fc2_weight_expert5(self, bridge):
        name = "mtp.layers.0.transformer_layer.mlp.experts.linear_fc2.weight5"
        result = _map_mtp(bridge, name)
        assert result == ["mtp.layers.0.mlp.experts.5.down_proj.weight"]

    def test_fc1_bias_expert0(self, bridge):
        name = "mtp.layers.0.transformer_layer.mlp.experts.linear_fc1.bias0"
        result = _map_mtp(bridge, name)
        assert len(result) == 2
        assert "mtp.layers.0.mlp.experts.0.gate_proj.bias" in result
        assert "mtp.layers.0.mlp.experts.0.up_proj.bias" in result

    def test_fc2_bias_expert1(self, bridge):
        name = "mtp.layers.0.transformer_layer.mlp.experts.linear_fc2.bias1"
        result = _map_mtp(bridge, name)
        assert result == ["mtp.layers.0.mlp.experts.1.down_proj.bias"]


# ---------------------------------------------------------------------------
# _weight_name_mapping_mtp -- expert mappings (sequential format)
# ---------------------------------------------------------------------------


class TestMTPExpertsSequential:
    @pytest.fixture
    def bridge(self):
        return _make_qwen3_mock()

    def test_sequential_fc1_weight_expert2(self, bridge):
        name = "mtp.layers.0.transformer_layer.mlp.experts.local_experts.2.linear_fc1.weight"
        result = _map_mtp(bridge, name)
        assert len(result) == 2
        assert "mtp.layers.0.mlp.experts.2.gate_proj.weight" in result
        assert "mtp.layers.0.mlp.experts.2.up_proj.weight" in result

    def test_sequential_fc2_weight_expert4(self, bridge):
        name = "mtp.layers.0.transformer_layer.mlp.experts.local_experts.4.linear_fc2.weight"
        result = _map_mtp(bridge, name)
        assert result == ["mtp.layers.0.mlp.experts.4.down_proj.weight"]

    def test_sequential_fc1_bias_expert0(self, bridge):
        name = "mtp.layers.0.transformer_layer.mlp.experts.local_experts.0.linear_fc1.bias"
        result = _map_mtp(bridge, name)
        assert len(result) == 2
        assert "mtp.layers.0.mlp.experts.0.gate_proj.bias" in result
        assert "mtp.layers.0.mlp.experts.0.up_proj.bias" in result

    def test_sequential_fc2_bias_expert3(self, bridge):
        name = "mtp.layers.0.transformer_layer.mlp.experts.local_experts.3.linear_fc2.bias"
        result = _map_mtp(bridge, name)
        assert result == ["mtp.layers.0.mlp.experts.3.down_proj.bias"]


# ---------------------------------------------------------------------------
# _weight_name_mapping_mtp -- unsupported raises
# ---------------------------------------------------------------------------


class TestMTPUnsupported:
    @pytest.fixture
    def bridge(self):
        return _make_qwen3_mock()

    def test_unsupported_mtp_module_raises(self, bridge):
        with pytest.raises(NotImplementedError, match="Unsupported MTP"):
            _map_mtp(bridge, "mtp.layers.0.transformer_layer.unknown.weight")

    def test_unsupported_mtp_no_transformer_layer_raises(self, bridge):
        with pytest.raises(NotImplementedError, match="Unsupported MTP"):
            _map_mtp(bridge, "mtp.layers.0.nonexistent.weight")


# ---------------------------------------------------------------------------
# _weight_to_hf_format (CPU tensors)
# ---------------------------------------------------------------------------


class TestWeightToHfFormat:
    """Test Qwen3NextBridge._weight_to_hf_format with CPU tensors."""

    def test_mtp_fc1_weight_splits_into_gate_and_up(self):
        """MTP path: linear_fc1.weight is chunked into 2 equal halves."""
        bridge = SimpleNamespace(
            _weight_name_mapping_mtp=lambda name: ["gate.weight", "up.weight"],
            _weight_name_mapping_mcore_to_hf=lambda name: ["single.weight"],
            make_vocab_size_divisible_by=None,
        )
        weight = torch.randn(64, 32)  # Will be chunked into 2x (32, 32)
        names, tensors = Qwen3NextBridge._weight_to_hf_format(bridge, "mtp.something.linear_fc1.weight", weight)
        assert len(names) == 2
        assert names == ["gate.weight", "up.weight"]
        assert tensors[0].shape == (32, 32)
        assert tensors[1].shape == (32, 32)
        torch.testing.assert_close(torch.cat(tensors), weight)

    def test_mtp_fc1_bias_splits_into_gate_and_up(self):
        """MTP path: linear_fc1.bias also splits."""
        bridge = SimpleNamespace(
            _weight_name_mapping_mtp=lambda name: ["gate.bias", "up.bias"],
            make_vocab_size_divisible_by=None,
        )
        bias = torch.randn(128)
        names, tensors = Qwen3NextBridge._weight_to_hf_format(bridge, "mtp.layer.linear_fc1.bias", bias)
        assert len(names) == 2
        assert tensors[0].shape == (64,)
        assert tensors[1].shape == (64,)
        torch.testing.assert_close(torch.cat(tensors), bias)

    def test_mtp_non_fc1_returns_same_tensor_for_each_name(self):
        """MTP path: non-fc1 weights replicate the tensor for each mapped name."""
        bridge = SimpleNamespace(
            _weight_name_mapping_mtp=lambda name: ["a.weight", "b.weight", "c.weight"],
            make_vocab_size_divisible_by=None,
        )
        weight = torch.randn(16, 16)
        names, tensors = Qwen3NextBridge._weight_to_hf_format(bridge, "mtp.layers.0.some_other.weight", weight)
        assert len(names) == 3
        assert len(tensors) == 3
        for t in tensors:
            assert t is weight  # same object, not a copy

    def test_non_mtp_single_name_passthrough(self):
        """Non-MTP path with 1 HF name: 1:1 mapping, no transform."""
        bridge = SimpleNamespace(
            _weight_name_mapping_mcore_to_hf=lambda name: ["hf.single.weight"],
            make_vocab_size_divisible_by=None,
        )
        weight = torch.randn(8, 8)
        names, tensors = Qwen3NextBridge._weight_to_hf_format(bridge, "decoder.layers.0.some.weight", weight)
        assert names == ["hf.single.weight"]
        assert len(tensors) == 1
        assert tensors[0] is weight

    def test_non_mtp_fc1_splits_gate_up(self):
        """Non-MTP path: linear_fc1.weight with 2 HF names splits into gate/up."""
        bridge = SimpleNamespace(
            _weight_name_mapping_mcore_to_hf=lambda name: [
                "model.layers.0.mlp.gate_proj.weight",
                "model.layers.0.mlp.up_proj.weight",
            ],
            make_vocab_size_divisible_by=None,
        )
        weight = torch.randn(256, 64)
        names, tensors = Qwen3NextBridge._weight_to_hf_format(bridge, "decoder.layers.0.mlp.linear_fc1.weight", weight)
        assert len(names) == 2
        assert tensors[0].shape == (128, 64)
        assert tensors[1].shape == (128, 64)
        torch.testing.assert_close(torch.cat(tensors), weight)

    def test_non_mtp_linear_qkv_returns_first_name_full_weight(self):
        """Non-MTP path: self_attention.linear_qkv returns first name with full weight (Qwen3Next override)."""
        bridge = SimpleNamespace(
            _weight_name_mapping_mcore_to_hf=lambda name: [
                "model.layers.0.self_attn.q_proj.weight",
                "model.layers.0.self_attn.k_proj.weight",
                "model.layers.0.self_attn.v_proj.weight",
            ],
            make_vocab_size_divisible_by=None,
        )
        weight = torch.randn(384, 128)
        names, tensors = Qwen3NextBridge._weight_to_hf_format(
            bridge, "decoder.layers.0.self_attention.linear_qkv.weight", weight
        )
        # Qwen3Next override: returns only first name with full weight (no QKV split)
        assert len(names) == 1
        assert names == ["model.layers.0.self_attn.q_proj.weight"]
        assert tensors[0] is weight

    def test_vocab_padding_trim_embedding(self):
        """Non-MTP: embedding.word_embeddings.weight trimmed to vocab_size."""
        bridge = SimpleNamespace(
            _weight_name_mapping_mcore_to_hf=lambda name: ["model.embed_tokens.weight"],
            make_vocab_size_divisible_by=128,
            padded_vocab_size=32128,
            vocab_size=32000,
        )
        weight = torch.randn(32128, 512)
        names, tensors = Qwen3NextBridge._weight_to_hf_format(bridge, "embedding.word_embeddings.weight", weight)
        assert names == ["model.embed_tokens.weight"]
        assert tensors[0].shape == (32000, 512)
        torch.testing.assert_close(tensors[0], weight[:32000])

    def test_vocab_padding_trim_output_layer(self):
        """Non-MTP: output_layer.weight trimmed to vocab_size."""
        bridge = SimpleNamespace(
            _weight_name_mapping_mcore_to_hf=lambda name: ["lm_head.weight"],
            make_vocab_size_divisible_by=64,
            padded_vocab_size=32064,
            vocab_size=32000,
        )
        weight = torch.randn(32064, 512)
        names, tensors = Qwen3NextBridge._weight_to_hf_format(bridge, "output_layer.weight", weight)
        assert names == ["lm_head.weight"]
        assert tensors[0].shape == (32000, 512)
        torch.testing.assert_close(tensors[0], weight[:32000])

    def test_vocab_no_trim_when_divisible_by_is_none(self):
        """Non-MTP: no trimming when make_vocab_size_divisible_by is None."""
        bridge = SimpleNamespace(
            _weight_name_mapping_mcore_to_hf=lambda name: ["model.embed_tokens.weight"],
            make_vocab_size_divisible_by=None,
        )
        weight = torch.randn(32128, 512)
        names, tensors = Qwen3NextBridge._weight_to_hf_format(bridge, "embedding.word_embeddings.weight", weight)
        assert tensors[0] is weight  # no slicing


# ---------------------------------------------------------------------------
# _weight_name_mapping_mcore_to_hf dispatch
# ---------------------------------------------------------------------------


class TestWeightNameMappingMcoreToHf:
    """Test the override that dispatches MTP names to _weight_name_mapping_mtp."""

    def test_mtp_name_dispatches_to_mtp_mapping(self):
        """Names containing 'mtp' should be handled by _weight_name_mapping_mtp."""
        sentinel = ["mtp_mapped:mtp.layers.0.enorm.weight"]
        bridge = SimpleNamespace(
            _weight_name_mapping_mtp=lambda name: [f"mtp_mapped:{name}"],
        )
        result = Qwen3NextBridge._weight_name_mapping_mcore_to_hf(bridge, "mtp.layers.0.enorm.weight")
        assert result == sentinel

    def test_non_mtp_name_calls_parent(self):
        """Names without 'mtp' should delegate to the parent class."""
        # We need to mock super()._weight_name_mapping_mcore_to_hf
        # The simplest way: create a real-ish bridge that can call super
        with patch.object(
            Qwen3NextBridge.__bases__[0],
            "_weight_name_mapping_mcore_to_hf",
            return_value=["parent_result"],
        ) as mock_parent:
            bridge = MagicMock(spec=Qwen3NextBridge)
            # Call the unbound method with the mock
            # We need to actually go through the real method, so use the descriptor
            Qwen3NextBridge._weight_name_mapping_mcore_to_hf(bridge, "decoder.layers.0.mlp.weight")
            # Since "mtp" not in name, it should call super()'s version
            # The mock on the parent class means bridge's method should reach it
            # Actually, the method calls super() which resolves to Qwen2MoEBridge
            mock_parent.assert_called_once_with("decoder.layers.0.mlp.weight")
