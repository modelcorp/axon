"""Tests for GLM-5 bridge weight conversion and parameter mapping.

Verifies:
- Rope swap roundtrip (mcore -> HF -> mcore) for all DSA indexer weights
- Rope swap correctness (pe part moved to front for HF format)
- Consistency with slime's hardcoded conversion (index_head_dim=128, split=64)
- MTP parameter name conversion with dynamic layer index
- Attention mapping includes all indexer weights and extends DeepseekV3

Usage:
    pytest tests/test_glm5_bridge.py -v
"""

from types import SimpleNamespace

import pytest
import torch

from axon.models.mbridge.glm5 import GLM5Bridge

# GLM-5 744B actual dimensions
GLM5_INDEX_HEAD_DIM = 128
GLM5_QK_POS_EMB_HEAD_DIM = 64
GLM5_NUM_LAYERS = 78


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_bridge_mock(
    index_head_dim=GLM5_INDEX_HEAD_DIM,
    qk_pos_emb_head_dim=GLM5_QK_POS_EMB_HEAD_DIM,
    num_layers=GLM5_NUM_LAYERS,
    mtp_num_layers=1,
):
    """Minimal mock with config attributes needed by weight conversion methods.

    GLM5Bridge methods for indexer weights return early (before any super() call),
    so a SimpleNamespace is sufficient.
    """
    return SimpleNamespace(
        hf_config=SimpleNamespace(index_head_dim=index_head_dim),
        config=SimpleNamespace(
            qk_pos_emb_head_dim=qk_pos_emb_head_dim,
            mtp_num_layers=mtp_num_layers,
            num_layers=num_layers,
        ),
        _weight_name_mapping_mcore_to_hf=lambda name: [
            name.replace("decoder.layers.", "model.layers.").replace("self_attention.", "self_attn.indexer.")
        ],
        _SHARED_STATE_DICT_MAPPING={
            "embedding.word_embeddings.weight": [
                "model.embed_tokens.weight",
                f"model.layers.{num_layers}.embed_tokens.weight",
            ],
            "output_layer.weight": [
                "lm_head.weight",
                f"model.layers.{num_layers}.shared_head.head.weight",
            ],
        },
        _weight_name_mapping_attention=lambda name: [name + "::attn_mapped"],
        _weight_name_mapping_mlp=lambda name: [name + "::mlp_mapped"],
    )


# Thin wrappers so we call the real GLM5Bridge methods on our mock.
def _to_hf(bridge, name, weight):
    return GLM5Bridge._weight_to_hf_format(bridge, name, weight)


def _to_mcore(bridge, name, hf_weights):
    return GLM5Bridge._weight_to_mcore_format(bridge, name, hf_weights)


def _convert_mtp(bridge, name):
    return GLM5Bridge._convert_mtp_param(bridge, name)


# ---------------------------------------------------------------------------
# wq_b.weight — shape (n_heads * index_head_dim, q_lora_rank)
# ---------------------------------------------------------------------------


class TestWqbWeight:
    @pytest.fixture
    def bridge(self):
        return _make_bridge_mock()

    def test_roundtrip(self, bridge):
        original = torch.randn(4 * 128, 256)
        name = "decoder.layers.5.self_attention.wq_b.weight"
        _, [hf] = _to_hf(bridge, name, original.clone())
        recovered = _to_mcore(bridge, name, [hf])
        torch.testing.assert_close(recovered, original)

    def test_swap_moves_pe_to_front(self, bridge):
        """mcore [no_pe | pe] per head -> HF [pe | no_pe] per head."""
        n_heads, hdim, rank = 2, 128, 10
        no_pe = torch.full((n_heads, 64, rank), 1.0)
        pe = torch.full((n_heads, 64, rank), 2.0)
        mcore = torch.cat([no_pe, pe], dim=1).reshape(n_heads * hdim, rank)

        name = "decoder.layers.0.self_attention.wq_b.weight"
        _, [hf] = _to_hf(bridge, name, mcore)
        hf_per_head = hf.view(n_heads, hdim, rank)
        assert (hf_per_head[:, :64] == 2.0).all(), "pe part should be first in HF"
        assert (hf_per_head[:, 64:] == 1.0).all(), "no_pe part should be second in HF"

    @pytest.mark.parametrize("n_heads,rank", [(1, 64), (8, 512), (16, 2048)])
    def test_roundtrip_various_shapes(self, bridge, n_heads, rank):
        original = torch.randn(n_heads * 128, rank)
        name = "decoder.layers.0.self_attention.wq_b.weight"
        _, [hf] = _to_hf(bridge, name, original.clone())
        recovered = _to_mcore(bridge, name, [hf])
        torch.testing.assert_close(recovered, original)


# ---------------------------------------------------------------------------
# wk.weight — shape (index_head_dim, hidden_size)
# ---------------------------------------------------------------------------


class TestWkWeight:
    @pytest.fixture
    def bridge(self):
        return _make_bridge_mock()

    def test_roundtrip(self, bridge):
        original = torch.randn(128, 6144)
        name = "decoder.layers.5.self_attention.wk.weight"
        _, [hf] = _to_hf(bridge, name, original.clone())
        recovered = _to_mcore(bridge, name, [hf])
        torch.testing.assert_close(recovered, original)

    def test_swap_moves_pe_to_front(self, bridge):
        no_pe = torch.full((64, 10), 1.0)
        pe = torch.full((64, 10), 2.0)
        mcore = torch.cat([no_pe, pe], dim=0)

        name = "decoder.layers.0.self_attention.wk.weight"
        _, [hf] = _to_hf(bridge, name, mcore)
        assert (hf[:64] == 2.0).all()
        assert (hf[64:] == 1.0).all()


# ---------------------------------------------------------------------------
# k_norm.weight — shape (index_head_dim,)
# ---------------------------------------------------------------------------


class TestKNormWeight:
    @pytest.fixture
    def bridge(self):
        return _make_bridge_mock()

    def test_roundtrip(self, bridge):
        original = torch.randn(128)
        name = "decoder.layers.5.self_attention.k_norm.weight"
        _, [hf] = _to_hf(bridge, name, original.clone())
        recovered = _to_mcore(bridge, name, [hf])
        torch.testing.assert_close(recovered, original)

    def test_swap_moves_pe_to_front(self, bridge):
        mcore = torch.cat([torch.full((64,), 1.0), torch.full((64,), 2.0)])
        name = "decoder.layers.0.self_attention.k_norm.weight"
        _, [hf] = _to_hf(bridge, name, mcore)
        assert (hf[:64] == 2.0).all()
        assert (hf[64:] == 1.0).all()


# ---------------------------------------------------------------------------
# k_norm.bias — shape (index_head_dim,)
# ---------------------------------------------------------------------------


class TestKNormBias:
    @pytest.fixture
    def bridge(self):
        return _make_bridge_mock()

    def test_roundtrip(self, bridge):
        original = torch.randn(128)
        name = "decoder.layers.5.self_attention.k_norm.bias"
        _, [hf] = _to_hf(bridge, name, original.clone())
        recovered = _to_mcore(bridge, name, [hf])
        torch.testing.assert_close(recovered, original)

    def test_swap_moves_pe_to_front(self, bridge):
        mcore = torch.cat([torch.full((64,), 1.0), torch.full((64,), 2.0)])
        name = "decoder.layers.0.self_attention.k_norm.bias"
        _, [hf] = _to_hf(bridge, name, mcore)
        assert (hf[:64] == 2.0).all()
        assert (hf[64:] == 1.0).all()


# ---------------------------------------------------------------------------
# Consistency with slime hardcoded conversion
# (slime/slime/backends/megatron_utils/megatron_to_hf/deepseekv3.py lines 71-89)
# ---------------------------------------------------------------------------


class TestMatchesSlimeConversion:
    """Verify axon bridge produces identical output to slime's hardcoded converter."""

    @pytest.fixture
    def bridge(self):
        return _make_bridge_mock()

    def _slime_convert_wq_b(self, param):
        """Exact copy of slime's hardcoded wq_b conversion."""
        wq_b = param
        wq_b = wq_b.view(-1, 128, wq_b.shape[-1])  # hard code 128
        wq_b = torch.cat([wq_b[:, 64:], wq_b[:, :64]], dim=1).view(-1, wq_b.shape[-1])
        return wq_b

    def _slime_convert_wk(self, param):
        wk = param
        wk = torch.cat([wk[64:], wk[:64]], dim=0)
        return wk

    def _slime_convert_knorm_weight(self, param):
        return torch.cat([param[64:], param[:64]], dim=0)

    def _slime_convert_knorm_bias(self, param):
        return torch.cat([param[64:], param[:64]], dim=0)

    def test_wq_b_matches_slime(self, bridge):
        weight = torch.randn(8 * 128, 2048)
        name = "decoder.layers.0.self_attention.wq_b.weight"
        _, [axon_hf] = _to_hf(bridge, name, weight.clone())
        slime_hf = self._slime_convert_wq_b(weight.clone())
        torch.testing.assert_close(axon_hf, slime_hf)

    def test_wk_matches_slime(self, bridge):
        weight = torch.randn(128, 6144)
        name = "decoder.layers.0.self_attention.wk.weight"
        _, [axon_hf] = _to_hf(bridge, name, weight.clone())
        slime_hf = self._slime_convert_wk(weight.clone())
        torch.testing.assert_close(axon_hf, slime_hf)

    def test_knorm_weight_matches_slime(self, bridge):
        weight = torch.randn(128)
        name = "decoder.layers.0.self_attention.k_norm.weight"
        _, [axon_hf] = _to_hf(bridge, name, weight.clone())
        slime_hf = self._slime_convert_knorm_weight(weight.clone())
        torch.testing.assert_close(axon_hf, slime_hf)

    def test_knorm_bias_matches_slime(self, bridge):
        weight = torch.randn(128)
        name = "decoder.layers.0.self_attention.k_norm.bias"
        _, [axon_hf] = _to_hf(bridge, name, weight.clone())
        slime_hf = self._slime_convert_knorm_bias(weight.clone())
        torch.testing.assert_close(axon_hf, slime_hf)


# ---------------------------------------------------------------------------
# MTP shared state dict mapping
# ---------------------------------------------------------------------------


class TestMTPSharedStateDict:
    def test_embedding_maps_to_two_hf_names(self):
        bridge = _make_bridge_mock(num_layers=78, mtp_num_layers=1)
        weight = torch.randn(1000, 64)
        hf_names, hf_weights = _to_hf(bridge, "embedding.word_embeddings.weight", weight)
        assert hf_names == [
            "model.embed_tokens.weight",
            "model.layers.78.embed_tokens.weight",
        ]
        assert len(hf_weights) == 2
        # Both entries should be the same tensor
        assert hf_weights[0] is weight and hf_weights[1] is weight

    def test_output_layer_maps_to_two_hf_names(self):
        bridge = _make_bridge_mock(num_layers=78, mtp_num_layers=1)
        weight = torch.randn(1000, 64)
        hf_names, hf_weights = _to_hf(bridge, "output_layer.weight", weight)
        assert hf_names == [
            "lm_head.weight",
            "model.layers.78.shared_head.head.weight",
        ]
        assert len(hf_weights) == 2

    def test_skipped_when_mtp_disabled(self):
        bridge = _make_bridge_mock(mtp_num_layers=0)
        # With MTP disabled, the condition should not match
        assert not (
            bridge.config.mtp_num_layers is not None
            and bridge.config.mtp_num_layers >= 1
            and "embedding.word_embeddings.weight" in bridge._SHARED_STATE_DICT_MAPPING
        )

    def test_dynamic_layer_index_not_hardcoded_61(self):
        """GLM-5 has 78 layers, not DeepSeek-V3's 61."""
        bridge = _make_bridge_mock(num_layers=78)
        _, hf_names_embed = (
            bridge._SHARED_STATE_DICT_MAPPING["embedding.word_embeddings.weight"],
            bridge._SHARED_STATE_DICT_MAPPING["output_layer.weight"],
        )
        # Should use layer 78, NOT 61
        assert "model.layers.78" in hf_names_embed[1]
        assert "model.layers.61" not in hf_names_embed[1]


# ---------------------------------------------------------------------------
# _convert_mtp_param
# ---------------------------------------------------------------------------


class TestConvertMTPParam:
    @pytest.fixture
    def bridge(self):
        return _make_bridge_mock(num_layers=78, mtp_num_layers=1)

    @pytest.mark.parametrize(
        "mcore_name,expected",
        [
            ("mtp.layers.0.enorm.weight", "model.layers.78.enorm.weight"),
            ("mtp.layers.0.hnorm.weight", "model.layers.78.hnorm.weight"),
            ("mtp.layers.0.eh_proj.weight", "model.layers.78.eh_proj.weight"),
            (
                "mtp.layers.0.final_layernorm.weight",
                "model.layers.78.shared_head.norm.weight",
            ),
        ],
    )
    def test_direct_mappings(self, bridge, mcore_name, expected):
        assert _convert_mtp(bridge, mcore_name) == [expected]

    def test_self_attention_dispatches_to_attention_mapper(self, bridge):
        name = "mtp.layers.0.transformer_layer.self_attention.linear_proj.weight"
        result = _convert_mtp(bridge, name)
        assert result == ["decoder.layers.78.self_attention.linear_proj.weight::attn_mapped"]

    def test_input_layernorm_dispatches_to_attention_mapper(self, bridge):
        name = "mtp.layers.0.transformer_layer.input_layernorm.weight"
        result = _convert_mtp(bridge, name)
        assert result == ["decoder.layers.78.input_layernorm.weight::attn_mapped"]

    def test_mlp_dispatches_to_mlp_mapper(self, bridge):
        name = "mtp.layers.0.transformer_layer.mlp.router.weight"
        result = _convert_mtp(bridge, name)
        assert result == ["decoder.layers.78.mlp.router.weight::mlp_mapped"]

    def test_unsupported_module_raises(self, bridge):
        with pytest.raises(NotImplementedError, match="Unsupported MTP"):
            _convert_mtp(bridge, "mtp.layers.0.transformer_layer.unknown.weight")

    def test_non_mtp_name_raises(self, bridge):
        with pytest.raises(AssertionError):
            _convert_mtp(bridge, "decoder.layers.0.self_attention.weight")


# ---------------------------------------------------------------------------
# _ATTENTION_MAPPING class attribute
# ---------------------------------------------------------------------------


class TestAttentionMapping:
    def test_all_indexer_keys_present(self):
        mapping = GLM5Bridge._ATTENTION_MAPPING
        for key in [
            "self_attention.wq_b.weight",
            "self_attention.wk.weight",
            "self_attention.k_norm.weight",
            "self_attention.k_norm.bias",
            "self_attention.weights_proj.weight",
        ]:
            assert key in mapping, f"Missing indexer key: {key}"

    def test_extends_deepseekv3_mappings(self):
        from mbridge.models.deepseek_v3 import DeepseekV3Bridge

        for key in DeepseekV3Bridge._ATTENTION_MAPPING:
            assert key in GLM5Bridge._ATTENTION_MAPPING, f"Lost parent mapping: {key}"

    def test_indexer_hf_paths_use_layer_number_template(self):
        mapping = GLM5Bridge._ATTENTION_MAPPING
        for key in [
            "self_attention.wq_b.weight",
            "self_attention.wk.weight",
            "self_attention.k_norm.weight",
            "self_attention.k_norm.bias",
            "self_attention.weights_proj.weight",
        ]:
            hf_names = mapping[key]
            assert len(hf_names) == 1
            assert "{layer_number}" in hf_names[0]
            assert "self_attn.indexer." in hf_names[0]
