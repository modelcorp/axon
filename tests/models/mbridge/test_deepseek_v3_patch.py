"""Tests for DeepSeek-V3 monkey patches applied by axon.

Verifies:
- _patched_init removes quantization_config and sets num_nextn_predict_layers=0
- _patched_build_config renames max_position_embeddings -> original_max_position_embeddings
- _patched_get_gptmodel_args handles missing max_position_embeddings,
  rope_scaling fallback, and default 163840 fallback
- apply_deepseek_v3_patch replaces the 3 methods on DeepseekV3Bridge

Usage:
    pytest tests/models/mbridge/test_deepseek_v3_patch.py -v
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import axon.models.mbridge.deepseek_v3 as dsv3_module
from axon.models.mbridge.deepseek_v3 import (
    _patched_build_config,
    _patched_get_gptmodel_args,
    _patched_init,
    apply_deepseek_v3_patch,
)

# ---------------------------------------------------------------------------
# Shared fixture: save/restore _original_init
# ---------------------------------------------------------------------------


@pytest.fixture
def save_restore_original_init():
    """Save and restore _original_init around tests that mock it."""
    original = dsv3_module._original_init
    yield
    dsv3_module._original_init = original


# ---------------------------------------------------------------------------
# _patched_init
# ---------------------------------------------------------------------------


class TestPatchedInit:
    def test_removes_quantization_config_and_zeros_mtp(self, save_restore_original_init):
        """Both quantization_config removal and num_nextn_predict_layers=0 in one call."""
        hf_config = SimpleNamespace(
            quantization_config={"quant_method": "fp8"},
            num_nextn_predict_layers=3,
        )
        init_called = {}

        def fake_init(self, hf_config, *args, **kwargs):
            init_called["hf_config"] = hf_config

        dsv3_module._original_init = fake_init
        bridge = SimpleNamespace()
        _patched_init(bridge, hf_config)
        assert not hasattr(hf_config, "quantization_config")
        assert hf_config.num_nextn_predict_layers == 0
        assert "hf_config" in init_called

    def test_no_quantization_config_is_noop(self, save_restore_original_init):
        """If hf_config has no quantization_config, patching should not fail."""
        hf_config = SimpleNamespace(num_nextn_predict_layers=2)

        dsv3_module._original_init = lambda self, hf_config, *a, **kw: None
        bridge = SimpleNamespace()
        _patched_init(bridge, hf_config)
        assert not hasattr(hf_config, "quantization_config")
        assert hf_config.num_nextn_predict_layers == 0

    def test_no_mtp_layers_is_noop(self, save_restore_original_init):
        """If hf_config has no num_nextn_predict_layers, patching should not fail."""
        hf_config = SimpleNamespace(
            quantization_config={"quant_method": "fp8"},
        )

        dsv3_module._original_init = lambda self, hf_config, *a, **kw: None
        bridge = SimpleNamespace()
        _patched_init(bridge, hf_config)
        assert not hasattr(hf_config, "quantization_config")
        assert not hasattr(hf_config, "num_nextn_predict_layers")


# ---------------------------------------------------------------------------
# _patched_get_gptmodel_args
# ---------------------------------------------------------------------------


class TestPatchedGetGptmodelArgs:
    def test_with_max_position_embeddings(self):
        bridge = SimpleNamespace(
            hf_config=SimpleNamespace(
                max_position_embeddings=4096,
                vocab_size=32000,
                rope_theta=10000.0,
            ),
        )
        result = _patched_get_gptmodel_args(bridge)
        assert result["max_sequence_length"] == 4096
        assert result["vocab_size"] == 32000
        assert result["position_embedding_type"] == "rope"
        assert result["rotary_base"] == 10000.0

    def test_fallback_to_rope_scaling(self):
        bridge = SimpleNamespace(
            hf_config=SimpleNamespace(
                vocab_size=32000,
                rope_theta=10000.0,
                rope_scaling={"original_max_position_embeddings": 8192},
            ),
        )
        result = _patched_get_gptmodel_args(bridge)
        assert result["max_sequence_length"] == 8192

    def test_fallback_to_default_163840(self):
        bridge = SimpleNamespace(
            hf_config=SimpleNamespace(
                vocab_size=32000,
                rope_theta=10000.0,
            ),
        )
        result = _patched_get_gptmodel_args(bridge)
        assert result["max_sequence_length"] == 163840

    def test_rope_scaling_none_falls_to_default(self):
        bridge = SimpleNamespace(
            hf_config=SimpleNamespace(
                vocab_size=32000,
                rope_theta=10000.0,
                rope_scaling=None,
            ),
        )
        result = _patched_get_gptmodel_args(bridge)
        assert result["max_sequence_length"] == 163840

    def test_rope_scaling_without_key_falls_to_default(self):
        bridge = SimpleNamespace(
            hf_config=SimpleNamespace(
                vocab_size=32000,
                rope_theta=10000.0,
                rope_scaling={"type": "dynamic"},
            ),
        )
        result = _patched_get_gptmodel_args(bridge)
        assert result["max_sequence_length"] == 163840

    def test_result_keys(self):
        bridge = SimpleNamespace(
            hf_config=SimpleNamespace(
                max_position_embeddings=4096,
                vocab_size=32000,
                rope_theta=10000.0,
            ),
        )
        result = _patched_get_gptmodel_args(bridge)
        assert set(result.keys()) == {
            "vocab_size",
            "max_sequence_length",
            "position_embedding_type",
            "rotary_base",
        }


# ---------------------------------------------------------------------------
# _patched_build_config
# ---------------------------------------------------------------------------


class TestPatchedBuildConfig:
    def test_renames_max_position_embeddings(self):
        """max_position_embeddings kwarg gets renamed to original_max_position_embeddings
        when calling _build_base_config."""
        captured_kwargs = {}
        build_config_result = SimpleNamespace(some_config="value")

        def fake_build_base_config(**kwargs):
            captured_kwargs.update(kwargs)
            return build_config_result

        def fake_original_build_config(self_arg):
            # Simulate what _original_build_config would do: call _build_base_config
            return self_arg._build_base_config(
                max_position_embeddings=4096,
                hidden_size=1024,
            )

        # Temporarily replace _original_build_config
        original_obc = dsv3_module._original_build_config
        dsv3_module._original_build_config = fake_original_build_config
        try:
            bridge = SimpleNamespace(_build_base_config=fake_build_base_config)
            _patched_build_config(bridge)
            # max_position_embeddings should have been renamed
            assert "max_position_embeddings" not in captured_kwargs
            assert captured_kwargs["original_max_position_embeddings"] == 4096
            # Other kwargs pass through unchanged
            assert captured_kwargs["hidden_size"] == 1024
        finally:
            dsv3_module._original_build_config = original_obc

    def test_non_max_position_embeddings_kwargs_pass_through(self):
        """kwargs without max_position_embeddings should pass through unchanged."""
        captured_kwargs = {}

        def fake_build_base_config(**kwargs):
            captured_kwargs.update(kwargs)
            return SimpleNamespace()

        def fake_original_build_config(self_arg):
            return self_arg._build_base_config(
                hidden_size=2048,
                num_layers=32,
            )

        original_obc = dsv3_module._original_build_config
        dsv3_module._original_build_config = fake_original_build_config
        try:
            bridge = SimpleNamespace(_build_base_config=fake_build_base_config)
            _patched_build_config(bridge)
            assert captured_kwargs == {"hidden_size": 2048, "num_layers": 32}
            assert "original_max_position_embeddings" not in captured_kwargs
        finally:
            dsv3_module._original_build_config = original_obc

    def test_restores_build_base_config_after_call(self):
        """_build_base_config should be restored even if _original_build_config raises."""
        original_bbc = MagicMock()

        def fake_original_build_config(self_arg):
            raise RuntimeError("simulated failure")

        original_obc = dsv3_module._original_build_config
        dsv3_module._original_build_config = fake_original_build_config
        try:
            bridge = SimpleNamespace(_build_base_config=original_bbc)
            with pytest.raises(RuntimeError, match="simulated failure"):
                _patched_build_config(bridge)
            # _build_base_config must be restored to the original
            assert bridge._build_base_config is original_bbc
        finally:
            dsv3_module._original_build_config = original_obc


# ---------------------------------------------------------------------------
# apply_deepseek_v3_patch
# ---------------------------------------------------------------------------


class TestApplyDeepseekV3Patch:
    def test_replaces_three_methods_on_class(self):
        from mbridge.models.deepseek_v3 import DeepseekV3Bridge

        # Save originals
        orig_init = DeepseekV3Bridge.__init__
        orig_build = DeepseekV3Bridge._build_config
        orig_gpt = DeepseekV3Bridge._get_gptmodel_args

        try:
            apply_deepseek_v3_patch()
            assert DeepseekV3Bridge.__init__ is _patched_init
            assert DeepseekV3Bridge._build_config is _patched_build_config
            assert DeepseekV3Bridge._get_gptmodel_args is _patched_get_gptmodel_args
        finally:
            # Restore to not break other tests
            DeepseekV3Bridge.__init__ = orig_init
            DeepseekV3Bridge._build_config = orig_build
            DeepseekV3Bridge._get_gptmodel_args = orig_gpt
