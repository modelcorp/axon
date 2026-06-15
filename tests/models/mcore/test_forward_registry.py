"""Tests for mcore forward function registry.

Verifies:
- SupportedVLM enum and the derived supported_vlm list as complete sets
- get_mcore_forward_fn, get_mcore_forward_fused_fn, get_mcore_forward_no_padding_fn
  dispatch correctly for VLM vs non-VLM architectures
- Edge cases: prefix-of-VLM names, cross-function differences, consistency

Usage:
    pytest tests/models/mcore/test_forward_registry.py -v
"""

from types import SimpleNamespace

import pytest

from axon.models.mcore.forward.registry import (
    SupportedVLM,
    get_mcore_forward_fn,
    get_mcore_forward_fused_fn,
    get_mcore_forward_no_padding_fn,
    gptmodel_forward_no_padding,
    supported_vlm,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(*architectures: str) -> SimpleNamespace:
    """Create a minimal hf_config mock with the given architectures list."""
    return SimpleNamespace(architectures=list(architectures))


# All three dispatching getter functions under test
_GETTER_FNS = [
    get_mcore_forward_fn,
    get_mcore_forward_fused_fn,
    get_mcore_forward_no_padding_fn,
]

_GETTER_IDS = ["forward", "fused", "no_padding"]


# ---------------------------------------------------------------------------
# SupportedVLM -- one exhaustive check
# ---------------------------------------------------------------------------


class TestSupportedVLM:
    def test_complete_set(self):
        """Enum values, member count, and derived list all match the expected set."""
        expected = {
            "Qwen2_5_VLForConditionalGeneration",
            "Qwen3VLMoeForConditionalGeneration",
            "Qwen3VLForConditionalGeneration",
        }
        assert len(SupportedVLM) == len(expected)
        assert {m.value for m in SupportedVLM} == expected
        assert set(supported_vlm) == expected


# ---------------------------------------------------------------------------
# Parametrized dispatch tests across all getter functions
# ---------------------------------------------------------------------------


class TestGetterDispatch:
    """Unified dispatch tests parametrized over the three getter functions."""

    @pytest.mark.parametrize("getter", _GETTER_FNS, ids=_GETTER_IDS)
    @pytest.mark.parametrize("vlm_arch", [m.value for m in SupportedVLM])
    def test_vlm_arch_returns_callable(self, getter, vlm_arch):
        fn = getter(_make_config(vlm_arch))
        assert callable(fn)

    @pytest.mark.parametrize("getter", _GETTER_FNS, ids=_GETTER_IDS)
    def test_non_vlm_arch_returns_callable(self, getter):
        fn = getter(_make_config("LlamaForCausalLM"))
        assert callable(fn)

    @pytest.mark.parametrize("getter", _GETTER_FNS, ids=_GETTER_IDS)
    def test_vlm_differs_from_non_vlm(self, getter):
        """VLM and non-VLM should produce different closures (except no_padding which is always the same)."""
        vlm_fn = getter(_make_config("Qwen2_5_VLForConditionalGeneration"))
        non_vlm_fn = getter(_make_config("LlamaForCausalLM"))
        if getter is get_mcore_forward_no_padding_fn:
            # no_padding always returns the same function regardless
            assert vlm_fn is non_vlm_fn
        else:
            assert vlm_fn is not non_vlm_fn

    @pytest.mark.parametrize("getter", _GETTER_FNS, ids=_GETTER_IDS)
    def test_multiple_archs_raises(self, getter):
        config = _make_config("LlamaForCausalLM", "GPT2LMHeadModel")
        with pytest.raises(AssertionError):
            getter(config)

    @pytest.mark.parametrize("getter", _GETTER_FNS, ids=_GETTER_IDS)
    def test_empty_archs_raises(self, getter):
        config = _make_config()
        with pytest.raises(AssertionError):
            getter(config)


# ---------------------------------------------------------------------------
# get_mcore_forward_no_padding_fn specifics
# ---------------------------------------------------------------------------


class TestGetMcoreForwardNoPaddingFn:
    def test_returns_gptmodel_forward_no_padding(self):
        """Returns the exact gptmodel_forward_no_padding function object."""
        fn = get_mcore_forward_no_padding_fn(_make_config("LlamaForCausalLM"))
        assert fn is gptmodel_forward_no_padding
        assert callable(fn)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_vlm_prefix_not_exact_match_treated_as_non_vlm(self):
        """An architecture that is a prefix of a VLM name but not exact should be non-VLM."""
        # "Qwen2_5_VL" is a prefix of "Qwen2_5_VLForConditionalGeneration"
        fn_prefix = get_mcore_forward_fn(_make_config("Qwen2_5_VL"))
        fn_vlm = get_mcore_forward_fn(_make_config("Qwen2_5_VLForConditionalGeneration"))
        fn_non_vlm = get_mcore_forward_fn(_make_config("LlamaForCausalLM"))
        # Prefix should behave like non-VLM (different closure from VLM)
        assert fn_prefix is not fn_vlm
        # Both non-VLM configs produce closures with the same __name__ / behavior
        # (model_forward_gen creates a new closure each time, so identity won't match)
        assert fn_prefix.__name__ == fn_non_vlm.__name__

    def test_forward_and_fused_return_different_functions(self):
        """get_mcore_forward_fn and get_mcore_forward_fused_fn return different functions for the same arch."""
        arch = "Qwen2_5_VLForConditionalGeneration"
        fn = get_mcore_forward_fn(_make_config(arch))
        fused_fn = get_mcore_forward_fused_fn(_make_config(arch))
        assert fn is not fused_fn

    def test_forward_and_fused_differ_for_non_vlm_too(self):
        fn = get_mcore_forward_fn(_make_config("LlamaForCausalLM"))
        fused_fn = get_mcore_forward_fused_fn(_make_config("LlamaForCausalLM"))
        assert fn is not fused_fn

    def test_consistent_dispatch_same_config_twice(self):
        """Calling the same getter twice with the same config returns equivalent results."""
        config = _make_config("Qwen3VLForConditionalGeneration")
        fn1 = get_mcore_forward_fn(config)
        fn2 = get_mcore_forward_fn(config)
        # model_forward_gen is called each time so objects differ, but both should be callable
        assert callable(fn1)
        assert callable(fn2)
        # For no_padding, should be identical object
        fn_np1 = get_mcore_forward_no_padding_fn(config)
        fn_np2 = get_mcore_forward_no_padding_fn(config)
        assert fn_np1 is fn_np2
