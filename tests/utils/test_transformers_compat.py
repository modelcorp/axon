"""Tests for axon.utils.transformers_compat module."""

import importlib.metadata

from axon.utils.transformers_compat import (
    flash_attn_supports_top_left_mask,
    is_transformers_version_in_range,
)


class TestIsTransformersVersionInRange:
    def setup_method(self):
        is_transformers_version_in_range.cache_clear()

    def test_no_bounds_always_true(self):
        assert is_transformers_version_in_range() is True

    def test_above_any_real_version(self):
        assert is_transformers_version_in_range(min_version="999.0.0") is False

    def test_below_any_real_version(self):
        assert is_transformers_version_in_range(max_version="0.0.1") is False

    def test_wide_range_covering_current(self):
        assert is_transformers_version_in_range(min_version="0.0.1", max_version="999.0.0") is True

    def test_impossible_range(self):
        assert is_transformers_version_in_range(min_version="999.0.0", max_version="999.0.1") is False

    def test_exact_current_version_inclusive(self):
        current = importlib.metadata.version("transformers")
        assert is_transformers_version_in_range(min_version=current, max_version=current) is True

    def test_min_equals_current(self):
        current = importlib.metadata.version("transformers")
        assert is_transformers_version_in_range(min_version=current) is True

    def test_max_equals_current(self):
        current = importlib.metadata.version("transformers")
        assert is_transformers_version_in_range(max_version=current) is True

    def test_lru_cache_hit(self):
        is_transformers_version_in_range(min_version="0.0.1")
        is_transformers_version_in_range(min_version="0.0.1")
        assert is_transformers_version_in_range.cache_info().hits >= 1


class TestFlashAttnFallback:
    def test_fallback_returns_bool(self):
        """Whether real or fallback, should return a bool."""
        result = flash_attn_supports_top_left_mask()
        assert isinstance(result, bool)
