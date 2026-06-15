"""Tests for axon.utils.metrics.repetition module."""

import math

import pytest

from axon.utils.metrics.repetition import compression_ratio, has_repetition

# =============================================================================
# compression_ratio
# =============================================================================


class TestCompressionRatio:
    def test_repetitive_vs_normal_ratio(self):
        normal = "The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs."
        repetitive = "aaa" * 500
        ratio_normal, _ = compression_ratio(normal)
        ratio_repetitive, _ = compression_ratio(repetitive)
        assert ratio_repetitive > ratio_normal

    def test_highly_repetitive_text(self):
        text = "abc" * 1000
        ratio, savings = compression_ratio(text)
        assert ratio > 10.0
        assert savings > 90.0

    def test_empty_string_and_bytes_return_inf(self):
        """Both empty string and empty bytes should return (inf, 0.0)."""
        for data in ("", b""):
            ratio, savings = compression_ratio(data)
            assert math.isinf(ratio)
            assert savings == 0.0

    def test_single_byte_compressed_larger_than_original(self):
        """A single byte compresses to more than 1 byte (header overhead),
        so the ratio should be < 1 and savings should be negative."""
        ratio, savings = compression_ratio("x")
        assert ratio < 1.0
        assert savings < 0.0

    def test_unicode_repetitive_text_high_ratio(self):
        text = "\u4e2d\u6587" * 1000  # Chinese characters repeated
        ratio, savings = compression_ratio(text)
        assert ratio > 10.0

    def test_null_bytes_extremely_high_ratio(self):
        data = b"\x00" * 10000
        ratio, savings = compression_ratio(data)
        assert ratio > 50.0
        assert savings > 95.0

    def test_all_four_algorithms_on_same_input(self):
        """Compare all 4 algorithms on the same repetitive input."""
        text = "hello world " * 200
        results = {}
        for algo in ("zlib", "gzip", "bz2", "lzma"):
            ratio, savings = compression_ratio(text, algorithm=algo)
            results[algo] = ratio
            assert ratio > 1.0
            assert 0.0 < savings <= 100.0

        # zlib and gzip should give similar (but not necessarily identical) ratios
        # All should agree that this text is compressible
        assert all(r > 5.0 for r in results.values())

    def test_algorithm_zlib_is_default(self):
        text = "test data " * 100
        ratio_default, _ = compression_ratio(text)
        ratio_zlib, _ = compression_ratio(text, algorithm="zlib")
        assert ratio_default == ratio_zlib

    def test_invalid_algorithm_raises(self):
        with pytest.raises(ValueError, match="Unsupported algorithm"):
            compression_ratio("test", algorithm="invalid_algo")


# =============================================================================
# has_repetition
# =============================================================================


class TestHasRepetition:
    def test_long_repetitive_text_returns_true(self):
        text = "abc" * 10000  # 30000 chars, well above default 10000
        assert has_repetition(text) is True

    def test_long_non_repetitive_text_returns_false(self):
        import hashlib

        parts = []
        for i in range(2000):
            parts.append(hashlib.sha256(str(i).encode()).hexdigest())
        text = " ".join(parts)
        assert len(text) > 10000
        assert has_repetition(text) is False

    def test_custom_threshold_low(self):
        text = "hello world " * 2000
        assert has_repetition(text, threshold=2.0) is True

    def test_custom_threshold_very_high(self):
        text = "abc" * 10000
        assert has_repetition(text, threshold=1000.0) is False

    def test_boundary_text_exactly_at_tail_chars_returns_false(self):
        """When len(text) == tail_chars, the condition is '>' not '>=',
        so it should return False."""
        text = "a" * 10000
        assert has_repetition(text, tail_chars=10000) is False

    def test_boundary_text_one_more_than_tail_chars_and_repetitive(self):
        """When len(text) == tail_chars + 1 and text is highly repetitive,
        has_repetition should detect it."""
        tail = 500
        text = "a" * (tail + 1)
        # Single-char repeated text is extremely compressible
        assert has_repetition(text, tail_chars=tail, threshold=5.0) is True

    def test_custom_tail_chars_small(self):
        text = "abc" * 200  # 600 chars
        result = has_repetition(text, tail_chars=500)
        assert result is True
