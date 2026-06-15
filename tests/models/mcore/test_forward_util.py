"""Tests for mcore forward utility functions (CPU-only).

Verifies validate_moe_routermap:
- Valid cases: right-padded, no padding, batched, single token, all padding,
  single expert, large batch, non-contiguous mask, max_prompt_length param
- Failure cases: non-padding with all -1, padding without -1, last token not -1,
  first token bad in long sequence, all non-last valid tokens bad
- Edge cases: shape mismatch, boundary conditions

Usage:
    pytest tests/models/mcore/test_forward_util.py -v
"""

import pytest
import torch

from axon.models.mcore.forward.util import validate_moe_routermap

N_EXPERTS = 4


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_valid_routermap_right_padded(
    seq_len: int,
    n_valid: int,
    n_experts: int = N_EXPERTS,
    batch_size: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create a valid (layer_map, attention_mask) pair for right-padded sequences.

    Tokens 0..n_valid-2 are valid non-last tokens (routermap != all -1).
    Token n_valid-1 is the last valid token (routermap = all -1, per vLLM convention).
    Tokens n_valid..seq_len-1 are padding (routermap = all -1).
    """
    layer_map = torch.full((batch_size, seq_len, n_experts), -1, dtype=torch.long)
    attention_mask = torch.zeros(batch_size, seq_len, dtype=torch.long)

    for b in range(batch_size):
        attention_mask[b, :n_valid] = 1
        # Non-last valid tokens get real routermap values
        for t in range(n_valid - 1):
            layer_map[b, t] = torch.arange(n_experts)

    return layer_map, attention_mask


# ---------------------------------------------------------------------------
# Valid cases (should NOT raise)
# ---------------------------------------------------------------------------


class TestValidCases:
    """Cases where validate_moe_routermap should pass without error."""

    def test_valid_simple(self):
        """Right-padded sequence: valid tokens have routermap, last + padding have -1."""
        layer_map, mask = _make_valid_routermap_right_padded(seq_len=6, n_valid=4)
        validate_moe_routermap(layer_map, mask)

    def test_valid_no_padding(self):
        """All tokens valid, last one has -1 routermap."""
        seq_len = 5
        layer_map = torch.full((1, seq_len, N_EXPERTS), -1, dtype=torch.long)
        mask = torch.ones(1, seq_len, dtype=torch.long)
        for t in range(seq_len - 1):
            layer_map[0, t] = torch.arange(N_EXPERTS)
        validate_moe_routermap(layer_map, mask)

    def test_valid_batch(self):
        """Batch of 3 sequences with different valid lengths."""
        batch_size = 3
        seq_len = 8
        layer_map = torch.full((batch_size, seq_len, N_EXPERTS), -1, dtype=torch.long)
        mask = torch.zeros(batch_size, seq_len, dtype=torch.long)

        valid_lengths = [6, 4, 8]
        for b, n_valid in enumerate(valid_lengths):
            mask[b, :n_valid] = 1
            for t in range(n_valid - 1):
                layer_map[b, t] = torch.arange(N_EXPERTS)

        validate_moe_routermap(layer_map, mask)

    def test_empty_sequence(self):
        """All padding (all-zero mask) -- should pass (skipped by the function)."""
        seq_len = 4
        layer_map = torch.full((1, seq_len, N_EXPERTS), -1, dtype=torch.long)
        mask = torch.zeros(1, seq_len, dtype=torch.long)
        validate_moe_routermap(layer_map, mask)

    def test_single_token(self):
        """Only one valid token -- it is both first and last, so routermap should be -1."""
        layer_map = torch.full((1, 1, N_EXPERTS), -1, dtype=torch.long)
        mask = torch.ones(1, 1, dtype=torch.long)
        validate_moe_routermap(layer_map, mask)

    def test_single_valid_among_padding(self):
        """One valid token in a longer sequence (right-padded)."""
        seq_len = 5
        layer_map = torch.full((1, seq_len, N_EXPERTS), -1, dtype=torch.long)
        mask = torch.zeros(1, seq_len, dtype=torch.long)
        mask[0, 0] = 1  # only first token is valid; it's the last valid -> -1
        validate_moe_routermap(layer_map, mask)

    def test_left_padded_valid(self):
        """Left-padded sequence: padding first, then valid tokens."""
        seq_len = 6
        n_valid = 4
        layer_map = torch.full((1, seq_len, N_EXPERTS), -1, dtype=torch.long)
        mask = torch.zeros(1, seq_len, dtype=torch.long)
        start = seq_len - n_valid
        mask[0, start:] = 1
        for t in range(start, seq_len - 1):
            layer_map[0, t] = torch.arange(N_EXPERTS)
        validate_moe_routermap(layer_map, mask)

    def test_n_experts_1(self):
        """Single expert (n_experts=1) should still follow the same validation rules."""
        seq_len = 5
        n_valid = 3
        layer_map, mask = _make_valid_routermap_right_padded(seq_len=seq_len, n_valid=n_valid, n_experts=1)
        validate_moe_routermap(layer_map, mask)

    def test_large_batch_varied_lengths(self):
        """Stress test: batch_size=16 with varied valid lengths per element."""
        batch_size = 16
        seq_len = 32
        layer_map = torch.full((batch_size, seq_len, N_EXPERTS), -1, dtype=torch.long)
        mask = torch.zeros(batch_size, seq_len, dtype=torch.long)

        # Each batch element has a different valid length (1 to 32)
        for b in range(batch_size):
            n_valid = (b * 2 + 1) % seq_len + 1  # 1, 3, 5, ..., wrapping around
            mask[b, :n_valid] = 1
            for t in range(n_valid - 1):
                layer_map[b, t] = torch.arange(N_EXPERTS)

        validate_moe_routermap(layer_map, mask)

    def test_non_contiguous_mask(self):
        """Non-contiguous mask [1, 0, 1, 1, 0]: valid tokens scattered among padding.

        Positions 0, 2, 3 are valid.  Last valid = position 3.
        Positions 0 and 2 need real routermap. Position 3 (last valid) needs -1.
        """
        seq_len = 5
        layer_map = torch.full((1, seq_len, N_EXPERTS), -1, dtype=torch.long)
        mask = torch.tensor([[1, 0, 1, 1, 0]], dtype=torch.long)
        # Token 0 and 2 are non-last valid -> need real routermap
        layer_map[0, 0] = torch.arange(N_EXPERTS)
        layer_map[0, 2] = torch.arange(N_EXPERTS)
        # Token 3 is last valid -> -1 (already set)
        # Tokens 1, 4 are padding -> -1 (already set)
        validate_moe_routermap(layer_map, mask)

    def test_max_prompt_length_does_not_affect_validation(self):
        """max_prompt_length is only for debug output; validation should pass regardless."""
        layer_map, mask = _make_valid_routermap_right_padded(seq_len=6, n_valid=4)
        # With debug=False
        validate_moe_routermap(layer_map, mask, debug=False, max_prompt_length=3)
        # With debug=True
        validate_moe_routermap(layer_map, mask, debug=True, max_prompt_length=3)

    def test_max_prompt_length_invalid_input_still_raises(self):
        """max_prompt_length should not suppress errors."""
        layer_map, mask = _make_valid_routermap_right_padded(seq_len=6, n_valid=4)
        layer_map[0, 5] = torch.arange(N_EXPERTS)  # corrupt padding
        with pytest.raises(AssertionError, match="padding tokens"):
            validate_moe_routermap(layer_map, mask, debug=True, max_prompt_length=3)

    def test_all_tokens_same_value(self):
        """All non-last valid tokens have the same routermap value (not -1)."""
        seq_len = 6
        n_valid = 4
        layer_map = torch.full((1, seq_len, N_EXPERTS), -1, dtype=torch.long)
        mask = torch.zeros(1, seq_len, dtype=torch.long)
        mask[0, :n_valid] = 1
        # All non-last valid tokens map to expert 2 for every expert slot
        for t in range(n_valid - 1):
            layer_map[0, t] = torch.full((N_EXPERTS,), 2, dtype=torch.long)
        validate_moe_routermap(layer_map, mask)

    def test_exactly_2_valid_tokens(self):
        """Exactly 2 valid tokens: token 0 needs real routermap, token 1 (last) needs -1."""
        seq_len = 5
        layer_map = torch.full((1, seq_len, N_EXPERTS), -1, dtype=torch.long)
        mask = torch.zeros(1, seq_len, dtype=torch.long)
        mask[0, :2] = 1
        layer_map[0, 0] = torch.arange(N_EXPERTS)  # non-last valid
        # Token 1 is last valid -> -1 (already set)
        validate_moe_routermap(layer_map, mask)

    def test_boundary_last_valid_at_position_0(self):
        """Single valid token at position 0 with rest padding.

        Only one valid token, so it is both first and last -> routermap = all -1.
        """
        seq_len = 8
        layer_map = torch.full((1, seq_len, N_EXPERTS), -1, dtype=torch.long)
        mask = torch.zeros(1, seq_len, dtype=torch.long)
        mask[0, 0] = 1
        validate_moe_routermap(layer_map, mask)

    def test_boundary_last_valid_at_end(self):
        """All tokens valid -- last valid token is at the final position."""
        seq_len = 6
        layer_map = torch.full((1, seq_len, N_EXPERTS), -1, dtype=torch.long)
        mask = torch.ones(1, seq_len, dtype=torch.long)
        for t in range(seq_len - 1):
            layer_map[0, t] = torch.arange(N_EXPERTS)
        # Last position (seq_len-1) is last valid -> -1 (already set)
        validate_moe_routermap(layer_map, mask)


# ---------------------------------------------------------------------------
# Failure cases (should raise AssertionError)
# ---------------------------------------------------------------------------


class TestFailureCases:
    """Cases where validate_moe_routermap should raise AssertionError."""

    def test_fail_non_padding_all_neg1(self):
        """A non-padding, non-last token has all -1 routermap -> fail."""
        seq_len = 5
        layer_map = torch.full((1, seq_len, N_EXPERTS), -1, dtype=torch.long)
        mask = torch.ones(1, seq_len, dtype=torch.long)
        mask[0, -1] = 0  # last position is padding, so position 3 is last valid

        layer_map[0, 0] = torch.arange(N_EXPERTS)
        # token 1 left as -1 -- non-padding, non-last with all -1
        layer_map[0, 2] = torch.arange(N_EXPERTS)

        with pytest.raises(AssertionError, match="non-padding tokens"):
            validate_moe_routermap(layer_map, mask)

    def test_fail_padding_not_neg1(self):
        """A padding token has non-(-1) routermap values -> fail."""
        layer_map, mask = _make_valid_routermap_right_padded(seq_len=6, n_valid=4)
        layer_map[0, 5] = torch.arange(N_EXPERTS)

        with pytest.raises(AssertionError, match="padding tokens"):
            validate_moe_routermap(layer_map, mask)

    def test_fail_last_token_not_neg1(self):
        """Last non-padding token has non-(-1) routermap -> fail."""
        seq_len = 4
        layer_map = torch.full((1, seq_len, N_EXPERTS), -1, dtype=torch.long)
        mask = torch.ones(1, seq_len, dtype=torch.long)

        for t in range(seq_len):
            layer_map[0, t] = torch.arange(N_EXPERTS)

        with pytest.raises(AssertionError, match="last non-padding token"):
            validate_moe_routermap(layer_map, mask)

    def test_fail_batch_second_sequence(self):
        """Failure in the second sequence of a batch is still caught."""
        batch_size = 2
        seq_len = 4
        layer_map = torch.full((batch_size, seq_len, N_EXPERTS), -1, dtype=torch.long)
        mask = torch.ones(batch_size, seq_len, dtype=torch.long)

        # First sequence is valid
        for t in range(seq_len - 1):
            layer_map[0, t] = torch.arange(N_EXPERTS)

        # Second sequence: token 0 is non-padding, non-last, but left as all -1 (BAD)
        layer_map[1, 1] = torch.arange(N_EXPERTS)
        layer_map[1, 2] = torch.arange(N_EXPERTS)

        with pytest.raises(AssertionError, match="non-padding tokens"):
            validate_moe_routermap(layer_map, mask)

    def test_fail_only_first_token_bad_in_long_sequence(self):
        """Only the first token is invalid in a long sequence.

        Verify the error message mentions the correct index (0).
        """
        seq_len = 20
        layer_map = torch.full((1, seq_len, N_EXPERTS), -1, dtype=torch.long)
        mask = torch.ones(1, seq_len, dtype=torch.long)

        # All non-last tokens get real values EXCEPT token 0
        for t in range(1, seq_len - 1):
            layer_map[0, t] = torch.arange(N_EXPERTS)
        # Token 0 stays as -1 -> BAD (non-padding, non-last)

        with pytest.raises(AssertionError, match="First bad: 0"):
            validate_moe_routermap(layer_map, mask)

    def test_fail_all_non_last_valid_tokens_bad(self):
        """All non-last valid tokens have all -1 routermap.

        Error message should mention the count of bad tokens.
        """
        seq_len = 10
        n_valid = 8
        layer_map = torch.full((1, seq_len, N_EXPERTS), -1, dtype=torch.long)
        mask = torch.zeros(1, seq_len, dtype=torch.long)
        mask[0, :n_valid] = 1
        # Leave ALL tokens as -1 (none get real routermap)
        # n_valid-1 = 7 non-last valid tokens are bad

        with pytest.raises(AssertionError, match="7 non-padding tokens"):
            validate_moe_routermap(layer_map, mask)


# ---------------------------------------------------------------------------
# Shape / dimension mismatch
# ---------------------------------------------------------------------------


class TestShapeMismatch:
    """Shape mismatches between layer_map and attention_mask should raise."""

    def test_batch_size_mismatch(self):
        layer_map = torch.full((2, 4, N_EXPERTS), -1, dtype=torch.long)
        mask = torch.ones(3, 4, dtype=torch.long)
        with pytest.raises(AssertionError, match="batch size mismatch"):
            validate_moe_routermap(layer_map, mask)

    def test_seq_len_mismatch(self):
        layer_map = torch.full((1, 4, N_EXPERTS), -1, dtype=torch.long)
        mask = torch.ones(1, 6, dtype=torch.long)
        with pytest.raises(AssertionError, match="seq_len mismatch"):
            validate_moe_routermap(layer_map, mask)
