"""
Thorough tests for axon.utils.torch.ops — tensor operations used throughout training.
"""

import pytest
import torch
import torch.nn.functional as F

from axon.utils.torch.ops import (
    entropy_from_logits,
    entropy_from_logits_with_chunking,
    get_response_mask,
    get_unpad_data,
    log_probs_from_logits_response,
    logprobs_from_logits_v2,
    masked_mean,
    masked_sum,
    masked_var,
    masked_whiten,
    pad_from_left,
    pad_sequence_to_length,
)

# ═══════════════════════════════════════════════════════════════════
#  logprobs_from_logits_v2
# ═══════════════════════════════════════════════════════════════════


class TestLogprobsFromLogitsV2:
    def test_basic_shape(self):
        logits = torch.randn(2, 5, 100)  # (batch, seq, vocab)
        labels = torch.randint(0, 100, (2, 5))
        result = logprobs_from_logits_v2(logits, labels)
        assert result.shape == (2, 5)

    def test_matches_log_softmax_gather(self):
        """Verify against the textbook formula: log_softmax + gather."""
        logits = torch.randn(3, 4, 50, dtype=torch.float32)
        labels = torch.randint(0, 50, (3, 4))
        result = logprobs_from_logits_v2(logits, labels)
        expected = F.log_softmax(logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        assert torch.allclose(result, expected, atol=1e-5)

    def test_uniform_distribution(self):
        """For uniform logits, all log-probs should be -log(vocab_size)."""
        vocab = 10
        logits = torch.zeros(1, 1, vocab)
        labels = torch.tensor([[5]])
        result = logprobs_from_logits_v2(logits, labels)
        expected = -torch.log(torch.tensor(float(vocab)))
        assert torch.allclose(result, expected.expand_as(result), atol=1e-5)

    def test_peaked_distribution(self):
        """If one logit is huge, its log-prob should be close to 0."""
        logits = torch.full((1, 1, 100), -100.0)
        logits[0, 0, 42] = 100.0
        labels = torch.tensor([[42]])
        result = logprobs_from_logits_v2(logits, labels)
        assert result.item() > -0.01

    def test_bfloat16_fallback(self):
        """bfloat16 triggers the per-row log_softmax path."""
        logits = torch.randn(2, 3, 50, dtype=torch.bfloat16)
        labels = torch.randint(0, 50, (2, 3))
        result = logprobs_from_logits_v2(logits, labels)
        assert result.shape == (2, 3)
        assert result.dtype == torch.bfloat16
        assert (result <= 1e-2).all()

    def test_float16(self):
        logits = torch.randn(2, 3, 50, dtype=torch.float16)
        labels = torch.randint(0, 50, (2, 3))
        result = logprobs_from_logits_v2(logits, labels)
        assert result.shape == (2, 3)
        assert result.dtype == torch.float16

    def test_float64(self):
        logits = torch.randn(2, 3, 20, dtype=torch.float64)
        labels = torch.randint(0, 20, (2, 3))
        result = logprobs_from_logits_v2(logits, labels)
        expected = F.log_softmax(logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        assert torch.allclose(result, expected, atol=1e-10)

    def test_gradient_flows(self):
        logits = torch.randn(2, 3, 10, requires_grad=True)
        labels = torch.randint(0, 10, (2, 3))
        result = logprobs_from_logits_v2(logits, labels)
        result.sum().backward()
        assert logits.grad is not None
        assert logits.grad.shape == logits.shape


# ═══════════════════════════════════════════════════════════════════
#  entropy_from_logits
# ═══════════════════════════════════════════════════════════════════


class TestEntropyFromLogits:
    def test_basic_shape(self):
        logits = torch.randn(4, 8, 100)
        result = entropy_from_logits(logits)
        assert result.shape == (4, 8)

    def test_uniform_distribution_max_entropy(self):
        """Uniform distribution has max entropy = log(vocab_size)."""
        vocab = 10
        logits = torch.zeros(1, 1, vocab)
        result = entropy_from_logits(logits)
        expected = torch.log(torch.tensor(float(vocab)))
        assert torch.allclose(result, expected.expand_as(result), atol=1e-4)

    def test_peaked_distribution_low_entropy(self):
        """Peaked distribution has near-zero entropy."""
        logits = torch.full((1, 1, 100), -100.0)
        logits[0, 0, 0] = 100.0
        result = entropy_from_logits(logits)
        assert result.item() < 0.01

    def test_matches_scipy_definition(self):
        """Cross-check: H = -sum(p * log(p))."""
        logits = torch.randn(2, 3, 20)
        result = entropy_from_logits(logits)
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        expected = -(probs * log_probs).sum(dim=-1)
        assert torch.allclose(result, expected, atol=1e-4)

    def test_2d_input(self):
        logits = torch.randn(5, 20)
        result = entropy_from_logits(logits)
        assert result.shape == (5,)


# ═══════════════════════════════════════════════════════════════════
#  entropy_from_logits_with_chunking
# ═══════════════════════════════════════════════════════════════════


class TestEntropyWithChunking:
    def test_matches_non_chunked(self):
        logits = torch.randn(10, 100)
        unchunked = entropy_from_logits(logits)
        chunked = entropy_from_logits_with_chunking(logits, chunk_size=3)
        assert torch.allclose(unchunked, chunked, atol=1e-4)


# ═══════════════════════════════════════════════════════════════════
#  masked_sum
# ═══════════════════════════════════════════════════════════════════


class TestMaskedSum:
    def test_basic(self):
        values = torch.tensor([1.0, 2.0, 3.0, 4.0])
        mask = torch.tensor([1.0, 0.0, 1.0, 0.0])
        result = masked_sum(values, mask)
        assert result.item() == pytest.approx(4.0)

    def test_all_masked(self):
        values = torch.tensor([1.0, 2.0, 3.0])
        mask = torch.zeros(3)
        result = masked_sum(values, mask)
        assert result.item() == pytest.approx(0.0)

    def test_no_mask(self):
        values = torch.tensor([1.0, 2.0, 3.0])
        mask = torch.ones(3)
        result = masked_sum(values, mask)
        assert result.item() == pytest.approx(6.0)

    def test_with_axis(self):
        values = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        mask = torch.tensor([[1.0, 0.0], [1.0, 1.0]])
        result = masked_sum(values, mask, axis=1)
        assert torch.allclose(result, torch.tensor([1.0, 7.0]))

    def test_nan_outside_mask_ignored(self):
        values = torch.tensor([1.0, float("nan"), 3.0])
        mask = torch.tensor([1.0, 0.0, 1.0])
        result = masked_sum(values, mask)
        assert result.item() == pytest.approx(4.0)


# ═══════════════════════════════════════════════════════════════════
#  masked_mean
# ═══════════════════════════════════════════════════════════════════


class TestMaskedMean:
    def test_basic(self):
        values = torch.tensor([1.0, 2.0, 3.0, 4.0])
        mask = torch.tensor([1.0, 0.0, 1.0, 0.0])
        result = masked_mean(values, mask)
        assert result.item() == pytest.approx(2.0, abs=1e-6)

    def test_all_ones_mask(self):
        values = torch.tensor([2.0, 4.0, 6.0])
        mask = torch.ones(3)
        result = masked_mean(values, mask)
        assert result.item() == pytest.approx(4.0, abs=1e-6)

    def test_single_element(self):
        values = torch.tensor([5.0, 0.0, 0.0])
        mask = torch.tensor([1.0, 0.0, 0.0])
        result = masked_mean(values, mask)
        assert result.item() == pytest.approx(5.0, abs=1e-6)

    def test_with_axis(self):
        values = torch.tensor([[10.0, 20.0], [30.0, 40.0]])
        mask = torch.ones(2, 2)
        result = masked_mean(values, mask, axis=1)
        assert torch.allclose(result, torch.tensor([15.0, 35.0]), atol=1e-5)

    def test_zero_mask_returns_near_zero(self):
        """With epsilon denominator, zero mask returns near-zero."""
        values = torch.tensor([100.0])
        mask = torch.tensor([0.0])
        result = masked_mean(values, mask)
        # Returns 0 / epsilon ≈ 0
        assert abs(result.item()) < 1.0

    def test_2d_values(self):
        values = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        mask = torch.tensor([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]])
        result = masked_mean(values, mask)
        # selected: 1, 2, 5, 6 -> mean = 3.5
        assert result.item() == pytest.approx(3.5, abs=1e-6)


# ═══════════════════════════════════════════════════════════════════
#  masked_var
# ═══════════════════════════════════════════════════════════════════


class TestMaskedVar:
    def test_basic_biased(self):
        values = torch.tensor([1.0, 2.0, 3.0, 4.0])
        mask = torch.ones(4)
        result = masked_var(values, mask, unbiased=False)
        expected = values.var(unbiased=False)
        assert result.item() == pytest.approx(expected.item(), abs=1e-5)

    def test_basic_unbiased(self):
        values = torch.tensor([1.0, 2.0, 3.0, 4.0])
        mask = torch.ones(4)
        result = masked_var(values, mask, unbiased=True)
        expected = values.var(unbiased=True)
        assert result.item() == pytest.approx(expected.item(), abs=1e-5)

    def test_constant_values_zero_variance(self):
        values = torch.tensor([5.0, 5.0, 5.0, 5.0])
        mask = torch.ones(4)
        result = masked_var(values, mask, unbiased=False)
        assert result.item() == pytest.approx(0.0, abs=1e-6)

    def test_partial_mask(self):
        values = torch.tensor([1.0, 100.0, 3.0, 100.0])
        mask = torch.tensor([1.0, 0.0, 1.0, 0.0])
        result = masked_var(values, mask, unbiased=False)
        # mean of selected = (1+3)/2 = 2, var = ((1-2)^2 + (3-2)^2) / 2 = 1
        assert result.item() == pytest.approx(1.0, abs=1e-5)

    def test_empty_mask_raises(self):
        values = torch.tensor([1.0, 2.0])
        mask = torch.zeros(2)
        with pytest.raises(ValueError, match="At least one element"):
            masked_var(values, mask, unbiased=True)

    def test_single_element_mask_raises(self):
        values = torch.tensor([1.0, 2.0])
        mask = torch.tensor([1.0, 0.0])
        with pytest.raises(ValueError, match="sum of the mask is one"):
            masked_var(values, mask, unbiased=True)


# ═══════════════════════════════════════════════════════════════════
#  masked_whiten
# ═══════════════════════════════════════════════════════════════════


class TestMaskedWhiten:
    def test_shift_mean_true(self):
        values = torch.tensor([1.0, 2.0, 3.0, 4.0])
        mask = torch.ones(4)
        result = masked_whiten(values, mask, shift_mean=True)
        # Result should have mean ≈ 0 and std ≈ 1
        assert masked_mean(result, mask).item() == pytest.approx(0.0, abs=1e-5)

    def test_shift_mean_false(self):
        values = torch.tensor([1.0, 2.0, 3.0, 4.0])
        mask = torch.ones(4)
        result = masked_whiten(values, mask, shift_mean=False)
        original_mean = masked_mean(values, mask).item()
        result_mean = masked_mean(result, mask).item()
        # Mean should be preserved
        assert result_mean == pytest.approx(original_mean, abs=1e-5)

    def test_whitening_normalizes(self):
        values = torch.randn(100)
        mask = torch.ones(100)
        result = masked_whiten(values, mask, shift_mean=True)
        # Variance should be ≈ 1
        var = masked_var(result, mask, unbiased=False)
        assert var.item() == pytest.approx(1.0, abs=0.2)

    def test_partial_mask(self):
        values = torch.tensor([10.0, 999.0, 20.0, 999.0])
        mask = torch.tensor([1.0, 0.0, 1.0, 0.0])
        result = masked_whiten(values, mask, shift_mean=True)
        # Should not raise, and masked values should be whitened
        assert result.shape == values.shape


# ═══════════════════════════════════════════════════════════════════
#  get_response_mask
# ═══════════════════════════════════════════════════════════════════


class TestGetResponseMask:
    def test_basic_single_eos(self):
        """Example from docstring."""
        response_id = torch.tensor(
            [
                [20, 10, 34, 1, 0, 0, 0],
                [78, 0, 76, 2, 1, 0, 0],
                [23, 98, 1, 0, 0, 0, 0],
                [33, 3, 98, 45, 1, 0, 0],
            ]
        )
        mask = get_response_mask(response_id, eos_token=1)
        expected = torch.tensor(
            [
                [1, 1, 1, 1, 0, 0, 0],
                [1, 1, 1, 1, 1, 0, 0],
                [1, 1, 1, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 0, 0],
            ],
            dtype=torch.int64,
        )
        assert torch.equal(mask, expected)

    def test_multiple_eos_tokens(self):
        """Example from docstring with eos_token=[1, 2]."""
        response_id = torch.tensor(
            [
                [20, 10, 34, 1, 0, 0, 0],
                [78, 0, 76, 2, 1, 0, 0],
                [23, 98, 1, 0, 0, 0, 0],
                [33, 3, 98, 45, 1, 0, 0],
            ]
        )
        mask = get_response_mask(response_id, eos_token=[1, 2])
        expected = torch.tensor(
            [
                [1, 1, 1, 1, 0, 0, 0],
                [1, 1, 1, 1, 0, 0, 0],
                [1, 1, 1, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 0, 0],
            ],
            dtype=torch.int64,
        )
        assert torch.equal(mask, expected)

    def test_no_eos_all_ones(self):
        response_id = torch.tensor([[5, 6, 7, 8]])
        mask = get_response_mask(response_id, eos_token=1)
        expected = torch.ones(1, 4, dtype=torch.int64)
        assert torch.equal(mask, expected)

    def test_eos_at_start(self):
        response_id = torch.tensor([[1, 5, 6, 7]])
        mask = get_response_mask(response_id, eos_token=1)
        expected = torch.tensor([[1, 0, 0, 0]], dtype=torch.int64)
        assert torch.equal(mask, expected)

    def test_all_eos(self):
        response_id = torch.tensor([[1, 1, 1]])
        mask = get_response_mask(response_id, eos_token=1)
        expected = torch.tensor([[1, 0, 0]], dtype=torch.int64)
        assert torch.equal(mask, expected)

    def test_custom_dtype(self):
        response_id = torch.tensor([[5, 1, 0]])
        mask = get_response_mask(response_id, eos_token=1, dtype=torch.float32)
        assert mask.dtype == torch.float32


# ═══════════════════════════════════════════════════════════════════
#  pad_from_left
# ═══════════════════════════════════════════════════════════════════


class TestPadFromLeft:
    def test_basic_padding(self):
        input_ids = [[1, 2, 3], [4, 5]]
        result = pad_from_left(input_ids, pad_token_id=0)
        assert len(result) == 2
        assert len(result[0]) == len(result[1])  # same length
        assert result[0] == [1, 2, 3]
        assert result[1] == [0, 4, 5]

    def test_already_same_length(self):
        input_ids = [[1, 2], [3, 4]]
        result = pad_from_left(input_ids, pad_token_id=0)
        assert result[0] == [1, 2]
        assert result[1] == [3, 4]

    def test_single_sequence_adds_random_padding(self):
        input_ids = [[1, 2, 3]]
        result = pad_from_left(input_ids, pad_token_id=0)
        # Single sequence gets extra random padding (1-100)
        assert len(result[0]) > 3
        assert result[0][-3:] == [1, 2, 3]
        assert all(x == 0 for x in result[0][:-3])

    def test_custom_pad_token(self):
        input_ids = [[1, 2], [3]]
        result = pad_from_left(input_ids, pad_token_id=99)
        assert result[1][0] == 99


# ═══════════════════════════════════════════════════════════════════
#  pad_sequence_to_length
# ═══════════════════════════════════════════════════════════════════


class TestPadSequenceToLength:
    def test_right_pad(self):
        t = torch.tensor([[1, 2, 3]])
        result = pad_sequence_to_length(t, max_seq_len=5, pad_token_id=0)
        assert torch.equal(result, torch.tensor([[1, 2, 3, 0, 0]]))

    def test_left_pad(self):
        t = torch.tensor([[1, 2, 3]])
        result = pad_sequence_to_length(t, max_seq_len=5, pad_token_id=0, left_pad=True)
        assert torch.equal(result, torch.tensor([[0, 0, 1, 2, 3]]))

    def test_no_padding_needed(self):
        t = torch.tensor([[1, 2, 3]])
        result = pad_sequence_to_length(t, max_seq_len=3, pad_token_id=0)
        assert torch.equal(result, t)

    def test_already_longer(self):
        t = torch.tensor([[1, 2, 3, 4, 5]])
        result = pad_sequence_to_length(t, max_seq_len=3, pad_token_id=0)
        assert torch.equal(result, t)  # no truncation

    def test_batch(self):
        t = torch.tensor([[1, 2], [3, 4]])
        result = pad_sequence_to_length(t, max_seq_len=4, pad_token_id=-1)
        expected = torch.tensor([[1, 2, -1, -1], [3, 4, -1, -1]])
        assert torch.equal(result, expected)


# ═══════════════════════════════════════════════════════════════════
#  log_probs_from_logits_response
# ═══════════════════════════════════════════════════════════════════


class TestLogProbsFromLogitsResponse:
    """log_probs_from_logits_response dispatches through logprobs_from_logits,
    which may use flash-attn cross-entropy (CUDA-only) when available.
    We monkeypatch to force the pure-torch v2 path so tests run on CPU."""

    @pytest.fixture(autouse=True)
    def _force_v2_path(self, monkeypatch):
        import axon.utils.torch.ops as ops_mod

        monkeypatch.setattr(ops_mod, "FLAH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE", False)
        monkeypatch.setattr(ops_mod, "NPU_CROSS_ENTROPY_LOSS_AVAILABLE", False)

    def test_basic_shape(self):
        batch_size, seq_len, vocab = 2, 10, 50
        input_ids = torch.randint(0, vocab, (batch_size, seq_len))
        logits = torch.randn(batch_size, seq_len, vocab)
        response_length = 4
        result = log_probs_from_logits_response(input_ids, logits, response_length)
        assert result.shape == (batch_size, response_length)

    def test_values_are_log_probs(self):
        batch_size, seq_len, vocab = 1, 6, 20
        input_ids = torch.randint(0, vocab, (batch_size, seq_len))
        logits = torch.randn(batch_size, seq_len, vocab)
        result = log_probs_from_logits_response(input_ids, logits, response_length=3)
        # Log probs should be <= 0
        assert (result <= 1e-5).all()

    def test_full_response(self):
        """When response_length == seq_len - 1, covers everything except first token."""
        batch_size, seq_len, vocab = 1, 5, 10
        input_ids = torch.randint(0, vocab, (batch_size, seq_len))
        logits = torch.randn(batch_size, seq_len, vocab)
        result = log_probs_from_logits_response(input_ids, logits, response_length=seq_len - 1)
        assert result.shape == (batch_size, seq_len - 1)

    def test_correct_logprobs_computed(self):
        """Verify the response log-probs match a manual computation."""
        vocab = 10
        input_ids = torch.tensor([[0, 1, 2, 3, 4]])
        logits = torch.randn(1, 5, vocab)
        response_length = 3
        result = log_probs_from_logits_response(input_ids, logits, response_length)
        # Manual: response_logits = logits[:, 1:4], response = input_ids[:, 2:5]
        expected = logprobs_from_logits_v2(logits[:, 1:4], input_ids[:, 2:5])
        assert torch.allclose(result, expected, atol=1e-5)


# ═══════════════════════════════════════════════════════════════════
#  get_unpad_data
# ═══════════════════════════════════════════════════════════════════


class TestGetUnpadData:
    def test_basic(self):
        attention_mask = torch.tensor(
            [
                [1, 1, 1, 0, 0],
                [1, 1, 0, 0, 0],
            ]
        )
        indices, cu_seqlens, max_seqlen = get_unpad_data(attention_mask)
        assert max_seqlen == 3
        assert cu_seqlens.shape[0] == 3  # batch_size + 1
        assert torch.equal(cu_seqlens, torch.tensor([0, 3, 5], dtype=torch.int32))
        assert len(indices) == 5  # total non-zero elements

    def test_all_ones(self):
        attention_mask = torch.ones(2, 4, dtype=torch.int32)
        indices, cu_seqlens, max_seqlen = get_unpad_data(attention_mask)
        assert max_seqlen == 4
        assert len(indices) == 8

    def test_single_sequence(self):
        attention_mask = torch.tensor([[1, 1, 1, 0, 0, 0]])
        indices, cu_seqlens, max_seqlen = get_unpad_data(attention_mask)
        assert max_seqlen == 3
        assert torch.equal(cu_seqlens, torch.tensor([0, 3], dtype=torch.int32))
        assert len(indices) == 3

    def test_indices_match_nonzero(self):
        attention_mask = torch.tensor(
            [
                [1, 0, 1, 0],
                [1, 1, 1, 1],
            ]
        )
        indices, cu_seqlens, max_seqlen = get_unpad_data(attention_mask)
        expected_indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
        assert torch.equal(indices, expected_indices)

    def test_variable_lengths(self):
        attention_mask = torch.tensor(
            [
                [1, 1, 1, 1, 1],
                [1, 0, 0, 0, 0],
                [1, 1, 1, 0, 0],
            ]
        )
        indices, cu_seqlens, max_seqlen = get_unpad_data(attention_mask)
        assert max_seqlen == 5
        assert torch.equal(cu_seqlens, torch.tensor([0, 5, 6, 9], dtype=torch.int32))


# ═══════════════════════════════════════════════════════════════════
#  Integration tests — cross-function consistency
# ═══════════════════════════════════════════════════════════════════


class TestOpsIntegration:
    def test_entropy_gradient(self):
        """Entropy should support gradient computation."""
        logits = torch.randn(2, 5, 20, requires_grad=True)
        entropy = entropy_from_logits(logits)
        entropy.sum().backward()
        assert logits.grad is not None

    def test_response_mask_and_pad_consistency(self):
        """Tokens after EOS should be maskable and paddable."""
        response_ids = torch.tensor([[5, 10, 1, 0, 0]])  # EOS=1 at position 2
        mask = get_response_mask(response_ids, eos_token=1)
        padded = pad_sequence_to_length(response_ids, max_seq_len=7, pad_token_id=0)
        assert padded.shape == (1, 7)
        # Original mask positions are preserved
        assert mask[0, :3].sum() == 3
        assert mask[0, 3:].sum() == 0


# ═══════════════════════════════════════════════════════════════════
#  Additional edge cases
# ═══════════════════════════════════════════════════════════════════


class TestLogprobsEdgeCases:
    def test_large_vocab(self):
        """Test with a large vocab (common in LLMs, e.g. 128k)."""
        logits = torch.randn(1, 2, 50000)
        labels = torch.randint(0, 50000, (1, 2))
        result = logprobs_from_logits_v2(logits, labels)
        assert result.shape == (1, 2)
        assert (result <= 1e-5).all()

    def test_very_negative_logits_numerical_stability(self):
        """All logits very negative — should not produce NaN/Inf."""
        logits = torch.full((2, 3, 50), -1000.0)
        logits[:, :, 0] = -999.0  # slightly less negative
        labels = torch.zeros(2, 3, dtype=torch.long)
        result = logprobs_from_logits_v2(logits, labels)
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()

    def test_very_large_logits_numerical_stability(self):
        """Very large logits — should not produce NaN/Inf."""
        logits = torch.full((1, 1, 50), 1000.0)
        logits[0, 0, 10] = 1001.0
        labels = torch.tensor([[10]])
        result = logprobs_from_logits_v2(logits, labels)
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()


class TestEntropyEdgeCases:
    def test_single_class_zero_entropy(self):
        """Single-element vocab has zero entropy."""
        logits = torch.tensor([[5.0]])
        result = entropy_from_logits(logits)
        assert result.item() == pytest.approx(0.0, abs=1e-5)

    def test_binary_balanced(self):
        """50/50 binary entropy = log(2)."""
        logits = torch.tensor([[0.0, 0.0]])
        result = entropy_from_logits(logits)
        assert result.item() == pytest.approx(torch.log(torch.tensor(2.0)).item(), abs=1e-4)

    def test_chunking_size_one(self):
        """Process one element at a time."""
        logits = torch.randn(5, 20)
        unchunked = entropy_from_logits(logits)
        chunked = entropy_from_logits_with_chunking(logits, chunk_size=1)
        assert torch.allclose(unchunked, chunked, atol=1e-4)


class TestMaskedOpsEdgeCases:
    def test_masked_sum_3d(self):
        values = torch.randn(2, 3, 4)
        mask = torch.ones(2, 3, 4)
        mask[0, :, :2] = 0
        result = masked_sum(values, mask)
        expected = (values * mask).sum()
        assert result.item() == pytest.approx(expected.item(), abs=1e-5)

    def test_masked_mean_high_dim_axis(self):
        """Reduce along last axis of 3D tensor."""
        values = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])
        mask = torch.tensor([[[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]]])
        result = masked_mean(values, mask, axis=2)
        # row 0: (1+2)/2=1.5, row 1: (5+6)/2=5.5
        assert torch.allclose(result, torch.tensor([[[1.5], [5.5]]]).squeeze(-1), atol=1e-5)

    def test_masked_whiten_two_elements(self):
        """Whiten with exactly 2 valid elements."""
        values = torch.tensor([0.0, 100.0, 1.0])
        mask = torch.tensor([1.0, 0.0, 1.0])
        result = masked_whiten(values, mask, shift_mean=True)
        # Should succeed without error and produce zero-mean result
        selected = result[mask.bool()]
        assert selected.mean().item() == pytest.approx(0.0, abs=1e-4)


class TestResponseMaskEdgeCases:
    def test_eos_at_every_position(self):
        """Each row has EOS at different positions."""
        response_id = torch.tensor(
            [
                [1, 5, 5, 5],  # EOS at 0
                [5, 1, 5, 5],  # EOS at 1
                [5, 5, 1, 5],  # EOS at 2
                [5, 5, 5, 1],  # EOS at 3
            ]
        )
        mask = get_response_mask(response_id, eos_token=1)
        assert mask.sum(dim=1).tolist() == [1, 2, 3, 4]

    def test_large_eos_token_value(self):
        response_id = torch.tensor([[5, 6, 99999, 0]])
        mask = get_response_mask(response_id, eos_token=99999)
        expected = torch.tensor([[1, 1, 1, 0]], dtype=torch.int64)
        assert torch.equal(mask, expected)

    def test_empty_sequence(self):
        """Zero-length sequences."""
        response_id = torch.zeros(3, 0, dtype=torch.long)
        mask = get_response_mask(response_id, eos_token=1)
        assert mask.shape == (3, 0)


class TestPadEdgeCases:
    def test_pad_from_left_many_sequences(self):
        input_ids = [[1], [1, 2], [1, 2, 3], [1, 2, 3, 4]]
        result = pad_from_left(input_ids, pad_token_id=0)
        assert all(len(r) == 4 for r in result)
        assert result[0] == [0, 0, 0, 1]
        assert result[3] == [1, 2, 3, 4]

    def test_pad_sequence_float_tensor(self):
        t = torch.tensor([[1.5, 2.5]])
        result = pad_sequence_to_length(t, max_seq_len=4, pad_token_id=0)
        assert result.dtype == torch.float32
        assert result.shape == (1, 4)
        assert torch.equal(result, torch.tensor([[1.5, 2.5, 0.0, 0.0]]))

    def test_pad_sequence_3d_tensor(self):
        """Padding should work on the last dim for any shape."""
        t = torch.randn(2, 3, 4)
        result = pad_sequence_to_length(t, max_seq_len=6, pad_token_id=0)
        assert result.shape == (2, 3, 6)
        # Original data preserved
        assert torch.equal(result[:, :, :4], t)
        # Padding is zeros
        assert (result[:, :, 4:] == 0).all()


class TestGetUnpadDataEdgeCases:
    def test_all_zeros(self):
        """All-zero mask means no valid tokens."""
        attention_mask = torch.zeros(2, 4, dtype=torch.int32)
        indices, cu_seqlens, max_seqlen = get_unpad_data(attention_mask)
        assert max_seqlen == 0
        assert len(indices) == 0
        assert torch.equal(cu_seqlens, torch.tensor([0, 0, 0], dtype=torch.int32))

    def test_single_token_per_sequence(self):
        attention_mask = torch.tensor(
            [
                [1, 0, 0, 0],
                [1, 0, 0, 0],
                [1, 0, 0, 0],
            ]
        )
        indices, cu_seqlens, max_seqlen = get_unpad_data(attention_mask)
        assert max_seqlen == 1
        assert torch.equal(cu_seqlens, torch.tensor([0, 1, 2, 3], dtype=torch.int32))

    def test_large_batch(self):
        attention_mask = torch.ones(100, 512, dtype=torch.int32)
        indices, cu_seqlens, max_seqlen = get_unpad_data(attention_mask)
        assert max_seqlen == 512
        assert len(indices) == 100 * 512
        assert cu_seqlens.shape[0] == 101
