"""Tests for axon.utils.metrics.data_metrics module."""

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from axon.protocol import DataProto
from axon.utils.metrics.data_metrics import (
    compute_data_metrics,
    compute_response_mask,
    reduce_data_metrics,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_data_batch(
    batch_size=4,
    prompt_len=10,
    response_len=8,
    use_critic=True,
    mask_value=1.0,
):
    """Create a DataProto with all fields needed by ``compute_data_metrics``.

    Keys required by the function:
        - token_level_scores  (batch_size, response_len)
        - token_level_rewards (batch_size, response_len)
        - advantages          (batch_size, response_len)
        - returns             (batch_size, response_len)
        - responses           (batch_size, response_len)
        - attention_mask      (batch_size, seq_len)
        - response_mask       (batch_size, response_len)
        - values              (batch_size, response_len)  -- only when use_critic
    """
    seq_len = prompt_len + response_len
    attn_mask = torch.full((batch_size, seq_len), mask_value)
    # response_mask spans the full sequence: 0 over prompt tokens, mask_value over response tokens.
    resp_mask = torch.zeros(batch_size, seq_len)
    resp_mask[:, prompt_len:] = mask_value

    # advantages / returns / values / token_level_* are full-seq-len because
    # ``compute_data_metrics`` does ``torch.masked_select(advantages, response_mask)``.
    tensors = {
        "input_ids": torch.ones(batch_size, seq_len, dtype=torch.long),
        "token_level_scores": torch.randn(batch_size, seq_len),
        "token_level_rewards": torch.randn(batch_size, seq_len),
        "advantages": torch.randn(batch_size, seq_len),
        "returns": torch.randn(batch_size, seq_len),
        "responses": torch.ones(batch_size, response_len),
        "attention_mask": attn_mask,
        "response_mask": resp_mask,
    }
    if use_critic:
        tensors["values"] = torch.randn(batch_size, seq_len)

    td = TensorDict(tensors, batch_size=[batch_size])
    return DataProto(batch=td)


# ===========================================================================
# compute_response_mask
# ===========================================================================


class TestComputeResponseMask:
    """Tests for ``compute_response_mask``."""

    def test_returns_response_mask_full_seq(self):
        """``compute_response_mask`` returns the stored ``response_mask`` tensor verbatim."""
        batch_size, response_len = 2, 6
        seq_len = 4 + response_len
        attn = torch.ones(batch_size, seq_len)
        resp = torch.zeros(batch_size, seq_len)
        resp[:, -response_len:] = 1

        td = TensorDict(
            {
                "responses": torch.ones(batch_size, response_len),
                "attention_mask": attn,
                "response_mask": resp,
            },
            batch_size=[batch_size],
        )
        batch = DataProto(batch=td)
        mask = compute_response_mask(batch)
        assert mask.shape == (batch_size, seq_len)
        assert torch.all(mask[:, :-response_len] == 0)
        assert torch.all(mask[:, -response_len:] == 1)


# ===========================================================================
# compute_data_metrics
# ===========================================================================


class TestComputeDataMetrics:
    """Tests for ``compute_data_metrics``."""

    def test_all_aborted_raises(self):
        """When all response lengths are zero, should raise ValueError."""
        batch_size, response_len = 2, 4
        batch = _make_data_batch(batch_size=batch_size, response_len=response_len, mask_value=0.0)
        # All attention_mask = 0 -> all response_length = 0 -> all aborted
        with pytest.raises(ValueError, match="All samples are aborted"):
            compute_data_metrics(batch)

    def test_mixed_aborted_and_valid(self):
        """Some samples aborted, some valid."""
        batch_size, response_len = 4, 5
        batch = _make_data_batch(batch_size=batch_size, response_len=response_len)
        # Zero out response mask for samples 2 and 3
        mask = batch.batch["attention_mask"].clone()
        mask[2, -response_len:] = 0
        mask[3, -response_len:] = 0
        batch.batch["attention_mask"] = mask
        resp_mask = batch.batch["response_mask"].clone()
        resp_mask[2, :] = 0
        resp_mask[3, :] = 0
        batch.batch["response_mask"] = resp_mask
        metrics = compute_data_metrics(batch)
        assert metrics["response/aborted_ratio"] == pytest.approx(0.5)

    def test_vf_explained_var_zero_when_random_values(self):
        """Random values should have low explained variance."""
        torch.manual_seed(42)
        batch = _make_data_batch(batch_size=100, response_len=10, prompt_len=10, use_critic=True)
        seq_len = batch.batch["input_ids"].shape[-1]
        # Make returns and values completely independent
        batch.batch["returns"] = torch.randn(100, seq_len)
        batch.batch["values"] = torch.randn(100, seq_len)
        metrics = compute_data_metrics(batch, use_critic=True)
        # Explained var should be near 0 (or negative) for random data
        assert metrics["critic/vf_explained_var"] < 0.5

    def test_is_last_step_and_is_padding_combined(self):
        """Both filters applied together."""
        batch = _make_data_batch(batch_size=4, response_len=5)
        batch.non_tensor_batch = {
            "is_last_step": np.array([True, True, True, False]),
            "is_padding": np.array([False, True, False, False]),
        }
        detail = compute_data_metrics(batch, include_detail=True)
        # Row 0: last_step=T, padding=F -> kept
        # Row 1: last_step=T, padding=T -> removed by padding filter
        # Row 2: last_step=T, padding=F -> kept
        # Row 3: last_step=F -> removed by last_step filter
        assert detail["_count_all"] == 2

    def test_response_length_values_correct(self):
        """All attention_mask = 1, so response_length should equal response_len for every sample."""
        batch_size, response_len = 4, 8
        batch = _make_data_batch(batch_size=batch_size, response_len=response_len)
        metrics = compute_data_metrics(batch)
        assert metrics["response_length/mean"] == float(response_len)
        assert metrics["response_length/max"] == float(response_len)
        assert metrics["response_length/min"] == float(response_len)

    def test_prompt_length_values_correct(self):
        batch_size, prompt_len = 4, 10
        batch = _make_data_batch(batch_size=batch_size, prompt_len=prompt_len)
        metrics = compute_data_metrics(batch)
        assert metrics["prompt_length/mean"] == float(prompt_len)

    def test_clip_ratio_all_clipped(self):
        """When every response is exactly max_seq_length, clip_ratio = 1."""
        # Set the entire sequence as the response (no prompt) so response_length
        # equals max_seq_length = input_ids.shape[-1].
        batch = _make_data_batch(batch_size=4, prompt_len=0, response_len=8)
        batch.batch["response_mask"] = torch.ones_like(batch.batch["response_mask"])
        metrics = compute_data_metrics(batch)
        assert metrics["response_length/clip_ratio"] == 1.0

    def test_score_statistics_match_manual(self):
        """Verify score statistics against manual calculation."""
        batch_size, response_len = 3, 5
        batch = _make_data_batch(batch_size=batch_size, response_len=response_len)
        # Set deterministic scores
        scores = torch.zeros(batch_size, response_len)
        scores[0, -1] = 1.0  # sequence score = 1
        scores[1, -1] = 2.0  # sequence score = 2
        scores[2, -1] = 3.0  # sequence score = 3
        batch.batch["token_level_scores"] = scores

        metrics = compute_data_metrics(batch)
        assert abs(metrics["critic/score/mean"] - 2.0) < 1e-5
        assert abs(metrics["critic/score/max"] - 3.0) < 1e-5
        assert abs(metrics["critic/score/min"] - 1.0) < 1e-5

    def test_is_last_step_filtering(self):
        """When ``is_last_step`` is provided, only those rows are used."""
        batch_size = 4
        batch = _make_data_batch(batch_size=batch_size, response_len=5)
        # Mark only the first two as last steps
        is_last = np.array([True, True, False, False])
        batch.non_tensor_batch = {"is_last_step": is_last}

        _ = compute_data_metrics(batch)
        # Internal batch should have been filtered to 2 samples
        # We can verify by checking that the counts are correct if detail is on
        detail = compute_data_metrics(batch, include_detail=True)
        assert detail["_count_all"] == 2

    def test_is_padding_filtering(self):
        """When ``is_padding`` is provided, padded rows are excluded."""
        batch_size = 4
        batch = _make_data_batch(batch_size=batch_size, response_len=5)
        is_pad = np.array([False, False, True, True])
        batch.non_tensor_batch = {"is_padding": is_pad}

        detail = compute_data_metrics(batch, include_detail=True)
        assert detail["_count_all"] == 2

    def test_deterministic_scores_with_partial_mask(self):
        """With partial response mask, only masked tokens contribute to advantages."""
        batch = _make_data_batch(batch_size=2, prompt_len=5, response_len=4)
        seq_len = batch.batch["input_ids"].shape[-1]
        # Deterministic advantages over the full sequence; only the response slice gets non-zero values.
        adv = torch.zeros(2, seq_len)
        adv[:, -4:] = torch.tensor([1.0, 2.0, 3.0, 4.0])
        batch.batch["advantages"] = adv
        # Response mask: only first 2 of the 4 response tokens are valid.
        resp_mask = torch.zeros(2, seq_len)
        resp_mask[:, -4:-2] = 1.0
        batch.batch["response_mask"] = resp_mask
        metrics = compute_data_metrics(batch)
        # Only advantages[-4:-2] = [1, 2] contribute (masked_select).
        assert metrics["critic/advantages/mean"] == pytest.approx(1.5)
        assert metrics["critic/advantages/max"] == pytest.approx(2.0)
        assert metrics["critic/advantages/min"] == pytest.approx(1.0)

    def test_clip_ratio_zero_when_all_short(self):
        """When no response fills the max length, clip_ratio = 0."""
        batch = _make_data_batch(batch_size=2, prompt_len=5, response_len=6)
        # Mask out last 2 response tokens for all samples -> response_length = 4 < 6
        mask = batch.batch["attention_mask"].clone()
        mask[:, -2:] = 0
        batch.batch["attention_mask"] = mask
        resp_mask = batch.batch["response_mask"].clone()
        resp_mask[:, -2:] = 0
        batch.batch["response_mask"] = resp_mask
        metrics = compute_data_metrics(batch)
        assert metrics["response_length/clip_ratio"] == 0.0

    def test_vf_explained_var_range(self):
        """Explained variance should typically be between -inf and 1."""
        batch = _make_data_batch(use_critic=True)
        # Make values = returns for perfect prediction
        batch.batch["values"] = batch.batch["returns"].clone()
        metrics = compute_data_metrics(batch, use_critic=True)
        # With perfect predictions, explained variance should be close to 1
        assert metrics["critic/vf_explained_var"] > 0.99


# ===========================================================================
# reduce_data_metrics
# ===========================================================================


class TestReduceDataMetrics:
    """Tests for ``reduce_data_metrics``."""

    def test_heterogeneous_workers_with_and_without_critic(self):
        """One worker has critic metrics, another doesn't."""
        batch1 = _make_data_batch(batch_size=2, use_critic=True)
        batch2 = _make_data_batch(batch_size=2, use_critic=False)
        m1 = compute_data_metrics(batch1, use_critic=True, include_detail=True)
        m2 = compute_data_metrics(batch2, use_critic=False, include_detail=True)
        result = reduce_data_metrics([m1, m2])
        # Should still have score/reward metrics but critic/values may be partial
        assert "critic/score/mean" in result

    def test_two_workers_score_mean_aggregated(self):
        """Score mean should be aggregated across workers using sufficient statistics."""
        # Worker 1: scores sum to 4.0, 2 non-aborted samples
        batch1 = _make_data_batch(batch_size=2, response_len=5)
        scores1 = torch.zeros(2, 5)
        scores1[0, -1] = 1.0
        scores1[1, -1] = 3.0
        batch1.batch["token_level_scores"] = scores1

        # Worker 2: scores sum to 6.0, 2 non-aborted samples
        batch2 = _make_data_batch(batch_size=2, response_len=5)
        scores2 = torch.zeros(2, 5)
        scores2[0, -1] = 2.0
        scores2[1, -1] = 4.0
        batch2.batch["token_level_scores"] = scores2

        m1 = compute_data_metrics(batch1, include_detail=True)
        m2 = compute_data_metrics(batch2, include_detail=True)

        result = reduce_data_metrics([m1, m2])
        # Global mean = (1+3+2+4) / 4 = 2.5
        assert abs(result["critic/score/mean"] - 2.5) < 1e-5

    def test_two_workers_max_is_global_max(self):
        batch1 = _make_data_batch(batch_size=2, response_len=5)
        scores1 = torch.zeros(2, 5)
        scores1[0, -1] = 10.0
        scores1[1, -1] = 1.0
        batch1.batch["token_level_scores"] = scores1

        batch2 = _make_data_batch(batch_size=2, response_len=5)
        scores2 = torch.zeros(2, 5)
        scores2[0, -1] = 5.0
        scores2[1, -1] = 7.0
        batch2.batch["token_level_scores"] = scores2

        m1 = compute_data_metrics(batch1, include_detail=True)
        m2 = compute_data_metrics(batch2, include_detail=True)

        result = reduce_data_metrics([m1, m2])
        assert result["critic/score/max"] == 10.0

    def test_two_workers_min_is_global_min(self):
        batch1 = _make_data_batch(batch_size=2, response_len=5)
        scores1 = torch.zeros(2, 5)
        scores1[0, -1] = 10.0
        scores1[1, -1] = 1.0
        batch1.batch["token_level_scores"] = scores1

        batch2 = _make_data_batch(batch_size=2, response_len=5)
        scores2 = torch.zeros(2, 5)
        scores2[0, -1] = 5.0
        scores2[1, -1] = -3.0
        batch2.batch["token_level_scores"] = scores2

        m1 = compute_data_metrics(batch1, include_detail=True)
        m2 = compute_data_metrics(batch2, include_detail=True)

        result = reduce_data_metrics([m1, m2])
        assert result["critic/score/min"] == -3.0

    def test_response_length_mean_aggregated(self):
        """Response length mean uses total sums / total counts."""
        batch1 = _make_data_batch(batch_size=2, response_len=6)
        batch2 = _make_data_batch(batch_size=2, response_len=6)

        m1 = compute_data_metrics(batch1, include_detail=True)
        m2 = compute_data_metrics(batch2, include_detail=True)

        result = reduce_data_metrics([m1, m2])
        # All masks are 1, so response_length = 6 for all 4 samples
        assert abs(result["response_length/mean"] - 6.0) < 1e-5

    def test_prompt_length_mean_aggregated(self):
        batch1 = _make_data_batch(batch_size=2, prompt_len=10, response_len=5)
        batch2 = _make_data_batch(batch_size=2, prompt_len=10, response_len=5)

        m1 = compute_data_metrics(batch1, include_detail=True)
        m2 = compute_data_metrics(batch2, include_detail=True)

        result = reduce_data_metrics([m1, m2])
        assert abs(result["prompt_length/mean"] - 10.0) < 1e-5

    def test_critic_values_aggregated_when_present(self):
        batch1 = _make_data_batch(batch_size=2, use_critic=True)
        batch2 = _make_data_batch(batch_size=2, use_critic=True)

        m1 = compute_data_metrics(batch1, use_critic=True, include_detail=True)
        m2 = compute_data_metrics(batch2, use_critic=True, include_detail=True)

        result = reduce_data_metrics([m1, m2])
        assert "critic/values/mean" in result
        assert "critic/vf_explained_var" in result

    def test_no_critic_keys_when_absent(self):
        batch1 = _make_data_batch(batch_size=2, use_critic=False)
        batch2 = _make_data_batch(batch_size=2, use_critic=False)

        m1 = compute_data_metrics(batch1, use_critic=False, include_detail=True)
        m2 = compute_data_metrics(batch2, use_critic=False, include_detail=True)

        result = reduce_data_metrics([m1, m2])
        assert "critic/values/mean" not in result
        assert "critic/vf_explained_var" not in result

    def test_three_workers_global_max_correct(self):
        """Global max/min should be the true max/min across all workers."""
        batches = []
        for score_val in [1.0, 5.0, -3.0]:
            b = _make_data_batch(batch_size=1, response_len=3)
            b.batch["token_level_scores"] = torch.tensor([[0.0, 0.0, score_val]])
            batches.append(b)
        metrics_list = [compute_data_metrics(b, include_detail=True) for b in batches]
        result = reduce_data_metrics(metrics_list)
        assert result["critic/score/max"] == 5.0
        assert result["critic/score/min"] == -3.0
        assert abs(result["critic/score/mean"] - 1.0) < 1e-5  # (1+5-3)/3

    def test_aborted_ratio_aggregated(self):
        batch1 = _make_data_batch(batch_size=2)
        batch2 = _make_data_batch(batch_size=2)

        m1 = compute_data_metrics(batch1, include_detail=True)
        m2 = compute_data_metrics(batch2, include_detail=True)

        result = reduce_data_metrics([m1, m2])
        # No aborted samples (all masks = 1)
        assert result["response/aborted_ratio"] == 0.0
