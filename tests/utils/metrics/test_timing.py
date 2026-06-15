"""Tests for axon.utils.metrics.timing module."""

import time

import pytest
import torch
from tensordict import TensorDict

from axon.protocol import DataProto
from axon.utils.metrics.timing import (
    compute_timing_metrics,
    marked_timer,
    reduce_timing_metrics,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_timing_batch(batch_size=4, prompt_len=10, response_len=8):
    """Create a minimal DataProto suitable for timing metric functions.

    ``compute_timing_metrics`` reads ``response_mask`` and ``attention_mask`` to
    derive prompt and response token counts.
    """
    seq_len = prompt_len + response_len
    response_mask = torch.zeros(batch_size, seq_len)
    response_mask[:, prompt_len:] = 1
    td = TensorDict(
        {
            "responses": torch.ones(batch_size, response_len),
            "attention_mask": torch.ones(batch_size, seq_len),
            "response_mask": response_mask,
        },
        batch_size=[batch_size],
    )
    return DataProto(batch=td)


# ===========================================================================
# marked_timer
# ===========================================================================


class TestMarkedTimer:
    """Tests for the ``marked_timer`` context-manager."""

    def test_accumulates_multiple_calls(self):
        timing_raw = {}
        with marked_timer("op", timing_raw):
            time.sleep(0.01)
        first = timing_raw["op"]
        with marked_timer("op", timing_raw):
            time.sleep(0.01)
        assert timing_raw["op"] > first  # accumulated

    def test_nested_timers_independent(self):
        timing_raw = {}
        with marked_timer("outer", timing_raw):
            time.sleep(0.02)
            with marked_timer("inner", timing_raw):
                time.sleep(0.01)
        # Both should be recorded, outer >= inner
        assert timing_raw["outer"] >= timing_raw["inner"]

    def test_exception_inside_block_propagates(self):
        timing_raw = {}
        with pytest.raises(ValueError, match="boom"):
            with marked_timer("err", timing_raw):
                raise ValueError("boom")
        # The _timer generator yield is interrupted by the exception,
        # so the post-yield accumulation code does not run.
        assert "err" not in timing_raw

    def test_multiple_names_independent(self):
        timing_raw = {}
        with marked_timer("a", timing_raw):
            time.sleep(0.01)
        with marked_timer("b", timing_raw):
            time.sleep(0.02)
        # 'b' should be roughly twice 'a'; at minimum both should be > 0
        assert timing_raw["a"] > 0
        assert timing_raw["b"] > 0

    def test_accumulation_is_additive(self):
        """Three invocations with the same name accumulate."""
        timing_raw = {}
        for _ in range(3):
            with marked_timer("acc", timing_raw):
                time.sleep(0.005)
        assert timing_raw["acc"] >= 0.01  # at least ~15ms total


# ===========================================================================
# compute_timing_metrics
# ===========================================================================


class TestComputeTimingMetrics:
    """Tests for ``compute_timing_metrics``."""

    def test_zero_response_tokens_gen_per_token(self):
        """When all response tokens are masked, gen per-token should use max(1, 0)=1."""
        batch_size, prompt_len, response_len = 2, 10, 4
        seq_len = prompt_len + response_len
        mask = torch.ones(batch_size, seq_len)
        mask[:, -response_len:] = 0  # zero out all response tokens
        # response_mask zero everywhere -> 0 response tokens
        td = TensorDict(
            {
                "responses": torch.ones(batch_size, response_len),
                "attention_mask": mask,
                "response_mask": torch.zeros(batch_size, seq_len),
            },
            batch_size=[batch_size],
        )
        batch = DataProto(batch=td)
        metrics = compute_timing_metrics(batch, {"gen": 1.0})
        # With 0 response tokens, falls back to max(1, 0) = 1
        assert metrics["timing_per_token_ms/gen"] == 1000.0  # 1.0 * 1000 / 1

    def test_per_token_ms_gen_uses_response_tokens(self):
        """The 'gen' stage normalises by response tokens only."""
        batch_size, response_len = 4, 8
        batch = _make_timing_batch(batch_size=batch_size, response_len=response_len)
        timing_raw = {"gen": 1.0}
        metrics = compute_timing_metrics(batch, timing_raw)

        num_response_tokens = batch_size * response_len  # all ones in mask
        expected = 1.0 * 1000 / num_response_tokens
        assert abs(metrics["timing_per_token_ms/gen"] - expected) < 1e-6

    def test_per_token_ms_ref_uses_all_tokens(self):
        """Non-gen stages normalise by prompt + response tokens."""
        batch_size, prompt_len, response_len = 4, 10, 8
        batch = _make_timing_batch(batch_size=batch_size, prompt_len=prompt_len, response_len=response_len)
        timing_raw = {"ref": 2.0}
        metrics = compute_timing_metrics(batch, timing_raw)

        num_all_tokens = batch_size * (prompt_len + response_len)
        expected = 2.0 * 1000 / num_all_tokens
        assert abs(metrics["timing_per_token_ms/ref"] - expected) < 1e-6

    def test_unknown_stage_gets_timing_s_but_not_per_token(self):
        """A stage name not in the known set gets ``timing_s/`` but not ``timing_per_token_ms/``."""
        batch = _make_timing_batch()
        timing_raw = {"custom_stage": 0.5}
        metrics = compute_timing_metrics(batch, timing_raw)

        assert "timing_s/custom_stage" in metrics
        assert "timing_per_token_ms/custom_stage" not in metrics

    def test_include_detail_adds_token_counts(self):
        batch = _make_timing_batch(batch_size=2, prompt_len=5, response_len=3)
        timing_raw = {"gen": 1.0}
        metrics = compute_timing_metrics(batch, timing_raw, include_detail=True)

        assert "_num_prompt_tokens" in metrics
        assert "_num_response_tokens" in metrics
        assert "_num_overall_tokens" in metrics
        assert metrics["_num_prompt_tokens"] == 2 * 5
        assert metrics["_num_response_tokens"] == 2 * 3
        assert metrics["_num_overall_tokens"] == 2 * (5 + 3)

    def test_partial_attention_mask(self):
        """When some tokens are masked out, token counts reflect actual ones."""
        batch_size, prompt_len, response_len = 2, 6, 4
        seq_len = prompt_len + response_len
        mask = torch.ones(batch_size, seq_len)
        # Mask out last 2 tokens for both samples
        mask[:, -2:] = 0
        # response_mask spans the full sequence; only the response portion is 1.
        # The last 2 response tokens are masked off here too.
        response_mask = torch.zeros(batch_size, seq_len)
        response_mask[:, prompt_len : seq_len - 2] = 1

        td = TensorDict(
            {
                "responses": torch.ones(batch_size, response_len),
                "attention_mask": mask,
                "response_mask": response_mask,
            },
            batch_size=[batch_size],
        )
        batch = DataProto(batch=td)
        timing_raw = {"gen": 1.0}
        metrics = compute_timing_metrics(batch, timing_raw, include_detail=True)

        # Each sample has 8 active tokens (10 - 2 masked)
        expected_prompt_tokens = batch_size * prompt_len  # prompt part untouched
        expected_response_tokens = batch_size * (response_len - 2)  # 2 masked per sample
        assert metrics["_num_prompt_tokens"] == expected_prompt_tokens
        assert metrics["_num_response_tokens"] == expected_response_tokens

    def test_empty_timing_raw(self):
        batch = _make_timing_batch()
        metrics = compute_timing_metrics(batch, {})
        assert metrics == {} or all(k.startswith("_") for k in metrics)

    def test_multiple_stages_mixed_known_and_unknown(self):
        """Known stages get per-token metrics, unknown ones don't, in same call."""
        batch = _make_timing_batch()
        timing_raw = {"gen": 1.0, "ref": 2.0, "my_custom": 0.5}
        metrics = compute_timing_metrics(batch, timing_raw)
        assert "timing_per_token_ms/gen" in metrics
        assert "timing_per_token_ms/ref" in metrics
        assert "timing_per_token_ms/my_custom" not in metrics
        assert "timing_s/my_custom" in metrics

    def test_all_known_stages(self):
        """All recognised stage names produce per-token metrics."""
        batch = _make_timing_batch()
        known_stages = ["gen", "ref", "values", "adv", "update_critic", "forward_backward"]
        timing_raw = {stage: 1.0 for stage in known_stages}
        metrics = compute_timing_metrics(batch, timing_raw)

        for stage in known_stages:
            assert f"timing_s/{stage}" in metrics
            assert f"timing_per_token_ms/{stage}" in metrics


# ===========================================================================
# reduce_timing_metrics
# ===========================================================================


class TestReduceTimingMetrics:
    """Tests for ``reduce_timing_metrics``."""

    def test_three_workers_token_counts_summed(self):
        """Verify token counts are summed correctly across 3 workers."""
        batches = [_make_timing_batch(batch_size=2, prompt_len=5, response_len=3) for _ in range(3)]
        metrics = [compute_timing_metrics(b, {"gen": 1.0}, include_detail=True) for b in batches]
        result = reduce_timing_metrics(metrics)
        total_resp = 3 * 2 * 3  # 3 workers, 2 samples, 3 response tokens each
        expected = 1.0 * 1000 / total_resp
        assert abs(result["timing_per_token_ms/gen"] - expected) < 1e-6

    def test_two_workers_timing_s_from_first(self):
        """``timing_s/`` values come from the first worker (shared clock)."""
        batch1 = _make_timing_batch(batch_size=2, prompt_len=5, response_len=3)
        batch2 = _make_timing_batch(batch_size=2, prompt_len=5, response_len=3)

        m1 = compute_timing_metrics(batch1, {"gen": 1.0}, include_detail=True)
        m2 = compute_timing_metrics(batch2, {"gen": 2.0}, include_detail=True)

        result = reduce_timing_metrics([m1, m2])
        # Uses first worker's timing
        assert result["timing_s/gen"] == 1.0

    def test_two_workers_per_token_aggregated(self):
        """Per-token metrics should use aggregated token counts from all workers."""
        batch_size, prompt_len, response_len = 2, 5, 3

        batch1 = _make_timing_batch(batch_size=batch_size, prompt_len=prompt_len, response_len=response_len)
        batch2 = _make_timing_batch(batch_size=batch_size, prompt_len=prompt_len, response_len=response_len)

        timing_raw = {"gen": 1.0}
        m1 = compute_timing_metrics(batch1, timing_raw, include_detail=True)
        m2 = compute_timing_metrics(batch2, timing_raw, include_detail=True)

        result = reduce_timing_metrics([m1, m2])

        total_response_tokens = 2 * batch_size * response_len
        expected_per_token = 1.0 * 1000 / total_response_tokens
        assert abs(result["timing_per_token_ms/gen"] - expected_per_token) < 1e-6

    def test_reduces_all_known_stages(self):
        batch = _make_timing_batch()
        stages = ["gen", "ref", "values", "adv", "update_critic", "forward_backward"]
        timing_raw = {s: float(i + 1) for i, s in enumerate(stages)}
        m = compute_timing_metrics(batch, timing_raw, include_detail=True)

        result = reduce_timing_metrics([m])
        for stage in stages:
            assert f"timing_s/{stage}" in result
            assert f"timing_per_token_ms/{stage}" in result

    def test_workers_with_different_batch_sizes(self):
        """Workers can have different batch sizes; token counts should still sum correctly."""
        b1 = _make_timing_batch(batch_size=2, prompt_len=5, response_len=3)
        b2 = _make_timing_batch(batch_size=6, prompt_len=5, response_len=3)
        m1 = compute_timing_metrics(b1, {"gen": 1.0}, include_detail=True)
        m2 = compute_timing_metrics(b2, {"gen": 1.0}, include_detail=True)
        result = reduce_timing_metrics([m1, m2])
        total_resp = (2 + 6) * 3  # 8 samples, 3 response tokens each
        expected = 1.0 * 1000 / total_resp
        assert abs(result["timing_per_token_ms/gen"] - expected) < 1e-6

    def test_unknown_stage_no_per_token_in_reduce(self):
        """Unknown stage names get timing_s but not timing_per_token_ms in reduction."""
        batch = _make_timing_batch()
        timing_raw = {"custom": 1.0}
        m = compute_timing_metrics(batch, timing_raw, include_detail=True)

        result = reduce_timing_metrics([m])
        assert "timing_s/custom" in result
        assert "timing_per_token_ms/custom" not in result
