"""
Comprehensive tests for axon.utils.rl.utils -- RL training data processing utilities.
"""

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from axon.protocol import DataProto
from axon.utils.rl.utils import (
    compute_pass_metrics,
    compute_reward_statistics,
    filter_zero_advantage_samples,
)


def _make_batch(tensor_dict_data: dict, non_tensor_data: dict = None, batch_size: int = None) -> DataProto:
    """Helper to construct a DataProto for testing."""
    if batch_size is None:
        # Infer from first tensor value
        for v in tensor_dict_data.values():
            batch_size = v.shape[0]
            break
    td = TensorDict(tensor_dict_data, batch_size=[batch_size])
    non_tensor_batch = {}
    if non_tensor_data:
        for k, v in non_tensor_data.items():
            non_tensor_batch[k] = np.array(v, dtype=object)
    return DataProto(batch=td, non_tensor_batch=non_tensor_batch)


# ===================================================================
#  filter_zero_advantage_samples
# ===================================================================


class TestFilterZeroAdvantageSamples:
    def test_some_zeros_filtered(self):
        advantages = torch.tensor([[1.0, 2.0], [0.0, 0.0], [5.0, 6.0]])
        batch = _make_batch({"advantages": advantages})
        result = filter_zero_advantage_samples(batch)
        assert len(result) == 2

    def test_all_zeros_returns_original(self):
        advantages = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
        batch = _make_batch({"advantages": advantages})
        result = filter_zero_advantage_samples(batch)
        # When all are zero, original batch is returned
        assert len(result) == 2

    def test_near_zero_filtered(self):
        # Values below eps (1e-5) should be treated as zero
        advantages = torch.tensor([[1e-7, 1e-8], [1.0, 0.0]])
        batch = _make_batch({"advantages": advantages})
        result = filter_zero_advantage_samples(batch)
        assert len(result) == 1

    def test_large_batch_filtering(self):
        """Performance: filter should work on large batches."""
        advantages = torch.zeros(1000, 10)
        advantages[:500] = 1.0  # first 500 are non-zero
        batch = _make_batch({"advantages": advantages})
        result = filter_zero_advantage_samples(batch)
        assert len(result) == 500

    def test_negative_advantages_kept(self):
        advantages = torch.tensor([[-1.0, -2.0], [0.0, 0.0], [-0.5, 0.0]])
        batch = _make_batch({"advantages": advantages})
        result = filter_zero_advantage_samples(batch)
        assert len(result) == 2

    def test_preserves_other_keys(self):
        advantages = torch.tensor([[1.0, 2.0], [0.0, 0.0], [3.0, 4.0]])
        other = torch.tensor([[10.0], [20.0], [30.0]])
        batch = _make_batch({"advantages": advantages, "other": other})
        result = filter_zero_advantage_samples(batch)
        assert len(result) == 2
        assert "other" in result.batch.keys()

    def test_missing_advantages_raises(self):
        batch = _make_batch({"some_key": torch.tensor([[1.0]])})
        with pytest.raises(AssertionError):
            filter_zero_advantage_samples(batch)

    def test_eps_boundary_exactly_at_threshold(self):
        """Value exactly at eps=1e-5 is NOT > eps, so it's treated as zero and filtered."""
        advantages = torch.tensor([[1e-5, 0.0], [2.0, 0.0]])
        batch = _make_batch({"advantages": advantages})
        result = filter_zero_advantage_samples(batch)
        # The code checks abs > eps (strict), so 1e-5 is NOT > 1e-5 -> first row is zero-advantage
        assert len(result) == 1  # only the row with 2.0 is kept

    def test_mixed_zero_nonzero_in_row(self):
        # Row has some zeros and some non-zeros -> kept (any > eps)
        advantages = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0]])
        batch = _make_batch({"advantages": advantages})
        result = filter_zero_advantage_samples(batch)
        assert len(result) == 1


# ===================================================================
#  compute_reward_statistics
# ===================================================================


class TestComputeRewardStatistics:
    def test_with_padding(self):
        token_level_scores = torch.tensor([[0.0, 1.0], [0.0, 2.0], [0.0, 100.0]])
        batch = _make_batch(
            {"token_level_scores": token_level_scores},
            non_tensor_data={
                "is_padding": [False, False, True],
                "is_last_step": [True, True, True],
            },
        )
        stats = compute_reward_statistics(batch)
        # Padded row (100.0) should be excluded
        assert stats["batch/reward/mean"] == pytest.approx(1.5)
        assert stats["batch/reward/max"] == pytest.approx(2.0)
        assert stats["batch/reward/min"] == pytest.approx(1.0)

    def test_with_is_last_step(self):
        token_level_scores = torch.tensor([[0.0, 1.0], [0.0, 2.0], [0.0, 3.0]])
        batch = _make_batch(
            {"token_level_scores": token_level_scores},
            non_tensor_data={
                "is_padding": [False, False, False],
                "is_last_step": [True, False, True],
            },
        )
        stats = compute_reward_statistics(batch)
        # Only rows 0 and 2 are last steps: rewards = [1.0, 3.0]
        assert stats["batch/reward/mean"] == pytest.approx(2.0)
        assert stats["batch/reward/max"] == pytest.approx(3.0)
        assert stats["batch/reward/min"] == pytest.approx(1.0)

    def test_all_padded_returns_zeros(self):
        token_level_scores = torch.tensor([[0.0, 1.0], [0.0, 2.0]])
        batch = _make_batch(
            {"token_level_scores": token_level_scores},
            non_tensor_data={
                "is_padding": [True, True],
                "is_last_step": [True, True],
            },
        )
        stats = compute_reward_statistics(batch)
        assert stats["reward_mean"] == 0.0
        assert stats["reward_max"] == 0.0
        assert stats["reward_min"] == 0.0

    def test_multi_step_only_last_step_summed(self):
        """Reward is sum of token_level_scores for last step rows only."""
        scores = torch.tensor([[0.0, 0.5, 0.5], [0.0, 0.0, 3.0], [0.0, 1.0, 1.0]])
        batch = _make_batch(
            {"token_level_scores": scores},
            non_tensor_data={"is_last_step": [True, False, True], "is_padding": [False, False, False]},
        )
        stats = compute_reward_statistics(batch)
        # Rows 0 and 2 are last steps: rewards = [1.0, 2.0]
        assert stats["batch/reward/mean"] == pytest.approx(1.5)

    def test_negative_rewards(self):
        token_level_scores = torch.tensor([[0.0, -1.0], [0.0, -5.0]])
        batch = _make_batch({"token_level_scores": token_level_scores})
        stats = compute_reward_statistics(batch)
        assert stats["batch/reward/mean"] == pytest.approx(-3.0)
        assert stats["batch/reward/max"] == pytest.approx(-1.0)
        assert stats["batch/reward/min"] == pytest.approx(-5.0)

    def test_scores_distributed_across_tokens(self):
        """Reward is sum of ALL token scores, not just the last."""
        token_level_scores = torch.tensor([[0.5, 0.3, 0.2]])  # sum = 1.0
        batch = _make_batch({"token_level_scores": token_level_scores})
        stats = compute_reward_statistics(batch)
        assert stats["batch/reward/mean"] == pytest.approx(1.0)


# ===================================================================
#  compute_pass_metrics
# ===================================================================


class TestComputePassMetrics:
    def test_all_fail(self):
        token_level_scores = torch.tensor([[0.0, 0.0], [0.0, -1.0], [0.0, 0.0], [0.0, -2.0]])
        batch = _make_batch(
            {"token_level_scores": token_level_scores},
            non_tensor_data={"uid": ["a", "a", "b", "b"]},
        )
        metrics, pass_rates = compute_pass_metrics(batch)
        assert metrics["batch/solve/none"] == 2
        assert metrics["batch/solve/all"] == 0
        assert metrics["batch/solve/partial"] == 0
        assert pass_rates["a"] == pytest.approx(0.0)
        assert pass_rates["b"] == pytest.approx(0.0)

    def test_partial(self):
        # uid "a": one pass (reward=1), one fail (reward=0) -> partial
        # uid "b": all pass -> solve_all
        token_level_scores = torch.tensor([[0.0, 1.0], [0.0, 0.0], [0.0, 2.0], [0.0, 1.0]])
        batch = _make_batch(
            {"token_level_scores": token_level_scores},
            non_tensor_data={"uid": ["a", "a", "b", "b"]},
        )
        metrics, pass_rates = compute_pass_metrics(batch)
        assert metrics["batch/solve/all"] == 1  # "b"
        assert metrics["batch/solve/none"] == 0
        assert metrics["batch/solve/partial"] == 1  # "a"
        assert pass_rates["a"] == pytest.approx(0.5)
        assert pass_rates["b"] == pytest.approx(1.0)

    def test_with_is_last_step(self):
        # Only is_last_step=True rows should be considered
        token_level_scores = torch.tensor([[0.0, 100.0], [0.0, 1.0], [0.0, 0.0], [0.0, 2.0]])
        batch = _make_batch(
            {"token_level_scores": token_level_scores},
            non_tensor_data={
                "uid": ["a", "a", "b", "b"],
                "is_last_step": [False, True, False, True],
            },
        )
        metrics, pass_rates = compute_pass_metrics(batch)
        # After filtering: rows 1 and 3 remain (uids "a" and "b")
        assert pass_rates["a"] == pytest.approx(1.0)
        assert pass_rates["b"] == pytest.approx(1.0)

    def test_many_uids(self):
        """Multiple uids with varying pass rates."""
        n = 100
        scores = torch.zeros(n, 2)
        scores[:50, -1] = 1.0  # first 50 pass
        uids = [f"uid_{i // 10}" for i in range(n)]  # 10 uids, 10 samples each
        batch = _make_batch({"token_level_scores": scores}, non_tensor_data={"uid": uids})
        metrics, pass_rates = compute_pass_metrics(batch)
        # First 5 uids (indices 0-49) have all passes, last 5 have all fails
        assert pass_rates["uid_0"] == pytest.approx(1.0)
        assert pass_rates["uid_5"] == pytest.approx(0.0)

    def test_single_uid(self):
        token_level_scores = torch.tensor([[0.0, 1.0], [0.0, 0.5]])
        batch = _make_batch(
            {"token_level_scores": token_level_scores},
            non_tensor_data={"uid": ["x", "x"]},
        )
        metrics, pass_rates = compute_pass_metrics(batch)
        # Row 0: sum=1.0 >= 1 -> pass. Row 1: sum=0.5 < 1 -> fail.
        assert pass_rates["x"] == pytest.approx(0.5)
        assert metrics["batch/solve/partial"] == 1

    def test_reward_exactly_one_counts_as_pass(self):
        """Reward == 1.0 exactly should count as a pass (>= 1)."""
        token_level_scores = torch.tensor([[0.0, 1.0]])
        batch = _make_batch({"token_level_scores": token_level_scores}, non_tensor_data={"uid": ["x"]})
        _, pass_rates = compute_pass_metrics(batch)
        assert pass_rates["x"] == pytest.approx(1.0)

    def test_reward_just_below_one_fails(self):
        """Reward = 0.999 should NOT count as pass."""
        token_level_scores = torch.tensor([[0.0, 0.999]])
        batch = _make_batch({"token_level_scores": token_level_scores}, non_tensor_data={"uid": ["x"]})
        _, pass_rates = compute_pass_metrics(batch)
        assert pass_rates["x"] == pytest.approx(0.0)
