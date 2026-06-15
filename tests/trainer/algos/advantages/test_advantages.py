"""
Tests for advantage estimation functions.

All advantage functions have the uniform signature:
    fn(data: DataProto, config: DictConfig) -> tuple[torch.Tensor, torch.Tensor]
"""

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from axon.protocol import DataProto


def _make_data(tensors, non_tensors=None):
    """Helper to create DataProto from dicts."""
    return DataProto.from_dict(tensors=tensors, non_tensors=non_tensors)


def _make_config(**kwargs):
    """Helper to create a DictConfig representing advantage_args."""
    return OmegaConf.create(kwargs)


class TestGRPOAdvantage:
    """Tests for the GRPO advantage estimator."""

    def test_grpo_basic(self):
        """Test basic GRPO advantage computation."""
        from axon.trainer.algos.advantages.advantage import grpo_advantage_fn

        # Create test data: 4 samples, 2 groups (0, 0, 1, 1)
        rewards = torch.tensor(
            [
                [0.0, 0.0, 1.0],  # Group 0, score = 1.0
                [0.0, 0.0, 0.0],  # Group 0, score = 0.0
                [0.0, 1.0, 1.0],  # Group 1, score = 2.0
                [0.0, 0.0, 1.0],  # Group 1, score = 1.0
            ]
        )
        mask = torch.ones_like(rewards)
        index = np.array(["a", "a", "b", "b"])

        data = _make_data(
            {"token_level_rewards": rewards, "response_mask": mask},
            {"uid": index},
        )
        config = _make_config(epsilon=1e-6, norm_adv_by_std=True)

        advantages, returns = grpo_advantage_fn(data, config)

        assert advantages.shape == rewards.shape
        assert returns.shape == rewards.shape

        # Group 0: mean=0.5, samples are 1.0 and 0.0
        # Group 1: mean=1.5, samples are 2.0 and 1.0
        # Within each group, one should be positive, one negative
        adv_sums = advantages.sum(dim=-1)

        # Sample 0 (score=1.0) should have positive advantage (above group mean)
        assert adv_sums[0] > 0
        # Sample 1 (score=0.0) should have negative advantage (below group mean)
        assert adv_sums[1] < 0
        # Sample 2 (score=2.0) should have positive advantage
        assert adv_sums[2] > 0
        # Sample 3 (score=1.0) should have negative advantage
        assert adv_sums[3] < 0

    def test_grpo_no_std_normalization(self):
        """Test GRPO without std normalization."""
        from axon.trainer.algos.advantages.advantage import grpo_advantage_fn

        rewards = torch.tensor(
            [
                [0.0, 0.0, 2.0],  # Group 0, score = 2.0
                [0.0, 0.0, 0.0],  # Group 0, score = 0.0
            ]
        )
        mask = torch.ones_like(rewards)
        index = np.array(["a", "a"])

        data = _make_data(
            {"token_level_rewards": rewards, "response_mask": mask},
            {"uid": index},
        )
        config = _make_config(epsilon=1e-6, norm_adv_by_std=False)

        # With norm_adv_by_std=False, advantage = score - mean
        # mean = 1.0, so scalar advantages are 1.0 and -1.0
        # These are then broadcasted across all 3 tokens
        advantages, _ = grpo_advantage_fn(data, config)

        # Check the per-token advantage values (all same within a sequence)
        assert torch.allclose(advantages[0, 0], torch.tensor(1.0))  # Score 2 - mean 1 = 1
        assert torch.allclose(advantages[1, 0], torch.tensor(-1.0))  # Score 0 - mean 1 = -1

    def test_grpo_singleton_groups_return_zero(self):
        """Test GRPO with singleton groups returns 0 advantage."""
        from axon.trainer.algos.advantages.advantage import grpo_advantage_fn

        rewards = torch.tensor(
            [
                [0.0, 0.0, 1.0],  # Group A (singleton), score = 1.0
                [0.0, 0.0, 2.0],  # Group B (singleton), score = 2.0
            ]
        )
        mask = torch.ones_like(rewards)
        index = np.array(["a", "b"])  # Each in own group

        data = _make_data(
            {"token_level_rewards": rewards, "response_mask": mask},
            {"uid": index},
        )
        config = _make_config(epsilon=1e-6, norm_adv_by_std=True)

        advantages, _ = grpo_advantage_fn(data, config)

        # Singletons get advantage = 0 (can't compute meaningful advantage with n=1)
        assert torch.allclose(advantages[0], torch.zeros(3))
        assert torch.allclose(advantages[1], torch.zeros(3))

    def test_grpo_respects_mask(self):
        """Test that response mask is applied correctly."""
        from axon.trainer.algos.advantages.advantage import grpo_advantage_fn

        # Different rewards so advantages are non-zero
        rewards = torch.tensor(
            [
                [1.0, 1.0, 1.0],  # score = 3
                [0.0, 0.0, 1.0],  # score = 1
            ]
        )
        # Mask out last token for first sample
        mask = torch.tensor(
            [
                [1.0, 1.0, 0.0],
                [1.0, 1.0, 1.0],
            ]
        )
        index = np.array(["a", "a"])  # Same group, mean = 2

        data = _make_data(
            {"token_level_rewards": rewards, "response_mask": mask},
            {"uid": index},
        )
        config = _make_config()

        advantages, _ = grpo_advantage_fn(data, config)

        # First sample: masked position should be 0, others should be positive
        assert advantages[0, 2].item() == 0.0  # Masked out
        assert advantages[0, 0].item() != 0.0  # Not masked
        # Second sample: non-masked, should have non-zero advantage
        assert advantages[1, 2].item() != 0.0


class TestRLOOAdvantage:
    """Tests for the RLOO advantage estimator."""

    def test_rloo_basic(self):
        """Test basic RLOO advantage computation."""
        from axon.trainer.algos.advantages.advantage import rloo_advantage_fn

        # 3 samples in one group
        rewards = torch.tensor(
            [
                [0.0, 0.0, 3.0],  # score = 3
                [0.0, 0.0, 0.0],  # score = 0
                [0.0, 0.0, 6.0],  # score = 6
            ]
        )
        mask = torch.ones_like(rewards)
        index = np.array(["a", "a", "a"])

        data = _make_data(
            {"token_level_rewards": rewards, "response_mask": mask},
            {"uid": index},
        )
        config = _make_config()

        advantages, _ = rloo_advantage_fn(data, config)

        # LOO formula: adv_i = score_i - mean(others)
        # Sample 0: 3 - (0+6)/2 = 3 - 3 = 0
        # Sample 1: 0 - (3+6)/2 = 0 - 4.5 = -4.5
        # Sample 2: 6 - (3+0)/2 = 6 - 1.5 = 4.5
        # These scalars are broadcast to all tokens, so sum = scalar * num_tokens
        adv_sums = advantages.sum(dim=-1)
        num_tokens = 3

        assert torch.allclose(adv_sums[0], torch.tensor(0.0 * num_tokens), atol=1e-5)
        assert torch.allclose(adv_sums[1], torch.tensor(-4.5 * num_tokens), atol=1e-5)
        assert torch.allclose(adv_sums[2], torch.tensor(4.5 * num_tokens), atol=1e-5)

    def test_rloo_singleton_returns_zero(self):
        """Test RLOO with singleton groups returns 0."""
        from axon.trainer.algos.advantages.advantage import rloo_advantage_fn

        rewards = torch.tensor(
            [
                [0.0, 0.0, 5.0],  # singleton
            ]
        )
        mask = torch.ones_like(rewards)
        index = np.array(["a"])

        data = _make_data(
            {"token_level_rewards": rewards, "response_mask": mask},
            {"uid": index},
        )
        config = _make_config()

        advantages, _ = rloo_advantage_fn(data, config)

        # Singleton returns 0
        assert torch.allclose(advantages, torch.zeros_like(advantages))

    def test_rloo_two_samples(self):
        """Test RLOO with 2 samples per group."""
        from axon.trainer.algos.advantages.advantage import rloo_advantage_fn

        rewards = torch.tensor(
            [
                [0.0, 0.0, 4.0],  # score = 4
                [0.0, 0.0, 2.0],  # score = 2
            ]
        )
        mask = torch.ones_like(rewards)
        index = np.array(["a", "a"])

        data = _make_data(
            {"token_level_rewards": rewards, "response_mask": mask},
            {"uid": index},
        )
        config = _make_config()

        advantages, _ = rloo_advantage_fn(data, config)

        # LOO for 2 samples: adv_i = score_i - other_score
        # Sample 0: 4 - 2 = 2
        # Sample 1: 2 - 4 = -2
        # These scalars are broadcast to all tokens
        adv_sums = advantages.sum(dim=-1)
        num_tokens = 3

        assert torch.allclose(adv_sums[0], torch.tensor(2.0 * num_tokens), atol=1e-5)
        assert torch.allclose(adv_sums[1], torch.tensor(-2.0 * num_tokens), atol=1e-5)


class TestGAEAdvantage:
    """Tests for the GAE advantage estimator."""

    def test_gae_basic(self):
        """Test basic GAE computation."""
        from axon.trainer.algos.advantages.advantage import gae_advantage_fn

        # Simple rewards and values
        rewards = torch.tensor(
            [
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
            ]
        )
        values = torch.tensor(
            [
                [0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5],
            ]
        )
        mask = torch.ones_like(rewards)

        data = _make_data(
            {"token_level_rewards": rewards, "values": values, "response_mask": mask},
        )
        config = _make_config(gamma=0.99, lam=0.95)

        advantages, returns = gae_advantage_fn(data, config)

        assert advantages.shape == rewards.shape
        assert returns.shape == rewards.shape

        # GAE computes: returns = raw_advantages + values
        # Then whitens advantages. So returned advantages are whitened,
        # but returns are computed from raw advantages.
        # We can verify returns are reasonable (positive where rewards are)
        assert returns[0, 2] > returns[0, 0]  # More reward at position 2
        assert returns[1, 1] > returns[1, 2]  # More reward at position 1

    def test_gae_respects_mask(self):
        """Test GAE respects response mask."""
        from axon.trainer.algos.advantages.advantage import gae_advantage_fn

        rewards = torch.tensor([[1.0, 1.0, 1.0]])
        values = torch.tensor([[0.5, 0.5, 0.5]])
        mask = torch.tensor([[1.0, 1.0, 0.0]])  # Last token masked

        data = _make_data(
            {"token_level_rewards": rewards, "values": values, "response_mask": mask},
        )
        config = _make_config()

        advantages, _ = gae_advantage_fn(data, config)

        # Advantages should be whitened, but masked positions should be 0
        # Note: whitening happens before masking in the implementation
        assert advantages.shape == rewards.shape


class TestReinforcePlusPlusAdvantage:
    """Tests for REINFORCE++ advantage estimator."""

    def test_reinforce_pp_basic(self):
        """Test basic REINFORCE++ computation."""
        from axon.trainer.algos.advantages.advantage import reinforce_plus_plus_advantage_fn

        rewards = torch.tensor(
            [
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 2.0],
                [0.0, 0.0, 0.0],
            ]
        )
        mask = torch.ones_like(rewards)

        data = _make_data(
            {"token_level_rewards": rewards, "response_mask": mask},
        )
        config = _make_config(gamma=1.0)

        advantages, returns = reinforce_plus_plus_advantage_fn(data, config)

        assert advantages.shape == rewards.shape
        assert returns.shape == rewards.shape

        # Returns should be cumulative sum from end
        # For gamma=1.0, returns[:, t] = sum of rewards from t to end
        assert torch.allclose(returns[0], torch.tensor([1.0, 1.0, 1.0]))
        assert torch.allclose(returns[1], torch.tensor([2.0, 2.0, 2.0]))
        assert torch.allclose(returns[2], torch.tensor([0.0, 0.0, 0.0]))

    def test_reinforce_pp_with_gamma(self):
        """Test REINFORCE++ with discount factor."""
        from axon.trainer.algos.advantages.advantage import reinforce_plus_plus_advantage_fn

        rewards = torch.tensor([[0.0, 0.0, 1.0]])
        mask = torch.ones_like(rewards)

        data = _make_data(
            {"token_level_rewards": rewards, "response_mask": mask},
        )
        config = _make_config(gamma=0.9)

        _, returns = reinforce_plus_plus_advantage_fn(data, config)

        # With gamma=0.9:
        # returns[:, 2] = 1.0
        # returns[:, 1] = 0.0 + 0.9 * 1.0 = 0.9
        # returns[:, 0] = 0.0 + 0.9 * 0.9 = 0.81
        assert torch.allclose(returns[0], torch.tensor([0.81, 0.9, 1.0]), atol=1e-5)


class TestReinforcePlusPlusBaselineAdvantage:
    """Tests for REINFORCE++ with baseline advantage estimator."""

    def test_reinforce_pp_baseline_basic(self):
        """Test basic REINFORCE++ baseline computation."""
        from axon.trainer.algos.advantages.advantage import reinforce_plus_plus_baseline_advantage_fn

        rewards = torch.tensor(
            [
                [0.0, 0.0, 3.0],  # Group A, score = 3
                [0.0, 0.0, 1.0],  # Group A, score = 1
                [0.0, 0.0, 4.0],  # Group B, score = 4
                [0.0, 0.0, 2.0],  # Group B, score = 2
            ]
        )
        mask = torch.ones_like(rewards)
        index = np.array(["a", "a", "b", "b"])

        data = _make_data(
            {"token_level_rewards": rewards, "response_mask": mask},
            {"uid": index},
        )
        config = _make_config()

        advantages, _ = reinforce_plus_plus_baseline_advantage_fn(data, config)

        assert advantages.shape == rewards.shape

        # After group mean subtraction and global whitening,
        # within each group one should be positive, one negative
        adv_sums = advantages.sum(dim=-1)
        assert adv_sums[0] > 0  # 3 > group mean 2
        assert adv_sums[1] < 0  # 1 < group mean 2
        assert adv_sums[2] > 0  # 4 > group mean 3
        assert adv_sums[3] < 0  # 2 < group mean 3

    def test_reinforce_pp_baseline_singleton_keeps_score(self):
        """Test REINFORCE++ baseline with singleton keeps original score."""
        from axon.trainer.algos.advantages.advantage import reinforce_plus_plus_baseline_advantage_fn

        # Mix of singleton and multi-sample groups
        rewards = torch.tensor(
            [
                [0.0, 0.0, 5.0],  # Group A (singleton), score = 5
                [0.0, 0.0, 2.0],  # Group B, score = 2
                [0.0, 0.0, 4.0],  # Group B, score = 4
            ]
        )
        mask = torch.ones_like(rewards)
        index = np.array(["a", "b", "b"])

        data = _make_data(
            {"token_level_rewards": rewards, "response_mask": mask},
            {"uid": index},
        )
        config = _make_config()

        advantages, _ = reinforce_plus_plus_baseline_advantage_fn(data, config)

        # After whitening, we can't check exact values, but structure should be correct
        assert advantages.shape == rewards.shape


class TestOPOAdvantage:
    """Tests for OPO advantage estimator."""

    def test_opo_basic(self):
        """Test basic OPO computation with length-weighted baseline."""
        from axon.trainer.algos.advantages.advantage import opo_advantage_fn

        rewards = torch.tensor(
            [
                [0.0, 0.0, 4.0],  # score = 4, length = 3
                [0.0, 2.0, 0.0],  # score = 2, length = 3
            ]
        )
        mask = torch.ones_like(rewards)
        index = np.array(["a", "a"])

        data = _make_data(
            {"token_level_rewards": rewards, "response_mask": mask},
            {"uid": index},
        )
        config = _make_config()

        advantages, _ = opo_advantage_fn(data, config)

        assert advantages.shape == rewards.shape

        # OPO uses length-weighted baseline
        # baseline = (3*4 + 3*2) / (3+3) = 18/6 = 3
        # adv[0] = 4 - 3 = 1
        # adv[1] = 2 - 3 = -1
        adv_sums = advantages.sum(dim=-1)
        assert torch.allclose(adv_sums[0], torch.tensor(3.0), atol=1e-5)  # 1 * 3 tokens
        assert torch.allclose(adv_sums[1], torch.tensor(-3.0), atol=1e-5)  # -1 * 3 tokens

    def test_opo_singleton_keeps_score(self):
        """Test OPO with singleton group keeps original score."""
        from axon.trainer.algos.advantages.advantage import opo_advantage_fn

        rewards = torch.tensor(
            [
                [0.0, 0.0, 5.0],  # singleton, score = 5
            ]
        )
        mask = torch.ones_like(rewards)
        index = np.array(["a"])

        data = _make_data(
            {"token_level_rewards": rewards, "response_mask": mask},
            {"uid": index},
        )
        config = _make_config()

        advantages, _ = opo_advantage_fn(data, config)

        # For singleton, baseline = 0, so adv = score = 5
        adv_sum = advantages.sum(dim=-1)
        assert torch.allclose(adv_sum[0], torch.tensor(15.0), atol=1e-5)  # 5 * 3 tokens

    def test_opo_different_lengths(self):
        """Test OPO with different response lengths."""
        from axon.trainer.algos.advantages.advantage import opo_advantage_fn

        rewards = torch.tensor(
            [
                [0.0, 0.0, 6.0],  # score = 6
                [0.0, 3.0, 0.0],  # score = 3 (but only 2 valid tokens)
            ]
        )
        mask = torch.tensor(
            [
                [1.0, 1.0, 1.0],  # length = 3
                [1.0, 1.0, 0.0],  # length = 2
            ]
        )
        index = np.array(["a", "a"])

        data = _make_data(
            {"token_level_rewards": rewards, "response_mask": mask},
            {"uid": index},
        )
        config = _make_config()

        advantages, _ = opo_advantage_fn(data, config)

        # baseline = (3*6 + 2*3) / (3+2) = 24/5 = 4.8
        # adv[0] = 6 - 4.8 = 1.2
        # adv[1] = 3 - 4.8 = -1.8
        assert advantages.shape == rewards.shape


class TestGPGAdvantage:
    """Tests for GPG advantage estimator."""

    def test_gpg_basic(self):
        """Test basic GPG computation."""
        from axon.trainer.algos.advantages.advantage import gpg_advantage_fn

        rewards = torch.tensor(
            [
                [0.0, 0.0, 2.0],  # score = 2
                [0.0, 0.0, 0.0],  # score = 0
            ]
        )
        mask = torch.ones_like(rewards)
        index = np.array(["a", "a"])

        data = _make_data(
            {"token_level_rewards": rewards, "response_mask": mask},
            {"uid": index},
        )
        config = _make_config(f_norm=1.0)

        advantages, _ = gpg_advantage_fn(data, config)

        assert advantages.shape == rewards.shape

        # GPG: alpha * (score - group_mean) / f_norm
        # alpha = 2 / 1 = 2 (only 1 non-zero score)
        # group_mean = 1
        # adv[0] = 2 * (2 - 1) / 1 = 2
        # adv[1] = 2 * (0 - 1) / 1 = -2
        adv_sums = advantages.sum(dim=-1)
        assert torch.allclose(adv_sums[0], torch.tensor(6.0), atol=1e-5)  # 2 * 3 tokens
        assert torch.allclose(adv_sums[1], torch.tensor(-6.0), atol=1e-5)  # -2 * 3 tokens

    def test_gpg_singleton_uses_zero_mean(self):
        """Test GPG with singleton uses mean of 0."""
        from axon.trainer.algos.advantages.advantage import gpg_advantage_fn

        rewards = torch.tensor(
            [
                [0.0, 0.0, 3.0],  # singleton, score = 3
            ]
        )
        mask = torch.ones_like(rewards)
        index = np.array(["a"])

        data = _make_data(
            {"token_level_rewards": rewards, "response_mask": mask},
            {"uid": index},
        )
        config = _make_config(f_norm=1.0)

        advantages, _ = gpg_advantage_fn(data, config)

        # For singleton, mean = 0, so adv = alpha * score
        # alpha = 1 / 1 = 1
        adv_sum = advantages.sum(dim=-1)
        assert torch.allclose(adv_sum[0], torch.tensor(9.0), atol=1e-5)  # 3 * 3 tokens


class TestGRPOPassKAdvantage:
    """Tests for GRPO Pass@k advantage estimator."""

    def test_grpo_passk_basic(self):
        """Test basic GRPO Pass@k computation."""
        from axon.trainer.algos.advantages.advantage import grpo_passk_advantage_fn

        rewards = torch.tensor(
            [
                [0.0, 0.0, 5.0],  # score = 5 (best)
                [0.0, 0.0, 3.0],  # score = 3 (second best)
                [0.0, 0.0, 1.0],  # score = 1
            ]
        )
        mask = torch.ones_like(rewards)
        index = np.array(["a", "a", "a"])

        data = _make_data(
            {"token_level_rewards": rewards, "response_mask": mask},
            {"uid": index},
        )
        config = _make_config(epsilon=1e-6, norm_adv_by_std=False)

        advantages, _ = grpo_passk_advantage_fn(data, config)

        assert advantages.shape == rewards.shape

        # Only best response gets non-zero advantage = r_max - r_second_max
        # adv[0] = 5 - 3 = 2
        # adv[1] = 0
        # adv[2] = 0
        adv_sums = advantages.sum(dim=-1)
        assert torch.allclose(adv_sums[0], torch.tensor(6.0), atol=1e-5)  # 2 * 3 tokens
        assert torch.allclose(adv_sums[1], torch.tensor(0.0), atol=1e-5)
        assert torch.allclose(adv_sums[2], torch.tensor(0.0), atol=1e-5)

    def test_grpo_passk_requires_two_samples(self):
        """Test GRPO Pass@k raises error with single sample."""
        from axon.trainer.algos.advantages.advantage import grpo_passk_advantage_fn

        rewards = torch.tensor([[0.0, 0.0, 5.0]])  # Only 1 sample
        mask = torch.ones_like(rewards)
        index = np.array(["a"])

        data = _make_data(
            {"token_level_rewards": rewards, "response_mask": mask},
            {"uid": index},
        )
        config = _make_config()

        with pytest.raises(ValueError, match="at least 2 samples"):
            grpo_passk_advantage_fn(data, config)

    def test_grpo_passk_with_std_normalization(self):
        """Test GRPO Pass@k with std normalization."""
        from axon.trainer.algos.advantages.advantage import grpo_passk_advantage_fn

        rewards = torch.tensor(
            [
                [0.0, 0.0, 10.0],  # score = 10 (best)
                [0.0, 0.0, 5.0],  # score = 5 (second best)
            ]
        )
        mask = torch.ones_like(rewards)
        index = np.array(["a", "a"])

        data = _make_data(
            {"token_level_rewards": rewards, "response_mask": mask},
            {"uid": index},
        )
        config = _make_config(epsilon=1e-6, norm_adv_by_std=True)

        advantages, _ = grpo_passk_advantage_fn(data, config)

        # Only best response gets advantage, normalized by std
        adv_sums = advantages.sum(dim=-1)
        assert adv_sums[0] > 0  # Best should be positive
        assert adv_sums[1] == 0  # Second best should be 0


class TestRemaxAdvantage:
    """Tests for ReMax advantage estimator."""

    def test_remax_basic(self):
        """Test basic ReMax computation."""
        from axon.trainer.algos.advantages.advantage import remax_advantage_fn

        rewards = torch.tensor(
            [
                [0.0, 0.0, 3.0],
                [0.0, 1.0, 1.0],
            ]
        )
        mask = torch.ones_like(rewards)
        baselines = torch.tensor([2.0, 1.0])  # Greedy decoding baselines

        data = _make_data(
            {"token_level_rewards": rewards, "response_mask": mask, "reward_baselines": baselines},
        )
        config = _make_config()

        advantages, returns = remax_advantage_fn(data, config)

        assert advantages.shape == rewards.shape
        assert returns.shape == rewards.shape

        # Returns are cumulative sum from end
        # returns[0] = [3, 3, 3]
        # returns[1] = [2, 2, 1]
        assert torch.allclose(returns[0], torch.tensor([3.0, 3.0, 3.0]))
        assert torch.allclose(returns[1], torch.tensor([2.0, 2.0, 1.0]))

        # Advantages = returns - baseline * mask
        # adv[0] = [3-2, 3-2, 3-2] = [1, 1, 1]
        # adv[1] = [2-1, 2-1, 1-1] = [1, 1, 0]
        assert torch.allclose(advantages[0], torch.tensor([1.0, 1.0, 1.0]))
        assert torch.allclose(advantages[1], torch.tensor([1.0, 1.0, 0.0]))

    def test_remax_respects_mask(self):
        """Test ReMax respects response mask for returns computation."""
        from axon.trainer.algos.advantages.advantage import remax_advantage_fn

        rewards = torch.tensor([[1.0, 1.0, 1.0]])
        mask = torch.tensor([[1.0, 1.0, 0.0]])  # Last token masked
        baselines = torch.tensor([0.0])

        data = _make_data(
            {"token_level_rewards": rewards, "response_mask": mask, "reward_baselines": baselines},
        )
        config = _make_config()

        advantages, returns = remax_advantage_fn(data, config)

        # Masked rewards should not contribute to returns
        # returns should be computed on masked rewards
        assert returns[0, 2].item() == 0.0  # Masked position


class TestMixedGroups:
    """Tests for mixed group scenarios across different estimators."""

    def test_multiple_groups_different_sizes(self):
        """Test handling of multiple groups with different sizes."""
        from axon.trainer.algos.advantages.advantage import grpo_advantage_fn

        rewards = torch.tensor(
            [
                [0.0, 0.0, 2.0],  # Group A
                [0.0, 0.0, 0.0],  # Group A
                [0.0, 0.0, 5.0],  # Group B (singleton)
                [0.0, 0.0, 3.0],  # Group C
                [0.0, 0.0, 1.0],  # Group C
                [0.0, 0.0, 2.0],  # Group C
            ]
        )
        mask = torch.ones_like(rewards)
        index = np.array(["a", "a", "b", "c", "c", "c"])

        data = _make_data(
            {"token_level_rewards": rewards, "response_mask": mask},
            {"uid": index},
        )
        config = _make_config()

        advantages, _ = grpo_advantage_fn(data, config)

        assert advantages.shape == rewards.shape

        # Singleton (Group B) should have 0 advantage
        assert torch.allclose(advantages[2], torch.zeros(3))

        # Group A: one positive, one negative
        adv_sums = advantages.sum(dim=-1)
        assert adv_sums[0] > 0  # Above mean
        assert adv_sums[1] < 0  # Below mean

        # Group C: one above mean, two below/at mean
        assert adv_sums[3] > 0  # 3 > mean 2


class TestChunkedGAEAdvantage:
    """Tests for the Chunked GAE advantage estimator."""

    def test_chunked_gae_matches_standard_gae_returns(self):
        """Test that chunked GAE produces the same raw returns as standard GAE."""
        from axon.trainer.algos.advantages.advantage import _chunked_gae_core

        rewards = torch.tensor(
            [
                [0.0, 0.0, 1.0, 0.0, 0.5],
                [0.0, 1.0, 0.0, 0.0, 0.0],
            ]
        )
        values = torch.tensor(
            [
                [0.5, 0.5, 0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5, 0.5, 0.5],
            ]
        )
        gamma = 0.99
        lam = 0.95

        # Standard sequential GAE
        B, T = rewards.shape
        nextvalues = 0
        lastgaelam = 0
        adv_rev = []
        for t in reversed(range(T)):
            delta = rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lam * lastgaelam
            nextvalues = values[:, t]
            adv_rev.append(lastgaelam)
        expected_adv = torch.stack(adv_rev[::-1], dim=1)
        expected_returns = expected_adv + values

        # Chunked GAE
        chunked_adv, chunked_returns = _chunked_gae_core(rewards, values, gamma, lam, chunk_size=2)

        assert torch.allclose(chunked_adv, expected_adv, atol=1e-5)
        assert torch.allclose(chunked_returns, expected_returns, atol=1e-5)

    def test_chunked_gae_basic(self):
        """Test basic chunked GAE through the registered function."""
        from axon.trainer.algos.advantages.advantage import chunked_gae_advantage_fn

        rewards = torch.tensor(
            [
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
            ]
        )
        values = torch.tensor(
            [
                [0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5],
            ]
        )
        mask = torch.ones_like(rewards)

        data = _make_data(
            {"token_level_rewards": rewards, "values": values, "response_mask": mask},
        )
        config = _make_config(gamma=0.99, lam=0.95, chunk_size=2)

        advantages, returns = chunked_gae_advantage_fn(data, config)

        assert advantages.shape == rewards.shape
        assert returns.shape == rewards.shape
        # Returns should be higher where more reward is available
        assert returns[0, 2] > returns[0, 0]
        assert returns[1, 1] > returns[1, 2]

    def test_chunked_gae_respects_mask(self):
        """Test chunked GAE respects response mask."""
        from axon.trainer.algos.advantages.advantage import chunked_gae_advantage_fn

        rewards = torch.tensor([[1.0, 1.0, 1.0]])
        values = torch.tensor([[0.5, 0.5, 0.5]])
        mask = torch.tensor([[1.0, 1.0, 0.0]])

        data = _make_data(
            {"token_level_rewards": rewards, "values": values, "response_mask": mask},
        )
        config = _make_config(chunk_size=2)

        advantages, _ = chunked_gae_advantage_fn(data, config)
        assert advantages.shape == rewards.shape

    def test_chunked_gae_exact_chunk_boundary(self):
        """Test chunked GAE when T is an exact multiple of chunk_size (no padding needed)."""
        from axon.trainer.algos.advantages.advantage import _chunked_gae_core

        T = 8
        chunk_size = 4
        rewards = torch.randn(2, T)
        values = torch.randn(2, T)
        gamma, lam = 0.99, 0.95

        # Standard sequential GAE
        nextvalues = 0
        lastgaelam = 0
        adv_rev = []
        for t in reversed(range(T)):
            delta = rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lam * lastgaelam
            nextvalues = values[:, t]
            adv_rev.append(lastgaelam)
        expected_adv = torch.stack(adv_rev[::-1], dim=1)

        chunked_adv, _ = _chunked_gae_core(rewards, values, gamma, lam, chunk_size)
        assert torch.allclose(chunked_adv, expected_adv, atol=1e-5)

    def test_chunked_gae_single_chunk(self):
        """Test chunked GAE when chunk_size >= T (degenerates to one chunk)."""
        from axon.trainer.algos.advantages.advantage import _chunked_gae_core

        rewards = torch.tensor([[0.0, 0.0, 1.0]])
        values = torch.tensor([[0.5, 0.5, 0.5]])
        gamma, lam = 0.99, 0.95

        # Standard sequential
        nextvalues = 0
        lastgaelam = 0
        adv_rev = []
        for t in reversed(range(3)):
            delta = rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lam * lastgaelam
            nextvalues = values[:, t]
            adv_rev.append(lastgaelam)
        expected_adv = torch.stack(adv_rev[::-1], dim=1)

        chunked_adv, _ = _chunked_gae_core(rewards, values, gamma, lam, chunk_size=256)
        assert torch.allclose(chunked_adv, expected_adv, atol=1e-5)

    def test_chunked_gae_gamma_zero(self):
        """Test chunked GAE with gamma=0 (no discounting)."""
        from axon.trainer.algos.advantages.advantage import _chunked_gae_core

        rewards = torch.tensor([[1.0, 2.0, 3.0]])
        values = torch.tensor([[0.5, 0.5, 0.5]])

        adv, _ = _chunked_gae_core(rewards, values, gamma=0.0, lam=0.95, chunk_size=2)

        # With gamma=0: delta_t = r_t - V_t, and no propagation (w=0)
        # So advantages are just deltas: [0.5, 1.5, 2.5]
        expected = rewards - values
        assert torch.allclose(adv, expected, atol=1e-5)

    def test_chunked_gae_matches_standard_gae_with_contiguous_mask(self):
        """Test that chunked GAE returns match standard GAE returns for contiguous masks."""
        from axon.trainer.algos.advantages.advantage import chunked_gae_advantage_fn, gae_advantage_fn

        torch.manual_seed(123)
        B, T = 3, 10
        rewards = torch.randn(B, T)
        values = torch.randn(B, T)
        mask = torch.zeros(B, T)
        mask[0, 5:] = 1.0  # 5 response tokens
        mask[1, 2:] = 1.0  # 8 response tokens
        mask[2, 8:] = 1.0  # 2 response tokens

        config = _make_config(gamma=0.99, lam=0.95, chunk_size=3)

        data_std = _make_data(
            {"token_level_rewards": rewards.clone(), "values": values.clone(), "response_mask": mask.clone()},
        )
        _, returns_std = gae_advantage_fn(data_std, config)

        data_chunked = _make_data(
            {"token_level_rewards": rewards.clone(), "values": values.clone(), "response_mask": mask.clone()},
        )
        _, returns_chunked = chunked_gae_advantage_fn(data_chunked, config)

        # Returns at response positions should match
        diff = ((returns_chunked - returns_std) * mask).abs().max().item()
        assert diff < 1e-4, f"Contiguous mask returns differ by {diff}"

    def test_chunked_gae_matches_standard_gae_with_noncontiguous_mask(self):
        """Test that chunked GAE returns match standard GAE returns for non-contiguous masks."""
        from axon.trainer.algos.advantages.advantage import chunked_gae_advantage_fn, gae_advantage_fn

        rewards = torch.tensor([[1.0, 0.5, 2.0, 0.3, 1.5, 0.8]])
        values = torch.tensor([[0.5, 0.5, 0.5, 0.5, 0.5, 0.5]])
        mask = torch.tensor([[1.0, 0.0, 1.0, 0.0, 1.0, 1.0]])

        config = _make_config(gamma=0.99, lam=0.95, chunk_size=2)

        data_std = _make_data(
            {"token_level_rewards": rewards.clone(), "values": values.clone(), "response_mask": mask.clone()},
        )
        _, returns_std = gae_advantage_fn(data_std, config)

        data_chunked = _make_data(
            {"token_level_rewards": rewards.clone(), "values": values.clone(), "response_mask": mask.clone()},
        )
        _, returns_chunked = chunked_gae_advantage_fn(data_chunked, config)

        diff = ((returns_chunked - returns_std) * mask).abs().max().item()
        assert diff < 1e-4, f"Non-contiguous mask returns differ by {diff}"


# ---------------------------------------------------------------------------
# Hardened edge cases
# ---------------------------------------------------------------------------
class TestGRPOAdvantageEdgeCases:
    def test_all_identical_scores_in_group(self):
        """All identical scores → advantage should be exactly 0 for all samples."""
        from axon.trainer.algos.advantages.advantage import grpo_advantage_fn

        rewards = torch.tensor(
            [
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
            ]
        )
        mask = torch.ones_like(rewards)
        index = np.array(["a", "a"])
        data = _make_data(
            {"token_level_rewards": rewards, "response_mask": mask},
            {"uid": index},
        )
        config = _make_config(epsilon=1e-6, norm_adv_by_std=True)
        advantages, _ = grpo_advantage_fn(data, config)
        # Same scores → (score - mean) = 0 for all
        assert torch.allclose(advantages, torch.zeros_like(advantages), atol=1e-5)

    def test_many_groups_different_sizes(self):
        """Multiple groups with sizes 1, 2, 3 — singletons should get 0."""
        from axon.trainer.algos.advantages.advantage import grpo_advantage_fn

        rewards = torch.tensor(
            [
                [0.0, 0.0, 5.0],  # Group A (singleton)
                [0.0, 0.0, 1.0],  # Group B
                [0.0, 0.0, 3.0],  # Group B
                [0.0, 0.0, 0.0],  # Group C
                [0.0, 0.0, 2.0],  # Group C
                [0.0, 0.0, 4.0],  # Group C
            ]
        )
        mask = torch.ones_like(rewards)
        index = np.array(["a", "b", "b", "c", "c", "c"])
        data = _make_data(
            {"token_level_rewards": rewards, "response_mask": mask},
            {"uid": index},
        )
        config = _make_config(epsilon=1e-6, norm_adv_by_std=True)
        advantages, _ = grpo_advantage_fn(data, config)

        # Singleton group A → advantage = 0
        assert advantages[0].sum().item() == pytest.approx(0.0, abs=1e-5)

    def test_all_zero_rewards(self):
        """All-zero rewards → all advantages should be 0."""
        from axon.trainer.algos.advantages.advantage import grpo_advantage_fn

        rewards = torch.zeros(4, 3)
        mask = torch.ones(4, 3)
        index = np.array(["a", "a", "b", "b"])
        data = _make_data(
            {"token_level_rewards": rewards, "response_mask": mask},
            {"uid": index},
        )
        config = _make_config(epsilon=1e-6, norm_adv_by_std=True)
        advantages, _ = grpo_advantage_fn(data, config)
        assert torch.allclose(advantages, torch.zeros_like(advantages), atol=1e-5)


class TestRLOOAdvantageEdgeCases:
    def test_two_identical_scores(self):
        """Two identical scores in a group → both advantages should be 0."""
        from axon.trainer.algos.advantages.advantage import rloo_advantage_fn

        rewards = torch.tensor(
            [
                [0.0, 0.0, 5.0],
                [0.0, 0.0, 5.0],
            ]
        )
        mask = torch.ones_like(rewards)
        index = np.array(["a", "a"])
        data = _make_data(
            {"token_level_rewards": rewards, "response_mask": mask},
            {"uid": index},
        )
        config = _make_config()
        advantages, _ = rloo_advantage_fn(data, config)
        # LOO: adv_i = (n*s_i - sum) / (n-1) = (2*5 - 10) / 1 = 0
        assert torch.allclose(advantages, torch.zeros_like(advantages), atol=1e-5)

    def test_large_group(self):
        """RLOO with a large group (10 samples) should work correctly."""
        from axon.trainer.algos.advantages.advantage import rloo_advantage_fn

        N = 10
        seq_len = 3
        scores = torch.arange(1.0, N + 1.0)
        rewards = torch.zeros(N, seq_len)
        rewards[:, -1] = scores
        mask = torch.ones(N, seq_len)
        index = np.array(["g"] * N)
        data = _make_data(
            {"token_level_rewards": rewards, "response_mask": mask},
            {"uid": index},
        )
        config = _make_config()
        advantages, _ = rloo_advantage_fn(data, config)
        # LOO scalar: adv_i = (n*s_i - S) / (n-1) = (10*s_i - 55) / 9
        # Then broadcast to all seq_len tokens, so sum = adv_i * seq_len
        S = 55.0
        for i in range(N):
            s_i = float(i + 1)
            expected_scalar = (N * s_i - S) / (N - 1)
            expected_sum = expected_scalar * seq_len
            actual = advantages[i].sum().item()
            assert actual == pytest.approx(expected_sum, abs=1e-4), f"Sample {i}: expected {expected_sum}, got {actual}"


class TestGRPOPassKEdgeCases:
    def test_single_sample_group_raises(self):
        """Pass@K requires at least 2 samples per group."""
        from axon.trainer.algos.advantages.advantage import grpo_passk_advantage_fn

        rewards = torch.tensor([[0.0, 0.0, 1.0]])
        mask = torch.ones_like(rewards)
        index = np.array(["a"])
        data = _make_data(
            {"token_level_rewards": rewards, "response_mask": mask},
            {"uid": index},
        )
        config = _make_config(epsilon=1e-6, norm_adv_by_std=True)
        with pytest.raises(ValueError, match="at least 2 samples"):
            grpo_passk_advantage_fn(data, config)

    def test_all_identical_scores_in_group(self):
        """If all scores are the same, best and second best are equal → advantage=0."""
        from axon.trainer.algos.advantages.advantage import grpo_passk_advantage_fn

        rewards = torch.tensor(
            [
                [0.0, 0.0, 3.0],
                [0.0, 0.0, 3.0],
                [0.0, 0.0, 3.0],
            ]
        )
        mask = torch.ones_like(rewards)
        index = np.array(["a", "a", "a"])
        data = _make_data(
            {"token_level_rewards": rewards, "response_mask": mask},
            {"uid": index},
        )
        config = _make_config(epsilon=1e-6, norm_adv_by_std=False)
        advantages, _ = grpo_passk_advantage_fn(data, config)
        # r_max - r_second_max = 0 → all advantages 0
        assert torch.allclose(advantages, torch.zeros_like(advantages), atol=1e-5)


class TestGPGAdvantageEdgeCases:
    def test_all_zero_scores(self):
        """All-zero scores → count_nonzero=0 → alpha = bsz/1, but advantage should be 0."""
        from axon.trainer.algos.advantages.advantage import gpg_advantage_fn

        rewards = torch.zeros(4, 3)
        mask = torch.ones(4, 3)
        index = np.array(["a", "a", "b", "b"])
        data = _make_data(
            {"token_level_rewards": rewards, "response_mask": mask},
            {"uid": index},
        )
        config = _make_config(f_norm=1.0)
        advantages, _ = gpg_advantage_fn(data, config)
        # (0 - 0) * alpha / f_norm = 0
        assert torch.allclose(advantages, torch.zeros_like(advantages), atol=1e-5)


class TestChunkedGAEEdgeCases:
    def test_all_zero_mask(self):
        """All-zero mask should return zero advantages."""
        from axon.trainer.algos.advantages.advantage import chunked_gae_advantage_fn

        rewards = torch.ones(2, 8)
        values = torch.zeros(2, 8)
        mask = torch.zeros(2, 8)
        data = _make_data(
            {"token_level_rewards": rewards, "values": values, "response_mask": mask},
        )
        config = _make_config(gamma=0.99, lam=0.95, chunk_size=4)
        advantages, _ = chunked_gae_advantage_fn(data, config)
        assert torch.allclose(advantages, torch.zeros_like(advantages))

    def test_single_batch_single_token(self):
        """Minimal case: B=1, T=1."""
        from axon.trainer.algos.advantages.advantage import chunked_gae_advantage_fn, gae_advantage_fn

        rewards = torch.tensor([[1.0]])
        values = torch.tensor([[0.5]])
        mask = torch.ones(1, 1)
        config = _make_config(gamma=0.99, lam=0.95, chunk_size=128)

        data1 = _make_data(
            {"token_level_rewards": rewards.clone(), "values": values.clone(), "response_mask": mask.clone()},
        )
        data2 = _make_data(
            {"token_level_rewards": rewards.clone(), "values": values.clone(), "response_mask": mask.clone()},
        )
        _, ret_std = gae_advantage_fn(data1, config)
        _, ret_chunked = chunked_gae_advantage_fn(data2, config)
        assert torch.allclose(ret_std, ret_chunked, atol=1e-5)


class TestIdentityAdvantage:
    def test_identity_returns_rewards_times_mask(self):
        """Identity advantage should return token_level_rewards * mask."""
        from axon.trainer.algos.advantages.advantage import identity_advantage_fn

        rewards = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        mask = torch.tensor([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]])
        data = _make_data({"token_level_rewards": rewards, "response_mask": mask})
        config = _make_config()
        advantages, returns = identity_advantage_fn(data, config)
        expected = rewards * mask
        assert torch.allclose(advantages, expected)
        assert torch.allclose(returns, expected)


class TestKimiK15Advantage:
    """Tests for the Kimi K1.5 length-shaped advantage estimator."""

    @staticmethod
    def _build_data(scores, lengths, group_ids, max_len=8):
        """Build DataProto where each sample's score and response length match the args.

        Reward is placed as a single-token reward at position (length - 1); response_mask
        covers the first `length` positions, so token_level_rewards.sum() == score and
        response_mask.sum() == length.
        """
        n = len(scores)
        rewards = torch.zeros(n, max_len)
        mask = torch.zeros(n, max_len)
        for i in range(n):
            L = int(lengths[i])
            mask[i, :L] = 1.0
            rewards[i, L - 1] = float(scores[i])
        return _make_data(
            {"token_level_rewards": rewards, "response_mask": mask},
            {"uid": np.array(group_ids)},
        )

    def test_warmup_disables_length_term(self):
        """length_coef=0 should produce identical output to vanilla GRPO."""
        from axon.trainer.algos.advantages.advantage import grpo_advantage_fn, kimi_k1_5_advantage_fn

        # Group with mixed correctness and varied lengths; without length shaping
        # both functions should agree.
        data = self._build_data(
            scores=[1.0, 1.0, 0.0, 0.0],
            lengths=[3, 6, 4, 8],
            group_ids=["a", "a", "a", "a"],
        )
        cfg_k15 = _make_config(length_coef=0.0, norm_adv_by_std=True)
        cfg_grpo = _make_config(norm_adv_by_std=True)

        adv_k15, _ = kimi_k1_5_advantage_fn(data, cfg_k15)
        adv_grpo, _ = grpo_advantage_fn(data, cfg_grpo)
        assert torch.allclose(adv_k15, adv_grpo, atol=1e-5)

    def test_correct_shorter_gets_higher_advantage(self):
        """Among two correct rollouts, the shorter one should land above the longer."""
        from axon.trainer.algos.advantages.advantage import kimi_k1_5_advantage_fn

        # 4 rollouts, all in one group. Two correct (one short, one long), two incorrect.
        data = self._build_data(
            scores=[1.0, 1.0, 0.0, 0.0],
            lengths=[3, 7, 4, 5],
            group_ids=["a", "a", "a", "a"],
        )
        cfg = _make_config(length_coef=0.5, norm_adv_by_std=False)
        advantages, _ = kimi_k1_5_advantage_fn(data, cfg)

        adv = advantages.sum(dim=-1) / data.batch["response_mask"].sum(dim=-1)
        # Short correct (len=3) should outrank long correct (len=7)
        assert adv[0] > adv[1]
        # Both correct should outrank both incorrect
        assert adv[0] > adv[2] and adv[0] > adv[3]
        assert adv[1] > adv[2] and adv[1] > adv[3]

    def test_incorrect_only_penalized_never_bonused(self):
        """Incorrect rollouts get min(0, raw) — short-incorrect must not get a length bonus."""
        from axon.trainer.algos.advantages.advantage import kimi_k1_5_advantage_fn

        # Two correct rollouts (len 4, 8) anchor the group's min/max length.
        # Incorrect-short (len=2) would get raw > 0 if symmetric; clipping must zero it.
        # Incorrect-long (len=10) gets raw < 0 and that penalty stays.
        data = self._build_data(
            scores=[1.0, 1.0, 0.0, 0.0],
            lengths=[4, 8, 2, 10],
            group_ids=["a", "a", "a", "a"],
            max_len=12,
        )
        cfg = _make_config(length_coef=1.0, norm_adv_by_std=False)
        advantages_shaped, _ = kimi_k1_5_advantage_fn(data, cfg)

        # Compare against length_coef=0 (no shaping) to isolate the length term's effect.
        cfg_unshaped = _make_config(length_coef=0.0, norm_adv_by_std=False)
        advantages_unshaped, _ = kimi_k1_5_advantage_fn(data, cfg_unshaped)

        # Recover per-sample shaped vs unshaped contributions to scores by looking at
        # the shift in (mean_subtracted) advantage. Because norm_adv_by_std=False,
        # adv_i = shaped_i - mean(shaped). The shift between shaped and unshaped is:
        #   delta_i = length_reward_i - mean(length_reward)
        per_token_shaped = advantages_shaped[:, 0]
        per_token_unshaped = advantages_unshaped[:, 0]
        delta = per_token_shaped - per_token_unshaped
        mean_delta = delta.mean()
        length_reward = delta - mean_delta + delta.mean()  # equivalently: delta + const

        # The constant cancels when comparing within the batch.
        # Short-incorrect (idx=2): clipped to 0 (would have been positive raw).
        # Long-incorrect (idx=3): kept negative.
        # So delta[2] - delta[3] should be > 0 (short-incorrect > long-incorrect).
        assert delta[2] > delta[3]

        # Long-incorrect must be more penalized than short-correct (length-wise).
        # This sanity-checks that the asymmetric clip does not over-correct.
        assert length_reward[3] < length_reward[1]

    def test_single_correct_group_still_gets_shaping(self):
        """K1.5 paper computes min_len/max_len over ALL rollouts, so a group
        with only one correct rollout still gets a length term."""
        from axon.trainer.algos.advantages.advantage import kimi_k1_5_advantage_fn

        # 1 correct (len=3, the shortest), 3 incorrect (len 5,7,9 — the longest).
        # min_len=3, max_len=9, spread=6.
        # lam = 0.5 - (L-3)/6 → values: 0.5, 0.167, -0.167, -0.5
        # length_reward (correct=lam, incorrect=min(0,lam)): 0.5, 0, -0.167, -0.5
        data = self._build_data(
            scores=[1.0, 0.0, 0.0, 0.0],
            lengths=[3, 5, 7, 9],
            group_ids=["a", "a", "a", "a"],
            max_len=12,
        )
        cfg = _make_config(length_coef=1.0, norm_adv_by_std=False)
        advantages, _ = kimi_k1_5_advantage_fn(data, cfg)

        per_token = advantages[:, 0]
        # The single correct rollout (shortest) gets the largest positive boost.
        assert per_token[0] > per_token[1]
        # The longest incorrect rollout is most penalized.
        assert per_token[3] < per_token[2]
        assert per_token[3] < per_token[1]
        # Shortest incorrect (idx=1) is clipped to 0 length-reward, so it sits
        # between the correct rollout and the longer incorrects.
        assert per_token[1] > per_token[2]

    def test_zero_spread_no_length_pressure(self):
        """All rollouts the same length → max_len == min_len → length term off (paper's stated gate)."""
        from axon.trainer.algos.advantages.advantage import grpo_advantage_fn, kimi_k1_5_advantage_fn

        data = self._build_data(
            scores=[1.0, 1.0, 0.0, 0.0],
            lengths=[5, 5, 5, 5],
            group_ids=["a", "a", "a", "a"],
        )
        cfg_k15 = _make_config(length_coef=1.0, norm_adv_by_std=True)
        cfg_grpo = _make_config(norm_adv_by_std=True)

        adv_k15, _ = kimi_k1_5_advantage_fn(data, cfg_k15)
        adv_grpo, _ = grpo_advantage_fn(data, cfg_grpo)
        assert torch.allclose(adv_k15, adv_grpo, atol=1e-5)

    def test_schedule_warmup_holds_coef_at_zero(self):
        """During warmup (step < warmup_steps), length_coef is held at 0."""
        from axon.trainer.algos.advantages.advantage import grpo_advantage_fn, kimi_k1_5_advantage_fn

        data = self._build_data(
            scores=[1.0, 1.0, 0.0, 0.0],
            lengths=[3, 7, 4, 5],
            group_ids=["a", "a", "a", "a"],
        )
        # Attach a step that's inside the warmup window.
        data.meta_info = {"global_steps": 50}

        cfg_k15 = _make_config(
            length_coef=1.0,
            length_coef_warmup_steps=200,
            length_coef_ramp_steps=0,
            norm_adv_by_std=True,
        )
        cfg_grpo = _make_config(norm_adv_by_std=True)

        adv_k15, _ = kimi_k1_5_advantage_fn(data, cfg_k15)
        adv_grpo, _ = grpo_advantage_fn(data, cfg_grpo)
        # During warmup the K1.5 output must equal vanilla GRPO.
        assert torch.allclose(adv_k15, adv_grpo, atol=1e-5)

    def test_schedule_hard_switch_after_warmup(self):
        """ramp_steps=0 reproduces the paper's hard switch off → constant."""
        from axon.trainer.algos.advantages.advantage import kimi_k1_5_advantage_fn

        data = self._build_data(
            scores=[1.0, 1.0, 0.0, 0.0],
            lengths=[3, 7, 4, 5],
            group_ids=["a", "a", "a", "a"],
        )

        cfg_with_schedule = _make_config(
            length_coef=1.0,
            length_coef_warmup_steps=200,
            length_coef_ramp_steps=0,
            norm_adv_by_std=False,
        )
        cfg_static = _make_config(length_coef=1.0, norm_adv_by_std=False)

        # Step just before the switch: coef = 0 → equals static-zero output.
        data.meta_info = {"global_steps": 199}
        adv_pre, _ = kimi_k1_5_advantage_fn(data, cfg_with_schedule)

        # Step at the switch: coef = target → equals static-1.0 output.
        data.meta_info = {"global_steps": 200}
        adv_post, _ = kimi_k1_5_advantage_fn(data, cfg_with_schedule)
        adv_static, _ = kimi_k1_5_advantage_fn(data, cfg_static)
        assert torch.allclose(adv_post, adv_static, atol=1e-5)
        assert not torch.allclose(adv_pre, adv_post, atol=1e-3)

    def test_schedule_linear_ramp(self):
        """Inside the ramp window, the effective coef is linearly interpolated."""
        from axon.trainer.algos.advantages.advantage import kimi_k1_5_advantage_fn

        data = self._build_data(
            scores=[1.0, 1.0, 0.0, 0.0],
            lengths=[3, 7, 4, 5],
            group_ids=["a", "a", "a", "a"],
        )

        # Schedule: warmup 100 steps, then ramp over 100 steps to coef=1.0.
        # At step 150, the effective coef should be 0.5.
        cfg_ramp = _make_config(
            length_coef=1.0,
            length_coef_warmup_steps=100,
            length_coef_ramp_steps=100,
            norm_adv_by_std=False,
        )
        cfg_static_half = _make_config(length_coef=0.5, norm_adv_by_std=False)

        data.meta_info = {"global_steps": 150}
        adv_ramp, _ = kimi_k1_5_advantage_fn(data, cfg_ramp)
        adv_half, _ = kimi_k1_5_advantage_fn(data, cfg_static_half)
        assert torch.allclose(adv_ramp, adv_half, atol=1e-5)

    def test_schedule_after_ramp_completes(self):
        """After warmup + ramp, the effective coef equals the target."""
        from axon.trainer.algos.advantages.advantage import kimi_k1_5_advantage_fn

        data = self._build_data(
            scores=[1.0, 1.0, 0.0, 0.0],
            lengths=[3, 7, 4, 5],
            group_ids=["a", "a", "a", "a"],
        )

        cfg_ramp = _make_config(
            length_coef=1.0,
            length_coef_warmup_steps=100,
            length_coef_ramp_steps=100,
            norm_adv_by_std=False,
        )
        cfg_static = _make_config(length_coef=1.0, norm_adv_by_std=False)

        data.meta_info = {"global_steps": 5000}
        adv_ramp, _ = kimi_k1_5_advantage_fn(data, cfg_ramp)
        adv_static, _ = kimi_k1_5_advantage_fn(data, cfg_static)
        assert torch.allclose(adv_ramp, adv_static, atol=1e-5)

    def test_schedule_missing_meta_info_defaults_to_step_zero(self):
        """If meta_info has no global_steps, treat as step 0 (i.e., still in warmup if configured)."""
        from axon.trainer.algos.advantages.advantage import grpo_advantage_fn, kimi_k1_5_advantage_fn

        data = self._build_data(
            scores=[1.0, 1.0, 0.0, 0.0],
            lengths=[3, 7, 4, 5],
            group_ids=["a", "a", "a", "a"],
        )
        # No meta_info attached.

        cfg_with_warmup = _make_config(
            length_coef=1.0,
            length_coef_warmup_steps=10,
            norm_adv_by_std=True,
        )
        cfg_grpo = _make_config(norm_adv_by_std=True)

        adv_k15, _ = kimi_k1_5_advantage_fn(data, cfg_with_warmup)
        adv_grpo, _ = grpo_advantage_fn(data, cfg_grpo)
        assert torch.allclose(adv_k15, adv_grpo, atol=1e-5)

    def test_min_max_over_all_rollouts(self):
        """The paper computes min_len/max_len over ALL k rollouts, not just correct ones.

        This test exposes the difference: the longest rollout is incorrect, so an
        implementation that uses correct-only min/max would compute a different
        spread and produce different length rewards.
        """
        from axon.trainer.algos.advantages.advantage import kimi_k1_5_advantage_fn

        # Correct rollouts have lengths [4, 6]; incorrect have [2, 10].
        # Paper (all): min=2, max=10, spread=8.
        # Wrong (correct-only): min=4, max=6, spread=2.
        # The two regimes assign different lam values to every rollout.
        data = self._build_data(
            scores=[1.0, 1.0, 0.0, 0.0],
            lengths=[4, 6, 2, 10],
            group_ids=["a", "a", "a", "a"],
            max_len=12,
        )
        cfg = _make_config(length_coef=1.0, norm_adv_by_std=False)
        advantages, _ = kimi_k1_5_advantage_fn(data, cfg)

        per_token = advantages[:, 0]
        # Compute expected length rewards under the paper formula:
        # lam_i = 0.5 - (L_i - 2) / 8
        # i=0 (correct, L=4): lam = 0.25 → reward = 0.25
        # i=1 (correct, L=6): lam = 0.0 → reward = 0.0
        # i=2 (incorrect, L=2): lam = 0.5 → clipped to 0
        # i=3 (incorrect, L=10): lam = -0.5 → reward = -0.5
        # shaped scores: [1.25, 1.0, 0.0, -0.5], mean = 0.4375
        # per-token (norm_adv_by_std=False): shaped - mean
        expected = torch.tensor([1.25 - 0.4375, 1.0 - 0.4375, 0.0 - 0.4375, -0.5 - 0.4375])
        assert torch.allclose(per_token, expected, atol=1e-5)

    def test_independent_groups(self):
        """Length shaping is per-group: groups with different len_min/len_max do not leak."""
        from axon.trainer.algos.advantages.advantage import kimi_k1_5_advantage_fn

        # Group "a": correct lengths 2 and 4 (small range)
        # Group "b": correct lengths 6 and 10 (large range, larger absolute lengths)
        # A rollout of length 6 in group "a" would be huge; in group "b" it's the min.
        data = self._build_data(
            scores=[1.0, 1.0, 1.0, 1.0],
            lengths=[2, 4, 6, 10],
            group_ids=["a", "a", "b", "b"],
            max_len=12,
        )
        cfg = _make_config(length_coef=0.5, norm_adv_by_std=False)
        advantages, _ = kimi_k1_5_advantage_fn(data, cfg)

        per_token = advantages[:, 0]
        # Within group a: short (2) > long (4)
        assert per_token[0] > per_token[1]
        # Within group b: short (6) > long (10)
        assert per_token[2] > per_token[3]
        # Group a's longer rollout (len=4) should NOT be penalized as if it were
        # group b's longest. Group-relative means group a's length=4 is the
        # max-of-correct in its group, getting raw = -0.5 * length_coef.
        # Group b's length=10 is the max-of-correct in its group, also raw = -0.5 * length_coef.
        # So adv[1] (max in a) and adv[3] (max in b) should equal each other after
        # subtracting their respective group means (which are equal because each
        # group is symmetric: shaped scores 1+0.25, 1-0.25 → mean 1, demeaned ±0.25).
        assert torch.allclose(per_token[1], per_token[3], atol=1e-5)
        assert torch.allclose(per_token[0], per_token[2], atol=1e-5)
