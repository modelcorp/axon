"""
Tests for policy loss functions.

All loss functions have the uniform signature:
    fn(data: DataProto, config: DictConfig) -> tuple[torch.Tensor, dict[str, Any]]
"""

import pytest
import torch
from omegaconf import OmegaConf

from axon.protocol import DataProto


def _make_data(tensors, non_tensors=None):
    """Helper to create DataProto from dicts."""
    return DataProto.from_dict(tensors=tensors, non_tensors=non_tensors)


def _make_config(**kwargs):
    """Helper to create a DictConfig representing loss_args."""
    return OmegaConf.create(kwargs)


def _base_tensors(batch_size=2, seq_len=4, *, same_logprobs=False):
    """Build the common tensor dict used by most loss functions.

    When same_logprobs=True, old_log_probs == log_probs so the ratio is 1.0
    everywhere. This is useful for testing that the loss formula reduces to a
    known value.
    """
    old_log_probs = torch.randn(batch_size, seq_len)
    if same_logprobs:
        log_probs = old_log_probs.clone()
    else:
        log_probs = old_log_probs + torch.randn(batch_size, seq_len) * 0.1
    advantages = torch.randn(batch_size, seq_len)
    response_mask = torch.ones(batch_size, seq_len)
    return {
        "old_log_probs": old_log_probs,
        "log_probs": log_probs,
        "advantages": advantages,
        "response_mask": response_mask,
    }


# ---------------------------------------------------------------------------
# PPO
# ---------------------------------------------------------------------------
class TestPPOLoss:
    def test_basic_output_shape_and_metrics(self):
        from axon.trainer.algos.loss.loss import ppo_loss_fn

        data = _make_data(_base_tensors())
        config = _make_config()
        loss, metrics = ppo_loss_fn(data, config)

        assert loss.shape == ()
        assert "pg_clipfrac" in metrics
        assert "ppo_kl" in metrics
        assert "pg_clipfrac_lower" in metrics

    def test_no_clipping_when_ratio_is_one(self):
        """When log probs are identical, ratio=1 and clipfrac should be 0."""
        from axon.trainer.algos.loss.loss import ppo_loss_fn

        tensors = _base_tensors(same_logprobs=True)
        data = _make_data(tensors)
        config = _make_config(clip_ratio=0.2)
        loss, metrics = ppo_loss_fn(data, config)

        assert metrics["pg_clipfrac"] == pytest.approx(0.0, abs=1e-6)
        assert metrics["ppo_kl"] == pytest.approx(0.0, abs=1e-6)

    def test_ratio_one_loss_equals_negative_advantage_mean(self):
        """With ratio=1, PPO loss = masked_mean(-advantages * 1)."""
        from axon.trainer.algos.loss.loss import ppo_loss_fn

        tensors = _base_tensors(batch_size=1, seq_len=3, same_logprobs=True)
        tensors["advantages"] = torch.tensor([[1.0, 2.0, 3.0]])
        tensors["response_mask"] = torch.ones(1, 3)
        data = _make_data(tensors)
        config = _make_config(clip_ratio=0.2)

        loss, _ = ppo_loss_fn(data, config)
        expected = -torch.tensor([1.0, 2.0, 3.0]).mean()
        assert torch.allclose(loss, expected, atol=1e-5)

    def test_custom_clip_ratios(self):
        from axon.trainer.algos.loss.loss import ppo_loss_fn

        data = _make_data(_base_tensors())
        config = _make_config(clip_ratio_low=0.1, clip_ratio_high=0.3)
        loss, metrics = ppo_loss_fn(data, config)
        assert loss.shape == ()

    def test_dual_clip(self):
        """clip_ratio_c controls dual-clip threshold."""
        from axon.trainer.algos.loss.loss import ppo_loss_fn

        data = _make_data(_base_tensors())
        config = _make_config(clip_ratio_c=2.0)
        loss, metrics = ppo_loss_fn(data, config)
        assert loss.shape == ()

    def test_clip_ratio_c_must_be_gt_1(self):
        from axon.trainer.algos.loss.loss import ppo_loss_fn

        data = _make_data(_base_tensors())
        config = _make_config(clip_ratio_c=0.5)
        with pytest.raises(AssertionError):
            ppo_loss_fn(data, config)

    def test_sampler_is_weights(self):
        from axon.trainer.algos.loss.loss import ppo_loss_fn

        tensors = _base_tensors()
        tensors["sampler_is_weights"] = torch.ones_like(tensors["response_mask"])
        data = _make_data(tensors)
        config = _make_config()
        loss, _ = ppo_loss_fn(data, config)
        assert loss.shape == ()

    def test_custom_agg_config(self):
        from axon.trainer.algos.loss.loss import ppo_loss_fn

        data = _make_data(_base_tensors())
        config = _make_config(token_reduce="mean", batch_reduce="step-mean")
        loss, _ = ppo_loss_fn(data, config)
        assert loss.shape == ()

    def test_gradient_flows(self):
        from axon.trainer.algos.loss.loss import ppo_loss_fn

        tensors = _base_tensors()
        tensors["log_probs"] = tensors["log_probs"].requires_grad_(True)
        data = _make_data(tensors)
        config = _make_config()
        loss, _ = ppo_loss_fn(data, config)
        loss.backward()
        assert tensors["log_probs"].grad is not None


# ---------------------------------------------------------------------------
# GSPO
# ---------------------------------------------------------------------------
class TestGSPOLoss:
    def test_basic_output(self):
        from axon.trainer.algos.loss.loss import gspo_loss_fn

        data = _make_data(_base_tensors())
        config = _make_config()
        loss, metrics = gspo_loss_fn(data, config)

        assert loss.shape == ()
        assert "pg_clipfrac" in metrics
        assert "ppo_kl" in metrics

    def test_no_clipping_when_ratio_is_one(self):
        from axon.trainer.algos.loss.loss import gspo_loss_fn

        data = _make_data(_base_tensors(same_logprobs=True))
        config = _make_config(clip_ratio=0.2)
        _, metrics = gspo_loss_fn(data, config)

        assert metrics["pg_clipfrac"] == pytest.approx(0.0, abs=1e-6)
        assert metrics["ppo_kl"] == pytest.approx(0.0, abs=1e-6)

    def test_sampler_is_weights(self):
        from axon.trainer.algos.loss.loss import gspo_loss_fn

        tensors = _base_tensors()
        tensors["sampler_is_weights"] = torch.ones_like(tensors["response_mask"])
        data = _make_data(tensors)
        config = _make_config()
        loss, _ = gspo_loss_fn(data, config)
        assert loss.shape == ()

    def test_gradient_flows(self):
        from axon.trainer.algos.loss.loss import gspo_loss_fn

        tensors = _base_tensors()
        tensors["log_probs"] = tensors["log_probs"].requires_grad_(True)
        data = _make_data(tensors)
        config = _make_config()
        loss, _ = gspo_loss_fn(data, config)
        loss.backward()
        assert tensors["log_probs"].grad is not None

    def test_partial_mask(self):
        from axon.trainer.algos.loss.loss import gspo_loss_fn

        tensors = _base_tensors(batch_size=2, seq_len=4)
        tensors["response_mask"] = torch.tensor(
            [
                [1.0, 1.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 0.0],
            ]
        )
        data = _make_data(tensors)
        config = _make_config()
        loss, _ = gspo_loss_fn(data, config)
        assert loss.shape == ()
        assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# GPG
# ---------------------------------------------------------------------------
class TestGPGLoss:
    """GPG loss is an alias of REINFORCE — same objective, same metrics surface.

    The GPG-specific shaping lives in ``advantage: gpg`` (the advantage
    estimator), not the loss.
    """

    def test_basic_output(self):
        from axon.trainer.algos.loss.loss import gpg_loss_fn

        data = _make_data(_base_tensors())
        config = _make_config()
        loss, metrics = gpg_loss_fn(data, config)

        assert loss.shape == ()
        # Aliases REINFORCE; both return a ppo_kl diagnostic metric.
        assert "ppo_kl" in metrics

    def test_known_value(self):
        """GPG loss = masked_mean(-log_prob * advantages) — identical to REINFORCE."""
        from axon.trainer.algos.loss.loss import gpg_loss_fn

        tensors = {
            "old_log_probs": torch.tensor([[1.0, 2.0]]),
            "log_probs": torch.tensor([[1.0, 2.0]]),
            "advantages": torch.tensor([[3.0, 4.0]]),
            "response_mask": torch.ones(1, 2),
        }
        data = _make_data(tensors)
        config = _make_config(token_reduce="sum", batch_reduce="token-mean")

        loss, _ = gpg_loss_fn(data, config)
        # -log_prob * advantages = [-3, -8], mean = -5.5
        assert torch.allclose(loss, torch.tensor(-5.5))

    def test_sampler_is_weights(self):
        from axon.trainer.algos.loss.loss import gpg_loss_fn

        tensors = _base_tensors()
        tensors["sampler_is_weights"] = 2.0 * torch.ones_like(tensors["response_mask"])
        data = _make_data(tensors)
        config = _make_config()
        loss_with_weights, _ = gpg_loss_fn(data, config)

        # Without weights
        data_no_w = _make_data({k: v for k, v in tensors.items() if k != "sampler_is_weights"})
        loss_no_weights, _ = gpg_loss_fn(data_no_w, config)

        # With weight=2, loss should scale by 2
        assert torch.allclose(loss_with_weights, 2.0 * loss_no_weights, atol=1e-5)

    def test_gradient_flows(self):
        from axon.trainer.algos.loss.loss import gpg_loss_fn

        tensors = _base_tensors()
        tensors["log_probs"] = tensors["log_probs"].requires_grad_(True)
        data = _make_data(tensors)
        config = _make_config()
        loss, _ = gpg_loss_fn(data, config)
        loss.backward()
        assert tensors["log_probs"].grad is not None


# ---------------------------------------------------------------------------
# CLIP_COV
# ---------------------------------------------------------------------------
class TestClipCovLoss:
    def test_basic_output(self):
        from axon.trainer.algos.loss.loss import clip_cov_loss_fn

        data = _make_data(_base_tensors())
        config = _make_config(clip_cov_ratio=0.001)
        loss, metrics = clip_cov_loss_fn(data, config)

        assert loss.shape == ()
        assert "pg_clipfrac" in metrics
        assert "ppo_kl" in metrics

    def test_clip_cov_ratio_must_be_positive(self):
        from axon.trainer.algos.loss.loss import clip_cov_loss_fn

        data = _make_data(_base_tensors())
        config = _make_config(clip_cov_ratio=0)
        with pytest.raises(AssertionError):
            clip_cov_loss_fn(data, config)

    def test_no_clipping_when_ratio_one(self):
        from axon.trainer.algos.loss.loss import clip_cov_loss_fn

        data = _make_data(_base_tensors(same_logprobs=True))
        config = _make_config(clip_cov_ratio=0.001)
        _, metrics = clip_cov_loss_fn(data, config)
        assert metrics["ppo_kl"] == pytest.approx(0.0, abs=1e-6)

    def test_sampler_is_weights(self):
        from axon.trainer.algos.loss.loss import clip_cov_loss_fn

        tensors = _base_tensors()
        tensors["sampler_is_weights"] = torch.ones_like(tensors["response_mask"])
        data = _make_data(tensors)
        config = _make_config(clip_cov_ratio=0.001)
        loss, _ = clip_cov_loss_fn(data, config)
        assert loss.shape == ()


# ---------------------------------------------------------------------------
# KL_COV
# ---------------------------------------------------------------------------
class TestKLCovLoss:
    def test_basic_output(self):
        from axon.trainer.algos.loss.loss import kl_cov_loss_fn

        data = _make_data(_base_tensors())
        config = _make_config(kl_cov_ratio=0.001)
        loss, metrics = kl_cov_loss_fn(data, config)

        assert loss.shape == ()
        assert "ppo_kl" in metrics

    def test_kl_cov_ratio_must_be_positive(self):
        from axon.trainer.algos.loss.loss import kl_cov_loss_fn

        data = _make_data(_base_tensors())
        config = _make_config(kl_cov_ratio=0)
        with pytest.raises(AssertionError):
            kl_cov_loss_fn(data, config)

    def test_no_kl_when_logprobs_same(self):
        from axon.trainer.algos.loss.loss import kl_cov_loss_fn

        data = _make_data(_base_tensors(same_logprobs=True))
        config = _make_config(kl_cov_ratio=0.001)
        _, metrics = kl_cov_loss_fn(data, config)
        assert metrics["ppo_kl"] == pytest.approx(0.0, abs=1e-6)

    def test_sampler_is_weights(self):
        from axon.trainer.algos.loss.loss import kl_cov_loss_fn

        tensors = _base_tensors()
        tensors["sampler_is_weights"] = torch.ones_like(tensors["response_mask"])
        data = _make_data(tensors)
        config = _make_config(kl_cov_ratio=0.001)
        loss, _ = kl_cov_loss_fn(data, config)
        assert loss.shape == ()

    def test_gradient_flows(self):
        from axon.trainer.algos.loss.loss import kl_cov_loss_fn

        tensors = _base_tensors()
        tensors["log_probs"] = tensors["log_probs"].requires_grad_(True)
        data = _make_data(tensors)
        config = _make_config(kl_cov_ratio=0.001)
        loss, _ = kl_cov_loss_fn(data, config)
        loss.backward()
        assert tensors["log_probs"].grad is not None


# ---------------------------------------------------------------------------
# GEO_MEAN
# ---------------------------------------------------------------------------
class TestGeoMeanLoss:
    def test_basic_output(self):
        from axon.trainer.algos.loss.loss import geo_mean_loss_fn

        data = _make_data(_base_tensors())
        config = _make_config()
        loss, metrics = geo_mean_loss_fn(data, config)

        assert loss.shape == ()
        assert "pg_clipfrac" in metrics
        assert "ppo_kl" in metrics
        assert "pg_clipfrac_lower" in metrics

    def test_no_clipping_when_ratio_one(self):
        from axon.trainer.algos.loss.loss import geo_mean_loss_fn

        data = _make_data(_base_tensors(same_logprobs=True))
        config = _make_config(clip_ratio=0.2)
        _, metrics = geo_mean_loss_fn(data, config)
        assert metrics["ppo_kl"] == pytest.approx(0.0, abs=1e-6)

    def test_ratio_one_loss_value(self):
        """When ratio=1 (logprobs same), loss = -mean(seq_advantage)."""
        from axon.trainer.algos.loss.loss import geo_mean_loss_fn

        tensors = _base_tensors(batch_size=1, seq_len=3, same_logprobs=True)
        tensors["advantages"] = torch.tensor([[1.0, 2.0, 3.0]])
        tensors["response_mask"] = torch.ones(1, 3)
        data = _make_data(tensors)
        config = _make_config(clip_ratio=0.2)

        loss, _ = geo_mean_loss_fn(data, config)
        # seq advantage = (1+2+3)/3 = 2.0, ratio exp(0)=1, loss = -2.0, mean over 1 seq = -2.0
        assert torch.allclose(loss, torch.tensor(-2.0), atol=1e-5)

    def test_sampler_is_weights(self):
        from axon.trainer.algos.loss.loss import geo_mean_loss_fn

        tensors = _base_tensors()
        tensors["sampler_is_weights"] = torch.ones_like(tensors["response_mask"])
        data = _make_data(tensors)
        config = _make_config()
        loss, _ = geo_mean_loss_fn(data, config)
        assert loss.shape == ()

    def test_gradient_flows(self):
        from axon.trainer.algos.loss.loss import geo_mean_loss_fn

        tensors = _base_tensors()
        tensors["log_probs"] = tensors["log_probs"].requires_grad_(True)
        data = _make_data(tensors)
        config = _make_config()
        loss, _ = geo_mean_loss_fn(data, config)
        loss.backward()
        assert tensors["log_probs"].grad is not None


# ---------------------------------------------------------------------------
# REINFORCE
# ---------------------------------------------------------------------------
class TestReinforceLoss:
    def test_basic_output(self):
        from axon.trainer.algos.loss.loss import reinforce_loss_fn

        data = _make_data(_base_tensors())
        config = _make_config()
        loss, metrics = reinforce_loss_fn(data, config)

        assert loss.shape == ()
        assert "ppo_kl" in metrics

    def test_known_value(self):
        """REINFORCE loss = agg(-log_prob * advantages)."""
        from axon.trainer.algos.loss.loss import reinforce_loss_fn

        tensors = {
            "old_log_probs": torch.tensor([[0.0, 0.0]]),
            "log_probs": torch.tensor([[1.0, 2.0]]),
            "advantages": torch.tensor([[3.0, 4.0]]),
            "response_mask": torch.ones(1, 2),
        }
        data = _make_data(tensors)
        # default agg mode: token_reduce=sum, batch_reduce=token-mean
        config = _make_config()

        loss, _ = reinforce_loss_fn(data, config)
        # pg_losses = -[3, 8], token-sum = -11, token-mean over 2 tokens = -5.5
        assert torch.allclose(loss, torch.tensor(-5.5))

    def test_sampler_is_weights(self):
        from axon.trainer.algos.loss.loss import reinforce_loss_fn

        tensors = _base_tensors()
        tensors["sampler_is_weights"] = 0.5 * torch.ones_like(tensors["response_mask"])
        data_w = _make_data(tensors)
        config = _make_config()
        loss_w, _ = reinforce_loss_fn(data_w, config)
        assert loss_w.shape == ()

    def test_kl_metric(self):
        """KL should be 0 when old and new log probs match."""
        from axon.trainer.algos.loss.loss import reinforce_loss_fn

        data = _make_data(_base_tensors(same_logprobs=True))
        config = _make_config()
        _, metrics = reinforce_loss_fn(data, config)
        assert metrics["ppo_kl"] == pytest.approx(0.0, abs=1e-6)

    def test_gradient_flows(self):
        from axon.trainer.algos.loss.loss import reinforce_loss_fn

        tensors = _base_tensors()
        tensors["log_probs"] = tensors["log_probs"].requires_grad_(True)
        data = _make_data(tensors)
        config = _make_config()
        loss, _ = reinforce_loss_fn(data, config)
        loss.backward()
        assert tensors["log_probs"].grad is not None

    def test_custom_agg_mode(self):
        from axon.trainer.algos.loss.loss import reinforce_loss_fn

        data = _make_data(_base_tensors())
        config = _make_config(token_reduce="sum", batch_reduce="token-mean")
        loss, _ = reinforce_loss_fn(data, config)
        assert loss.shape == ()


# ---------------------------------------------------------------------------
# VALUE
# ---------------------------------------------------------------------------
def _value_tensors(batch_size=2, seq_len=4):
    """Build the tensor dict used by the value loss function."""
    return {
        "vpreds": torch.randn(batch_size, seq_len),
        "values": torch.randn(batch_size, seq_len),
        "returns": torch.randn(batch_size, seq_len),
        "response_mask": torch.ones(batch_size, seq_len),
    }


class TestValueLoss:
    def test_basic_output_shape_and_metrics(self):
        from axon.trainer.algos.loss.loss import value_loss_fn

        data = _make_data(_value_tensors())
        config = _make_config()
        loss, metrics = value_loss_fn(data, config)

        assert loss.shape == ()
        assert "vf_loss" in metrics
        assert "vf_clipfrac" in metrics
        assert "vpred_mean" in metrics

    def test_no_clipping_when_vpreds_equal_values(self):
        """When vpreds == values, clipping has no effect and clipfrac should be 0."""
        from axon.trainer.algos.loss.loss import value_loss_fn

        tensors = _value_tensors()
        tensors["vpreds"] = tensors["values"].clone()
        data = _make_data(tensors)
        config = _make_config(cliprange_value=0.5)
        loss, metrics = value_loss_fn(data, config)

        assert metrics["vf_clipfrac"] == pytest.approx(0.0, abs=1e-6)

    def test_known_value(self):
        """With vpreds == values, loss = 0.5 * mean((vpreds - returns)^2)."""
        from axon.trainer.algos.loss.loss import value_loss_fn

        values = torch.tensor([[1.0, 2.0, 3.0]])
        returns = torch.tensor([[2.0, 2.0, 2.0]])
        tensors = {
            "vpreds": values.clone(),
            "values": values.clone(),
            "returns": returns,
            "response_mask": torch.ones(1, 3),
        }
        data = _make_data(tensors)
        config = _make_config(cliprange_value=10.0)

        loss, _ = value_loss_fn(data, config)
        # (1-2)^2=1, (2-2)^2=0, (3-2)^2=1 -> mean=2/3, *0.5 = 1/3
        expected = 0.5 * torch.tensor([1.0, 0.0, 1.0]).mean()
        assert torch.allclose(loss, expected, atol=1e-5)

    def test_custom_cliprange(self):
        from axon.trainer.algos.loss.loss import value_loss_fn

        data = _make_data(_value_tensors())
        config = _make_config(cliprange_value=0.1)
        loss, metrics = value_loss_fn(data, config)
        assert loss.shape == ()

    def test_custom_agg_config(self):
        from axon.trainer.algos.loss.loss import value_loss_fn

        data = _make_data(_value_tensors())
        config = _make_config(token_reduce="mean", batch_reduce="step-mean")
        loss, _ = value_loss_fn(data, config)
        assert loss.shape == ()

    def test_gradient_flows(self):
        from axon.trainer.algos.loss.loss import value_loss_fn

        tensors = _value_tensors()
        tensors["vpreds"] = tensors["vpreds"].requires_grad_(True)
        data = _make_data(tensors)
        config = _make_config()
        loss, _ = value_loss_fn(data, config)
        loss.backward()
        assert tensors["vpreds"].grad is not None


# ---------------------------------------------------------------------------
# Hardened edge cases — loss utilities
# ---------------------------------------------------------------------------
class TestLossUtilsEdgeCases:
    """Tests for edge cases in loss utility functions."""

    def test_masked_mean_all_zero_mask(self):
        """masked_mean with all-zero mask should return 0, not NaN or crash."""
        from axon.trainer.algos.loss.utils import masked_mean

        values = torch.tensor([1.0, 2.0, 3.0])
        mask = torch.zeros(3)
        result = masked_mean(values, mask)
        assert not torch.isnan(result), "masked_mean returned NaN for all-zero mask"
        assert result.item() == pytest.approx(0.0)

    def test_agg_loss_invalid_token_reduce(self):
        """Invalid token_reduce should raise ValueError."""
        from axon.trainer.algos.loss.utils import agg_loss

        loss_mat = torch.ones(2, 4)
        mask = torch.ones(2, 4)
        with pytest.raises(ValueError, match="Invalid token_reduce"):
            agg_loss(loss_mat, mask, token_reduce="invalid_mode")

    def test_agg_loss_invalid_batch_reduce(self):
        """Invalid batch_reduce should raise ValueError."""
        from axon.trainer.algos.loss.utils import agg_loss

        loss_mat = torch.ones(2, 4)
        mask = torch.ones(2, 4)
        with pytest.raises(ValueError, match="Invalid batch_reduce"):
            agg_loss(loss_mat, mask, batch_reduce="invalid_mode")

    def test_agg_loss_all_zero_mask(self):
        """All-zero mask should not crash and should return 0 loss."""
        from axon.trainer.algos.loss.utils import agg_loss

        loss_mat = torch.ones(2, 4)
        mask = torch.zeros(2, 4)
        loss = agg_loss(loss_mat, mask, token_reduce="sum", batch_reduce="token-mean")
        assert loss.item() == pytest.approx(0.0)

    def test_agg_loss_single_token(self):
        """Single-token sequence should work for all reduce modes."""
        from axon.trainer.algos.loss.utils import agg_loss

        loss_mat = torch.tensor([[5.0]])
        mask = torch.tensor([[1.0]])
        for tr in ["sum", "mean", "mean-norm"]:
            loss = agg_loss(loss_mat, mask, token_reduce=tr, batch_reduce="step-mean")
            assert loss.shape == ()
            assert torch.isfinite(loss)

    def test_entropy_from_logits_uniform(self):
        """Uniform logits should give max entropy = log(V)."""
        import math

        from axon.trainer.algos.loss.utils import entropy_from_logits

        V = 10
        logits = torch.zeros(2, V)
        entropy = entropy_from_logits(logits)
        expected = math.log(V)
        assert torch.allclose(entropy, torch.full((2,), expected), atol=1e-4)

    def test_entropy_from_logits_peaked(self):
        """Very peaked logits should give near-zero entropy."""
        from axon.trainer.algos.loss.utils import entropy_from_logits

        logits = torch.tensor([[100.0, -100.0, -100.0]])
        entropy = entropy_from_logits(logits)
        assert entropy.item() < 0.01

    def test_clip_by_value_with_scalar_bounds(self):
        """clip_by_value should work with plain float min/max args."""
        from axon.trainer.algos.loss.utils import clip_by_value

        tensor = torch.tensor([1.0, 5.0, 10.0])
        result = clip_by_value(tensor, min_val=2.0, max_val=8.0)
        assert result.tolist() == [2.0, 5.0, 8.0]


class TestPPOLossEdgeCases:
    """Edge cases for PPO loss that could break the implementation."""

    def test_ppo_all_zero_advantages(self):
        """All-zero advantages: loss should be exactly 0."""
        from axon.trainer.algos.loss.loss import ppo_loss_fn

        tensors = _base_tensors(same_logprobs=True)
        tensors["advantages"] = torch.zeros_like(tensors["advantages"])
        data = _make_data(tensors)
        config = _make_config()
        loss, metrics = ppo_loss_fn(data, config)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_ppo_single_token_sequence(self):
        """PPO with sequence length 1 should work."""
        from axon.trainer.algos.loss.loss import ppo_loss_fn

        tensors = _base_tensors(batch_size=1, seq_len=1, same_logprobs=True)
        tensors["advantages"] = torch.tensor([[1.0]])
        data = _make_data(tensors)
        config = _make_config()
        loss, metrics = ppo_loss_fn(data, config)
        assert loss.shape == ()
        assert torch.isfinite(loss)

    def test_ppo_nan_advantages_propagate(self):
        """NaN in advantages should propagate to loss, not be silently masked."""
        from axon.trainer.algos.loss.loss import ppo_loss_fn

        tensors = _base_tensors(batch_size=1, seq_len=3, same_logprobs=True)
        tensors["advantages"] = torch.tensor([[1.0, float("nan"), 3.0]])
        data = _make_data(tensors)
        config = _make_config()
        loss, _ = ppo_loss_fn(data, config)
        # NaN in advantages should result in NaN loss (not silently ignored)
        assert torch.isnan(loss), "NaN advantages should propagate to NaN loss"

    def test_ppo_very_large_log_ratio_is_clamped(self):
        """Very large log probability differences should be clamped, not overflow."""
        from axon.trainer.algos.loss.loss import ppo_loss_fn

        tensors = _base_tensors(batch_size=1, seq_len=2, same_logprobs=False)
        tensors["log_probs"] = tensors["old_log_probs"] + 100.0  # huge difference
        data = _make_data(tensors)
        config = _make_config()
        loss, _ = ppo_loss_fn(data, config)
        assert torch.isfinite(loss), "Large log ratio should be clamped, not overflow"

    def test_ppo_clip_ratio_c_exactly_one_raises(self):
        """clip_ratio_c must be > 1.0."""
        from axon.trainer.algos.loss.loss import ppo_loss_fn

        data = _make_data(_base_tensors())
        config = _make_config(clip_ratio_c=1.0)
        with pytest.raises(AssertionError, match="clip_ratio_c must be > 1.0"):
            ppo_loss_fn(data, config)

    def test_ppo_partial_mask(self):
        """PPO with partial response mask should only count masked tokens."""
        from axon.trainer.algos.loss.loss import ppo_loss_fn

        tensors = _base_tensors(batch_size=2, seq_len=4, same_logprobs=True)
        tensors["advantages"] = torch.ones(2, 4)
        tensors["response_mask"] = torch.tensor(
            [
                [1.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 1.0],
            ]
        )
        data = _make_data(tensors)
        config = _make_config()
        loss, _ = ppo_loss_fn(data, config)
        assert torch.isfinite(loss)
