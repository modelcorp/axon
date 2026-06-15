"""Tests for axon.utils.rl.sampler module."""

import pytest
import torch

from axon.utils.rl.sampler import (
    SAFETY_BOUND,
    compute_offpolicy_metrics,
    compute_sampler_correction_and_rejection_mask,
    compute_sampler_correction_weights,
    compute_sampler_rejection_mask,
)


def _make_data(batch_size=4, seq_len=8):
    log_ratio = torch.randn(batch_size, seq_len)
    response_mask = torch.ones(batch_size, seq_len)
    response_mask[:, -2:] = 0
    return log_ratio, response_mask


# ---------------------------------------------------------------------------
# compute_sampler_rejection_mask
# ---------------------------------------------------------------------------
class TestRejectionMask:
    @pytest.mark.parametrize("level", ["token", "sequence", "geometric"])
    def test_all_levels_produce_valid_mask(self, level):
        log_ratio, mask = _make_data()
        modified, metrics = compute_sampler_rejection_mask(
            log_ratio=log_ratio,
            response_mask=mask,
            sampler_rs=level,
            sampler_rs_threshold=3.0,
        )
        assert modified.shape == mask.shape
        assert (modified <= mask).all()
        assert (modified >= 0).all()
        assert "sampler_rs_masked_fraction" in metrics
        assert "sampler_rs_seq_masked_fraction" in metrics

    def test_invalid_level_raises(self):
        log_ratio, mask = _make_data()
        with pytest.raises(ValueError, match="Invalid sampler_rs"):
            compute_sampler_rejection_mask(log_ratio, mask, sampler_rs="invalid", sampler_rs_threshold=2.0)

    def test_none_threshold_raises(self):
        log_ratio, mask = _make_data()
        with pytest.raises(ValueError, match="must be provided"):
            compute_sampler_rejection_mask(log_ratio, mask, sampler_rs="token", sampler_rs_threshold=None)

    def test_zero_log_ratio_no_rejection(self):
        """When policies match (log_ratio=0), IS weight=1.0 → within [0.5, 2.0]."""
        mask = torch.ones(2, 4)
        log_ratio = torch.zeros(2, 4)
        modified, metrics = compute_sampler_rejection_mask(
            log_ratio,
            mask,
            sampler_rs="token",
            sampler_rs_threshold=2.0,
        )
        assert (modified == mask).all()
        assert metrics["sampler_rs_masked_fraction"] == pytest.approx(0.0)

    def test_extreme_positive_log_ratio_fully_rejected(self):
        """exp(10) ≈ 22026 >> threshold=2 → all tokens rejected."""
        mask = torch.ones(2, 4)
        log_ratio = torch.full((2, 4), 10.0)
        modified, metrics = compute_sampler_rejection_mask(
            log_ratio,
            mask,
            sampler_rs="token",
            sampler_rs_threshold=2.0,
        )
        assert modified.sum() == 0
        assert metrics["sampler_rs_masked_fraction"] == pytest.approx(1.0)

    def test_extreme_negative_log_ratio_fully_rejected(self):
        """exp(-5) ≈ 0.0067 < lower_threshold=0.5 → all tokens rejected."""
        mask = torch.ones(2, 4)
        log_ratio = torch.full((2, 4), -5.0)
        modified, _ = compute_sampler_rejection_mask(
            log_ratio,
            mask,
            sampler_rs="token",
            sampler_rs_threshold=2.0,
        )
        assert modified.sum() == 0

    def test_sequence_level_rejects_whole_sequence(self):
        """Sequence-level: one extreme sequence rejected, one normal preserved."""
        mask = torch.ones(2, 4)
        log_ratio = torch.zeros(2, 4)
        log_ratio[0, :] = 10.0  # extreme → sequence-level IS >> threshold
        modified, _ = compute_sampler_rejection_mask(
            log_ratio,
            mask,
            sampler_rs="sequence",
            sampler_rs_threshold=2.0,
        )
        # Row 0 should be fully masked, row 1 should be preserved
        assert modified[0].sum() == 0
        assert modified[1].sum() == 4.0

    def test_single_token_sequence(self):
        mask = torch.tensor([[1.0]])
        log_ratio = torch.tensor([[0.0]])
        modified, _ = compute_sampler_rejection_mask(
            log_ratio,
            mask,
            sampler_rs="token",
            sampler_rs_threshold=2.0,
        )
        assert modified.item() == 1.0


# ---------------------------------------------------------------------------
# compute_sampler_correction_weights
# ---------------------------------------------------------------------------
class TestCorrectionWeights:
    def test_token_level_clamped_and_detached(self):
        log_ratio, mask = _make_data()
        weights, metrics = compute_sampler_correction_weights(
            log_ratio,
            mask,
            sampler_is="token",
            sampler_is_threshold=2.0,
        )
        assert (weights <= 2.0).all()
        assert (weights >= 0).all()
        assert not weights.requires_grad
        assert "sampler_is_mean" in metrics
        assert "sampler_is_eff_sample_size" in metrics

    def test_sequence_level_weights_constant_per_row(self):
        """Sequence-level: all tokens in a row share the same IS weight."""
        mask = torch.ones(2, 4)
        log_ratio = torch.randn(2, 4)
        weights, _ = compute_sampler_correction_weights(
            log_ratio,
            mask,
            sampler_is="sequence",
            sampler_is_threshold=1e6,
        )
        # Within each row, weights at valid positions should be equal
        assert torch.allclose(weights[0, 0].expand(4), weights[0])
        assert torch.allclose(weights[1, 0].expand(4), weights[1])

    def test_invalid_level_raises(self):
        log_ratio, mask = _make_data()
        with pytest.raises(ValueError, match="Invalid sampler_is"):
            compute_sampler_correction_weights(log_ratio, mask, sampler_is="bad")

    def test_zero_threshold_raises(self):
        log_ratio, mask = _make_data()
        with pytest.raises(ValueError, match="must be positive"):
            compute_sampler_correction_weights(log_ratio, mask, sampler_is="token", sampler_is_threshold=0.0)

    def test_zero_ratio_gives_exact_unit_weights(self):
        """log_ratio=0 → exp(0)=1.0 everywhere."""
        mask = torch.ones(2, 4)
        log_ratio = torch.zeros(2, 4)
        weights, metrics = compute_sampler_correction_weights(log_ratio, mask, sampler_is="token")
        assert torch.allclose(weights, mask, atol=1e-6)
        assert metrics["sampler_is_mean"] == pytest.approx(1.0, abs=1e-5)

    def test_safety_bound_prevents_overflow(self):
        """Log ratios beyond ±SAFETY_BOUND are clamped so exp() stays finite."""
        mask = torch.ones(1, 2)
        log_ratio = torch.tensor([[SAFETY_BOUND + 50.0, -(SAFETY_BOUND + 50.0)]])
        weights, _ = compute_sampler_correction_weights(
            log_ratio,
            mask,
            sampler_is="token",
            sampler_is_threshold=1e12,
        )
        assert torch.isfinite(weights).all()
        expected_upper = torch.exp(torch.tensor(SAFETY_BOUND)).item()
        assert weights[0, 0].item() == pytest.approx(expected_upper, rel=1e-4)

    def test_masked_tokens_get_zero_weight(self):
        mask = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
        log_ratio = torch.ones(1, 4)  # exp(1) ≈ 2.72
        weights, _ = compute_sampler_correction_weights(log_ratio, mask, sampler_is="token")
        assert weights[0, 2].item() == 0.0
        assert weights[0, 3].item() == 0.0
        assert weights[0, 0].item() > 0.0


# ---------------------------------------------------------------------------
# compute_sampler_correction_and_rejection_mask – unified interface
# ---------------------------------------------------------------------------
class TestUnifiedInterface:
    def _make_log_probs(self, batch=2, seq=4):
        old_lp = torch.randn(batch, seq) - 3
        sampler_lp = torch.randn(batch, seq) - 3
        mask = torch.ones(batch, seq)
        return old_lp, sampler_lp, mask

    def test_is_only(self):
        """Enable IS weights only, no RS."""
        old, sampler, mask = self._make_log_probs()
        proto, mod_mask, metrics = compute_sampler_correction_and_rejection_mask(
            old,
            sampler,
            mask,
            sampler_is="token",
            sampler_is_threshold=2.0,
        )
        assert proto is not None
        assert "sampler_is_weights" in proto.batch
        assert (mod_mask == mask).all()  # no RS → mask unchanged
        assert any("sampler_corr/sampler_is" in k for k in metrics)

    def test_rs_only(self):
        """Enable RS only, no IS."""
        old, sampler, mask = self._make_log_probs()
        proto, mod_mask, metrics = compute_sampler_correction_and_rejection_mask(
            old,
            sampler,
            mask,
            sampler_rs="token",
            sampler_rs_threshold=2.0,
        )
        assert proto is None  # no IS weights
        assert any("sampler_corr/sampler_rs" in k for k in metrics)

    def test_both_is_and_rs(self):
        """Both IS and RS enabled."""
        old, sampler, mask = self._make_log_probs()
        proto, mod_mask, metrics = compute_sampler_correction_and_rejection_mask(
            old,
            sampler,
            mask,
            sampler_is="token",
            sampler_is_threshold=2.0,
            sampler_rs="token",
            sampler_rs_threshold=3.0,
        )
        assert proto is not None
        assert any("sampler_corr/sampler_is" in k for k in metrics)
        assert any("sampler_corr/sampler_rs" in k for k in metrics)

    def test_neither_is_nor_rs(self):
        """No IS, no RS → mask unchanged, no IS weights."""
        old, sampler, mask = self._make_log_probs()
        proto, mod_mask, metrics = compute_sampler_correction_and_rejection_mask(
            old,
            sampler,
            mask,
        )
        assert proto is None
        assert (mod_mask == mask).all()

    def test_per_token_veto(self):
        """Per-token veto: one catastrophic token vetoes the whole sequence."""
        mask = torch.ones(2, 4)
        old_lp = torch.zeros(2, 4)
        sampler_lp = torch.zeros(2, 4)
        # Make one token in row 0 catastrophic: old - sampler = -10 < log(0.01) ≈ -4.6
        old_lp[0, 1] = -10.0
        proto, mod_mask, metrics = compute_sampler_correction_and_rejection_mask(
            old_lp,
            sampler_lp,
            mask,
            sampler_token_veto_threshold=0.01,
        )
        # Row 0 should be vetoed (all zeros), row 1 should be preserved
        assert mod_mask[0].sum() == 0.0
        assert mod_mask[1].sum() == 4.0
        assert metrics["sampler_corr/sampler_is_veto_fraction"] == pytest.approx(0.5)

    def test_per_token_veto_negative_threshold_raises(self):
        old, sampler, mask = self._make_log_probs()
        with pytest.raises(ValueError, match="must be positive"):
            compute_sampler_correction_and_rejection_mask(
                old,
                sampler,
                mask,
                sampler_token_veto_threshold=-1.0,
            )

    def test_shape_mismatch_raises(self):
        old = torch.randn(2, 4)
        sampler = torch.randn(2, 5)
        mask = torch.ones(2, 4)
        with pytest.raises(ValueError, match="does not match"):
            compute_sampler_correction_and_rejection_mask(old, sampler, mask)

    def test_empty_mask_returns_noop(self):
        """All-padding micro-batches can occur under dynamic batching; the function
        returns a no-op tuple rather than raising."""
        old = torch.randn(2, 4)
        sampler = torch.randn(2, 4)
        mask = torch.zeros(2, 4)
        weights, returned_mask, metrics = compute_sampler_correction_and_rejection_mask(old, sampler, mask)
        assert weights is None
        assert torch.equal(returned_mask, mask)
        assert metrics == {}

    def test_metrics_have_sampler_corr_prefix(self):
        old, sampler, mask = self._make_log_probs()
        _, _, metrics = compute_sampler_correction_and_rejection_mask(
            old,
            sampler,
            mask,
            sampler_is="token",
            sampler_is_threshold=2.0,
        )
        for key in metrics:
            assert key.startswith("sampler_corr/"), f"Key {key} missing prefix"


# ---------------------------------------------------------------------------
# compute_offpolicy_metrics
# ---------------------------------------------------------------------------
class TestOffpolicyMetrics:
    def test_identical_policies_exact_zero_kl(self):
        """KL between identical distributions must be exactly 0."""
        lp = torch.tensor([[-1.0, -2.0, -3.0, -4.0]])
        mask = torch.ones(1, 4)
        metrics = compute_offpolicy_metrics(lp, lp.clone(), mask)
        assert metrics["kl"] == pytest.approx(0.0, abs=1e-7)
        assert metrics["k3_kl"] == pytest.approx(0.0, abs=1e-7)
        assert metrics["chi2_token"] == pytest.approx(0.0, abs=1e-5)
        assert metrics["ppl_ratio"] == pytest.approx(1.0, abs=1e-5)

    def test_none_sampler_returns_only_training_metrics(self):
        lp = torch.randn(2, 4) - 2
        mask = torch.ones(2, 4)
        metrics = compute_offpolicy_metrics(lp, None, mask)
        assert "training_ppl" in metrics
        assert "training_log_ppl" in metrics
        assert "kl" not in metrics
        assert "sampler_ppl" not in metrics
        assert "chi2_token" not in metrics

    def test_all_expected_metrics_present(self):
        old_lp = torch.randn(2, 4) - 2
        sampler_lp = torch.randn(2, 4) - 2
        mask = torch.ones(2, 4)
        metrics = compute_offpolicy_metrics(old_lp, sampler_lp, mask)
        expected = {
            "training_ppl",
            "training_log_ppl",
            "kl",
            "k3_kl",
            "sampler_ppl",
            "sampler_log_ppl",
            "log_ppl_diff",
            "log_ppl_abs_diff",
            "log_ppl_diff_max",
            "log_ppl_diff_min",
            "ppl_ratio",
            "chi2_token",
            "chi2_seq",
        }
        assert expected == set(metrics.keys())

    def test_known_kl_value(self):
        """KL(q || p) = E_q[log q - log p]. With known values, verify manually."""
        # q = sampler, p = training. mask all tokens.
        # log_q = [-1, -2], log_p = [-1.5, -2.5]
        # KL = mean((-1 - (-1.5)), (-2 - (-2.5))) = mean(0.5, 0.5) = 0.5
        old_lp = torch.tensor([[-1.5, -2.5]])
        sampler_lp = torch.tensor([[-1.0, -2.0]])
        mask = torch.ones(1, 2)
        metrics = compute_offpolicy_metrics(old_lp, sampler_lp, mask)
        assert metrics["kl"] == pytest.approx(0.5, abs=1e-6)

    def test_k3_kl_nonnegative_property(self):
        """K3 estimator exp(r)-r-1 >= 0 for all r (convexity of exp)."""
        torch.manual_seed(42)
        for _ in range(10):
            old_lp = torch.randn(4, 16) - 3
            sampler_lp = torch.randn(4, 16) - 3
            mask = torch.ones(4, 16)
            metrics = compute_offpolicy_metrics(old_lp, sampler_lp, mask)
            assert metrics["k3_kl"] >= -1e-6

    def test_partial_mask_ignores_masked_tokens(self):
        """Masked tokens should not affect metrics."""
        old_lp = torch.tensor([[-1.0, -2.0, -999.0, -999.0]])
        sampler_lp = torch.tensor([[-1.0, -2.0, 0.0, 0.0]])
        mask = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
        metrics = compute_offpolicy_metrics(old_lp, sampler_lp, mask)
        # Identical over valid tokens → KL = 0
        assert abs(metrics["kl"]) < 1e-5

    def test_extreme_log_probs_no_nan(self):
        """Extreme log probs may produce inf ppl (exp(100)=inf), but never NaN."""
        old_lp = torch.full((2, 4), -100.0)
        sampler_lp = torch.full((2, 4), -100.0)
        mask = torch.ones(2, 4)
        metrics = compute_offpolicy_metrics(old_lp, sampler_lp, mask)
        for k, v in metrics.items():
            assert not (isinstance(v, float) and v != v), f"NaN in metric {k}"
