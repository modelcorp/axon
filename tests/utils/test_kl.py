"""
Comprehensive tests for axon.utils.rl.kl -- KL divergence utilities for PPO training.
"""

import math
import random

import pytest
import torch

from axon.utils.rl.kl import (
    AdaptiveKLController,
    FixedKLController,
    apply_kl_penalty,
    compute_rewards_with_kl,
    get_kl_controller,
    kl_penalty,
    kl_penalty_forward,
)

# ===================================================================
#  AdaptiveKLController
# ===================================================================


class TestAdaptiveKLController:
    def test_kl_above_target_increases_value(self):
        ctrl = AdaptiveKLController(init_kl_coef=0.1, target_kl=6.0, horizon=10000)
        original = ctrl.value
        ctrl.update(current_kl=12.0, n_steps=32)
        assert ctrl.value > original

    def test_kl_below_target_decreases_value(self):
        ctrl = AdaptiveKLController(init_kl_coef=0.1, target_kl=6.0, horizon=10000)
        original = ctrl.value
        ctrl.update(current_kl=3.0, n_steps=32)
        assert ctrl.value < original

    def test_kl_at_target_value_unchanged(self):
        ctrl = AdaptiveKLController(init_kl_coef=0.1, target_kl=6.0, horizon=10000)
        original = ctrl.value
        ctrl.update(current_kl=6.0, n_steps=32)
        assert ctrl.value == original

    def test_exact_update_calculation(self):
        ctrl = AdaptiveKLController(init_kl_coef=0.5, target_kl=4.0, horizon=1000)
        # current_kl / target - 1 = 8.0 / 4.0 - 1 = 1.0
        # clipped to 0.2
        # mult = 1 + 0.2 * 10 / 1000 = 1.002
        ctrl.update(current_kl=8.0, n_steps=10)
        expected = 0.5 * (1 + 0.2 * 10 / 1000)
        assert ctrl.value == pytest.approx(expected)

    def test_error_clipped_high(self):
        """When kl >> target, error is clipped to 0.2."""
        ctrl = AdaptiveKLController(init_kl_coef=1.0, target_kl=1.0, horizon=100)
        # current_kl / target - 1 = 100.0 -> clipped to 0.2
        ctrl.update(current_kl=100.0, n_steps=10)
        expected = 1.0 * (1 + 0.2 * 10 / 100)
        assert ctrl.value == pytest.approx(expected)

    def test_error_clipped_low(self):
        """When kl << target, error is clipped to -0.2."""
        ctrl = AdaptiveKLController(init_kl_coef=1.0, target_kl=10.0, horizon=100)
        # current_kl / target - 1 = 0.001 / 10.0 - 1 = -0.9999 -> clipped to -0.2
        ctrl.update(current_kl=0.001, n_steps=10)
        expected = 1.0 * (1 + (-0.2) * 10 / 100)
        assert ctrl.value == pytest.approx(expected)

    def test_multiple_updates_accumulate(self):
        ctrl = AdaptiveKLController(init_kl_coef=1.0, target_kl=5.0, horizon=100)
        for _ in range(3):
            ctrl.update(current_kl=10.0, n_steps=10)
        # Each update: proportional_error = clip(10/5 - 1, -0.2, 0.2) = 0.2
        # mult = 1 + 0.2 * 10/100 = 1.02
        expected = 1.0 * (1.02**3)
        assert ctrl.value == pytest.approx(expected)

    def test_zero_n_steps_no_change(self):
        ctrl = AdaptiveKLController(init_kl_coef=0.5, target_kl=5.0, horizon=100)
        original = ctrl.value
        ctrl.update(current_kl=100.0, n_steps=0)
        # mult = 1 + error * 0 / horizon = 1.0
        assert ctrl.value == original

    def test_convergence_stays_bounded(self):
        """Run 1000 updates with random KL values; verify value stays bounded (not 0 or inf)."""
        rng = random.Random(42)
        ctrl = AdaptiveKLController(init_kl_coef=0.1, target_kl=6.0, horizon=10000)
        for _ in range(1000):
            kl_val = rng.uniform(0.01, 50.0)
            ctrl.update(current_kl=kl_val, n_steps=32)
            assert ctrl.value > 0, "KL coefficient went to 0"
            assert ctrl.value < 1e10, "KL coefficient exploded"
            assert math.isfinite(ctrl.value), "KL coefficient is not finite"

    def test_update_with_zero_target_kl_does_not_crash(self):
        """target_kl=0 causes division by zero in proportional_error calculation."""
        ctrl = AdaptiveKLController(init_kl_coef=0.1, target_kl=0.0, horizon=10)
        # Should not raise ZeroDivisionError
        ctrl.update(current_kl=0.5, n_steps=1)
        assert math.isfinite(ctrl.value), "KL coefficient should remain finite"

    def test_update_with_very_small_target_kl(self):
        """Very small target_kl should not cause numerical overflow."""
        ctrl = AdaptiveKLController(init_kl_coef=0.1, target_kl=1e-30, horizon=10)
        ctrl.update(current_kl=0.5, n_steps=1)
        assert ctrl.value > 0
        assert math.isfinite(ctrl.value)

    def test_update_with_negative_current_kl(self):
        """Negative current_kl is nonsensical but should not crash."""
        ctrl = AdaptiveKLController(init_kl_coef=0.1, target_kl=6.0, horizon=10000)
        ctrl.update(current_kl=-1.0, n_steps=32)
        assert math.isfinite(ctrl.value)


# ===================================================================
#  FixedKLController
# ===================================================================


class TestFixedKLController:
    def test_update_is_noop(self):
        ctrl = FixedKLController(kl_coef=0.05)
        ctrl.update(current_kl=999.0, n_steps=1000)
        assert ctrl.value == 0.05

    def test_repeated_update_still_noop(self):
        ctrl = FixedKLController(kl_coef=0.3)
        for _ in range(100):
            ctrl.update(current_kl=50.0, n_steps=64)
        assert ctrl.value == 0.3


# ===================================================================
#  get_kl_controller
# ===================================================================


class _FakeConfig:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class TestGetKLController:
    def test_adaptive_horizon_zero_raises(self):
        cfg = _FakeConfig(type="adaptive", kl_coef=0.1, target_kl=6.0, horizon=0)
        with pytest.raises(AssertionError):
            get_kl_controller(cfg)

    def test_unknown_type_raises(self):
        cfg = _FakeConfig(type="magic")
        with pytest.raises(NotImplementedError, match="Unknown KL controller type"):
            get_kl_controller(cfg)

    def test_fixed_returns_fixed_controller(self):
        cfg = _FakeConfig(type="fixed", kl_coef=0.1)
        ctrl = get_kl_controller(cfg)
        assert isinstance(ctrl, FixedKLController)
        assert ctrl.value == 0.1

    def test_adaptive_returns_adaptive_controller(self):
        cfg = _FakeConfig(type="adaptive", kl_coef=0.2, target_kl=6.0, horizon=10000)
        ctrl = get_kl_controller(cfg)
        assert isinstance(ctrl, AdaptiveKLController)
        assert ctrl.value == 0.2
        assert ctrl.target == 6.0
        assert ctrl.horizon == 10000


# ===================================================================
#  kl_penalty_forward
# ===================================================================


class TestKLPenaltyForward:
    @pytest.fixture
    def log_tensors(self):
        logprob = torch.tensor([[-1.0, -2.0, -3.0], [-0.5, -1.5, -2.5]])
        ref_logprob = torch.tensor([[-1.5, -1.0, -3.5], [-0.5, -2.0, -2.0]])
        return logprob, ref_logprob

    # --- kl / k1 ---

    def test_kl_basic(self, log_tensors):
        logprob, ref_logprob = log_tensors
        result = kl_penalty_forward(logprob, ref_logprob, "kl")
        expected = logprob - ref_logprob
        assert torch.allclose(result, expected)

    def test_k1_matches_kl(self, log_tensors):
        logprob, ref_logprob = log_tensors
        r1 = kl_penalty_forward(logprob, ref_logprob, "kl")
        r2 = kl_penalty_forward(logprob, ref_logprob, "k1")
        assert torch.allclose(r1, r2)

    # --- abs ---

    def test_abs_basic(self, log_tensors):
        logprob, ref_logprob = log_tensors
        result = kl_penalty_forward(logprob, ref_logprob, "abs")
        expected = (logprob - ref_logprob).abs()
        assert torch.allclose(result, expected)

    def test_abs_always_nonnegative(self, log_tensors):
        logprob, ref_logprob = log_tensors
        result = kl_penalty_forward(logprob, ref_logprob, "abs")
        assert (result >= 0).all()

    # --- mse / k2 ---

    def test_mse_basic(self, log_tensors):
        logprob, ref_logprob = log_tensors
        result = kl_penalty_forward(logprob, ref_logprob, "mse")
        expected = 0.5 * (logprob - ref_logprob).square()
        assert torch.allclose(result, expected)

    def test_k2_matches_mse(self, log_tensors):
        logprob, ref_logprob = log_tensors
        r1 = kl_penalty_forward(logprob, ref_logprob, "mse")
        r2 = kl_penalty_forward(logprob, ref_logprob, "k2")
        assert torch.allclose(r1, r2)

    def test_mse_always_nonnegative(self, log_tensors):
        logprob, ref_logprob = log_tensors
        result = kl_penalty_forward(logprob, ref_logprob, "mse")
        assert (result >= 0).all()

    # --- low_var_kl / k3 ---

    def test_low_var_kl_basic(self, log_tensors):
        logprob, ref_logprob = log_tensors
        result = kl_penalty_forward(logprob, ref_logprob, "low_var_kl")
        kl = ref_logprob - logprob
        kl = torch.clamp(kl, min=-20, max=20)
        ratio = torch.exp(kl)
        expected = torch.clamp(ratio - kl - 1, min=-10, max=10)
        assert torch.allclose(result, expected)

    def test_k3_matches_low_var_kl(self, log_tensors):
        logprob, ref_logprob = log_tensors
        r1 = kl_penalty_forward(logprob, ref_logprob, "low_var_kl")
        r2 = kl_penalty_forward(logprob, ref_logprob, "k3")
        assert torch.allclose(r1, r2)

    def test_low_var_kl_zero_when_equal(self):
        """When logprob == ref_logprob, kl=0, ratio=1, result = 1 - 0 - 1 = 0."""
        logprob = torch.tensor([[-1.0, -2.0, -3.0]])
        result = kl_penalty_forward(logprob, logprob, "low_var_kl")
        assert torch.allclose(result, torch.zeros_like(result))

    def test_low_var_kl_nonnegative_by_convexity(self):
        """e^x - x - 1 >= 0 for all x. Result should be non-negative before clamp."""
        logprob = torch.randn(10, 20)
        ref_logprob = torch.randn(10, 20)
        result = kl_penalty_forward(logprob, ref_logprob, "k3")
        # Due to clamping, it should still be >= -10, but for reasonable inputs >= 0
        assert (result >= -1e-6).all()

    def test_low_var_kl_clamped_output(self):
        """Extreme inputs should be clamped to [-10, 10]."""
        logprob = torch.tensor([[100.0]])
        ref_logprob = torch.tensor([[-100.0]])
        result = kl_penalty_forward(logprob, ref_logprob, "k3")
        assert result.item() <= 10.0
        assert result.item() >= -10.0

    # --- Mathematical properties ---

    def test_kl_penalty_zero_when_equal_all_types(self):
        """kl_penalty(logp, logp) should be 0 for all types."""
        logprob = torch.randn(4, 8)
        for ptype in ("kl", "k1", "abs", "mse", "k2", "low_var_kl", "k3"):
            result = kl_penalty_forward(logprob, logprob, ptype)
            assert torch.allclose(result, torch.zeros_like(result), atol=1e-6), (
                f"kl_penalty({ptype}) should be 0 when logp == ref_logp"
            )

    def test_nonnegative_for_nonneg_types(self):
        """abs, mse, k3 should always be non-negative."""
        torch.manual_seed(42)
        logprob = torch.randn(10, 20)
        ref_logprob = torch.randn(10, 20)
        for ptype in ("abs", "mse", "k2"):
            result = kl_penalty_forward(logprob, ref_logprob, ptype)
            assert (result >= -1e-7).all(), f"kl_penalty({ptype}) should be non-negative"

    # --- shape preservation ---

    def test_output_shape_matches_input(self, log_tensors):
        logprob, ref_logprob = log_tensors
        for ptype in ("kl", "k1", "abs", "mse", "k2", "low_var_kl", "k3"):
            result = kl_penalty_forward(logprob, ref_logprob, ptype)
            assert result.shape == logprob.shape, f"Shape mismatch for {ptype}"

    # --- errors ---

    def test_full_raises(self, log_tensors):
        logprob, ref_logprob = log_tensors
        with pytest.raises(NotImplementedError, match="Full KL"):
            kl_penalty_forward(logprob, ref_logprob, "full")

    def test_unknown_type_raises(self, log_tensors):
        logprob, ref_logprob = log_tensors
        with pytest.raises(NotImplementedError, match="Unknown KL penalty type"):
            kl_penalty_forward(logprob, ref_logprob, "bogus")

    # --- forward with "+" suffix strips correctly ---

    def test_k1_plus_forward_matches_k1(self, log_tensors):
        logprob, ref_logprob = log_tensors
        r1 = kl_penalty_forward(logprob, ref_logprob, "k1")
        r2 = kl_penalty_forward(logprob, ref_logprob, "k1+")
        assert torch.allclose(r1, r2)

    def test_k3_plus_forward_matches_k3(self, log_tensors):
        logprob, ref_logprob = log_tensors
        r1 = kl_penalty_forward(logprob, ref_logprob, "k3")
        r2 = kl_penalty_forward(logprob, ref_logprob, "k3+")
        assert torch.allclose(r1, r2)

    # --- numerical stability ---

    def test_large_log_ratio_no_nan_inf(self):
        """Very large log ratio (e.g., 100.0) should not produce NaN/Inf."""
        logprob = torch.tensor([[100.0, -100.0, 50.0]])
        ref_logprob = torch.tensor([[-100.0, 100.0, -50.0]])
        for ptype in ("kl", "k1", "abs", "mse", "k2", "low_var_kl", "k3"):
            result = kl_penalty_forward(logprob, ref_logprob, ptype)
            assert torch.isfinite(result).all(), f"Non-finite result for {ptype}"


# ===================================================================
#  kl_penalty (with straight-through trick)
# ===================================================================


class TestKLPenalty:
    @pytest.fixture
    def grad_tensors(self):
        logprob = torch.tensor([[-1.0, -2.0, -3.0]], requires_grad=True)
        ref_logprob = torch.tensor([[-1.5, -1.0, -3.5]])
        return logprob, ref_logprob

    @pytest.mark.parametrize("ptype", ["kl", "k1", "abs", "mse", "k2", "low_var_kl", "k3"])
    def test_non_plus_returns_forward(self, ptype):
        logprob = torch.tensor([[1.0, 2.0]])
        ref_logprob = torch.tensor([[0.5, 1.5]])
        result = kl_penalty(logprob, ref_logprob, ptype)
        expected = kl_penalty_forward(logprob, ref_logprob, ptype)
        assert torch.allclose(result, expected)

    def test_k1_plus_value_matches_k1(self, grad_tensors):
        logprob, ref_logprob = grad_tensors
        result = kl_penalty(logprob, ref_logprob, "k1+")
        expected_forward = kl_penalty_forward(logprob.detach(), ref_logprob, "k1")
        assert torch.allclose(result.detach(), expected_forward)

    def test_k3_plus_value_matches_k3(self, grad_tensors):
        logprob, ref_logprob = grad_tensors
        result = kl_penalty(logprob, ref_logprob, "k3+")
        expected_forward = kl_penalty_forward(logprob.detach(), ref_logprob, "k3")
        assert torch.allclose(result.detach(), expected_forward)

    def test_k1_plus_gradient_matches_k2(self, grad_tensors):
        """k1+ uses straight-through: forward from k1, gradient from k2."""
        logprob, ref_logprob = grad_tensors
        result = kl_penalty(logprob, ref_logprob, "k1+")
        loss = result.sum()
        loss.backward()
        grad_k1_plus = logprob.grad.clone()

        # Compare with pure k2 gradient
        logprob2 = logprob.detach().clone().requires_grad_(True)
        result2 = kl_penalty(logprob2, ref_logprob, "k2")
        result2.sum().backward()
        grad_k2 = logprob2.grad.clone()

        assert torch.allclose(grad_k1_plus, grad_k2)

    def test_k3_plus_gradient_matches_k2(self, grad_tensors):
        """k3+ uses straight-through: forward from k3, gradient from k2."""
        logprob, ref_logprob = grad_tensors
        result = kl_penalty(logprob, ref_logprob, "k3+")
        loss = result.sum()
        loss.backward()
        grad_k3_plus = logprob.grad.clone()

        # Compare with pure k2 gradient
        logprob2 = logprob.detach().clone().requires_grad_(True)
        result2 = kl_penalty(logprob2, ref_logprob, "k2")
        result2.sum().backward()
        grad_k2 = logprob2.grad.clone()

        assert torch.allclose(grad_k3_plus, grad_k2)

    def test_k1_plus_gradient_differs_from_k1(self):
        """k1+ gradient should differ from plain k1 gradient (unless they happen to align)."""
        logprob = torch.tensor([[0.0, -1.0, -2.0]], requires_grad=True)
        ref_logprob = torch.tensor([[-0.5, -1.5, -3.0]])

        result_k1 = kl_penalty(logprob, ref_logprob, "k1")
        result_k1.sum().backward()
        grad_k1 = logprob.grad.clone()

        logprob2 = logprob.detach().clone().requires_grad_(True)
        result_k1p = kl_penalty(logprob2, ref_logprob, "k1+")
        result_k1p.sum().backward()
        grad_k1p = logprob2.grad.clone()

        # k1 gradient is constant (all ones), k2 gradient is (logprob - ref_logprob)
        # They only coincide when diff == 1, which is not the case for all elements here
        assert not torch.allclose(grad_k1, grad_k1p)

    def test_default_penalty_type(self):
        """Default penalty type should be low_var_kl."""
        logprob = torch.tensor([[0.0, -1.0]])
        ref_logprob = torch.tensor([[-0.5, -1.5]])
        result = kl_penalty(logprob, ref_logprob)
        expected = kl_penalty(logprob, ref_logprob, "low_var_kl")
        assert torch.allclose(result, expected)

    def test_k1_plus_straight_through_forward_is_k1_gradient_is_k2(self):
        """Comprehensive test: k1+ FORWARD value matches k1, GRADIENT matches k2."""
        logprob = torch.tensor([[0.5, -0.3, -1.2, 0.8]], requires_grad=True)
        ref_logprob = torch.tensor([[-0.2, -0.9, -0.5, 0.1]])

        # Forward
        result_k1_plus = kl_penalty(logprob, ref_logprob, "k1+")
        expected_k1_forward = kl_penalty_forward(logprob.detach(), ref_logprob, "k1")
        assert torch.allclose(result_k1_plus.detach(), expected_k1_forward), "k1+ forward should match k1"

        # Gradient
        result_k1_plus.sum().backward()
        grad_k1_plus = logprob.grad.clone()

        logprob2 = logprob.detach().clone().requires_grad_(True)
        result_k2 = kl_penalty(logprob2, ref_logprob, "k2")
        result_k2.sum().backward()
        grad_k2 = logprob2.grad.clone()

        assert torch.allclose(grad_k1_plus, grad_k2), "k1+ gradient should match k2"

    def test_gradient_correctness_analytical_k1(self):
        """k1 gradient: d/d(logprob) of (logprob - ref_logprob) = 1.0 for each element."""
        logprob = torch.tensor([[0.5, -1.0, 2.0]], requires_grad=True)
        ref_logprob = torch.tensor([[0.0, -0.5, 1.0]])
        result = kl_penalty(logprob, ref_logprob, "k1")
        result.sum().backward()
        expected_grad = torch.ones_like(logprob)
        assert torch.allclose(logprob.grad, expected_grad)

    def test_gradient_correctness_analytical_k2(self):
        """k2 gradient: d/d(logprob) of 0.5*(logprob - ref)^2 = (logprob - ref)."""
        logprob = torch.tensor([[0.5, -1.0, 2.0]], requires_grad=True)
        ref_logprob = torch.tensor([[0.0, -0.5, 1.0]])
        result = kl_penalty(logprob, ref_logprob, "k2")
        result.sum().backward()
        expected_grad = logprob.detach() - ref_logprob
        assert torch.allclose(logprob.grad, expected_grad)


# ===================================================================
#  compute_rewards_with_kl
# ===================================================================


class TestComputeRewardsWithKL:
    def test_basic_arithmetic(self):
        scores = torch.tensor([[1.0, 2.0, 3.0]])
        old_lp = torch.tensor([[-0.5, -1.0, -1.5]])
        ref_lp = torch.tensor([[-1.0, -1.5, -2.0]])
        kl_coef = 0.1
        result = compute_rewards_with_kl(scores, old_lp, ref_lp, kl_coef)
        # kl = old - ref = [0.5, 0.5, 0.5]
        # result = scores - 0.1 * kl = [1 - 0.05, 2 - 0.05, 3 - 0.05]
        expected = torch.tensor([[0.95, 1.95, 2.95]])
        assert torch.allclose(result, expected)

    def test_zero_kl_coef(self):
        scores = torch.tensor([[1.0, 2.0]])
        old_lp = torch.tensor([[-0.5, -1.0]])
        ref_lp = torch.tensor([[-1.0, -1.5]])
        result = compute_rewards_with_kl(scores, old_lp, ref_lp, kl_coef=0.0)
        assert torch.allclose(result, scores)

    def test_equal_logprobs_no_penalty(self):
        scores = torch.tensor([[5.0, 6.0]])
        lp = torch.tensor([[-1.0, -2.0]])
        result = compute_rewards_with_kl(scores, lp, lp, kl_coef=10.0)
        assert torch.allclose(result, scores)

    def test_shape_preserved(self):
        batch, seq = 4, 16
        scores = torch.randn(batch, seq)
        old_lp = torch.randn(batch, seq)
        ref_lp = torch.randn(batch, seq)
        result = compute_rewards_with_kl(scores, old_lp, ref_lp, kl_coef=0.01)
        assert result.shape == (batch, seq)

    def test_negative_kl_increases_reward(self):
        """When old_log_prob < ref_log_prob (policy closer to target), reward increases."""
        scores = torch.tensor([[0.0]])
        old_lp = torch.tensor([[-2.0]])
        ref_lp = torch.tensor([[-1.0]])
        # kl = -2 - (-1) = -1, reward = 0 - 0.1 * (-1) = 0.1
        result = compute_rewards_with_kl(scores, old_lp, ref_lp, kl_coef=0.1)
        assert result.item() == pytest.approx(0.1)


# ===================================================================
#  apply_kl_penalty
# ===================================================================


class TestApplyKLPenalty:
    @pytest.fixture
    def setup(self):
        batch, seq = 2, 4
        token_scores = torch.ones(batch, seq)
        old_lp = torch.tensor([[-0.5, -1.0, -1.5, -2.0], [-0.3, -0.8, -1.3, -1.8]])
        ref_lp = torch.tensor([[-1.0, -1.5, -2.0, -2.5], [-0.8, -1.3, -1.8, -2.3]])
        response_mask = torch.tensor([[1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0]])
        kl_ctrl = FixedKLController(kl_coef=0.1)
        return token_scores, old_lp, ref_lp, response_mask, kl_ctrl

    def test_returns_tuple(self, setup):
        token_scores, old_lp, ref_lp, mask, ctrl = setup
        result = apply_kl_penalty(token_scores, old_lp, ref_lp, mask, ctrl, "kl")
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_metrics_keys(self, setup):
        token_scores, old_lp, ref_lp, mask, ctrl = setup
        _, metrics = apply_kl_penalty(token_scores, old_lp, ref_lp, mask, ctrl, "kl")
        assert "actor/reward_kl_penalty" in metrics
        assert "actor/reward_kl_penalty_coeff" in metrics

    def test_metrics_coeff_is_beta(self, setup):
        token_scores, old_lp, ref_lp, mask, ctrl = setup
        _, metrics = apply_kl_penalty(token_scores, old_lp, ref_lp, mask, ctrl, "kl")
        assert metrics["actor/reward_kl_penalty_coeff"] == 0.1

    def test_output_shape(self, setup):
        token_scores, old_lp, ref_lp, mask, ctrl = setup
        rewards, _ = apply_kl_penalty(token_scores, old_lp, ref_lp, mask, ctrl, "kl")
        assert rewards.shape == token_scores.shape

    def test_mask_zeroes_kl_outside_response(self, setup):
        token_scores, old_lp, ref_lp, mask, ctrl = setup
        rewards, _ = apply_kl_penalty(token_scores, old_lp, ref_lp, mask, ctrl, "kl")
        # Where mask is 0, kl penalty should be 0, so reward == token_scores
        unmasked_positions = mask == 0
        assert torch.allclose(rewards[unmasked_positions], token_scores[unmasked_positions])

    def test_reward_computation_manual(self):
        """Manual computation check for kl type."""
        token_scores = torch.tensor([[2.0, 3.0]])
        old_lp = torch.tensor([[-1.0, -2.0]])
        ref_lp = torch.tensor([[-1.5, -2.5]])
        mask = torch.tensor([[1.0, 1.0]])
        beta = 0.2
        ctrl = FixedKLController(kl_coef=beta)

        rewards, metrics = apply_kl_penalty(token_scores, old_lp, ref_lp, mask, ctrl, "kl")

        # kl_penalty_forward("kl") = old_lp - ref_lp = [0.5, 0.5]
        # kld masked = [0.5, 0.5]
        # rewards = [2.0, 3.0] - 0.2 * [0.5, 0.5] = [1.9, 2.9]
        expected = torch.tensor([[1.9, 2.9]])
        assert torch.allclose(rewards, expected)

        # mean KL = mean of per-seq means: seq has 2 tokens each
        # seq0: (0.5+0.5)/2 = 0.5
        # mean([0.5]) = 0.5
        assert metrics["actor/reward_kl_penalty"] == pytest.approx(0.5, abs=1e-5)

    def test_adaptive_controller_updated(self):
        """Check that apply_kl_penalty updates adaptive controller."""
        token_scores = torch.tensor([[1.0, 1.0, 1.0]])
        old_lp = torch.tensor([[-0.5, -1.0, -1.5]])
        ref_lp = torch.tensor([[-1.0, -1.5, -2.0]])
        mask = torch.ones(1, 3)
        ctrl = AdaptiveKLController(init_kl_coef=0.1, target_kl=0.01, horizon=100)
        original = ctrl.value

        apply_kl_penalty(token_scores, old_lp, ref_lp, mask, ctrl, "kl")
        # kl = [0.5, 0.5, 0.5], mean_kl = 0.5 which is >> target 0.01
        # So value should increase
        assert ctrl.value > original

    def test_all_zero_mask_produces_zero_kl(self):
        """When response_mask is all zeros, KL penalty should be zero."""
        token_scores = torch.tensor([[1.0, 2.0]])
        old_lp = torch.tensor([[-0.5, -1.0]])
        ref_lp = torch.tensor([[-1.5, -2.0]])
        mask = torch.zeros(1, 2)
        ctrl = FixedKLController(kl_coef=1.0)

        rewards, metrics = apply_kl_penalty(token_scores, old_lp, ref_lp, mask, ctrl, "kl")
        # All masked, so kld * mask = 0 => rewards = token_scores
        assert torch.allclose(rewards, token_scores)
        # mean KL: sum is 0, lengths clamped to 1, so 0/1 = 0 per seq
        assert metrics["actor/reward_kl_penalty"] == pytest.approx(0.0, abs=1e-6)

    def test_numerical_stability_large_log_ratio(self):
        """Very large log ratio should not produce NaN/Inf in rewards."""
        token_scores = torch.tensor([[1.0, 2.0]])
        old_lp = torch.tensor([[100.0, -100.0]])
        ref_lp = torch.tensor([[-100.0, 100.0]])
        mask = torch.ones(1, 2)
        ctrl = FixedKLController(kl_coef=0.01)

        for ptype in ("kl", "abs", "mse", "low_var_kl", "k3"):
            rewards, metrics = apply_kl_penalty(token_scores, old_lp, ref_lp, mask, ctrl, ptype)
            assert torch.isfinite(rewards).all(), f"Non-finite reward for {ptype}"
            assert math.isfinite(metrics["actor/reward_kl_penalty"]), f"Non-finite metric for {ptype}"

    def test_different_penalty_types(self):
        """apply_kl_penalty works with various penalty types."""
        token_scores = torch.ones(2, 3)
        old_lp = torch.randn(2, 3)
        ref_lp = torch.randn(2, 3)
        mask = torch.ones(2, 3)
        for ptype in ("kl", "abs", "mse", "low_var_kl", "k3"):
            ctrl = FixedKLController(kl_coef=0.1)
            rewards, metrics = apply_kl_penalty(token_scores, old_lp, ref_lp, mask, ctrl, ptype)
            assert rewards.shape == token_scores.shape
            assert "actor/reward_kl_penalty" in metrics

    def test_mean_kl_per_sequence(self):
        """Mean KL should be computed per sequence then averaged across batch."""
        token_scores = torch.zeros(2, 4)
        # Construct known diff
        old_lp = torch.tensor([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
        ref_lp = torch.tensor([[-1.0, -1.0, -1.0, -1.0], [-2.0, -2.0, -2.0, -2.0]])
        # mask: seq0 has 4 valid, seq1 has 2 valid
        mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=torch.float)
        ctrl = FixedKLController(kl_coef=0.0)  # 0 so rewards aren't affected

        _, metrics = apply_kl_penalty(token_scores, old_lp, ref_lp, mask, ctrl, "kl")

        # kl_penalty("kl") = old - ref = [[1,1,1,1],[2,2,2,2]]
        # masked = [[1,1,1,1],[2,2,0,0]]
        # per-seq mean: seq0 = (1+1+1+1)/4 = 1.0, seq1 = (2+2+0+0)/2 = 2.0
        # batch mean = (1.0 + 2.0) / 2 = 1.5
        assert metrics["actor/reward_kl_penalty"] == pytest.approx(1.5, abs=1e-5)


# ===================================================================
#  Hardened edge cases
# ===================================================================


class TestKLPenaltyForwardEdgeCases:
    def test_unknown_penalty_type_raises(self):
        """Unknown KL penalty type should raise NotImplementedError."""
        logprob = torch.tensor([[0.0]])
        ref = torch.tensor([[0.0]])
        with pytest.raises(NotImplementedError, match="Unknown KL penalty"):
            kl_penalty_forward(logprob, ref, "bogus_type")

    def test_identical_logprobs_all_types_yield_zero(self):
        """When logprob == ref_logprob, all KL estimators should return 0."""
        logprob = torch.tensor([[-1.0, -2.0, -3.0]])
        ref = logprob.clone()
        for ptype in ("kl", "k1", "abs", "mse", "k2", "low_var_kl", "k3"):
            result = kl_penalty_forward(logprob, ref, ptype)
            assert torch.allclose(result, torch.zeros_like(result), atol=1e-5), (
                f"KL penalty type '{ptype}' should be 0 for identical logprobs, got {result}"
            )

    def test_kl_penalty_shape_preserved_batch(self):
        """All penalty types should preserve input shape for batched inputs."""
        logprob = torch.randn(4, 8)
        ref = torch.randn(4, 8)
        for ptype in ("kl", "abs", "mse", "low_var_kl"):
            result = kl_penalty_forward(logprob, ref, ptype)
            assert result.shape == (4, 8), f"Shape mismatch for {ptype}"


class TestKLPenaltyEdgeCases:
    def test_straight_through_k1_plus(self):
        """k1+ should use k2 for gradients but k1 for forward values."""
        logprob = torch.tensor([[0.0, -1.0]], requires_grad=True)
        ref = torch.tensor([[-0.5, -0.5]])
        result = kl_penalty(logprob, ref, "k1+")
        # Forward value should match k1
        k1_val = kl_penalty_forward(logprob.detach(), ref, "k1")
        assert torch.allclose(result.detach(), k1_val, atol=1e-5)

    def test_straight_through_k3_plus(self):
        """k3+ should use k2 for gradients but k3 for forward values."""
        logprob = torch.tensor([[0.0, -1.0]], requires_grad=True)
        ref = torch.tensor([[-0.5, -0.5]])
        result = kl_penalty(logprob, ref, "k3+")
        k3_val = kl_penalty_forward(logprob.detach(), ref, "k3")
        assert torch.allclose(result.detach(), k3_val, atol=1e-5)


class TestComputeRewardsWithKLEdgeCases:
    def test_zero_kl_coef(self):
        """With kl_coef=0, rewards should equal token_level_scores."""
        scores = torch.randn(2, 4)
        old_lp = torch.randn(2, 4)
        ref_lp = torch.randn(2, 4)
        result = compute_rewards_with_kl(scores, old_lp, ref_lp, kl_coef=0.0)
        assert torch.allclose(result, scores)

    def test_identical_policies(self):
        """When old_log_prob == ref_log_prob, KL=0 so rewards = scores."""
        scores = torch.randn(2, 4)
        lp = torch.randn(2, 4)
        result = compute_rewards_with_kl(scores, lp, lp.clone(), kl_coef=1.0)
        assert torch.allclose(result, scores)
