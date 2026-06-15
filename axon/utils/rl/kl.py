# Copyright 2025 Model AI Corp.
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# KL controllers/penalties adapted from verl core_algos.py (github.com/volcengine/verl), Apache-2.0.
"""
KL divergence loss utilities for PPO training.

Provides multiple KL estimators for policy regularization:
- k1/kl: Simple log ratio (biased but stable)
- k2/mse: Squared log ratio (unbiased gradients)
- k3/low_var_kl: Low variance approximation (default, recommended)
- abs: Absolute log ratio

Also includes adaptive and fixed KL controllers for dynamic coefficient adjustment.

Reference: http://joschu.net/blog/kl-approx.html
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch

__all__ = [
    "kl_penalty",
    "kl_penalty_forward",
    "compute_rewards_with_kl",
    "apply_kl_penalty",
    "AdaptiveKLController",
    "FixedKLController",
    "get_kl_controller",
    "KLPenaltyType",
]

# Supported KL penalty types
KLPenaltyType = Literal["kl", "k1", "abs", "mse", "k2", "low_var_kl", "k3", "k1+", "k3+"]


class AdaptiveKLController:
    """Adaptive KL controller that adjusts coefficient based on observed KL.

    Described in: https://arxiv.org/pdf/1909.08593.pdf

    The controller increases the KL coefficient when observed KL exceeds the target,
    and decreases it when observed KL is below target. This helps maintain a
    consistent level of policy divergence throughout training.

    Attributes:
        value: Current KL coefficient
        target: Target KL divergence
        horizon: Number of steps over which to adjust (controls adaptation speed)
    """

    def __init__(self, init_kl_coef: float, target_kl: float, horizon: int):
        """Initialize adaptive KL controller.

        Args:
            init_kl_coef: Initial KL coefficient
            target_kl: Target KL divergence to maintain
            horizon: Adaptation horizon (larger = slower adaptation)
        """
        self.value = init_kl_coef
        self.target = target_kl
        self.horizon = horizon

    def update(self, current_kl: float, n_steps: int) -> None:
        """Update the KL coefficient based on current KL divergence.

        Uses proportional control with clipped error to smoothly adjust
        the coefficient towards maintaining the target KL.

        Args:
            current_kl: Current observed KL divergence
            n_steps: Number of steps taken (for scaling the update)
        """
        target = self.target
        if target == 0:
            # Avoid division by zero; treat any non-zero KL as max positive error
            proportional_error = np.clip(np.sign(current_kl) * 0.2, -0.2, 0.2) if current_kl != 0 else 0.0
        else:
            proportional_error = np.clip(current_kl / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult


class FixedKLController:
    """Fixed KL controller that maintains a constant coefficient.

    Use this when you want a stable KL penalty throughout training
    without adaptive adjustment.

    Attributes:
        value: Fixed KL coefficient
    """

    def __init__(self, kl_coef: float):
        """Initialize fixed KL controller.

        Args:
            kl_coef: Fixed KL coefficient to use
        """
        self.value = kl_coef

    def update(self, current_kl: float, n_steps: int) -> None:
        """Update method for interface compatibility (no-op).

        Args:
            current_kl: Current KL divergence value (unused)
            n_steps: Number of steps taken (unused)
        """
        pass


def get_kl_controller(kl_ctrl) -> AdaptiveKLController | FixedKLController:
    """Factory function to create appropriate KL controller based on configuration.

    Args:
        kl_ctrl: Configuration object containing KL controller settings

    Returns:
        KL controller instance (FixedKLController or AdaptiveKLController)

    Raises:
        NotImplementedError: If controller type is not "fixed" or "adaptive"
        AssertionError: If adaptive controller horizon is not positive
    """
    if kl_ctrl.type == "fixed":
        return FixedKLController(kl_coef=kl_ctrl.kl_coef)
    elif kl_ctrl.type == "adaptive":
        assert kl_ctrl.horizon > 0, f"horizon must be larger than 0. Got {kl_ctrl.horizon}"
        return AdaptiveKLController(
            init_kl_coef=kl_ctrl.kl_coef,
            target_kl=kl_ctrl.target_kl,
            horizon=kl_ctrl.horizon,
        )
    else:
        raise NotImplementedError(f"Unknown KL controller type: {kl_ctrl.type}")


def kl_penalty(
    logprob: torch.Tensor,
    ref_logprob: torch.Tensor,
    kl_penalty_type: KLPenaltyType = "low_var_kl",
) -> torch.Tensor:
    """Compute KL divergence given logprob and ref_logprob.

    Optionally uses straight-through gradient trick to bind k2 on other
    KL penalty methods for unbiased KL gradient estimation.

    See more description in http://joschu.net/blog/kl-approx.html

    Args:
        logprob: Log probabilities from current policy (batch_size, seq_len)
        ref_logprob: Log probabilities from reference policy (batch_size, seq_len)
        kl_penalty_type: Type of KL penalty estimator. Options:
            - "kl", "k1": Simple log ratio
            - "abs": Absolute log ratio
            - "mse", "k2": Squared log ratio (unbiased gradients)
            - "low_var_kl", "k3": Low variance approximation (recommended)
            - "k1+", "k3+": With straight-through gradient correction

    Returns:
        KL estimate tensor of same shape as inputs
    """
    forward_score = kl_penalty_forward(logprob, ref_logprob, kl_penalty_type)

    # For basic types or mse/k2 (which already has unbiased gradients), return directly
    if not kl_penalty_type.endswith("+") or kl_penalty_type in ("mse", "k2"):
        return forward_score

    # Straight-through trick for unbiased gradient estimation:
    # The expectation of k1 and k3 estimators equals the expected KL value,
    # but their gradients are biased. The k2 estimator gives unbiased gradients.
    # We use k2 for backprop while keeping forward values from k1/k3.
    backward_score = 0.5 * (logprob - ref_logprob).square()

    return backward_score - backward_score.detach() + forward_score.detach()


def kl_penalty_forward(
    logprob: torch.Tensor,
    ref_logprob: torch.Tensor,
    kl_penalty_type: KLPenaltyType = "low_var_kl",
) -> torch.Tensor:
    """Compute KL divergence forward pass.

    Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py
    See more description in http://joschu.net/blog/kl-approx.html

    Args:
        logprob: Log probabilities from current policy
        ref_logprob: Log probabilities from reference policy
        kl_penalty_type: Type of KL penalty estimator

    Returns:
        KL estimate tensor

    Raises:
        NotImplementedError: If kl_penalty_type is not supported
    """
    # Strip '+' suffix for forward computation
    penalty_type = kl_penalty_type.rstrip("+")

    # K1 estimator: simple log ratio
    # E[log π/π_ref] = KL(π || π_ref)
    if penalty_type in ("kl", "k1"):
        return logprob - ref_logprob

    # Absolute difference
    if penalty_type == "abs":
        return (logprob - ref_logprob).abs()

    # K2 estimator: squared log ratio (MSE)
    # Provides unbiased gradient estimates
    if penalty_type in ("mse", "k2"):
        return 0.5 * (logprob - ref_logprob).square()

    # K3 estimator: low variance KL approximation (recommended)
    # Based on J. Schulman "Approximating KL Divergence", 2020
    # http://joschu.net/blog/kl-approx.html
    if penalty_type in ("low_var_kl", "k3"):
        kl = ref_logprob - logprob
        # Clamp for numerical stability
        kl = torch.clamp(kl, min=-20, max=20)
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1).contiguous()
        return torch.clamp(kld, min=-10, max=10)

    if penalty_type == "full":
        # Full KL requires per-token logits over vocabulary
        raise NotImplementedError("Full KL divergence requires logits, not just log probs")

    raise NotImplementedError(f"Unknown KL penalty type: {kl_penalty_type}")


def compute_rewards_with_kl(
    token_level_scores: torch.Tensor,
    old_log_prob: torch.Tensor,
    ref_log_prob: torch.Tensor,
    kl_coef: float,
) -> torch.Tensor:
    """Compute token-level rewards with KL penalty.

    Applies KL penalty as: r_t = score_t - kl_coef * (log π - log π_ref)

    This encourages the policy to stay close to the reference policy
    while maximizing the reward signal.

    Args:
        token_level_scores: Token-level reward scores (batch_size, seq_len)
        old_log_prob: Log probabilities from current/old policy (batch_size, seq_len)
        ref_log_prob: Log probabilities from reference policy (batch_size, seq_len)
        kl_coef: KL penalty coefficient (typically 0.001 - 0.1)

    Returns:
        Token-level rewards with KL penalty applied (batch_size, seq_len)
    """
    kl = old_log_prob - ref_log_prob
    return token_level_scores - kl * kl_coef


def apply_kl_penalty(
    token_level_scores: torch.Tensor,
    old_log_probs: torch.Tensor,
    ref_log_prob: torch.Tensor,
    response_mask: torch.Tensor,
    kl_ctrl: AdaptiveKLController | FixedKLController,
    kl_penalty_type: KLPenaltyType = "kl",
) -> tuple[torch.Tensor, dict[str, float]]:
    """Apply KL penalty to token-level rewards with controller update.

    Computes KL divergence between current and reference policy, applies it
    as a penalty to the reward signal, and updates the KL controller.

    Args:
        token_level_scores: Token-level reward scores (batch_size, seq_len)
        old_log_probs: Log probs from current/old policy (batch_size, seq_len)
        ref_log_prob: Log probs from reference policy (batch_size, seq_len)
        response_mask: Mask for valid response tokens (batch_size, seq_len)
        kl_ctrl: KL controller (fixed or adaptive) managing the coefficient
        kl_penalty_type: Type of KL divergence estimator

    Returns:
        tuple of (token_level_rewards, metrics dict with KL stats)
    """
    kld = kl_penalty(old_log_probs, ref_log_prob, kl_penalty_type=kl_penalty_type)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    # Mean KL per sequence, then mean across batch
    response_lengths = response_mask.sum(dim=-1).clamp(min=1)
    current_kl = (kld.sum(dim=-1) / response_lengths).mean().item()
    batch_size = token_level_scores.shape[0]
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)

    metrics = {
        "actor/reward_kl_penalty": current_kl,
        "actor/reward_kl_penalty_coeff": beta,
    }
    return token_level_rewards, metrics
