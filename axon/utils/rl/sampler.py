# Copyright 2025 Model AI Corp.
"""Sampler correction utilities for off-policy RL training.

Provides a complete pipeline to address off-policy issues in RL training:
1. Policy mismatch between sampler and training implementations (e.g. vLLM BF16 vs FSDP FP32)
2. Model update staleness (training on programs from older checkpoints)
3. General distribution shifts between data collection and training

Core capabilities:
- Importance sampling (IS) weights: token-level or sequence-level
- Rejection sampling (RS): token/sequence/geometric outlier filtering
- Catastrophic outlier veto: per-token veto for extreme outliers
- Off-policy diagnostics: KL, PPL, chi-squared metrics

Reference: https://richardli.xyz/rl-collapse
"""

from typing import Any

import torch

import axon.utils.torch.ops as axon_F
from axon.protocol import DataProto

# Safety bound to prevent numerical overflow/underflow when exponentiating
# exp(20) ≈ 485 million (upper limit for stable weights), exp(-20) ≈ 2e-9 (lower limit)
SAFETY_BOUND = 20.0


# =============================================================================
# Rejection Sampling
# =============================================================================


def compute_sampler_rejection_mask(
    log_ratio: torch.Tensor,
    response_mask: torch.Tensor,
    sampler_rs: str = "token",
    sampler_rs_threshold: float | None = None,
    sampler_rs_threshold_lower: float | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute rejection mask for outlier handling in off-policy RL training.

    Identifies and masks outlier tokens/sequences using precomputed log ratios
    (log(pi_train / pi_sampler)).

    Args:
        log_ratio: Log ratio of training to sampler policy, shape (batch_size, seq_length).
        response_mask: Binary mask for valid tokens, shape (batch_size, seq_length).
        sampler_rs: Aggregation level: "token", "sequence", or "geometric".
        sampler_rs_threshold: Upper threshold for valid IS weights.
        sampler_rs_threshold_lower: Lower threshold. Defaults to 1/upper if None.

    Returns:
        (modified_response_mask, metrics) where outliers are masked to 0.
    """
    valid_rs_levels = {"token", "sequence", "geometric"}
    if sampler_rs not in valid_rs_levels:
        raise ValueError(f"Invalid sampler_rs: {sampler_rs}. Must be one of {valid_rs_levels}.")
    if sampler_rs_threshold is None:
        raise ValueError("sampler_rs_threshold must be provided for rejection sampling.")

    upper_threshold = sampler_rs_threshold
    lower_threshold = sampler_rs_threshold_lower if sampler_rs_threshold_lower is not None else 1.0 / upper_threshold

    if sampler_rs == "token":
        log_ratio_for_metrics = log_ratio
        log_ratio_safe = torch.clamp(log_ratio, min=-SAFETY_BOUND, max=SAFETY_BOUND)
        sampler_is_weights = torch.exp(log_ratio_safe)
    elif sampler_rs == "sequence":
        log_ratio_sum = axon_F.masked_sum(log_ratio, response_mask, axis=-1).unsqueeze(-1)
        log_ratio_for_metrics = log_ratio_sum
        log_ratio_sum_safe = torch.clamp(log_ratio_sum, min=-SAFETY_BOUND, max=SAFETY_BOUND)
        sampler_is_weights = torch.exp(log_ratio_sum_safe).expand_as(log_ratio)
    elif sampler_rs == "geometric":
        log_ratio_mean = axon_F.masked_mean(log_ratio, response_mask, axis=-1).unsqueeze(-1)
        log_ratio_for_metrics = log_ratio_mean
        log_ratio_mean_safe = torch.clamp(log_ratio_mean, min=-SAFETY_BOUND, max=SAFETY_BOUND)
        sampler_is_weights = torch.exp(log_ratio_mean_safe).expand_as(log_ratio)
    else:
        raise ValueError(f"Unsupported sampler_rs: {sampler_rs}")

    mask = ((sampler_is_weights >= lower_threshold) & (sampler_is_weights <= upper_threshold)).float()

    metrics = _compute_rs_metrics(
        sampler_is_weights=sampler_is_weights,
        log_ratio_for_metrics=log_ratio_for_metrics,
        response_mask=response_mask,
        sampler_rs=sampler_rs,
        sampler_rs_threshold=upper_threshold,
        sampler_rs_threshold_lower=lower_threshold,
    )

    metrics["sampler_rs_masked_fraction"] = axon_F.masked_mean(1 - mask, response_mask).item()

    if sampler_rs == "token":
        seq_has_masked = axon_F.masked_sum(1 - mask, response_mask, axis=-1) > 0
        metrics["sampler_rs_seq_masked_fraction"] = seq_has_masked.float().mean().item()
    else:
        metrics["sampler_rs_seq_masked_fraction"] = (1 - mask[:, 0]).mean().item()

    modified_response_mask = response_mask * mask
    return modified_response_mask, metrics


def _compute_rs_metrics(
    sampler_is_weights: torch.Tensor,
    log_ratio_for_metrics: torch.Tensor,
    response_mask: torch.Tensor,
    sampler_rs: str,
    sampler_rs_threshold: float,
    sampler_rs_threshold_lower: float,
) -> dict[str, float]:
    """Compute metrics for rejection sampling weights."""
    if not response_mask.any():
        raise ValueError("response_mask must contain at least one valid token (1).")

    metrics: dict[str, float] = {}
    device = sampler_is_weights.device

    log_threshold_upper = torch.log(torch.tensor(sampler_rs_threshold, device=device))
    log_threshold_lower = torch.log(torch.tensor(sampler_rs_threshold_lower, device=device))

    if sampler_rs in ["sequence", "geometric"]:
        log_max = log_ratio_for_metrics.max()
        log_min = log_ratio_for_metrics.min()
        metrics["sampler_rs_max"] = torch.exp(torch.clamp(log_max, max=SAFETY_BOUND)).item()
        metrics["sampler_rs_min"] = torch.exp(log_min).item()
        metrics["sampler_rs_mean"] = axon_F.masked_mean(sampler_is_weights, response_mask).item()
        exceeds_upper = log_ratio_for_metrics > log_threshold_upper
        below_lower = log_ratio_for_metrics < log_threshold_lower
        metrics["sampler_rs_ratio_fraction_high"] = exceeds_upper.float().mean().item()
        metrics["sampler_rs_ratio_fraction_low"] = below_lower.float().mean().item()
    else:
        metrics["sampler_rs_mean"] = axon_F.masked_mean(sampler_is_weights, response_mask).item()
        metrics["sampler_rs_ratio_fraction_high"] = axon_F.masked_mean(
            (sampler_is_weights > sampler_rs_threshold).float(), response_mask
        ).item()
        metrics["sampler_rs_ratio_fraction_low"] = axon_F.masked_mean(
            (sampler_is_weights < sampler_rs_threshold_lower).float(), response_mask
        ).item()
        mask_bool = response_mask.bool()
        metrics["sampler_rs_max"] = sampler_is_weights.masked_fill(~mask_bool, float("-inf")).max().item()
        metrics["sampler_rs_min"] = sampler_is_weights.masked_fill(~mask_bool, float("inf")).min().item()

    mask_count = response_mask.sum()
    if mask_count > 1:
        weights_for_std = sampler_is_weights.clamp(min=sampler_rs_threshold_lower, max=sampler_rs_threshold)
        mean_clamped = axon_F.masked_mean(weights_for_std, response_mask)
        sampler_is_var = axon_F.masked_mean(weights_for_std.square(), response_mask) - mean_clamped.square()
        metrics["sampler_rs_std"] = torch.sqrt(torch.clamp(sampler_is_var, min=0.0)).item()
    else:
        metrics["sampler_rs_std"] = 0.0

    weights_for_ess = sampler_is_weights.clamp(min=sampler_rs_threshold_lower, max=sampler_rs_threshold)
    mean_for_ess = axon_F.masked_mean(weights_for_ess, response_mask)
    is_weights_normalized = weights_for_ess / (mean_for_ess + 1e-8)
    metrics["sampler_rs_eff_sample_size"] = (
        1.0 / axon_F.masked_mean(is_weights_normalized.square(), response_mask).item()
    )

    if sampler_is_weights.dim() > 1:
        seq_mean_weights = axon_F.masked_mean(sampler_is_weights, response_mask, axis=-1)
        metrics["sampler_rs_seq_mean"] = seq_mean_weights.mean().item()
        metrics["sampler_rs_seq_std"] = seq_mean_weights.std().item() if seq_mean_weights.numel() > 1 else 0.0
        metrics["sampler_rs_seq_max"] = seq_mean_weights.max().item()
        metrics["sampler_rs_seq_min"] = seq_mean_weights.min().item()
        seq_deviation = (seq_mean_weights - 1.0).abs()
        metrics["sampler_rs_seq_max_deviation"] = seq_deviation.max().item()
        metrics["sampler_rs_seq_fraction_high"] = (seq_mean_weights > sampler_rs_threshold).float().mean().item()
        metrics["sampler_rs_seq_fraction_low"] = (seq_mean_weights < sampler_rs_threshold_lower).float().mean().item()

    return metrics


# =============================================================================
# Importance Sampling Weights
# =============================================================================


def compute_sampler_correction_weights(
    log_ratio: torch.Tensor,
    response_mask: torch.Tensor,
    sampler_is: str = "token",
    sampler_is_threshold: float = 2.0,
    sampler_is_batch_normalize: bool = False,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute truncated importance sampling weights for off-policy correction.

    Args:
        log_ratio: Log ratio of training to sampler policy, shape (batch_size, seq_length).
        response_mask: Binary mask for valid tokens, shape (batch_size, seq_length).
        sampler_is: Aggregation level: "token" or "sequence".
        sampler_is_threshold: Upper truncation threshold. Default 2.0.
        sampler_is_batch_normalize: Normalize weights to mean=1.0 per batch.

    Returns:
        (sampler_is_weights, metrics) where weights are detached and truncated.
    """
    valid_is_levels = {"token", "sequence"}
    if sampler_is not in valid_is_levels:
        raise ValueError(f"Invalid sampler_is: {sampler_is}. Must be one of {valid_is_levels}.")
    if sampler_is_threshold <= 0:
        raise ValueError(f"sampler_is_threshold must be positive, got {sampler_is_threshold}.")

    if sampler_is == "token":
        log_ratio_for_metrics = log_ratio
        log_ratio_safe = torch.clamp(log_ratio, min=-SAFETY_BOUND, max=SAFETY_BOUND)
        sampler_is_weights = torch.exp(log_ratio_safe)
    elif sampler_is == "sequence":
        log_ratio_sum = axon_F.masked_sum(log_ratio, response_mask, axis=-1).unsqueeze(-1)
        log_ratio_for_metrics = log_ratio_sum
        log_ratio_sum_safe = torch.clamp(log_ratio_sum, min=-SAFETY_BOUND, max=SAFETY_BOUND)
        sampler_is_weights = torch.exp(log_ratio_sum_safe).expand_as(log_ratio)
    else:
        raise ValueError(f"Unsupported sampler_is: {sampler_is}")

    sampler_is_weights = sampler_is_weights * response_mask

    metrics = _compute_is_metrics(
        sampler_is_weights=sampler_is_weights,
        log_ratio_for_metrics=log_ratio_for_metrics,
        response_mask=response_mask,
        sampler_is=sampler_is,
        sampler_is_threshold=sampler_is_threshold,
    )

    # Truncated Importance Sampling
    sampler_is_weights = sampler_is_weights.clamp(max=sampler_is_threshold)
    # Detach: IS weights change the measure, not the objective
    sampler_is_weights = sampler_is_weights.detach()

    if sampler_is_batch_normalize:
        mask_float = response_mask.to(dtype=sampler_is_weights.dtype)
        if sampler_is == "token":
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                weights_mean = axon_F.distributed_masked_mean(sampler_is_weights, mask_float)
            else:
                weights_mean = axon_F.masked_mean(sampler_is_weights, response_mask)
        elif sampler_is == "sequence":
            seq_weights = axon_F.masked_mean(sampler_is_weights, response_mask, axis=-1)
            seq_mask = (response_mask.sum(dim=-1) > 0).to(dtype=sampler_is_weights.dtype)
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                weights_mean = axon_F.distributed_masked_mean(seq_weights, seq_mask)
            else:
                weights_mean = (seq_weights * seq_mask).sum() / seq_mask.sum().clamp_min(1e-8)
        else:
            raise ValueError(f"Unsupported sampler_is: {sampler_is}")

        if weights_mean > 1e-8:
            sampler_is_weights = sampler_is_weights / weights_mean
            metrics["sampler_is_batch_norm_factor"] = weights_mean.item()
        else:
            metrics["sampler_is_batch_norm_factor"] = 1.0

    return sampler_is_weights, metrics


def _compute_is_metrics(
    sampler_is_weights: torch.Tensor,
    log_ratio_for_metrics: torch.Tensor,
    response_mask: torch.Tensor,
    sampler_is: str,
    sampler_is_threshold: float,
) -> dict[str, float]:
    """Compute metrics for truncated importance sampling weights."""
    if not response_mask.any():
        raise ValueError("response_mask must contain at least one valid token (1).")

    metrics: dict[str, float] = {}
    device = sampler_is_weights.device
    sampler_is_threshold_lower = 1.0 / sampler_is_threshold

    log_threshold_upper = torch.log(torch.tensor(sampler_is_threshold, device=device))
    log_threshold_lower = torch.log(torch.tensor(sampler_is_threshold_lower, device=device))

    if sampler_is == "sequence":
        log_max = log_ratio_for_metrics.max()
        log_min = log_ratio_for_metrics.min()
        metrics["sampler_is_max"] = torch.exp(torch.clamp(log_max, max=SAFETY_BOUND)).item()
        metrics["sampler_is_min"] = torch.exp(log_min).item()
        metrics["sampler_is_mean"] = axon_F.masked_mean(sampler_is_weights, response_mask).item()
        exceeds_upper = log_ratio_for_metrics > log_threshold_upper
        below_lower = log_ratio_for_metrics < log_threshold_lower
        metrics["sampler_is_ratio_fraction_high"] = exceeds_upper.float().mean().item()
        metrics["sampler_is_ratio_fraction_low"] = below_lower.float().mean().item()
    else:
        metrics["sampler_is_mean"] = axon_F.masked_mean(sampler_is_weights, response_mask).item()
        metrics["sampler_is_ratio_fraction_high"] = axon_F.masked_mean(
            (sampler_is_weights > sampler_is_threshold).float(), response_mask
        ).item()
        metrics["sampler_is_ratio_fraction_low"] = axon_F.masked_mean(
            (sampler_is_weights < sampler_is_threshold_lower).float(), response_mask
        ).item()
        mask_bool = response_mask.bool()
        metrics["sampler_is_max"] = sampler_is_weights.masked_fill(~mask_bool, float("-inf")).max().item()
        metrics["sampler_is_min"] = sampler_is_weights.masked_fill(~mask_bool, float("inf")).min().item()

    mask_count = response_mask.sum()
    if mask_count > 1:
        weights_for_std = sampler_is_weights.clamp(min=sampler_is_threshold_lower, max=sampler_is_threshold)
        mean_clamped = axon_F.masked_mean(weights_for_std, response_mask)
        sampler_is_var = axon_F.masked_mean(weights_for_std.square(), response_mask) - mean_clamped.square()
        metrics["sampler_is_std"] = torch.sqrt(torch.clamp(sampler_is_var, min=0.0)).item()
    else:
        metrics["sampler_is_std"] = 0.0

    weights_for_ess = sampler_is_weights.clamp(min=sampler_is_threshold_lower, max=sampler_is_threshold)
    mean_for_ess = axon_F.masked_mean(weights_for_ess, response_mask)
    is_weights_normalized = weights_for_ess / (mean_for_ess + 1e-8)
    metrics["sampler_is_eff_sample_size"] = (
        1.0 / axon_F.masked_mean(is_weights_normalized.square(), response_mask).item()
    )

    if sampler_is_weights.dim() > 1:
        seq_mean_weights = axon_F.masked_mean(sampler_is_weights, response_mask, axis=-1)
        metrics["sampler_is_seq_mean"] = seq_mean_weights.mean().item()
        metrics["sampler_is_seq_std"] = seq_mean_weights.std().item() if seq_mean_weights.numel() > 1 else 0.0
        metrics["sampler_is_seq_max"] = seq_mean_weights.max().item()
        metrics["sampler_is_seq_min"] = seq_mean_weights.min().item()
        seq_deviation = (seq_mean_weights - 1.0).abs()
        metrics["sampler_is_seq_max_deviation"] = seq_deviation.max().item()
        metrics["sampler_is_seq_fraction_high"] = (seq_mean_weights > sampler_is_threshold).float().mean().item()
        metrics["sampler_is_seq_fraction_low"] = (seq_mean_weights < sampler_is_threshold_lower).float().mean().item()

    return metrics


# =============================================================================
# Unified Interface
# =============================================================================


def compute_sampler_correction_and_rejection_mask(
    old_log_prob: torch.Tensor,
    sampler_log_prob: torch.Tensor,
    response_mask: torch.Tensor,
    sampler_is: str | None = None,
    sampler_is_threshold: float | None = 2.0,
    sampler_rs: str | None = None,
    sampler_rs_threshold: float | None = 2.0,
    sampler_rs_threshold_lower: float | None = None,
    sampler_token_veto_threshold: float | None = None,
    sampler_is_batch_normalize: bool = False,
) -> tuple[DataProto | None, torch.Tensor, dict[str, float]]:
    """Unified interface for computing IS weights and rejection masks.

    Combines IS weight calculation, rejection sampling, and per-token veto
    into a single pipeline.

    Args:
        old_log_prob: Log probs from training policy, shape (batch_size, seq_length).
        sampler_log_prob: Log probs from sampler policy, shape (batch_size, seq_length).
        response_mask: Binary mask for valid tokens, shape (batch_size, seq_length).
        sampler_is: IS aggregation level ("token"/"sequence"), or None to disable.
        sampler_is_threshold: Upper truncation threshold for IS weights.
        sampler_rs: RS aggregation level ("token"/"sequence"/"geometric"), or None to disable.
        sampler_rs_threshold: Upper threshold for rejection sampling.
        sampler_rs_threshold_lower: Lower RS threshold. Defaults to 1/upper.
        sampler_token_veto_threshold: Min token-level IS weight before sequence veto.
        sampler_is_batch_normalize: Normalize IS weights to mean=1.0.

    Returns:
        (sampler_is_weights_proto, modified_response_mask, metrics) where
        sampler_is_weights_proto is a DataProto (or None), and metrics are
        prefixed with "sampler_corr/".
    """
    if not response_mask.any():
        # All-padding micro-batch (can happen with dynamic batching). Return no-op.
        return None, response_mask, {}
    if old_log_prob.shape != sampler_log_prob.shape:
        raise ValueError(
            f"old_log_prob shape {old_log_prob.shape} does not match sampler_log_prob shape {sampler_log_prob.shape}."
        )
    if old_log_prob.shape != response_mask.shape:
        raise ValueError(
            f"log_prob shape {old_log_prob.shape} does not match response_mask shape {response_mask.shape}."
        )

    log_ratio = old_log_prob - sampler_log_prob
    device = log_ratio.device
    metrics: dict[str, float] = {}

    # IS weights
    sampler_is_weights: torch.Tensor | None = None
    if sampler_is is not None and sampler_is_threshold is not None:
        sampler_is_weights, is_metrics = compute_sampler_correction_weights(
            log_ratio=log_ratio,
            response_mask=response_mask,
            sampler_is=sampler_is,
            sampler_is_threshold=sampler_is_threshold,
            sampler_is_batch_normalize=sampler_is_batch_normalize,
        )
        metrics.update(is_metrics)

    # Rejection mask
    modified_response_mask = response_mask.clone()
    if sampler_rs is not None:
        if sampler_rs_threshold is None:
            raise ValueError(
                "sampler_rs_threshold must be explicitly provided when sampler_rs is enabled. "
                "Set sampler_rs_threshold to the desired threshold value."
            )
        modified_response_mask, rs_metrics = compute_sampler_rejection_mask(
            log_ratio=log_ratio,
            response_mask=response_mask,
            sampler_rs=sampler_rs,
            sampler_rs_threshold=sampler_rs_threshold,
            sampler_rs_threshold_lower=sampler_rs_threshold_lower,
        )
        metrics.update(rs_metrics)

    # Per-token veto
    if sampler_token_veto_threshold is not None:
        if sampler_token_veto_threshold <= 0:
            raise ValueError(f"sampler_token_veto_threshold must be positive, got {sampler_token_veto_threshold}.")

        log_veto_threshold = torch.log(torch.tensor(sampler_token_veto_threshold, device=device))
        catastrophic_tokens = (log_ratio < log_veto_threshold) & response_mask.bool()
        has_catastrophic = catastrophic_tokens.any(dim=-1, keepdim=True)
        veto_mask = (~has_catastrophic).float()

        metrics["sampler_is_veto_fraction"] = has_catastrophic.float().mean().item()
        metrics["sampler_is_catastrophic_token_fraction"] = axon_F.masked_mean(
            catastrophic_tokens.float(), response_mask
        ).item()

        modified_response_mask = modified_response_mask * veto_mask
    else:
        metrics["sampler_is_veto_fraction"] = 0.0
        metrics["sampler_is_catastrophic_token_fraction"] = 0.0

    # Prefix all metrics
    metrics_scalar: dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(value, torch.Tensor):
            metrics_scalar[f"sampler_corr/{key}"] = value.item()
        else:
            metrics_scalar[f"sampler_corr/{key}"] = value

    # Wrap IS weights in DataProto
    sampler_is_weights_proto: DataProto | None = None
    if sampler_is_weights is not None:
        sampler_is_weights_proto = DataProto.from_dict(tensors={"sampler_is_weights": sampler_is_weights})

    return sampler_is_weights_proto, modified_response_mask, metrics_scalar


# =============================================================================
# Off-Policy Diagnostics
# =============================================================================


def compute_offpolicy_metrics(
    old_log_prob: torch.Tensor,
    sampler_log_prob: torch.Tensor | None,
    response_mask: torch.Tensor,
) -> dict[str, Any]:
    """Compute off-policy diagnostic metrics (KL, PPL, chi-squared).

    Args:
        old_log_prob: Log probs from training policy, shape (batch_size, seq_length).
        sampler_log_prob: Log probs from sampler policy, shape (batch_size, seq_length).
        response_mask: Mask for valid tokens, shape (batch_size, seq_length).

    Returns:
        Dictionary of off-policy metrics (without prefix).
    """
    if not response_mask.any():
        return {}

    metrics: dict[str, Any] = {}

    # Training policy perplexity
    mean_log_prob_training = axon_F.masked_mean(old_log_prob, response_mask, axis=-1)
    training_ppl = torch.exp(-mean_log_prob_training).mean()
    metrics["training_ppl"] = training_ppl.detach().item()
    metrics["training_log_ppl"] = (-mean_log_prob_training).mean().detach().item()

    if sampler_log_prob is not None:
        # KL(pi_sampler || pi_training)
        metrics["kl"] = axon_F.masked_mean(sampler_log_prob - old_log_prob, response_mask).detach().item()

        # K3 KL estimator (more stable for small KL)
        log_ratio = old_log_prob - sampler_log_prob
        k3_kl_matrix = torch.exp(log_ratio) - log_ratio - 1
        metrics["k3_kl"] = axon_F.masked_mean(k3_kl_matrix, response_mask).detach().item()

        # Sampler policy perplexity
        mean_log_prob_sampler = axon_F.masked_mean(sampler_log_prob, response_mask, axis=-1)
        sampler_ppl = torch.exp(-mean_log_prob_sampler).mean()
        metrics["sampler_ppl"] = sampler_ppl.detach().item()
        metrics["sampler_log_ppl"] = (-mean_log_prob_sampler).mean().detach().item()

        # Log PPL difference
        log_ppl_diff = mean_log_prob_sampler - mean_log_prob_training
        metrics["log_ppl_diff"] = log_ppl_diff.mean().detach().item()
        metrics["log_ppl_abs_diff"] = log_ppl_diff.abs().mean().detach().item()
        metrics["log_ppl_diff_max"] = log_ppl_diff.max().detach().item()
        metrics["log_ppl_diff_min"] = log_ppl_diff.min().detach().item()

        # PPL ratio
        ppl_ratio = torch.exp(log_ppl_diff).mean()
        metrics["ppl_ratio"] = ppl_ratio.detach().item()

        # Chi-squared divergence
        log_ratio_safe = torch.clamp(log_ratio, min=-SAFETY_BOUND, max=SAFETY_BOUND)
        rho_token = torch.exp(log_ratio_safe)
        chi2_token = axon_F.masked_mean(rho_token.square(), response_mask) - 1.0
        metrics["chi2_token"] = chi2_token.detach().item()

        log_ratio_sum = axon_F.masked_sum(log_ratio, response_mask, axis=-1)
        log_ratio_sum_safe = torch.clamp(log_ratio_sum, min=-SAFETY_BOUND, max=SAFETY_BOUND)
        rho_squared_seq = torch.exp(2.0 * log_ratio_sum_safe)
        chi2_seq = rho_squared_seq.mean() - 1.0
        metrics["chi2_seq"] = chi2_seq.detach().item()

    return metrics


# =============================================================================
# Batch-Level Helpers
# =============================================================================


def compute_sampler_correction_and_add_to_batch(
    batch: DataProto,
    config: dict[str, Any],
) -> tuple[DataProto, dict]:
    """Compute sampler correction weights and apply rejection sampling to a batch.

    Computes IS weights and/or rejection masks, then updates the batch.
    Always updates response_mask; conditionally adds sampler_is_weights.

    Args:
        batch: DataProto with old_log_probs, sampler_log_probs, response_mask.
        config: Config dict (typically loss_args) containing sampler correction
            keys directly: sampler_is, sampler_is_threshold, sampler_rs, etc.

    Returns:
        (updated_batch, metrics) where metrics are prefixed with "sampler_corr/".
    """
    with torch.no_grad():
        sampler_is_weights, modified_response_mask, sampler_corr_metrics = (
            compute_sampler_correction_and_rejection_mask(
                old_log_prob=batch.batch["old_log_probs"],
                sampler_log_prob=batch.batch["sampler_log_probs"],
                response_mask=batch.batch["response_mask"],
                sampler_is=config.get("sampler_is", None),
                sampler_is_threshold=config.get("sampler_is_threshold", 2.0),
                sampler_rs=config.get("sampler_rs", None),
                sampler_rs_threshold=config.get("sampler_rs_threshold", None),
                sampler_rs_threshold_lower=config.get("sampler_rs_threshold_lower", None),
                sampler_token_veto_threshold=config.get("sampler_token_veto_threshold", None),
                sampler_is_batch_normalize=config.get("sampler_is_batch_normalize", False),
            )
        )
    batch.batch["response_mask"] = modified_response_mask

    if sampler_is_weights is not None:
        batch = batch.union(sampler_is_weights)

    return batch, sampler_corr_metrics


def compute_sampler_corr_metrics_from_logprobs(
    log_prob: torch.Tensor,
    sampler_log_prob: torch.Tensor,
    response_mask: torch.Tensor,
) -> dict[str, float]:
    """Compute sampler correction metrics from current vs sampler log probs.

    Used in the actor to track the evolving off-policy gap as pi_theta
    updates during mini-batch training.

    Args:
        log_prob: Current policy log probs, shape (batch_size, seq_length).
        sampler_log_prob: Sampler policy log probs, shape (batch_size, seq_length).
        response_mask: Valid token mask, shape (batch_size, seq_length).

    Returns:
        Dictionary of metrics with "sampler_corr/" prefix.
    """
    offpolicy_metrics = compute_offpolicy_metrics(
        old_log_prob=log_prob,
        sampler_log_prob=sampler_log_prob,
        response_mask=response_mask,
    )

    metrics_with_prefix = {}
    for key, value in offpolicy_metrics.items():
        if isinstance(value, torch.Tensor):
            metrics_with_prefix[f"sampler_corr/{key}"] = value.item()
        else:
            metrics_with_prefix[f"sampler_corr/{key}"] = value

    return metrics_with_prefix
