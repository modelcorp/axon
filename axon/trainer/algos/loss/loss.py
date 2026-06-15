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
# Policy/value loss functions adapted from verl core_algos.py (github.com/volcengine/verl), Apache-2.0.
"""
Policy loss functions for reinforcement learning algorithms.

All functions have the uniform signature:
    fn(data: DataProto, config: DictConfig) -> tuple[torch.Tensor, dict[str, Any]]

Each function extracts its own data from DataProto and config from DictConfig.
Config is the loss_args dict directly (not the full training config).
"""

from typing import Any

import torch
from omegaconf import DictConfig

from axon.protocol import DataProto
from axon.trainer.algos.loss.registry import LossFn, register_loss
from axon.trainer.algos.loss.utils import agg_loss, clip_by_value, masked_mean


def _valid_counts(batch):
    """Extract the valid-count tensors from a batch dict."""
    return dict(
        valid_token_count=batch.get("valid_token_count", None),
        valid_batch_size=batch.get("valid_batch_size", None),
        valid_program_count=batch.get("valid_program_count", None),
        per_row_token_count=batch.get("per_row_token_count", None),  # For context parallel.
    )


@register_loss(LossFn.PPO)
def ppo_loss_fn(
    data: DataProto,
    config: DictConfig,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the clipped policy objective and related metrics for PPO (vanilla).

    Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py
    """
    old_log_prob = data.batch["old_log_probs"]
    log_prob = data.batch["log_probs"]
    advantages = data.batch["advantages"]
    response_mask = data.batch["response_mask"]
    num_program_tokens = data.batch.get("num_program_tokens", None)
    num_program_steps = data.batch.get("num_program_steps", None)
    sampler_is_weights = data.batch.get("sampler_is_weights", None)

    clip_ratio = config.get("clip_ratio", 0.2)
    clip_ratio_low = config.get("clip_ratio_low", None)
    clip_ratio_high = config.get("clip_ratio_high", None)
    clip_ratio_c = config.get("clip_ratio_c", 3.0)

    cliprange_low = clip_ratio_low if clip_ratio_low is not None else clip_ratio
    cliprange_high = clip_ratio_high if clip_ratio_high is not None else clip_ratio

    assert clip_ratio_c > 1.0, f"clip_ratio_c must be > 1.0, got {clip_ratio_c}"

    negative_approx_kl = log_prob - old_log_prob
    # Clamp for numerical stability
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = masked_mean(-negative_approx_kl, response_mask)

    pg_losses1 = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high)
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)
    pg_clipfrac = masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)

    # Dual-clip PPO
    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    pg_clipfrac_lower = masked_mean(torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), response_mask)

    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)

    if sampler_is_weights is not None:
        pg_losses = pg_losses * sampler_is_weights

    pg_loss = agg_loss(
        loss_mat=pg_losses,
        loss_mask=response_mask,
        token_reduce=config.get("token_reduce", "sum"),
        batch_reduce=config.get("batch_reduce", "token-mean"),
        num_program_tokens=num_program_tokens,
        num_program_steps=num_program_steps,
        **_valid_counts(data.batch),
    )

    pg_metrics = {
        "pg_clipfrac": pg_clipfrac.detach().item(),
        "ppo_kl": ppo_kl.detach().item(),
        "pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
    }
    return pg_loss, pg_metrics


@register_loss(LossFn.GSPO)
def gspo_loss_fn(
    data: DataProto,
    config: DictConfig,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the clipped policy objective for GSPO (Group Sequence Policy Optimization).

    See https://arxiv.org/pdf/2507.18071 for details.
    """
    old_log_prob = data.batch["old_log_probs"]
    log_prob = data.batch["log_probs"]
    advantages = data.batch["advantages"]
    response_mask = data.batch["response_mask"]
    num_program_tokens = data.batch.get("num_program_tokens", None)
    num_program_steps = data.batch.get("num_program_steps", None)
    sampler_is_weights = data.batch.get("sampler_is_weights", None)

    clip_ratio = config.get("clip_ratio", 0.2)
    clip_ratio_low = config.get("clip_ratio_low", None)
    clip_ratio_high = config.get("clip_ratio_high", None)

    cliprange_low = clip_ratio_low if clip_ratio_low is not None else clip_ratio
    cliprange_high = clip_ratio_high if clip_ratio_high is not None else clip_ratio

    # Token-level log ratio
    negative_approx_kl = log_prob - old_log_prob

    # Compute sequence-level importance ratio:
    # s_i(θ) = (π_θ(y_i|x) / π_θold(y_i|x))^(1/|y_i|)
    seq_lengths = torch.sum(response_mask, dim=-1).clamp(min=1)
    negative_approx_kl_seq = torch.sum(negative_approx_kl * response_mask, dim=-1) / seq_lengths

    # Combined ratio at token level with stop-gradient
    log_seq_importance_ratio = log_prob - log_prob.detach() + negative_approx_kl_seq.detach().unsqueeze(-1)
    # Clamp for numerical stability
    log_seq_importance_ratio = torch.clamp(log_seq_importance_ratio, min=-20.0, max=20.0)
    seq_importance_ratio = torch.exp(log_seq_importance_ratio)

    pg_losses1 = -advantages * seq_importance_ratio
    pg_losses2 = -advantages * torch.clamp(seq_importance_ratio, 1 - cliprange_low, 1 + cliprange_high)
    pg_losses = torch.maximum(pg_losses1, pg_losses2)

    # Apply sampler correction weights if provided
    if sampler_is_weights is not None:
        pg_losses = pg_losses * sampler_is_weights

    # GSPO uses step-level mean aggregation
    pg_loss = agg_loss(
        loss_mat=pg_losses,
        loss_mask=response_mask,
        token_reduce=config.get("token_reduce", "mean"),
        batch_reduce=config.get("batch_reduce", "token-mean"),
        num_program_tokens=num_program_tokens,
        num_program_steps=num_program_steps,
        **_valid_counts(data.batch),
    )

    pg_clipfrac = masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)
    ppo_kl = masked_mean(-negative_approx_kl, response_mask)

    pg_metrics = {
        "pg_clipfrac": pg_clipfrac.detach().item(),
        "ppo_kl": ppo_kl.detach().item(),
        "pg_clipfrac_lower": 0.0,
    }
    return pg_loss, pg_metrics


@register_loss(LossFn.GPG)
def gpg_loss_fn(
    data: DataProto,
    config: DictConfig,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """GPG (Group Policy Gradient) loss.

    Adapted from https://github.com/AMAP-ML/GPG. The objective ``L = -log π(a|s) * A`` is
    identical to REINFORCE, so this delegates to :func:`reinforce_loss_fn` and pairs with
    the GPG advantage estimator (:data:`AdvantageFn.GPG`) which provides the GPG-specific
    advantage shaping.
    """
    return reinforce_loss_fn(data, config)


@register_loss(LossFn.CLIP_COV)
def clip_cov_loss_fn(
    data: DataProto,
    config: DictConfig,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute Clip-Cov loss for entropy-aware policy optimization.

    Adapted from https://github.com/PRIME-RL/Entropy-Mechanism-of-RL
    """
    old_log_prob = data.batch["old_log_probs"]
    log_prob = data.batch["log_probs"]
    advantages = data.batch["advantages"]
    response_mask = data.batch["response_mask"]
    num_program_tokens = data.batch.get("num_program_tokens", None)
    num_program_steps = data.batch.get("num_program_steps", None)
    sampler_is_weights = data.batch.get("sampler_is_weights", None)

    clip_ratio = config.get("clip_ratio", 0.2)
    clip_ratio_low = config.get("clip_ratio_low", None)
    clip_ratio_high = config.get("clip_ratio_high", None)
    clip_cov_ratio = config.get("clip_cov_ratio", 0.0002)
    clip_cov_lb = config.get("clip_cov_lb", 1.0)
    clip_cov_ub = config.get("clip_cov_ub", 5.0)

    cliprange_low = clip_ratio_low if clip_ratio_low is not None else clip_ratio
    cliprange_high = clip_ratio_high if clip_ratio_high is not None else clip_ratio

    assert clip_cov_ratio > 0, "clip_cov_ratio must be > 0"

    negative_approx_kl = log_prob - old_log_prob
    # Clamp for numerical stability
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = masked_mean(-negative_approx_kl, response_mask)

    corr = torch.ones_like(advantages)
    pg_losses1 = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high)
    clip_by_origin = (pg_losses2 > pg_losses1) & (response_mask > 0)

    # Compute covariance for selective clipping
    cov_all = (advantages - masked_mean(advantages, response_mask)) * (
        log_prob.detach() - masked_mean(log_prob.detach(), response_mask)
    )
    cov_all = cov_all.clone()
    cov_all[response_mask == 0] = -torch.inf
    cov_all[clip_by_origin] = -torch.inf

    clip_num = max(int(clip_cov_ratio * response_mask.sum().item()), 1)
    top_k_idx = (cov_all < clip_cov_ub) & (cov_all > clip_cov_lb) & (response_mask > 0)
    top_k_idx = torch.nonzero(top_k_idx)

    if len(top_k_idx) > 0:
        perm = torch.randperm(len(top_k_idx), device=top_k_idx.device)
        top_k_idx = top_k_idx[perm[: min(clip_num, len(top_k_idx))]]
    else:
        top_k_idx = torch.empty((0, 2), device=cov_all.device, dtype=torch.long)

    corr[top_k_idx[:, 0], top_k_idx[:, 1]] = 0

    pg_clipfrac = masked_mean((corr == 0).float(), response_mask)
    pg_losses = torch.maximum(pg_losses1, pg_losses2) * corr

    # Apply sampler correction weights if provided
    if sampler_is_weights is not None:
        pg_losses = pg_losses * sampler_is_weights

    pg_loss = agg_loss(
        loss_mat=pg_losses,
        loss_mask=response_mask,
        token_reduce=config.get("token_reduce", "mean"),
        batch_reduce=config.get("batch_reduce", "token-mean"),
        num_program_tokens=num_program_tokens,
        num_program_steps=num_program_steps,
        **_valid_counts(data.batch),
    )

    pg_metrics = {
        "pg_clipfrac": pg_clipfrac.detach().item(),
        "ppo_kl": ppo_kl.detach().item(),
    }
    return pg_loss, pg_metrics


@register_loss(LossFn.KL_COV)
def kl_cov_loss_fn(
    data: DataProto,
    config: DictConfig,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute KL-Cov loss for entropy-aware policy optimization.

    Adapted from https://github.com/PRIME-RL/Entropy-Mechanism-of-RL
    """
    old_log_prob = data.batch["old_log_probs"]
    log_prob = data.batch["log_probs"]
    advantages = data.batch["advantages"]
    response_mask = data.batch["response_mask"]
    num_program_tokens = data.batch.get("num_program_tokens", None)
    num_program_steps = data.batch.get("num_program_steps", None)
    sampler_is_weights = data.batch.get("sampler_is_weights", None)

    kl_cov_ratio = config.get("kl_cov_ratio", 0.0002)
    ppo_kl_coef = config.get("ppo_kl_coef", 1.0)

    assert kl_cov_ratio > 0, "kl_cov_ratio must be > 0"

    negative_approx_kl = log_prob - old_log_prob
    # Clamp for numerical stability
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    abs_kl = negative_approx_kl.abs()
    ratio = torch.exp(negative_approx_kl)
    ppo_kl_abs = masked_mean(abs_kl, response_mask)

    pg_losses1 = -advantages * ratio
    pg_losses_kl = -advantages * ratio + ppo_kl_coef * abs_kl
    pg_losses = pg_losses1.clone()

    all_valid = response_mask > 0
    all_valid_idx = torch.nonzero(all_valid.reshape(-1), as_tuple=True)[0]
    all_valid_adv = advantages[all_valid].detach().reshape(-1).cpu()
    all_valid_logp = log_prob[all_valid].detach().reshape(-1).cpu()

    k = min(kl_cov_ratio, len(all_valid_adv))

    if k != 0 and len(all_valid_adv) > 0:
        cov_lst_all = (all_valid_adv - all_valid_adv.mean()) * (all_valid_logp - all_valid_logp.mean())
        k_percent_nums = max(1, int(len(cov_lst_all) * kl_cov_ratio))
        large_cov_idxs = torch.topk(cov_lst_all, k_percent_nums, largest=True).indices

        if len(large_cov_idxs) != 0:
            large_cov_idxs = all_valid_idx[large_cov_idxs]
            pg_losses[large_cov_idxs // advantages.shape[1], large_cov_idxs % advantages.shape[1]] = pg_losses_kl[
                large_cov_idxs // advantages.shape[1], large_cov_idxs % advantages.shape[1]
            ]

    # Apply sampler correction weights if provided
    if sampler_is_weights is not None:
        pg_losses = pg_losses * sampler_is_weights

    pg_loss = agg_loss(
        loss_mat=pg_losses,
        loss_mask=response_mask,
        token_reduce=config.get("token_reduce", "sum"),
        batch_reduce=config.get("batch_reduce", "token-mean"),
        num_program_tokens=num_program_tokens,
        num_program_steps=num_program_steps,
        **_valid_counts(data.batch),
    )

    pg_metrics = {
        "ppo_kl": ppo_kl_abs.detach().item(),
    }
    return pg_loss, pg_metrics


@register_loss(LossFn.GEO_MEAN)
def geo_mean_loss_fn(
    data: DataProto,
    config: DictConfig,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute GMPO (Geometric Mean Policy Optimization) loss.

    Based on https://arxiv.org/abs/2507.20673
    """
    old_log_prob = data.batch["old_log_probs"]
    log_prob = data.batch["log_probs"]
    advantages = data.batch["advantages"]
    response_mask = data.batch["response_mask"]
    sampler_is_weights = data.batch.get("sampler_is_weights", None)

    clip_ratio = config.get("clip_ratio", 0.2)
    clip_ratio_low = config.get("clip_ratio_low", None)
    clip_ratio_high = config.get("clip_ratio_high", None)

    cliprange_low = clip_ratio_low if clip_ratio_low is not None else clip_ratio
    cliprange_high = clip_ratio_high if clip_ratio_high is not None else clip_ratio

    negative_approx_kl = log_prob - old_log_prob
    ppo_kl = masked_mean(-negative_approx_kl, response_mask)

    # Token-level clipping based on advantage sign
    sgn_advantage = torch.sign(advantages)
    negative_approx_kl_clamp = torch.clamp(negative_approx_kl, -cliprange_low, cliprange_high)
    negative_approx_kl_min = torch.min(sgn_advantage * negative_approx_kl, sgn_advantage * negative_approx_kl_clamp)
    negative_approx_kl_min = sgn_advantage * negative_approx_kl_min

    # Geometric-Mean Policy Optimization
    response_mask_sum = response_mask.sum(dim=-1)
    ratio = torch.exp((negative_approx_kl_min * response_mask).sum(dim=-1) / (response_mask_sum + 1e-8))
    # Sequence-level advantage
    advantage = (advantages * response_mask).sum(dim=-1) / (response_mask_sum + 1e-8)
    pg_losses = -advantage * ratio

    # Apply sampler correction weights if provided
    if sampler_is_weights is not None:
        # Aggregate token-level weights to sequence level using geometric mean
        seq_is_weights = torch.exp(
            (torch.log(sampler_is_weights + 1e-10) * response_mask).sum(dim=-1) / (response_mask_sum + 1e-8)
        )
        pg_losses = pg_losses * seq_is_weights

    valid_mask = response_mask_sum > 0
    pg_loss = pg_losses[valid_mask].mean() if valid_mask.any() else pg_losses.mean()

    # Compute clip fractions
    clipped = torch.ne(negative_approx_kl, negative_approx_kl_clamp)
    pg_clipfrac = masked_mean((clipped * (advantages > 0)).float(), response_mask)
    pg_clipfrac_lower = masked_mean((clipped * (advantages < 0)).float(), response_mask)

    pg_metrics = {
        "pg_clipfrac": pg_clipfrac.detach().item(),
        "ppo_kl": ppo_kl.detach().item(),
        "pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
    }
    return pg_loss, pg_metrics


@register_loss(LossFn.CISPO)
def cispo_loss_fn(
    data: DataProto,
    config: DictConfig,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute CISPO (Clipped IS-weight Policy Optimization) loss.

    Based on https://arxiv.org/abs/2506.13585 (MiniMax-M1, Section 3.1).

    Instead of clipping the policy ratio in the objective (as in PPO), CISPO clips
    the importance sampling weight and applies it with stop-gradient. Gradients flow
    only through log π_θ, preserving gradient contributions from all tokens.

    J_CISPO = E[ 1/(Σ|o_i|) Σ_i Σ_t sg(r̂_{i,t}) · Â_{i,t} · log π_θ(o_{i,t}) · M_{i,t} ]
    r̂_{i,t} = clip(r_{i,t}, 1 - ε_low, 1 + ε_high)

    The unified formulation (Eq 6-7) adds an optional token-wise mask M_{i,t} that drops
    tokens where the unclipped ratio exceeds the clip bounds in the advantage direction:
        M_{i,t} = 0 if Â > 0 and r > 1 + ε_high  (positive advantage, ratio too high)
        M_{i,t} = 0 if Â < 0 and r < 1 - ε_low   (negative advantage, ratio too low)
        M_{i,t} = 1 otherwise
    Enable with token_level_mask: true. Disabled by default (pure CISPO).
    """
    old_log_prob = data.batch["old_log_probs"]
    log_prob = data.batch["log_probs"]
    advantages = data.batch["advantages"]
    response_mask = data.batch["response_mask"]
    num_program_tokens = data.batch.get("num_program_tokens", None)
    num_program_steps = data.batch.get("num_program_steps", None)

    clip_ratio = config.get("clip_ratio", 0.2)
    clip_ratio_low = config.get("clip_ratio_low", None)
    clip_ratio_high = config.get("clip_ratio_high", None)
    token_level_mask = config.get("token_level_mask", False)

    # Paper: "we did not impose a lower bound on the IS weight by setting ε_low to a large value"
    cliprange_low = clip_ratio_low if clip_ratio_low is not None else 1e10
    cliprange_high = clip_ratio_high if clip_ratio_high is not None else clip_ratio

    # Token-level importance sampling ratio: r_{i,t} = π_θ / π_old
    negative_approx_kl = log_prob - old_log_prob
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = masked_mean(-negative_approx_kl, response_mask)

    # Clip the IS weight (stop-gradient): r̂ = clip(r, 1 - ε_low, 1 + ε_high)
    clipped_ratio = torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high).detach()

    # Loss = -sg(r̂) · A · log π_θ  (negated for minimization)
    pg_losses = -clipped_ratio * advantages * log_prob

    # Unified formulation (Eq 7): optional token-level mask M_{i,t}
    # Drops tokens where the unclipped ratio exceeds clip bounds in the advantage direction
    if token_level_mask:
        mask = torch.ones_like(ratio)
        mask[(advantages > 0) & (ratio > 1 + cliprange_high)] = 0.0
        mask[(advantages < 0) & (ratio < 1 - cliprange_low)] = 0.0
        pg_losses = pg_losses * mask

    pg_loss = agg_loss(
        loss_mat=pg_losses,
        loss_mask=response_mask,
        token_reduce=config.get("token_reduce", "sum"),
        batch_reduce=config.get("batch_reduce", "token-mean"),
        num_program_tokens=num_program_tokens,
        num_program_steps=num_program_steps,
        **_valid_counts(data.batch),
    )

    # Clip fraction metrics
    pg_clipfrac = masked_mean(
        ((ratio > 1 + cliprange_high) | (ratio < 1 - cliprange_low)).float(),
        response_mask,
    )

    pg_metrics = {
        "pg_clipfrac": pg_clipfrac.detach().item(),
        "ppo_kl": ppo_kl.detach().item(),
    }
    return pg_loss, pg_metrics


@register_loss(LossFn.REINFORCE)
def reinforce_loss_fn(
    data: DataProto,
    config: DictConfig,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute REINFORCE-style policy gradient loss.

    Standard policy gradient: L = -E[log π(a|s) * A]
    """
    old_log_prob = data.batch["old_log_probs"]
    log_prob = data.batch["log_probs"]
    advantages = data.batch["advantages"]
    response_mask = data.batch["response_mask"]
    num_program_tokens = data.batch.get("num_program_tokens", None)
    num_program_steps = data.batch.get("num_program_steps", None)
    sampler_is_weights = data.batch.get("sampler_is_weights", None)

    # Standard REINFORCE: L = -E[log π · A]
    # With IS: L = -E[w · log π · A]
    if sampler_is_weights is not None:
        pg_losses = -advantages * log_prob * sampler_is_weights
    else:
        pg_losses = -advantages * log_prob

    pg_loss = agg_loss(
        loss_mat=pg_losses,
        loss_mask=response_mask,
        token_reduce=config.get("token_reduce", "sum"),
        batch_reduce=config.get("batch_reduce", "token-mean"),
        num_program_tokens=num_program_tokens,
        num_program_steps=num_program_steps,
        **_valid_counts(data.batch),
    )

    # Compute KL divergence between current and old policy
    negative_approx_kl = log_prob - old_log_prob
    kl_divergence = masked_mean(-negative_approx_kl, response_mask)

    pg_metrics = {
        "ppo_kl": kl_divergence.detach().item(),
    }

    return pg_loss, pg_metrics


@register_loss(LossFn.VALUE)
def value_loss_fn(
    data: DataProto,
    config: DictConfig,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the clipped value-function loss for PPO.

    Takes the current value predictions (vpreds), the old baseline values, and
    the computed returns, then applies PPO-style value clipping.
    """
    vpreds = data.batch["vpreds"]
    values = data.batch["values"]
    returns = data.batch["returns"]
    response_mask = data.batch["response_mask"]
    num_program_tokens = data.batch.get("num_program_tokens", None)
    num_program_steps = data.batch.get("num_program_steps", None)

    cliprange_value = config.get("cliprange_value", 0.5)

    vpredclipped = clip_by_value(vpreds, values - cliprange_value, values + cliprange_value)
    vf_losses1 = (vpreds - returns) ** 2
    vf_losses2 = (vpredclipped - returns) ** 2
    clipped_vf_losses = torch.max(vf_losses1, vf_losses2)

    vf_loss = 0.5 * agg_loss(
        loss_mat=clipped_vf_losses,
        loss_mask=response_mask,
        token_reduce=config.get("token_reduce", "sum"),
        batch_reduce=config.get("batch_reduce", "token-mean"),
        num_program_tokens=num_program_tokens,
        num_program_steps=num_program_steps,
        **_valid_counts(data.batch),
    )

    vf_clipfrac = masked_mean(torch.gt(vf_losses2, vf_losses1).float(), response_mask)
    vpred_mean = masked_mean(vpreds, response_mask)

    vf_metrics = {
        "vf_loss": vf_loss.detach().item(),
        "vf_clipfrac": vf_clipfrac.detach().item(),
        "vpred_mean": vpred_mean.detach().item(),
    }
    return vf_loss, vf_metrics
