# Copyright 2025 Model AI Corp.
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
# The chunked-GAE estimator is adapted from THUDM/slime (https://github.com/THUDM/slime), Apache License 2.0.
"""
Advantage estimation functions for reinforcement learning algorithms.

All functions have the uniform signature:
    fn(data: DataProto, config: DictConfig) -> tuple[torch.Tensor, torch.Tensor]

Each function extracts its own data from DataProto and config from DictConfig.
"""

from collections import defaultdict

import torch
import torch.nn.functional as F
from omegaconf import DictConfig

from axon.protocol import DataProto
from axon.trainer.algos.advantages.registry import AdvantageFn, register_advantage
from axon.utils.rl.advantage import (
    as_torch_index,
    group_mean_std,
    masked_whiten,
)


@register_advantage(AdvantageFn.GAE)
def gae_advantage_fn(
    data: DataProto,
    config: DictConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Generalized Advantage Estimation (GAE) advantages.

    Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py
    """
    token_level_rewards = data.batch["token_level_rewards"]
    values = data.batch["values"]
    response_mask = data.batch["response_mask"]
    gamma = config.get("gamma", 0.99)
    lam = config.get("lam", 0.95)

    with torch.no_grad():
        nextvalues = 0
        lastgaelam = 0
        advantages_reversed = []
        gen_len = token_level_rewards.shape[-1]

        for t in reversed(range(gen_len)):
            delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam_ = delta + gamma * lam * lastgaelam

            # skip values and TD-error on observation tokens
            nextvalues = values[:, t] * response_mask[:, t] + (1 - response_mask[:, t]) * nextvalues
            lastgaelam = lastgaelam_ * response_mask[:, t] + (1 - response_mask[:, t]) * lastgaelam

            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)

        returns = advantages + values
        advantages = masked_whiten(advantages, response_mask)
    return advantages, returns


@register_advantage(AdvantageFn.GRPO)
def grpo_advantage_fn(
    data: DataProto,
    config: DictConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Group Relative Policy Optimization (GRPO) advantages.

    For each group g, computes advantage as:
        a_i = (r_i - μ_g) / σ_g  (if norm_adv_by_std=True)
        a_i = r_i - μ_g          (if norm_adv_by_std=False)

    For single-sample groups (n=1), advantage is set to 0.
    """
    token_level_rewards = data.batch["token_level_rewards"]
    response_mask = data.batch["response_mask"]
    index = data.non_tensor_batch["uid"]
    epsilon = config.get("epsilon", 1e-6)
    norm_adv_by_std = config.get("norm_adv_by_std", True)

    with torch.no_grad():
        scores = token_level_rewards.sum(dim=-1)
        g = as_torch_index(index, device=scores.device)
        mean_g, std_g, count_g = group_mean_std(scores, g, eps=epsilon)
        if norm_adv_by_std:
            scalars = (scores - mean_g[g]) / (std_g[g] + epsilon)
        else:
            scalars = scores - mean_g[g]
        # Zero out singleton groups (can't compute meaningful advantage with n=1)
        scalars = scalars * (count_g[g] > 1)
        advantages = scalars.unsqueeze(-1) * response_mask
        return advantages, advantages


@register_advantage(AdvantageFn.RLOO)
@register_advantage(AdvantageFn.LOOP)
def rloo_advantage_fn(
    data: DataProto,
    config: DictConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute RLOO (Reinforcement Learning with Leave-One-Out) advantages.
    Also registered as LOOP (alias).

    For each sample i in group g, computes the leave-one-out advantage:
        a_i = (n * s_i - S_g) / (n - 1)

    For single-sample groups (n=1), advantage is set to 0.

    Reference: https://arxiv.org/abs/2402.14740
    """
    token_level_rewards = data.batch["token_level_rewards"]
    response_mask = data.batch["response_mask"]
    index = data.non_tensor_batch["uid"]

    with torch.no_grad():
        scores = token_level_rewards.sum(dim=-1)
        g = as_torch_index(index, device=scores.device)

        # Compute group counts and sums
        c = torch.bincount(g)[g].to(scores.dtype)  # count per sample's group
        group_sum = torch.bincount(g, weights=scores)[g]  # sum per sample's group

        # LOO advantage: (n * s_i - S_g) / (n - 1)
        # For n=1: set to 0 (can't compute LOO with single sample)
        adv = ((c * scores - group_sum) / (c - 1).clamp_min(1)) * (c > 1)

        advantages = adv.unsqueeze(-1) * response_mask
        return advantages, advantages


@register_advantage(AdvantageFn.REINFORCE_PLUS_PLUS)
def reinforce_plus_plus_advantage_fn(
    data: DataProto,
    config: DictConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute REINFORCE++ advantages with discounted returns.

    Based on https://arxiv.org/abs/2501.03262
    """
    token_level_rewards = data.batch["token_level_rewards"]
    response_mask = data.batch["response_mask"]
    gamma = config.get("gamma", 1.0)

    with torch.no_grad():
        returns = torch.zeros_like(token_level_rewards)
        running_return = 0

        for t in reversed(range(token_level_rewards.shape[1])):
            running_return = token_level_rewards[:, t] + gamma * running_return
            returns[:, t] = running_return
            # Reset after EOS
            running_return = running_return * response_mask[:, t]

        advantages = masked_whiten(returns, response_mask)
        advantages = advantages * response_mask

    return advantages, returns


@register_advantage(AdvantageFn.REINFORCE_PLUS_PLUS_BASELINE)
def reinforce_plus_plus_baseline_advantage_fn(
    data: DataProto,
    config: DictConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute REINFORCE++ with baseline advantages.

    Based on https://arxiv.org/abs/2501.03262

    Two-step normalization:
    1. Subtract GROUP mean (reward reshaping) - for n=1, baseline=0 so keeps original score
    2. Whiten GLOBALLY across the entire batch (stability)
    """
    token_level_rewards = data.batch["token_level_rewards"]
    response_mask = data.batch["response_mask"]
    index = data.non_tensor_batch["uid"]

    with torch.no_grad():
        scores = token_level_rewards.sum(dim=-1)
        g = as_torch_index(index, device=scores.device)

        # Compute group counts and sums
        c = torch.bincount(g)[g].to(scores.dtype)
        group_sum = torch.bincount(g, weights=scores)[g]
        group_mean = group_sum / c.clamp_min(1)

        # Subtract group mean baseline
        # For n=1: baseline = 0, so baselined = score - 0 = score
        # For n>1: baseline = group_mean
        baseline = torch.where(c > 1, group_mean, torch.zeros_like(scores))
        baselined = scores - baseline

        # Broadcast to token dimension and whiten globally
        response_length = token_level_rewards.shape[-1]
        scores_expanded = baselined.unsqueeze(-1).expand(-1, response_length) * response_mask
        scores_whitened = masked_whiten(scores_expanded, response_mask) * response_mask

    return scores_whitened, scores_whitened


@register_advantage(AdvantageFn.REMAX)
def remax_advantage_fn(
    data: DataProto,
    config: DictConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute ReMax advantages using greedy decoding baseline.

    Based on https://arxiv.org/abs/2310.10505
    """
    token_level_rewards = data.batch["token_level_rewards"]
    response_mask = data.batch["response_mask"]
    reward_baselines = data.batch["reward_baselines"]

    with torch.no_grad():
        # Compute cumulative returns from end to start
        # flip -> cumsum -> flip gives us sum from position t to end
        returns = (token_level_rewards * response_mask).flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1])
        advantages = returns - reward_baselines.unsqueeze(-1) * response_mask

    return advantages, returns


@register_advantage(AdvantageFn.OPO)
def opo_advantage_fn(
    data: DataProto,
    config: DictConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute OPO (Optimal Policy Optimization) advantages.

    Based on https://arxiv.org/pdf/2505.23585

    Uses a length-weighted baseline: baseline = sum(len_i * score_i) / sum(len_i)
    For single-sample groups (n=1), baseline = 0, so advantage = original score.
    """
    token_level_rewards = data.batch["token_level_rewards"]
    response_mask = data.batch["response_mask"]
    index = data.non_tensor_batch["uid"]

    with torch.no_grad():
        response_lengths = response_mask.sum(dim=-1)  # (batch_size,)
        scores = token_level_rewards.sum(dim=-1)  # (batch_size,)
        g = as_torch_index(index, device=scores.device)

        # Compute group counts
        c = torch.bincount(g)[g].to(scores.dtype)

        # Compute length-weighted baseline per group
        # baseline_g = sum(len_i * score_i for i in g) / sum(len_i for i in g)
        weighted_scores = response_lengths * scores
        group_weighted_sum = torch.bincount(g, weights=weighted_scores)[g]
        group_len_sum = torch.bincount(g, weights=response_lengths)[g]

        # For singleton groups (n=1), baseline = 0, so adv = score - 0 = score
        # For multi-sample groups, baseline = length-weighted average
        baseline = torch.where(c > 1, group_weighted_sum / group_len_sum.clamp_min(1), torch.zeros_like(scores))
        adv = scores - baseline

        advantages = adv.unsqueeze(-1) * response_mask
        return advantages, advantages


@register_advantage(AdvantageFn.GRPO_PASSK)
def grpo_passk_advantage_fn(
    data: DataProto,
    config: DictConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Pass@k advantages using GRPO-style formulation.

    Based on https://arxiv.org/abs/2503.19595

    Only the best response per group gets a non-zero advantage:
        advantage = r_max - r_second_max

    Requires at least 2 samples per group.
    """
    token_level_rewards = data.batch["token_level_rewards"]
    response_mask = data.batch["response_mask"]
    index = data.non_tensor_batch["uid"]
    epsilon = config.get("epsilon", 1e-6)
    norm_adv_by_std = config.get("norm_adv_by_std", True)

    scores = token_level_rewards.sum(dim=-1)  # (bs,)
    advantages = torch.zeros_like(scores)

    id2scores: dict = defaultdict(list)
    id2indices: dict = defaultdict(list)

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            idx = index[i]
            id2scores[idx].append(scores[i])
            id2indices[idx].append(i)

        for idx in id2scores:
            rewards = torch.stack(id2scores[idx])  # (k,)
            if rewards.numel() < 2:
                raise ValueError(
                    f"Pass@k requires at least 2 samples per group. Got {rewards.numel()} for group {idx}."
                )
            topk, topk_idx = torch.topk(rewards, 2)
            r_max, r_second_max = topk[0], topk[1]
            i_max = id2indices[idx][topk_idx[0].item()]
            advantage = r_max - r_second_max
            if norm_adv_by_std:
                std = torch.std(rewards)
                advantage = advantage / (std + epsilon)
            advantages[i_max] = advantage

    advantages = advantages.unsqueeze(-1) * response_mask
    return advantages, advantages


@register_advantage(AdvantageFn.GPG)
def gpg_advantage_fn(
    data: DataProto,
    config: DictConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute GPG (Group Policy Gradient) advantages.

    Computes advantage as: alpha * (score - group_mean) / f_norm
    where alpha = batch_size / count_nonzero(scores)

    For single-sample groups (n=1), uses mean of 0.
    """
    token_level_rewards = data.batch["token_level_rewards"]
    response_mask = data.batch["response_mask"]
    index = data.non_tensor_batch["uid"]
    f_norm = config.get("f_norm", 1.0)

    with torch.no_grad():
        scores = token_level_rewards.sum(dim=-1)
        g = as_torch_index(index, device=scores.device)

        bsz = scores.shape[0]
        m = torch.count_nonzero(scores)
        alpha = bsz / m.clamp(min=1).to(scores.dtype)

        # Compute group counts and means
        c = torch.bincount(g)[g].to(scores.dtype)
        group_sum = torch.bincount(g, weights=scores)[g]
        group_mean = torch.where(c > 1, group_sum / c, torch.zeros_like(group_sum))

        # GPG advantage: alpha * (score - group_mean) / f_norm
        adv = alpha * (scores - group_mean) / f_norm

        advantages = adv.unsqueeze(-1) * response_mask
        return advantages, advantages


def _chunked_gae_core(
    rewards: torch.Tensor,
    values: torch.Tensor,
    gamma: float,
    lam: float,
    chunk_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute GAE using a parallel prefix scan within chunks and recurrent propagation across chunks.

    This reduces the sequential dependency from O(T) to O(T / chunk_size) while keeping
    intra-chunk computation fully parallel via a matrix multiply.

    Adapted from slime (FlashLinearAttention-inspired chunked GAE).

    Args:
        rewards: [B, T] reward sequence.
        values:  [B, T] value predictions.  V_T is assumed zero.
        gamma: Discount factor.
        lam: GAE lambda.
        chunk_size: Chunk length for the parallel scan.

    Returns:
        advantages: [B, T]
        returns:    [B, T]  (advantages + values)
    """
    assert rewards.ndim == 2 and values.ndim == 2
    B, T = rewards.shape
    assert values.shape == (B, T)

    device = rewards.device
    dtype = rewards.dtype

    # delta_t = r_t + gamma * V_{t+1} - V_t   with V_T = 0
    next_values = torch.cat(
        [values[:, 1:], torch.zeros(B, 1, device=device, dtype=dtype)],
        dim=1,
    )
    deltas = rewards + gamma * next_values - values

    # Reformulate backward GAE as a forward scan on the reversed sequence:
    #   S[i] = delta[i] + w * S[i-1],   w = gamma * lam
    w = gamma * lam
    deltas_rev = torch.flip(deltas, dims=[1])  # [B, T]

    # Pad to a multiple of chunk_size
    if T % chunk_size != 0:
        pad = chunk_size - (T % chunk_size)
        deltas_rev = F.pad(deltas_rev, (0, pad))
    else:
        pad = 0

    _, T_pad = deltas_rev.shape
    n_chunks = T_pad // chunk_size

    deltas_chunks = deltas_rev.view(B, n_chunks, chunk_size)

    # Intra-chunk parallel scan kernel M
    # M[i, j] = w^(j-i) if j >= i, else 0
    idx = torch.arange(chunk_size, device=device)
    diff = idx[None, :] - idx[:, None]
    M = torch.zeros(chunk_size, chunk_size, device=device, dtype=dtype)
    mask = diff >= 0
    if w == 0.0:
        M[mask & (diff == 0)] = 1.0
    else:
        M[mask] = w ** diff[mask].to(dtype)

    # pow_vec[t] = w^(t+1), used to inject recurrent state s_prev
    if w == 0.0:
        pow_vec = torch.zeros(chunk_size, device=device, dtype=dtype)
    else:
        pow_vec = w ** torch.arange(1, chunk_size + 1, device=device, dtype=dtype)

    # Parallel local chunk computation (assuming initial state = 0)
    deltas_flat = deltas_chunks.reshape(B * n_chunks, chunk_size)
    S_local_flat = deltas_flat @ M
    S_local_chunks = S_local_flat.view(B, n_chunks, chunk_size)

    lengths = [chunk_size] * n_chunks
    if pad > 0:
        lengths[-1] = chunk_size - pad

    # Recurrent propagation between chunks
    S_rev = deltas_rev.new_zeros(B, T_pad)
    s_prev = torch.zeros(B, device=device, dtype=dtype)

    for c in range(n_chunks):
        Lc = lengths[c]
        start = c * chunk_size
        end = start + Lc

        S_local = S_local_chunks[:, c, :Lc]
        S_global = S_local + s_prev.unsqueeze(1) * pow_vec[:Lc]

        S_rev[:, start:end] = S_global
        s_prev = S_global[:, -1]

    # Remove padding and flip back
    if pad > 0:
        S_rev = S_rev[:, :T]

    advantages = torch.flip(S_rev, dims=[1])
    returns = advantages + values
    return advantages, returns


@register_advantage(AdvantageFn.CHUNKED_GAE)
def chunked_gae_advantage_fn(
    data: DataProto,
    config: DictConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute GAE advantages using a chunked parallel prefix scan.

    Reduces the sequential loop length from O(T) to O(T / chunk_size) while
    producing results equivalent to standard GAE.  Useful for very long
    sequences where the sequential backward pass in :func:`gae_advantage_fn`
    becomes a bottleneck.

    Handles arbitrary (including non-contiguous) response masks by gathering
    response-only tokens into a compressed sequence before running the scan.
    """
    token_level_rewards = data.batch["token_level_rewards"]
    values = data.batch["values"]
    response_mask = data.batch["response_mask"]
    gamma = config.get("gamma", 0.99)
    lam = config.get("lam", 0.95)
    chunk_size = config.get("chunk_size", 128)

    B, T = token_level_rewards.shape
    device = token_level_rewards.device
    dtype = token_level_rewards.dtype

    with torch.no_grad():
        # Gather response-only tokens into compressed sequences so that the
        # scan never propagates a discount factor across observation gaps.
        resp_counts = response_mask.sum(dim=1).long()  # [B]
        max_resp = resp_counts.max().item()

        if max_resp == 0:
            advantages = torch.zeros_like(token_level_rewards)
            return advantages, advantages

        compressed_rewards = torch.zeros(B, max_resp, device=device, dtype=dtype)
        compressed_values = torch.zeros(B, max_resp, device=device, dtype=dtype)
        resp_indices = []

        for b in range(B):
            idx = torch.where(response_mask[b] > 0)[0]
            n = idx.shape[0]
            compressed_rewards[b, :n] = token_level_rewards[b, idx]
            compressed_values[b, :n] = values[b, idx]
            resp_indices.append(idx)

        raw_adv, raw_ret = _chunked_gae_core(compressed_rewards, compressed_values, gamma, lam, chunk_size)

        # Scatter back to full sequence length
        advantages = torch.zeros(B, T, device=device, dtype=dtype)
        returns = torch.zeros(B, T, device=device, dtype=dtype)

        for b in range(B):
            idx = resp_indices[b]
            n = idx.shape[0]
            advantages[b, idx] = raw_adv[b, :n]
            returns[b, idx] = raw_ret[b, :n]

        advantages = masked_whiten(advantages, response_mask)
    return advantages, returns


@register_advantage(AdvantageFn.KIMI_K1_5)
def kimi_k1_5_advantage_fn(
    data: DataProto,
    config: DictConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Kimi K1.5 length-shaped advantage (arXiv:2501.12599 sec 2.3.3).

    Per group: lam_i = 0.5 - (len_i - min_len) / (max_len - min_len), with
    min/max over ALL k rollouts. length_reward = lam if correct else min(0, lam).
    Gated to 0 when max_len == min_len. Then GRPO advantage on shaped scores.

    Config:
        length_coef: scales length_reward; 1.0 = paper formula, 0 = disabled.
        length_coef_warmup_steps: hold coef at 0 for step < this.
        length_coef_ramp_steps: linear ramp from 0 to length_coef after warmup;
            0 = paper's hard switch.
        correct_threshold: score >= threshold counts as correct.
        norm_adv_by_std, epsilon: GRPO knobs.

    Reads data.meta_info["global_steps"] to resolve the schedule.
    """
    token_level_rewards = data.batch["token_level_rewards"]
    response_mask = data.batch["response_mask"]
    index = data.non_tensor_batch["uid"]

    epsilon = config.get("epsilon", 1e-6)
    norm_adv_by_std = config.get("norm_adv_by_std", True)
    length_coef_target = float(config.get("length_coef", 0.0))
    warmup_steps = int(config.get("length_coef_warmup_steps", 0))
    ramp_steps = int(config.get("length_coef_ramp_steps", 0))
    correct_threshold = float(config.get("correct_threshold", 1.0))

    global_step = 0
    if getattr(data, "meta_info", None):
        global_step = int(data.meta_info.get("global_steps", 0) or 0)

    if global_step < warmup_steps:
        length_coef = 0.0
    elif ramp_steps > 0 and global_step < warmup_steps + ramp_steps:
        length_coef = length_coef_target * (global_step - warmup_steps) / ramp_steps
    else:
        length_coef = length_coef_target

    with torch.no_grad():
        scores = token_level_rewards.sum(dim=-1)
        response_lengths = response_mask.sum(dim=-1).to(scores.dtype)
        g = as_torch_index(index, device=scores.device)

        if length_coef > 0.0 and g.numel() > 0:
            G = int(g.max().item()) + 1
            correct_mask = (scores >= correct_threshold).to(scores.dtype)

            # min/max over ALL rollouts (paper).
            len_min_per_group = torch.full((G,), float("inf"), device=scores.device, dtype=scores.dtype)
            len_max_per_group = torch.full((G,), float("-inf"), device=scores.device, dtype=scores.dtype)
            len_min_per_group.scatter_reduce_(0, g, response_lengths, reduce="amin", include_self=False)
            len_max_per_group.scatter_reduce_(0, g, response_lengths, reduce="amax", include_self=False)

            len_min = len_min_per_group[g]
            len_max = len_max_per_group[g]
            spread = len_max - len_min

            normalized = (response_lengths - len_min) / spread.clamp_min(1.0)
            lam = length_coef * (0.5 - normalized)

            # Asymmetric: incorrect rollouts capped at 0 (paper).
            length_reward = torch.where(correct_mask > 0, lam, torch.minimum(lam, torch.zeros_like(lam)))

            # Gate when max_len == min_len.
            length_reward = length_reward * (spread > 0).to(scores.dtype)

            shaped_scores = scores + length_reward
        else:
            shaped_scores = scores

        mean_g, std_g, count_g = group_mean_std(shaped_scores, g, eps=epsilon)
        if norm_adv_by_std:
            scalars = (shaped_scores - mean_g[g]) / (std_g[g] + epsilon)
        else:
            scalars = shaped_scores - mean_g[g]
        scalars = scalars * (count_g[g] > 1)

        advantages = scalars.unsqueeze(-1) * response_mask
        return advantages, advantages


@register_advantage(AdvantageFn.IDENTITY)
def identity_advantage_fn(
    data: DataProto,
    config: DictConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Identity advantage: use token_level_rewards directly as advantages.

    No normalization, no baseline subtraction. Useful when the program has
    already computed per-step advantages (e.g. stage-conditioned baselines
    in ParallelThinkerProgram).
    """
    token_level_rewards = data.batch["token_level_rewards"]
    response_mask = data.batch["response_mask"]

    with torch.no_grad():
        advantages = token_level_rewards * response_mask
    return advantages, advantages
