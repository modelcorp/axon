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
# Adapted from verl trainer/ppo/metric_utils.py (github.com/volcengine/verl), Apache-2.0.
"""
Data metrics for PPO training.
"""

from typing import Any

import numpy as np
import torch

from axon.protocol import DataProto


def compute_response_mask(data: DataProto):
    """Compute the attention mask for the response part of the sequence.

    This function extracts the response_mask from the data, which indicates
    which tokens are response tokens (as opposed to prompt tokens).

    Args:
        data (DataProto): The data containing batched model outputs and inputs.

    Returns:
        torch.Tensor: The attention mask for the response tokens.
    """
    return data.batch["response_mask"]


def _compute_response_info(batch: DataProto) -> dict[str, Any]:
    """
    Computes information about prompts and responses from a batch.

    This is an internal helper function that extracts masks and lengths for prompts and responses.

    Args:
        batch: A DataProto object containing batch data with input_ids, attention_mask, and response_mask.

    Returns:
        A dictionary containing:
            - response_mask: Mask for the response tokens
            - prompt_length: Tensor of prompt lengths for each item in the batch
            - response_length: Tensor of response lengths for each item in the batch
    """
    response_mask = batch.batch["response_mask"]
    attention_mask = batch.batch["attention_mask"]

    # Prompt tokens are those in the attention mask but NOT in the response mask
    prompt_mask = attention_mask.bool() & ~response_mask.bool()

    prompt_length = prompt_mask.sum(-1).float()
    response_length = response_mask.sum(-1).float()  # (batch_size,)

    return dict(
        response_mask=response_mask,
        prompt_length=prompt_length,
        response_length=response_length,
    )


def compute_data_metrics(batch: DataProto, use_critic: bool = True, include_detail: bool = False) -> dict[str, Any]:
    """
    Computes various metrics from a batch of data for PPO training.

    This function calculates metrics related to scores, rewards, advantages, returns, values,
    and sequence lengths from a batch of data. It provides statistical information (mean, max, min)
    for each metric category.

    Args:
        batch: A DataProto object containing batch data with token-level scores, rewards, advantages, etc.
        use_critic: Whether to include critic-specific metrics. Defaults to True.
        include_detail: If True, returns sufficient statistics for aggregation.
                       If False, returns computed metrics (backward compatible).

    Returns:
        A dictionary of metrics including:
            - critic/score/mean, max, min: Statistics about sequence scores
            - critic/rewards/mean, max, min: Statistics about sequence rewards
            - critic/advantages/mean, max, min: Statistics about advantages
            - critic/returns/mean, max, min: Statistics about returns
            - critic/values/mean, max, min: Statistics about critic values (if use_critic=True)
            - critic/vf_explained_var: Explained variance of the value function (if use_critic=True)
            - response_length/mean, max, min, clip_ratio: Statistics about response lengths
            - prompt_length/mean, max, min, clip_ratio: Statistics about prompt lengths
            - num_turns/mean, max, min: Statistics about the number of multi-turn conversations
    """
    if "is_last_step" in batch.non_tensor_batch:
        is_last_step = batch.non_tensor_batch["is_last_step"]
        last_step_indices = np.where(is_last_step == True)[0]  # noqa: E712
        batch = batch.select_idxs(last_step_indices)

    if "is_padding" in batch.non_tensor_batch:
        is_padding = batch.non_tensor_batch["is_padding"]
        valid_step_indices = np.where(is_padding == False)[0]  # noqa: E712
        batch = batch.select_idxs(valid_step_indices)

    if not batch:
        # empty batch
        return {}

    # Need to log only task step scores
    sequence_score = batch.batch["token_level_scores"].sum(-1)
    sequence_reward = batch.batch["token_level_rewards"].sum(-1)

    advantages = batch.batch["advantages"]
    returns = batch.batch["returns"]

    max_seq_length = batch.batch["input_ids"].shape[-1]

    response_mask = batch.batch["response_mask"].bool()

    response_info = _compute_response_info(batch)
    prompt_length = response_info["prompt_length"]
    response_length = response_info["response_length"]

    aborted_mask = (response_length == 0).bool()
    non_aborted_mask = ~aborted_mask

    non_aborted_sequence_score = sequence_score[non_aborted_mask]
    non_aborted_sequence_reward = sequence_reward[non_aborted_mask]

    if non_aborted_sequence_score.numel() == 0:
        raise ValueError("All samples are aborted (response_length == 0). Cannot compute data metrics.")

    score_mean = torch.mean(non_aborted_sequence_score).detach().item()
    score_max = torch.max(non_aborted_sequence_score).detach().item()
    score_min = torch.min(non_aborted_sequence_score).detach().item()

    reward_mean = torch.mean(non_aborted_sequence_reward).detach().item()
    reward_max = torch.max(non_aborted_sequence_reward).detach().item()
    reward_min = torch.min(non_aborted_sequence_reward).detach().item()

    valid_adv = torch.masked_select(advantages, response_mask)
    valid_returns = torch.masked_select(returns, response_mask)

    if use_critic:
        values = batch.batch["values"]
        valid_values = torch.masked_select(values, response_mask)
        return_diff_var = torch.var(valid_returns - valid_values)
        return_var = torch.var(valid_returns)

    # Aborted samples and non-aborted response length statistics
    # response_length_non_aborted/*: statistics computed on non-aborted samples only
    aborted_ratio = torch.mean(aborted_mask.float()).detach().item()

    non_aborted_response_length = response_length[non_aborted_mask]
    if non_aborted_response_length.numel() > 0:
        non_aborted_response_length_mean = torch.mean(non_aborted_response_length).detach().item()
        non_aborted_response_length_max = torch.max(non_aborted_response_length).detach().item()
        non_aborted_response_length_min = torch.min(non_aborted_response_length).detach().item()
        non_aborted_response_length_clip_ratio = (
            torch.mean(torch.eq(non_aborted_response_length, max_seq_length).float()).detach().item()
        )
    else:
        raise ValueError("All samples are aborted, this should not happen.")

    metrics = {
        # score
        "critic/score/mean": score_mean,
        "critic/score/max": score_max,
        "critic/score/min": score_min,
        # reward
        "critic/rewards/mean": reward_mean,
        "critic/rewards/max": reward_max,
        "critic/rewards/min": reward_min,
        # adv
        "critic/advantages/mean": torch.mean(valid_adv).detach().item(),
        "critic/advantages/max": torch.max(valid_adv).detach().item(),
        "critic/advantages/min": torch.min(valid_adv).detach().item(),
        # returns
        "critic/returns/mean": torch.mean(valid_returns).detach().item(),
        "critic/returns/max": torch.max(valid_returns).detach().item(),
        "critic/returns/min": torch.min(valid_returns).detach().item(),
        **(
            {
                # values
                "critic/values/mean": torch.mean(valid_values).detach().item(),
                "critic/values/max": torch.max(valid_values).detach().item(),
                "critic/values/min": torch.min(valid_values).detach().item(),
                # vf explained var
                "critic/vf_explained_var": (1.0 - return_diff_var / (return_var + 1e-5)).detach().item(),
            }
            if use_critic
            else {}
        ),
        # response length
        "response_length/mean": torch.mean(response_length).detach().item(),
        "response_length/max": torch.max(response_length).detach().item(),
        "response_length/min": torch.min(response_length).detach().item(),
        "response_length/clip_ratio": torch.mean(torch.eq(response_length, max_seq_length).float()).detach().item(),
        # response length (non-aborted only)
        # These statistics exclude aborted samples to avoid skew from zeros
        "response_length_non_aborted/mean": non_aborted_response_length_mean,
        "response_length_non_aborted/max": non_aborted_response_length_max,
        "response_length_non_aborted/min": non_aborted_response_length_min,
        "response_length_non_aborted/clip_ratio": non_aborted_response_length_clip_ratio,
        # aborted ratio
        # Fraction of samples whose response length is zero
        "response/aborted_ratio": aborted_ratio,
        # prompt length
        "prompt_length/mean": torch.mean(prompt_length).detach().item(),
        "prompt_length/max": torch.max(prompt_length).detach().item(),
        "prompt_length/min": torch.min(prompt_length).detach().item(),
        "prompt_length/clip_ratio": torch.mean(torch.eq(prompt_length, max_seq_length).float()).detach().item(),
    }

    # Add sufficient statistics for aggregation if requested
    if include_detail:
        metrics.update(
            {
                # Counts
                "_count_non_aborted": non_aborted_sequence_score.numel(),
                "_count_valid_tokens": valid_adv.numel(),
                "_count_all": len(batch),
                "_count_aborted": aborted_mask.sum().item(),
                "_count_non_aborted_response": non_aborted_mask.sum().item(),
                # Sums for means
                "_sum_score": torch.sum(non_aborted_sequence_score).detach().item(),
                "_sum_reward": torch.sum(non_aborted_sequence_reward).detach().item(),
                "_sum_adv": torch.sum(valid_adv).detach().item(),
                "_sum_returns": torch.sum(valid_returns).detach().item(),
                "_sum_response_length": torch.sum(response_length).detach().item(),
                "_sum_non_aborted_response_length": torch.sum(non_aborted_response_length).detach().item(),
                "_sum_prompt_length": torch.sum(prompt_length).detach().item(),
                # Counts for ratios
                "_count_clipped_response": torch.eq(response_length, max_seq_length).sum().item(),
                "_count_non_aborted_clipped": torch.eq(non_aborted_response_length, max_seq_length).sum().item(),
                "_count_clipped_prompt": torch.eq(prompt_length, max_seq_length).sum().item(),
            }
        )

        if use_critic:
            metrics.update(
                {
                    "_sum_values": torch.sum(valid_values).detach().item(),
                    # For variance calculation
                    "_sum_returns_squared": torch.sum(valid_returns**2).detach().item(),
                    "_sum_values_squared": torch.sum(valid_values**2).detach().item(),
                    "_sum_returns_times_values": torch.sum(valid_returns * valid_values).detach().item(),
                }
            )

    return metrics


def reduce_data_metrics(metrics_list: list[dict]) -> dict[str, Any]:
    """
    Aggregate data metrics from multiple workers.
    Expects metrics computed with include_detail=True.
    """
    if not metrics_list:
        return {}

    metrics_list = [m for m in metrics_list if m]
    if not metrics_list:
        return {}

    # Aggregate counts
    total_count_non_aborted = sum(m.get("_count_non_aborted", 0) for m in metrics_list)
    total_count_valid_tokens = sum(m.get("_count_valid_tokens", 0) for m in metrics_list)
    total_count_all = sum(m.get("_count_all", 0) for m in metrics_list)
    total_count_aborted = sum(m.get("_count_aborted", 0) for m in metrics_list)
    total_count_non_aborted_response = sum(m.get("_count_non_aborted_response", 0) for m in metrics_list)

    if total_count_all == 0:
        return {}

    # Global max/min
    aggregated = {
        "critic/score/max": max(m.get("critic/score/max", float("-inf")) for m in metrics_list),
        "critic/score/min": min(m.get("critic/score/min", float("inf")) for m in metrics_list),
        "critic/rewards/max": max(m.get("critic/rewards/max", float("-inf")) for m in metrics_list),
        "critic/rewards/min": min(m.get("critic/rewards/min", float("inf")) for m in metrics_list),
        "critic/advantages/max": max(m.get("critic/advantages/max", float("-inf")) for m in metrics_list),
        "critic/advantages/min": min(m.get("critic/advantages/min", float("inf")) for m in metrics_list),
        "critic/returns/max": max(m.get("critic/returns/max", float("-inf")) for m in metrics_list),
        "critic/returns/min": min(m.get("critic/returns/min", float("inf")) for m in metrics_list),
        "response_length/max": max(m.get("response_length/max", float("-inf")) for m in metrics_list),
        "response_length/min": min(m.get("response_length/min", float("inf")) for m in metrics_list),
        "response_length_non_aborted/max": max(
            m.get("response_length_non_aborted/max", float("-inf")) for m in metrics_list
        ),
        "response_length_non_aborted/min": min(
            m.get("response_length_non_aborted/min", float("inf")) for m in metrics_list
        ),
        "prompt_length/max": max(m.get("prompt_length/max", float("-inf")) for m in metrics_list),
        "prompt_length/min": min(m.get("prompt_length/min", float("inf")) for m in metrics_list),
    }

    # Global means using sufficient statistics
    if total_count_non_aborted > 0:
        aggregated["critic/score/mean"] = sum(m.get("_sum_score", 0) for m in metrics_list) / total_count_non_aborted
        aggregated["critic/rewards/mean"] = sum(m.get("_sum_reward", 0) for m in metrics_list) / total_count_non_aborted

    if total_count_valid_tokens > 0:
        aggregated["critic/advantages/mean"] = (
            sum(m.get("_sum_adv", 0) for m in metrics_list) / total_count_valid_tokens
        )
        aggregated["critic/returns/mean"] = (
            sum(m.get("_sum_returns", 0) for m in metrics_list) / total_count_valid_tokens
        )

    # Response length stats
    aggregated["response_length/mean"] = sum(m.get("_sum_response_length", 0) for m in metrics_list) / total_count_all
    aggregated["response_length/clip_ratio"] = (
        sum(m.get("_count_clipped_response", 0) for m in metrics_list) / total_count_all
    )
    aggregated["response/aborted_ratio"] = total_count_aborted / total_count_all

    # Non-aborted response length
    if total_count_non_aborted_response > 0:
        aggregated["response_length_non_aborted/mean"] = (
            sum(m.get("_sum_non_aborted_response_length", 0) for m in metrics_list) / total_count_non_aborted_response
        )
        aggregated["response_length_non_aborted/clip_ratio"] = (
            sum(m.get("_count_non_aborted_clipped", 0) for m in metrics_list) / total_count_non_aborted_response
        )

    # Prompt length stats
    aggregated["prompt_length/mean"] = sum(m.get("_sum_prompt_length", 0) for m in metrics_list) / total_count_all
    aggregated["prompt_length/clip_ratio"] = (
        sum(m.get("_count_clipped_prompt", 0) for m in metrics_list) / total_count_all
    )

    # Critic-specific metrics
    if any("_sum_values" in m for m in metrics_list) and total_count_valid_tokens > 0:
        aggregated["critic/values/max"] = max(m.get("critic/values/max", float("-inf")) for m in metrics_list)
        aggregated["critic/values/min"] = min(m.get("critic/values/min", float("inf")) for m in metrics_list)
        aggregated["critic/values/mean"] = sum(m.get("_sum_values", 0) for m in metrics_list) / total_count_valid_tokens

        # Explained variance
        returns_mean = aggregated["critic/returns/mean"]

        returns_var = (sum(m.get("_sum_returns_squared", 0) for m in metrics_list) / total_count_valid_tokens) - (
            returns_mean**2
        )
        returns_values_diff_var = (
            sum(m.get("_sum_returns_squared", 0) for m in metrics_list) / total_count_valid_tokens
            - 2 * sum(m.get("_sum_returns_times_values", 0) for m in metrics_list) / total_count_valid_tokens
            + sum(m.get("_sum_values_squared", 0) for m in metrics_list) / total_count_valid_tokens
        )
        aggregated["critic/vf_explained_var"] = 1.0 - returns_values_diff_var / (returns_var + 1e-5)

    # Multi-turn conversation
    if any("_sum_num_turns" in m for m in metrics_list):
        aggregated["num_turns/min"] = min(m.get("num_turns/min", float("inf")) for m in metrics_list)
        aggregated["num_turns/max"] = max(m.get("num_turns/max", float("-inf")) for m in metrics_list)
        aggregated["num_turns/mean"] = sum(m.get("_sum_num_turns", 0) for m in metrics_list) / total_count_all

    if any("_sum_tool_call_counts" in m for m in metrics_list):
        aggregated["tool_call_counts/min"] = min(m.get("tool_call_counts/min", float("inf")) for m in metrics_list)
        aggregated["tool_call_counts/max"] = max(m.get("tool_call_counts/max", float("-inf")) for m in metrics_list)
        aggregated["tool_call_counts/mean"] = (
            sum(m.get("_sum_tool_call_counts", 0) for m in metrics_list) / total_count_all
        )

    return aggregated
