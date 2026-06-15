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

"""Sampler-actor log-prob mismatch metrics.

Computes how much the actor's recomputed log-probs differ from the sampler's
original log-probs. This is a key diagnostic for PPO: large mismatches
indicate stale rollouts or numerical issues.

Design: always returns **sufficient statistics** (sums, sum-of-squares, counts)
so that per-worker results can be aggregated exactly via
``aggregate_trainer_sampler_metrics``. When running on a single process
(controller-resident), just wrap the result in a list and aggregate.
"""

import torch

from axon.protocol import DataProto


def _to_scalar(t: torch.Tensor) -> float:
    return t.detach().item()


def compute_trainer_sampler_mismatch_metrics(batch: DataProto) -> dict:
    """Compute sufficient statistics for sampler-actor log-prob mismatch.

    Always returns sufficient statistics that can be aggregated across
    workers. Call ``aggregate_trainer_sampler_metrics([result])`` to get
    human-readable metrics (works for both single-worker and multi-worker).

    Args:
        batch: DataProto with ``old_log_probs``, ``sampler_log_probs``,
            and ``response_mask`` in its batch dict.

    Returns:
        Dict of sufficient statistics (prefixed with ``_``) plus per-shard
        min/max values for global min/max aggregation.
    """
    mask = batch.batch["response_mask"].bool()

    actor_logprobs = batch.batch["old_log_probs"]
    sampler_logprobs = batch.batch["sampler_log_probs"]

    actor_probs, sampler_probs = actor_logprobs.exp(), sampler_logprobs.exp()

    probs_diff = torch.abs(actor_probs - sampler_probs)[mask]
    logprobs_diff = torch.abs(actor_logprobs - sampler_logprobs)[mask]
    actor_masked, sampler_masked = actor_probs[mask], sampler_probs[mask]

    return {
        # Per-shard extremes (aggregated via global min/max)
        "batch/sampler_probs_diff_max": _to_scalar(probs_diff.max()) if mask.any() else 0.0,
        "batch/sampler_probs_diff_min": _to_scalar(probs_diff.min()) if mask.any() else 0.0,
        "batch/sampler_logprobs_diff_max": _to_scalar(logprobs_diff.max()) if mask.any() else 0.0,
        "batch/sampler_logprobs_diff_min": _to_scalar(logprobs_diff.min()) if mask.any() else 0.0,
        # Sufficient statistics for mean/std/pearson
        "_valid_count": mask.sum().item(),
        "_sum": _to_scalar(probs_diff.sum()) if mask.any() else 0.0,
        "_sum_squares": _to_scalar((probs_diff**2).sum()) if mask.any() else 0.0,
        "_logprobs_sum": _to_scalar(logprobs_diff.sum()) if mask.any() else 0.0,
        "_logprobs_sum_squares": _to_scalar((logprobs_diff**2).sum()) if mask.any() else 0.0,
        "_actor_sum": _to_scalar(actor_masked.sum()) if mask.any() else 0.0,
        "_sampler_sum": _to_scalar(sampler_masked.sum()) if mask.any() else 0.0,
        "_actor_sum_squares": _to_scalar((actor_masked**2).sum()) if mask.any() else 0.0,
        "_sampler_sum_squares": _to_scalar((sampler_masked**2).sum()) if mask.any() else 0.0,
        "_product_sum": _to_scalar((actor_masked * sampler_masked).sum()) if mask.any() else 0.0,
    }


_ZERO_METRICS = {
    "batch/sampler_probs_diff_max": 0.0,
    "batch/sampler_probs_diff_mean": 0.0,
    "batch/sampler_probs_diff_min": 0.0,
    "batch/sampler_probs_diff_std": 0.0,
    "batch/sampler_logprobs_diff_max": 0.0,
    "batch/sampler_logprobs_diff_mean": 0.0,
    "batch/sampler_logprobs_diff_min": 0.0,
    "batch/sampler_logprobs_diff_std": 0.0,
    "batch/sampler_actor_probs_pearson_corr": 0.0,
}


def aggregate_trainer_sampler_metrics(metrics_list: list[dict]) -> dict:
    """Aggregate sufficient statistics from one or more workers into final metrics.

    Works for both single-worker (controller-resident, list of 1) and
    multi-worker (worker-resident, list of N) cases.

    Args:
        metrics_list: List of dicts from ``compute_trainer_sampler_mismatch_metrics``.

    Returns:
        Dict with human-readable metric keys (mean, std, min, max, pearson).
    """
    if not metrics_list:
        return {}

    def _sum_key(key: str) -> float:
        return sum(m.get(key, 0.0) for m in metrics_list)

    n = _sum_key("_valid_count")
    if n == 0:
        return dict(_ZERO_METRICS)

    # Probs diff: mean, std
    mean = _sum_key("_sum") / n
    variance = (_sum_key("_sum_squares") / n) - mean**2
    std = variance**0.5 if variance > 0 else 0.0

    # Logprobs diff: mean, std
    logprobs_mean = _sum_key("_logprobs_sum") / n
    logprobs_variance = (_sum_key("_logprobs_sum_squares") / n) - logprobs_mean**2
    logprobs_std = logprobs_variance**0.5 if logprobs_variance > 0 else 0.0

    # Pearson correlation: Cov(X,Y) / (Std(X) * Std(Y))
    actor_mean = _sum_key("_actor_sum") / n
    sampler_mean = _sum_key("_sampler_sum") / n
    covariance = (_sum_key("_product_sum") / n) - (actor_mean * sampler_mean)
    actor_std = max(0, _sum_key("_actor_sum_squares") / n - actor_mean**2) ** 0.5
    sampler_std = max(0, _sum_key("_sampler_sum_squares") / n - sampler_mean**2) ** 0.5
    denom = actor_std * sampler_std
    pearson = covariance / denom if denom > 0 else 0.0

    return {
        "batch/sampler_probs_diff_max": max(m["batch/sampler_probs_diff_max"] for m in metrics_list),
        "batch/sampler_probs_diff_mean": mean,
        "batch/sampler_probs_diff_min": min(m["batch/sampler_probs_diff_min"] for m in metrics_list),
        "batch/sampler_probs_diff_std": std,
        "batch/sampler_logprobs_diff_max": max(m["batch/sampler_logprobs_diff_max"] for m in metrics_list),
        "batch/sampler_logprobs_diff_mean": logprobs_mean,
        "batch/sampler_logprobs_diff_min": min(m["batch/sampler_logprobs_diff_min"] for m in metrics_list),
        "batch/sampler_logprobs_diff_std": logprobs_std,
        "batch/sampler_actor_probs_pearson_corr": pearson,
    }
