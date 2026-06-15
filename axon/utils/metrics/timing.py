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
Timing metrics and utilities.
"""

from contextlib import contextmanager
from typing import Any

from codetiming import Timer

from axon.protocol import DataProto


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

    # Prompt tokens are those in attention_mask but NOT in response_mask
    prompt_mask = attention_mask.bool() & ~response_mask.bool()

    prompt_length = prompt_mask.sum(-1).float()
    response_length = response_mask.sum(-1).float()  # (batch_size,)

    return dict(
        response_mask=response_mask,
        prompt_length=prompt_length,
        response_length=response_length,
    )


def _timer(name: str, timing_raw: dict[str, float]):
    """Inner function that handles the core timing logic.

    Args:
        name (str): The name/identifier for this timing measurement.
        timing_raw (Dict[str, float]): Dictionary to store timing information.
    """
    with Timer(name=name, logger=None) as timer:
        yield
    if name not in timing_raw:
        timing_raw[name] = 0
    timing_raw[name] += timer.last


@contextmanager
def marked_timer(
    name: str,
    timing_raw: dict[str, float],
    color: str = None,
    domain: str | None = None,
    category: str | None = None,
):
    """Context manager for timing with optional NVTX markers.

    Measures the execution time of code within its context, accumulates the timing
    information into ``timing_raw[name]``, and emits NVTX range markers when NVTX is
    available so the same call sites work for both timing metrics and Nsight profiling.

    Args:
        name: The name/identifier for this timing measurement.
        timing_raw: Dictionary to accumulate timing information into.
        color: Color for the NVTX marker (ignored when NVTX is unavailable).
        domain: Domain for the NVTX marker (ignored when NVTX is unavailable).
        category: Category for the NVTX marker (ignored when NVTX is unavailable).
    """
    from axon.utils.import_utils import is_nvtx_available

    if is_nvtx_available():
        from axon.utils.profiler.nvtx_profile import mark_end_range, mark_start_range

        mark_range = mark_start_range(message=name, color=color, domain=domain, category=category)
        try:
            yield from _timer(name, timing_raw)
        finally:
            mark_end_range(mark_range)
    else:
        yield from _timer(name, timing_raw)


def compute_timing_metrics(
    batch: DataProto, timing_raw: dict[str, float], include_detail: bool = False
) -> dict[str, Any]:
    """
    Computes timing metrics for different processing stages in PPO training.

    This function calculates both raw timing metrics (in seconds) and per-token timing metrics
    (in milliseconds) for various processing stages like generation, reference computation,
    value computation, advantage computation, and model updates.

    Args:
        batch: A DataProto object containing batch data with responses and attention masks.
        timing_raw: A dictionary mapping stage names to their execution times in seconds.
        include_detail: If True, returns sufficient statistics for aggregation.
                       If False, returns computed metrics (backward compatible).

    Returns:
        A dictionary containing:
            - timing_s/{name}: Raw timing in seconds for each stage
            - timing_per_token_ms/{name}: Per-token timing in milliseconds for each stage

    Note:
        Different stages use different token counts for normalization:
        - "gen" uses only response tokens
        - Other stages ("ref", "values", "adv", "update_critic", "forward_backward") use all tokens
          (prompt + response)
    """
    response_info = _compute_response_info(batch)
    num_prompt_tokens = response_info["prompt_length"].sum().item()
    num_response_tokens = response_info["response_length"].sum().item()
    num_overall_tokens = num_prompt_tokens + num_response_tokens

    num_tokens_of_section = {
        "gen": num_response_tokens,
        **{name: num_overall_tokens for name in ["ref", "values", "adv", "update_critic", "forward_backward"]},
    }

    metrics = {
        **{f"timing_s/{name}": value for name, value in timing_raw.items()},
        **{
            f"timing_per_token_ms/{name}": timing_raw[name] * 1000 / max(1, num_tokens_of_section[name])
            for name in set(num_tokens_of_section.keys()) & set(timing_raw.keys())
        },
    }

    if include_detail:
        metrics.update(
            {
                "_num_prompt_tokens": num_prompt_tokens,
                "_num_response_tokens": num_response_tokens,
                "_num_overall_tokens": num_overall_tokens,
            }
        )

    return metrics


def reduce_timing_metrics(metrics_list: list[dict]) -> dict[str, Any]:
    """
    Aggregate timing metrics from multiple workers.
    Expects metrics computed with include_detail=True.
    """
    if not metrics_list:
        return {}

    # Find all timing keys
    all_timing_keys = set()
    for m in metrics_list:
        all_timing_keys.update(k for k in m.keys() if k.startswith("timing_s/"))

    # Sum all timing values
    aggregated = {}
    for key in all_timing_keys:
        # Timing is shared across worker metrics
        aggregated[key] = metrics_list[0].get(key, 0)

    # Aggregate token counts
    total_response_tokens = sum(m.get("_num_response_tokens", 0) for m in metrics_list)
    total_overall_tokens = sum(m.get("_num_overall_tokens", 0) for m in metrics_list)

    # Compute per-token timing
    num_tokens_of_section = {
        "gen": total_response_tokens,
        "ref": total_overall_tokens,
        "values": total_overall_tokens,
        "adv": total_overall_tokens,
        "update_critic": total_overall_tokens,
        "forward_backward": total_overall_tokens,
    }

    for key in all_timing_keys:
        stage = key.replace("timing_s/", "")
        if stage in num_tokens_of_section and num_tokens_of_section[stage] > 0:
            aggregated[f"timing_per_token_ms/{stage}"] = aggregated[key] * 1000 / num_tokens_of_section[stage]

    return aggregated
