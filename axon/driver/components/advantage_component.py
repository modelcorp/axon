# Copyright 2025 Model AI Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Advantage computation component for PPO training.

Orchestrates the full advantage computation sequence: pass-rate statistics,
reward statistics, stepwise advantage broadcasting (for multi-step agent
programs), optional zero-advantage filtering, and padding/balancing for
data-parallel distribution.

These functions sit between the raw RL utilities in :mod:`axon.utils.rl`
(which compute individual metrics) and the training controllers (which
call the component as a single step).
"""

import numpy as np
import torch

from axon.protocol import DataProto
from axon.trainer.algos.advantages import compute_advantage
from axon.utils.rl.utils import compute_pass_metrics, compute_reward_statistics, filter_zero_advantage_samples


def stepwise_advantage_broadcast(
    last_step_batch: DataProto,
    other_step_batch: DataProto,
) -> tuple[DataProto, DataProto]:
    """Broadcast advantages from the last step to all earlier steps of a program.

    For multi-step agent programs, advantages are computed only on the final
    step (where rewards are assigned). This function propagates those
    advantages back to earlier steps so that all steps share the same
    per-program advantage. Uses ``program_uids`` to link steps.

    Args:
        last_step_batch: DataProto for the last step (has computed advantages).
        other_step_batch: DataProto for earlier steps (needs advantages assigned).

    Returns:
        Tuple of (other_step_batch, last_step_batch), both with advantages set.
    """
    last_step_program_uids = last_step_batch.non_tensor_batch["program_uids"]
    src_advantages = last_step_batch.batch["advantages"]
    src_mask = last_step_batch.batch["response_mask"]

    other_step_program_uids = other_step_batch.non_tensor_batch["program_uids"]
    tgt_mask = other_step_batch.batch["response_mask"]

    # Build program_uid -> scalar advantage (mean over response tokens)
    program_uid_to_scalar_adv = {}
    for i, program_uid in enumerate(last_step_program_uids):
        mask = src_mask[i].bool()
        masked_adv = src_advantages[i][mask]
        scalar = masked_adv.mean().item() if masked_adv.numel() > 0 else 0.0
        program_uid_to_scalar_adv[program_uid] = scalar

    # Create advantage tensor for other_step_batch
    if len(other_step_program_uids) > 0:
        scalar_rows = torch.stack(
            [
                torch.full_like(
                    tgt_mask[i],
                    fill_value=program_uid_to_scalar_adv[program_uid],
                    dtype=torch.float32,
                )
                for i, program_uid in enumerate(other_step_program_uids)
            ]
        )
    else:
        scalar_rows = torch.zeros_like(tgt_mask, dtype=torch.float32)

    final_advantage = scalar_rows * tgt_mask
    other_step_batch.batch["advantages"] = final_advantage
    other_step_batch.batch["returns"] = final_advantage

    return other_step_batch, last_step_batch


def compute_advantage_component(
    batch: DataProto,
    config,
    metrics: dict,
) -> DataProto:
    """Compute advantages on a training batch.

    Pure advantage computation — no padding, balancing, or count injection.
    Callers handle DP-specific concerns (pad, balance, process_mini_batch).

    Steps:
        1. Compute pass rate and reward statistics
        2. Set token_level_rewards from token_level_scores
        3. Split last-step vs other-step data
        4. Compute advantages (GRPO, GAE, etc.) on last steps
        5. Broadcast advantages to earlier steps
        6. Optionally filter zero-advantage samples

    Args:
        batch: DataProto with ``token_level_scores``, ``program_uids``,
            ``is_last_step``, ``uid``, and ``attention_mask`` fields.
        config: Training config with ``stepwise_advantage_mode``,
            ``drop_zero_advantage_samples``, and advantage estimator settings.
        metrics: Dict to update with pass rate and reward metrics.

    Returns:
        Batch with advantages computed.
    """
    # Pass rate statistics
    pass_metrics, pass_rates_dict = compute_pass_metrics(batch)
    metrics.update(pass_metrics)

    # Reward statistics (mean, min, max across batch)
    reward_metrics = compute_reward_statistics(batch)
    metrics.update(reward_metrics)

    # Assign per-row pass rates
    pass_rates = np.zeros((len(batch),))
    for i in range(len(batch)):
        pass_rates[i] = pass_rates_dict.get(batch[i].non_tensor_batch["uid"], 0)
    batch.non_tensor_batch["pass_rate"] = pass_rates

    # Set token-level rewards from raw scores unless already pre-computed
    # (e.g. KL penalty applied by controller-resident mode before calling).
    if "token_level_rewards" not in batch.batch:
        batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

    # When programs provide per-step rewards (has_step_rewards=True), each
    # step already carries its own score — compute advantage on all steps
    # directly.  Otherwise, compute on last steps only and broadcast to
    # earlier steps of the same program.
    has_per_step = "has_step_rewards" in batch.non_tensor_batch and np.any(
        np.asarray(batch.non_tensor_batch["has_step_rewards"], dtype=bool)
    )

    if has_per_step:
        batch = compute_advantage(batch, config=config)
    else:
        # Broadcast: compute on last steps, then broadcast to earlier steps
        is_last_step = batch.non_tensor_batch["is_last_step"]
        last_step_indices = np.where(is_last_step == True)[0]  # noqa: E712
        other_step_indices = np.where(is_last_step == False)[0]  # noqa: E712
        other_step_batch = batch.select_idxs(other_step_indices)
        batch = batch.select_idxs(last_step_indices)

        batch = compute_advantage(batch, config=config)

        other_step_batch, batch = stepwise_advantage_broadcast(batch, other_step_batch)
        batch = DataProto.concat([batch, other_step_batch])

    # Filter groups where all samples got the same reward (zero advantage)
    if config.drop_zero_advantage_samples:
        pre_filter_size = len(batch)
        batch = filter_zero_advantage_samples(batch)
        post_filter_size = len(batch)
        metrics["batch/filtered_zero_adv_count"] = pre_filter_size - post_filter_size
        metrics["batch/filtered_zero_adv_frac"] = (pre_filter_size - post_filter_size) / max(pre_filter_size, 1)

    if "advantages" in batch.batch:
        adv = batch.batch["advantages"]
        mask = batch.batch.get("response_mask", None)
        if mask is not None:
            valid_adv = adv[mask.bool()]
            if valid_adv.numel() > 0:
                metrics["batch/advantages_std"] = valid_adv.std().item()
                metrics["batch/advantages_mean"] = valid_adv.mean().item()
                metrics["batch/advantages_abs_max"] = valid_adv.abs().max().item()

    return batch
