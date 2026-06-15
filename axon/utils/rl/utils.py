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
"""Reinforcement learning utilities for training data processing."""

import numpy as np
import torch

from axon.protocol import DataProto
from axon.utils.print_utils import colorful_print


def filter_zero_advantage_samples(batch: DataProto):
    """
    Filter out samples where advantage is zero.

    Args:
        batch: DataProto containing the batch data with advantages

    Returns:
        Filtered DataProto with zero-advantage samples removed
    """
    assert "advantages" in batch.batch
    advantages = batch.batch["advantages"]
    eps = 1e-5
    mask_nonzero_advantages_mask = torch.any(advantages.abs() > eps, dim=1)
    keep_samples_idxes = torch.nonzero(mask_nonzero_advantages_mask, as_tuple=False).squeeze(1).tolist()
    total_samples = len(batch)
    drop_samples = total_samples - len(keep_samples_idxes)

    if drop_samples > 0:
        colorful_print(f"Filtered {drop_samples}/{total_samples} samples with zero advantage ", fg="yellow", bold=True)

    # Filter the batch
    if len(keep_samples_idxes) > 0:
        filtered_batch = batch.select_idxs(keep_samples_idxes)
        return filtered_batch
    else:
        # All advantages are zero — the loss gradient will be zero regardless.
        # Return a minimal subset (one program group) to avoid OOM from
        # processing the full batch (which can be 8-64x larger than normal
        # filtered batches). We keep a full group so that downstream padding
        # and DP sharding get enough rows for each GPU rank.
        colorful_print("WARNING: All groups have zero advantage! Keeping one group.", fg="red", bold=True)
        uids = batch.non_tensor_batch.get("uid", None)
        if uids is not None:
            first_uid = uids[0]
            group_idxs = [i for i, u in enumerate(uids) if u == first_uid]
            return batch.select_idxs(group_idxs)
        return batch


def compute_reward_statistics(batch: DataProto):
    """
    Compute reward statistics across programs, not steps.
    Excludes padded rows (is_padding=True) and uses only the last step
    of each program to compute a single reward per program.
    """
    token_level_scores = batch.batch["token_level_scores"]
    batch_size = token_level_scores.size(0)

    # Build mask for valid rows: non-padded and last step of each program
    is_padding = batch.non_tensor_batch.get("is_padding", np.zeros(batch_size, dtype=bool))
    is_last_step = batch.non_tensor_batch.get("is_last_step", np.ones(batch_size, dtype=bool))

    valid_rows = (~np.asarray(is_padding, dtype=bool)) & np.asarray(is_last_step, dtype=bool)

    if not valid_rows.any():
        # No valid program rows; return zeros to avoid NaNs
        return {"reward_mean": 0.0, "reward_max": 0.0, "reward_min": 0.0}

    # Each program's reward is the sum across its response tokens (only last token is non-zero)
    program_rewards = token_level_scores[torch.from_numpy(valid_rows).to(token_level_scores.device)].sum(dim=1)

    return {
        "batch/reward/mean": program_rewards.mean().detach().item(),
        "batch/reward/max": program_rewards.max().detach().item(),
        "batch/reward/min": program_rewards.min().detach().item(),
    }


def compute_pass_metrics(batch: DataProto):
    """
    Compute pass statistics across programs (by uid): solve_none, solve_all, solve_partial,
    and a mapping of uid to pass rate (fraction of samples with reward >= 1).

    Uses only last steps per program when is_last_step is available.
    Returns a metrics dict (for logging) and pass_rates_dict (uid -> pass rate float).
    """
    # Filter to last steps only if available
    if "is_last_step" in batch.non_tensor_batch:
        batch = batch.select_idxs(np.where(batch.non_tensor_batch["is_last_step"])[0])

    reward_tensor = batch.batch["token_level_scores"]
    uids = batch.non_tensor_batch["uid"]

    pass_rates_dict = {}
    solve_none = solve_all = 0

    for uid in np.unique(uids):
        uid_rewards = reward_tensor[uids == uid].sum(-1)
        pass_rates_dict[uid] = (uid_rewards >= 1).float().mean().item()

        if (uid_rewards <= 0).all():
            solve_none += 1
        elif (uid_rewards >= 1).all():
            solve_all += 1

    return {
        "batch/solve/none": solve_none,
        "batch/solve/all": solve_all,
        "batch/solve/partial": len(pass_rates_dict) - solve_none - solve_all,
    }, pass_rates_dict
