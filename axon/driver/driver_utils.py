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
"""
PPO training utilities.

Shared helpers used by the PPO driver controllers (SyncPPO,
AsyncPPO) and worker mixins (AsyncTrainerMixin, AsyncSamplerMixin):
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from functools import reduce
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from axon.core import ResourcePool
from axon.protocol import DataProto, pad_dataproto_to_divisor
from axon.utils.metrics import reduce_metrics
from axon.utils.print_utils import append_to_dict
from axon.utils.seqlen_balancing import calculate_workload, get_seqlen_balanced_partitions, log_seqlen_unbalance

if TYPE_CHECKING:
    from axon.core.worker import Worker

# Filler id for neutralized padding-row input_ids (see _mark_padding_rows). Must not be
# any model's image/video placeholder id (Qwen2-VL: 151655/151656); 0 is safe everywhere.
PAD_FILLER_TOKEN_ID = 0

# ---------------------------------------------------------------------------
# Role-to-worker mapping
# ---------------------------------------------------------------------------


@dataclass
class RoleWorkerConfig:
    """Configuration for a role's worker class and resource pool.

    Attributes:
        cls: The Ray remote worker class for this role.
        resource_pool: The ResourcePool containing CPU/GPU resource bundles for this role.
        init_kwargs: Extra keyword arguments forwarded to RayActorWithInitArgs.
        max_concurrency: Max concurrent method calls for this Ray actor.
            Controls the size of the internal thread pool. Lower values prevent
            cuBLAS thread-local handle proliferation (~67 MB per thread).
    """

    cls: type[Worker]
    resource_pool: ResourcePool
    init_kwargs: dict[str, Any] = field(default_factory=dict)
    max_concurrency: int | None = None


# ---------------------------------------------------------------------------
# Dataloader dict → DataProto conversion
# ---------------------------------------------------------------------------


def convert_batch_dict_to_dataproto(
    batch_dict: dict,
    global_steps: int,
    n_samples: int = 1,
    temperature_scheduler=None,
    temperature_config=None,
    val: bool = False,
    val_n_samples: int | None = None,
) -> DataProto:
    """Convert a dataloader dict to a DataProto batch.

    Args:
        batch_dict: Raw dict from the dataloader (keys: ``env_args``, optionally ``index``).
        global_steps: Current training step.
        n_samples: Number of samples per prompt (``config.decoding.n``).
        temperature_scheduler: Optional ``TemperatureScheduler`` for dynamic temperature.
        temperature_config: The ``config.decoding.temperature_schedule`` sub-config
            (needs ``.enable`` attribute).
        val: If True, this is a validation batch.
        val_n_samples: Number of samples for validation (``config.validation.decoding.n``).

    Returns:
        DataProto ready for sampling / program execution.
    """
    env_args = batch_dict.get("env_args", [])
    batch_size = len(env_args)

    non_tensor_batch = {
        "uid": np.array([str(uuid.uuid4()) for _ in range(batch_size)], dtype=object),
    }

    if env_args:
        non_tensor_batch["env_args"] = np.array(env_args, dtype=object)

    if "index" in batch_dict:
        non_tensor_batch["index"] = np.array(batch_dict["index"])

    batch = DataProto.from_dict(tensors={}, non_tensors=non_tensor_batch)
    batch.batch = None
    batch.meta_info.update(
        {
            "sample_params": {},
            "global_steps": global_steps,
        }
    )

    if not val and temperature_scheduler is not None and temperature_config is not None and temperature_config.enable:
        batch.meta_info["sample_params"]["temperature"] = temperature_scheduler.get_temperature(
            max(global_steps - 1, 0)
        )

    repeat_n = val_n_samples if val else n_samples
    if repeat_n and repeat_n > 1:
        batch = batch.repeat(repeat_times=repeat_n, interleave=True)

    return batch


# ---------------------------------------------------------------------------
# Validation metric accumulator
# ---------------------------------------------------------------------------


class ValidationResult:
    """Accumulates per-sample validation rewards and computes summary metrics.

    Usage::

        result = ValidationResult(n_samples=16)
        for program in programs:
            result.add(reward=program.reward, uid=program.group_id,
                       data_source=getattr(program, 'data_source', 'unknown'))
        metrics = result.compute_metrics()
    """

    def __init__(self, n_samples: int = 1):
        self.n_samples = n_samples
        self.rewards: list[float] = []
        self.uids: list[str] = []
        self.data_sources: list[str] = []

    def add(self, reward: float, uid: str, data_source: str = "unknown"):
        self.rewards.append(reward)
        self.uids.append(uid)
        self.data_sources.append(data_source)

    def compute_metrics(self) -> dict:
        metric_dict: dict[str, float] = {}
        if not self.rewards:
            return metric_dict

        rewards_array = np.array(self.rewards)

        metric_dict["val/reward"] = float(np.mean(rewards_array))
        metric_dict["val/reward_clip[-1, 1]"] = float(np.mean(np.clip(rewards_array, -1, 1)))
        metric_dict["val/reward_clip[0, 1]"] = float(np.mean(np.clip(rewards_array, 0, 1)))

        # Best score per uid for pass@k
        uid_best_scores: dict[str, float] = {}
        for reward, uid in zip(self.rewards, self.uids, strict=False):
            current_best = uid_best_scores.get(uid, float("-inf"))
            uid_best_scores[uid] = max(current_best, reward)

        solved_problems = [score >= 1 for score in uid_best_scores.values()]
        metric_dict[f"val/pass@{self.n_samples}"] = float(np.mean(solved_problems))

        return metric_dict


# ---------------------------------------------------------------------------
# Batch balancing and padding
# ---------------------------------------------------------------------------


def _mark_padding_rows(batch: DataProto, pad_slice: slice) -> None:
    """Normalize metadata for rows copied in by DataProto padding."""
    start = pad_slice.start or 0
    stop = pad_slice.stop if pad_slice.stop is not None else len(batch)

    if batch.non_tensor_batch is not None:
        for row_idx in range(start, stop):
            group_uid = str(uuid.uuid4())
            program_uid = str(uuid.uuid4())
            step_uid = str(uuid.uuid4())

            if "uid" in batch.non_tensor_batch:
                batch.non_tensor_batch["uid"][row_idx] = group_uid
            if "program_group_ids" in batch.non_tensor_batch:
                batch.non_tensor_batch["program_group_ids"][row_idx] = group_uid
            if "program_uids" in batch.non_tensor_batch:
                batch.non_tensor_batch["program_uids"][row_idx] = program_uid
            if "step_ids" in batch.non_tensor_batch:
                batch.non_tensor_batch["step_ids"][row_idx] = step_uid
            if "program_step_ids" in batch.non_tensor_batch:
                batch.non_tensor_batch["program_step_ids"][row_idx] = f"{program_uid}_padding_step0"
            if "multi_modal_inputs" in batch.non_tensor_batch:
                # Clear the feature side of this padding row. Its token side (the cloned
                # placeholder ids) is zeroed below; the two must stay in sync, or the
                # actor forward hits the image-token/feature mismatch. See PAD_FILLER_TOKEN_ID.
                batch.non_tensor_batch["multi_modal_inputs"][row_idx] = {}

        if "num_program_steps" in batch.non_tensor_batch:
            batch.non_tensor_batch["num_program_steps"][pad_slice] = 1
        if "is_last_step" in batch.non_tensor_batch:
            batch.non_tensor_batch["is_last_step"][pad_slice] = False
        if "has_step_rewards" in batch.non_tensor_batch:
            batch.non_tensor_batch["has_step_rewards"][pad_slice] = False
        if "is_padding" in batch.non_tensor_batch:
            batch.non_tensor_batch["is_padding"][pad_slice] = True
        if "index" in batch.non_tensor_batch:
            batch.non_tensor_batch["index"][pad_slice] = -1

    if batch.batch is not None:
        if "response_mask" in batch.batch:
            batch.batch["response_mask"][pad_slice] = 0
        # Token side of the multi_modal_inputs={} scrub above. Padding rows are deep
        # copies of real rows (pad_dataproto_to_divisor), so their input_ids still carry
        # image/video placeholder tokens after the feature side was cleared. The actor
        # forward then counts placeholders with no matching pixel_values and crashes:
        # "Image features and image tokens do not match" (FSDP) / "video token start
        # index" (Megatron). Zeroing the whole row drops every placeholder id without
        # per-model knowledge (the row is pure filler: response_mask=0). Keep
        # attention_mask=1 — a 0-token row breaks Megatron mRoPE (get_rope_index ->
        # torch.cat([])). Multimodal batches only; text-only runs are untouched.
        if (
            batch.non_tensor_batch is not None
            and "multi_modal_inputs" in batch.non_tensor_batch
            and "input_ids" in batch.batch
        ):
            batch.batch["input_ids"][pad_slice] = PAD_FILLER_TOKEN_ID


def balance_batch(
    batch: DataProto,
    world_size: int,
    metrics: dict,
    logging_prefix: str = "global_seqlen",
) -> DataProto:
    """Reorder batch rows so each DP rank gets similar total tokens.

    When ``program_uids`` is present in the batch, all steps of each program
    are kept on the same DP rank (program-aware balancing). This ensures
    ``valid_program_count`` is correct for ``program-mean`` loss reduction
    and that each worker has intact programs for mini-batch splitting.

    Extra padding rows (with zeroed masks) are added when needed to
    equalize partition sizes after program assignment, so the returned
    batch may be larger than the input.

    Falls back to row-level balancing when ``program_uids`` is absent.

    Args:
        batch: Training batch to reorder. May be extended with padding.
        world_size: Number of data-parallel ranks.
        metrics: Dict to update with balance statistics.
        logging_prefix: Prefix for balance metric keys.

    Returns:
        The reordered (and possibly padded) batch.
    """
    program_uids = batch.non_tensor_batch.get("program_uids") if batch.non_tensor_batch else None
    is_padding = batch.non_tensor_batch.get("is_padding") if batch.non_tensor_batch else None

    if program_uids is not None:
        batch, global_partition_lst = _balance_by_programs(batch, world_size, program_uids, is_padding)
    else:
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = attention_mask.view(batch_size, -1).sum(-1)
        workload_lst = calculate_workload(global_seqlen_lst)
        global_partition_lst = get_seqlen_balanced_partitions(
            workload_lst,
            k_partitions=world_size,
            equal_size=True,
        )

    # Sort within each partition to reduce bubbles in pipeline parallel.
    attention_mask = batch.batch["attention_mask"]
    global_seqlen_lst = attention_mask.view(len(batch), -1).sum(-1)
    workload_lst = calculate_workload(global_seqlen_lst)
    for idx, partition in enumerate(global_partition_lst):
        partition.sort(key=lambda x: (workload_lst[x], x))
        ordered_partition = partition[::2] + partition[1::2][::-1]
        global_partition_lst[idx] = ordered_partition

    global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
    batch.reorder(global_idx)

    global_balance_stats = log_seqlen_unbalance(
        seqlen_list=global_seqlen_lst,
        partitions=global_partition_lst,
        prefix=logging_prefix,
    )
    metrics.update(global_balance_stats)

    return batch


def _balance_by_programs(
    batch: DataProto,
    world_size: int,
    program_uids: np.ndarray,
    is_padding: np.ndarray | None,
) -> tuple[DataProto, list[list[int]]]:
    """Assign programs to DP partitions keeping all steps together.

    Uses LPT (Longest Processing Time first) scheduling: sort programs
    by total workload descending, then greedily assign each to the
    partition with the smallest current row count (ties broken by
    workload). Padding rows are distributed last to equalize partition
    sizes for ``chunk()``.

    If there aren't enough padding rows to equalize, extra padding rows
    are appended to the batch (with zeroed masks and ``is_padding=True``).

    Returns:
        Tuple of (possibly padded batch, partition list).
    """
    attention_mask = batch.batch["attention_mask"]
    batch_size = len(batch)
    global_seqlen_lst = attention_mask.view(batch_size, -1).sum(-1)

    # Group rows by program uid
    uid_to_indices: dict[str, list[int]] = {}
    for i, uid in enumerate(program_uids):
        uid_to_indices.setdefault(uid, []).append(i)

    # Separate real programs from padding rows
    real_programs: list[tuple[str, list[int], float]] = []
    padding_indices: list[int] = []

    for uid, indices in uid_to_indices.items():
        if is_padding is not None and all(is_padding[i] for i in indices):
            padding_indices.extend(indices)
        else:
            workload = sum(global_seqlen_lst[i].item() for i in indices)
            real_programs.append((uid, indices, workload))

    # LPT: sort by workload descending for better balance
    real_programs.sort(key=lambda x: x[2], reverse=True)

    # Greedy assignment: put each program on the partition with fewest rows
    # (ties broken by smallest workload)
    partitions: list[list[int]] = [[] for _ in range(world_size)]
    partition_sizes = [0] * world_size
    partition_workloads = [0.0] * world_size

    for _uid, indices, workload in real_programs:
        best = min(range(world_size), key=lambda p: (partition_sizes[p], partition_workloads[p]))
        partitions[best].extend(indices)
        partition_sizes[best] += len(indices)
        partition_workloads[best] += workload

    # Distribute existing padding rows to fill smaller partitions
    target = max(partition_sizes)  # at least as large as the biggest partition
    # Round up to next multiple of world_size so chunk() divides evenly
    target = math.ceil(target / world_size) * world_size
    # Make total divisible by world_size
    total_needed = target * world_size

    padding_iter = iter(padding_indices)
    used_padding = 0
    for p in range(world_size):
        while partition_sizes[p] < target:
            try:
                idx = next(padding_iter)
                partitions[p].append(idx)
                partition_sizes[p] += 1
                used_padding += 1
            except StopIteration:
                break

    # If we still need more rows, add new padding to the batch
    extra_needed = total_needed - sum(partition_sizes)
    if extra_needed > 0:
        original_size = len(batch)
        batch, _pad_size = pad_dataproto_to_divisor(batch, original_size + extra_needed)

        # Mark new rows as padding with zeroed masks and unique program metadata.
        new_start = original_size
        new_end = len(batch)
        pad_slice = slice(new_start, new_end)
        _mark_padding_rows(batch, pad_slice)

        # Distribute new padding rows to partitions that need them
        new_idx = new_start
        for p in range(world_size):
            while partition_sizes[p] < target:
                partitions[p].append(new_idx)
                partition_sizes[p] += 1
                new_idx += 1

    return batch, partitions


def pad_dataproto_to_world_size(batch: DataProto, world_sizes: list[int]) -> DataProto:
    """Pad batch so its size is divisible by the LCM of all worker world sizes.

    Padded rows get unique program metadata (avoiding GRPO grouping),
    ``is_last_step=False``, ``is_padding=True``, and zeroed-out masks.

    Args:
        batch: Training batch to pad.
        world_sizes: List of worker group world sizes.

    Returns:
        The padded batch.
    """
    if not world_sizes:
        return batch

    world_size = reduce(math.lcm, world_sizes)
    original_batch_size = len(batch)
    batch, pad_size = pad_dataproto_to_divisor(batch, world_size)

    if pad_size > 0:
        pad_slice = slice(original_batch_size, original_batch_size + pad_size)
        _mark_padding_rows(batch, pad_slice)

    return batch


def process_mini_batch(mini_batch: DataProto) -> None:
    """Inject valid counts into a mini-batch so ``agg_loss`` divides correctly.

    Computes LOCAL counts from the current slice:

    * ``valid_batch_size``  — rows with ≥ 1 valid token.
    * ``valid_token_count`` — total valid tokens.
    * ``valid_program_count`` — unique ``program_uids`` among valid rows
      (only when ``program_uids`` is present).

    Both training paths need GLOBAL mini-batch counts (all D workers combined):

    * **Controller-resident** (``update_trainer``): called on the full
      mini-batch before DP dispatch — counts are already global.

    * **Async** (``AsyncTrainerMixin.train_on_batch``): called per
      worker's local mini-batch — counts are local (~1/D of global).
      The worker then scales ×D to get global counts.

    Backend gradient compensation is handled separately:
    * FSDP (``fsdp_models.py``): ``loss *= dp_replicas`` (D for AVG, 1 for SUM).
    * Megatron (``megatron_models.py``): ``loss *= n_micro_batches``.
    """
    mask = mini_batch.batch.get("response_mask", None)
    if mask is None:
        return

    n = len(mini_batch)
    valid_mask = mask.sum(dim=-1) > 0

    mini_batch.batch["valid_batch_size"] = torch.full((n,), int(valid_mask.sum().item()), dtype=torch.long)
    mini_batch.batch["valid_token_count"] = torch.full((n,), int(mask.sum().item()), dtype=torch.long)
    mini_batch.batch["per_row_token_count"] = mask.sum(dim=-1).to(torch.long)

    program_uids = mini_batch.non_tensor_batch.get("program_uids", None) if mini_batch.non_tensor_batch else None
    if program_uids is not None:
        valid_idx = valid_mask.nonzero(as_tuple=True)[0].tolist()
        count = len({program_uids[i] for i in valid_idx})
        mini_batch.batch["valid_program_count"] = torch.full((n,), count, dtype=torch.long)


def split_mini_batches_by_programs(
    batch: DataProto,
    mini_batch_size: int,
    world_size: int,
    shuffle: bool = False,
    generator: torch.Generator | None = None,
) -> list[DataProto]:
    """Split batch into mini-batches of ``mini_batch_size`` programs each.

    All steps (rows) of a program stay together.  Each mini-batch is
    padded so its row count is divisible by ``world_size`` for even DP
    sharding.  ``mini_batch_size`` is the number of *programs*, not rows.
    """
    program_uids_arr = batch.non_tensor_batch["program_uids"]

    # Group row indices by program uid.
    uid_to_indices: dict[str, list[int]] = {}
    for i, uid in enumerate(program_uids_arr):
        uid_to_indices.setdefault(uid, []).append(i)

    program_uids = list(uid_to_indices.keys())

    # Fast path: everything fits in one mini-batch — just pad, no reorder.
    if mini_batch_size >= len(program_uids):
        if shuffle:
            perm = torch.randperm(len(batch), generator=generator).tolist()
            batch = batch.select_idxs(perm)
        return [_pad_mini_batch(batch, world_size)]

    if shuffle:
        perm = torch.randperm(len(program_uids), generator=generator).tolist()
        program_uids = [program_uids[j] for j in perm]

    # Chunk programs into groups of mini_batch_size.
    mini_batches = []
    for start in range(0, len(program_uids), mini_batch_size):
        chunk_uids = program_uids[start : start + mini_batch_size]
        row_indices = []
        for uid in chunk_uids:
            row_indices.extend(uid_to_indices[uid])
        mb = batch.select_idxs(row_indices)
        mini_batches.append(_pad_mini_batch(mb, world_size))

    return mini_batches


def _pad_mini_batch(mini_batch: DataProto, world_size: int) -> DataProto:
    """Pad a mini-batch so its size is divisible by ``world_size``.

    Padded rows get zeroed-out masks so they don't affect the loss.
    """
    if world_size <= 1:
        return mini_batch

    original_size = len(mini_batch)
    mini_batch, pad_size = pad_dataproto_to_divisor(mini_batch, world_size)

    if pad_size > 0:
        pad_slice = slice(original_size, original_size + pad_size)
        _mark_padding_rows(mini_batch, pad_slice)

    return mini_batch


def update_trainer(
    batch: DataProto,
    training_client,
    loss_fn: str,
    loss_fn_args: dict,
    epochs: int = 1,
    mini_batch_size: int = None,
    mini_batch_shuffle: bool = False,
    mini_batch_seed: int | None = None,
    world_size: int = 1,
) -> dict:
    """Run PPO training epochs with mini-batch splitting.

    The PPO epoch and mini-batch loop runs on the trainer (driver process).
    Each mini-batch is dispatched to workers via the TrainingClient.

    **Online SGD mode** (current): ``optim_step`` runs after each mini-batch.
    Each mini-batch is an independent gradient update. ``process_mini_batch``
    computes per-mini-batch valid counts, so each mini-batch's loss is
    self-normalized. The gradient does not depend on how the full batch is
    split.

    .. note:: **Blueprint for batch-SGD mode** (gradient accumulation):

       To switch to accumulating gradients across all mini-batches and
       stepping the optimizer once, make these changes:

       1. Call ``process_mini_batch(batch)`` ONCE on the full batch
          (before the mini-batch loop) to get **global** valid counts.
       2. Pass these global counts to each mini-batch (instead of
          recomputing per-mini-batch).
       3. Move ``optim_step`` outside the inner loop (call once after
          all mini-batches).

       This makes mini-batch splitting a pure memory optimization —
       the gradient equals the full-batch gradient regardless of split.

    Args:
        batch: Full global training batch (DataProto).
        training_client: TrainingClient wrapping the trainer worker group.
        loss_fn: Loss function name (e.g., "ppo").
        loss_fn_args: Loss function arguments.
        epochs: Number of PPO epochs.
        mini_batch_size: Number of *programs* per mini-batch.
        mini_batch_shuffle: Whether to shuffle programs before splitting each epoch.
        mini_batch_seed: Random seed for reproducible shuffling.
        world_size: Trainer worker world size (for padding mini-batches).

    Returns:
        dict: Aggregated metrics (already reduced to scalars).
    """
    metrics = {}
    if not mini_batch_size:
        mini_batch_size = len(batch)

    generator = torch.Generator().manual_seed(mini_batch_seed) if mini_batch_seed is not None else None

    for epoch in range(epochs):
        mini_batches = split_mini_batches_by_programs(
            batch,
            mini_batch_size,
            world_size,
            shuffle=mini_batch_shuffle,
            generator=generator,
        )
        last_epoch = epoch == epochs - 1

        for i, mini_batch in enumerate(mini_batches):
            # Per-mini-batch valid counts (online SGD — each mini-batch
            # is an independent update with its own normalization).
            process_mini_batch(mini_batch)
            fb_output = training_client.forward_backward(mini_batch, loss_fn=loss_fn, loss_fn_args=loss_fn_args)
            append_to_dict(metrics, fb_output.meta_info["metrics"])

            last_mini_batch = last_epoch and i == len(mini_batches) - 1
            optim_output = training_client.optim_step(step_lr=last_mini_batch)
            append_to_dict(metrics, optim_output.meta_info["metrics"])
    return reduce_metrics(metrics)
