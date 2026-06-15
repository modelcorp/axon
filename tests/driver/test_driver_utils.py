"""Tests for axon.driver.driver_utils module."""

import math

import numpy as np
import pytest
import torch

from axon.core import ResourcePool
from axon.core.worker import Worker
from axon.driver.driver_utils import (
    RoleWorkerConfig,
    ValidationResult,
    pad_dataproto_to_world_size,
    process_mini_batch,
    split_mini_batches_by_programs,
)
from axon.protocol import DataProto

# ---------------------------------------------------------------------------
# ValidationResult - pass@k solved/unsolved
# ---------------------------------------------------------------------------


class TestValidationResultPassAtK:
    def test_pass_at_k_solved(self):
        """If any sample for a uid >= 1.0, that uid counts as solved."""
        vr = ValidationResult(n_samples=4)
        vr.add(reward=0.0, uid="u1")
        vr.add(reward=0.5, uid="u1")
        vr.add(reward=1.0, uid="u1")  # solved
        vr.add(reward=0.2, uid="u1")
        metrics = vr.compute_metrics()
        assert metrics["val/pass@4"] == pytest.approx(1.0)
        assert metrics["val/reward"] == pytest.approx(0.425)

    def test_pass_at_k_unsolved(self):
        """If no sample for a uid >= 1.0, that uid is not solved."""
        vr = ValidationResult(n_samples=3)
        vr.add(reward=0.2, uid="u1")
        vr.add(reward=0.5, uid="u1")
        vr.add(reward=0.99, uid="u1")
        metrics = vr.compute_metrics()
        assert metrics["val/pass@3"] == pytest.approx(0.0)

    def test_exact_boundary_1_0_solved(self):
        """Exact boundary: reward == 1.0 counts as solved."""
        vr = ValidationResult(n_samples=2)
        vr.add(reward=1.0, uid="u1")
        vr.add(reward=0.0, uid="u1")
        metrics = vr.compute_metrics()
        assert metrics["val/pass@2"] == pytest.approx(1.0)

    def test_exact_boundary_0_999_not_solved(self):
        """reward=0.999 is below 1.0, so not solved."""
        vr = ValidationResult(n_samples=1)
        vr.add(reward=0.999, uid="u1")
        metrics = vr.compute_metrics()
        assert metrics["val/pass@1"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# ValidationResult - multiple uids
# ---------------------------------------------------------------------------


class TestValidationResultMultipleUids:
    def test_different_uids(self):
        vr = ValidationResult(n_samples=2)
        vr.add(reward=0.5, uid="u1")
        vr.add(reward=1.0, uid="u1")
        vr.add(reward=0.3, uid="u2")
        vr.add(reward=0.8, uid="u2")
        metrics = vr.compute_metrics()
        # 1 of 2 uids solved
        assert metrics["val/pass@2"] == pytest.approx(0.5)
        assert metrics["val/reward"] == pytest.approx(0.65)

    def test_many_uids_some_solved_some_not(self):
        """10 uids, 4 solved (reward=1.0), 6 unsolved (reward=0.5). Ratio = 0.4."""
        vr = ValidationResult(n_samples=1)
        for i in range(10):
            reward = 1.0 if i < 4 else 0.5
            vr.add(reward=reward, uid=f"u{i}")
        metrics = vr.compute_metrics()
        assert metrics["val/pass@1"] == pytest.approx(0.4)


# ---------------------------------------------------------------------------
# ValidationResult - negative rewards
# ---------------------------------------------------------------------------


class TestValidationResultNegativeRewards:
    def test_negative_rewards_clipping(self):
        vr = ValidationResult(n_samples=1)
        vr.add(reward=-2.0, uid="u1")
        vr.add(reward=0.5, uid="u2")
        metrics = vr.compute_metrics()
        assert metrics["val/reward"] == pytest.approx(-0.75)
        assert metrics["val/reward_clip[-1, 1]"] == pytest.approx(-0.25)
        assert metrics["val/reward_clip[0, 1]"] == pytest.approx(0.25)

    def test_all_negative_rewards_best_score_is_actual_best(self):
        """With float('-inf') as initial best, the best score for a uid with
        all-negative rewards should be the actual maximum (e.g., -0.3),
        not 0. Since -0.3 < 1.0, the uid is not solved."""
        vr = ValidationResult(n_samples=3)
        vr.add(reward=-1.0, uid="u1")
        vr.add(reward=-0.5, uid="u1")
        vr.add(reward=-0.3, uid="u1")
        metrics = vr.compute_metrics()
        assert metrics["val/pass@3"] == pytest.approx(0.0)
        # Mean of [-1.0, -0.5, -0.3]
        assert metrics["val/reward"] == pytest.approx(-0.6)


# ---------------------------------------------------------------------------
# ValidationResult - edge cases
# ---------------------------------------------------------------------------


class TestValidationResultEdgeCases:
    def test_empty_returns_empty_dict(self):
        vr = ValidationResult(n_samples=1)
        assert vr.compute_metrics() == {}

    def test_nan_reward_propagation(self):
        """NaN rewards should propagate to mean (result is NaN)."""
        vr = ValidationResult(n_samples=1)
        vr.add(reward=float("nan"), uid="u1")
        vr.add(reward=0.5, uid="u2")
        metrics = vr.compute_metrics()
        # Mean of [nan, 0.5] is nan
        assert math.isnan(metrics["val/reward"])

    def test_inf_reward(self):
        """Inf reward: best score is inf >= 1.0, so uid is solved."""
        vr = ValidationResult(n_samples=1)
        vr.add(reward=float("inf"), uid="u1")
        metrics = vr.compute_metrics()
        assert metrics["val/pass@1"] == pytest.approx(1.0)
        assert math.isinf(metrics["val/reward"])

    def test_large_number_of_samples(self):
        """Stress test with 1000 samples across 100 uids."""
        vr = ValidationResult(n_samples=10)
        for uid_idx in range(100):
            for sample in range(10):
                # Every 3rd uid gets a perfect score on the last sample
                reward = 1.0 if (uid_idx % 3 == 0 and sample == 9) else 0.5
                vr.add(reward=reward, uid=f"u{uid_idx}")
        metrics = vr.compute_metrics()
        # uids 0,3,6,...,99 -> 34 uids solved out of 100
        solved_count = len([i for i in range(100) if i % 3 == 0])
        assert metrics["val/pass@10"] == pytest.approx(solved_count / 100.0)


# ---------------------------------------------------------------------------
# RoleWorkerConfig
# ---------------------------------------------------------------------------


class TestRoleWorkerConfig:
    def test_construction(self):
        pool = ResourcePool(process_on_nodes=[1])
        cfg = RoleWorkerConfig(cls=Worker, resource_pool=pool)
        assert cfg.cls is Worker
        assert cfg.resource_pool is pool
        assert cfg.init_kwargs == {}


# ---------------------------------------------------------------------------
# Program-aware padding
# ---------------------------------------------------------------------------


def _make_program_batch(num_rows: int = 3) -> DataProto:
    return DataProto.from_dict(
        tensors={
            "response_mask": torch.ones(num_rows, 4),
        },
        non_tensors={
            "uid": np.array([f"group-{i}" for i in range(num_rows)], dtype=object),
            "step_ids": np.array([f"step-{i}" for i in range(num_rows)], dtype=object),
            "program_uids": np.array([f"program-{i}" for i in range(num_rows)], dtype=object),
            "program_group_ids": np.array([f"group-{i}" for i in range(num_rows)], dtype=object),
            "program_step_ids": np.array([f"group-{i}_step0" for i in range(num_rows)], dtype=object),
            "num_program_steps": np.array([1 for _ in range(num_rows)], dtype=object),
            "is_last_step": np.array([True for _ in range(num_rows)], dtype=object),
            "has_step_rewards": np.array([True for _ in range(num_rows)], dtype=object),
            "is_padding": np.array([False for _ in range(num_rows)], dtype=object),
            "index": np.arange(num_rows),
        },
    )


class TestProgramAwarePadding:
    def test_padding_rows_get_non_trainable_program_metadata(self):
        batch = _make_program_batch(num_rows=3)
        original_uids = set(batch.non_tensor_batch["uid"])
        original_program_uids = set(batch.non_tensor_batch["program_uids"])

        padded = pad_dataproto_to_world_size(batch, [2])

        assert len(padded) == 4
        pad_idx = 3
        assert padded.batch["response_mask"][pad_idx].sum().item() == 0
        assert padded.non_tensor_batch["is_padding"][pad_idx] is True
        assert padded.non_tensor_batch["is_last_step"][pad_idx] is False
        assert padded.non_tensor_batch["has_step_rewards"][pad_idx] is False
        assert padded.non_tensor_batch["num_program_steps"][pad_idx] == 1
        assert padded.non_tensor_batch["index"][pad_idx] == -1
        assert padded.non_tensor_batch["uid"][pad_idx] not in original_uids
        assert padded.non_tensor_batch["program_uids"][pad_idx] not in original_program_uids

    def test_split_mini_batches_keeps_program_rows_together(self):
        batch = DataProto.from_dict(
            tensors={
                "response_mask": torch.ones(4, 4),
            },
            non_tensors={
                "uid": np.array(["group-a", "group-a", "group-b", "group-c"], dtype=object),
                "step_ids": np.array(["a-0", "a-1", "b-0", "c-0"], dtype=object),
                "program_uids": np.array(["program-a", "program-a", "program-b", "program-c"], dtype=object),
                "program_group_ids": np.array(["group-a", "group-a", "group-b", "group-c"], dtype=object),
                "program_step_ids": np.array(["a-step0", "a-step1", "b-step0", "c-step0"], dtype=object),
                "num_program_steps": np.array([2, 2, 1, 1], dtype=object),
                "is_last_step": np.array([False, True, True, True], dtype=object),
                "has_step_rewards": np.array([False, False, False, False], dtype=object),
                "is_padding": np.array([False, False, False, False], dtype=object),
            },
        )

        mini_batches = split_mini_batches_by_programs(batch, mini_batch_size=2, world_size=2)

        assert len(mini_batches) == 2
        uid_to_batch_indices = {}
        for batch_idx, mini_batch in enumerate(mini_batches):
            program_uids = mini_batch.non_tensor_batch["program_uids"]
            real_program_uids = {
                uid for i, uid in enumerate(program_uids) if not mini_batch.non_tensor_batch["is_padding"][i]
            }
            for uid in real_program_uids:
                row_indices = np.where(program_uids == uid)[0]
                uid_to_batch_indices.setdefault(uid, set()).add(batch_idx)
                assert len(row_indices) in (1, 2)
                assert all(not mini_batch.non_tensor_batch["is_padding"][idx] for idx in row_indices)

            padding_indices = np.where(mini_batch.non_tensor_batch["is_padding"] == True)[0]  # noqa: E712
            for pad_idx in padding_indices:
                assert mini_batch.batch["response_mask"][pad_idx].sum().item() == 0

        assert uid_to_batch_indices == {
            "program-a": {0},
            "program-b": {0},
            "program-c": {1},
        }

    def test_process_mini_batch_counts_only_valid_programs(self):
        batch = DataProto.from_dict(
            tensors={
                "response_mask": torch.tensor(
                    [
                        [1, 1, 0],
                        [1, 0, 0],
                        [0, 0, 0],
                    ]
                ),
            },
            non_tensors={
                "program_uids": np.array(["program-a", "program-a", "program-padding"], dtype=object),
                "is_padding": np.array([False, False, True], dtype=object),
            },
        )

        process_mini_batch(batch)

        assert batch.batch["valid_batch_size"][0].item() == 2
        assert batch.batch["valid_token_count"][0].item() == 3
        assert batch.batch["valid_program_count"][0].item() == 1


# ---------------------------------------------------------------------------
# Multimodal padding — orphaned image/video placeholder regression
# ---------------------------------------------------------------------------
#
# Divisor-padding deep-copies real rows, so a multimodal batch's padding rows inherit the
# source rows' placeholder tokens. _mark_padding_rows must zero those tokens to match its
# multi_modal_inputs={} feature-side scrub, else the actor forward sees more image tokens
# than features and crashes. These tests pin that invariant on the non-divisible
# multimodal batch path — the trigger that had no coverage and kept the bug latent.

_IMAGE_TOKEN_ID = 151655  # Qwen2-VL image placeholder
_VIDEO_TOKEN_ID = 151656  # Qwen2.5-VL video placeholder


def _make_multimodal_batch(num_rows: int = 3, seqlen: int = 8) -> DataProto:
    """Batch whose FRONT rows carry image+video placeholders and real
    multi_modal_inputs, so padding actually clones the placeholders into pad rows."""
    input_ids = torch.arange(1, num_rows * seqlen + 1).reshape(num_rows, seqlen)
    # Placeholders must live in the cloned (front) rows or the test is vacuous.
    input_ids[0, 1:4] = _IMAGE_TOKEN_ID  # 3 image placeholders
    input_ids[0, 5:7] = _VIDEO_TOKEN_ID  # 2 video placeholders
    if num_rows > 1:
        input_ids[1, 2:5] = _IMAGE_TOKEN_ID
    return DataProto.from_dict(
        tensors={
            "input_ids": input_ids,
            "attention_mask": torch.ones(num_rows, seqlen, dtype=torch.long),
            "response_mask": torch.ones(num_rows, seqlen),
        },
        non_tensors={
            "multi_modal_inputs": np.array(
                [{"pixel_values": torch.randn(4, 1176)} for _ in range(num_rows)],
                dtype=object,
            ),
            "uid": np.array([f"group-{i}" for i in range(num_rows)], dtype=object),
            "is_padding": np.array([False for _ in range(num_rows)], dtype=object),
            "index": np.arange(num_rows),
        },
    )


class TestMultimodalPadding:
    def test_padding_rows_strip_image_and_video_placeholders(self):
        batch = _make_multimodal_batch(num_rows=3, seqlen=8)
        padded = pad_dataproto_to_world_size(batch, [2])  # 3 -> 4, pad_size=1

        assert len(padded) == 4
        pad_slice = slice(3, 4)
        pad_ids = padded.batch["input_ids"][pad_slice]
        # (1) core invariant: no image/video placeholders survive in the pad rows.
        #     Reverting the input_ids scrub in _mark_padding_rows makes this fail.
        assert (pad_ids == _IMAGE_TOKEN_ID).sum().item() == 0
        assert (pad_ids == _VIDEO_TOKEN_ID).sum().item() == 0
        # (2) feature side cleared
        assert padded.non_tensor_batch["multi_modal_inputs"][3] == {}
        # (3) attention_mask deliberately kept = 1 (the fragile half — guards the
        #     Megatron mRoPE get_rope_index torch.cat([]) regression).
        assert (padded.batch["attention_mask"][pad_slice] == 1).all()
        # (4) no loss leakage from pad rows
        assert padded.batch["response_mask"][pad_slice].sum().item() == 0
        # (5) negative control: a real (non-pad) row keeps its placeholders untouched.
        assert (padded.batch["input_ids"][0] == _IMAGE_TOKEN_ID).sum().item() == 3
        assert (padded.batch["input_ids"][0] == _VIDEO_TOKEN_ID).sum().item() == 2

    def test_padding_strips_placeholders_with_wraparound(self):
        # pad_size (6) > num_rows (2): pad_dataproto_to_divisor re-clones front rows,
        # so every pad row — not just the first — must still be stripped.
        batch = _make_multimodal_batch(num_rows=2, seqlen=8)
        padded = pad_dataproto_to_world_size(batch, [8])  # 2 -> 8, pad_size=6

        assert len(padded) == 8
        pad_slice = slice(2, 8)
        pad_ids = padded.batch["input_ids"][pad_slice]
        assert (pad_ids == _IMAGE_TOKEN_ID).sum().item() == 0
        assert (pad_ids == _VIDEO_TOKEN_ID).sum().item() == 0
        for i in range(2, 8):
            assert padded.non_tensor_batch["multi_modal_inputs"][i] == {}
        assert (padded.batch["attention_mask"][pad_slice] == 1).all()

    def test_text_only_padding_is_not_zeroed(self):
        # No multi_modal_inputs -> the input_ids scrub is scoped OUT; the text-only
        # padding row must keep its deep-copied real tokens (only loss is masked).
        batch = DataProto.from_dict(
            tensors={
                "input_ids": torch.arange(1, 3 * 8 + 1).reshape(3, 8),
                "attention_mask": torch.ones(3, 8, dtype=torch.long),
                "response_mask": torch.ones(3, 8),
            },
            non_tensors={
                "uid": np.array([f"group-{i}" for i in range(3)], dtype=object),
            },
        )
        padded = pad_dataproto_to_world_size(batch, [2])  # 3 -> 4

        assert padded.batch["response_mask"][3].sum().item() == 0
        assert (padded.batch["input_ids"][3] != 0).any()  # not blanket-zeroed
