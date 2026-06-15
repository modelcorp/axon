"""Tests for axon.driver.components.advantage_component module."""

import pytest
import torch

from axon.driver.components.advantage_component import stepwise_advantage_broadcast
from axon.protocol import DataProto


def _make_batch(
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    program_uids: list[str],
) -> DataProto:
    """Build a DataProto with the fields required by stepwise_advantage_broadcast.

    Args:
        advantages: Tensor of shape (n, seq_len) with advantage values.
        response_mask: Tensor of shape (n, seq_len) with 0/1 mask.
        program_uids: List of program uid strings, length n.

    Returns:
        DataProto with batch keys {advantages, response_mask} and
        non_tensor_batch key {program_uids}.
    """
    return DataProto.from_dict(
        tensors={
            "advantages": advantages,
            "response_mask": response_mask,
        },
        non_tensors={
            "program_uids": program_uids,
        },
    )


def _make_other_batch(
    response_mask: torch.Tensor,
    program_uids: list[str],
) -> DataProto:
    """Build a DataProto for other (non-last) steps, which have no advantages yet.

    Args:
        response_mask: Tensor of shape (n, seq_len) with 0/1 mask.
        program_uids: List of program uid strings, length n.

    Returns:
        DataProto with batch key {response_mask} and non_tensor_batch key {program_uids}.
    """
    return DataProto.from_dict(
        tensors={
            "response_mask": response_mask,
        },
        non_tensors={
            "program_uids": program_uids,
        },
    )


# ---------------------------------------------------------------------------
# Basic two-program broadcast
# ---------------------------------------------------------------------------


class TestStepwiseAdvantageBroadcastBasic:
    """Two programs, each with a last step and one earlier step."""

    def test_two_programs_broadcast(self):
        """Advantages from last steps are broadcast to earlier steps."""
        # Last steps: program A has advantages [2, 4, 0, 0] with mask [1, 1, 0, 0]
        #             program B has advantages [6, 6, 6, 0] with mask [1, 1, 1, 0]
        last_step_advantages = torch.tensor([[2.0, 4.0, 0.0, 0.0], [6.0, 6.0, 6.0, 0.0]])
        last_step_mask = torch.tensor([[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 1.0, 0.0]])
        last_step_batch = _make_batch(
            advantages=last_step_advantages,
            response_mask=last_step_mask,
            program_uids=["prog_A", "prog_B"],
        )

        # Other (earlier) steps for both programs
        other_mask = torch.tensor([[1.0, 1.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
        other_step_batch = _make_other_batch(
            response_mask=other_mask,
            program_uids=["prog_A", "prog_B"],
        )

        result_other, result_last = stepwise_advantage_broadcast(last_step_batch, other_step_batch)

        # Scalar advantage for prog_A: mean([2, 4]) = 3.0
        # Scalar advantage for prog_B: mean([6, 6, 6]) = 6.0
        expected_A = 3.0
        expected_B = 6.0

        # other_step_batch advantages should be scalar * mask
        expected_other_adv = torch.tensor(
            [
                [expected_A, expected_A, expected_A, 0.0],
                [expected_B, 0.0, 0.0, 0.0],
            ]
        )
        torch.testing.assert_close(result_other.batch["advantages"], expected_other_adv)
        torch.testing.assert_close(result_other.batch["returns"], expected_other_adv)

        # last_step_batch should be returned unchanged
        torch.testing.assert_close(result_last.batch["advantages"], last_step_advantages)

    def test_broadcast_values_are_per_program(self):
        """Each earlier step receives the scalar advantage of its own program,
        not a global average."""
        last_step_batch = _make_batch(
            advantages=torch.tensor([[10.0, 0.0], [2.0, 4.0]]),
            response_mask=torch.tensor([[1.0, 0.0], [1.0, 1.0]]),
            program_uids=["X", "Y"],
        )
        other_step_batch = _make_other_batch(
            response_mask=torch.tensor([[1.0, 1.0], [1.0, 1.0]]),
            program_uids=["X", "Y"],
        )

        result_other, _ = stepwise_advantage_broadcast(last_step_batch, other_step_batch)

        # X scalar = mean([10]) = 10.0, Y scalar = mean([2, 4]) = 3.0
        assert result_other.batch["advantages"][0, 0].item() == pytest.approx(10.0)
        assert result_other.batch["advantages"][0, 1].item() == pytest.approx(10.0)
        assert result_other.batch["advantages"][1, 0].item() == pytest.approx(3.0)
        assert result_other.batch["advantages"][1, 1].item() == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# Single program
# ---------------------------------------------------------------------------


class TestStepwiseAdvantageBroadcastSingle:
    """Only one program is present."""

    def test_single_program(self):
        """One program with one last step and one earlier step."""
        last_step_batch = _make_batch(
            advantages=torch.tensor([[1.0, 3.0, 5.0]]),
            response_mask=torch.tensor([[1.0, 1.0, 1.0]]),
            program_uids=["solo"],
        )
        other_step_batch = _make_other_batch(
            response_mask=torch.tensor([[1.0, 1.0, 0.0]]),
            program_uids=["solo"],
        )

        result_other, result_last = stepwise_advantage_broadcast(last_step_batch, other_step_batch)

        # scalar = mean([1, 3, 5]) = 3.0
        expected = torch.tensor([[3.0, 3.0, 0.0]])
        torch.testing.assert_close(result_other.batch["advantages"], expected)
        torch.testing.assert_close(result_other.batch["returns"], expected)

    def test_single_program_multiple_earlier_steps(self):
        """One program with multiple earlier steps receives the same scalar."""
        last_step_batch = _make_batch(
            advantages=torch.tensor([[4.0, 8.0]]),
            response_mask=torch.tensor([[1.0, 1.0]]),
            program_uids=["t1"],
        )
        other_step_batch = _make_other_batch(
            response_mask=torch.tensor([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]]),
            program_uids=["t1", "t1", "t1"],
        )

        result_other, _ = stepwise_advantage_broadcast(last_step_batch, other_step_batch)

        # scalar = mean([4, 8]) = 6.0
        expected = torch.tensor([[6.0, 6.0], [6.0, 0.0], [0.0, 6.0]])
        torch.testing.assert_close(result_other.batch["advantages"], expected)


# ---------------------------------------------------------------------------
# Empty other_step_batch (all steps are last steps)
# ---------------------------------------------------------------------------


class TestStepwiseAdvantageBroadcastEmpty:
    """Edge case: no earlier steps exist (every step is a last step)."""

    def test_empty_other_step_batch(self):
        """When there are no earlier steps, the function should handle gracefully."""
        last_step_batch = _make_batch(
            advantages=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            response_mask=torch.tensor([[1.0, 1.0], [1.0, 1.0]]),
            program_uids=["a", "b"],
        )
        # Empty other_step_batch with matching seq_len shape
        other_step_batch = _make_other_batch(
            response_mask=torch.zeros(0, 2),
            program_uids=[],
        )

        result_other, result_last = stepwise_advantage_broadcast(last_step_batch, other_step_batch)

        # Other step batch should have zero-length advantages
        assert result_other.batch["advantages"].shape == (0, 2)
        assert result_other.batch["returns"].shape == (0, 2)

        # Last step batch unchanged
        torch.testing.assert_close(
            result_last.batch["advantages"],
            torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        )


# ---------------------------------------------------------------------------
# Response mask correctly filters advantage computation
# ---------------------------------------------------------------------------


class TestStepwiseAdvantageBroadcastMasking:
    """Verify that the scalar advantage is computed only over masked (response) tokens."""

    def test_masked_tokens_excluded_from_scalar(self):
        """Non-response tokens (mask=0) should not contribute to the scalar advantage."""
        # Advantage tensor has values everywhere, but mask selects only some.
        # advantages = [10, 20, 99, 99], mask = [1, 1, 0, 0] -> scalar = mean([10, 20]) = 15
        last_step_batch = _make_batch(
            advantages=torch.tensor([[10.0, 20.0, 99.0, 99.0]]),
            response_mask=torch.tensor([[1.0, 1.0, 0.0, 0.0]]),
            program_uids=["m1"],
        )
        other_step_batch = _make_other_batch(
            response_mask=torch.tensor([[1.0, 1.0, 1.0, 1.0]]),
            program_uids=["m1"],
        )

        result_other, _ = stepwise_advantage_broadcast(last_step_batch, other_step_batch)

        # All response tokens in the other step get 15.0
        expected = torch.tensor([[15.0, 15.0, 15.0, 15.0]])
        torch.testing.assert_close(result_other.batch["advantages"], expected)

    def test_all_tokens_masked_out_yields_zero(self):
        """If all tokens are masked out (mask all zeros), scalar advantage should be 0."""
        last_step_batch = _make_batch(
            advantages=torch.tensor([[5.0, 5.0]]),
            response_mask=torch.tensor([[0.0, 0.0]]),
            program_uids=["z1"],
        )
        other_step_batch = _make_other_batch(
            response_mask=torch.tensor([[1.0, 1.0]]),
            program_uids=["z1"],
        )

        result_other, _ = stepwise_advantage_broadcast(last_step_batch, other_step_batch)

        # scalar = 0.0 (no masked-in tokens), so other advantages = 0 * mask = 0
        expected = torch.tensor([[0.0, 0.0]])
        torch.testing.assert_close(result_other.batch["advantages"], expected)

    def test_other_step_mask_applied_to_broadcast(self):
        """The broadcasted scalar is multiplied by the other_step response_mask,
        so non-response positions in earlier steps remain zero."""
        last_step_batch = _make_batch(
            advantages=torch.tensor([[8.0, 8.0]]),
            response_mask=torch.tensor([[1.0, 1.0]]),
            program_uids=["q1"],
        )
        other_step_batch = _make_other_batch(
            response_mask=torch.tensor([[1.0, 0.0]]),
            program_uids=["q1"],
        )

        result_other, _ = stepwise_advantage_broadcast(last_step_batch, other_step_batch)

        # scalar = 8.0, but second position is masked out in other step
        expected = torch.tensor([[8.0, 0.0]])
        torch.testing.assert_close(result_other.batch["advantages"], expected)

    def test_negative_advantages_broadcast_correctly(self):
        """Negative advantage values should propagate correctly."""
        last_step_batch = _make_batch(
            advantages=torch.tensor([[-2.0, -6.0, 0.0]]),
            response_mask=torch.tensor([[1.0, 1.0, 0.0]]),
            program_uids=["neg"],
        )
        other_step_batch = _make_other_batch(
            response_mask=torch.tensor([[1.0, 1.0, 1.0]]),
            program_uids=["neg"],
        )

        result_other, _ = stepwise_advantage_broadcast(last_step_batch, other_step_batch)

        # scalar = mean([-2, -6]) = -4.0
        expected = torch.tensor([[-4.0, -4.0, -4.0]])
        torch.testing.assert_close(result_other.batch["advantages"], expected)

    def test_returns_match_advantages(self):
        """The function sets both 'advantages' and 'returns' to the same value."""
        last_step_batch = _make_batch(
            advantages=torch.tensor([[5.0, 5.0]]),
            response_mask=torch.tensor([[1.0, 1.0]]),
            program_uids=["r1"],
        )
        other_step_batch = _make_other_batch(
            response_mask=torch.tensor([[1.0, 1.0]]),
            program_uids=["r1"],
        )

        result_other, _ = stepwise_advantage_broadcast(last_step_batch, other_step_batch)

        torch.testing.assert_close(
            result_other.batch["advantages"],
            result_other.batch["returns"],
        )


# ---------------------------------------------------------------------------
# Hardened: many programs, numerical precision, large batches
# ---------------------------------------------------------------------------


class TestStepwiseAdvantageBroadcastHardened:
    def test_many_programs_correct_routing(self):
        """10 programs — each earlier step must get its own program's scalar."""
        n_prog = 10
        seq_len = 3
        # Last steps: each program has advantages [i, 2*i, 0] with mask [1, 1, 0]
        last_advs = torch.tensor([[float(i), float(2 * i), 0.0] for i in range(n_prog)])
        last_mask = torch.tensor([[1.0, 1.0, 0.0]] * n_prog)
        last_uids = [f"prog_{i}" for i in range(n_prog)]
        last_step_batch = _make_batch(last_advs, last_mask, last_uids)

        # Other steps: one earlier step per program, full mask
        other_mask = torch.ones(n_prog, seq_len)
        other_step_batch = _make_other_batch(other_mask, last_uids)

        result_other, _ = stepwise_advantage_broadcast(last_step_batch, other_step_batch)

        for i in range(n_prog):
            expected_scalar = (float(i) + float(2 * i)) / 2.0  # mean of [i, 2*i]
            for j in range(seq_len):
                assert result_other.batch["advantages"][i, j].item() == pytest.approx(expected_scalar, abs=1e-5), (
                    f"prog_{i}, pos {j}"
                )

    def test_very_small_advantages_precision(self):
        """Very small advantage values should not be lost to floating point."""
        last_step_batch = _make_batch(
            advantages=torch.tensor([[1e-7, 3e-7]]),
            response_mask=torch.tensor([[1.0, 1.0]]),
            program_uids=["tiny"],
        )
        other_step_batch = _make_other_batch(
            response_mask=torch.tensor([[1.0, 1.0]]),
            program_uids=["tiny"],
        )
        result_other, _ = stepwise_advantage_broadcast(last_step_batch, other_step_batch)
        expected = 2e-7  # mean of [1e-7, 3e-7]
        assert result_other.batch["advantages"][0, 0].item() == pytest.approx(expected, rel=1e-4)

    def test_very_large_advantages(self):
        """Large advantage values should broadcast correctly."""
        last_step_batch = _make_batch(
            advantages=torch.tensor([[1e6, 1e6]]),
            response_mask=torch.tensor([[1.0, 1.0]]),
            program_uids=["big"],
        )
        other_step_batch = _make_other_batch(
            response_mask=torch.tensor([[1.0, 1.0]]),
            program_uids=["big"],
        )
        result_other, _ = stepwise_advantage_broadcast(last_step_batch, other_step_batch)
        assert result_other.batch["advantages"][0, 0].item() == pytest.approx(1e6)

    def test_mixed_positive_negative_advantages(self):
        """Advantages that mix positive and negative should average correctly."""
        last_step_batch = _make_batch(
            advantages=torch.tensor([[-10.0, 10.0, -10.0, 10.0]]),
            response_mask=torch.tensor([[1.0, 1.0, 1.0, 1.0]]),
            program_uids=["mixed"],
        )
        other_step_batch = _make_other_batch(
            response_mask=torch.tensor([[1.0, 1.0, 1.0, 1.0]]),
            program_uids=["mixed"],
        )
        result_other, _ = stepwise_advantage_broadcast(last_step_batch, other_step_batch)
        # mean([-10, 10, -10, 10]) = 0
        expected = torch.tensor([[0.0, 0.0, 0.0, 0.0]])
        torch.testing.assert_close(result_other.batch["advantages"], expected, atol=1e-5, rtol=0)

    def test_multiple_earlier_steps_per_program_different_masks(self):
        """Multiple earlier steps for same program with varying masks."""
        last_step_batch = _make_batch(
            advantages=torch.tensor([[4.0, 8.0, 12.0]]),
            response_mask=torch.tensor([[1.0, 1.0, 1.0]]),
            program_uids=["t"],
        )
        # 3 earlier steps with different masks
        other_step_batch = _make_other_batch(
            response_mask=torch.tensor(
                [
                    [1.0, 1.0, 1.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            ),
            program_uids=["t", "t", "t"],
        )
        result_other, _ = stepwise_advantage_broadcast(last_step_batch, other_step_batch)
        scalar = 8.0  # mean([4, 8, 12])

        torch.testing.assert_close(
            result_other.batch["advantages"],
            torch.tensor(
                [
                    [scalar, scalar, scalar],
                    [scalar, 0.0, 0.0],
                    [0.0, 0.0, scalar],
                ]
            ),
        )

    def test_last_step_batch_not_modified(self):
        """The function should not mutate the last_step_batch advantages."""
        original_advantages = torch.tensor([[1.0, 2.0, 3.0]])
        last_step_batch = _make_batch(
            advantages=original_advantages.clone(),
            response_mask=torch.tensor([[1.0, 1.0, 1.0]]),
            program_uids=["t"],
        )
        other_step_batch = _make_other_batch(
            response_mask=torch.tensor([[1.0, 1.0, 1.0]]),
            program_uids=["t"],
        )
        _, result_last = stepwise_advantage_broadcast(last_step_batch, other_step_batch)
        torch.testing.assert_close(result_last.batch["advantages"], original_advantages)
