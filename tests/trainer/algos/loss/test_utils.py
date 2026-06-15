"""
Tests for policy loss utility functions.
"""

import pytest
import torch

from axon.trainer.algos.loss.utils import (
    agg_loss,
    clip_by_value,
    entropy_from_logits,
    masked_mean,
)


class TestMaskedMean:
    def test_basic(self):
        values = torch.tensor([1.0, 2.0, 3.0, 4.0])
        mask = torch.tensor([1.0, 1.0, 0.0, 0.0])
        result = masked_mean(values, mask)
        assert torch.allclose(result, torch.tensor(1.5))

    def test_all_masked(self):
        values = torch.tensor([1.0, 2.0, 3.0])
        mask = torch.tensor([0.0, 0.0, 0.0])
        result = masked_mean(values, mask)
        # clamp_min(1.0) prevents division by zero, result should be 0
        assert torch.allclose(result, torch.tensor(0.0))

    def test_all_unmasked(self):
        values = torch.tensor([2.0, 4.0, 6.0])
        mask = torch.tensor([1.0, 1.0, 1.0])
        result = masked_mean(values, mask)
        assert torch.allclose(result, torch.tensor(4.0))

    def test_2d(self):
        values = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        mask = torch.tensor([[1.0, 0.0], [1.0, 1.0]])
        result = masked_mean(values, mask)
        # (1 + 3 + 4) / 3 = 8/3
        assert torch.allclose(result, torch.tensor(8.0 / 3.0))

    def test_boolean_mask(self):
        values = torch.tensor([10.0, 20.0, 30.0])
        mask = torch.tensor([True, False, True])
        result = masked_mean(values, mask)
        assert torch.allclose(result, torch.tensor(20.0))

    def test_gradient_flows(self):
        values = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        mask = torch.tensor([1.0, 1.0, 0.0])
        result = masked_mean(values, mask)
        result.backward()
        assert values.grad is not None
        assert torch.allclose(values.grad, torch.tensor([0.5, 0.5, 0.0]))


class TestClipByValue:
    def test_basic_scalar_bounds(self):
        tensor = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        result = clip_by_value(tensor, torch.tensor(-1.0), torch.tensor(1.0))
        expected = torch.tensor([-1.0, -1.0, 0.0, 1.0, 1.0])
        assert torch.allclose(result, expected)

    def test_tensor_bounds(self):
        tensor = torch.tensor([0.5, 1.5, 2.5])
        min_val = torch.tensor([0.0, 1.0, 2.0])
        max_val = torch.tensor([1.0, 2.0, 3.0])
        result = clip_by_value(tensor, min_val, max_val)
        expected = torch.tensor([0.5, 1.5, 2.5])
        assert torch.allclose(result, expected)

    def test_values_already_in_range(self):
        tensor = torch.tensor([0.0, 0.5, 1.0])
        result = clip_by_value(tensor, torch.tensor(-1.0), torch.tensor(2.0))
        assert torch.allclose(result, tensor)

    def test_all_below_min(self):
        tensor = torch.tensor([-5.0, -3.0, -1.0])
        result = clip_by_value(tensor, torch.tensor(0.0), torch.tensor(10.0))
        expected = torch.tensor([0.0, 0.0, 0.0])
        assert torch.allclose(result, expected)


class TestEntropyFromLogits:
    def test_uniform_distribution(self):
        # Uniform distribution over 4 classes -> entropy = ln(4)
        logits = torch.zeros(1, 4)
        entropy = entropy_from_logits(logits)
        expected = torch.tensor([torch.log(torch.tensor(4.0))])
        assert torch.allclose(entropy, expected, atol=1e-5)

    def test_peaked_distribution(self):
        # Very peaked distribution -> entropy near 0
        logits = torch.tensor([[100.0, -100.0, -100.0]])
        entropy = entropy_from_logits(logits)
        assert entropy.item() < 0.01

    def test_shape_preserved(self):
        logits = torch.randn(2, 5, 10)  # (batch, seq_len, vocab)
        entropy = entropy_from_logits(logits)
        assert entropy.shape == (2, 5)

    def test_non_negative(self):
        logits = torch.randn(3, 8)
        entropy = entropy_from_logits(logits)
        assert (entropy >= -1e-6).all()

    def test_shift_invariance(self):
        logits = torch.randn(2, 5)
        shifted = logits + 100.0
        assert torch.allclose(
            entropy_from_logits(logits),
            entropy_from_logits(shifted),
            atol=1e-5,
        )


class TestAggLoss:
    def test_token_mean(self):
        loss_mat = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        loss_mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])
        result = agg_loss(loss_mat, loss_mask, token_reduce="sum", batch_reduce="token-mean")
        # masked values: 1, 2, 4, 5, 6 -> mean = 18/5 = 3.6
        assert torch.allclose(result, torch.tensor(3.6))

    def test_step_mean_token_mean(self):
        loss_mat = torch.tensor([[2.0, 4.0], [6.0, 8.0]])
        loss_mask = torch.ones(2, 2)
        result = agg_loss(loss_mat, loss_mask, token_reduce="mean", batch_reduce="step-mean")
        # seq0: (2+4)/2 = 3, seq1: (6+8)/2 = 7, mean = (3+7)/2 = 5
        assert torch.allclose(result, torch.tensor(5.0))

    def test_step_mean_token_sum(self):
        loss_mat = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        loss_mask = torch.ones(2, 2)
        result = agg_loss(loss_mat, loss_mask, token_reduce="sum", batch_reduce="step-mean")
        # seq0: 1+2=3, seq1: 3+4=7, mean = (3+7)/2 = 5
        assert torch.allclose(result, torch.tensor(5.0))

    def test_step_mean_token_mean_norm(self):
        loss_mat = torch.tensor([[1.0, 2.0, 0.0], [3.0, 4.0, 0.0]])
        loss_mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
        result = agg_loss(loss_mat, loss_mask, token_reduce="mean-norm", batch_reduce="step-mean")
        # mean-norm divides by T=3 (full dim), not valid count
        # seq0: (1+2)/3 = 1.0, seq1: (3+4)/3 = 7/3, mean = (1 + 7/3)/2 = 10/6 = 5/3
        expected = (1.0 + 7.0 / 3.0) / 2.0
        assert torch.allclose(result, torch.tensor(expected))

    def test_mask_zeros_out_padded_tokens(self):
        loss_mat = torch.tensor([[1.0, 999.0]])
        loss_mask = torch.tensor([[1.0, 0.0]])
        result = agg_loss(loss_mat, loss_mask, token_reduce="sum", batch_reduce="token-mean")
        assert torch.allclose(result, torch.tensor(1.0))

    def test_with_num_program_tokens(self):
        loss_mat = torch.tensor([[1.0, 2.0, 3.0]])
        loss_mask = torch.ones(1, 3)
        num_program_tokens = torch.tensor([6.0])
        result = agg_loss(
            loss_mat,
            loss_mask,
            token_reduce="mean-program",
            batch_reduce="step-mean",
            num_program_tokens=num_program_tokens,
        )
        # token sum = 6, divided by num_program_tokens=6 -> 1.0, mean over 1 row = 1.0
        assert torch.allclose(result, torch.tensor(1.0))

    def test_program_mean_token_mean(self):
        """program-mean + token_reduce=mean: mean of per-step means, averaged over programs.

        /nps is applied because "mean" is per-step, so we need to average
        a program's step-means to get its contribution.
        """
        loss_mat = torch.tensor([[2.0, 4.0], [6.0, 8.0], [10.0, 12.0]])
        loss_mask = torch.ones(3, 2)
        num_program_steps = torch.tensor([2.0, 2.0, 1.0])
        result = agg_loss(
            loss_mat,
            loss_mask,
            token_reduce="mean",
            batch_reduce="program-mean",
            num_program_steps=num_program_steps,
            valid_program_count=2,
        )
        # row0: mean=3.0, /nps=2 → 1.5   (program A, step 1)
        # row1: mean=7.0, /nps=2 → 3.5   (program A, step 2)
        # row2: mean=11.0, /nps=1 → 11.0 (program B, step 1)
        # sum = 1.5 + 3.5 + 11.0 = 16.0
        # / 2 programs = 8.0
        assert torch.allclose(result, torch.tensor(8.0))

    def test_program_mean_token_mean_program(self):
        """program-mean + token_reduce=mean-program: per-program token mean, averaged over programs.

        /nps IS applied (program-mean always divides by nps).
        """
        # Program A: 2 steps, 4 total tokens, total loss = 2+4+6+8 = 20
        # Program B: 1 step,  2 total tokens, total loss = 10+12 = 22
        loss_mat = torch.tensor([[2.0, 4.0], [6.0, 8.0], [10.0, 12.0]])
        loss_mask = torch.ones(3, 2)
        num_program_steps = torch.tensor([2.0, 2.0, 1.0])
        num_program_tokens = torch.tensor([4.0, 4.0, 2.0])
        result = agg_loss(
            loss_mat,
            loss_mask,
            token_reduce="mean-program",
            batch_reduce="program-mean",
            num_program_tokens=num_program_tokens,
            num_program_steps=num_program_steps,
            valid_program_count=2,
        )
        # row0: sum=6  / T_A=4 → 1.5,  /nps=2 → 0.75
        # row1: sum=14 / T_A=4 → 3.5,  /nps=2 → 1.75
        # row2: sum=22 / T_B=2 → 11.0, /nps=1 → 11.0
        # sum = 0.75 + 1.75 + 11.0 = 13.5
        # / 2 programs = 6.75
        assert torch.allclose(result, torch.tensor(6.75))

    def test_program_mean_requires_valid_program_count(self):
        """program-mean must have valid_program_count (programs can span micro-batches)."""
        loss_mat = torch.tensor([[1.0, 2.0]])
        loss_mask = torch.ones(1, 2)
        with pytest.raises(AssertionError, match="valid_program_count required"):
            agg_loss(
                loss_mat,
                loss_mask,
                token_reduce="mean",
                batch_reduce="program-mean",
                num_program_steps=torch.tensor([1.0]),
            )

    def test_program_mean_with_padding(self):
        """Padding rows should not count as programs."""
        loss_mat = torch.tensor([[2.0, 4.0], [6.0, 8.0], [999.0, 999.0]])
        loss_mask = torch.tensor([[1.0, 1.0], [1.0, 1.0], [0.0, 0.0]])
        num_program_steps = torch.tensor([2.0, 2.0, 1.0])
        result = agg_loss(
            loss_mat,
            loss_mask,
            token_reduce="mean",
            batch_reduce="program-mean",
            num_program_steps=num_program_steps,
            valid_program_count=1,  # only program A is valid
        )
        # row0: mean=3.0, /nps=2 → 1.5  (program A, step 1)
        # row1: mean=7.0, /nps=2 → 3.5  (program A, step 2)
        # row2: padding → 0
        # sum = 5.0,  / 1 program = 5.0
        assert torch.allclose(result, torch.tensor(5.0))

    def test_step_mean_with_padding_rows(self):
        """Padding rows (all-zero mask) should not dilute batch mean."""
        loss_mat = torch.tensor([[1.0, 2.0], [3.0, 4.0], [999.0, 999.0]])
        loss_mask = torch.tensor([[1.0, 1.0], [1.0, 1.0], [0.0, 0.0]])
        result = agg_loss(loss_mat, loss_mask, token_reduce="mean", batch_reduce="step-mean")
        # row0: (1+2)/2 = 1.5, row1: (3+4)/2 = 3.5, row2: padding → 0
        # mean over 2 valid rows = (1.5 + 3.5) / 2 = 2.5
        assert torch.allclose(result, torch.tensor(2.5))

    def test_token_mean_with_padding_rows(self):
        """token-mean ignores padding rows (mask=0 contributes 0 to both num/denom)."""
        loss_mat = torch.tensor([[1.0, 2.0], [3.0, 4.0], [999.0, 999.0]])
        loss_mask = torch.tensor([[1.0, 1.0], [1.0, 1.0], [0.0, 0.0]])
        result = agg_loss(loss_mat, loss_mask, token_reduce="sum", batch_reduce="token-mean")
        # valid tokens: 1, 2, 3, 4 → mean = 10/4 = 2.5
        assert torch.allclose(result, torch.tensor(2.5))

    def test_all_padding_rows(self):
        """All-padding batch should return 0, not NaN."""
        loss_mat = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        loss_mask = torch.zeros(2, 2)
        result = agg_loss(loss_mat, loss_mask, token_reduce="mean", batch_reduce="step-mean")
        assert torch.allclose(result, torch.tensor(0.0))
        assert not torch.isnan(result)

    def test_invalid_token_reduce_raises(self):
        loss_mat = torch.ones(2, 3)
        loss_mask = torch.ones(2, 3)
        with pytest.raises(ValueError, match="Invalid token_reduce"):
            agg_loss(loss_mat, loss_mask, token_reduce="invalid-mode")

    def test_invalid_batch_reduce_raises(self):
        loss_mat = torch.ones(2, 3)
        loss_mask = torch.ones(2, 3)
        with pytest.raises(ValueError, match="Invalid batch_reduce"):
            agg_loss(loss_mat, loss_mask, batch_reduce="invalid-mode")
