"""
Comprehensive tests for axon.utils.rl.advantage -- group-wise helpers for RL training.
"""

import numpy as np
import pytest
import torch

from axon.utils.rl.advantage import (
    _to_1d_numpy_object_array,
    as_torch_index,
    group_mean_std,
    masked_mean,
    masked_var,
    masked_whiten,
)

# ===================================================================
#  _to_1d_numpy_object_array
# ===================================================================


class TestTo1dNumpyObjectArray:
    def test_2d_array_flattened(self):
        arr = _to_1d_numpy_object_array(np.array([[1, 2], [3, 4]]))
        assert arr.ndim == 1 and len(arr) == 4

    def test_3d_array_flattened(self):
        arr = _to_1d_numpy_object_array(np.ones((2, 3, 4)))
        assert arr.ndim == 1 and len(arr) == 24

    def test_scalar(self):
        arr = _to_1d_numpy_object_array(42)
        assert arr.ndim == 1
        # A scalar turned into a 0-d array then reshaped to 1-d gives length 1
        assert len(arr) == 1

    def test_strings(self):
        arr = _to_1d_numpy_object_array(["a", "b", "c"])
        assert arr.ndim == 1 and len(arr) == 3

    def test_mixed_types(self):
        arr = _to_1d_numpy_object_array([1, "two", 3.0])
        assert arr.ndim == 1 and len(arr) == 3


# ===================================================================
#  as_torch_index
# ===================================================================


class TestAsTorchIndex:
    def test_string_labels(self):
        idx = as_torch_index(["a", "b", "a", "c"], device="cpu")
        assert idx.dtype == torch.long
        assert idx.shape == (4,)
        # Same labels should map to same index
        assert idx[0].item() == idx[2].item()
        assert idx[0].item() != idx[1].item()

    def test_torch_tensor_bool(self):
        t = torch.tensor([True, False, True])
        idx = as_torch_index(t, device="cpu")
        assert idx.dtype == torch.long
        assert idx.tolist() == [1, 0, 1]

    def test_torch_tensor_float_roundable(self):
        t = torch.tensor([0.0, 1.0, 2.0])
        idx = as_torch_index(t, device="cpu")
        assert idx.dtype == torch.long
        assert idx.tolist() == [0, 1, 2]

    def test_torch_tensor_float_not_roundable(self):
        # Non-integer float values should be treated as string labels
        t = torch.tensor([0.1, 0.2, 0.1])
        idx = as_torch_index(t, device="cpu")
        assert idx.dtype == torch.long
        # 0.1 appears twice, should map to same index
        assert idx[0].item() == idx[2].item()

    def test_numpy_string_labels(self):
        arr = np.array(["x", "y", "x"], dtype=object)
        idx = as_torch_index(arr, device="cpu")
        assert idx.dtype == torch.long
        assert idx[0].item() == idx[2].item()
        assert idx[0].item() != idx[1].item()

    def test_result_is_contiguous(self):
        idx = as_torch_index([2, 0, 1], device="cpu")
        assert idx.is_contiguous()

    def test_multidim_tensor_flattened(self):
        t = torch.tensor([[0, 1], [2, 3]])
        idx = as_torch_index(t, device="cpu")
        assert idx.ndim == 1
        assert idx.shape == (4,)

    def test_bfloat16_tensor(self):
        t = torch.tensor([0.0, 1.0, 2.0], dtype=torch.bfloat16)
        idx = as_torch_index(t, device="cpu")
        assert idx.dtype == torch.long
        assert idx.tolist() == [0, 1, 2]


# ===================================================================
#  group_mean_std
# ===================================================================


class TestGroupMeanStd:
    def test_single_group_std(self):
        scores = torch.tensor([1.0, 2.0, 3.0])
        gidx = torch.tensor([0, 0, 0])
        mean, std, count = group_mean_std(scores, gidx, device="cpu")
        # Bessel corrected std: sqrt(sum((x-2)^2) / (3-1)) = sqrt(2/2) = 1.0
        assert std[0].item() == pytest.approx(1.0)

    def test_two_groups_std(self):
        scores = torch.tensor([1.0, 3.0, 2.0, 4.0])
        gidx = torch.tensor([0, 0, 1, 1])
        mean, std, count = group_mean_std(scores, gidx, device="cpu")
        # Group 0: var = (1-2)^2 + (3-2)^2 / (2-1) = 2/1 = 2, std = sqrt(2)
        assert std[0].item() == pytest.approx(2**0.5, abs=1e-5)
        # Group 1: var = (2-3)^2 + (4-3)^2 / (2-1) = 2/1 = 2, std = sqrt(2)
        assert std[1].item() == pytest.approx(2**0.5, abs=1e-5)

    def test_singleton_group(self):
        scores = torch.tensor([5.0])
        gidx = torch.tensor([0])
        mean, std, count = group_mean_std(scores, gidx, device="cpu")
        # Singleton: mean=0, std=1
        assert mean[0].item() == pytest.approx(0.0)
        assert std[0].item() == pytest.approx(1.0)
        assert count[0].item() == pytest.approx(1.0)

    def test_mixed_singleton_and_multi(self):
        # Group 0 has 1 element, group 1 has 3
        scores = torch.tensor([10.0, 1.0, 2.0, 3.0])
        gidx = torch.tensor([0, 1, 1, 1])
        mean, std, count = group_mean_std(scores, gidx, device="cpu")
        # Singleton group 0: mean=0, std=1
        assert mean[0].item() == pytest.approx(0.0)
        assert std[0].item() == pytest.approx(1.0)
        # Multi group 1: mean=2
        assert mean[1].item() == pytest.approx(2.0)
        assert count[1].item() == pytest.approx(3.0)

    def test_empty(self):
        scores = torch.tensor([])
        gidx = torch.tensor([], dtype=torch.long)
        mean, std, count = group_mean_std(scores, gidx, device="cpu")
        assert mean.numel() == 0
        assert std.numel() == 0
        assert count.numel() == 0

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="mismatch"):
            group_mean_std(torch.tensor([1.0, 2.0]), torch.tensor([0]), device="cpu")

    def test_three_groups_uneven(self):
        scores = torch.tensor([1.0, 2.0, 3.0, 10.0, 20.0])
        gidx = torch.tensor([0, 0, 0, 1, 2])
        mean, std, count = group_mean_std(scores, gidx, device="cpu")
        assert mean[0].item() == pytest.approx(2.0)
        assert count[0].item() == pytest.approx(3.0)
        # Group 1: singleton -> mean=0, std=1
        assert mean[1].item() == pytest.approx(0.0)
        assert std[1].item() == pytest.approx(1.0)
        # Group 2: singleton -> mean=0, std=1
        assert mean[2].item() == pytest.approx(0.0)
        assert std[2].item() == pytest.approx(1.0)

    def test_sparse_group_ids_create_empty_intermediate_groups(self):
        """Groups [0, 5] with gap create G=6, groups 1-4 have count=0 and thus nan-safe mean."""
        scores = torch.tensor([10.0, 20.0])
        gidx = torch.tensor([0, 5])
        mean, std, count = group_mean_std(scores, gidx, device="cpu")
        assert count.shape == (6,)
        # Empty groups 1-4 have count=0
        for i in range(1, 5):
            assert count[i].item() == 0.0
        # Singleton groups: mean=0, std=1
        assert mean[0].item() == pytest.approx(0.0)
        assert mean[5].item() == pytest.approx(0.0)

    def test_float64_precision(self):
        """Scores are cast to float32 internally; verify large values don't lose precision badly."""
        scores = torch.tensor([1e7, 1e7 + 1, 1e7 + 2], dtype=torch.float64)
        gidx = torch.tensor([0, 0, 0])
        mean, std, count = group_mean_std(scores, gidx, device="cpu")
        # Mean should be ~1e7+1, but float32 has ~7 decimal digits of precision
        assert abs(mean[0].item() - (1e7 + 1)) < 10  # within float32 precision

    def test_all_same_scores(self):
        scores = torch.tensor([5.0, 5.0, 5.0])
        gidx = torch.tensor([0, 0, 0])
        mean, std, count = group_mean_std(scores, gidx, device="cpu")
        assert mean[0].item() == pytest.approx(5.0)
        # Var numerator is 0, eps prevents std from being 0
        assert std[0].item() == pytest.approx(1e-6**0.5, abs=1e-4)

    def test_bessel_correction_matches_torch(self):
        """Verify Bessel-corrected std matches torch.std for non-singleton groups."""
        scores = torch.tensor([1.0, 3.0, 5.0, 7.0])
        gidx = torch.tensor([0, 0, 0, 0])
        mean, std, _ = group_mean_std(scores, gidx, device="cpu")
        expected_std = torch.std(scores).item()
        assert std[0].item() == pytest.approx(expected_std, abs=1e-4)

    def test_large_group_ids(self):
        # gidx values don't need to be contiguous; max determines G
        scores = torch.tensor([1.0, 2.0])
        gidx = torch.tensor([0, 5])
        mean, std, count = group_mean_std(scores, gidx, device="cpu")
        # G = 6 (0..5)
        assert mean.shape == (6,)
        assert count[0].item() == pytest.approx(1.0)
        assert count[5].item() == pytest.approx(1.0)
        # Groups 1-4 have count=0
        assert count[1].item() == pytest.approx(0.0)


# ===================================================================
#  masked_mean
# ===================================================================


class TestMaskedMean:
    def test_partial_mask(self):
        values = torch.tensor([1.0, 2.0, 3.0, 4.0])
        mask = torch.tensor([1.0, 0.0, 1.0, 0.0])
        assert masked_mean(values, mask).item() == pytest.approx(2.0)

    def test_zero_mask(self):
        values = torch.tensor([1.0, 2.0])
        mask = torch.tensor([0.0, 0.0])
        # Should not raise, clamp_min prevents div by zero
        result = masked_mean(values, mask)
        assert result.item() == pytest.approx(0.0)

    def test_single_element_selected(self):
        values = torch.tensor([10.0, 20.0, 30.0])
        mask = torch.tensor([0.0, 1.0, 0.0])
        assert masked_mean(values, mask).item() == pytest.approx(20.0)

    def test_weighted_mask(self):
        """Mask values > 1 act as weights (mask is cast to values.dtype)."""
        values = torch.tensor([1.0, 2.0, 3.0])
        mask = torch.tensor([2.0, 0.0, 1.0])  # weight 2 on first, 1 on third
        # weighted_sum = 1*2 + 2*0 + 3*1 = 5, weight_sum = 3
        assert masked_mean(values, mask).item() == pytest.approx(5.0 / 3.0)


# ===================================================================
#  masked_var
# ===================================================================


class TestMaskedVar:
    def test_basic_variance_bessel(self):
        values = torch.tensor([1.0, 2.0, 3.0])
        mask = torch.ones(3)
        var = masked_var(values, mask)
        # Bessel corrected: population_var * n/(n-1) = (2/3) * 3/2 = 1.0
        assert var.item() == pytest.approx(1.0)

    def test_no_bessel(self):
        values = torch.tensor([1.0, 2.0, 3.0])
        mask = torch.ones(3)
        var = masked_var(values, mask, bessel_correction=False)
        # Population variance: sum((x-2)^2)/3 = 2/3
        assert var.item() == pytest.approx(2.0 / 3.0)

    def test_all_same_values(self):
        values = torch.tensor([5.0, 5.0, 5.0])
        mask = torch.ones(3)
        var = masked_var(values, mask)
        assert var.item() == pytest.approx(0.0)

    def test_partial_mask(self):
        values = torch.tensor([1.0, 100.0, 3.0])
        mask = torch.tensor([1.0, 0.0, 1.0])
        # Masked values: [1.0, 3.0], mean=2.0
        # Population var: ((1-2)^2 + (3-2)^2) / 2 = 1.0
        # Bessel: 1.0 * 2 / (2-1) = 2.0
        var = masked_var(values, mask, bessel_correction=True)
        assert var.item() == pytest.approx(2.0)

    def test_single_element_no_bessel_correction(self):
        values = torch.tensor([5.0, 10.0])
        mask = torch.tensor([1.0, 0.0])
        # Only one element -> bessel correction not applied (mask_sum == 1, not > 1)
        var = masked_var(values, mask, bessel_correction=True)
        assert var.item() == pytest.approx(0.0)

    def test_zero_mask_returns_zero(self):
        values = torch.tensor([1.0, 2.0])
        mask = torch.tensor([0.0, 0.0])
        var = masked_var(values, mask)
        # With clamp_min(1.0), should be 0.0
        assert var.item() == pytest.approx(0.0, abs=1e-5)

    def test_large_variance(self):
        values = torch.tensor([0.0, 100.0])
        mask = torch.ones(2)
        var = masked_var(values, mask, bessel_correction=False)
        # mean = 50, pop_var = (50^2 + 50^2)/2 = 2500
        assert var.item() == pytest.approx(2500.0)

    def test_bessel_vs_no_bessel_relationship(self):
        """Bessel-corrected variance = pop_variance * n / (n-1)."""
        values = torch.tensor([1.0, 2.0, 3.0, 4.0])
        mask = torch.ones(4)
        pop_var = masked_var(values, mask, bessel_correction=False).item()
        bessel_var = masked_var(values, mask, bessel_correction=True).item()
        assert bessel_var == pytest.approx(pop_var * 4.0 / 3.0)


# ===================================================================
#  masked_whiten
# ===================================================================


class TestMaskedWhiten:
    def test_zero_mean_unit_var(self):
        values = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        mask = torch.ones(5)
        whitened = masked_whiten(values, mask, shift_mean=True)
        assert whitened.mean().item() == pytest.approx(0.0, abs=1e-5)

    def test_unit_variance_after_whiten(self):
        values = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float32)
        mask = torch.ones(5)
        whitened = masked_whiten(values, mask, shift_mean=True)
        # Whitened values should have approximately unit variance
        var = whitened.var(unbiased=False).item()
        assert var == pytest.approx(1.0, abs=0.2)

    def test_no_shift_mean(self):
        values = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        mask = torch.ones(5)
        whitened = masked_whiten(values, mask, shift_mean=False)
        # Mean should be approximately the original mean (3.0) added back
        assert whitened.mean().item() == pytest.approx(3.0, abs=0.5)

    def test_shift_mean_true_vs_false_difference(self):
        values = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        mask = torch.ones(5)
        w_shifted = masked_whiten(values, mask, shift_mean=True)
        w_not_shifted = masked_whiten(values, mask, shift_mean=False)
        # The difference should be the original mean
        diff = (w_not_shifted - w_shifted).mean().item()
        assert diff == pytest.approx(3.0, abs=0.5)

    def test_constant_values(self):
        values = torch.tensor([5.0, 5.0, 5.0])
        mask = torch.ones(3)
        whitened = masked_whiten(values, mask, shift_mean=True)
        # All values equal -> (x - mean) = 0 for all, so whitened should be ~0
        assert torch.allclose(whitened, torch.zeros(3), atol=1e-3)

    def test_partial_mask_stats_applied_to_all(self):
        """Stats come from masked subset, but whitening is applied to ALL values."""
        values = torch.tensor([1.0, 100.0, 3.0])
        mask = torch.tensor([1.0, 0.0, 1.0])
        whitened = masked_whiten(values, mask, shift_mean=True)
        # Stats from [1, 3]: mean=2, var=2 (bessel), std=sqrt(2)
        # whitened[1] = (100 - 2) / sqrt(2 + 1e-8) ~ 69.3 (huge because 100 is an outlier unmasked value)
        assert abs(whitened[1].item()) > 50

    def test_whitened_values_sorted_order_preserved(self):
        values = torch.tensor([1.0, 5.0, 3.0, 2.0, 4.0])
        mask = torch.ones(5)
        whitened = masked_whiten(values, mask, shift_mean=True)
        # Relative ordering should be preserved
        sorted_orig = torch.argsort(values)
        sorted_whitened = torch.argsort(whitened)
        assert sorted_orig.tolist() == sorted_whitened.tolist()

    def test_two_element(self):
        values = torch.tensor([0.0, 10.0])
        mask = torch.ones(2)
        whitened = masked_whiten(values, mask, shift_mean=True)
        # mean=5, bessel var = 50, std=~7.07
        # whitened[0] = (0-5)/std, whitened[1] = (10-5)/std
        # Should be symmetric around 0
        assert whitened[0].item() == pytest.approx(-whitened[1].item(), abs=1e-5)

    def test_all_zero_mask_produces_large_values(self):
        """With all-zero mask, mean=0 and var=0, so rsqrt(0+1e-8)=10000.
        This means whitened = (values - 0) * 10000, producing huge values.
        Callers must ensure the mask is not all-zero.
        """
        values = torch.tensor([1.0, 2.0, 3.0])
        mask = torch.zeros(3)
        whitened = masked_whiten(values, mask, shift_mean=True)
        # rsqrt(1e-8) = 10000, so values get scaled by ~10000
        assert torch.all(torch.abs(whitened) > 1000)


class TestAsTorchIndexEdgeCases:
    def test_integer_tensor_passed_through_raw(self):
        """Integer tensors are passed through as-is, NOT remapped to contiguous 0..G-1.
        This means non-zero-based labels like [5, 10] create G=11 groups in group_mean_std.
        """
        index = torch.tensor([5, 5, 10, 10])
        result = as_torch_index(index, device="cpu")
        # Passed through raw: values stay as [5, 5, 10, 10]
        assert result.tolist() == [5, 5, 10, 10]

    def test_integer_list_passed_through_raw(self):
        """Integer lists are also passed through raw, not remapped."""
        result = as_torch_index([100, 100, 200, 200], device="cpu")
        assert result.tolist() == [100, 100, 200, 200]

    def test_mixed_type_labels(self):
        """Mixed types (str/int/None) in group labels should not crash."""
        result = as_torch_index(["a", 1, None, "a"], device="cpu")
        assert result.dtype == torch.long
        assert result[0].item() == result[3].item()  # "a" == "a"

    def test_none_labels(self):
        """None values as group labels should be handled."""
        result = as_torch_index([None, None, None], device="cpu")
        assert result.dtype == torch.long
        # All same label → all same index
        assert result[0].item() == result[1].item() == result[2].item()

    def test_numpy_float_non_roundable_uses_string_labels(self):
        """Non-integer numpy floats fall through to unique-based labeling."""
        arr = np.array([0.5, 1.5, 0.5])
        idx = as_torch_index(arr, device="cpu")
        assert idx[0].item() == idx[2].item()
        assert idx[0].item() != idx[1].item()

    def test_uint8_tensor(self):
        t = torch.tensor([0, 1, 255], dtype=torch.uint8)
        idx = as_torch_index(t, device="cpu")
        assert idx.dtype == torch.long
        assert idx.tolist() == [0, 1, 255]

    def test_negative_integer_tensor_passed_through(self):
        """Negative integer indices are passed through raw.
        This would crash in downstream group_mean_std (index_add doesn't support negatives).
        Callers must ensure non-negative indices for integer tensors.
        """
        index = torch.tensor([-1, -1, 0, 0])
        result = as_torch_index(index, device="cpu")
        # Passed through as-is: [-1, -1, 0, 0]
        assert result.tolist() == [-1, -1, 0, 0]


class TestGroupMeanStdEdgeCases:
    def test_inf_scores(self):
        """Inf in scores should not crash."""
        scores = torch.tensor([1.0, float("inf"), 3.0])
        gidx = torch.tensor([0, 0, 0])
        mean, std, count = group_mean_std(scores, gidx, device="cpu")
        assert count[0].item() == 3.0

    def test_very_large_group_count(self):
        """Many groups should work without memory issues."""
        N = 1000
        scores = torch.randn(N)
        gidx = torch.arange(N)  # Each element is its own group (all singletons)
        mean, std, count = group_mean_std(scores, gidx, device="cpu")
        assert mean.shape == (N,)
        # All singletons → mean=0, std=1
        assert torch.allclose(mean, torch.zeros(N), atol=1e-5)
        assert torch.allclose(std, torch.ones(N), atol=1e-5)
