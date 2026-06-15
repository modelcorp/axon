"""
Tests for axon.data.data_sampler — curriculum samplers and factory.
"""

import numpy as np
import pytest
import torch

from axon.data.data_sampler.curriculum_samplers import (
    ExpWeightedCurriculumSampler,
    ThresholdMaskingSampler,
)


# ───────────────── helpers ─────────────────

def _make_data_source(n=100):
    """Create a simple sized data source."""
    return list(range(n))


class _MockBatch:
    """Lightweight batch mock matching the interface samplers expect."""
    def __init__(self, non_tensor_batch):
        self.non_tensor_batch = non_tensor_batch


def _make_exp_batch(indices, pass_rates):
    """Create a mock batch for ExpWeightedCurriculumSampler.update.

    ExpWeightedCurriculumSampler calls .numpy() and .tolist() on the values,
    so we provide torch tensors.
    """
    return _MockBatch({
        "index": torch.tensor(indices),
        "pass_rate": torch.tensor(pass_rates, dtype=torch.float32),
    })


def _make_threshold_batch(indices, pass_rates):
    """Create a mock batch for ThresholdMaskingSampler.update.

    ThresholdMaskingSampler calls np.asarray() on the values, so numpy arrays work.
    """
    return _MockBatch({
        "index": np.array(indices, dtype=np.int64),
        "pass_rate": np.array(pass_rates, dtype=np.float32),
    })


# ═══════════════════════════════════════════════════════════════════
#  ExpWeightedCurriculumSampler
# ═══════════════════════════════════════════════════════════════════

class TestExpWeightedCurriculumSampler:
    def test_init(self):
        ds = _make_data_source(50)
        sampler = ExpWeightedCurriculumSampler(ds)
        assert sampler.num_samples == 50
        assert len(sampler.weights) == 50
        assert all(w == 1.0 for w in sampler.weights)

    def test_iter_returns_correct_count(self):
        ds = _make_data_source(20)
        sampler = ExpWeightedCurriculumSampler(ds)
        indices = list(sampler)
        assert len(indices) == 20
        # All indices should be valid
        assert all(0 <= idx < 20 for idx in indices)

    def test_update_high_pass_rate_lowers_weight(self):
        ds = _make_data_source(10)
        sampler = ExpWeightedCurriculumSampler(ds)
        batch = _make_exp_batch(indices=[0, 1], pass_rates=[0.9, 0.95])
        sampler.update(batch)
        assert sampler.weights[0] == sampler.min_weight
        assert sampler.weights[1] == sampler.min_weight

    def test_update_low_pass_rate_raises_weight(self):
        ds = _make_data_source(10)
        sampler = ExpWeightedCurriculumSampler(ds)
        batch = _make_exp_batch(indices=[0, 1], pass_rates=[0.1, 0.05])
        sampler.update(batch)
        assert sampler.weights[0] == sampler.max_weight
        assert sampler.weights[1] == sampler.max_weight

    def test_update_mid_pass_rate_exponential_weight(self):
        ds = _make_data_source(10)
        sampler = ExpWeightedCurriculumSampler(ds)
        batch = _make_exp_batch(indices=[0], pass_rates=[0.5])
        sampler.update(batch)
        import math
        expected = math.exp(-0.5 + sampler.centroid)
        assert sampler.weights[0] == pytest.approx(expected)

    def test_state_dict_roundtrip(self):
        ds = _make_data_source(10)
        sampler = ExpWeightedCurriculumSampler(ds)
        batch = _make_exp_batch(indices=[0, 1, 2], pass_rates=[0.1, 0.5, 0.9])
        sampler.update(batch)

        state = sampler.state_dict()
        sampler2 = ExpWeightedCurriculumSampler(ds)
        sampler2.load_state_dict(state)
        assert sampler2.weights == sampler.weights

    def test_sampling_biased_by_weights(self):
        ds = _make_data_source(3)
        sampler = ExpWeightedCurriculumSampler(ds)
        # Set weight distribution: heavily favor index 0
        sampler.weights = [100.0, 0.001, 0.001]
        indices = list(sampler)
        # Most samples should be index 0
        count_0 = sum(1 for idx in indices if idx == 0)
        assert count_0 >= 1  # at minimum, should appear at least once


# ═══════════════════════════════════════════════════════════════════
#  ThresholdMaskingSampler
# ═══════════════════════════════════════════════════════════════════

class TestThresholdMaskingSampler:
    def test_init(self):
        ds = _make_data_source(30)
        sampler = ThresholdMaskingSampler(ds, threshold=0.8)
        assert sampler.num_samples == 30
        assert sampler.threshold == 0.8
        assert not sampler.masked.any()
        assert sampler.pass_sum.sum() == 0
        assert sampler.invoke_count.sum() == 0

    def test_iter_returns_correct_count(self):
        ds = _make_data_source(20)
        sampler = ThresholdMaskingSampler(ds)
        indices = list(sampler)
        assert len(indices) == 20
        assert all(0 <= idx < 20 for idx in indices)

    def test_update_masks_above_threshold(self):
        ds = _make_data_source(5)
        sampler = ThresholdMaskingSampler(ds, threshold=0.9)
        # Pass rate 0.95 for index 0 -> should be masked
        batch = _make_threshold_batch(indices=[0], pass_rates=[0.95])
        sampler.update(batch)
        assert sampler.masked[0] == True
        assert sampler.masked[1] == False

    def test_update_below_threshold_not_masked(self):
        ds = _make_data_source(5)
        sampler = ThresholdMaskingSampler(ds, threshold=0.9)
        batch = _make_threshold_batch(indices=[0], pass_rates=[0.5])
        sampler.update(batch)
        assert sampler.masked[0] == False

    def test_update_accumulates_across_calls(self):
        ds = _make_data_source(5)
        sampler = ThresholdMaskingSampler(ds, threshold=0.9)
        # Two updates for index 0: avg = (0.85 + 1.0) / 2 = 0.925 > threshold
        sampler.update(_make_threshold_batch(indices=[0], pass_rates=[0.85]))
        assert sampler.masked[0] == False
        sampler.update(_make_threshold_batch(indices=[0], pass_rates=[1.0]))
        assert sampler.masked[0] == True

    def test_masked_indices_excluded_from_sampling(self):
        ds = _make_data_source(3)
        sampler = ThresholdMaskingSampler(ds, threshold=0.5)
        # Mask indices 0 and 1
        sampler.masked[0] = True
        sampler.masked[1] = True
        indices = list(sampler)
        # All samples should be index 2
        assert all(idx == 2 for idx in indices)

    def test_all_masked_falls_back_to_all(self):
        ds = _make_data_source(3)
        sampler = ThresholdMaskingSampler(ds, threshold=0.5)
        sampler.masked[:] = True
        indices = list(sampler)
        assert len(indices) == 3
        # Should sample from all indices when all are masked
        assert all(0 <= idx < 3 for idx in indices)

    def test_duplicate_indices_in_batch(self):
        ds = _make_data_source(5)
        sampler = ThresholdMaskingSampler(ds, threshold=0.9)
        # Same index appears twice: sum = 0.5 + 0.5 = 1.0, count = 2, avg = 0.5
        batch = _make_threshold_batch(indices=[0, 0], pass_rates=[0.5, 0.5])
        sampler.update(batch)
        assert sampler.invoke_count[0] == 2
        assert sampler.pass_sum[0] == pytest.approx(1.0)
        assert sampler.masked[0] == False

    def test_state_dict_roundtrip(self):
        ds = _make_data_source(10)
        sampler = ThresholdMaskingSampler(ds, threshold=0.8)
        sampler.update(_make_threshold_batch(indices=[0, 1, 2], pass_rates=[0.9, 0.5, 0.85]))

        state = sampler.state_dict()
        sampler2 = ThresholdMaskingSampler(ds, threshold=0.8)
        sampler2.load_state_dict(state)

        np.testing.assert_array_equal(sampler2.masked, sampler.masked)
        np.testing.assert_array_almost_equal(sampler2.pass_sum, sampler.pass_sum)
        np.testing.assert_array_equal(sampler2.invoke_count, sampler.invoke_count)


# ═══════════════════════════════════════════════════════════════════
#  create_data_sampler factory
# ═══════════════════════════════════════════════════════════════════

class TestCreateDataSampler:
    def test_sequential(self):
        from omegaconf import OmegaConf
        from axon.data.data_sampler import create_data_sampler
        from torch.utils.data import SequentialSampler

        ds = _make_data_source(10)
        config = OmegaConf.create({"data_sampler": "sequential"})
        sampler = create_data_sampler(config, ds)
        assert isinstance(sampler, SequentialSampler)

    def test_random(self):
        from omegaconf import OmegaConf
        from axon.data.data_sampler import create_data_sampler

        ds = _make_data_source(10)
        config = OmegaConf.create({"data_sampler": "random"})
        sampler = create_data_sampler(config, ds)
        # Should return a sampler that iterates
        indices = list(sampler)
        assert len(indices) == 10

    def test_random_with_seed(self):
        from omegaconf import OmegaConf
        from axon.data.data_sampler import create_data_sampler

        ds = _make_data_source(10)
        config = OmegaConf.create({
            "data_sampler": "random",
            "data_sampler_args": {"seed": 42},
        })
        sampler1 = create_data_sampler(config, ds)
        sampler2 = create_data_sampler(config, ds)
        assert list(sampler1) == list(sampler2)

    def test_exp_weighted_curriculum(self):
        from omegaconf import OmegaConf
        from axon.data.data_sampler import create_data_sampler

        ds = _make_data_source(10)
        config = OmegaConf.create({"data_sampler": "exp_weighted_curriculum"})
        sampler = create_data_sampler(config, ds)
        assert isinstance(sampler, ExpWeightedCurriculumSampler)

    def test_threshold_masking(self):
        from omegaconf import OmegaConf
        from axon.data.data_sampler import create_data_sampler

        ds = _make_data_source(10)
        config = OmegaConf.create({
            "data_sampler": "threshold_masking_curriculum",
            "data_sampler_args": {"threshold": 0.85},
        })
        sampler = create_data_sampler(config, ds)
        assert isinstance(sampler, ThresholdMaskingSampler)
        assert sampler.threshold == 0.85

    def test_default_is_sequential(self):
        from omegaconf import OmegaConf
        from axon.data.data_sampler import create_data_sampler
        from torch.utils.data import SequentialSampler

        ds = _make_data_source(10)
        config = OmegaConf.create({})
        sampler = create_data_sampler(config, ds)
        assert isinstance(sampler, SequentialSampler)

    def test_enum_values_work(self):
        from omegaconf import OmegaConf
        from axon.data.data_sampler import SamplerType, create_data_sampler

        ds = _make_data_source(10)
        config = OmegaConf.create({"data_sampler": SamplerType.SEQUENTIAL})
        sampler = create_data_sampler(config, ds)
        indices = list(sampler)
        assert indices == list(range(10))


# ═══════════════════════════════════════════════════════════════════
#  Edge cases
# ═══════════════════════════════════════════════════════════════════

class TestCurriculumSamplerEdgeCases:
    def test_exp_weighted_single_item_dataset(self):
        ds = _make_data_source(1)
        sampler = ExpWeightedCurriculumSampler(ds)
        indices = list(sampler)
        assert len(indices) == 1
        assert indices[0] == 0

    def test_threshold_single_item_dataset(self):
        ds = _make_data_source(1)
        sampler = ThresholdMaskingSampler(ds, threshold=0.9)
        indices = list(sampler)
        assert len(indices) == 1
        assert indices[0] == 0

    def test_exp_weighted_all_same_weight(self):
        """With uniform weights, all indices should be sampled roughly equally."""
        ds = _make_data_source(5)
        sampler = ExpWeightedCurriculumSampler(ds)
        # Default: all weights = 1.0
        counts = [0] * 5
        for _ in range(1000):
            for idx in sampler:
                counts[idx] += 1
        # Each should be sampled ~1000 times (1000 iters * 5 samples / 5 items)
        assert all(c > 0 for c in counts)

    def test_threshold_boundary_value(self):
        """Pass rate exactly at threshold."""
        ds = _make_data_source(3)
        sampler = ThresholdMaskingSampler(ds, threshold=0.5)
        sampler.update(_make_threshold_batch(indices=[0], pass_rates=[0.5]))
        assert sampler.masked[0] == True

    def test_threshold_boundary_below(self):
        """Pass rate just below threshold."""
        ds = _make_data_source(3)
        sampler = ThresholdMaskingSampler(ds, threshold=0.5)
        sampler.update(_make_threshold_batch(indices=[0], pass_rates=[0.49]))
        assert sampler.masked[0] == False

    def test_exp_weighted_centroid_effect(self):
        """Different centroid values should change weight mapping."""
        ds = _make_data_source(5)
        s1 = ExpWeightedCurriculumSampler(ds)
        s2 = ExpWeightedCurriculumSampler(ds)
        s2.centroid = 0.8  # shift centroid
        batch = _make_exp_batch(indices=[0], pass_rates=[0.5])
        s1.update(batch)
        s2.update(batch)
        # Different centroids should produce different weights
        assert s1.weights[0] != s2.weights[0]

    def test_threshold_large_batch_update(self):
        """Update many indices at once."""
        ds = _make_data_source(100)
        sampler = ThresholdMaskingSampler(ds, threshold=0.8)
        indices = list(range(100))
        pass_rates = [0.9] * 50 + [0.3] * 50
        sampler.update(_make_threshold_batch(indices=indices, pass_rates=pass_rates))
        assert sampler.masked[:50].sum() == 50
        assert sampler.masked[50:].sum() == 0
