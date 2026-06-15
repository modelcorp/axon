"""
Tests for axon.data.dataloader — DynamicDataLoader.
"""

import pytest
import torch
from torch.utils.data import Dataset

from axon.data.dataloader import DynamicDataLoader


# ───────────────── fixtures ─────────────────

class SimpleDataset(Dataset):
    """Minimal dataset for testing."""
    def __init__(self, size=20):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return {"value": idx, "squared": idx ** 2}


class TensorDataset(Dataset):
    """Dataset returning tensors."""
    def __init__(self, size=12):
        self.data = torch.arange(size).float()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


@pytest.fixture
def simple_ds():
    return SimpleDataset(20)


@pytest.fixture
def tensor_ds():
    return TensorDataset(12)


def _sum_collate(batch):
    """Collate that returns the list as-is for easy inspection."""
    return batch


# ═══════════════════════════════════════════════════════════════════
#  Construction
# ═══════════════════════════════════════════════════════════════════

class TestDynamicDataLoaderConstruction:
    def test_basic_construction(self, simple_ds):
        dl = DynamicDataLoader(simple_ds, batch_size=4)
        assert dl.batch_size == 4
        assert dl.infinite is True
        assert dl.dataset is simple_ds

    def test_finite_mode(self, simple_ds):
        dl = DynamicDataLoader(simple_ds, batch_size=4, infinite=False)
        assert dl.infinite is False

    def test_len(self, simple_ds):
        dl = DynamicDataLoader(simple_ds, batch_size=4)
        assert len(dl) == 20

    def test_sampler_accessible(self, simple_ds):
        dl = DynamicDataLoader(simple_ds, batch_size=4)
        assert dl.sampler is not None


# ═══════════════════════════════════════════════════════════════════
#  Iteration — __next__ (default batch_size)
# ═══════════════════════════════════════════════════════════════════

class TestDynamicDataLoaderIteration:
    def test_basic_iteration(self, simple_ds):
        dl = DynamicDataLoader(simple_ds, batch_size=5, collate_fn=_sum_collate)
        it = iter(dl)
        batch = next(it)
        assert len(batch) == 5

    def test_finite_exhaustion(self):
        ds = SimpleDataset(10)
        dl = DynamicDataLoader(ds, batch_size=4, infinite=False, collate_fn=_sum_collate)
        it = iter(dl)
        batches = []
        try:
            while True:
                batches.append(next(it))
        except StopIteration:
            pass
        # 10 items / 4 per batch -> should get at least 2 full batches
        total_items = sum(len(b) for b in batches)
        assert total_items == 10

    def test_infinite_wraps_around(self):
        ds = SimpleDataset(4)
        dl = DynamicDataLoader(ds, batch_size=2, infinite=True, collate_fn=_sum_collate)
        it = iter(dl)
        # Fetch more than the dataset size to force wrap-around
        batches = [next(it) for _ in range(5)]
        total_items = sum(len(b) for b in batches)
        assert total_items == 10  # 5 batches * 2 items each


# ═══════════════════════════════════════════════════════════════════
#  Dynamic batch size via .next()
# ═══════════════════════════════════════════════════════════════════

class TestDynamicDataLoaderNext:
    def test_varying_batch_sizes(self, simple_ds):
        dl = DynamicDataLoader(simple_ds, batch_size=4, collate_fn=_sum_collate)
        it = iter(dl)
        batch_3 = it.next(3)
        assert len(batch_3) == 3
        batch_7 = it.next(7)
        assert len(batch_7) == 7

    def test_next_zero_raises(self, simple_ds):
        dl = DynamicDataLoader(simple_ds, batch_size=4, collate_fn=_sum_collate)
        it = iter(dl)
        with pytest.raises(AssertionError, match="positive"):
            it.next(0)

    def test_next_larger_than_internal_batch(self):
        ds = SimpleDataset(100)
        dl = DynamicDataLoader(ds, batch_size=4, collate_fn=_sum_collate)
        it = iter(dl)
        # Request more than the internal batch_size=4
        batch = it.next(10)
        assert len(batch) == 10

    def test_next_finite_stop_iteration(self):
        ds = SimpleDataset(5)
        dl = DynamicDataLoader(ds, batch_size=5, infinite=False, collate_fn=_sum_collate)
        it = iter(dl)
        batch1 = it.next(5)
        assert len(batch1) == 5
        with pytest.raises(StopIteration):
            it.next(5)


# ═══════════════════════════════════════════════════════════════════
#  Custom collate_fn
# ═══════════════════════════════════════════════════════════════════

class TestDynamicDataLoaderCollate:
    def test_default_collate(self):
        ds = TensorDataset(8)
        dl = DynamicDataLoader(ds, batch_size=4)
        it = iter(dl)
        batch = next(it)
        assert isinstance(batch, torch.Tensor)
        assert batch.shape == (4,)

    def test_custom_collate(self, simple_ds):
        def my_collate(items):
            return {"count": len(items), "items": items}

        dl = DynamicDataLoader(simple_ds, batch_size=3, collate_fn=my_collate)
        it = iter(dl)
        batch = next(it)
        assert batch["count"] == 3


# ═══════════════════════════════════════════════════════════════════
#  State dict (checkpoint/resume)
# ═══════════════════════════════════════════════════════════════════

class TestDynamicDataLoaderState:
    def test_state_dict_structure(self, simple_ds):
        dl = DynamicDataLoader(simple_ds, batch_size=4, collate_fn=_sum_collate)
        it = iter(dl)
        next(it)  # consume one batch
        state = dl.state_dict()
        assert "loader_state" in state
        assert "buffer" in state
        assert "exhausted" in state

    def test_save_load_roundtrip(self, simple_ds):
        dl = DynamicDataLoader(simple_ds, batch_size=4, collate_fn=_sum_collate)
        it = iter(dl)
        next(it)
        state = dl.state_dict()

        # Create a new loader and restore state
        dl2 = DynamicDataLoader(simple_ds, batch_size=4, collate_fn=_sum_collate)
        dl2.load_state_dict(state)
        assert dl2._buffer == state["buffer"]
        assert dl2._exhausted == state["exhausted"]



# ═══════════════════════════════════════════════════════════════════
#  Edge cases
# ═══════════════════════════════════════════════════════════════════

class TestDynamicDataLoaderEdgeCases:
    def test_single_item_dataset(self):
        ds = SimpleDataset(1)
        dl = DynamicDataLoader(ds, batch_size=1, collate_fn=_sum_collate)
        it = iter(dl)
        batch = next(it)
        assert len(batch) == 1

    def test_reinitialize_iter(self, simple_ds):
        dl = DynamicDataLoader(simple_ds, batch_size=4, collate_fn=_sum_collate)
        it1 = iter(dl)
        next(it1)
        # Reinitialize
        it2 = iter(dl)
        batch = next(it2)
        assert len(batch) == 4


# ═══════════════════════════════════════════════════════════════════
#  Additional edge cases
# ═══════════════════════════════════════════════════════════════════

class TestDynamicDataLoaderMoreEdgeCases:
    def test_batch_size_one(self):
        ds = SimpleDataset(5)
        dl = DynamicDataLoader(ds, batch_size=1, collate_fn=_sum_collate)
        it = iter(dl)
        batch = next(it)
        assert len(batch) == 1

    def test_next_returns_all_data_finite(self):
        """In finite mode, dynamic .next() should also exhaust data."""
        ds = SimpleDataset(7)
        dl = DynamicDataLoader(ds, batch_size=3, infinite=False, collate_fn=_sum_collate)
        it = iter(dl)
        all_items = []
        try:
            while True:
                all_items.extend(it.next(2))
        except StopIteration:
            pass
        values = {item["value"] for item in all_items}
        assert values == set(range(7))

    def test_large_batch_size_with_small_dataset(self):
        """Request larger batch than dataset in infinite mode."""
        ds = SimpleDataset(3)
        dl = DynamicDataLoader(ds, batch_size=10, infinite=True, collate_fn=_sum_collate)
        it = iter(dl)
        batch = next(it)
        assert len(batch) == 10
