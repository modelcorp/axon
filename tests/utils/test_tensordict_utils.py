import pytest
import torch
from tensordict import TensorDict

from axon.utils.tensordict_utils import (
    get,
    get_keys,
    pad_to_divisor,
    pop,
    pop_keys,
    union_tensor_dict,
    unpad,
)


def _make_td(**kwargs):
    batch_size = None
    for v in kwargs.values():
        if isinstance(v, torch.Tensor):
            batch_size = v.shape[0]
            break
    if batch_size is None:
        batch_size = []
    else:
        batch_size = [batch_size]
    return TensorDict(kwargs, batch_size=batch_size)


# ===================================================================
#  pad_to_divisor
# ===================================================================


class TestPadToDivisor:
    def test_needs_padding(self):
        td = _make_td(x=torch.tensor([[1, 2], [3, 4], [5, 6]]))
        padded, pad_size = pad_to_divisor(td, size_divisor=4)
        assert pad_size == 1
        assert len(padded) == 4
        # First 3 should be original data
        assert torch.equal(padded["x"][:3], td["x"])
        # Padded element should be a copy from the beginning
        assert torch.equal(padded["x"][3], td["x"][0])

    def test_needs_more_padding(self):
        td = _make_td(x=torch.tensor([[1], [2]]))
        padded, pad_size = pad_to_divisor(td, size_divisor=5)
        assert pad_size == 3
        assert len(padded) == 5
        assert torch.equal(padded["x"][:2], td["x"])

    def test_divisor_larger_than_batch(self):
        """When divisor > batch size, pads up to the divisor."""
        td = _make_td(x=torch.tensor([[1]]))
        padded, pad_size = pad_to_divisor(td, size_divisor=7)
        assert len(padded) == 7
        assert pad_size == 6

    def test_roundtrip_preserves_values_exactly(self):
        """Pad then unpad should yield identical tensor values."""
        original = torch.randn(5, 3)
        td = _make_td(x=original)
        padded, pad_size = pad_to_divisor(td, size_divisor=4)
        result = unpad(padded, pad_size)
        assert torch.equal(result["x"], original)

    def test_padding_larger_than_data(self):
        td = _make_td(x=torch.tensor([[10]]))
        padded, pad_size = pad_to_divisor(td, size_divisor=4)
        assert pad_size == 3
        assert len(padded) == 4
        # All padded elements should be copies of the single element
        for i in range(4):
            assert padded["x"][i].item() == 10


# ===================================================================
#  unpad
# ===================================================================


class TestUnpad:
    def test_unpad_removes_correct_elements(self):
        td = _make_td(x=torch.tensor([[1], [2], [3], [4]]))
        result = unpad(td, pad_size=2)
        assert len(result) == 2
        assert result["x"][0].item() == 1
        assert result["x"][1].item() == 2

    def test_roundtrip_pad_unpad(self):
        td = _make_td(x=torch.arange(7).unsqueeze(1))
        padded, pad_size = pad_to_divisor(td, size_divisor=4)
        assert len(padded) == 8
        result = unpad(padded, pad_size)
        assert len(result) == 7
        assert torch.equal(result["x"], td["x"])


# ===================================================================
#  union_tensor_dict
# ===================================================================


class TestUnionTensorDict:
    def test_disjoint_keys_merge(self):
        td1 = _make_td(a=torch.tensor([1.0, 2.0]))
        td2 = _make_td(b=torch.tensor([3.0, 4.0]))

        result = union_tensor_dict(td1, td2)

        assert torch.equal(result["a"], torch.tensor([1.0, 2.0]))
        assert torch.equal(result["b"], torch.tensor([3.0, 4.0]))

    def test_overlapping_same_value_keys(self):
        shared = torch.tensor([1.0, 2.0])
        td1 = _make_td(a=shared.clone())
        td2 = _make_td(a=shared.clone())

        result = union_tensor_dict(td1, td2)
        assert torch.equal(result["a"], shared)

    def test_overlapping_different_value_raises(self):
        td1 = _make_td(a=torch.tensor([1.0, 2.0]))
        td2 = _make_td(a=torch.tensor([3.0, 4.0]))

        with pytest.raises(AssertionError, match="not the same object"):
            union_tensor_dict(td1, td2)

    def test_different_batch_size_raises(self):
        td1 = _make_td(a=torch.tensor([1.0, 2.0]))
        td2 = _make_td(b=torch.tensor([1.0, 2.0, 3.0]))

        with pytest.raises(AssertionError, match="identical batch size"):
            union_tensor_dict(td1, td2)

    def test_union_with_many_keys(self):
        td1 = _make_td(**{f"k{i}": torch.tensor([float(i)]) for i in range(10)})
        td2 = _make_td(**{f"j{i}": torch.tensor([float(i)]) for i in range(10)})
        result = union_tensor_dict(td1, td2)
        assert len(list(result.keys())) == 20


# ===================================================================
#  get
# ===================================================================


class TestGet:
    def test_existing_key_returns_value(self):
        td = _make_td(x=torch.tensor([1.0, 2.0, 3.0]))
        result = get(td, "x")
        assert isinstance(result, torch.Tensor)
        assert torch.equal(result, torch.tensor([1.0, 2.0, 3.0]))

    def test_missing_key_returns_default(self):
        td = _make_td(x=torch.tensor([1.0, 2.0]))
        result = get(td, "y", default=42)
        assert result == 42


# ===================================================================
#  pop
# ===================================================================


class TestPop:
    def test_existing_key_returns_and_removes(self):
        td = _make_td(x=torch.tensor([1.0, 2.0]), y=torch.tensor([3.0, 4.0]))
        result = pop(td, "x")
        assert isinstance(result, torch.Tensor)
        assert torch.equal(result, torch.tensor([1.0, 2.0]))
        assert "x" not in td.keys()
        assert "y" in td.keys()

    def test_missing_key_returns_default(self):
        td = _make_td(x=torch.tensor([1.0, 2.0]))
        result = pop(td, "z", default=-1)
        assert result == -1


# ===================================================================
#  get_keys
# ===================================================================


class TestGetKeys:
    def test_returns_subset_tensordict(self):
        td = _make_td(
            a=torch.tensor([1.0, 2.0]),
            b=torch.tensor([3.0, 4.0]),
            c=torch.tensor([5.0, 6.0]),
        )

        result = get_keys(td, ["a", "c"])

        assert "a" in result.keys()
        assert "c" in result.keys()
        assert "b" not in result.keys()
        assert torch.equal(result["a"], torch.tensor([1.0, 2.0]))
        assert torch.equal(result["c"], torch.tensor([5.0, 6.0]))

    def test_missing_key_raises(self):
        td = _make_td(a=torch.tensor([1.0, 2.0]))

        with pytest.raises(KeyError, match="not in tensordict"):
            get_keys(td, ["a", "nonexistent"])

    def test_original_unchanged(self):
        td = _make_td(
            a=torch.tensor([1.0, 2.0]),
            b=torch.tensor([3.0, 4.0]),
        )

        get_keys(td, ["a"])

        assert "a" in td.keys()
        assert "b" in td.keys()


# ===================================================================
#  pop_keys
# ===================================================================


class TestPopKeys:
    def test_returns_and_removes_subset(self):
        td = _make_td(
            a=torch.tensor([1.0, 2.0]),
            b=torch.tensor([3.0, 4.0]),
            c=torch.tensor([5.0, 6.0]),
        )

        result = pop_keys(td, ["a", "c"])

        # Result should contain the popped keys
        assert "a" in result.keys()
        assert "c" in result.keys()
        assert torch.equal(result["a"], torch.tensor([1.0, 2.0]))
        assert torch.equal(result["c"], torch.tensor([5.0, 6.0]))

        # Original should no longer have the popped keys
        assert "a" not in td.keys()
        assert "c" not in td.keys()
        assert "b" in td.keys()

    def test_missing_key_raises(self):
        td = _make_td(a=torch.tensor([1.0, 2.0]))

        with pytest.raises(KeyError, match="not in tensordict"):
            pop_keys(td, ["a", "nonexistent"])

    def test_pop_all_keys(self):
        td = _make_td(
            a=torch.tensor([1.0, 2.0]),
            b=torch.tensor([3.0, 4.0]),
        )

        result = pop_keys(td, ["a", "b"])

        assert "a" in result.keys()
        assert "b" in result.keys()
        assert len(list(td.keys())) == 0

    def test_pop_all_leaves_empty(self):
        td = _make_td(a=torch.tensor([1.0]), b=torch.tensor([2.0]))
        popped = pop_keys(td, ["a", "b"])
        assert len(list(td.keys())) == 0
        assert len(list(popped.keys())) == 2


# ===================================================================
#  get_keys edge cases
# ===================================================================


class TestGetKeysEdgeCases:
    def test_single_key(self):
        td = _make_td(a=torch.tensor([1.0, 2.0]), b=torch.tensor([3.0, 4.0]))
        result = get_keys(td, ["a"])
        assert "a" in result.keys()
        assert "b" not in result.keys()

    def test_all_keys(self):
        td = _make_td(a=torch.tensor([1.0]), b=torch.tensor([2.0]))
        result = get_keys(td, ["a", "b"])
        assert len(list(result.keys())) == 2
