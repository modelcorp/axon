"""
Thorough tests for axon.protocol — DataProto, DataProtoItem, DataProtoConfig,
and all helper / free-standing functions.
"""

import copy
import os
import pickle
import tempfile

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from axon.protocol import (
    DataProto,
    DataProtoConfig,
    DataProtoItem,
    _deep_equal,
    collate_fn,
    deserialize_single_tensor,
    deserialize_tensordict,
    fold_batch_dim,
    list_of_dict_to_dict_of_list,
    pad_dataproto_to_divisor,
    serialize_single_tensor,
    serialize_tensordict,
    unfold_batch_dim,
    union_numpy_dict,
    union_tensor_dict,
    unpad_dataproto,
)

# ───────────────────────────── helpers ─────────────────────────────


def _make_proto(batch_size=4, seq_len=8, with_non_tensor=True, meta=None):
    """Create a simple DataProto for testing."""
    tensors = {
        "input_ids": torch.randint(0, 1000, (batch_size, seq_len)),
        "scores": torch.randn(batch_size, seq_len),
    }
    non_tensors = {}
    if with_non_tensor:
        non_tensors = {
            "text": np.array([f"sample_{i}" for i in range(batch_size)], dtype=object),
        }
    return DataProto.from_dict(tensors=tensors, non_tensors=non_tensors, meta_info=meta or {})


# ═══════════════════════════════════════════════════════════════════
#  union_tensor_dict
# ═══════════════════════════════════════════════════════════════════


class TestUnionTensorDict:
    def test_merge_disjoint_keys(self):
        td1 = TensorDict({"a": torch.ones(3)}, batch_size=(3,))
        td2 = TensorDict({"b": torch.zeros(3)}, batch_size=(3,))
        result = union_tensor_dict(td1, td2)
        assert "a" in result.keys() and "b" in result.keys()

    def test_merge_overlapping_equal_keys(self):
        t = torch.tensor([1.0, 2.0])
        td1 = TensorDict({"x": t.clone()}, batch_size=(2,))
        td2 = TensorDict({"x": t.clone()}, batch_size=(2,))
        result = union_tensor_dict(td1, td2)
        assert torch.equal(result["x"], t)

    def test_merge_overlapping_conflicting_keys_raises(self):
        td1 = TensorDict({"x": torch.ones(2)}, batch_size=(2,))
        td2 = TensorDict({"x": torch.zeros(2)}, batch_size=(2,))
        with pytest.raises(AssertionError, match="Conflicting"):
            union_tensor_dict(td1, td2)

    def test_batch_size_mismatch_raises(self):
        td1 = TensorDict({"a": torch.ones(2)}, batch_size=(2,))
        td2 = TensorDict({"b": torch.ones(3)}, batch_size=(3,))
        with pytest.raises(AssertionError, match="Batch sizes"):
            union_tensor_dict(td1, td2)


# ═══════════════════════════════════════════════════════════════════
#  union_numpy_dict
# ═══════════════════════════════════════════════════════════════════


class TestUnionNumpyDict:
    def test_merge_disjoint(self):
        d1 = {"a": np.array([1, 2])}
        d2 = {"b": np.array([3, 4])}
        result = union_numpy_dict(d1, d2)
        assert "a" in result and "b" in result

    def test_merge_overlapping_equal(self):
        arr = np.array([1, 2, 3])
        d1 = {"x": arr.copy()}
        d2 = {"x": arr.copy()}
        result = union_numpy_dict(d1, d2)
        np.testing.assert_array_equal(result["x"], arr)

    def test_merge_overlapping_conflicting_raises(self):
        d1 = {"x": np.array([1, 2])}
        d2 = {"x": np.array([3, 4])}
        with pytest.raises(AssertionError, match="Conflicting"):
            union_numpy_dict(d1, d2)


# ═══════════════════════════════════════════════════════════════════
#  _deep_equal
# ═══════════════════════════════════════════════════════════════════


class TestDeepEqual:
    def test_equal_primitives(self):
        assert _deep_equal(1, 1, set())
        assert _deep_equal("abc", "abc", set())
        assert not _deep_equal(1, 2, set())

    def test_nan_equality(self):
        assert _deep_equal(float("nan"), float("nan"), set())

    def test_type_mismatch(self):
        assert not _deep_equal(1, 1.0, set())

    def test_ndarray_equality(self):
        a = np.array([1.0, 2.0, np.nan])
        b = np.array([1.0, 2.0, np.nan])
        assert _deep_equal(a, b, set())

    def test_ndarray_object_dtype(self):
        a = np.array(["hello", "world"], dtype=object)
        b = np.array(["hello", "world"], dtype=object)
        assert _deep_equal(a, b, set())

    def test_ndarray_shape_mismatch(self):
        assert not _deep_equal(np.array([1, 2]), np.array([1, 2, 3]), set())

    def test_dict_containers(self):
        assert _deep_equal({"a": 1}, {"a": 1}, set())
        assert not _deep_equal({"a": 1}, {"a": 2}, set())

    def test_list_containers(self):
        assert _deep_equal([1, 2, 3], [1, 2, 3], set())
        assert not _deep_equal([1, 2], [1, 2, 3], set())


# ═══════════════════════════════════════════════════════════════════
#  list_of_dict_to_dict_of_list
# ═══════════════════════════════════════════════════════════════════


class TestListOfDictToDictOfList:
    def test_basic(self):
        inp = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
        result = list_of_dict_to_dict_of_list(inp)
        assert result == {"a": [1, 3], "b": [2, 4]}

    def test_empty_list(self):
        assert list_of_dict_to_dict_of_list([]) == {}

    def test_single_item(self):
        result = list_of_dict_to_dict_of_list([{"x": 10}])
        assert result == {"x": [10]}


# ═══════════════════════════════════════════════════════════════════
#  Serialization round-trips
# ═══════════════════════════════════════════════════════════════════


class TestSerialization:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.int64, torch.bfloat16])
    def test_single_tensor_roundtrip(self, dtype):
        t = torch.randn(3, 4).to(dtype) if dtype.is_floating_point else torch.randint(0, 100, (3, 4)).to(dtype)
        serialized = serialize_single_tensor(t)
        restored = deserialize_single_tensor(serialized)
        assert restored.shape == t.shape
        assert restored.dtype == t.dtype
        assert torch.equal(restored, t)

    def test_tensordict_roundtrip(self):
        td = TensorDict(
            {"a": torch.randn(2, 3), "b": torch.randint(0, 10, (2,))},
            batch_size=(2,),
        )
        serialized = serialize_tensordict(td)
        restored = deserialize_tensordict(serialized)
        assert torch.equal(restored["a"], td["a"])
        assert torch.equal(restored["b"], td["b"])

    def test_empty_shape_tensor(self):
        t = torch.tensor(42.0)
        serialized = serialize_single_tensor(t)
        restored = deserialize_single_tensor(serialized)
        assert torch.equal(restored, t)


# ═══════════════════════════════════════════════════════════════════
#  DataProtoConfig
# ═══════════════════════════════════════════════════════════════════


class TestDataProtoConfig:
    def setup_method(self):
        # Reset config between tests
        DataProtoConfig._config.clear()
        os.environ.pop("VERL_AUTO_PADDING", None)

    def teardown_method(self):
        DataProtoConfig._config.clear()
        os.environ.pop("AXON_AUTO_PADDING", None)

    def test_auto_padding_default_false(self):
        assert DataProtoConfig.auto_padding is False

    def test_auto_padding_setter(self):
        DataProtoConfig.auto_padding = True
        assert DataProtoConfig.auto_padding is True

    def test_auto_padding_env_var(self):
        os.environ["AXON_AUTO_PADDING"] = "TRUE"
        assert DataProtoConfig.auto_padding is True

    def test_auto_padding_env_var_1(self):
        os.environ["AXON_AUTO_PADDING"] = "1"
        assert DataProtoConfig.auto_padding is True

    def test_auto_padding_setter_must_be_bool(self):
        with pytest.raises(AssertionError):
            DataProtoConfig.auto_padding = 1


# ═══════════════════════════════════════════════════════════════════
#  DataProto — construction
# ═══════════════════════════════════════════════════════════════════


class TestDataProtoConstruction:
    def test_from_dict_tensors_only(self):
        dp = DataProto.from_dict(tensors={"x": torch.ones(3, 2)})
        assert len(dp) == 3
        assert dp.non_tensor_batch == {}

    def test_from_dict_non_tensors_only(self):
        dp = DataProto.from_dict(non_tensors={"labels": ["a", "b", "c"]})
        assert len(dp) == 3
        assert dp.batch is None

    def test_from_dict_mixed(self):
        dp = DataProto.from_dict(
            tensors={"t": torch.zeros(2, 4)},
            non_tensors={"s": ["hello", "world"]},
        )
        assert len(dp) == 2
        assert "t" in dp.batch.keys()
        assert "s" in dp.non_tensor_batch

    def test_from_dict_batch_size_mismatch_raises(self):
        with pytest.raises(AssertionError, match="Batch size mismatch"):
            DataProto.from_dict(tensors={"a": torch.ones(3), "b": torch.ones(4)})

    def test_from_single_dict(self):
        dp = DataProto.from_single_dict(
            {
                "tensor_field": torch.randn(5, 3),
                "array_field": np.array([1, 2, 3, 4, 5]),
            }
        )
        assert len(dp) == 5
        assert "tensor_field" in dp.batch.keys()
        assert "array_field" in dp.non_tensor_batch

    def test_from_single_dict_unsupported_type_raises(self):
        with pytest.raises(ValueError, match="Unsupported type"):
            DataProto.from_single_dict({"bad": [1, 2, 3]})

    def test_from_dict_with_meta_info(self):
        dp = DataProto.from_dict(
            tensors={"x": torch.ones(2)},
            meta_info={"lr": 0.01},
        )
        assert dp.meta_info["lr"] == 0.01

    def test_from_dict_tensordict_input(self):
        td = TensorDict({"a": torch.randn(4, 3)}, batch_size=(4,))
        dp = DataProto.from_dict(tensors=td)
        assert len(dp) == 4

    def test_empty_dataproto(self):
        dp = DataProto()
        assert len(dp) == 0


# ═══════════════════════════════════════════════════════════════════
#  DataProto — consistency checking
# ═══════════════════════════════════════════════════════════════════


class TestDataProtoConsistency:
    def test_batch_non_tensor_size_mismatch_raises(self):
        with pytest.raises(AssertionError, match="has length"):
            DataProto(
                batch=TensorDict({"x": torch.ones(3)}, batch_size=(3,)),
                non_tensor_batch={"y": np.array([1, 2])},
            )

    def test_non_tensor_not_ndarray_raises(self):
        with pytest.raises(AssertionError, match="must be ndarray"):
            DataProto(
                batch=TensorDict({"x": torch.ones(3)}, batch_size=(3,)),
                non_tensor_batch={"y": [1, 2, 3]},
            )

    def test_multi_batch_dim_raises(self):
        with pytest.raises(AssertionError, match="num_batch_dims=1"):
            DataProto(batch=TensorDict({"x": torch.ones(2, 3)}, batch_size=(2, 3)))


# ═══════════════════════════════════════════════════════════════════
#  DataProto — indexing (__getitem__)
# ═══════════════════════════════════════════════════════════════════


class TestDataProtoIndexing:
    def test_int_index_returns_item(self):
        dp = _make_proto(batch_size=4)
        item = dp[0]
        assert isinstance(item, DataProtoItem)
        assert item.batch is not None
        assert "text" in item.non_tensor_batch

    def test_negative_int_index(self):
        dp = _make_proto(batch_size=4)
        item = dp[-1]
        assert isinstance(item, DataProtoItem)
        assert torch.equal(item.batch["input_ids"], dp.batch["input_ids"][-1])

    def test_slice_returns_dataproto(self):
        dp = _make_proto(batch_size=8)
        sliced = dp[2:5]
        assert isinstance(sliced, DataProto)
        assert len(sliced) == 3

    def test_slice_with_step(self):
        dp = _make_proto(batch_size=8)
        sliced = dp[::2]
        assert isinstance(sliced, DataProto)
        assert len(sliced) == 4

    def test_list_index(self):
        dp = _make_proto(batch_size=6)
        selected = dp[[0, 2, 4]]
        assert isinstance(selected, DataProto)
        assert len(selected) == 3

    def test_tensor_index(self):
        dp = _make_proto(batch_size=6)
        selected = dp[torch.tensor([1, 3, 5])]
        assert len(selected) == 3

    def test_numpy_index(self):
        dp = _make_proto(batch_size=6)
        selected = dp[np.array([0, 1])]
        assert len(selected) == 2

    def test_boolean_mask(self):
        dp = _make_proto(batch_size=4)
        mask = torch.tensor([True, False, True, False])
        selected = dp[mask]
        assert len(selected) == 2

    def test_unsupported_index_type_raises(self):
        dp = _make_proto(batch_size=4)
        with pytest.raises(TypeError, match="not supported"):
            dp["bad"]


# ═══════════════════════════════════════════════════════════════════
#  DataProto — select
# ═══════════════════════════════════════════════════════════════════


class TestDataProtoSelect:
    def test_select_batch_keys(self):
        dp = _make_proto(batch_size=4)
        selected = dp.select(batch_keys=["input_ids"])
        assert "input_ids" in selected.batch.keys()
        assert "scores" not in selected.batch.keys()

    def test_select_non_tensor_keys(self):
        dp = _make_proto(batch_size=4)
        selected = dp.select(non_tensor_batch_keys=["text"])
        assert "text" in selected.non_tensor_batch

    def test_select_meta_keys(self):
        dp = _make_proto(batch_size=4, meta={"a": 1, "b": 2})
        selected = dp.select(meta_info_keys=["a"])
        assert "a" in selected.meta_info
        assert "b" not in selected.meta_info

    def test_select_deepcopy(self):
        dp = _make_proto(batch_size=4, meta={"key": [1, 2]})
        selected = dp.select(deepcopy=True)
        selected.meta_info["key"].append(3)
        assert len(dp.meta_info["key"]) == 2  # original unchanged

    def test_select_none_keeps_all(self):
        dp = _make_proto(batch_size=4)
        selected = dp.select()
        assert set(selected.batch.keys()) == set(dp.batch.keys())


# ═══════════════════════════════════════════════════════════════════
#  DataProto — slice
# ═══════════════════════════════════════════════════════════════════


class TestDataProtoSlice:
    def test_slice_basic(self):
        dp = _make_proto(batch_size=10)
        sliced = dp.slice(2, 7)
        assert len(sliced) == 5

    def test_slice_from_start(self):
        dp = _make_proto(batch_size=10)
        sliced = dp.slice(end=3)
        assert len(sliced) == 3

    def test_slice_to_end(self):
        dp = _make_proto(batch_size=10)
        sliced = dp.slice(start=8)
        assert len(sliced) == 2

    def test_slice_preserves_meta(self):
        dp = _make_proto(batch_size=4, meta={"epoch": 1})
        sliced = dp.slice(0, 2)
        assert sliced.meta_info["epoch"] == 1

    def test_slice_non_tensor(self):
        dp = _make_proto(batch_size=4)
        sliced = dp.slice(1, 3)
        assert len(sliced.non_tensor_batch["text"]) == 2


# ═══════════════════════════════════════════════════════════════════
#  DataProto — pop
# ═══════════════════════════════════════════════════════════════════


class TestDataProtoPop:
    def test_pop_batch_key(self):
        dp = _make_proto(batch_size=4)
        popped = dp.pop(batch_keys=["scores"])
        assert "scores" not in dp.batch.keys()
        assert "scores" in popped.batch.keys()

    def test_pop_non_tensor_key(self):
        dp = _make_proto(batch_size=4)
        popped = dp.pop(non_tensor_batch_keys=["text"])
        assert "text" not in dp.non_tensor_batch
        assert "text" in popped.non_tensor_batch

    def test_pop_meta_key(self):
        dp = _make_proto(batch_size=4, meta={"lr": 0.01})
        popped = dp.pop(meta_info_keys=["lr"])
        assert "lr" not in dp.meta_info
        assert "lr" in popped.meta_info

    def test_pop_missing_key_raises(self):
        dp = _make_proto(batch_size=4)
        with pytest.raises(AssertionError, match="not found"):
            dp.pop(batch_keys=["nonexistent"])


# ═══════════════════════════════════════════════════════════════════
#  DataProto — rename
# ═══════════════════════════════════════════════════════════════════


class TestDataProtoRename:
    def test_rename_single_key(self):
        dp = _make_proto(batch_size=4)
        dp.rename("scores", "rewards")
        assert "rewards" in dp.batch.keys()
        assert "scores" not in dp.batch.keys()

    def test_rename_multiple_keys_sequentially(self):
        dp = _make_proto(batch_size=4)
        dp.rename("input_ids", "tokens")
        dp.rename("scores", "rewards")
        assert "tokens" in dp.batch.keys()
        assert "rewards" in dp.batch.keys()

    def test_rename_length_mismatch_raises(self):
        dp = _make_proto(batch_size=4)
        with pytest.raises(ValueError, match="Length mismatch"):
            dp.rename(["a"], ["b", "c"])

    def test_rename_none_is_noop(self):
        dp = _make_proto(batch_size=4)
        keys_before = set(dp.batch.keys())
        dp.rename(None, None)
        assert set(dp.batch.keys()) == keys_before


# ═══════════════════════════════════════════════════════════════════
#  DataProto — union
# ═══════════════════════════════════════════════════════════════════


class TestDataProtoUnion:
    def test_union_disjoint(self):
        dp1 = DataProto.from_dict(tensors={"a": torch.ones(3, 2)})
        dp2 = DataProto.from_dict(tensors={"b": torch.zeros(3, 2)})
        dp1.union(dp2)
        assert "a" in dp1.batch.keys() and "b" in dp1.batch.keys()

    def test_union_non_tensor(self):
        dp1 = DataProto.from_dict(
            tensors={"a": torch.ones(3)},
            non_tensors={"x": ["a", "b", "c"]},
        )
        dp2 = DataProto.from_dict(
            tensors={"b": torch.zeros(3)},
            non_tensors={"y": ["d", "e", "f"]},
        )
        dp1.union(dp2)
        assert "x" in dp1.non_tensor_batch and "y" in dp1.non_tensor_batch


# ═══════════════════════════════════════════════════════════════════
#  DataProto — concat
# ═══════════════════════════════════════════════════════════════════


class TestDataProtoConcat:
    def test_concat_basic(self):
        dp1 = _make_proto(batch_size=3)
        dp2 = _make_proto(batch_size=5)
        result = DataProto.concat([dp1, dp2])
        assert len(result) == 8

    def test_concat_preserves_tensor_data(self):
        dp1 = DataProto.from_dict(tensors={"x": torch.tensor([1.0, 2.0])})
        dp2 = DataProto.from_dict(tensors={"x": torch.tensor([3.0, 4.0])})
        result = DataProto.concat([dp1, dp2])
        assert torch.equal(result.batch["x"], torch.tensor([1.0, 2.0, 3.0, 4.0]))

    def test_concat_non_tensor(self):
        dp1 = DataProto.from_dict(non_tensors={"t": ["a", "b"]})
        dp2 = DataProto.from_dict(non_tensors={"t": ["c"]})
        result = DataProto.concat([dp1, dp2])
        assert len(result.non_tensor_batch["t"]) == 3

    def test_concat_empty_list(self):
        result = DataProto.concat([])
        assert len(result) == 0

    def test_concat_single_item(self):
        dp = _make_proto(batch_size=4)
        result = DataProto.concat([dp])
        assert len(result) == 4

    def test_concat_merges_meta_info(self):
        dp1 = _make_proto(batch_size=2, meta={"key": "val"})
        dp2 = _make_proto(batch_size=2, meta={"key": "val"})
        result = DataProto.concat([dp1, dp2])
        assert result.meta_info["key"] == "val"

    def test_concat_conflicting_meta_raises(self):
        dp1 = _make_proto(batch_size=2, meta={"key": "val1"})
        dp2 = _make_proto(batch_size=2, meta={"key": "val2"})
        with pytest.raises(AssertionError, match="Conflicting"):
            DataProto.concat([dp1, dp2])

    def test_concat_merges_metrics(self):
        dp1 = _make_proto(batch_size=2, meta={"metrics": {"loss": 0.5}})
        dp2 = _make_proto(batch_size=2, meta={"metrics": {"loss": 0.3}})
        result = DataProto.concat([dp1, dp2])
        assert "metrics" in result.meta_info
        assert result.meta_info["metrics"]["loss"] == [0.5, 0.3]


# ═══════════════════════════════════════════════════════════════════
#  DataProto — chunk / split
# ═══════════════════════════════════════════════════════════════════


class TestDataProtoChunkSplit:
    def test_chunk_even(self):
        dp = _make_proto(batch_size=6)
        chunks = dp.chunk(3)
        assert len(chunks) == 3
        assert all(len(c) == 2 for c in chunks)

    def test_chunk_uneven_without_padding_raises(self):
        dp = _make_proto(batch_size=7)
        with pytest.raises(AssertionError, match="not divisible"):
            dp.chunk(3)

    def test_chunk_preserves_meta(self):
        dp = _make_proto(batch_size=4, meta={"key": "val"})
        chunks = dp.chunk(2)
        assert all(c.meta_info["key"] == "val" for c in chunks)

    def test_chunk_non_tensor_split(self):
        dp = _make_proto(batch_size=6)
        chunks = dp.chunk(3)
        for c in chunks:
            assert len(c.non_tensor_batch["text"]) == 2

    def test_split_even(self):
        dp = _make_proto(batch_size=6)
        parts = dp.split(2)
        assert len(parts) == 3
        assert all(len(p) == 2 for p in parts)

    def test_split_uneven(self):
        dp = _make_proto(batch_size=7)
        parts = dp.split(3)
        assert len(parts) == 3
        assert len(parts[0]) == 3
        assert len(parts[1]) == 3
        assert len(parts[2]) == 1

    def test_split_larger_than_batch(self):
        dp = _make_proto(batch_size=3)
        parts = dp.split(10)
        assert len(parts) == 1
        assert len(parts[0]) == 3

    def test_chunk_then_concat_roundtrip(self):
        dp = _make_proto(batch_size=8)
        chunks = dp.chunk(4)
        restored = DataProto.concat(chunks)
        assert len(restored) == 8
        assert torch.equal(restored.batch["input_ids"], dp.batch["input_ids"])


# ═══════════════════════════════════════════════════════════════════
#  DataProto — repeat
# ═══════════════════════════════════════════════════════════════════


class TestDataProtoRepeat:
    def test_repeat_interleave(self):
        dp = DataProto.from_dict(
            tensors={"x": torch.tensor([1.0, 2.0, 3.0])},
            non_tensors={"t": ["a", "b", "c"]},
        )
        repeated = dp.repeat(2, interleave=True)
        assert len(repeated) == 6
        assert torch.equal(repeated.batch["x"], torch.tensor([1.0, 1.0, 2.0, 2.0, 3.0, 3.0]))
        assert list(repeated.non_tensor_batch["t"]) == ["a", "a", "b", "b", "c", "c"]

    def test_repeat_tile(self):
        dp = DataProto.from_dict(
            tensors={"x": torch.tensor([1.0, 2.0, 3.0])},
            non_tensors={"t": ["a", "b", "c"]},
        )
        repeated = dp.repeat(2, interleave=False)
        assert len(repeated) == 6
        assert torch.equal(repeated.batch["x"], torch.tensor([1.0, 2.0, 3.0, 1.0, 2.0, 3.0]))
        assert list(repeated.non_tensor_batch["t"]) == ["a", "b", "c", "a", "b", "c"]

    def test_repeat_default_interleave(self):
        dp = _make_proto(batch_size=3)
        repeated = dp.repeat(3)
        assert len(repeated) == 9


# ═══════════════════════════════════════════════════════════════════
#  DataProto — sample_level_repeat
# ═══════════════════════════════════════════════════════════════════


class TestDataProtoSampleLevelRepeat:
    def test_basic(self):
        dp = DataProto.from_dict(
            tensors={"x": torch.tensor([10.0, 20.0, 30.0])},
            non_tensors={"t": ["a", "b", "c"]},
        )
        repeated = dp.sample_level_repeat([1, 2, 3])
        assert len(repeated) == 6
        assert torch.equal(
            repeated.batch["x"],
            torch.tensor([10.0, 20.0, 20.0, 30.0, 30.0, 30.0]),
        )

    def test_from_tuple(self):
        dp = DataProto.from_dict(tensors={"x": torch.ones(3)})
        repeated = dp.sample_level_repeat((2, 2, 2))
        assert len(repeated) == 6

    def test_from_numpy(self):
        dp = DataProto.from_dict(tensors={"x": torch.ones(3)})
        repeated = dp.sample_level_repeat(np.array([1, 1, 1]))
        assert len(repeated) == 3

    def test_from_tensor(self):
        dp = DataProto.from_dict(tensors={"x": torch.ones(3)})
        repeated = dp.sample_level_repeat(torch.tensor([3, 1, 2]))
        assert len(repeated) == 6


# ═══════════════════════════════════════════════════════════════════
#  DataProto — repeat_by_counts
# ═══════════════════════════════════════════════════════════════════


class TestDataProtoRepeatByCounts:
    def test_interleave(self):
        dp = DataProto.from_dict(
            tensors={"x": torch.tensor([1.0, 2.0, 3.0])},
            non_tensors={"t": ["a", "b", "c"]},
        )
        result = dp.repeat_by_counts([1, 2, 3], interleave=True)
        assert len(result) == 6
        assert torch.equal(
            result.batch["x"],
            torch.tensor([1.0, 2.0, 2.0, 3.0, 3.0, 3.0]),
        )

    def test_non_interleave(self):
        dp = DataProto.from_dict(
            tensors={"x": torch.tensor([1.0, 2.0, 3.0])},
            non_tensors={"t": ["a", "b", "c"]},
        )
        result = dp.repeat_by_counts([2, 1, 3], interleave=False)
        assert len(result) == 6


# ═══════════════════════════════════════════════════════════════════
#  DataProto — reorder
# ═══════════════════════════════════════════════════════════════════


class TestDataProtoReorder:
    def test_reorder(self):
        dp = DataProto.from_dict(
            tensors={"x": torch.tensor([10.0, 20.0, 30.0])},
            non_tensors={"t": ["a", "b", "c"]},
        )
        dp.reorder(torch.tensor([2, 0, 1]))
        assert torch.equal(dp.batch["x"], torch.tensor([30.0, 10.0, 20.0]))
        assert list(dp.non_tensor_batch["t"]) == ["c", "a", "b"]


# ═══════════════════════════════════════════════════════════════════
#  DataProto — to device
# ═══════════════════════════════════════════════════════════════════


class TestDataProtoTo:
    def test_to_cpu(self):
        dp = _make_proto(batch_size=4)
        dp.to("cpu")
        assert dp.batch.device == torch.device("cpu")

    def test_to_with_none_batch(self):
        dp = DataProto.from_dict(non_tensors={"a": ["x"]})
        dp.to("cpu")  # should not raise


# ═══════════════════════════════════════════════════════════════════
#  DataProto — make_iterator
# ═══════════════════════════════════════════════════════════════════


class TestDataProtoMakeIterator:
    def test_basic_iteration(self):
        dp = _make_proto(batch_size=8, with_non_tensor=False)
        batches = list(dp.make_iterator(mini_batch_size=4, epochs=1))
        assert len(batches) == 2
        assert all(len(b) == 4 for b in batches)

    def test_multiple_epochs(self):
        dp = _make_proto(batch_size=4, with_non_tensor=False)
        batches = list(dp.make_iterator(mini_batch_size=2, epochs=3))
        assert len(batches) == 6

    def test_mini_batch_not_divisible_raises(self):
        dp = _make_proto(batch_size=5, with_non_tensor=False)
        with pytest.raises(AssertionError):
            list(dp.make_iterator(mini_batch_size=3, epochs=1))

    def test_seeded_iteration_deterministic(self):
        dp = _make_proto(batch_size=8, with_non_tensor=False)
        batches1 = [b.batch["input_ids"].clone() for b in dp.make_iterator(mini_batch_size=4, epochs=1, seed=42)]
        batches2 = [b.batch["input_ids"].clone() for b in dp.make_iterator(mini_batch_size=4, epochs=1, seed=42)]
        assert all(torch.equal(a, b) for a, b in zip(batches1, batches2, strict=True))

    def test_meta_info_propagated(self):
        dp = _make_proto(batch_size=4, with_non_tensor=False, meta={"lr": 0.01})
        batches = list(dp.make_iterator(mini_batch_size=2, epochs=1))
        assert all(b.meta_info["lr"] == 0.01 for b in batches)


# ═══════════════════════════════════════════════════════════════════
#  DataProto — padding
# ═══════════════════════════════════════════════════════════════════


class TestDataProtoPadding:
    def test_is_padding_enabled_default_false(self):
        dp = _make_proto(batch_size=4)
        assert dp.is_padding_enabled() is False

    def test_is_padding_enabled_from_meta(self):
        dp = _make_proto(batch_size=4, meta={DataProtoConfig.auto_padding_key: True})
        assert dp.is_padding_enabled() is True

    def test_padding_adds_elements(self):
        dp = _make_proto(batch_size=4)
        original_len = len(dp)
        dp.padding(2)
        assert len(dp) == original_len + 2

    def test_padding_zero_is_noop(self):
        dp = _make_proto(batch_size=4)
        dp.padding(0)
        assert len(dp) == 4


# ═══════════════════════════════════════════════════════════════════
#  DataProto — unfold_column_chunks
# ═══════════════════════════════════════════════════════════════════


class TestDataProtoUnfoldColumnChunks:
    def test_basic_unfold(self):
        dp = DataProto.from_dict(
            tensors={
                "responses": torch.randn(2, 6, 10),
                "prompt": torch.randn(2, 5),
            },
        )
        result = dp.unfold_column_chunks(n_split=3, split_keys=["responses"])
        assert len(result) == 6
        assert result.batch["responses"].shape == (6, 2, 10)
        assert result.batch["prompt"].shape == (6, 5)

    def test_non_tensor_unfold(self):
        dp = DataProto.from_dict(
            tensors={"x": torch.randn(2, 4)},
            non_tensors={
                "split_me": np.arange(8).reshape(2, 4),
                "repeat_me": np.array(["a", "b"], dtype=object),
            },
        )
        result = dp.unfold_column_chunks(n_split=2, split_keys=["split_me"])
        assert len(result) == 4
        assert result.non_tensor_batch["split_me"].shape == (4, 2)
        assert len(result.non_tensor_batch["repeat_me"]) == 4


# ═══════════════════════════════════════════════════════════════════
#  pad_dataproto_to_divisor / unpad_dataproto
# ═══════════════════════════════════════════════════════════════════


class TestPadUnpadDataproto:
    def test_pad_when_divisible(self):
        dp = _make_proto(batch_size=6)
        padded, pad_size = pad_dataproto_to_divisor(dp, 3)
        assert pad_size == 0
        assert len(padded) == 6

    def test_pad_when_not_divisible(self):
        dp = _make_proto(batch_size=7)
        padded, pad_size = pad_dataproto_to_divisor(dp, 4)
        assert pad_size == 1
        assert len(padded) == 8

    def test_unpad_removes_padding(self):
        dp = _make_proto(batch_size=7)
        padded, pad_size = pad_dataproto_to_divisor(dp, 4)
        unpadded = unpad_dataproto(padded, pad_size)
        assert len(unpadded) == 7

    def test_unpad_zero_is_noop(self):
        dp = _make_proto(batch_size=4)
        result = unpad_dataproto(dp, 0)
        assert len(result) == 4

    def test_pad_larger_than_data(self):
        dp = _make_proto(batch_size=2)
        padded, pad_size = pad_dataproto_to_divisor(dp, 5)
        assert pad_size == 3
        assert len(padded) == 5


# ═══════════════════════════════════════════════════════════════════
#  fold_batch_dim / unfold_batch_dim
# ═══════════════════════════════════════════════════════════════════


class TestFoldUnfoldBatchDim:
    def test_fold(self):
        dp = _make_proto(batch_size=6, seq_len=4, with_non_tensor=False)
        folded = fold_batch_dim(dp, new_batch_size=2)
        assert folded.batch.batch_size == (2,)
        assert folded.batch["input_ids"].shape == (2, 3, 4)

    def test_fold_non_tensor(self):
        dp = _make_proto(batch_size=6, seq_len=4)
        folded = fold_batch_dim(dp, new_batch_size=2)
        assert folded.non_tensor_batch["text"].shape == (2, 3)

    def test_unfold(self):
        dp = _make_proto(batch_size=6, seq_len=4, with_non_tensor=False)
        folded = fold_batch_dim(dp, new_batch_size=2)
        unfolded = unfold_batch_dim(folded, batch_dims=2)
        assert len(unfolded) == 6
        assert unfolded.batch["input_ids"].shape == (6, 4)

    def test_fold_unfold_roundtrip(self):
        dp = _make_proto(batch_size=6, seq_len=4, with_non_tensor=False)
        folded = fold_batch_dim(dp, new_batch_size=3)
        unfolded = unfold_batch_dim(folded, batch_dims=2)
        assert torch.equal(unfolded.batch["input_ids"], dp.batch["input_ids"])

    def test_fold_not_divisible_raises(self):
        dp = _make_proto(batch_size=7, with_non_tensor=False)
        with pytest.raises(AssertionError):
            fold_batch_dim(dp, new_batch_size=3)


# ═══════════════════════════════════════════════════════════════════
#  collate_fn
# ═══════════════════════════════════════════════════════════════════


class TestCollateFn:
    def test_basic_collate(self):
        items = [
            DataProtoItem(
                batch=TensorDict({"x": torch.tensor([1.0, 2.0])}, batch_size=()),
                non_tensor_batch={"t": "a"},
            ),
            DataProtoItem(
                batch=TensorDict({"x": torch.tensor([3.0, 4.0])}, batch_size=()),
                non_tensor_batch={"t": "b"},
            ),
        ]
        result = collate_fn(items)
        assert isinstance(result, DataProto)
        assert len(result) == 2
        assert result.batch["x"].shape == (2, 2)


# ═══════════════════════════════════════════════════════════════════
#  DataProto — pickle serialization
# ═══════════════════════════════════════════════════════════════════


class TestDataProtoPickle:
    def test_pickle_roundtrip(self):
        dp = _make_proto(batch_size=4, meta={"lr": 0.01})
        data = pickle.dumps(dp)
        restored = pickle.loads(data)
        assert len(restored) == 4
        assert torch.equal(restored.batch["input_ids"], dp.batch["input_ids"])
        assert np.array_equal(restored.non_tensor_batch["text"], dp.non_tensor_batch["text"])
        assert restored.meta_info["lr"] == 0.01

    def test_pickle_empty(self):
        dp = DataProto()
        data = pickle.dumps(dp)
        restored = pickle.loads(data)
        assert len(restored) == 0

    def test_save_load_disk(self):
        dp = _make_proto(batch_size=4, meta={"epoch": 5})
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name
        try:
            dp.save_to_disk(path)
            loaded = DataProto.load_from_disk(path)
            assert len(loaded) == 4
            assert torch.equal(loaded.batch["input_ids"], dp.batch["input_ids"])
            assert loaded.meta_info["epoch"] == 5
        finally:
            os.unlink(path)


# ═══════════════════════════════════════════════════════════════════
#  DataProto — get_data_info / print_size
# ═══════════════════════════════════════════════════════════════════


class TestDataProtoInfo:
    def test_get_data_info(self):
        dp = _make_proto(batch_size=4, meta={"lr": 0.01})
        info = dp.get_data_info()
        assert "input_ids" in info
        assert "scores" in info
        assert "text" in info
        assert "lr" in info

    def test_print_size(self, capsys):
        dp = _make_proto(batch_size=4)
        dp.print_size("test")
        captured = capsys.readouterr()
        assert "GB" in captured.out
        # Also verify no-prefix variant works
        dp.print_size()
        captured2 = capsys.readouterr()
        assert "GB" in captured2.out


# ═══════════════════════════════════════════════════════════════════
#  DataProtoItem
# ═══════════════════════════════════════════════════════════════════


class TestDataProtoItem:
    def test_default_fields(self):
        item = DataProtoItem()
        assert item.batch is None
        assert item.non_tensor_batch == {}
        assert item.meta_info == {}

    def test_from_dataproto_indexing(self):
        dp = _make_proto(batch_size=4)
        item = dp[2]
        assert isinstance(item, DataProtoItem)
        assert torch.equal(item.batch["input_ids"], dp.batch["input_ids"][2])
        assert item.non_tensor_batch["text"] == dp.non_tensor_batch["text"][2]
        assert item.meta_info == dp.meta_info


# ═══════════════════════════════════════════════════════════════════
#  Edge cases and integration
# ═══════════════════════════════════════════════════════════════════


class TestDataProtoEdgeCases:
    def test_batch_only_no_non_tensor(self):
        dp = _make_proto(batch_size=4, with_non_tensor=False)
        assert dp.non_tensor_batch == {}
        sliced = dp[1:3]
        assert len(sliced) == 2
        chunks = dp.chunk(2)
        assert len(chunks) == 2

    def test_non_tensor_only_no_batch(self):
        dp = DataProto.from_dict(non_tensors={"labels": ["a", "b", "c", "d"]})
        assert dp.batch is None
        assert len(dp) == 4
        sliced = dp[1:3]
        assert len(sliced) == 2

    def test_select_idxs_boolean_list(self):
        dp = _make_proto(batch_size=4)
        selected = dp[[True, False, True, False]]
        assert len(selected) == 2

    def test_copy_independence(self):
        dp = _make_proto(batch_size=4)
        dp_copy = copy.deepcopy(dp)
        dp_copy.batch["scores"].fill_(999)
        assert not torch.equal(dp.batch["scores"], dp_copy.batch["scores"])

    def test_select_idxs_preserves_device(self):
        dp = _make_proto(batch_size=4)
        selected = dp.select_idxs([0, 2])
        assert selected.batch.device == dp.batch.device

    def test_multi_dim_tensors(self):
        dp = DataProto.from_dict(
            tensors={
                "embeddings": torch.randn(4, 8, 16),
                "mask": torch.ones(4, 8),
            }
        )
        assert len(dp) == 4
        sliced = dp[1:3]
        assert sliced.batch["embeddings"].shape == (2, 8, 16)

    def test_auto_padding_from_dict(self):
        dp = DataProto.from_dict(
            tensors={"x": torch.ones(3)},
            auto_padding=True,
        )
        assert dp.is_padding_enabled()

    def test_select_idxs_with_numpy_bool_mask(self):
        dp = _make_proto(batch_size=4)
        mask = np.array([True, False, True, False])
        selected = dp.select_idxs(mask)
        assert len(selected) == 2

    def test_chunk_non_tensor_only(self):
        dp = DataProto.from_dict(non_tensors={"t": ["a", "b", "c", "d"]})
        chunks = dp.chunk(2)
        assert len(chunks) == 2
        assert len(chunks[0]) == 2
        assert len(chunks[1]) == 2

    def test_repeat_non_tensor_only(self):
        dp = DataProto.from_dict(non_tensors={"t": ["a", "b"]})
        repeated = dp.repeat(3, interleave=True)
        assert len(repeated) == 6
        assert list(repeated.non_tensor_batch["t"]) == ["a", "a", "a", "b", "b", "b"]

    def test_split_non_tensor_only(self):
        dp = DataProto.from_dict(non_tensors={"t": ["a", "b", "c", "d", "e"]})
        parts = dp.split(2)
        assert len(parts) == 3
        assert len(parts[0]) == 2
        assert len(parts[2]) == 1

    def test_from_dict_num_batch_dims_gt1_rejected(self):
        """DataProto only supports 1D batch dims; multi-dim raises."""
        with pytest.raises(AssertionError, match="only support num_batch_dims=1"):
            DataProto.from_dict(
                tensors={"x": torch.randn(2, 3, 4)},
                num_batch_dims=2,
            )

    def test_pad_unpad_preserves_tensor_values(self):
        dp = DataProto.from_dict(
            tensors={"x": torch.tensor([10.0, 20.0, 30.0])},
            non_tensors={"t": ["a", "b", "c"]},
        )
        padded, pad_size = pad_dataproto_to_divisor(dp, 4)
        unpadded = unpad_dataproto(padded, pad_size)
        assert torch.equal(unpadded.batch["x"], dp.batch["x"])
        assert list(unpadded.non_tensor_batch["t"]) == ["a", "b", "c"]

    def test_select_idxs_empty_list(self):
        dp = _make_proto(batch_size=4)
        selected = dp.select_idxs([])
        assert len(selected) == 0

    def test_chunk_with_auto_padding_handles_uneven(self):
        dp = DataProto.from_dict(
            tensors={"x": torch.randn(7, 4)},
            auto_padding=True,
        )
        chunks = dp.chunk(3)
        assert len(chunks) == 3
        total = sum(len(c) for c in chunks)
        assert total == 7  # original size, not padded

    def test_repeat_by_counts_with_zero(self):
        dp = DataProto.from_dict(
            tensors={"x": torch.tensor([1.0, 2.0, 3.0])},
            non_tensors={"t": ["a", "b", "c"]},
        )
        result = dp.repeat_by_counts([0, 2, 1], interleave=True)
        assert len(result) == 3
        assert torch.equal(result.batch["x"], torch.tensor([2.0, 2.0, 3.0]))

    def test_reorder_identity(self):
        dp = DataProto.from_dict(
            tensors={"x": torch.tensor([1.0, 2.0, 3.0])},
        )
        original = dp.batch["x"].clone()
        dp.reorder(torch.tensor([0, 1, 2]))
        assert torch.equal(dp.batch["x"], original)

    def test_sample_level_repeat_with_zeros(self):
        dp = DataProto.from_dict(
            tensors={"x": torch.tensor([1.0, 2.0, 3.0])},
        )
        result = dp.sample_level_repeat([0, 0, 3])
        assert len(result) == 3
        assert torch.equal(result.batch["x"], torch.tensor([3.0, 3.0, 3.0]))

    def test_get_data_info_nested_meta(self):
        dp = _make_proto(
            batch_size=2,
            meta={
                "config": {"lr": 0.01},
                "tags": ["a", "b"],
                "empty_dict": {},
            },
        )
        info = dp.get_data_info()
        assert "config" in info
        assert "tags" in info
        assert "empty_dict" in info

    def test_pickle_with_only_non_tensor(self):
        dp = DataProto.from_dict(non_tensors={"labels": ["a", "b", "c"]})
        data = pickle.dumps(dp)
        restored = pickle.loads(data)
        assert len(restored) == 3
        assert list(restored.non_tensor_batch["labels"]) == ["a", "b", "c"]

    def test_make_iterator_all_data_seen(self):
        """Verify every element appears in exactly one mini-batch per epoch."""
        dp = DataProto.from_dict(tensors={"x": torch.arange(8).float()})
        batches = list(dp.make_iterator(mini_batch_size=4, epochs=1, seed=0))
        all_vals = torch.cat([b.batch["x"] for b in batches]).sort().values
        assert torch.equal(all_vals, torch.arange(8).float())


# ═══════════════════════════════════════════════════════════════════
#  Hardened edge cases
# ═══════════════════════════════════════════════════════════════════


class TestDeepEqualEdgeCases:
    def test_nan_tensor_equality(self):
        """Two tensors with NaN in same positions should be considered equal."""
        t1 = torch.tensor([1.0, float("nan"), 3.0])
        t2 = torch.tensor([1.0, float("nan"), 3.0])
        assert _deep_equal(t1, t2, set()), "Tensors with NaN in same positions should be equal"

    def test_nan_tensor_inequality(self):
        """Two tensors with NaN in different positions should NOT be equal."""
        t1 = torch.tensor([1.0, float("nan"), 3.0])
        t2 = torch.tensor([float("nan"), 2.0, 3.0])
        assert not _deep_equal(t1, t2, set())

    def test_nested_dict_equality(self):
        d1 = {"a": {"b": [1, 2, 3]}, "c": torch.tensor([1.0])}
        d2 = {"a": {"b": [1, 2, 3]}, "c": torch.tensor([1.0])}
        assert _deep_equal(d1, d2, set())

    def test_different_types_not_equal(self):
        assert not _deep_equal(1, "1", set())
        assert not _deep_equal([1], (1,), set())

    def test_empty_structures_equal(self):
        assert _deep_equal({}, {}, set())
        assert _deep_equal([], [], set())

    def test_none_equality(self):
        assert _deep_equal(None, None, set())
        assert not _deep_equal(None, 0, set())


class TestUnionTensorDictEdgeCases:
    def test_nan_tensor_conflict(self):
        """Two TensorDicts with NaN tensors for the same key should not raise
        if NaN positions match (semantically equal)."""
        td1 = TensorDict({"a": torch.tensor([1.0, float("nan")])}, batch_size=(2,))
        td2 = TensorDict({"a": torch.tensor([1.0, float("nan")])}, batch_size=(2,))
        # Should not raise — NaN in same positions
        result = union_tensor_dict(td1, td2)
        assert "a" in result.keys()


class TestUnionNumpyDictEdgeCases:
    def test_nan_numpy_conflict(self):
        """NaN values in numpy arrays at same positions should not cause assertion failure."""
        d1 = {"a": np.array([1.0, float("nan")])}
        d2 = {"a": np.array([1.0, float("nan")])}
        result = union_numpy_dict(d1, d2)
        assert "a" in result

    def test_object_array_conflict(self):
        """Object arrays with same values should merge."""
        d1 = {"labels": np.array(["cat", "dog"], dtype=object)}
        d2 = {"labels": np.array(["cat", "dog"], dtype=object)}
        result = union_numpy_dict(d1, d2)
        assert "labels" in result


class TestDataProtoHardenedEdgeCases:
    def test_single_element_dataproto(self):
        dp = DataProto.from_dict(tensors={"x": torch.tensor([42.0])})
        assert len(dp) == 1
        assert dp.batch["x"].item() == 42.0

    def test_from_dict_with_only_non_tensors(self):
        dp = DataProto.from_dict(non_tensors={"labels": ["a", "b", "c"]})
        assert len(dp) == 3
        assert list(dp.non_tensor_batch["labels"]) == ["a", "b", "c"]

    def test_getitem_returns_dataproto_item(self):
        dp = DataProto.from_dict(tensors={"x": torch.arange(5).float()})
        item = dp[0]
        assert isinstance(item, DataProtoItem)

    def test_slice_preserves_data(self):
        dp = DataProto.from_dict(tensors={"x": torch.arange(10).float()})
        sliced = dp[:5]
        assert len(sliced) == 5
        assert torch.equal(sliced.batch["x"], torch.arange(5).float())

    def test_make_iterator_mini_batch_larger_than_data_raises(self):
        """Mini-batch size must evenly divide data size."""
        dp = DataProto.from_dict(tensors={"x": torch.arange(3).float()})
        with pytest.raises(AssertionError):
            list(dp.make_iterator(mini_batch_size=10, epochs=1, seed=0))


class TestPadUnpadEdgeCases:
    def test_pad_already_divisible(self):
        """If size is already divisible, no padding should be added."""
        dp = _make_proto(batch_size=4, with_non_tensor=False)
        padded, pad_size = pad_dataproto_to_divisor(dp, size_divisor=4)
        assert pad_size == 0
        assert len(padded) == 4

    def test_pad_then_unpad_roundtrip(self):
        """Padding then unpadding should recover original data."""
        dp = _make_proto(batch_size=5, with_non_tensor=True)
        padded, pad_size = pad_dataproto_to_divisor(dp, size_divisor=4)
        assert len(padded) == 8  # next multiple of 4
        assert pad_size == 3
        unpadded = unpad_dataproto(padded, pad_size)
        assert len(unpadded) == 5
        assert torch.equal(unpadded.batch["input_ids"], dp.batch["input_ids"])


class TestDataProtoConfigEdgeCases:
    def test_auto_padding_default_false(self):
        """Auto-padding should default to False."""
        # Reset config
        DataProtoConfig._config = {}
        old = os.environ.pop("AXON_AUTO_PADDING", None)
        try:
            assert DataProtoConfig.auto_padding is False
        finally:
            if old is not None:
                os.environ["AXON_AUTO_PADDING"] = old

    def test_auto_padding_env_var_true(self):
        old = os.environ.get("AXON_AUTO_PADDING")
        try:
            os.environ["AXON_AUTO_PADDING"] = "TRUE"
            assert DataProtoConfig.auto_padding is True
        finally:
            if old is None:
                os.environ.pop("AXON_AUTO_PADDING", None)
            else:
                os.environ["AXON_AUTO_PADDING"] = old

    def test_auto_padding_setter(self):
        old_config = DataProtoConfig._config.copy()
        try:
            DataProtoConfig.auto_padding = True
            assert DataProtoConfig._config.get(DataProtoConfig.auto_padding_key) is True
        finally:
            DataProtoConfig._config = old_config

    def test_auto_padding_setter_rejects_non_bool(self):
        with pytest.raises(AssertionError, match="must be bool"):
            DataProtoConfig.auto_padding = "yes"
