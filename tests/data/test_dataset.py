"""
Tests for axon.data.dataset — RLDataset and collate_rl_dataset.
"""

import json
import tempfile
import os

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from axon.data.dataset import RLDataset, collate_rl_dataset


# ───────────────── fixtures ─────────────────

@pytest.fixture
def sample_records():
    return [
        {"question": "What is 2+2?", "answer": "4"},
        {"question": "Capital of France?", "answer": "Paris"},
        {"question": "Largest planet?", "answer": "Jupiter"},
    ]


@pytest.fixture
def parquet_file(sample_records, tmp_path):
    path = str(tmp_path / "data.parquet")
    table = pa.table({k: [r[k] for r in sample_records] for k in sample_records[0]})
    pq.write_table(table, path)
    return path


@pytest.fixture
def jsonl_file(sample_records, tmp_path):
    path = str(tmp_path / "data.jsonl")
    with open(path, "w") as f:
        for record in sample_records:
            f.write(json.dumps(record) + "\n")
    return path


# ═══════════════════════════════════════════════════════════════════
#  RLDataset — construction
# ═══════════════════════════════════════════════════════════════════

class TestRLDatasetConstruction:
    def test_no_data_files(self):
        ds = RLDataset()
        assert ds.data is None

    def test_load_parquet(self, parquet_file):
        ds = RLDataset(data_files=parquet_file)
        assert ds.data is not None
        assert len(ds) == 3

    def test_load_jsonl(self, jsonl_file):
        ds = RLDataset(data_files=jsonl_file)
        assert ds.data is not None
        assert len(ds) == 3

    def test_load_list_of_files(self, parquet_file, jsonl_file):
        ds = RLDataset(data_files=[parquet_file, jsonl_file])
        assert len(ds) == 6

    def test_load_multiple_parquets(self, sample_records, tmp_path):
        paths = []
        for i in range(3):
            path = str(tmp_path / f"data_{i}.parquet")
            table = pa.table({k: [r[k] for r in sample_records] for k in sample_records[0]})
            pq.write_table(table, path)
            paths.append(path)
        ds = RLDataset(data_files=paths)
        assert len(ds) == 9

    def test_unsupported_format_raises(self, tmp_path):
        path = str(tmp_path / "data.csv")
        with open(path, "w") as f:
            f.write("a,b\n1,2\n")
        with pytest.raises(ValueError, match="Unsupported format"):
            RLDataset(data_files=path)


# ═══════════════════════════════════════════════════════════════════
#  RLDataset — __len__
# ═══════════════════════════════════════════════════════════════════

class TestRLDatasetLen:
    def test_len_without_data(self):
        ds = RLDataset()
        assert len(ds) == 0


# ═══════════════════════════════════════════════════════════════════
#  RLDataset — __getitem__
# ═══════════════════════════════════════════════════════════════════

class TestRLDatasetGetItem:
    def test_getitem_with_data(self, parquet_file):
        ds = RLDataset(data_files=parquet_file)
        item = ds[0]
        assert "data" in item
        assert "index" in item
        assert item["index"] == 0
        assert item["data"]["question"] == "What is 2+2?"

    def test_getitem_without_data_raises(self):
        ds = RLDataset()
        with pytest.raises(IndexError):
            ds[42]



# ═══════════════════════════════════════════════════════════════════
#  collate_rl_dataset
# ═══════════════════════════════════════════════════════════════════

class TestCollateRlDataset:
    def test_basic_collation(self, parquet_file):
        ds = RLDataset(data_files=parquet_file)
        batch_items = [ds[i] for i in range(3)]
        collated = collate_rl_dataset(batch_items)
        assert "env_args" in collated
        assert "index" in collated
        assert isinstance(collated["index"], torch.Tensor)
        assert len(collated["env_args"]) == 3
        assert torch.equal(collated["index"], torch.tensor([0, 1, 2]))

    def test_collation_preserves_data(self, parquet_file):
        ds = RLDataset(data_files=parquet_file)
        batch_items = [ds[0], ds[2]]
        collated = collate_rl_dataset(batch_items)
        assert collated["env_args"][0]["question"] == "What is 2+2?"
        assert collated["env_args"][1]["question"] == "Largest planet?"

    def test_single_item_batch(self, parquet_file):
        ds = RLDataset(data_files=parquet_file)
        collated = collate_rl_dataset([ds[0]])
        assert len(collated["env_args"]) == 1
        assert collated["index"].shape == (1,)


# ═══════════════════════════════════════════════════════════════════
#  Edge cases
# ═══════════════════════════════════════════════════════════════════

class TestRLDatasetEdgeCases:
    def test_nonexistent_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            RLDataset(data_files=str(tmp_path / "nonexistent.parquet"))

    def test_single_row_parquet(self, tmp_path):
        path = str(tmp_path / "single.parquet")
        table = pa.table({"q": ["hello"], "a": ["world"]})
        pq.write_table(table, path)
        ds = RLDataset(data_files=path)
        assert len(ds) == 1
        item = ds[0]
        assert item["data"]["q"] == "hello"

    def test_large_field_values(self, tmp_path):
        path = str(tmp_path / "large.parquet")
        big_text = "x" * 100000
        table = pa.table({"text": [big_text]})
        pq.write_table(table, path)
        ds = RLDataset(data_files=path)
        assert ds[0]["data"]["text"] == big_text

    def test_mixed_file_formats(self, tmp_path):
        """Mix of parquet and jsonl files."""
        parquet_path = str(tmp_path / "data.parquet")
        jsonl_path = str(tmp_path / "data.jsonl")
        table = pa.table({"q": ["q1"], "a": ["a1"]})
        pq.write_table(table, parquet_path)
        with open(jsonl_path, "w") as f:
            f.write(json.dumps({"q": "q2", "a": "a2"}) + "\n")
        ds = RLDataset(data_files=[parquet_path, jsonl_path])
        assert len(ds) == 2

    def test_empty_jsonl_raises(self, tmp_path):
        """datasets library cannot infer schema from empty files."""
        from datasets.exceptions import DatasetGenerationError
        path = str(tmp_path / "empty.jsonl")
        with open(path, "w") as f:
            pass  # empty file
        # Depending on datasets version, this raises DatasetGenerationError or StopIteration
        with pytest.raises((DatasetGenerationError, StopIteration)):
            RLDataset(data_files=path)


# ---------------------------------------------------------------------------
# Hardened edge cases
# ---------------------------------------------------------------------------
class TestRLDatasetHardenedEdgeCases:
    def test_getitem_on_empty_dataset_should_raise_index_error(self):
        """Accessing any index on empty dataset should raise IndexError, not return {}."""
        ds = RLDataset()
        assert len(ds) == 0
        # Accessing index on empty dataset should raise IndexError per Python convention
        with pytest.raises(IndexError):
            ds[0]

    def test_getitem_out_of_bounds_should_raise(self, tmp_path):
        """Accessing beyond dataset length should raise IndexError."""
        parquet_path = str(tmp_path / "data.parquet")
        table = pa.table({"q": ["q1"], "a": ["a1"]})
        pq.write_table(table, parquet_path)
        ds = RLDataset(data_files=parquet_path)
        assert len(ds) == 1
        _ = ds[0]  # should work
        with pytest.raises((IndexError, KeyError)):
            ds[999]

    def test_collate_with_empty_dict_items_raises(self):
        """collate_rl_dataset with items missing 'data'/'index' keys should fail."""
        from axon.data.dataset import collate_rl_dataset

        # An empty dataset returns {} from __getitem__, which breaks collate
        with pytest.raises(KeyError):
            collate_rl_dataset([{}])

    def test_file_weights_length_mismatch_raises(self, tmp_path):
        """file_weights must match number of data_files."""
        parquet_path = str(tmp_path / "data.parquet")
        table = pa.table({"q": ["q1"]})
        pq.write_table(table, parquet_path)
        with pytest.raises(ValueError, match="data_mix length"):
            RLDataset(data_files=[parquet_path], file_weights=[1.0, 2.0])

    def test_file_weights_applied_correctly(self, tmp_path):
        """File weights should be normalized per-sample."""
        p1 = str(tmp_path / "d1.parquet")
        p2 = str(tmp_path / "d2.parquet")
        table1 = pa.table({"q": ["q1", "q2"]})  # 2 rows
        table2 = pa.table({"q": ["q3"]})  # 1 row
        pq.write_table(table1, p1)
        pq.write_table(table2, p2)
        ds = RLDataset(data_files=[p1, p2], file_weights=[2.0, 3.0])
        assert ds.sample_weights is not None
        assert len(ds.sample_weights) == 3
        # First file: weight 2.0 / 2 rows = 1.0 each
        assert ds.sample_weights[0] == pytest.approx(1.0)
        assert ds.sample_weights[1] == pytest.approx(1.0)
        # Second file: weight 3.0 / 1 row = 3.0
        assert ds.sample_weights[2] == pytest.approx(3.0)

    def test_negative_index(self, tmp_path):
        """Negative index should work (HuggingFace datasets support it)."""
        parquet_path = str(tmp_path / "data.parquet")
        table = pa.table({"q": ["q1", "q2", "q3"]})
        pq.write_table(table, parquet_path)
        ds = RLDataset(data_files=parquet_path)
        item = ds[-1]
        assert item["index"] == -1
        assert "data" in item

    def test_collate_preserves_all_fields(self, tmp_path):
        """Collated batch should preserve all fields from original data."""
        from axon.data.dataset import collate_rl_dataset

        batch = [
            {"data": {"q": "What is 2+2?", "a": "4", "meta": {"difficulty": "easy"}}, "index": 0},
            {"data": {"q": "What is 3+3?", "a": "6", "meta": {"difficulty": "easy"}}, "index": 1},
        ]
        result = collate_rl_dataset(batch)
        assert len(result["env_args"]) == 2
        assert result["env_args"][0]["q"] == "What is 2+2?"
        assert result["env_args"][1]["meta"]["difficulty"] == "easy"
        assert result["index"].tolist() == [0, 1]
