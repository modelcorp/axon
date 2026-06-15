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
import torch
from typing import Optional
from torch.utils.data import Dataset
import datasets


def collate_rl_dataset(batch: list[dict]) -> dict:
    """Custom collate that keeps 'data' as list of dicts, index as tensor."""
    return {
        "env_args": [item["data"] for item in batch],
        "index": torch.tensor([item["index"] for item in batch]),
    }


class RLDataset(Dataset):
    """
    Minimal RL dataset that loads parquet or jsonl files and returns dictionaries.
    If no data_files provided, yields empty dicts indefinitely.
    """

    def __init__(self, data_files: Optional[str | list[str]] = None, file_weights: Optional[list[float]] = None):
        self.sample_weights: Optional[list[float]] = None
        self.data = self._load(data_files, file_weights) if data_files else None

    def _load(self, data_files: str | list[str], file_weights: Optional[list[float]] = None) -> datasets.Dataset:
        if isinstance(data_files, str):
            data_files = [data_files]
        else:
            data_files = [str(p) for p in data_files]

        if file_weights is not None and len(file_weights) != len(data_files):
            raise ValueError(
                f"data_mix length ({len(file_weights)}) must match "
                f"train_files length ({len(data_files)})"
            )

        dfs = []
        for path in data_files:
            if path.endswith(".parquet"):
                dfs.append(self._load_parquet(path))
            elif path.endswith(".jsonl"):
                dfs.append(datasets.Dataset.from_json(path))
            else:
                raise ValueError(f"Unsupported format: {path}. Use .parquet or .jsonl")

        if file_weights is not None:
            self.sample_weights = []
            for df, w in zip(dfs, file_weights):
                n = len(df)
                per_sample = w / n if n > 0 else 0.0
                self.sample_weights.extend([per_sample] * n)

        return datasets.concatenate_datasets(dfs)

    def _load_parquet(self, path: str) -> datasets.Dataset:
        try:
            return datasets.Dataset.from_parquet(path)
        except Exception as e:
            print(f"Error loading {path}: {e}; falling back to polars")
            import polars as pl
            return datasets.Dataset.from_pandas(pl.read_parquet(path).to_pandas())

    def __len__(self):
        return len(self.data) if self.data else 0

    def __getitem__(self, index: int) -> dict:
        if not self.data:
            raise IndexError(f"index {index} out of range for empty dataset")
        return {
            "data": self.data[index],
            "index": index,
        }
