"""
Geo3k Dataset Generator

This script generates training and test datasets for the Geo3k geometry problem environment.
Each dataset entry contains geometry problem data including images and answers.

The generated datasets are saved as Parquet files and can be used for training
reinforcement learning agents on geometry problems.

Usage:
    python recipes/geo3k/data.py --local_dir data/axon-geo3k

The script generates:
- Training dataset: Geometry problems for agent training
- Test dataset: Separate set of problems for evaluation
"""

import argparse
import os

import datasets
import numpy as np
from datasets import Dataset

import axon

# Get the directory for Axon repo (axon.__file__)
AXON_DIR = os.path.dirname(os.path.dirname(os.path.abspath(axon.__file__)))


def main():
    parser = argparse.ArgumentParser(description="Generate programs using specified environment and policy.")
    parser.add_argument("--local_dir", default=os.path.join(AXON_DIR, "data/geo3k"))

    args = parser.parse_args()

    local_dir = args.local_dir
    os.makedirs(os.path.expanduser(local_dir), exist_ok=True)

    np.random.seed(42)

    data = datasets.load_dataset(
        "hiyouga/geometry3k",
    )
    train_data = data["train"]
    test_data = data["test"]

    # Convert to Parquet with env_name routing field
    train_ds = Dataset.from_list(train_data).add_column("env_name", ["geo3k"] * len(train_data))
    train_ds.to_parquet(os.path.join(local_dir, "train.parquet"))

    test_ds = Dataset.from_list(test_data).add_column("env_name", ["geo3k"] * len(test_data))
    test_ds.to_parquet(os.path.join(local_dir, "test.parquet"))


if __name__ == "__main__":
    main()
