"""Script to prepare code datasets for training and testing.

This script processes code problem datasets into a standardized format for training
and testing models. It loads problems from various code datasets (APPS, CodeForces,
LiveCodeBench etc.), adds appropriate instruction prompts, and saves the processed
data as parquet files.
"""

import argparse
import json
import os
from typing import Any

import pandas as pd

# Get the axon package path
import axon
from axon.utils.dataset_utils import TestDataset, TrainDataset, fetch_live_code_bench_system_prompt, load_dataset

AXON_PATH = os.path.dirname(os.path.dirname(axon.__file__))


def process_fn(example: dict[str, Any], dataset_name=None) -> dict[str, Any] | None:
    question = example.pop("problem")
    tests = example.pop("tests")

    if example.get("metadata", {}):
        assert "func_name" in example["metadata"], (
            f"Function name is not found, check if your LCB data is preprocessed correctly: {example['metadata']}"
        )
        if isinstance(tests, dict):
            tests["metadata"] = example["metadata"]
        else:
            for test in tests:
                assert isinstance(test, dict), "Test is not a dict"
                test["metadata"] = example["metadata"]

    tests = json.dumps(tests)

    if dataset_name == "livecodebench":
        starter_code = example.get("starter_code", None)
        question = fetch_live_code_bench_system_prompt(question, starter_code)
    if isinstance(question, dict):
        question = json.dumps(question)
    return {"env_name": "competition_coding", "data_source": dataset_name, "question": question, "ground_truth": tests}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process datasets for DeepScaler training")
    parser.add_argument(
        "--local_dir",
        default=os.path.join(AXON_PATH, "data", "code"),
        help="Local directory to save processed datasets",
    )
    args = parser.parse_args()

    local_dir = args.local_dir
    train_dir = os.path.join(local_dir, "train")
    test_dir = os.path.join(local_dir, "test")

    # Make local directories if they don't exist
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    # Initialize datasets
    train_datasets = [TrainDataset.Code.PRIMEINTELLECT, TrainDataset.Code.TACO, TrainDataset.Code.LIVECODEBENCH]
    test_datasets = [TestDataset.Code.LIVECODEBENCH, TestDataset.Code.HUMANEVALPLUS]

    test_datasets_data = [load_dataset(d) for d in test_datasets]
    train_dataset_data = [load_dataset(d) for d in train_datasets]

    # Print dataset sizes
    for test_dataset, data in zip(test_datasets, test_datasets_data, strict=False):
        print(f"Test dataset {test_dataset.value}: {len(data)} examples")
    for train_dataset, data in zip(train_datasets, train_dataset_data, strict=False):
        print(f"Train dataset {train_dataset.value}: {len(data)} examples")

    # Process training data
    all_train_data = []

    for train_dataset, train_data_raw in zip(train_datasets, train_dataset_data, strict=False):
        train_data: list[dict[str, Any]] = []
        dataset_name = train_dataset.value.lower()  # Extract name from enum
        for idx, example in enumerate(train_data_raw):
            processed_example = process_fn(example, dataset_name)
            if not processed_example:
                continue  # Break here to inspect the problematic example
            if processed_example is not None:
                train_data.append(processed_example)
                all_train_data.append(processed_example)
        train_df = pd.DataFrame(train_data)
        train_df.to_parquet(os.path.join(train_dir, f"train_{dataset_name}.parquet"))

    # save all code dataset
    all_train_df = pd.DataFrame(all_train_data)
    all_train_df.to_parquet(os.path.join(train_dir, "deepcoder.parquet"))

    # Process and save each test dataset separately
    all_test_data = []
    for test_dataset, test_data_list in zip(test_datasets, test_datasets_data, strict=False):
        test_data: list[dict[str, Any]] = []
        dataset_name = test_dataset.value.lower()  # Extract name from enum
        for idx, example in enumerate(test_data_list):
            processed_example = process_fn(example, dataset_name)
            if processed_example is not None:
                test_data.append(processed_example)
                all_test_data.append(processed_example)
        test_df = pd.DataFrame(test_data)
        test_df.to_parquet(os.path.join(test_dir, f"test_{dataset_name}.parquet"))
