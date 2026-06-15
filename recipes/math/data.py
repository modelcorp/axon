"""Script to prepare DeepScaler training and test datasets.

This script processes math problem datasets into a standardized format for training
and testing DeepScaler models. It loads problems from specified datasets, adds
instruction prompts, and saves the processed data as parquet files.
"""

import argparse
import os
from typing import Any

import pandas as pd

import axon
from axon.utils.dataset_utils import TestDataset, TrainDataset, load_dataset

AXON_PATH = os.path.dirname(os.path.dirname(axon.__file__))


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx : right_brace_idx + 1]

    return retval


def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[: len(left)] == left
        assert s[-1] == "}"
        return s[len(left) : -1]
    except Exception:
        return None


def extract_solution(solution_str: str) -> str:
    """Extract the final boxed solution from a solution string.

    Args:
        solution_str: Raw solution string that may contain multiple boxed answers

    Returns:
        The final boxed answer with box notation removed
    """
    return remove_boxed(last_boxed_only_string(solution_str))


def process_fn(example: dict[str, Any], instruction: str = None) -> dict[str, Any] | None:
    question = example.pop("problem")

    if instruction is None:
        instruction = "Let's think step by step and output the final answer within \\boxed{}."

    answer = example.pop("answer")

    return {"env_name": "math", "question": question, "answer": answer}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process datasets for DeepScaler training")
    parser.add_argument(
        "--local_dir",
        default=os.path.join(AXON_PATH, "data", "math"),
        help="Local directory to save processed datasets",
    )
    args = parser.parse_args()

    local_dir = args.local_dir

    # Make local directory if it doesn't exist
    os.makedirs(local_dir, exist_ok=True)

    # Initialize datasets - Load DeepScaler dataset from HuggingFace
    train_datasets = [
        TrainDataset.Math.DEEPSCALER,
        TrainDataset.Math.MATH,
        TrainDataset.Math.POLARIS,
        TrainDataset.Math.ACEREASON_MATH,
        TrainDataset.Math.NEMOTRON_RL_MATH,
    ]
    test_datasets = [
        TestDataset.Math.AIME24,
        TestDataset.Math.AIME25,
        TestDataset.Math.AIME26,
        TestDataset.Math.AMC23,
        TestDataset.Math.MATH,
        TestDataset.Math.MINERVA,
        TestDataset.Math.OLYMPIAD_BENCH,
    ]

    train_dataset_data = [load_dataset(d) for d in train_datasets]
    test_datasets_data = [load_dataset(d) for d in test_datasets]
    # Process training data
    for train_dataset, train_data_list in zip(train_datasets, train_dataset_data, strict=False):
        train_data: list[dict[str, Any]] = []

        df = pd.DataFrame(train_data_list)
        processed_df = df.reset_index().apply(lambda row: process_fn(row.to_dict()), axis=1)
        processed_examples = processed_df.dropna().tolist()
        train_data.extend(processed_examples)

        train_df = pd.DataFrame(train_data)
        os.makedirs(os.path.join(local_dir, "train"), exist_ok=True)
        train_df.to_parquet(os.path.join(local_dir, "train", f"{train_dataset.value.lower()}.parquet"))
        print(f"{train_dataset.value} train data size:", len(processed_examples))

    # Process and save each test dataset separately
    for test_dataset, test_data_list in zip(test_datasets, test_datasets_data, strict=False):
        test_data: list[dict[str, Any]] = []

        df = pd.DataFrame(test_data_list)
        processed_df = df.reset_index().apply(lambda row: process_fn(row.to_dict()), axis=1)
        processed_examples = processed_df.dropna().tolist()
        test_data.extend(processed_examples)

        dataset_name = test_dataset.value.lower()
        test_df = pd.DataFrame(test_data)
        os.makedirs(os.path.join(local_dir, "test"), exist_ok=True)
        test_df.to_parquet(os.path.join(local_dir, "test", f"{dataset_name}.parquet"))
        print(f"{test_dataset.value} test data size:", len(test_data))
