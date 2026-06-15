"""
2048 Dataset Generator

Generates training and test datasets for the 2048 environment. Each entry
contains environment configuration parameters (seed, size, target_value) used
to create a Game2048Env instance.

Usage:
    python recipes/2048/data.py --train_size 10000 --test_size 100
"""

import argparse
import os

import numpy as np
import pandas as pd

import axon

AXON_DIR = os.path.dirname(os.path.dirname(os.path.abspath(axon.__file__)))


def get_2048_dict(seed: int, size: int, target_value: int, max_turns: int) -> dict:
    return {
        "env_name": "2048",
        "seed": int(seed),
        "size": int(size),
        "target_value": int(target_value),
        "max_turns": int(max_turns),
    }


def generate_dataset_parameters(size: int, random_seed: int = 42) -> np.ndarray:
    np.random.seed(random_seed)
    return np.random.randint(0, 1_000_000, size=size)


def save_dataset(data: list[dict], filepath: str) -> None:
    df = pd.DataFrame(data)
    df.to_parquet(filepath)
    print(f"Saved {len(data)} entries to {filepath}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate 2048 environment configuration datasets for training and testing."
    )
    parser.add_argument(
        "--local_dir",
        default=os.path.join(AXON_DIR, "data/2048"),
        help="Local directory to save the datasets",
    )
    parser.add_argument("--train_size", type=int, default=10000)
    parser.add_argument("--test_size", type=int, default=100)
    parser.add_argument("--size", type=int, default=4, help="Board size (default 4x4)")
    parser.add_argument(
        "--target_value",
        type=int,
        default=128,
        help=(
            "Target tile value that counts as a win. Default is 128 to match "
            "openpipe/art's 2048 example. Raise to 256/512/2048 for a harder curriculum."
        ),
    )
    parser.add_argument("--max_turns", type=int, default=64)

    args = parser.parse_args()

    local_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_dir, exist_ok=True)
    print(f"Using local directory: {local_dir}")

    train_seeds = generate_dataset_parameters(args.train_size, random_seed=42)
    train_data = [get_2048_dict(seed, args.size, args.target_value, args.max_turns) for seed in train_seeds]

    test_seeds = generate_dataset_parameters(args.test_size, random_seed=123)
    test_data = [get_2048_dict(seed, args.size, args.target_value, args.max_turns) for seed in test_seeds]

    save_dataset(train_data, os.path.join(local_dir, "train.parquet"))
    save_dataset(test_data, os.path.join(local_dir, "test.parquet"))


if __name__ == "__main__":
    main()
