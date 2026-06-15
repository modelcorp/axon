"""
Sudoku Dataset Generator — with curriculum learning support.

Generates training and test datasets spanning multiple difficulty tiers,
ordered from easy to hard. Each entry contains env configuration params
(seed, size, num_clues, max_turns, difficulty) used to construct a
SudokuEnv at rollout time.

Difficulty tiers (default):
  1  easy       4x4, 12 clues (4 blanks),  max_turns 20
  2  medium     4x4,  6 clues (10 blanks), max_turns 40
  3  standard   9x9, 50 clues (31 blanks), max_turns 120
  4  hard       9x9, 36 clues (45 blanks), max_turns 200
  5  expert     9x9, 28 clues (53 blanks), max_turns 200

The dataset is sorted by difficulty so a curriculum-aware training loop
can progress through tiers. The `difficulty` column (1..5) lets the
trainer filter or weight samples per tier.

Usage:
    # Default curriculum (10k train, 100 test, all 5 tiers)
    python recipes/sudoku/data.py

    # Easy-only for quick experiments
    python recipes/sudoku/data.py --tiers easy

    # 9x9 only
    python recipes/sudoku/data.py --tiers standard,hard,expert

    # Custom proportions (must sum to 1.0)
    python recipes/sudoku/data.py --tier_weights 0.1,0.2,0.2,0.3,0.2
"""

import argparse
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

import axon

AXON_DIR = os.path.dirname(os.path.dirname(os.path.abspath(axon.__file__)))


@dataclass(frozen=True)
class DifficultyTier:
    name: str
    level: int  # 1 = easiest
    size: int
    num_clues: int
    max_turns: int


# Ordered by difficulty. Tune these to match your model's capability.
ALL_TIERS = [
    DifficultyTier("easy", 1, size=4, num_clues=12, max_turns=20),
    DifficultyTier("medium", 2, size=4, num_clues=6, max_turns=40),
    DifficultyTier("standard", 3, size=9, num_clues=50, max_turns=120),
    DifficultyTier("hard", 4, size=9, num_clues=36, max_turns=200),
    DifficultyTier("expert", 5, size=9, num_clues=28, max_turns=200),
]
TIER_BY_NAME = {t.name: t for t in ALL_TIERS}

# Default proportions: heavier on medium difficulty for bootstrap signal
DEFAULT_WEIGHTS = [0.10, 0.15, 0.25, 0.30, 0.20]


def make_entry(seed: int, tier: DifficultyTier) -> dict:
    return {
        "env_name": "sudoku",
        "seed": int(seed),
        "size": tier.size,
        "num_clues": tier.num_clues,
        "max_turns": tier.max_turns,
        "difficulty": tier.level,
        "difficulty_name": tier.name,
    }


def generate_seeds(n: int, random_seed: int) -> np.ndarray:
    rng = np.random.RandomState(random_seed)
    return rng.randint(0, 10_000_000, size=n)


def build_dataset(
    total_size: int,
    tiers: list[DifficultyTier],
    weights: list[float],
    random_seed: int,
) -> list[dict]:
    """Build a dataset with `total_size` entries distributed across `tiers`
    according to `weights`, sorted by difficulty (easy first)."""
    assert len(tiers) == len(weights)
    # Normalise weights.
    w_sum = sum(weights)
    normed = [w / w_sum for w in weights]

    # Allocate counts (rounding remainder goes to last tier).
    counts = [int(round(total_size * w)) for w in normed]
    remainder = total_size - sum(counts)
    counts[-1] += remainder

    data: list[dict] = []
    for tier, n in zip(tiers, counts, strict=False):
        seeds = generate_seeds(n, random_seed=random_seed + tier.level)
        for s in seeds:
            data.append(make_entry(int(s), tier))

    # Stable sort by difficulty level — preserves random order within tier.
    data.sort(key=lambda x: x["difficulty"])
    return data


def save_dataset(data: list[dict], filepath: str) -> None:
    df = pd.DataFrame(data)
    df.to_parquet(filepath)
    # Summary per tier.
    for name, group in df.groupby("difficulty_name", sort=False):
        print(
            f"  {name:>10s}: {len(group):>6d} entries  "
            f"(size={group['size'].iloc[0]}, clues={group['num_clues'].iloc[0]})"
        )
    print(f"  {'TOTAL':>10s}: {len(df):>6d} entries  ->  {filepath}")


def parse_tiers(tier_str: str) -> list[DifficultyTier]:
    names = [n.strip().lower() for n in tier_str.split(",")]
    tiers = []
    for n in names:
        if n not in TIER_BY_NAME:
            raise ValueError(f"Unknown tier '{n}'. Choose from: {list(TIER_BY_NAME)}")
        tiers.append(TIER_BY_NAME[n])
    # Sort by level.
    tiers.sort(key=lambda t: t.level)
    return tiers


def parse_weights(weight_str: str, n_tiers: int) -> list[float]:
    ws = [float(w.strip()) for w in weight_str.split(",")]
    if len(ws) != n_tiers:
        raise ValueError(f"Expected {n_tiers} weights, got {len(ws)}")
    if any(w < 0 for w in ws):
        raise ValueError("Weights must be non-negative")
    return ws


def main():
    parser = argparse.ArgumentParser(description="Generate Sudoku curriculum datasets for RL training.")
    parser.add_argument(
        "--local_dir",
        default=os.path.join(AXON_DIR, "data/sudoku"),
    )
    parser.add_argument("--train_size", type=int, default=10000)
    parser.add_argument("--test_size", type=int, default=100)
    parser.add_argument(
        "--tiers",
        type=str,
        default=",".join(t.name for t in ALL_TIERS),
        help="Comma-separated tier names to include (default: all).",
    )
    parser.add_argument(
        "--tier_weights",
        type=str,
        default=None,
        help=(
            "Comma-separated proportions per tier (must match number of tiers). "
            "Default: heavier on standard/hard for bootstrap signal."
        ),
    )

    args = parser.parse_args()

    tiers = parse_tiers(args.tiers)
    if args.tier_weights:
        weights = parse_weights(args.tier_weights, len(tiers))
    else:
        # Use default weights for the selected tiers.
        selected_defaults = [DEFAULT_WEIGHTS[i] for i, t in enumerate(ALL_TIERS) if t in tiers]
        if len(selected_defaults) != len(tiers):
            # Fallback: uniform
            selected_defaults = [1.0 / len(tiers)] * len(tiers)
        weights = selected_defaults

    local_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_dir, exist_ok=True)

    print(f"Generating curriculum dataset in {local_dir}")
    print(f"Tiers: {[t.name for t in tiers]}")
    print(f"Weights: {weights}")
    print()

    print("Train set:")
    train_data = build_dataset(args.train_size, tiers, weights, random_seed=42)
    save_dataset(train_data, os.path.join(local_dir, "train.parquet"))

    print("\nTest set:")
    test_data = build_dataset(args.test_size, tiers, weights, random_seed=123)
    save_dataset(test_data, os.path.join(local_dir, "test.parquet"))


if __name__ == "__main__":
    main()
