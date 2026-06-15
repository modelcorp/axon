"""Script to prepare formal math (Lean 4) training and test datasets.

Processes FineLeanCorpus and MiniF2F datasets into axon's standardized parquet format
for training Lean 4 proof generation models.

Usage:
    python recipes/formal_math/data.py
    python recipes/formal_math/data.py --train_size 5000 --local_dir /path/to/output

Prerequisites:
    pip install polars datasets
"""

import argparse
import os
import re
from typing import Any

import pandas as pd

import axon

AXON_PATH = os.path.dirname(os.path.dirname(axon.__file__))

PROMPT_TEMPLATE = """Complete the following Lean 4 code:

```lean4
{code}
```

Before producing the Lean 4 code to formally prove the given theorem, provide a detailed proof plan outlining the main proof steps and strategies.
The plan should highlight key ideas, intermediate lemmas, and proof structures that will guide the construction of the final formal proof."""

_NEEDLE_THEOREM = "theorem "


def _convert_to_by_sorry(s: str) -> str:
    """Normalize theorem ending to ':= by\\n  sorry'."""
    pattern = r" *:=\s*(?:by\s*)?(?:sorry\s*)?$"
    assert re.search(pattern, s, flags=re.MULTILINE), f"Pattern not found in: {s[-100:]}"
    return re.sub(pattern, "", s, flags=re.MULTILINE) + " := by\n  sorry"


def process_flc(
    local_dir: str,
    train_size: int = 20000,
    test_size: int = 100,
):
    """Process FineLeanCorpus dataset into train/test parquet files."""
    from datasets import load_dataset

    ds = load_dataset("m-a-p/FineLeanCorpus", split="train")
    print(f"Loaded FineLeanCorpus: {len(ds)} examples")

    # Filter to single-theorem statements
    ds = ds.filter(
        lambda batch: [code.count(_NEEDLE_THEOREM) == 1 for code in batch["lean_code"]],
        batched=True,
        num_proc=16,
    )
    print(f"After single-theorem filter: {len(ds)} examples")

    ds = ds.shuffle(seed=42)
    total_needed = min(len(ds), train_size + test_size)
    ds = ds.select(range(total_needed))
    split = ds.train_test_split(test_size=min(test_size, total_needed // 2), shuffle=False, seed=42)

    def make_question(statement, lean_code):
        # Insert statement as comment before the theorem
        code = lean_code.replace(_NEEDLE_THEOREM, f"/- {statement} -/\n{_NEEDLE_THEOREM}")
        code = _convert_to_by_sorry(code)
        return PROMPT_TEMPLATE.format(code=code)

    for split_name, split_ds in [("train", split["train"]), ("test", split["test"])]:
        rows: list[dict[str, Any]] = []
        for i, example in enumerate(split_ds):
            try:
                question = make_question(example["statement"], example["lean_code"])
                rows.append(
                    {
                        "env_name": "formal_math",
                        "question": question,
                        "answer": "",  # No ground-truth proof; reward comes from the verifier
                        "question_id": f"flc_{i}",
                    }
                )
            except (AssertionError, Exception) as e:
                print(f"Skipping example {i}: {e}")
                continue

        df = pd.DataFrame(rows)
        out_dir = os.path.join(local_dir, split_name)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "flc.parquet")
        df.to_parquet(out_path)
        print(f"Wrote {len(df)} {split_name} examples to {out_path}")


def process_minif2f(local_dir: str):
    """Process MiniF2F test set into a parquet file for evaluation."""
    from datasets import load_dataset

    ds = load_dataset("AI-MO/minif2f_test", split="train")
    print(f"Loaded MiniF2F: {len(ds)} examples")

    rows: list[dict[str, Any]] = []
    for i, example in enumerate(ds):
        try:
            code = _convert_to_by_sorry(example["formal_statement"])
            question = PROMPT_TEMPLATE.format(code=code)
            rows.append(
                {
                    "env_name": "formal_math",
                    "question": question,
                    "answer": "",
                    "question_id": f"minif2f_{i}",
                }
            )
        except (AssertionError, Exception) as e:
            print(f"Skipping MiniF2F example {i}: {e}")
            continue

    df = pd.DataFrame(rows)
    out_dir = os.path.join(local_dir, "test")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "minif2f.parquet")
    df.to_parquet(out_path)
    print(f"Wrote {len(df)} MiniF2F examples to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare formal math datasets for axon training")
    parser.add_argument(
        "--local_dir",
        default=os.path.join(AXON_PATH, "data", "formal_math"),
        help="Output directory for processed datasets",
    )
    parser.add_argument("--train_size", type=int, default=20000, help="Number of training examples")
    parser.add_argument("--test_size", type=int, default=100, help="Number of test examples")
    args = parser.parse_args()

    os.makedirs(args.local_dir, exist_ok=True)
    process_flc(args.local_dir, train_size=args.train_size, test_size=args.test_size)
    process_minif2f(args.local_dir)
    print(f"\nDone! Data saved to {args.local_dir}")
