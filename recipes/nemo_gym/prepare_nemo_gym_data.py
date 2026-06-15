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
"""
Data Preparation for NeMo Gym Environments
===========================================

Converts NeMo Gym JSONL datasets into parquet files for Axon training.

NeMo Gym datasets are JSONL files where each line is::

    {"responses_create_params": {"input": [...], "tools": [...]}}

This script converts them into parquet with two columns::

    task              : dict   – the full NeMo Gym task (responses_create_params + metadata)
    resource_server   : str    – name tag for the resource server (for logging/provenance)

Usage::

    # From a local NeMo Gym JSONL file
    python prepare_nemo_gym_data.py \\
        --input data/workplace_assistant/train.jsonl \\
        --test-input data/workplace_assistant/validation.jsonl \\
        --resource-server workplace_assistant

    # From HuggingFace
    python prepare_nemo_gym_data.py \\
        --hf-repo nvidia/Nemotron-RL-agent-workplace_assistant \\
        --hf-train-split train \\
        --hf-test-split validation \\
        --resource-server workplace_assistant

    # Inspect dataset only
    python prepare_nemo_gym_data.py --input data/train.jsonl --info-only

    # Limit / shuffle
    python prepare_nemo_gym_data.py --input data/train.jsonl \\
        --max-examples 5000 --shuffle

    # Custom output directory
    python prepare_nemo_gym_data.py --input data/train.jsonl --local-dir ./my_data
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import axon

    AXON_DIR = os.path.dirname(os.path.dirname(os.path.abspath(axon.__file__)))
except ImportError:
    AXON_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_DATA_DIR = os.path.join(AXON_DIR, "data", "nemo_gym")


# ═══════════════════════════════════════════════════════════════════════════════
# Path helpers
# ═══════════════════════════════════════════════════════════════════════════════


def get_output_paths(
    name: str,
    local_dir: str | None = None,
    output: str | None = None,
    test_output: str | None = None,
) -> tuple[str, str]:
    """
    Resolve output file paths.

    Priority:
        1. Explicit ``--output`` / ``--test-output``
        2. ``--local-dir``  (custom base, auto-named files)
        3. Default: ``{AXON_DIR}/data/nemo_gym/{name}/``
    """
    dir_name = name.split("/")[-1].replace("-", "_")

    if output and test_output:
        return output, test_output

    base_dir = local_dir or os.path.join(DEFAULT_DATA_DIR, dir_name)
    train_path = output or os.path.join(base_dir, "train.parquet")
    test_path = test_output or os.path.join(base_dir, "test.parquet")
    return train_path, test_path


# ═══════════════════════════════════════════════════════════════════════════════
# Loading
# ═══════════════════════════════════════════════════════════════════════════════


def load_jsonl(path: str) -> list[dict]:
    """Load a NeMo Gym JSONL file. Each line must have ``responses_create_params``."""
    tasks: list[dict] = []
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Not found: {path}")

    with open(p) as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning("Line %d: bad JSON: %s", i, e)
                continue
            if "responses_create_params" not in row:
                logger.warning("Line %d: missing 'responses_create_params', skipping", i)
                continue
            tasks.append(row)

    return tasks


def load_from_huggingface(
    repo_id: str,
    split: str = "train",
    artifact_fpath: str | None = None,
) -> list[dict]:
    """Load a NeMo Gym dataset from HuggingFace."""
    if artifact_fpath:
        from huggingface_hub import hf_hub_download

        local_path = hf_hub_download(repo_id=repo_id, filename=artifact_fpath, repo_type="dataset")
        return load_jsonl(local_path)

    from datasets import load_dataset

    ds = load_dataset(repo_id, split=split)
    tasks = []
    for row in ds:
        d = dict(row)
        if "responses_create_params" in d:
            tasks.append(d)
        elif "input" in d:
            tasks.append({"responses_create_params": d})
        else:
            logger.warning("Skipping unrecognized row: %s", list(d.keys()))
    return tasks


# ═══════════════════════════════════════════════════════════════════════════════
# Prepare
# ═══════════════════════════════════════════════════════════════════════════════


def prepare_parquet(
    tasks: list[dict],
    output_path: str,
    resource_server: str = "",
    max_examples: int | None = None,
    shuffle: bool = False,
    seed: int = 42,
) -> int:
    """
    Write tasks to parquet. Returns number of rows written.

    Each row has:
        task             : dict  – the full NeMo Gym task
        resource_server  : str   – name tag for provenance
    """
    import pandas as pd

    indices = list(range(len(tasks)))
    if shuffle:
        random.seed(seed)
        random.shuffle(indices)
    if max_examples:
        indices = indices[:max_examples]

    records = [{"task": tasks[i], "resource_server": resource_server} for i in indices]

    if not records:
        raise ValueError("No valid tasks to write")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_parquet(output_path, index=False)
    return len(records)


# ═══════════════════════════════════════════════════════════════════════════════
# Info / inspection
# ═══════════════════════════════════════════════════════════════════════════════


def print_info(tasks: list[dict], source: str) -> None:
    """Print summary statistics for a NeMo Gym dataset."""
    if not tasks:
        print(f"\n{source}: empty dataset")
        return

    tool_counts: list[int] = []
    tool_names: set[str] = set()
    n_tool_calling = 0

    for task in tasks:
        params = task.get("responses_create_params", {})
        tools = params.get("tools", [])
        tool_counts.append(len(tools))
        if tools:
            n_tool_calling += 1
        for t in tools:
            name = t.get("function", t).get("name", "")
            if name:
                tool_names.add(name)

    print(f"\n{'=' * 60}")
    print(f"  {source}")
    print(f"{'=' * 60}")
    print(f"  Total tasks:       {len(tasks)}")
    print(f"  Tool-calling:      {n_tool_calling}")
    print(f"  Single-turn:       {len(tasks) - n_tool_calling}")
    if tool_counts:
        print(f"  Avg tools/task:    {sum(tool_counts) / len(tool_counts):.1f}")
        print(f"  Max tools/task:    {max(tool_counts)}")
    if tool_names:
        names = sorted(tool_names)
        preview = ", ".join(names[:10])
        if len(names) > 10:
            preview += f", ... ({len(names)} total)"
        print(f"  Tool names:        {preview}")

    # Sample task
    params = tasks[0].get("responses_create_params", {})
    raw_input = params.get("input", [])
    for item in reversed(raw_input) if isinstance(raw_input, list) else []:
        if item.get("role") in ("user",):
            prompt = item.get("content", "")
            print(f"\n  Sample prompt:     {prompt[:120]}{'...' if len(prompt) > 120 else ''}")
            break

    other_keys = [k for k in tasks[0] if k != "responses_create_params"]
    if other_keys:
        print(f"  Extra fields:      {', '.join(other_keys)}")

    print()


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    p = argparse.ArgumentParser(
        description="Prepare Axon training data from NeMo Gym environments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --input data/train.jsonl --info-only
  %(prog)s --input data/train.jsonl --resource-server workplace_assistant
  %(prog)s --input data/train.jsonl --test-input data/val.jsonl --resource-server math_with_judge
  %(prog)s --hf-repo nvidia/Nemotron-RL-agent-workplace_assistant --resource-server workplace_assistant
  %(prog)s --input data/train.jsonl --max-examples 1000 --shuffle
""",
    )

    # ── Input ──────────────────────────────────────────────────────────
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", help="Path to NeMo Gym JSONL file")
    src.add_argument("--hf-repo", metavar="REPO", help="HuggingFace dataset repo ID")

    p.add_argument("--hf-train-split", default="train", help="HF train split (default: train)")
    p.add_argument("--hf-test-split", default="validation", help="HF test split (default: validation)")
    p.add_argument("--hf-artifact", help="Specific JSONL file path within HF repo")

    p.add_argument("--test-input", help="Path to test/validation JSONL file")

    # ── Output ─────────────────────────────────────────────────────────
    p.add_argument("--resource-server", default="", help="Resource server name tag (also used for default output path)")
    p.add_argument("--local-dir", help=f"Base output directory (default: {DEFAULT_DATA_DIR}/<name>/)")
    p.add_argument("--output", help="Explicit train parquet output path")
    p.add_argument("--test-output", help="Explicit test parquet output path")

    # ── Options ────────────────────────────────────────────────────────
    p.add_argument("--max-examples", type=int, help="Limit number of examples")
    p.add_argument("--shuffle", action="store_true", help="Shuffle examples")
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    p.add_argument("--info-only", action="store_true", help="Print dataset info and exit")

    args = p.parse_args()

    # ── Resolve name ───────────────────────────────────────────────────
    name = args.resource_server
    if not name:
        if args.input:
            name = Path(args.input).stem
            if name in ("train", "test", "validation", "example"):
                name = Path(args.input).parent.name
        elif args.hf_repo:
            name = args.hf_repo.split("/")[-1]
        else:
            name = "nemo_gym"

    # ── Load ───────────────────────────────────────────────────────────
    print("Loading dataset...")

    if args.input:
        train_tasks = load_jsonl(args.input)
        source = args.input
    else:
        train_tasks = load_from_huggingface(args.hf_repo, args.hf_train_split, args.hf_artifact)
        source = f"{args.hf_repo} ({args.hf_train_split})"

    print(f"  Train: {len(train_tasks)} tasks from {source}")

    test_tasks: list[dict] = []
    if args.test_input:
        test_tasks = load_jsonl(args.test_input)
        print(f"  Test:  {len(test_tasks)} tasks from {args.test_input}")
    elif args.hf_repo:
        try:
            test_tasks = load_from_huggingface(args.hf_repo, args.hf_test_split, args.hf_artifact)
            print(f"  Test:  {len(test_tasks)} tasks from {args.hf_repo} ({args.hf_test_split})")
        except Exception:
            print(f"  Test:  no '{args.hf_test_split}' split found")

    # ── Info only ──────────────────────────────────────────────────────
    if args.info_only:
        print_info(train_tasks, source)
        if test_tasks:
            print_info(test_tasks, f"{source} (test)")
        return

    # ── Prepare ────────────────────────────────────────────────────────
    train_path, test_path = get_output_paths(
        name=name,
        local_dir=args.local_dir,
        output=args.output,
        test_output=args.test_output,
    )

    n_train = prepare_parquet(
        train_tasks,
        train_path,
        resource_server=name,
        max_examples=args.max_examples,
        shuffle=args.shuffle,
        seed=args.seed,
    )

    n_test = 0
    if test_tasks:
        n_test = prepare_parquet(
            test_tasks,
            test_path,
            resource_server=name,
            max_examples=args.max_examples,
            shuffle=args.shuffle,
            seed=args.seed,
        )

    # ── Summary ────────────────────────────────────────────────────────
    print("\nOutput:")
    print(f"  Train: {n_train} rows → {train_path}")
    if n_test:
        print(f"  Test:  {n_test} rows → {test_path}")


if __name__ == "__main__":
    main()
