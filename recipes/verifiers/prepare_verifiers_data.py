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
Data Preparation for Verifiers Environments

Prepares training data from Verifiers environments for use with Axon's VerifiersProgram.

Installation:
    pip install verifiers
    uv tool install prime
    prime login
    prime env install will/wordle  # For Hub environments

Usage:
    # Basic - reference mode (smaller files, requires env at training time)
    python prepare_verifiers_data.py --env-module wordle

    # Embedded mode (self-contained, larger files)
    python prepare_verifiers_data.py --env-module wordle --embed-task

    # View environment info only
    python prepare_verifiers_data.py --env-module wordle --info-only

Output Format:
    Parquet file with columns:
    - idx: int - sequential index
    - env_args: dict - arguments for VerifiersProgram
    - prompt: str - task prompt (for inspection)
    - answer: str - expected answer (for inspection)

    The env_args dict is passed directly to VerifiersProgram and contains:
    - env_module: str - module name
    - env_kwargs: dict - arguments for load_environment()
    - task_idx: int (reference mode) OR task: dict (embedded mode)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections.abc import Iterator
from pathlib import Path

import axon

# Get the directory for Axon repo (axon.__file__)
AXON_DIR = os.path.dirname(os.path.dirname(os.path.abspath(axon.__file__)))
DEFAULT_DATA_DIR = os.path.join(AXON_DIR, "data", "verifiers")


def get_output_paths(
    env_module: str,
    local_dir: str | None = None,
    output: str | None = None,
    test_output: str | None = None,
) -> tuple[str, str]:
    """
    Resolve output paths with sensible defaults.

    Priority:
    1. Explicit --output/--test-output (full control)
    2. --local-dir (custom base, auto-named files)
    3. Default: {AXON_DIR}/data/verifiers/{env_module}/
    """
    # Normalize env_module for directory name
    # "will/wordle" → "wordle", "math-python" → "math_python"
    dir_name = env_module.split("/")[-1].replace("-", "_")

    # Explicit output takes precedence
    if output and test_output:
        train_path = output
        test_path = test_output  # May be None
    else:
        # Use local_dir or default
        base_dir = local_dir or os.path.join(DEFAULT_DATA_DIR, dir_name)
        train_path = os.path.join(base_dir, "train.parquet")
        test_path = os.path.join(base_dir, "test.parquet")

    return train_path, test_path


def load_verifiers_env(env_module: str, env_kwargs: dict | None = None):
    """Load using verifiers' official API."""
    try:
        import verifiers as vf
    except ImportError:
        print("Error: verifiers not installed. Run: pip install verifiers")
        sys.exit(1)

    return vf.load_environment(env_module, **(env_kwargs or {}))


def get_env_info(env) -> dict:
    """Get information about a Verifiers environment."""
    # Detect environment type
    has_tools = bool(getattr(env, "tools", []))
    has_env_response = hasattr(env, "env_response")
    max_turns = getattr(env, "max_turns", 1)

    if has_tools:
        env_type = "ToolEnv"
    elif has_env_response and max_turns > 1:
        env_type = "MultiTurnEnv"
    else:
        env_type = "SingleTurnEnv"

    return {
        "env_type": env_type,
        "dataset_size": len(env.dataset) if env.dataset else 0,
        "max_turns": max_turns,
        "has_system_prompt": bool(getattr(env, "system_prompt", None)),
        "has_rubric": bool(getattr(env, "rubric", None)),
        "num_tools": len(getattr(env, "tools", [])),
        "tool_names": [t.get("name", "") for t in getattr(env, "tools", [])],
    }


def iter_tasks(
    env_module: str,
    env_kwargs: dict | None = None,
    embed_task: bool = False,
    max_examples: int | None = None,
    shuffle: bool = False,
    seed: int = 42,
    use_eval_dataset: bool = False,
) -> Iterator[dict]:
    """
    Iterate over tasks from a Verifiers environment.

    Args:
        env_module: Module name or Hub path
        env_kwargs: Arguments for load_environment()
        embed_task: If True, embed full task data. If False, use task_idx reference.
        max_examples: Maximum tasks to yield (None = all)
        shuffle: Whether to shuffle task order
        seed: Random seed for shuffling

    Yields:
        Dict with idx, env_args, prompt, answer
    """
    env_kwargs = env_kwargs or {}
    env = load_verifiers_env(env_module, env_kwargs)

    dataset = env.dataset if not use_eval_dataset else env.eval_dataset
    if not dataset:
        raise ValueError(f"Environment '{env_module}' has no dataset for {'eval' if use_eval_dataset else 'train'}")
    indices = list(range(len(dataset)))

    if shuffle:
        random.seed(seed)
        random.shuffle(indices)

    if max_examples:
        indices = indices[:max_examples]

    for output_idx, dataset_idx in enumerate(indices):
        row = dict(dataset[dataset_idx])

        # Extract prompt and answer for inspection
        prompt = row.get("prompt") or row.get("question") or ""
        answer = row.get("answer", "")

        # Build env_args for VerifiersProgram
        if embed_task:
            # Self-contained: embed full task
            task_data = {"prompt": prompt, "answer": answer}
            # Include other fields
            for k, v in row.items():
                if k not in ("prompt", "question", "answer"):
                    task_data[k] = v

            env_args = {
                "env_module": env_module,
                "env_kwargs": env_kwargs,
                "task": task_data,
                "eval": use_eval_dataset,
            }
        else:
            # Reference mode: use task_idx
            env_args = {
                "env_module": env_module,
                "env_kwargs": env_kwargs,
                "task_idx": dataset_idx,
                "eval": use_eval_dataset,
            }

        if not env_args["env_kwargs"]:
            env_args.pop("env_kwargs")
        yield env_args


def prepare_data(
    env_module: str,
    output_path: str,
    test_output_path: str,
    env_kwargs: dict | None = None,
    embed_task: bool = False,
    max_examples: int | None = None,
    shuffle: bool = False,
    seed: int = 42,
) -> dict:
    """
    Prepare Axon training data from a Verifiers environment.

    Returns:
        Dict with statistics
    """
    import pandas as pd

    # Get env info
    env = load_verifiers_env(env_module, env_kwargs)
    env_info = get_env_info(env)

    # Collect records
    train_records = list(
        iter_tasks(
            env_module=env_module,
            env_kwargs=env_kwargs,
            embed_task=embed_task,
            max_examples=max_examples,
            shuffle=shuffle,
            seed=seed,
            use_eval_dataset=False,
        )
    )

    test_records = list(
        iter_tasks(
            env_module=env_module,
            env_kwargs=env_kwargs,
            embed_task=embed_task,
            max_examples=max_examples,
            shuffle=shuffle,
            seed=seed,
            use_eval_dataset=True,
        )
    )

    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(train_records).to_parquet(output_path, index=False)

    Path(test_output_path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(test_records).to_parquet(test_output_path, index=False)

    return {
        "env_module": env_module,
        "env_info": env_info,
        "mode": "embedded" if embed_task else "reference",
        "train_examples": len(train_records),
        "test_examples": len(test_records),
        "train_path": output_path,
        "test_path": test_output_path,
    }


def print_env_info(env_module: str, env_kwargs: dict | None = None):
    """Print detailed information about an environment."""
    print(f"\n{'=' * 60}")
    print(f"Environment: {env_module}")
    print(f"{'=' * 60}")

    env = load_verifiers_env(env_module, env_kwargs)
    info = get_env_info(env)

    print(f"\nType: {info['env_type']}")
    print(f"Dataset size: {info['dataset_size']}")
    print(f"Max turns: {info['max_turns']}")
    print(f"Has rubric: {info['has_rubric']}")

    if info["num_tools"] > 0:
        print(f"Tools ({info['num_tools']}): {', '.join(info['tool_names'])}")

    # System prompt
    system_prompt = getattr(env, "system_prompt", "")
    if system_prompt:
        print("\nSystem prompt:")
        print(f"  {system_prompt[:300]}{'...' if len(system_prompt) > 300 else ''}")

    # Sample task
    if env.dataset:
        print("\nSample task (index 0):")
        row = dict(env.dataset[0])
        prompt = row.get("prompt") or row.get("question", "")
        answer = row.get("answer", "")
        print(f"  Prompt: {str(prompt)[:200]}{'...' if len(str(prompt)) > 200 else ''}")
        print(f"  Answer: {str(answer)[:100]}{'...' if len(str(answer)) > 100 else ''}")

        # Other fields
        other_keys = [k for k in row.keys() if k not in ("prompt", "question", "answer")]
        if other_keys:
            print(f"  Other fields: {', '.join(other_keys)}")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Prepare Axon training data from Verifiers environments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # View environment info
  %(prog)s --env-module wordle --info-only
  
  # Basic preparation (reference mode)
  %(prog)s --env-module wordle
  
  # Embedded mode (self-contained)
  %(prog)s --env-module wordle --embed-task
  
  # Limit examples
  %(prog)s --env-module wordle --max-examples 1000 --shuffle
""",
    )

    parser.add_argument("--env-module", required=True, help="Module name or Hub path")
    parser.add_argument("--env-kwargs", type=json.loads, default={}, help="JSON kwargs for load_environment()")
    parser.add_argument(
        "--local-dir", default=None, help=f"Base directory for output (default: {DEFAULT_DATA_DIR}/<env_module>/)"
    )
    parser.add_argument("--output", default=None, help="Explicit train parquet path (overrides --local-dir)")
    parser.add_argument("--test-output", default=None, help="Explicit test parquet path")
    parser.add_argument("--embed-task", action="store_true", help="Embed full task data (default: use task_idx)")
    parser.add_argument("--max-examples", type=int, help="Max examples")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle examples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--info-only", action="store_true", help="Only print env info")

    args = parser.parse_args()

    if args.info_only:
        print_env_info(args.env_module, args.env_kwargs)
        return

    # Resolve paths
    train_path, test_path = get_output_paths(
        env_module=args.env_module,
        local_dir=args.local_dir,
        output=args.output,
        test_output=args.test_output,
    )

    print(f"Preparing data from: {args.env_module}")
    print(f"Mode: {'embedded' if args.embed_task else 'reference'}")

    stats = prepare_data(
        env_module=args.env_module,
        output_path=train_path,
        env_kwargs=args.env_kwargs,
        embed_task=args.embed_task,
        max_examples=args.max_examples,
        shuffle=args.shuffle,
        seed=args.seed,
        test_output_path=test_path,
    )

    print(f"\nEnvironment: {stats['env_info']['env_type']}")
    print(f"Dataset size: {stats['env_info']['dataset_size']}")
    print("\nOutput:")
    print(f"  Train: {stats['train_examples']} examples → {stats['train_path']}")
    if stats["test_examples"] > 0:
        print(f"  Test: {stats['test_examples']} examples → {stats['test_path']}")

    print("\nDone!")


if __name__ == "__main__":
    main()
