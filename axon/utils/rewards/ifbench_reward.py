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
"""Reward for IFBench (instruction-following benchmark) using the official evaluation library."""

from __future__ import annotations

import importlib
import logging
import os
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from axon.utils.rewards.base import RewardOutput

logger = logging.getLogger(__name__)

_AXON_ROOT = Path(__file__).resolve().parents[3]

JsonDict = dict[str, Any]
KwargsDict = dict[str, str | int | float | None]

# Lazily loaded evaluation library.
_evaluation_lib = None
_InputExample = None


def _ensure_ifbench_available():
    """Clone IFBench repo if needed and import evaluation_lib."""
    global _evaluation_lib, _InputExample

    if _evaluation_lib is not None:
        return

    repo_path = _AXON_ROOT.parent / "IFBench"

    if not repo_path.exists():
        try:
            subprocess.run(
                ["git", "clone", "https://github.com/allenai/IFBench.git", str(repo_path)],
                check=True,
                capture_output=True,
            )
        except Exception as exc:
            raise ImportError(
                "Unable to clone IFBench. Please clone "
                "https://github.com/allenai/IFBench.git next to the axon repo root."
            ) from exc

    repo_str = str(repo_path)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)

    current_pp = os.environ.get("PYTHONPATH", "")
    if repo_str not in current_pp.split(os.pathsep):
        os.environ["PYTHONPATH"] = os.pathsep.join([repo_str, current_pp]) if current_pp else repo_str

    _evaluation_lib = importlib.import_module("evaluation_lib")
    _InputExample = _evaluation_lib.InputExample


def _normalize_instruction_ids(raw_ids: Sequence[Any]) -> list[str]:
    return [str(e).strip() for e in (raw_ids or []) if e is not None and str(e).strip()]


def _coerce_kwargs_list(raw_kwargs: Any, n: int) -> list[KwargsDict]:
    if isinstance(raw_kwargs, list):
        processed = [dict(e) if isinstance(e, dict) else {} for e in raw_kwargs]
    elif isinstance(raw_kwargs, dict):
        processed = [dict(raw_kwargs) for _ in range(n)]
    else:
        processed = [{} for _ in range(n)]

    # Pad or trim.
    if len(processed) < n:
        tail = processed[-1] if processed else {}
        processed.extend([dict(tail) for _ in range(n - len(processed))])
    processed = processed[:n]

    return [{k: v for k, v in entry.items() if v is not None} for entry in processed]


def _build_input_example(metadata: JsonDict):
    instruction_ids = _normalize_instruction_ids(metadata.get("instruction_id_list", []))
    if not instruction_ids:
        return None

    prompt_text = str(metadata.get("prompt_text", "") or "")
    kwargs_list = _coerce_kwargs_list(metadata.get("kwargs"), len(instruction_ids))

    return _InputExample(
        key=int(metadata.get("record_id", 0) or 0),
        instruction_id_list=instruction_ids,
        prompt=prompt_text,
        kwargs=kwargs_list,
    )


def ifbench_reward_fn(task_info: dict, action: str) -> RewardOutput:
    """Score a response using official IFBench strict evaluation.

    Expected *task_info* keys:
      - ``instruction_id_list``: list of instruction identifiers
      - ``prompt_text``: the original prompt
      - (optional) ``kwargs``: per-instruction kwargs
      - (optional) ``record_id``: integer record ID
    """
    if action is None:
        return RewardOutput(reward=0.0, is_correct=False)

    _ensure_ifbench_available()

    inp = _build_input_example(task_info)
    if inp is None:
        return RewardOutput(reward=0.0, is_correct=False)

    prompt_to_response = {inp.prompt: str(action)}
    output = _evaluation_lib.test_instruction_following_strict(inp, prompt_to_response)
    score = 1.0 if output.follow_all_instructions else 0.0
    return RewardOutput(reward=score, is_correct=score == 1.0)
