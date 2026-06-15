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
"""F1-score based reward for open-domain QA tasks (e.g. HotpotQA, Natural Questions)."""

from __future__ import annotations

import re
import string
from collections import Counter

from axon.utils.rewards.base import RewardOutput


def _normalize_answer(s: str) -> str:
    """Lowercase, remove articles/punctuation, and collapse whitespace."""

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def remove_punc(text: str) -> str:
        return "".join(ch for ch in text if ch not in set(string.punctuation))

    return " ".join(remove_articles(remove_punc(s.lower())).split())


def compute_f1(prediction: str, ground_truth: str) -> tuple[float, float, float]:
    """Token-level F1 between *prediction* and *ground_truth*.

    Returns (f1, precision, recall).
    """
    if not prediction:
        return 0.0, 0.0, 0.0

    pred_norm = _normalize_answer(prediction)
    gold_norm = _normalize_answer(ground_truth)

    # Short-circuit yes/no/noanswer mismatches.
    if pred_norm in {"yes", "no", "noanswer"} and pred_norm != gold_norm:
        return 0.0, 0.0, 0.0
    if gold_norm in {"yes", "no", "noanswer"} and pred_norm != gold_norm:
        return 0.0, 0.0, 0.0

    pred_tokens = pred_norm.split()
    gold_tokens = gold_norm.split()
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0, 0.0, 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def f1_reward_fn(task_info: dict, action: str) -> RewardOutput:
    """Compute F1 reward against one or more ground-truth answers.

    Expected *task_info* keys:
      - ``answer`` or ``ground_truth``: str or list[str]
    """
    ground_truths = task_info.get("ground_truth") or task_info.get("answer")
    if ground_truths is None:
        return RewardOutput(reward=0.0, is_correct=False)

    if isinstance(ground_truths, str):
        ground_truths = [ground_truths]

    best_f1 = 0.0
    for gt in ground_truths:
        f1, _, _ = compute_f1(action, str(gt))
        best_f1 = max(best_f1, f1)

    return RewardOutput(
        reward=best_f1,
        metadata={"f1": best_f1},
        is_correct=best_f1 >= 0.5,
    )
