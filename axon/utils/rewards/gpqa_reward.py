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
"""Rule-based reward for GPQA-style multiple-choice evaluation."""

from __future__ import annotations

import re
import string
from collections.abc import Iterable

from axon.utils.rewards.base import RewardOutput

_DEFAULT_VALID_LETTERS = list(string.ascii_uppercase[:8])


def _strip_chain_of_thought(text: str) -> str:
    if not text:
        return ""
    if "</think>" in text:
        return text.rsplit("</think>", 1)[-1]
    return text


def _normalize_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _extract_letter(response: str, valid_letters: Iterable[str]) -> str | None:
    """Best-effort extraction of the selected option letter from a model response."""
    if not response:
        return None

    text = _strip_chain_of_thought(response)
    patterns = [
        r"(?:answer|option|choice)\s*(?:is|:)?\s*([A-Z])",
        r"([A-Z])\s*(?:is\s*(?:the)?\s*correct)",
        r"final\s*(?:answer|option)\s*(?:is|:)?\s*([A-Z])",
    ]

    valid_set = {letter.upper() for letter in valid_letters}
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            letter = match.group(1).upper()
            if letter in valid_set:
                return letter

    # Fallback: last standalone capital letter that is valid.
    for letter in reversed(re.findall(r"\b([A-Z])\b", text)):
        if letter.upper() in valid_set:
            return letter.upper()

    return None


def _compute_gpqa(response: str, label: str | int | float | None, metadata: dict) -> float:
    """Core scoring logic for GPQA multiple-choice."""
    if response is None:
        return 0.0

    choices = metadata.get("choices")
    if isinstance(choices, dict):
        choices = list(choices.values())
    elif choices is not None:
        choices = list(choices)

    valid_letters: list[str] = metadata.get("valid_letters") or (
        list(string.ascii_uppercase[: len(choices)]) if choices else _DEFAULT_VALID_LETTERS
    )
    valid_letters = [str(v).upper() for v in valid_letters]

    # Determine the correct letter.
    correct_letter: str | None = metadata.get("correct_letter")
    if isinstance(correct_letter, str):
        correct_letter = correct_letter.strip().upper()
    else:
        correct_letter = None

    label_text: str | None = None
    if isinstance(label, str):
        label_text = label.strip()
        if len(label_text) == 1 and label_text.upper() in valid_letters and not correct_letter:
            correct_letter = label_text.upper()
    elif isinstance(label, int | float):
        idx = int(label)
        if 0 <= idx < len(valid_letters):
            correct_letter = valid_letters[idx]

    if not correct_letter and choices and label_text:
        normalized_label = _normalize_text(label_text)
        for idx, choice in enumerate(choices):
            if _normalize_text(str(choice)) == normalized_label:
                correct_letter = valid_letters[idx]
                break

    extracted = _extract_letter(response, valid_letters)
    if extracted and correct_letter:
        return 1.0 if extracted == correct_letter else 0.0

    # Fallback: substring match against candidate answers.
    candidates: list[str] = []
    if correct_letter and choices:
        try:
            idx = valid_letters.index(correct_letter)
            if idx < len(choices):
                candidates.append(str(choices[idx]))
        except ValueError:
            pass
    for key in ("correct_answer", "answer_text"):
        if metadata.get(key):
            candidates.append(str(metadata[key]))
    if label_text:
        candidates.append(label_text)

    norm_response = _normalize_text(_strip_chain_of_thought(response))
    for c in candidates:
        if _normalize_text(c) and _normalize_text(c) in norm_response:
            return 1.0

    if extracted and not correct_letter and label_text:
        return 1.0 if extracted == label_text.strip().upper() else 0.0

    return 0.0


def gpqa_reward_fn(task_info: dict, action: str) -> RewardOutput:
    """GPQA multiple-choice reward.

    Expected *task_info* keys:
      - ``answer`` or ``ground_truth``: correct letter, index, or text
      - (optional) ``choices``: list or dict of answer options
      - (optional) ``correct_letter``, ``valid_letters``, ``correct_answer``
    """
    label = task_info.get("ground_truth") or task_info.get("answer")
    score = _compute_gpqa(action, label, metadata=task_info)
    return RewardOutput(reward=score, is_correct=score == 1.0)
