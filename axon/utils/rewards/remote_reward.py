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
"""HTTP-based remote reward model client."""

from __future__ import annotations

import logging
import os

import requests

from axon.utils.rewards.base import RewardOutput

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 30


def remote_reward_fn(task_info: dict, action: str) -> RewardOutput:
    """Send the prompt/response to a remote reward model HTTP endpoint.

    The endpoint URL is read from ``task_info["rm_url"]`` or the
    ``AXON_REMOTE_RM_URL`` environment variable.

    The endpoint should accept a JSON POST with keys ``prompt``, ``response``,
    ``label`` and return a JSON object with at least a ``reward`` (float) field.

    Expected *task_info* keys:
      - ``rm_url`` (or env var ``AXON_REMOTE_RM_URL``): endpoint URL
      - ``question`` or ``prompt``: the input prompt
      - ``answer`` or ``ground_truth``: the ground-truth label (optional)
    """
    url = task_info.get("rm_url") or os.environ.get("AXON_REMOTE_RM_URL")
    if not url:
        logger.warning("No rm_url provided; returning zero reward.")
        return RewardOutput(reward=0.0, is_correct=False)

    payload = {
        "prompt": task_info.get("question") or task_info.get("prompt", ""),
        "response": action,
        "label": task_info.get("ground_truth") or task_info.get("answer", ""),
    }

    try:
        resp = requests.post(url, json=payload, timeout=_DEFAULT_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        logger.exception("Remote RM call failed for %s", url)
        return RewardOutput(reward=0.0, is_correct=False)

    reward = float(data) if isinstance(data, int | float) else float(data.get("reward", 0.0))
    is_correct = data.get("is_correct") if isinstance(data, dict) else None
    metadata = data if isinstance(data, dict) else {"raw": data}

    return RewardOutput(reward=reward, metadata=metadata, is_correct=is_correct)
