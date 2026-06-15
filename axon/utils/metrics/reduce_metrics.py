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
Metrics reduction utilities.
"""

import re
from typing import Any

import numpy as np

# Match "max" or "min" as a whole word or at a word boundary (e.g. "max_reward",
# "maximum", "minimize") but NOT as an embedded substring of an unrelated word
# (e.g. "admin", "terminal", "diminish").
_MAX_RE = re.compile(r"(?:^|_)max")
_MIN_RE = re.compile(r"(?:^|_)min")


def reduce_metrics(metrics: dict[str, list[Any]]) -> dict[str, Any]:
    """
    Reduces a dictionary of metric lists by computing the mean, max, or min of each list.
    The reduce operation is determined by the key name:
    - If the key contains a "max" token (word-boundary aware), np.max is used
    - If the key contains a "min" token (word-boundary aware), np.min is used
    - Otherwise, np.mean is used

    Args:
        metrics: A dictionary mapping metric names to lists of metric values.

    Returns:
        A dictionary with the same keys but with each list replaced by its reduced value.

    Example:
        >>> metrics = {
        ...     "loss": [1.0, 2.0, 3.0],
        ...     "accuracy": [0.8, 0.9, 0.7],
        ...     "max_reward": [5.0, 8.0, 6.0],
        ...     "min_error": [0.1, 0.05, 0.2]
        ... }
        >>> reduce_metrics(metrics)
        {"loss": 2.0, "accuracy": 0.8, "max_reward": 8.0, "min_error": 0.05}
    """
    result = {}
    for key, val in metrics.items():
        # Flatten in case values are inhomogeneous (e.g. arrays of different lengths)
        flat = np.concatenate([np.atleast_1d(v) for v in val])
        if _MAX_RE.search(key):
            result[key] = np.max(flat)
        elif _MIN_RE.search(key):
            result[key] = np.min(flat)
        else:
            result[key] = np.mean(flat)
    return result
