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
"""Miscellaneous utility functions for printing, dict manipulation, and data processing."""

import warnings
from collections import defaultdict

import click

# ---------------------------------------------------------------------------
# Printing & logging
# ---------------------------------------------------------------------------


def colorful_print(string: str, *args, **kwargs) -> None:
    """Print *string* styled with ``click.style``, flushing immediately."""
    end = kwargs.pop("end", "\n")
    print(click.style(string, *args, **kwargs), end=end, flush=True)


def colorful_warning(string: str, *args, **kwargs) -> None:
    """Emit a styled warning via ``warnings.warn``."""
    warnings.warn(click.style(string, *args, **kwargs), stacklevel=2)


def log_metrics(metrics: dict, title: str = "Metrics", color: str = "white") -> None:
    """Print *metrics* as a formatted, sorted table with a banner *title*.

    Args:
        metrics: Metric names mapped to values.
        title: Banner text centred above the table.
        color: Click colour name applied to every line.
    """
    if not metrics:
        return

    sorted_items = sorted(metrics.items())
    width = max(len(title) + 4, max(len(k) for k in metrics) + 20)
    sep = "=" * width

    colorful_print(sep, color)
    colorful_print(f"{title:^{width}}", color)
    colorful_print(sep, color)
    for key, value in sorted_items:
        colorful_print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}", color)
    colorful_print(sep, color)


# ---------------------------------------------------------------------------
# Dict helpers
# ---------------------------------------------------------------------------


def merge_dicts(dict_list: list[dict]) -> dict[str, list]:
    """Merge a list of dicts, collecting values into lists keyed by their original keys."""
    merged: dict[str, list] = defaultdict(list)
    for d in dict_list:
        for key, value in d.items():
            merged[key].append(value)
    return dict(merged)


def append_to_dict(data: dict, new_data: dict, prefix: str = "") -> None:
    """Append values from *new_data* into list-valued *data* in-place.

    Args:
        data: Target dict whose values are lists.
        new_data: Source dict with values to append.
        prefix: If given, prepended to keys that don't already start with it.
    """
    for key, val in new_data.items():
        new_key = f"{prefix}{key}" if prefix and not key.startswith(prefix) else key
        data.setdefault(new_key, []).extend(val if isinstance(val, list) else [val])
