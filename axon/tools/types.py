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
Unified data types for tool handling across Axon.
"""

from __future__ import annotations

import copy
import json
import uuid
from dataclasses import dataclass, field
from typing import Any

# =============================================================================
# ToolCall — what to invoke
# =============================================================================


@dataclass
class ToolCall:
    """
    A single parsed tool call extracted from model output.

    ``name`` and ``arguments`` are always present.  ``id`` is auto-generated
    but can be overridden (e.g. when reconstructing from OpenAI format).

    Conversion helpers
    ------------------
    * ``to_openai_dict()``  → OpenAI chat-completion tool_call format
    * ``to_dict()``         → minimal ``{"name": ..., "arguments": ...}``
    * ``from_raw_tool_call()``        → normalize any SDK/dict/object shape into ToolCall
    """

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: f"call_{uuid.uuid4().hex[:12]}")

    # Original raw object before normalization (for round-trip fidelity).
    raw_tool_call: Any = field(default=None, repr=False, compare=False)

    # -- Serialisation helpers ------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Minimal dict (for prompts, logging, etc.)."""
        return {"name": self.name, "arguments": self.arguments}

    def to_openai_dict(self) -> dict[str, Any]:
        """
        OpenAI chat-completion ``tool_calls[]`` element.

        Use case 3 — building the ``/v1/chat/completions`` response and
        converting to message dicts for multi-turn conversations.
        """
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": (json.dumps(self.arguments) if isinstance(self.arguments, dict) else str(self.arguments)),
            },
        }

    @classmethod
    def from_raw_tool_call(cls, d: Any) -> ToolCall:
        """
        Accepts the following shapes and convert to internal ToolCall format

        * ToolCall:         returned as-is
        * OpenAI dict:      ``{"id": ..., "function": {"name": ..., "arguments": ...}}``
        * Direct dict:      ``{"name": ..., "arguments": ...}``
        * Pydantic / obj:   ``obj.function.name``, ``obj.function.arguments``
        * Direct obj:       ``obj.name``, ``obj.arguments``
        """
        raw_tool_call_copy = copy.deepcopy(d)
        if isinstance(d, ToolCall):
            return d

        # pydantic/SDK object with .function attribute
        if hasattr(d, "function"):
            d = {
                "function": {
                    "name": getattr(d.function, "name", ""),
                    "arguments": getattr(d.function, "arguments", {}),
                },
                "id": getattr(d, "id", None),
            }

        if not isinstance(d, dict):
            raise TypeError(f"Expected dict or tool call object, got {type(d)}")

        # OpenAI {"id": ..., "function": {"name": ..., "arguments": ...}}
        if "function" in d:
            func = d["function"]
            if hasattr(func, "name"):  # nested pydantic
                name = func.name
                arguments = func.arguments
            else:
                name = func.get("name", "")
                arguments = func.get("arguments", {})
            call_id = d.get("id")
        # flat {"name": ..., "arguments": ...}
        else:
            if "name" not in d:
                raise ValueError(f"Tool call dict must contain 'name' or 'function' key, got keys: {list(d.keys())}")
            name = d["name"]
            arguments = d.get("arguments", d.get("parameters", {}))
            call_id = d.get("id")

        # Normalize arguments to dict
        if arguments is None:
            arguments = {}
        elif isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {"raw": arguments}
        elif not isinstance(arguments, dict):
            arguments = {"raw": arguments}

        kwargs = {"name": name, "arguments": arguments}
        if call_id:
            kwargs["id"] = call_id
        kwargs["raw_tool_call"] = raw_tool_call_copy
        return cls(**kwargs)


# =============================================================================
# ToolResult — execution outcome (ToolCall → execute → ToolResult)
# =============================================================================


@dataclass
class ToolResult:
    """
    Result from executing a tool call.
    """

    content: str
    name: str = ""
    tool_call_id: str = ""

    def to_openai_dict(self) -> dict[str, Any]:
        """OpenAI-compatible tool-result message."""
        return {
            "role": "tool",
            "tool_call_id": self.tool_call_id,
            "content": self.content,
        }


# =============================================================================
# ToolOutput — raw return from Tool.forward()
# =============================================================================


@dataclass
class ToolOutput:
    """
    Raw output from ``Tool.forward()`` / ``Tool.async_forward()``.

    This is the *execution-level* result before it gets wrapped into
    a ``ToolResult`` for conversation building.
    """

    name: str
    output: str | list | dict | None = None
    error: str | None = None
    metadata: dict | None = None

    def to_content_string(self) -> str:
        """Convert to a plain string suitable for ``ToolResult.content``."""
        if self.error is not None:
            return f"Error: {self.error}"
        if self.output is None:
            return ""
        if isinstance(self.output, (list | dict)):
            return json.dumps(self.output)
        return str(self.output)

    def __str__(self) -> str:
        return self.to_content_string()
