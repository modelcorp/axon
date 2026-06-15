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
Fenced-JSON tool-call parser.

Format::

    ```json
    [{"name": "function_name", "arguments": {"key": "value"}}]
    ```

Or a single object::

    ```json
    {"name": "function_name", "arguments": {"key": "value"}}
    ```
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from axon.tools.parsers.base_parser import ToolCallParser, register_parser
from axon.tools.types import ToolCall, ToolResult

logger = logging.getLogger(__name__)


@register_parser("json")
class JsonToolCallParser(ToolCallParser):
    """Fenced JSON blocks: ````json [{"name":...,"arguments":...}]```."""

    _BLOCK_RE = re.compile(
        r"```(?:json)?\s*(\[.*?\]|\{.*?\})\s*```",
        re.DOTALL,
    )

    def parse(self, response: str) -> tuple[list[ToolCall], str]:
        calls: list[ToolCall] = []
        text = response

        for m in self._BLOCK_RE.finditer(response):
            try:
                items = json.loads(m.group(1))
                if isinstance(items, dict):
                    items = [items]
                for d in items:
                    if isinstance(d, dict) and "name" in d:
                        calls.append(
                            ToolCall(
                                name=d["name"],
                                arguments=d.get("arguments", {}),
                            )
                        )
                text = text.replace(m.group(0), "")
            except (json.JSONDecodeError, KeyError):
                continue

        return calls, text.strip()

    def format_tool_call(self, tool_call: ToolCall) -> str:
        """Single call as fenced JSON object."""
        obj = {"name": tool_call.name, "arguments": tool_call.arguments}
        return f"```json\n{json.dumps(obj, indent=2, ensure_ascii=False)}\n```"

    def format_tool_calls(self, tool_calls: list[ToolCall]) -> str:
        """Override: single fenced array, not N separate fenced blocks."""
        items = [{"name": tc.name, "arguments": tc.arguments} for tc in tool_calls]
        return f"```json\n{json.dumps(items, indent=2, ensure_ascii=False)}\n```"

    def format_tool_result(self, content: Any, name: str = "") -> str:
        """Single result as fenced JSON."""
        obj = {"name": name, "content": content} if name else {"content": content}
        return f"```json\n{json.dumps(obj, indent=2)}\n```"

    def format_tool_results(self, results: list[ToolResult]) -> str:
        """Override: single fenced array."""
        items = [{"name": r.name, "content": r.content} for r in results]
        return f"```json\n{json.dumps(items, indent=2)}\n```"

    def get_tool_system_prompt(self, tools_json: list[dict[str, Any]]) -> str:
        tools_str = json.dumps(tools_json, indent=2, ensure_ascii=False)
        return (
            "You have access to the following tools:\n"
            f"```json\n{tools_str}\n```\n\n"
            "To call a tool, respond with a fenced JSON code block containing "
            "an array of tool call objects:\n"
            "```json\n"
            '[{"name": "tool_name", "arguments": {"key": "value"}}]\n'
            "```"
        )
