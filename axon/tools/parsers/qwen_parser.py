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
Hermes-style tool-call parser.

Format::

    <tool_call>
    {"name": "function_name", "arguments": {"key": "value"}}
    </tool_call>
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from axon.tools.parsers.base_parser import ToolCallParser, register_parser
from axon.tools.types import ToolCall

logger = logging.getLogger(__name__)


def _try_fix_json(json_str: str) -> str:
    """Try to fix common JSON formatting errors from model output."""
    s = json_str.strip()

    # Remove trailing punctuation that shouldn't be there
    while s and s[-1] in ".;,)":
        if s[-1] == ")":
            s = s[:-1] + "}"
        else:
            s = s[:-1]

    # Fix unbalanced braces
    open_braces = s.count("{") - s.count("}")
    if open_braces > 0:
        s += "}" * open_braces

    return s


@register_parser("qwen")
class QwenToolCallParser(ToolCallParser):
    """
    ``<tool_call>{"name": ..., "arguments": ...}</tool_call>``

    Handles clean JSON, mixed reasoning+JSON, common JSON errors.
    """

    _CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
    _JSON_OBJ_RE = re.compile(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", re.DOTALL)

    def parse(self, response: str) -> tuple[list[ToolCall], str]:
        calls: list[ToolCall] = []
        text = response

        for m in self._CALL_RE.finditer(response):
            raw = m.group(1).strip()
            if not raw:
                text = text.replace(m.group(0), "")
                continue

            d = self._extract_json(raw)
            if d is not None and isinstance(d, dict) and "name" in d:
                calls.append(
                    ToolCall(
                        name=d["name"],
                        arguments=d.get("arguments", {}),
                    )
                )
            else:
                logger.warning(
                    "QwenParser: could not extract valid tool call from: %s",
                    raw[:200],
                )
            text = text.replace(m.group(0), "")

        return calls, text.strip()

    def format_tool_call(self, tool_call: ToolCall) -> str:
        """
        Render tool calls as ``<tool_call>`` blocks.

        Produces::

            <tool_call>
            {"name": "func", "arguments": {"key": "value"}}
            </tool_call>
        """
        args_str = json.dumps(tool_call.arguments, ensure_ascii=False)
        return f'<tool_call>\n{{"name": "{tool_call.name}", "arguments": {args_str}}}\n</tool_call>'

    def format_tool_result(self, content: Any, name: str = "") -> str:
        """
        Render results as ``<tool_response>`` blocks.
        """
        return f"<tool_response>\n{content}\n</tool_response>"

    def get_tool_system_prompt(self, tools_json: list[dict[str, Any]]) -> str:
        tools_str = "\n".join(json.dumps(t, ensure_ascii=False) for t in tools_json)
        return (
            "# Tools\n\n"
            "You may call one or more functions to assist with the user query.\n\n"
            "You are provided with function signatures within <tools></tools> XML tags:\n"
            f"<tools>\n{tools_str}\n</tools>\n\n"
            "For each function call, return a json object with function "
            "name and arguments within <tool_call></tool_call> XML tags:\n"
            "<tool_call>\n"
            '{"name": <function-name>, "arguments": <args-json-object>}\n'
            "</tool_call>"
        )

    # -- Internal helpers -----------------------------------------------------

    def _extract_json(self, raw: str) -> dict | None:
        # Strategy 1: direct parse
        try:
            d = json.loads(raw)
            if isinstance(d, dict):
                return d
        except json.JSONDecodeError:
            pass

        # Strategy 2: fix common errors
        try:
            fixed = _try_fix_json(raw)
            d = json.loads(fixed)
            if isinstance(d, dict):
                return d
        except json.JSONDecodeError:
            pass

        # Strategy 3: extract first JSON object via regex
        obj_match = self._JSON_OBJ_RE.search(raw)
        if obj_match:
            try:
                d = json.loads(obj_match.group(0))
                if isinstance(d, dict):
                    return d
            except json.JSONDecodeError:
                pass
            try:
                fixed = _try_fix_json(obj_match.group(0))
                d = json.loads(fixed)
                if isinstance(d, dict):
                    return d
            except json.JSONDecodeError:
                pass

        return None
