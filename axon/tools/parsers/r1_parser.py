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
DeepSeek R1 tool-call parser.

Format (uses fullwidth Unicode special tokens)::

    <｜tool▁calls▁begin｜>
    <｜tool▁call▁begin｜>function<｜tool▁sep｜>function_name
    ```json
    {"param": "value"}
    ```
    <｜tool▁call▁end｜>
    <｜tool▁calls▁end｜>
"""

from __future__ import annotations

import json
import logging
from typing import Any

from axon.tools.parsers.base_parser import ToolCallParser, register_parser
from axon.tools.types import ToolCall

logger = logging.getLogger(__name__)


@register_parser("r1")
class R1ToolCallParser(ToolCallParser):
    """Parser for DeepSeek R1 special-token tool call format."""

    def __init__(self):
        self.tool_calls_begin = "<｜tool▁calls▁begin｜>"
        self.tool_calls_end = "<｜tool▁calls▁end｜>"
        self.tool_call_begin = "<｜tool▁call▁begin｜>"
        self.tool_call_end = "<｜tool▁call▁end｜>"
        self.tool_sep = "<｜tool▁sep｜>"

    def parse(self, response: str) -> tuple[list[ToolCall], str]:
        calls: list[ToolCall] = []
        text = response
        call_idx = 0

        while True:
            call_idx = text.find(self.tool_call_begin, call_idx)
            if call_idx == -1:
                break

            call_start = call_idx + len(self.tool_call_begin)
            call_end = text.find(self.tool_call_end, call_start)
            if call_end == -1:
                break

            call_content = text[call_start:call_end].strip()

            # Parse function name
            func_prefix = "function" + self.tool_sep
            func_start = call_content.find(func_prefix)

            if func_start == -1:
                call_idx = call_end + len(self.tool_call_end)
                continue

            func_name_start = func_start + len(func_prefix)
            func_name_end = call_content.find("\n", func_name_start)
            if func_name_end == -1:
                function_name = call_content[func_name_start:].strip()
            else:
                function_name = call_content[func_name_start:func_name_end].strip()

            # Extract JSON arguments from ```json ... ``` block
            json_start = call_content.find("```json\n")
            if json_start == -1:
                json_start = call_content.find("```json")
                if json_start == -1:
                    call_idx = call_end + len(self.tool_call_end)
                    continue
                json_start += len("```json")
            else:
                json_start += len("```json\n")

            json_end = call_content.find("```", json_start)
            if json_end == -1:
                call_idx = call_end + len(self.tool_call_end)
                continue

            args_str = call_content[json_start:json_end].strip()

            try:
                args_json = json.loads(args_str)
            except json.JSONDecodeError:
                logger.warning("R1Parser: bad JSON: %s", args_str[:200])
                call_idx = call_end + len(self.tool_call_end)
                continue

            calls.append(ToolCall(name=function_name, arguments=args_json))
            call_idx = call_end + len(self.tool_call_end)

        # Strip tool call blocks from remaining text
        remaining = response
        for tc_text in self._iter_raw_blocks(response):
            remaining = remaining.replace(tc_text, "")

        return calls, remaining.strip()

    def format_tool_call(self, tool_call: ToolCall) -> str:
        """
        Render tool calls in R1 special-token format::

            <｜tool▁call▁begin｜>function<｜tool▁sep｜>func_name
            ```json
            {"key": "value"}
            ```
            <｜tool▁call▁end｜>
        """
        args_str = json.dumps(tool_call.arguments, indent=2, ensure_ascii=False)
        return (
            f"{self.tool_call_begin}function{self.tool_sep}{tool_call.name}\n"
            f"```json\n{args_str}\n```\n"
            f"{self.tool_call_end}"
        )

    def format_tool_result(self, content: Any, name: str = "") -> str:
        return f'<tool_response>\n{{"name": "{name}", "content": {json.dumps(content)}}}\n</tool_response>'

    def get_tool_system_prompt(self, tools_json: list[dict[str, Any]]) -> str:
        tools_str = "\n".join(json.dumps(t, ensure_ascii=False) for t in tools_json)
        return (
            "# Tools\n\n"
            "You may call one or more functions to assist with the user query.\n"
            f"<tools>\n{tools_str}\n</tools>\n\n"
            f"For function call returns, you should first print {self.tool_calls_begin}\n\n"
            "For each function call, you should return object like:\n"
            f"{self.tool_call_begin}function{self.tool_sep}<function_name>\n"
            "```json\n"
            '{"param": "value"}\n'
            "```\n"
            f"{self.tool_call_end}"
        )

    def _iter_raw_blocks(self, text: str):
        """Yield raw tool-call block strings for stripping."""
        idx = 0
        while True:
            start = text.find(self.tool_call_begin, idx)
            if start == -1:
                break
            end = text.find(self.tool_call_end, start)
            if end == -1:
                break
            end += len(self.tool_call_end)
            yield text[start:end]
            idx = end
