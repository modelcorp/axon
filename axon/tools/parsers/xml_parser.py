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
XML-style tool-call parser.

Format::

    <function=file_editor>
      <parameter=command>view</parameter>
      <parameter=path>./src/main.py</parameter>
    </function>
"""

from __future__ import annotations

import re
from typing import Any

from axon.tools.parsers.base_parser import ToolCallParser, register_parser
from axon.tools.types import ToolCall


@register_parser("xml")
class XMLToolCallParser(ToolCallParser):
    """Parses XML-style tool calls (SWE-bench style)."""

    _FUNC_RE = re.compile(
        r"<function\s*=\s*([^>]+)>(.*?)</function>",
        re.DOTALL,
    )
    _PARAM_RE = re.compile(
        r"<parameter\s*=\s*([^>]+)>(.*?)</parameter>",
        re.DOTALL,
    )

    def parse(self, response: str) -> tuple[list[ToolCall], str]:
        calls: list[ToolCall] = []
        text = response

        for m in self._FUNC_RE.finditer(response):
            function_name = m.group(1).strip()
            function_body = m.group(2)

            params: dict[str, str] = {}
            for pm in self._PARAM_RE.finditer(function_body):
                params[pm.group(1).strip()] = pm.group(2).strip()

            calls.append(ToolCall(name=function_name, arguments=params))
            text = text.replace(m.group(0), "")

        return calls, text.strip()

    def format_tool_call(self, tool_call: ToolCall) -> str:
        """
        Render tool calls in XML format::

            <function=name>
            <parameter=key>value</parameter>
            </function>
        """
        body = f"<function={tool_call.name}>\n"
        for key, value in tool_call.arguments.items():
            body += f"<parameter={key}>{value}</parameter>\n"
        body += "</function>"
        return body

    def format_tool_result(self, content: Any, name: str = "") -> str:
        return f"<function_result>\n<n>{name}</n>\n<o>{content}</o>\n</function_result>"

    def get_tool_system_prompt(self, tools_json: list[dict[str, Any]]) -> str:
        return ""

    @classmethod
    def parse_single(cls, action_str: str) -> ToolCall:
        """Convenience: parse a single <function=...>...</function> block."""
        parser = cls()
        calls, _ = parser.parse(action_str)
        if calls:
            return calls[0]
        return ToolCall(name="", arguments={})
