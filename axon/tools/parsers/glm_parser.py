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
GLM (ChatGLM) tool-call parser.

Format::

    <tool_call>function_name
    <arg_key>key1</arg_key>
    <arg_value>value1</arg_value>
    <arg_key>key2</arg_key>
    <arg_value>value2</arg_value>
    </tool_call>
"""

from __future__ import annotations

import json
import re
from typing import Any

from axon.tools.parsers.base_parser import ToolCallParser, register_parser
from axon.tools.types import ToolCall


@register_parser("glm")
class GlmToolCallParser(ToolCallParser):
    """Parser for ChatGLM's tool call format with XML-style arguments."""

    _CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
    _ARG_KEY_RE = re.compile(r"<arg_key>(.*?)</arg_key>", re.DOTALL)
    _ARG_VAL_RE = re.compile(r"<arg_value>(.*?)</arg_value>", re.DOTALL)

    def parse(self, response: str) -> tuple[list[ToolCall], str]:
        calls: list[ToolCall] = []
        text = response

        for m in self._CALL_RE.finditer(response):
            content = m.group(1).strip()
            lines = content.split("\n", 1)
            function_name = lines[0].strip()
            body = lines[1] if len(lines) > 1 else ""

            keys = self._ARG_KEY_RE.findall(body)
            vals = self._ARG_VAL_RE.findall(body)

            arguments: dict[str, Any] = {}
            for k, v in zip(keys, vals, strict=False):
                k, v = k.strip(), v.strip()
                try:
                    arguments[k] = json.loads(v)
                except (json.JSONDecodeError, ValueError):
                    arguments[k] = v

            calls.append(ToolCall(name=function_name, arguments=arguments))
            text = text.replace(m.group(0), "")

        return calls, text.strip()

    def format_tool_call(self, tool_call: ToolCall) -> str:
        """
        Render tool calls in GLM XML-arg format::

            <tool_call>function_name
            <arg_key>key</arg_key>
            <arg_value>value</arg_value>
            </tool_call>
        """
        body = tool_call.name + "\n"
        for key, value in tool_call.arguments.items():
            if isinstance(value, str):
                val_str = value
            else:
                val_str = json.dumps(value, ensure_ascii=False)
            body += f"<arg_key>{key}</arg_key>\n"
            body += f"<arg_value>{val_str}</arg_value>\n"
        return f"<tool_call>{body}</tool_call>"

    def format_tool_result(self, content: Any, name: str = "") -> str:
        return f"<tool_response>\n{content}\n</tool_response>"

    def get_tool_system_prompt(self, tools_json: list[dict[str, Any]]) -> str:
        tools_str = "\n".join(json.dumps(t, ensure_ascii=False) for t in tools_json)
        return (
            "# Tools\n"
            "You may call one or more functions to assist with the user query.\n"
            "You are provided with function signatures within <tools></tools> XML tags:\n"
            f"<tools>\n{tools_str}\n</tools>\n"
            "For each function call, output the function name and arguments "
            "within the following XML format:\n"
            "<tool_call>{function-name}\n"
            "<arg_key>{arg-key-1}</arg_key>\n"
            "<arg_value>{arg-value-1}</arg_value>\n"
            "<arg_key>{arg-key-2}</arg_key>\n"
            "<arg_value>{arg-value-2}</arg_value>\n"
            "...\n"
            "</tool_call>"
        )
