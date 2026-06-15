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
OpenAI Harmony tool-call parser.

This is the format used by models trained with the OpenAI Harmony chat template
(e.g. GPT-compatible models using ``<|start|>``, ``<|end|>``, ``<|call|>`` tokens).

Tool call output format::

    <|start|>assistant to=functions.get_weather<|channel|>commentary json<|message|>{"location": "SF"}<|call|>

Tool result format::

    <|start|>functions.get_weather to=assistant<|channel|>commentary<|message|>{"temp": 72}<|end|>

The parser extracts:
  - Function name from ``to=functions.{name}``
  - Arguments from the JSON between ``<|message|>`` and ``<|call|>``
  - Content type hint from ``<|channel|>commentary {content_type}``

Notes
-----
- The Harmony format supports only ONE tool call per assistant message
  (the chat template uses ``tool_calls[0]``).
- Analysis/thinking content before the tool call is treated as remaining text.
- Tool results are JSON-encoded in the formatted output.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from axon.tools.parsers.base_parser import ToolCallParser, register_parser
from axon.tools.types import ToolCall

logger = logging.getLogger(__name__)


@register_parser("openai_harmony")
class OpenAIHarmonyToolCallParser(ToolCallParser):
    """Parser for OpenAI Harmony special-token tool call format."""

    _CALL_RE = re.compile(
        r"to=functions\.(\S+)"  # group 1: function name
        r"<\|channel\|>commentary\s*(\w*)"  # group 2: content type
        r"<\|message\|>"
        r"(.*?)"  # group 3: arguments
        r"<\|call\|>",
        re.DOTALL,
    )

    _ANALYSIS_RE = re.compile(
        r"<\|channel\|>analysis<\|message\|>(.*?)(?:<\|end\|>|$)",
        re.DOTALL,
    )

    def parse(self, response: str) -> tuple[list[ToolCall], str]:
        calls: list[ToolCall] = []
        text = response

        for m in self._CALL_RE.finditer(response):
            function_name = m.group(1).strip()
            content_type = m.group(2).strip() or "json"
            args_raw = m.group(3).strip()

            arguments = self._parse_arguments(args_raw, content_type)
            if arguments is not None:
                calls.append(ToolCall(name=function_name, arguments=arguments))
            text = text.replace(m.group(0), "")

        # Extract analysis/thinking text
        remaining_parts = []
        for m in self._ANALYSIS_RE.finditer(text):
            analysis = m.group(1).strip()
            if analysis:
                remaining_parts.append(analysis)
            text = text.replace(m.group(0), "")

        # Capture <|channel|>final<|message|> content
        final_re = re.compile(
            r"<\|channel\|>final<\|message\|>(.*?)(?:<\|end\|>|<\|return\|>|$)",
            re.DOTALL,
        )
        for m in final_re.finditer(text):
            final_text = m.group(1).strip()
            if final_text:
                remaining_parts.append(final_text)
            text = text.replace(m.group(0), "")

        # Clean up remaining control tokens
        remaining = text
        for token in [
            "<|start|>",
            "<|end|>",
            "<|return|>",
            "<|call|>",
            "assistant",
            "<|channel|>",
            "<|message|>",
        ]:
            remaining = remaining.replace(token, "")
        remaining = remaining.strip()

        if remaining_parts:
            remaining = "\n".join(remaining_parts) + ("\n" + remaining if remaining else "")

        return calls, remaining.strip()

    def format_tool_call(self, tool_call: ToolCall) -> str:
        """
        Render tool calls in Harmony format.

        Produces::

            to=functions.{name}<|channel|>commentary json<|message|>{args}<|call|>

        Note: Only one tool call per Harmony message.
        """
        args_str = json.dumps(tool_call.arguments, ensure_ascii=False)
        content_type = (tool_call.raw_tool_call or {}).get("content_type", "json")
        return f" to=functions.{tool_call.name}<|channel|>commentary {content_type}<|message|>{args_str}<|call|>"

    def format_tool_calls(self, tool_calls: list[ToolCall]) -> str:
        """Render multiple ToolCalls."""
        return "".join(self.format_tool_call(tc) for tc in tool_calls)

    def format_tool_result(self, content: Any, name: str = "") -> str:
        """
        Render results in Harmony ``functions.{name} to=assistant`` format.
        """
        content = json.dumps(content, ensure_ascii=False)
        return f"functions.{name} to=assistant<|channel|>commentary<|message|>{content}"

    def get_tool_system_prompt(self, tools_json: list[dict[str, Any]]) -> str:
        """
        Generate TypeScript namespace tool definitions.

        This is the authoritative implementation — moved from
        OpenAIHarmonyChatTemplateParser._render_tool_namespace().
        """
        if len(tools_json) == 1 and "_builtin" in tools_json[0]:
            # Hack to also render builtin tools.
            builtin_tools = tools_json[0]["_builtin"]
            return self._render_builtin_tools(builtin_tools)
        return self._render_tool_namespace("functions", tools_json)

    # =========================================================================
    # TypeScript namespace rendering
    # =========================================================================

    def _render_tool_namespace(self, namespace_name: str, tools: list[dict]) -> str:
        """Render tools as a TypeScript namespace."""
        result = f"## {namespace_name}\n\n"
        result += f"namespace {namespace_name} {{\n\n"

        for tool in tools:
            tool_func = tool.get("function", tool)
            tool_name = tool_func.get("name", "")
            tool_desc = tool_func.get("description", "")
            parameters = tool_func.get("parameters", {})

            result += f"// {tool_desc}\n"
            result += f"type {tool_name} = "

            if parameters and parameters.get("properties"):
                result += "(_: {\n"
                props = parameters.get("properties", {})
                required = parameters.get("required", [])

                prop_lines = []
                for param_name, param_spec in props.items():
                    param_desc = param_spec.get("description", "")
                    if param_desc:
                        prop_lines.append(f"// {param_desc}")

                    optional = "?" if param_name not in required else ""
                    is_nested_obj = param_spec.get("type") == "object"
                    param_type = self._render_typescript_type(param_spec, required, is_nested=is_nested_obj)

                    default_comment = ""
                    if "default" in param_spec:
                        if param_spec.get("enum"):
                            default_comment = f", // default: {param_spec['default']}"
                        else:
                            default_comment = f", // default: {json.dumps(param_spec['default'])}"

                    prop_lines.append(f"{param_name}{optional}: {param_type}{default_comment},")

                result += "\n".join(prop_lines) + "\n"
                result += "}) => any;\n\n"
            else:
                result += "() => any;\n\n"

        result += f"}} // namespace {namespace_name}"
        return result

    def _render_typescript_type(self, param_spec: dict, required_params: list = None, is_nested: bool = False) -> str:
        """Render a JSON Schema type as TypeScript."""
        ptype = param_spec.get("type", "any")

        if ptype == "array":
            items = param_spec.get("items", {})
            if items.get("type") == "string":
                return "string[]"
            elif items.get("type") in ("number", "integer"):
                return "number[]"
            elif items.get("type") == "boolean":
                return "boolean[]"
            else:
                return "any[]"
        elif ptype == "string":
            if "enum" in param_spec:
                return '"' + '" | "'.join(param_spec["enum"]) + '"'
            return "string"
        elif ptype in ("number", "integer"):
            return "number"
        elif ptype == "boolean":
            return "boolean"
        elif ptype == "object":
            if "properties" in param_spec:
                if is_nested:
                    # Nested objects use specific indentation to match Jinja template
                    result = "{"
                    prop_items = list(param_spec["properties"].items())
                    for idx, (prop_name, prop_spec) in enumerate(prop_items):
                        optional = "?" if prop_name not in param_spec.get("required", []) else ""
                        prop_type = self._render_typescript_type(
                            prop_spec, param_spec.get("required", []), is_nested=True
                        )
                        if idx == 0:
                            result += f"\n{prop_name}{optional}: \n                {prop_type}"
                        else:
                            result += f" {prop_name}{optional}: \n                {prop_type}"
                        if idx < len(prop_items) - 1:
                            result += ","
                    result += "}"
                    return result
                else:
                    props = []
                    for prop_name, prop_spec in param_spec["properties"].items():
                        optional = "?" if prop_name not in param_spec.get("required", []) else ""
                        prop_type = self._render_typescript_type(
                            prop_spec, param_spec.get("required", []), is_nested=False
                        )
                        props.append(f"{prop_name}{optional}: {prop_type}")
                    return "{\n" + ",\n".join(props) + "\n}"
            return "object"
        return "any"

    def _render_builtin_tools(self, builtin_tools):
        """Render builtin tools documentation."""
        result = "# Tools\n\n"

        if "browser" in builtin_tools:
            result += """## browser

// Tool for browsing.
// The `cursor` appears in brackets before each browsing display: `[{cursor}]`.
// Cite information from the tool using the following format:
// `【{cursor}†L{line_start}(-L{line_end})?】`, for example: `【6†L9-L11】` or `【8†L3】`.
// Do not quote more than 10 words directly from the tool output.
// sources=web (default: web)
namespace browser {

// Searches for information related to `query` and displays `topn` results.
type search = (_: {
query: string,
topn?: number, // default: 10
source?: string,
}) => any;

// Opens the link `id` from the page indicated by `cursor` starting at line number `loc`, showing `num_lines` lines.
// Valid link ids are displayed with the formatting: `【{id}†.*】`.
// If `cursor` is not provided, the most recent page is implied.
// If `id` is a string, it is treated as a fully qualified URL associated with `source`.
// If `loc` is not provided, the viewport will be positioned at the beginning of the document or centered on the most relevant passage, if available.
// Use this function without `id` to scroll to a new location of an opened page.
type open = (_: {
id?: number | string, // default: -1
cursor?: number, // default: -1
loc?: number, // default: -1
num_lines?: number, // default: -1
view_source?: boolean, // default: false
source?: string,
}) => any;

// Finds exact matches of `pattern` in the current page, or the page given by `cursor`.
type find = (_: {
pattern: string,
cursor?: number, // default: -1
}) => any;

} // namespace browser
"""

        if "python" in builtin_tools:
            # Add newline only if browser was already added
            prefix = "\n" if "browser" in builtin_tools else ""
            result += (
                prefix
                + """## python

Use this tool to execute Python code in your chain of thought. The code will not be shown to the user. This tool should be used for internal reasoning, but not for code that is intended to be visible to the user (e.g. when creating plots, tables, or files).

When you send a message containing Python code to python, it will be executed in a stateful Jupyter notebook environment. python will respond with the output of the execution or time out after 120.0 seconds. The drive at '/mnt/data' can be used to save and persist user files. Internet access for this session is UNKNOWN. Depends on the cluster.
"""
            )

        return result

    # -- Internal helpers -----------------------------------------------------

    def _parse_arguments(self, raw: str, content_type: str) -> dict | None:
        if not raw:
            return {}

        if content_type == "json" or not content_type:
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    return parsed
                if isinstance(parsed, str):
                    try:
                        inner = json.loads(parsed)
                        if isinstance(inner, dict):
                            return inner
                    except json.JSONDecodeError:
                        pass
                return {"raw": parsed}
            except json.JSONDecodeError:
                logger.warning(
                    "HarmonyParser: bad JSON in tool call args: %s",
                    raw[:200],
                )
                return None
        else:
            return {"raw": raw}
