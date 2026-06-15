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
"""Gemma4 native tool-call parser.

Format::

    <|tool_call>call:name{arg:<|"|>value<|"|>}<tool_call|>
    <|tool_response>response:name{value:<|"|>ok<|"|>}<tool_response|>
"""

from __future__ import annotations

from typing import Any

from axon.tools.parsers.base_parser import ToolCallParser, register_parser
from axon.tools.types import ToolCall

TOOL_CALL_START = "<|tool_call>"
TOOL_CALL_END = "<tool_call|>"
TOOL_RESPONSE_START = "<|tool_response>"
TOOL_RESPONSE_END = "<tool_response|>"
TURN_END = "<turn|>"
STRING_DELIM = '<|"|>'


def _format_required(required) -> str:
    return "[" + ",".join(f"{STRING_DELIM}{item}{STRING_DELIM}" for item in (required or [])) + "]"


def _format_argument(arg: Any, escape_keys: bool = False) -> str:
    """Serialize a Python value into Gemma4's compact argument format."""
    if isinstance(arg, str):
        return f"{STRING_DELIM}{arg}{STRING_DELIM}"
    if isinstance(arg, bool):
        return "true" if arg else "false"
    if arg is None:
        return "null"
    if isinstance(arg, dict):
        parts = []
        for k in sorted(arg):
            key = f"{STRING_DELIM}{k}{STRING_DELIM}" if escape_keys else k
            parts.append(f"{key}:{_format_argument(arg[k], escape_keys=escape_keys)}")
        return "{" + ",".join(parts) + "}"
    if isinstance(arg, list | tuple):
        return "[" + ",".join(_format_argument(v, escape_keys=escape_keys) for v in arg) + "]"
    return str(arg)


def _parse_gemma4_value(value_str: str) -> object:
    value_str = value_str.strip()
    if not value_str:
        return value_str
    if value_str == "true":
        return True
    if value_str == "false":
        return False
    if value_str == "null":
        return None
    try:
        if "." in value_str:
            return float(value_str)
        return int(value_str)
    except ValueError:
        return value_str


def _parse_gemma4_array(arr_str: str) -> list:
    items: list = []
    i = 0
    n = len(arr_str)

    while i < n:
        while i < n and arr_str[i] in (" ", ",", "\n", "\t"):
            i += 1
        if i >= n:
            break

        if arr_str[i:].startswith(STRING_DELIM):
            i += len(STRING_DELIM)
            end_pos = arr_str.find(STRING_DELIM, i)
            if end_pos == -1:
                items.append(arr_str[i:])
                break
            items.append(arr_str[i:end_pos])
            i = end_pos + len(STRING_DELIM)
        elif arr_str[i] == "{":
            end_pos = _find_balanced_end(arr_str, i)
            if end_pos == -1:
                items.append(_parse_gemma4_args(arr_str[i + 1 :]))
                break
            items.append(_parse_gemma4_args(arr_str[i + 1 : end_pos]))
            i = end_pos + 1
        elif arr_str[i] == "[":
            end_pos = _find_balanced_end(arr_str, i)
            if end_pos == -1:
                items.append(_parse_gemma4_array(arr_str[i + 1 :]))
                break
            items.append(_parse_gemma4_array(arr_str[i + 1 : end_pos]))
            i = end_pos + 1
        else:
            val_start = i
            while i < n and arr_str[i] not in (",", "]"):
                i += 1
            items.append(_parse_gemma4_value(arr_str[val_start:i]))

    return items


def _parse_gemma4_args(args_str: str) -> dict:
    if not args_str or not args_str.strip():
        return {}

    result: dict = {}
    i = 0
    n = len(args_str)

    while i < n:
        while i < n and args_str[i] in (" ", ",", "\n", "\t"):
            i += 1
        if i >= n:
            break

        key_start = i
        while i < n and args_str[i] != ":":
            i += 1
        if i >= n:
            break
        key = args_str[key_start:i].strip()
        i += 1

        while i < n and args_str[i] in (" ", "\n", "\t"):
            i += 1
        if i >= n:
            result[key] = ""
            break

        if args_str[i:].startswith(STRING_DELIM):
            i += len(STRING_DELIM)
            val_start = i
            end_pos = args_str.find(STRING_DELIM, i)
            if end_pos == -1:
                result[key] = args_str[val_start:]
                break
            result[key] = args_str[val_start:end_pos]
            i = end_pos + len(STRING_DELIM)
        elif args_str[i] == "{":
            end_pos = _find_balanced_end(args_str, i)
            if end_pos == -1:
                result[key] = _parse_gemma4_args(args_str[i + 1 :])
                break
            result[key] = _parse_gemma4_args(args_str[i + 1 : end_pos])
            i = end_pos + 1
        elif args_str[i] == "[":
            end_pos = _find_balanced_end(args_str, i)
            if end_pos == -1:
                result[key] = _parse_gemma4_array(args_str[i + 1 :])
                break
            result[key] = _parse_gemma4_array(args_str[i + 1 : end_pos])
            i = end_pos + 1
        else:
            val_start = i
            while i < n and args_str[i] not in (",", "}", "]"):
                i += 1
            result[key] = _parse_gemma4_value(args_str[val_start:i])

    return result


def _find_balanced_end(text: str, start: int) -> int:
    """Find the matching ``}`` or ``]`` for a Gemma4 object/array."""
    opener = text[start]
    closer = "}" if opener == "{" else "]"
    depth = 0
    i = start
    n = len(text)

    while i < n:
        if text[i:].startswith(STRING_DELIM):
            i += len(STRING_DELIM)
            end = text.find(STRING_DELIM, i)
            if end == -1:
                return -1
            i = end + len(STRING_DELIM)
            continue
        if text[i] == opener:
            depth += 1
        elif text[i] == closer:
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def _format_parameters(properties: dict, required: list | None, filter_keys: bool = False) -> str:
    """Render Gemma4 native tool parameter declarations."""
    standard_keys = {"description", "type", "properties", "required", "nullable"}
    parts = []
    for key in sorted(properties or {}):
        value = properties[key]
        if filter_keys and key in standard_keys:
            continue
        if not isinstance(value, dict):
            parts.append(f"{key}:{{type:{STRING_DELIM}{str(value).upper()}{STRING_DELIM}}}")
            continue

        body = []
        desc = value.get("description")
        if desc:
            body.append(f"description:{STRING_DELIM}{desc}{STRING_DELIM}")

        value_type = value.get("type")
        value_type_upper = value_type.upper() if isinstance(value_type, str) else None

        if value_type_upper == "STRING" and value.get("enum"):
            body.append(f"enum:{_format_argument(value['enum'])}")
        elif value_type_upper == "ARRAY":
            items = value.get("items")
            if isinstance(items, dict) and items:
                item_parts = []
                for item_key in sorted(items):
                    item_value = items[item_key]
                    if item_value is None:
                        continue
                    if item_key == "properties" and isinstance(item_value, dict):
                        item_parts.append(
                            "properties:{" + _format_parameters(item_value, items.get("required", [])) + "}"
                        )
                    elif item_key == "required":
                        item_parts.append("required:" + _format_required(item_value))
                    elif item_key == "type":
                        if isinstance(item_value, str):
                            item_parts.append("type:" + _format_argument(item_value.upper()))
                        else:
                            item_parts.append("type:" + _format_argument([v.upper() for v in item_value]))
                    else:
                        item_parts.append(f"{item_key}:{_format_argument(item_value)}")
                body.append("items:{" + ",".join(item_parts) + "}")

        if value.get("nullable"):
            body.append("nullable:true")

        if value_type_upper == "OBJECT":
            child_properties = value.get("properties")
            if isinstance(child_properties, dict):
                body.append("properties:{" + _format_parameters(child_properties, value.get("required", [])) + "}")
            elif isinstance(value, dict):
                nested = _format_parameters(value, value.get("required", []), filter_keys=True)
                if nested:
                    body.append("properties:{" + nested + "}")
            if value.get("required"):
                body.append("required:" + _format_required(value.get("required")))

        if value_type_upper:
            body.append(f"type:{STRING_DELIM}{value_type_upper}{STRING_DELIM}")

        parts.append(f"{key}:{{" + ",".join(body) + "}")
    return ",".join(parts)


def _format_function_declaration(tool_data: dict) -> str:
    """Render a Gemma4 ``declaration:...`` tool payload."""
    fn = tool_data.get("function", tool_data)
    name = fn.get("name", "")
    description = fn.get("description", "")
    result = f"declaration:{name}{{description:{STRING_DELIM}{description}{STRING_DELIM}"

    params = fn.get("parameters")
    if isinstance(params, dict) and params:
        param_parts = []
        properties = params.get("properties")
        if isinstance(properties, dict) and properties:
            param_parts.append("properties:{" + _format_parameters(properties, params.get("required", [])) + "}")
        if params.get("required"):
            param_parts.append("required:" + _format_required(params.get("required")))
        if params.get("type"):
            param_parts.append(f"type:{STRING_DELIM}{str(params['type']).upper()}{STRING_DELIM}")
        result += ",parameters:{" + ",".join(param_parts) + "}"

    response = fn.get("response")
    if isinstance(response, dict):
        response_parts = []
        if response.get("description"):
            response_parts.append(f"description:{STRING_DELIM}{response['description']}{STRING_DELIM}")
        if response.get("type") and str(response["type"]).upper() == "OBJECT":
            response_parts.append(f"type:{STRING_DELIM}{str(response['type']).upper()}{STRING_DELIM}")
        result += ",response:{" + ",".join(response_parts) + "}"

    return result + "}"


@register_parser("gemma4")
class Gemma4ToolCallParser(ToolCallParser):
    """Parser/formatter for Gemma4's compact native tool format."""

    def parse(self, response: str) -> tuple[list[ToolCall], str]:
        calls: list[ToolCall] = []
        chunks: list[str] = []
        pos = 0

        while True:
            start = response.find(TOOL_CALL_START, pos)
            if start == -1:
                chunks.append(response[pos:])
                break

            call_start = start + len(TOOL_CALL_START)
            if not response.startswith("call:", call_start):
                chunks.append(response[pos:call_start])
                pos = call_start
                continue

            name_start = call_start + len("call:")
            brace = response.find("{", name_start)
            if brace == -1:
                chunks.append(response[pos:])
                break

            name = response[name_start:brace].strip()
            close = _find_balanced_end(response, brace)
            if close == -1:
                chunks.append(response[pos:])
                break

            args_str = response[brace + 1 : close]
            end = close + 1
            if response.startswith(TOOL_CALL_END, end):
                end += len(TOOL_CALL_END)
            elif response.startswith(TURN_END, end):
                end += len(TURN_END)
            if response.startswith(TOOL_RESPONSE_START, end):
                end += len(TOOL_RESPONSE_START)

            chunks.append(response[pos:start])
            calls.append(ToolCall(name=name, arguments=_parse_gemma4_args(args_str)))
            pos = end

        return calls, "".join(chunks).strip()

    def format_tool_call(self, tool_call: ToolCall) -> str:
        args = tool_call.arguments if isinstance(tool_call.arguments, dict) else {"raw": tool_call.arguments}
        args_str = ",".join(f"{k}:{_format_argument(args[k])}" for k in sorted(args))
        return f"{TOOL_CALL_START}call:{tool_call.name}{{{args_str}}}{TOOL_CALL_END}"

    def format_tool_calls(self, tool_calls: list[ToolCall]) -> str:
        return "".join(self.format_tool_call(tc) for tc in tool_calls)

    def format_tool_result(self, content: Any, name: str = "") -> str:
        tool_name = name or "unknown"
        if isinstance(content, dict):
            inner = ",".join(f"{k}:{_format_argument(content[k])}" for k in sorted(content))
        else:
            inner = f"value:{_format_argument(content)}"
        return f"{TOOL_RESPONSE_START}response:{tool_name}{{{inner}}}{TOOL_RESPONSE_END}"

    def get_tool_system_prompt(self, tools_json: list[dict[str, Any]]) -> str:
        return "".join(f"<|tool>{_format_function_declaration(tool).strip()}<tool|>" for tool in tools_json)
