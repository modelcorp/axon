"""Tests for axon.tools.parsers.openai_harmony_parser module."""

import json

import pytest

from axon.tools.parsers.openai_harmony_parser import OpenAIHarmonyToolCallParser
from axon.tools.types import ToolCall, ToolResult


class TestOpenAIHarmonyParserParse:
    def setup_method(self):
        self.parser = OpenAIHarmonyToolCallParser()

    # -- single tool call ------------------------------------------------------

    def test_parse_single_call(self):
        response = 'to=functions.get_weather<|channel|>commentary json<|message|>{"city": "NYC"}<|call|>'
        calls, remaining = self.parser.parse(response)
        assert len(calls) == 1
        assert calls[0].name == "get_weather"
        assert calls[0].arguments == {"city": "NYC"}

    def test_parse_with_assistant_prefix(self):
        response = '<|start|>assistant to=functions.search<|channel|>commentary json<|message|>{"q": "hello"}<|call|>'
        calls, remaining = self.parser.parse(response)
        assert len(calls) == 1
        assert calls[0].name == "search"
        assert calls[0].arguments == {"q": "hello"}

    # -- with analysis text ----------------------------------------------------

    def test_parse_with_analysis_text(self):
        response = (
            "<|channel|>analysis<|message|>I need to look up the weather<|end|>"
            'to=functions.get_weather<|channel|>commentary json<|message|>{"city": "SF"}<|call|>'
        )
        calls, remaining = self.parser.parse(response)
        assert len(calls) == 1
        assert calls[0].name == "get_weather"
        assert "I need to look up the weather" in remaining

    # -- with final channel text -----------------------------------------------

    def test_parse_with_final_channel_text(self):
        response = "<|channel|>final<|message|>Here is your answer.<|end|>"
        calls, remaining = self.parser.parse(response)
        assert len(calls) == 0
        assert "Here is your answer." in remaining

    def test_parse_with_final_and_tool_call(self):
        response = (
            'to=functions.calc<|channel|>commentary json<|message|>{"expr": "1+1"}<|call|>'
            "<|channel|>final<|message|>The result is 2.<|end|>"
        )
        calls, remaining = self.parser.parse(response)
        assert len(calls) == 1
        assert calls[0].name == "calc"
        assert "The result is 2." in remaining

    # -- no tool calls (plain text) --------------------------------------------

    def test_parse_plain_text_no_calls(self):
        response = "Just a normal response with no tool calls."
        calls, remaining = self.parser.parse(response)
        assert len(calls) == 0
        assert "Just a normal response" in remaining

    def test_parse_empty_response(self):
        calls, remaining = self.parser.parse("")
        assert calls == []
        assert remaining == ""

    # -- bad JSON returns None / skipped call ----------------------------------

    def test_parse_bad_json_skips_call(self):
        response = "to=functions.broken<|channel|>commentary json<|message|>{not valid json}<|call|>"
        calls, remaining = self.parser.parse(response)
        assert len(calls) == 0

    # -- content type ----------------------------------------------------------

    def test_parse_non_json_content_type(self):
        response = "to=functions.render<|channel|>commentary text<|message|>some raw text<|call|>"
        calls, remaining = self.parser.parse(response)
        assert len(calls) == 1
        assert calls[0].name == "render"
        assert calls[0].arguments == {"raw": "some raw text"}

    def test_parse_empty_content_type_defaults_to_json(self):
        response = 'to=functions.foo<|channel|>commentary<|message|>{"key": "val"}<|call|>'
        calls, _ = self.parser.parse(response)
        assert len(calls) == 1
        assert calls[0].arguments == {"key": "val"}

    # -- empty arguments -------------------------------------------------------

    def test_parse_empty_arguments(self):
        response = "to=functions.noop<|channel|>commentary json<|message|><|call|>"
        calls, _ = self.parser.parse(response)
        assert len(calls) == 1
        assert calls[0].name == "noop"
        assert calls[0].arguments == {}

    # -- control token cleanup -------------------------------------------------

    def test_control_tokens_stripped_from_remaining(self):
        response = '<|start|>assistant to=functions.test<|channel|>commentary json<|message|>{"a": 1}<|call|><|end|>'
        calls, remaining = self.parser.parse(response)
        assert len(calls) == 1
        assert "<|start|>" not in remaining
        assert "<|end|>" not in remaining
        assert "<|call|>" not in remaining


class TestOpenAIHarmonyParserFormat:
    def setup_method(self):
        self.parser = OpenAIHarmonyToolCallParser()

    # -- format_tool_call ------------------------------------------------------

    def test_format_tool_call_basic(self):
        tc = ToolCall(name="get_weather", arguments={"city": "NYC"})
        result = self.parser.format_tool_call(tc)
        assert "to=functions.get_weather" in result
        assert "<|channel|>commentary json<|message|>" in result
        assert "<|call|>" in result
        # arguments should be valid JSON
        json_start = result.index("<|message|>") + len("<|message|>")
        json_end = result.index("<|call|>")
        parsed_args = json.loads(result[json_start:json_end])
        assert parsed_args == {"city": "NYC"}

    def test_format_tool_call_roundtrip(self):
        tc = ToolCall(name="search", arguments={"query": "test", "limit": 10})
        formatted = self.parser.format_tool_call(tc)
        calls, _ = self.parser.parse(formatted)
        assert len(calls) == 1
        assert calls[0].name == tc.name
        assert calls[0].arguments == tc.arguments

    def test_format_tool_calls_concatenates(self):
        """Harmony format_tool_calls joins without newline (overridden method)."""
        tcs = [
            ToolCall(name="a", arguments={"x": 1}),
            ToolCall(name="b", arguments={"y": 2}),
        ]
        result = self.parser.format_tool_calls(tcs)
        assert "to=functions.a" in result
        assert "to=functions.b" in result

    # -- format_tool_result ----------------------------------------------------

    def test_format_tool_result(self):
        result = self.parser.format_tool_result({"temp": 72}, name="get_weather")
        assert "functions.get_weather to=assistant" in result
        assert "<|channel|>commentary<|message|>" in result
        # Content should be JSON-encoded
        assert '"temp"' in result
        assert "72" in result

    def test_format_tool_result_string_content(self):
        result = self.parser.format_tool_result("success", name="action")
        assert "functions.action to=assistant" in result
        assert '"success"' in result

    # -- format_tool_results ---------------------------------------------------

    def test_format_tool_results(self):
        results = [
            ToolResult(content="data1", name="tool_a"),
            ToolResult(content="data2", name="tool_b"),
        ]
        output = self.parser.format_tool_results(results)
        assert "functions.tool_a to=assistant" in output
        assert "functions.tool_b to=assistant" in output


class TestOpenAIHarmonyParserTypeScript:
    def setup_method(self):
        self.parser = OpenAIHarmonyToolCallParser()

    # -- _render_typescript_type (parametrized) --------------------------------

    @pytest.mark.parametrize(
        "spec,expected",
        [
            ({"type": "string"}, "string"),
            ({"type": "number"}, "number"),
            ({"type": "integer"}, "number"),
            ({"type": "boolean"}, "boolean"),
            ({"type": "foobar"}, "any"),
            ({}, "any"),
            ({"type": "object"}, "object"),
            ({"type": "array", "items": {"type": "string"}}, "string[]"),
            ({"type": "array", "items": {"type": "number"}}, "number[]"),
            ({"type": "array", "items": {"type": "integer"}}, "number[]"),
            ({"type": "array", "items": {"type": "boolean"}}, "boolean[]"),
            ({"type": "array", "items": {}}, "any[]"),
        ],
    )
    def test_scalar_and_array_types(self, spec, expected):
        assert self.parser._render_typescript_type(spec) == expected

    def test_string_enum(self):
        result = self.parser._render_typescript_type({"type": "string", "enum": ["a", "b", "c"]})
        assert result == '"a" | "b" | "c"'

    def test_object_with_required_and_optional_props(self):
        spec = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "number"},
            },
            "required": ["name"],
        }
        result = self.parser._render_typescript_type(spec, spec["required"])
        assert "name: string" in result  # required — no ?
        assert "age" in result

    def test_nested_object_indentation(self):
        spec = {
            "type": "object",
            "properties": {"inner": {"type": "string"}},
            "required": ["inner"],
        }
        result = self.parser._render_typescript_type(spec, spec["required"], is_nested=True)
        assert "inner:" in result
        assert "string" in result

    # -- get_tool_system_prompt ------------------------------------------------

    def test_system_prompt_with_tools(self):
        tools_json = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string", "description": "City name"},
                            "units": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"],
                                "default": "celsius",
                            },
                        },
                        "required": ["city"],
                    },
                },
            }
        ]
        prompt = self.parser.get_tool_system_prompt(tools_json)
        assert "namespace functions" in prompt
        assert "get_weather" in prompt
        assert "Get current weather" in prompt
        assert "city: string" in prompt
        assert '"celsius" | "fahrenheit"' in prompt

    def test_system_prompt_no_parameters(self):
        tools_json = [
            {
                "type": "function",
                "function": {
                    "name": "get_time",
                    "description": "Get current time",
                    "parameters": {},
                },
            }
        ]
        prompt = self.parser.get_tool_system_prompt(tools_json)
        assert "get_time" in prompt
        assert "() => any;" in prompt

    def test_system_prompt_multiple_tools(self):
        tools_json = [
            {
                "function": {
                    "name": "tool_a",
                    "description": "Tool A",
                    "parameters": {
                        "type": "object",
                        "properties": {"x": {"type": "number"}},
                        "required": ["x"],
                    },
                }
            },
            {
                "function": {
                    "name": "tool_b",
                    "description": "Tool B",
                    "parameters": {
                        "type": "object",
                        "properties": {"y": {"type": "string"}},
                        "required": [],
                    },
                }
            },
        ]
        prompt = self.parser.get_tool_system_prompt(tools_json)
        assert "tool_a" in prompt
        assert "tool_b" in prompt
        assert "x: number" in prompt
        assert "y?" in prompt or "y: string" in prompt

    def test_system_prompt_with_default_values(self):
        tools_json = [
            {
                "function": {
                    "name": "search",
                    "description": "Search things",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "limit": {"type": "number", "default": 10},
                        },
                        "required": ["query"],
                    },
                }
            }
        ]
        prompt = self.parser.get_tool_system_prompt(tools_json)
        assert "default: 10" in prompt

    def test_system_prompt_with_array_parameter(self):
        tools_json = [
            {
                "function": {
                    "name": "filter",
                    "description": "Filter items",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "tags": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": ["tags"],
                    },
                }
            }
        ]
        prompt = self.parser.get_tool_system_prompt(tools_json)
        assert "string[]" in prompt


class TestOpenAIHarmonyParserBuiltinTools:
    def setup_method(self):
        self.parser = OpenAIHarmonyToolCallParser()

    def test_builtin_browser(self):
        tools_json = [{"_builtin": ["browser"]}]
        prompt = self.parser.get_tool_system_prompt(tools_json)
        assert "## browser" in prompt
        assert "namespace browser" in prompt
        assert "type search" in prompt
        assert "type open" in prompt
        assert "type find" in prompt
        assert "python" not in prompt.lower() or "python" not in prompt.split("## browser")[0]

    def test_builtin_python(self):
        tools_json = [{"_builtin": ["python"]}]
        prompt = self.parser.get_tool_system_prompt(tools_json)
        assert "## python" in prompt
        assert "Jupyter" in prompt

    def test_builtin_both(self):
        tools_json = [{"_builtin": ["browser", "python"]}]
        prompt = self.parser.get_tool_system_prompt(tools_json)
        assert "## browser" in prompt
        assert "## python" in prompt
        assert "namespace browser" in prompt
        assert "Jupyter" in prompt

    def test_builtin_empty_list(self):
        tools_json = [{"_builtin": []}]
        prompt = self.parser.get_tool_system_prompt(tools_json)
        assert "# Tools" in prompt
        # No specific tool sections
        assert "namespace browser" not in prompt
        assert "## python" not in prompt


class TestOpenAIHarmonyParserParseArguments:
    def setup_method(self):
        self.parser = OpenAIHarmonyToolCallParser()

    def test_parse_arguments_valid_json(self):
        result = self.parser._parse_arguments('{"key": "value"}', "json")
        assert result == {"key": "value"}

    def test_parse_arguments_empty_string(self):
        result = self.parser._parse_arguments("", "json")
        assert result == {}

    def test_parse_arguments_invalid_json(self):
        result = self.parser._parse_arguments("{bad json}", "json")
        assert result is None

    def test_parse_arguments_non_dict_json(self):
        """A JSON array parsed as non-dict gets wrapped in {"raw": ...}."""
        result = self.parser._parse_arguments("[1, 2, 3]", "json")
        assert result == {"raw": [1, 2, 3]}

    def test_parse_arguments_double_encoded_json(self):
        """A JSON string containing a JSON dict gets unwrapped."""
        inner = json.dumps({"a": 1})
        outer = json.dumps(inner)
        result = self.parser._parse_arguments(outer, "json")
        assert result == {"a": 1}

    def test_parse_arguments_raw_content_type(self):
        result = self.parser._parse_arguments("some raw text", "text")
        assert result == {"raw": "some raw text"}


# =============================================================================
# Hardened edge cases
# =============================================================================


class TestOpenAIHarmonyEdgeCases:
    def setup_method(self):
        self.parser = OpenAIHarmonyToolCallParser()

    def test_unicode_in_tool_arguments(self):
        args = json.dumps({"text": "\u4f60\u597d\u4e16\u754c \u00e9\u00e0\u00fc"})
        response = f"to=functions.translate<|channel|>commentary json<|message|>{args}<|call|>"
        calls, _ = self.parser.parse(response)
        assert calls[0].arguments["text"] == "\u4f60\u597d\u4e16\u754c \u00e9\u00e0\u00fc"

    def test_deeply_nested_json_arguments(self):
        nested = {"config": {"model": {"layers": [{"type": "attention", "heads": 8}]}}}
        args = json.dumps(nested)
        response = f"to=functions.setup<|channel|>commentary json<|message|>{args}<|call|>"
        calls, _ = self.parser.parse(response)
        assert calls[0].arguments == nested

    def test_tool_call_followed_by_analysis_and_final(self):
        """Complex response with all three sections."""
        response = (
            "<|channel|>analysis<|message|>Let me think about this...<|end|>"
            '<|start|>assistant to=functions.search<|channel|>commentary json<|message|>{"q": "test"}<|call|>'
            "<|channel|>final<|message|>Here is what I found.<|end|>"
        )
        calls, remaining = self.parser.parse(response)
        assert len(calls) == 1
        assert calls[0].name == "search"
        assert "Let me think about this..." in remaining
        assert "Here is what I found." in remaining

    def test_multiple_analysis_blocks(self):
        response = (
            "<|channel|>analysis<|message|>First thought<|end|><|channel|>analysis<|message|>Second thought<|end|>"
        )
        calls, remaining = self.parser.parse(response)
        assert len(calls) == 0
        assert "First thought" in remaining
        assert "Second thought" in remaining

    def test_return_token_in_final(self):
        response = "<|channel|>final<|message|>The answer is 42.<|return|>"
        calls, remaining = self.parser.parse(response)
        assert "The answer is 42." in remaining

    def test_arguments_with_newlines(self):
        args = json.dumps({"code": "def hello():\n    print('hi')\n"})
        response = f"to=functions.run_code<|channel|>commentary json<|message|>{args}<|call|>"
        calls, _ = self.parser.parse(response)
        assert "def hello():" in calls[0].arguments["code"]

    def test_function_name_with_underscores_and_numbers(self):
        response = 'to=functions.get_item_v2<|channel|>commentary json<|message|>{"id": 42}<|call|>'
        calls, _ = self.parser.parse(response)
        assert calls[0].name == "get_item_v2"

    def test_typescript_optional_parameters(self):
        tools_json = [
            {
                "function": {
                    "name": "search",
                    "description": "Search",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query"},
                            "limit": {"type": "number", "description": "Max results"},
                            "offset": {"type": "number"},
                        },
                        "required": ["query"],
                    },
                }
            }
        ]
        prompt = self.parser.get_tool_system_prompt(tools_json)
        assert "query: string" in prompt  # required, no ?
        assert "limit?" in prompt  # optional
        assert "offset?" in prompt  # optional

    def test_format_roundtrip_preserves_arguments(self):
        """Full roundtrip: create ToolCall, format, parse, verify."""
        original = ToolCall(
            name="complex_tool",
            arguments={
                "nested": {"key": [1, 2, 3]},
                "flag": True,
                "text": 'He said "hello"',
            },
        )
        formatted = self.parser.format_tool_call(original)
        calls, _ = self.parser.parse(formatted)
        assert len(calls) == 1
        assert calls[0].name == "complex_tool"
        assert calls[0].arguments == original.arguments
