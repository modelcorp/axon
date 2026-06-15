"""Tests for axon.tools.parsers.glm_parser module."""

import json

from axon.tools.parsers.glm_parser import GlmToolCallParser
from axon.tools.types import ToolCall


class TestGlmToolCallParserParse:
    def setup_method(self):
        self.parser = GlmToolCallParser()

    # -- single tool call ------------------------------------------------------

    def test_parse_single_call(self):
        response = "<tool_call>get_weather\n<arg_key>city</arg_key>\n<arg_value>San Francisco</arg_value>\n</tool_call>"
        calls, remaining = self.parser.parse(response)
        assert len(calls) == 1
        assert calls[0].name == "get_weather"
        assert calls[0].arguments == {"city": "San Francisco"}
        assert remaining == ""

    # -- multiple tool calls ---------------------------------------------------

    def test_parse_multiple_calls(self):
        response = (
            "<tool_call>search\n"
            "<arg_key>query</arg_key>\n"
            "<arg_value>python</arg_value>\n"
            "</tool_call>\n"
            "<tool_call>calculate\n"
            "<arg_key>expression</arg_key>\n"
            "<arg_value>2+2</arg_value>\n"
            "</tool_call>"
        )
        calls, remaining = self.parser.parse(response)
        assert len(calls) == 2
        assert calls[0].name == "search"
        assert calls[0].arguments == {"query": "python"}
        assert calls[1].name == "calculate"
        assert calls[1].arguments == {"expression": "2+2"}

    # -- remaining text --------------------------------------------------------

    def test_parse_with_remaining_text(self):
        response = (
            "Let me check the weather.\n"
            "<tool_call>get_weather\n"
            "<arg_key>city</arg_key>\n"
            "<arg_value>NYC</arg_value>\n"
            "</tool_call>\n"
            "I will get back to you."
        )
        calls, remaining = self.parser.parse(response)
        assert len(calls) == 1
        assert calls[0].name == "get_weather"
        assert "Let me check the weather." in remaining
        assert "I will get back to you." in remaining

    # -- JSON values in arg_value ----------------------------------------------

    def test_parse_json_values(self):
        response = (
            "<tool_call>config\n"
            "<arg_key>count</arg_key>\n"
            "<arg_value>42</arg_value>\n"
            "<arg_key>enabled</arg_key>\n"
            "<arg_value>true</arg_value>\n"
            "<arg_key>tags</arg_key>\n"
            '<arg_value>["a", "b"]</arg_value>\n'
            "</tool_call>"
        )
        calls, remaining = self.parser.parse(response)
        assert len(calls) == 1
        assert calls[0].arguments["count"] == 42
        assert calls[0].arguments["enabled"] is True
        assert calls[0].arguments["tags"] == ["a", "b"]

    def test_parse_json_dict_value(self):
        response = '<tool_call>update\n<arg_key>data</arg_key>\n<arg_value>{"x": 1, "y": 2}</arg_value>\n</tool_call>'
        calls, _ = self.parser.parse(response)
        assert calls[0].arguments["data"] == {"x": 1, "y": 2}

    # -- empty response --------------------------------------------------------

    def test_parse_empty_response(self):
        calls, remaining = self.parser.parse("")
        assert calls == []
        assert remaining == ""

    def test_parse_no_tool_calls(self):
        response = "Just some plain text without any tool calls."
        calls, remaining = self.parser.parse(response)
        assert calls == []
        assert remaining == response

    # -- multiple arguments ----------------------------------------------------

    def test_parse_multiple_arguments(self):
        response = (
            "<tool_call>send_email\n"
            "<arg_key>to</arg_key>\n"
            "<arg_value>alice@example.com</arg_value>\n"
            "<arg_key>subject</arg_key>\n"
            "<arg_value>Hello</arg_value>\n"
            "<arg_key>body</arg_key>\n"
            "<arg_value>How are you?</arg_value>\n"
            "</tool_call>"
        )
        calls, _ = self.parser.parse(response)
        assert len(calls) == 1
        assert calls[0].arguments == {
            "to": "alice@example.com",
            "subject": "Hello",
            "body": "How are you?",
        }

    # -- no arguments ----------------------------------------------------------

    def test_parse_no_arguments(self):
        response = "<tool_call>get_time\n</tool_call>"
        calls, _ = self.parser.parse(response)
        assert len(calls) == 1
        assert calls[0].name == "get_time"
        assert calls[0].arguments == {}


class TestGlmToolCallParserFormat:
    def setup_method(self):
        self.parser = GlmToolCallParser()

    # -- format_tool_call ------------------------------------------------------

    def test_format_tool_call_basic(self):
        tc = ToolCall(name="search", arguments={"query": "hello"})
        result = self.parser.format_tool_call(tc)
        assert result.startswith("<tool_call>")
        assert result.endswith("</tool_call>")
        assert "search" in result
        assert "<arg_key>query</arg_key>" in result
        assert "<arg_value>hello</arg_value>" in result

    def test_format_tool_call_json_value(self):
        tc = ToolCall(name="config", arguments={"items": [1, 2, 3]})
        result = self.parser.format_tool_call(tc)
        assert "<arg_value>[1, 2, 3]</arg_value>" in result

    def test_format_tool_call_empty_arguments(self):
        tc = ToolCall(name="noop", arguments={})
        result = self.parser.format_tool_call(tc)
        assert "<tool_call>noop\n</tool_call>" == result

    # -- roundtrip: format then parse ------------------------------------------

    def test_roundtrip_single(self):
        tc = ToolCall(name="weather", arguments={"city": "London", "units": "metric"})
        formatted = self.parser.format_tool_call(tc)
        calls, remaining = self.parser.parse(formatted)
        assert len(calls) == 1
        assert calls[0].name == tc.name
        assert calls[0].arguments == tc.arguments
        assert remaining == ""

    def test_roundtrip_json_values(self):
        tc = ToolCall(name="update", arguments={"count": 5, "active": True, "tags": ["a", "b"]})
        formatted = self.parser.format_tool_call(tc)
        calls, _ = self.parser.parse(formatted)
        assert len(calls) == 1
        assert calls[0].arguments == tc.arguments

    def test_roundtrip_multiple(self):
        tcs = [
            ToolCall(name="a", arguments={"x": 1}),
            ToolCall(name="b", arguments={"y": "two"}),
        ]
        formatted = self.parser.format_tool_calls(tcs)
        calls, _ = self.parser.parse(formatted)
        assert len(calls) == 2
        for orig, parsed in zip(tcs, calls, strict=False):
            assert parsed.name == orig.name
            assert parsed.arguments == orig.arguments

    # -- format_tool_result ----------------------------------------------------

    def test_format_tool_result_wrapping(self):
        result = self.parser.format_tool_result("the answer is 42", name="calculator")
        assert result == "<tool_response>\nthe answer is 42\n</tool_response>"

    def test_format_tool_result_empty_name(self):
        result = self.parser.format_tool_result("ok")
        assert result == "<tool_response>\nok\n</tool_response>"

    # -- format_tool_results ---------------------------------------------------

    def test_format_tool_results(self):
        from axon.tools.types import ToolResult

        results = [
            ToolResult(content="result1", name="tool_a"),
            ToolResult(content="result2", name="tool_b"),
        ]
        output = self.parser.format_tool_results(results)
        assert "<tool_response>\nresult1\n</tool_response>" in output
        assert "<tool_response>\nresult2\n</tool_response>" in output

    # -- get_tool_system_prompt ------------------------------------------------

    def test_get_tool_system_prompt_with_tools(self):
        tools_json = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather for a city",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                    },
                },
            }
        ]
        prompt = self.parser.get_tool_system_prompt(tools_json)
        assert "# Tools" in prompt
        assert "<tools>" in prompt
        assert "</tools>" in prompt
        assert "get_weather" in prompt
        assert "<tool_call>" in prompt
        assert "<arg_key>" in prompt
        assert "<arg_value>" in prompt

    def test_get_tool_system_prompt_empty_tools(self):
        prompt = self.parser.get_tool_system_prompt([])
        assert "# Tools" in prompt
        assert "<tools>" in prompt

    def test_get_tool_system_prompt_multiple_tools(self):
        tools_json = [
            {"name": "tool_a", "description": "does A"},
            {"name": "tool_b", "description": "does B"},
        ]
        prompt = self.parser.get_tool_system_prompt(tools_json)
        assert "tool_a" in prompt
        assert "tool_b" in prompt


# =============================================================================
# Hardened edge cases
# =============================================================================


class TestGlmParserEdgeCases:
    def setup_method(self):
        self.parser = GlmToolCallParser()

    def test_unicode_in_arguments(self):
        response = (
            "<tool_call>translate\n"
            "<arg_key>text</arg_key>\n"
            "<arg_value>\u4f60\u597d\u4e16\u754c \u00e9\u00e0\u00fc\u00f1</arg_value>\n"
            "</tool_call>"
        )
        calls, _ = self.parser.parse(response)
        assert calls[0].arguments["text"] == "\u4f60\u597d\u4e16\u754c \u00e9\u00e0\u00fc\u00f1"

    def test_multiline_arg_value(self):
        response = (
            "<tool_call>write_file\n"
            "<arg_key>content</arg_key>\n"
            "<arg_value>line 1\nline 2\nline 3</arg_value>\n"
            "</tool_call>"
        )
        calls, _ = self.parser.parse(response)
        assert "line 1\nline 2\nline 3" == calls[0].arguments["content"]

    def test_nested_xml_like_content_in_value(self):
        """Values that look like XML tags should not confuse the parser."""
        response = "<tool_call>render\n<arg_key>html</arg_key>\n<arg_value><div>hello</div></arg_value>\n</tool_call>"
        calls, _ = self.parser.parse(response)
        assert calls[0].arguments["html"] == "<div>hello</div>"

    def test_tool_call_with_whitespace_variations(self):
        response = (
            "  <tool_call>  search  \n"
            "  <arg_key>  q  </arg_key>  \n"
            "  <arg_value>  hello  </arg_value>  \n"
            "  </tool_call>  "
        )
        calls, _ = self.parser.parse(response)
        assert len(calls) == 1
        assert calls[0].name == "search"
        assert calls[0].arguments["q"] == "hello"

    def test_mismatched_keys_and_values_uses_zip(self):
        """More keys than values: zip stops at shorter."""
        response = (
            "<tool_call>partial\n<arg_key>a</arg_key>\n<arg_key>b</arg_key>\n<arg_value>1</arg_value>\n</tool_call>"
        )
        calls, _ = self.parser.parse(response)
        assert calls[0].arguments == {"a": 1}  # b has no value

    def test_interleaved_text_and_multiple_calls(self):
        response = (
            "I'll search first.\n"
            "<tool_call>search\n<arg_key>q</arg_key>\n<arg_value>test</arg_value>\n</tool_call>\n"
            "Now I'll calculate.\n"
            "<tool_call>calc\n<arg_key>expr</arg_key>\n<arg_value>1+1</arg_value>\n</tool_call>\n"
            "All done."
        )
        calls, remaining = self.parser.parse(response)
        assert len(calls) == 2
        assert "I'll search first." in remaining
        assert "Now I'll calculate." in remaining
        assert "All done." in remaining

    def test_roundtrip_unicode_and_special_chars(self):
        tc = ToolCall(
            name="write",
            arguments={
                "path": "/tmp/\u00e9\u00e0.txt",
                "data": 'quote: "hello" & <brackets>',
            },
        )
        formatted = self.parser.format_tool_call(tc)
        calls, _ = self.parser.parse(formatted)
        assert calls[0].arguments["path"] == tc.arguments["path"]
        assert calls[0].arguments["data"] == tc.arguments["data"]

    def test_deeply_nested_json_value(self):
        nested = {"a": {"b": {"c": [1, 2, {"d": True}]}}}
        response = (
            f"<tool_call>config\n<arg_key>settings</arg_key>\n<arg_value>{json.dumps(nested)}</arg_value>\n</tool_call>"
        )
        calls, _ = self.parser.parse(response)
        assert calls[0].arguments["settings"] == nested
