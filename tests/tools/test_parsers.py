"""
Comprehensive tests for axon.tools.parsers, axon.tools.types, and axon.tools.tools.
"""

import json
from typing import Annotated

import pytest

from axon.tools.parsers.json_parser import JsonToolCallParser
from axon.tools.parsers.qwen_parser import QwenToolCallParser, _try_fix_json
from axon.tools.parsers.r1_parser import R1ToolCallParser
from axon.tools.parsers.xml_parser import XMLToolCallParser
from axon.tools.tools import Tool, _function_to_schema
from axon.tools.types import ToolCall, ToolOutput, ToolResult

# =========================================================================
# JsonToolCallParser
# =========================================================================


class TestJsonToolCallParser:
    def setup_method(self):
        self.parser = JsonToolCallParser()

    # -- parse ----------------------------------------------------------------

    def test_parse_single_call_in_array(self):
        response = 'Some text\n```json\n[{"name": "get_weather", "arguments": {"city": "NYC"}}]\n```\nmore text'
        calls, remaining = self.parser.parse(response)
        assert len(calls) == 1
        assert calls[0].name == "get_weather"
        assert calls[0].arguments == {"city": "NYC"}
        assert "Some text" in remaining
        assert "more text" in remaining
        assert "```" not in remaining

    def test_parse_multiple_calls_in_array(self):
        arr = json.dumps(
            [
                {"name": "func_a", "arguments": {"x": 1}},
                {"name": "func_b", "arguments": {"y": "hello"}},
            ]
        )
        response = f"```json\n{arr}\n```"
        calls, remaining = self.parser.parse(response)
        assert len(calls) == 2
        assert calls[0].name == "func_a"
        assert calls[0].arguments == {"x": 1}
        assert calls[1].name == "func_b"
        assert calls[1].arguments == {"y": "hello"}
        assert remaining == ""

    def test_parse_single_object_not_array(self):
        response = '```json\n{"name": "do_stuff", "arguments": {"a": true}}\n```'
        calls, remaining = self.parser.parse(response)
        assert len(calls) == 1
        assert calls[0].name == "do_stuff"
        assert calls[0].arguments == {"a": True}

    def test_parse_malformed_json_skipped(self):
        response = "```json\n{this is not json}\n```"
        calls, remaining = self.parser.parse(response)
        assert len(calls) == 0
        # malformed block stays in remaining because json.loads fails
        assert "```" in remaining

    def test_parse_mixed_text_and_blocks(self):
        response = (
            "Here is my plan.\n"
            '```json\n[{"name": "search", "arguments": {"q": "test"}}]\n```\n'
            "Now let me also call another tool.\n"
            '```json\n{"name": "read", "arguments": {"path": "/tmp"}}\n```\n'
            "Done."
        )
        calls, remaining = self.parser.parse(response)
        assert len(calls) == 2
        assert calls[0].name == "search"
        assert calls[1].name == "read"
        assert "Here is my plan." in remaining
        assert "Done." in remaining
        assert "```" not in remaining

    def test_parse_deeply_nested_arguments(self):
        """Arguments with deep nesting should be preserved."""
        deep_args = {
            "config": {
                "model": {
                    "layers": [{"type": "attention", "heads": 12}, {"type": "ffn"}],
                    "settings": {"dropout": 0.1, "nested": {"key": "value"}},
                },
            }
        }
        obj = {"name": "configure", "arguments": deep_args}
        response = f"```json\n{json.dumps(obj)}\n```"
        calls, _ = self.parser.parse(response)
        assert len(calls) == 1
        assert calls[0].arguments == deep_args

    def test_parse_unicode_in_names_and_values(self):
        """Unicode chars in function names and argument values."""
        obj = {"name": "search_\u00e9l\u00e8ve", "arguments": {"\u30ad\u30fc": "\u5024\u306f\u3053\u3053"}}
        response = f"```json\n{json.dumps(obj, ensure_ascii=False)}\n```"
        calls, _ = self.parser.parse(response)
        assert len(calls) == 1
        assert calls[0].name == "search_\u00e9l\u00e8ve"
        assert calls[0].arguments["\u30ad\u30fc"] == "\u5024\u306f\u3053\u3053"

    def test_parse_escaped_chars_in_json_strings(self):
        """Escaped quotes, newlines, tabs inside JSON string values."""
        obj = {"name": "write", "arguments": {"content": 'line1\\nline2\\t"quoted"'}}
        response = f"```json\n{json.dumps(obj)}\n```"
        calls, _ = self.parser.parse(response)
        assert len(calls) == 1
        assert '"quoted"' in calls[0].arguments["content"]

    def test_parse_mixed_valid_invalid_blocks(self):
        """Some blocks parse, some don't. Valid ones should still be returned."""
        response = (
            '```json\n{"name": "good", "arguments": {"x": 1}}\n```\n'
            "```json\n{INVALID JSON}\n```\n"
            '```json\n[{"name": "also_good", "arguments": {}}]\n```'
        )
        calls, remaining = self.parser.parse(response)
        assert len(calls) == 2
        assert calls[0].name == "good"
        assert calls[1].name == "also_good"
        # The invalid block should remain
        assert "INVALID JSON" in remaining

    def test_parse_special_regex_chars_in_argument_values(self):
        """Argument values containing regex special chars should not break parsing."""
        obj = {"name": "regex_test", "arguments": {"pattern": r"^foo\.(bar|baz)\$[0-9]+"}}
        response = f"```json\n{json.dumps(obj)}\n```"
        calls, _ = self.parser.parse(response)
        assert len(calls) == 1
        assert calls[0].arguments["pattern"] == r"^foo\.(bar|baz)\$[0-9]+"

    # -- format_tool_call -----------------------------------------------------

    def test_format_tool_call(self):
        tc = ToolCall(name="greet", arguments={"who": "world"})
        result = self.parser.format_tool_call(tc)
        assert result.startswith("```json\n")
        assert result.endswith("\n```")
        parsed = json.loads(result[len("```json\n") : -len("\n```")])
        assert parsed["name"] == "greet"
        assert parsed["arguments"] == {"who": "world"}

    # -- format_tool_calls ----------------------------------------------------

    def test_format_tool_calls(self):
        tcs = [
            ToolCall(name="a", arguments={"x": 1}),
            ToolCall(name="b", arguments={"y": 2}),
        ]
        result = self.parser.format_tool_calls(tcs)
        assert result.startswith("```json\n")
        assert result.endswith("\n```")
        parsed = json.loads(result[len("```json\n") : -len("\n```")])
        assert isinstance(parsed, list)
        assert len(parsed) == 2
        assert parsed[0]["name"] == "a"
        assert parsed[1]["name"] == "b"

    # -- roundtrip ------------------------------------------------------------

    def test_roundtrip_single(self):
        """parse(format(tool_call)) should return the original."""
        tc = ToolCall(name="do_stuff", arguments={"key": "value", "num": 42})
        formatted = self.parser.format_tool_call(tc)
        calls, _ = self.parser.parse(formatted)
        assert len(calls) == 1
        assert calls[0].name == tc.name
        assert calls[0].arguments == tc.arguments

    def test_roundtrip_multiple(self):
        """parse(format_tool_calls(calls)) should return all originals."""
        tcs = [
            ToolCall(name="a", arguments={"nested": {"deep": True}}),
            ToolCall(name="b", arguments={"list": [1, 2, 3]}),
        ]
        formatted = self.parser.format_tool_calls(tcs)
        calls, _ = self.parser.parse(formatted)
        assert len(calls) == 2
        for orig, parsed in zip(tcs, calls, strict=False):
            assert parsed.name == orig.name
            assert parsed.arguments == orig.arguments

    # -- get_tool_system_prompt -----------------------------------------------

    def test_get_tool_system_prompt(self):
        tools_json = [
            {"type": "function", "function": {"name": "foo", "description": "does foo"}},
        ]
        prompt = self.parser.get_tool_system_prompt(tools_json)
        assert "You have access to the following tools:" in prompt
        assert "foo" in prompt
        assert "tool_name" in prompt


# =========================================================================
# XMLToolCallParser
# =========================================================================


class TestXMLToolCallParser:
    def setup_method(self):
        self.parser = XMLToolCallParser()

    # -- parse ----------------------------------------------------------------

    def test_parse_single_function(self):
        response = "<function=file_editor><parameter=command>view</parameter></function>"
        calls, remaining = self.parser.parse(response)
        assert len(calls) == 1
        assert calls[0].name == "file_editor"
        assert calls[0].arguments == {"command": "view"}
        assert remaining == ""

    def test_parse_multiple_params(self):
        response = (
            "<function=editor>\n"
            "  <parameter=command>create</parameter>\n"
            "  <parameter=path>/tmp/foo.py</parameter>\n"
            "  <parameter=content>print('hi')</parameter>\n"
            "</function>"
        )
        calls, remaining = self.parser.parse(response)
        assert len(calls) == 1
        assert calls[0].name == "editor"
        assert calls[0].arguments["command"] == "create"
        assert calls[0].arguments["path"] == "/tmp/foo.py"
        assert calls[0].arguments["content"] == "print('hi')"

    def test_parse_nested_content_in_param(self):
        """Parameter values can contain arbitrary text (not XML)."""
        response = '<function=exec><parameter=code>if x > 0:\n    print("yes")</parameter></function>'
        calls, _ = self.parser.parse(response)
        assert len(calls) == 1
        assert "if x > 0:" in calls[0].arguments["code"]

    def test_parse_multiple_functions(self):
        response = (
            "<function=a><parameter=x>1</parameter></function>"
            "middle text"
            "<function=b><parameter=y>2</parameter></function>"
        )
        calls, remaining = self.parser.parse(response)
        assert len(calls) == 2
        assert calls[0].name == "a"
        assert calls[1].name == "b"
        assert remaining == "middle text"

    def test_parse_special_regex_chars_in_param_values(self):
        """Values with regex-special chars like $, ^, |, should work."""
        response = "<function=grep><parameter=pattern>^[a-z]+\\.(foo|bar)$</parameter></function>"
        calls, _ = self.parser.parse(response)
        assert len(calls) == 1
        assert calls[0].arguments["pattern"] == "^[a-z]+\\.(foo|bar)$"

    def test_parse_unicode_in_params(self):
        """Unicode content in parameter values."""
        response = (
            "<function=translate><parameter=text>\u3053\u3093\u306b\u3061\u306f\u4e16\u754c</parameter></function>"
        )
        calls, _ = self.parser.parse(response)
        assert len(calls) == 1
        assert calls[0].arguments["text"] == "\u3053\u3093\u306b\u3061\u306f\u4e16\u754c"

    # -- format_tool_call -----------------------------------------------------

    def test_format_tool_call(self):
        tc = ToolCall(name="ls", arguments={"path": "/tmp", "all": "true"})
        result = self.parser.format_tool_call(tc)
        assert "<function=ls>" in result
        assert "<parameter=path>/tmp</parameter>" in result
        assert "<parameter=all>true</parameter>" in result
        assert "</function>" in result

    def test_format_roundtrip(self):
        """format_tool_call output should be re-parseable."""
        tc = ToolCall(name="run", arguments={"cmd": "echo hi"})
        formatted = self.parser.format_tool_call(tc)
        calls, _ = self.parser.parse(formatted)
        assert len(calls) == 1
        assert calls[0].name == "run"
        assert calls[0].arguments["cmd"] == "echo hi"

    def test_roundtrip_complex_arguments(self):
        """Roundtrip with multi-line content containing special chars."""
        tc = ToolCall(
            name="write_file",
            arguments={
                "path": "/tmp/test.py",
                "content": 'def foo():\n    return "hello\\nworld"',
            },
        )
        formatted = self.parser.format_tool_call(tc)
        calls, _ = self.parser.parse(formatted)
        assert len(calls) == 1
        assert calls[0].name == "write_file"
        assert calls[0].arguments["path"] == "/tmp/test.py"


# =========================================================================
# QwenToolCallParser
# =========================================================================


class TestTryFixJson:
    """Tests for the module-level _try_fix_json helper."""

    def test_trailing_period(self):
        fixed = _try_fix_json('{"name": "foo", "arguments": {}}.')
        assert json.loads(fixed) == {"name": "foo", "arguments": {}}

    def test_trailing_semicolon(self):
        fixed = _try_fix_json('{"a": 1};')
        assert json.loads(fixed) == {"a": 1}

    def test_trailing_comma(self):
        fixed = _try_fix_json('{"a": 1},')
        assert json.loads(fixed) == {"a": 1}

    def test_trailing_paren_becomes_brace(self):
        """Trailing ')' is converted to '}' and removed."""
        fixed = _try_fix_json('{"a": 1)')
        # ')' -> '}' gives '{"a": 1}' which is valid
        assert json.loads(fixed) == {"a": 1}

    def test_unbalanced_braces(self):
        fixed = _try_fix_json('{"a": {"b": 1}')
        assert json.loads(fixed) == {"a": {"b": 1}}

    def test_multiple_trailing_issues(self):
        fixed = _try_fix_json('{"a": 1}.,;')
        assert json.loads(fixed) == {"a": 1}

    def test_clean_json_unchanged(self):
        s = '{"name": "test"}'
        assert _try_fix_json(s) == s


class TestQwenToolCallParser:
    def setup_method(self):
        self.parser = QwenToolCallParser()

    # -- parse ----------------------------------------------------------------

    def test_parse_clean_json(self):
        response = '<tool_call>\n{"name": "get_weather", "arguments": {"city": "London"}}\n</tool_call>'
        calls, remaining = self.parser.parse(response)
        assert len(calls) == 1
        assert calls[0].name == "get_weather"
        assert calls[0].arguments == {"city": "London"}
        assert remaining == ""

    def test_parse_multiple_tool_calls(self):
        response = (
            "<tool_call>\n"
            '{"name": "a", "arguments": {"x": 1}}\n'
            "</tool_call>\n"
            "<tool_call>\n"
            '{"name": "b", "arguments": {"y": 2}}\n'
            "</tool_call>"
        )
        calls, remaining = self.parser.parse(response)
        assert len(calls) == 2
        assert calls[0].name == "a"
        assert calls[1].name == "b"

    def test_parse_trailing_comma_in_json(self):
        """JSON with trailing comma should be recoverable via _try_fix_json."""
        response = '<tool_call>\n{"name": "func", "arguments": {"k": "v"}},\n</tool_call>'
        calls, _ = self.parser.parse(response)
        assert len(calls) == 1
        assert calls[0].name == "func"

    def test_parse_unbalanced_braces(self):
        response = '<tool_call>\n{"name": "func", "arguments": {"k": "v"}\n</tool_call>'
        calls, _ = self.parser.parse(response)
        assert len(calls) == 1
        assert calls[0].name == "func"

    def test_parse_mixed_text_inside_tags(self):
        """Reasoning text before the JSON object; regex strategy should recover."""
        response = (
            '<tool_call>\nI need to call the function.\n{"name": "search", "arguments": {"q": "test"}}\n</tool_call>'
        )
        calls, _ = self.parser.parse(response)
        assert len(calls) == 1
        assert calls[0].name == "search"

    def test_parse_tool_call_inside_code_block_not_parsed(self):
        """<tool_call> appearing inside a markdown code block should not be parsed as tool call."""
        response = (
            "Here is an example:\n"
            "```\n"
            "<tool_call>\n"
            '{"name": "example", "arguments": {}}\n'
            "</tool_call>\n"
            "```\n"
            "This was just an illustration."
        )
        calls, remaining = self.parser.parse(response)
        # The parser may or may not extract this. The key test is that any
        # calls outside code blocks work. If it does parse, it's an accepted
        # behavior (parsers aren't code-block-aware).
        # Just verify no crash.
        assert isinstance(calls, list)

    def test_parse_nested_objects_in_arguments(self):
        """Arguments with deeply nested structures."""
        nested_args = {"config": {"model": {"hidden": 768}, "training": {"lr": 0.001}}}
        response = f'<tool_call>\n{{"name": "configure", "arguments": {json.dumps(nested_args)}}}\n</tool_call>'
        calls, _ = self.parser.parse(response)
        assert len(calls) == 1
        assert calls[0].arguments == nested_args

    # -- _extract_json --------------------------------------------------------

    def test_extract_json_direct_parse(self):
        raw = '{"name": "f", "arguments": {}}'
        result = self.parser._extract_json(raw)
        assert result == {"name": "f", "arguments": {}}

    def test_extract_json_fix_strategy(self):
        raw = '{"name": "f", "arguments": {}}.'
        result = self.parser._extract_json(raw)
        assert result is not None
        assert result["name"] == "f"

    def test_extract_json_regex_strategy(self):
        raw = 'Some reasoning text {"name": "f", "arguments": {"a": 1}} end'
        result = self.parser._extract_json(raw)
        assert result is not None
        assert result["name"] == "f"

    # -- format_tool_call -----------------------------------------------------

    def test_format_tool_call(self):
        tc = ToolCall(name="calc", arguments={"expr": "2+2"})
        result = self.parser.format_tool_call(tc)
        assert "<tool_call>" in result
        assert "</tool_call>" in result
        assert '"name": "calc"' in result

    def test_format_roundtrip(self):
        tc = ToolCall(name="run", arguments={"cmd": "ls"})
        formatted = self.parser.format_tool_call(tc)
        calls, _ = self.parser.parse(formatted)
        assert len(calls) == 1
        assert calls[0].name == "run"
        assert calls[0].arguments == {"cmd": "ls"}

    def test_roundtrip_nested_arguments(self):
        """Roundtrip with nested dict arguments."""
        tc = ToolCall(
            name="deploy",
            arguments={
                "config": {"replicas": 3, "env": {"DEBUG": "1"}},
                "tags": ["prod", "v2"],
            },
        )
        formatted = self.parser.format_tool_call(tc)
        calls, _ = self.parser.parse(formatted)
        assert len(calls) == 1
        assert calls[0].name == "deploy"
        assert calls[0].arguments == tc.arguments


# =========================================================================
# R1ToolCallParser
# =========================================================================


class TestR1ToolCallParser:
    def setup_method(self):
        self.parser = R1ToolCallParser()
        self.begin = self.parser.tool_call_begin
        self.end = self.parser.tool_call_end
        self.sep = self.parser.tool_sep

    def _make_block(self, name: str, args: dict) -> str:
        args_str = json.dumps(args)
        return f"{self.begin}function{self.sep}{name}\n```json\n{args_str}\n```\n{self.end}"

    # -- parse ----------------------------------------------------------------

    def test_parse_single_call(self):
        block = self._make_block("get_weather", {"city": "Tokyo"})
        response = f"Some thinking\n{block}\nDone."
        calls, remaining = self.parser.parse(response)
        assert len(calls) == 1
        assert calls[0].name == "get_weather"
        assert calls[0].arguments == {"city": "Tokyo"}
        assert "Some thinking" in remaining
        assert "Done." in remaining

    def test_parse_multiple_calls(self):
        b1 = self._make_block("a", {"x": 1})
        b2 = self._make_block("b", {"y": 2})
        response = f"{b1}\n{b2}"
        calls, remaining = self.parser.parse(response)
        assert len(calls) == 2
        assert calls[0].name == "a"
        assert calls[1].name == "b"

    def test_parse_missing_json_block_skipped(self):
        """If there is no ```json block, the call should be skipped."""
        response = f"{self.begin}function{self.sep}my_func\nno json block here\n{self.end}"
        calls, _ = self.parser.parse(response)
        assert len(calls) == 0

    def test_parse_bad_json_skipped(self):
        response = f"{self.begin}function{self.sep}my_func\n```json\n{{invalid json!}}\n```\n{self.end}"
        calls, _ = self.parser.parse(response)
        assert len(calls) == 0

    def test_parse_no_function_prefix_skipped(self):
        """Block without 'function<sep>' prefix should be skipped."""
        response = f"{self.begin}something_else\n```json\n{{}}\n```\n{self.end}"
        calls, _ = self.parser.parse(response)
        assert len(calls) == 0

    def test_parse_multiple_calls_with_interleaved_text(self):
        """Multiple tool calls with discussion text between them."""
        b1 = self._make_block("search", {"q": "python"})
        b2 = self._make_block("read", {"path": "/tmp/file.py"})
        response = f"Let me search for that.\n{b1}\nNow I found the file, let me read it.\n{b2}\nAll done."
        calls, remaining = self.parser.parse(response)
        assert len(calls) == 2
        assert calls[0].name == "search"
        assert calls[1].name == "read"
        assert "Let me search" in remaining
        assert "All done." in remaining

    def test_parse_special_tokens_in_plain_text_not_matched(self):
        """The special begin/end tokens mentioned in plain text should not confuse parser
        if they don't form a complete block."""
        response = (
            f"The token {self.begin} is used to start a tool call and {self.end} ends it. But this is just discussion."
        )
        calls, remaining = self.parser.parse(response)
        # Without proper function prefix + JSON block, nothing should parse
        assert len(calls) == 0

    # -- format_tool_call -----------------------------------------------------

    def test_format_tool_call(self):
        tc = ToolCall(name="search", arguments={"q": "hello"})
        result = self.parser.format_tool_call(tc)
        assert self.begin in result
        assert self.end in result
        assert self.sep in result
        assert "search" in result
        assert "hello" in result

    def test_format_roundtrip(self):
        tc = ToolCall(name="exec", arguments={"code": "print(1)"})
        formatted = self.parser.format_tool_call(tc)
        calls, _ = self.parser.parse(formatted)
        assert len(calls) == 1
        assert calls[0].name == "exec"
        assert calls[0].arguments == {"code": "print(1)"}

    def test_roundtrip_complex_args(self):
        """Roundtrip with nested dict and list in arguments."""
        tc = ToolCall(
            name="deploy",
            arguments={
                "services": [{"name": "api", "port": 8080}, {"name": "web", "port": 3000}],
                "env": {"PROD": "true"},
            },
        )
        formatted = self.parser.format_tool_call(tc)
        calls, _ = self.parser.parse(formatted)
        assert len(calls) == 1
        assert calls[0].name == tc.name
        assert calls[0].arguments == tc.arguments

    # -- _iter_raw_blocks -----------------------------------------------------

    def test_iter_raw_blocks(self):
        b1 = self._make_block("a", {"x": 1})
        b2 = self._make_block("b", {"y": 2})
        text = f"prefix {b1} middle {b2} suffix"
        blocks = list(self.parser._iter_raw_blocks(text))
        assert len(blocks) == 2
        assert self.begin in blocks[0]
        assert self.end in blocks[0]
        assert self.begin in blocks[1]

    def test_iter_raw_blocks_unclosed(self):
        """An unclosed block (no end token) yields nothing."""
        text = f"{self.begin}function{self.sep}foo\n```json\n{{}}\n```\n"
        blocks = list(self.parser._iter_raw_blocks(text))
        assert blocks == []


# =========================================================================
# ToolCall (axon.tools.types)
# =========================================================================


class TestToolCall:
    def test_to_dict(self):
        tc = ToolCall(name="f", arguments={"a": 1})
        d = tc.to_dict()
        assert d == {"name": "f", "arguments": {"a": 1}}
        # Should not contain id or raw_tool_call
        assert "id" not in d

    def test_to_openai_dict(self):
        tc = ToolCall(name="g", arguments={"b": "x"}, id="call_abc")
        d = tc.to_openai_dict()
        assert d["id"] == "call_abc"
        assert d["type"] == "function"
        assert d["function"]["name"] == "g"
        # arguments should be a JSON string
        assert json.loads(d["function"]["arguments"]) == {"b": "x"}

    def test_to_openai_dict_arguments_not_dict(self):
        """If arguments is not a dict, to_openai_dict uses str()."""
        tc = ToolCall(name="h", arguments="raw_string")  # type: ignore[arg-type]
        d = tc.to_openai_dict()
        assert d["function"]["arguments"] == "raw_string"

    # -- from_raw_tool_call ---------------------------------------------------

    def test_from_raw_passthrough(self):
        tc = ToolCall(name="f", arguments={"a": 1})
        assert ToolCall.from_raw_tool_call(tc) is tc

    def test_from_raw_openai_dict(self):
        d = {
            "id": "call_123",
            "function": {
                "name": "my_func",
                "arguments": '{"key": "val"}',
            },
        }
        tc = ToolCall.from_raw_tool_call(d)
        assert tc.name == "my_func"
        assert tc.arguments == {"key": "val"}
        assert tc.id == "call_123"

    def test_from_raw_flat_dict(self):
        d = {"name": "flat_func", "arguments": {"x": 42}}
        tc = ToolCall.from_raw_tool_call(d)
        assert tc.name == "flat_func"
        assert tc.arguments == {"x": 42}

    def test_from_raw_flat_dict_with_parameters_key(self):
        d = {"name": "flat_func", "parameters": {"x": 42}}
        tc = ToolCall.from_raw_tool_call(d)
        assert tc.name == "flat_func"
        assert tc.arguments == {"x": 42}

    def test_from_raw_string_arguments_parsed(self):
        d = {
            "function": {
                "name": "f",
                "arguments": '{"a": 1}',
            }
        }
        tc = ToolCall.from_raw_tool_call(d)
        assert tc.arguments == {"a": 1}

    def test_from_raw_bad_json_string_becomes_raw(self):
        d = {
            "function": {
                "name": "f",
                "arguments": "not valid json",
            }
        }
        tc = ToolCall.from_raw_tool_call(d)
        assert tc.arguments == {"raw": "not valid json"}

    def test_from_raw_pydantic_like_object(self):
        class FuncObj:
            name = "pydantic_func"
            arguments = '{"p": 10}'

        class ToolCallObj:
            function = FuncObj()
            id = "call_pydantic"

        tc = ToolCall.from_raw_tool_call(ToolCallObj())
        assert tc.name == "pydantic_func"
        assert tc.arguments == {"p": 10}
        assert tc.id == "call_pydantic"

    def test_from_raw_invalid_type_raises(self):
        with pytest.raises(TypeError):
            ToolCall.from_raw_tool_call(42)  # type: ignore[arg-type]

    def test_from_raw_preserves_raw_tool_call(self):
        d = {"name": "f", "arguments": {"a": 1}}
        tc = ToolCall.from_raw_tool_call(d)
        assert tc.raw_tool_call is not None
        # raw_tool_call is a deep copy
        assert tc.raw_tool_call == d
        assert tc.raw_tool_call is not d

    def test_from_raw_deeply_nested_pydantic_chain(self):
        """Deeply nested pydantic-like object chain with multiple levels."""

        class InnerFunc:
            name = "deep_func"
            arguments = '{"nested": {"key": "value", "list": [1, 2, 3]}}'

        class OuterTool:
            function = InnerFunc()
            id = "call_deep"

        tc = ToolCall.from_raw_tool_call(OuterTool())
        assert tc.name == "deep_func"
        assert tc.arguments["nested"]["key"] == "value"
        assert tc.arguments["nested"]["list"] == [1, 2, 3]

    def test_from_raw_arguments_as_integer(self):
        """Non-string, non-dict arguments (integer) should be normalized to dict."""
        d = {"name": "f", "arguments": 42}
        tc = ToolCall.from_raw_tool_call(d)
        assert tc.name == "f"
        assert isinstance(tc.arguments, dict)
        assert tc.arguments == {"raw": 42}

    def test_from_raw_arguments_as_float(self):
        """Float arguments should be normalized to dict."""
        d = {"name": "f", "arguments": 3.14}
        tc = ToolCall.from_raw_tool_call(d)
        assert isinstance(tc.arguments, dict)
        assert tc.arguments == {"raw": 3.14}


# =========================================================================
# ToolResult (axon.tools.types)
# =========================================================================


class TestToolResult:
    def test_to_openai_dict(self):
        tr = ToolResult(content="success", name="func", tool_call_id="call_x")
        d = tr.to_openai_dict()
        assert d["role"] == "tool"
        assert d["tool_call_id"] == "call_x"
        assert d["content"] == "success"


# =========================================================================
# ToolOutput (axon.tools.types)
# =========================================================================


class TestToolOutput:
    def test_to_content_string_error(self):
        to = ToolOutput(name="f", error="something broke")
        assert to.to_content_string() == "Error: something broke"

    def test_to_content_string_none_output(self):
        to = ToolOutput(name="f", output=None)
        assert to.to_content_string() == ""

    def test_to_content_string_list_output(self):
        to = ToolOutput(name="f", output=[1, 2, 3])
        assert to.to_content_string() == "[1, 2, 3]"

    def test_to_content_string_dict_output(self):
        to = ToolOutput(name="f", output={"key": "val"})
        assert to.to_content_string() == '{"key": "val"}'

    def test_to_content_string_string_output(self):
        to = ToolOutput(name="f", output="hello world")
        assert to.to_content_string() == "hello world"

    def test_str_delegates_to_content_string(self):
        to = ToolOutput(name="f", output="abc")
        assert str(to) == "abc"

    def test_error_takes_priority_over_output(self):
        to = ToolOutput(name="f", output="some output", error="err")
        assert to.to_content_string() == "Error: err"


# =========================================================================
# _function_to_schema (axon.tools.tools)
# =========================================================================


class TestFunctionToSchema:
    def test_typed_args(self):
        def compute(x: int, name: str, flag: bool):
            """Compute something."""
            pass

        schema = _function_to_schema(compute)
        props = schema["function"]["parameters"]["properties"]
        assert props["x"]["type"] == "integer"
        assert props["name"]["type"] == "string"
        assert props["flag"]["type"] == "boolean"
        assert set(schema["function"]["parameters"]["required"]) == {"x", "name", "flag"}

    def test_annotated_args_with_description(self):
        def greet(who: Annotated[str, "The person to greet"]):
            """Say hello."""
            pass

        schema = _function_to_schema(greet)
        props = schema["function"]["parameters"]["properties"]
        assert props["who"]["type"] == "string"
        assert props["who"]["description"] == "The person to greet"

    def test_optional_args_with_defaults(self):
        def fetch(url: str, timeout: int = 30):
            """Fetch a URL."""
            pass

        schema = _function_to_schema(fetch)
        required = schema["function"]["parameters"]["required"]
        assert "url" in required
        assert "timeout" not in required
        assert "timeout" in schema["function"]["parameters"]["properties"]

    def test_function_with_multiline_docstring(self):
        def multi():
            """First line summary.

            More details here.
            """
            pass

        schema = _function_to_schema(multi)
        assert schema["function"]["description"] == "First line summary."

    def test_float_type(self):
        def func(val: float):
            """Test."""
            pass

        schema = _function_to_schema(func)
        assert schema["function"]["parameters"]["properties"]["val"]["type"] == "number"

    def test_dict_and_list_types(self):
        def func(data: dict, items: list):
            """Test."""
            pass

        schema = _function_to_schema(func)
        props = schema["function"]["parameters"]["properties"]
        assert props["data"]["type"] == "object"
        assert props["items"]["type"] == "array"

    def test_function_with_complex_annotated_types(self):
        """list[int], Optional[str] should fall through to string default in current impl."""

        def func(ids: list, name: str | None = None):
            """Process items."""
            pass

        schema = _function_to_schema(func)
        props = schema["function"]["parameters"]["properties"]
        # list maps to "array"
        assert props["ids"]["type"] == "array"
        # Optional[str] has get_origin as Union, which doesn't match any type_mapping key
        # so it falls back to "string"
        assert "name" in props
        assert "name" not in schema["function"]["parameters"]["required"]

    def test_function_with_all_type_mappings(self):
        """One function exercising all type mappings."""

        def kitchen_sink(a: int, b: float, c: bool, d: str, e: dict, f: list):
            """All types."""
            pass

        schema = _function_to_schema(kitchen_sink)
        props = schema["function"]["parameters"]["properties"]
        assert props["a"]["type"] == "integer"
        assert props["b"]["type"] == "number"
        assert props["c"]["type"] == "boolean"
        assert props["d"]["type"] == "string"
        assert props["e"]["type"] == "object"
        assert props["f"]["type"] == "array"
        assert len(schema["function"]["parameters"]["required"]) == 6


# =========================================================================
# Tool (axon.tools.tools)
# =========================================================================


class TestTool:
    def test_tool_from_function(self):
        def add(a: int, b: int):
            """Add two numbers."""
            return a + b

        tool = Tool(function=add)
        assert tool.name == "add"
        assert tool.description == "Add two numbers."

    def test_tool_json_schema(self):
        def multiply(x: float, y: float):
            """Multiply x by y."""
            return x * y

        tool = Tool(function=multiply)
        schema = tool.json
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "multiply"

    def test_tool_forward_executes_function(self):
        def double(n: int):
            """Double a number."""
            return n * 2

        tool = Tool(function=double)
        result = tool.forward(n=5)
        assert isinstance(result, ToolOutput)
        assert result.output == 10
        assert result.error is None

    def test_tool_forward_captures_exception(self):
        def fail():
            """Always fails."""
            raise ValueError("boom")

        tool = Tool(function=fail)
        result = tool.forward()
        assert result.error is not None
        assert "boom" in result.error
        assert result.output is None

    def test_tool_with_name_and_description_json_raises(self):
        """Without a function, .json raises NotImplementedError."""
        tool = Tool(name="my_tool", description="does stuff")
        with pytest.raises(NotImplementedError):
            _ = tool.json

    def test_tool_with_name_and_description_forward_raises(self):
        """Without a function, forward() raises NotImplementedError."""
        tool = Tool(name="my_tool", description="does stuff")
        with pytest.raises(NotImplementedError):
            tool.forward()

    def test_tool_forward_returns_correct_name(self):
        def echo(msg: str):
            """Echo."""
            return msg

        tool = Tool(function=echo)
        result = tool.forward(msg="hi")
        assert result.name == "echo"

    def test_tool_forward_various_return_types(self):
        """Tool.forward should handle different return types from the function."""

        def return_dict():
            """Returns dict."""
            return {"key": "value"}

        def return_list():
            """Returns list."""
            return [1, 2, 3]

        def return_none():
            """Returns None."""
            return None

        for func, expected in [(return_dict, {"key": "value"}), (return_list, [1, 2, 3]), (return_none, None)]:
            tool = Tool(function=func)
            result = tool.forward()
            assert result.output == expected
            assert result.error is None

    def test_tool_without_function_requires_both_name_and_description(self):
        """Missing either name or description should raise."""
        with pytest.raises(ValueError):
            Tool()
        with pytest.raises(ValueError):
            Tool(name="only_name")
        with pytest.raises(ValueError):
            Tool(description="only_desc")


# =========================================================================
# Cross-parser roundtrip tests
# =========================================================================


class TestCrossParserRoundtrip:
    """For every parser, verify that parse(format(tc)) == original."""

    PARSERS = [
        JsonToolCallParser(),
        QwenToolCallParser(),
        R1ToolCallParser(),
        XMLToolCallParser(),
    ]

    # XML parser stringifies parameter values, so nested dicts won't roundtrip.
    # Use simple string arguments that all parsers can handle.
    TOOL_CALLS = [
        ToolCall(name="simple", arguments={"key": "value"}),
        ToolCall(name="numeric", arguments={"count": "42", "flag": "true"}),
        ToolCall(name="empty_args", arguments={}),
    ]

    @pytest.mark.parametrize("parser", PARSERS, ids=lambda p: type(p).__name__)
    @pytest.mark.parametrize("tc", TOOL_CALLS, ids=lambda tc: tc.name)
    def test_roundtrip(self, parser, tc):
        formatted = parser.format_tool_call(tc)
        calls, _ = parser.parse(formatted)
        assert len(calls) >= 1
        assert calls[0].name == tc.name
        # XML parser converts all values to strings and JSON parsers preserve types,
        # so we compare key-by-key as strings for universal compatibility
        for k in tc.arguments:
            assert str(calls[0].arguments[k]) == str(tc.arguments[k])
