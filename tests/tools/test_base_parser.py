"""Tests for axon.tools.parsers.base_parser — registry and cross-parser integration."""

import logging

import pytest

from axon.tools.parsers.base_parser import (
    TOOL_CALL_PARSERS,
    ToolCallParser,
    get_tool_call_parser,
    register_parser,
)
from axon.tools.types import ToolCall

# =============================================================================
# Minimal stub for testing the ABC
# =============================================================================


class _StubParser(ToolCallParser):
    def parse(self, response):
        return [], response

    def format_tool_call(self, tool_call):
        return f"CALL:{tool_call.name}"

    def format_tool_result(self, content, name=""):
        return f"RESULT:{name}={content}"


# =============================================================================
# Registry
# =============================================================================


class TestRegistry:
    def test_register_and_retrieve(self):
        @register_parser("_test_reg")
        class P(_StubParser):
            pass

        assert TOOL_CALL_PARSERS["_test_reg"] is P
        instance = get_tool_call_parser("_test_reg")
        assert isinstance(instance, P)

    def test_unknown_name_raises(self):
        with pytest.raises(ValueError, match="Unknown"):
            get_tool_call_parser("_nonexistent_xyz")

    def test_kwargs_forwarded(self):
        @register_parser("_test_kw")
        class P(_StubParser):
            def __init__(self, **kw):
                self.val = kw.get("val")

        assert get_tool_call_parser("_test_kw", val=42).val == 42

    def test_overwrite_warns(self, caplog):
        @register_parser("_test_ow")
        class A(_StubParser):
            pass

        with caplog.at_level(logging.WARNING):

            @register_parser("_test_ow")
            class B(_StubParser):
                pass

        assert any("Overwriting" in r.message for r in caplog.records)
        assert TOOL_CALL_PARSERS["_test_ow"] is B

    def test_all_builtins_registered(self):
        for name in ("gemma4", "glm", "json", "openai_harmony", "qwen", "r1", "xml"):
            assert name in TOOL_CALL_PARSERS, f"Missing builtin parser: {name}"


# =============================================================================
# Cross-parser roundtrip integration
# =============================================================================


# These parsers support format→parse roundtrip for simple tool calls.
# Some parsers (r1, xml, json) may have different roundtrip characteristics,
# so we test the ones that are format-stable.
_ROUNDTRIP_PARSERS = ["gemma4", "glm", "openai_harmony", "qwen"]


class TestCrossParserRoundtrip:
    """Every registered parser that supports roundtrip should preserve tool call data."""

    @pytest.mark.parametrize("parser_name", _ROUNDTRIP_PARSERS)
    def test_simple_roundtrip(self, parser_name):
        parser = get_tool_call_parser(parser_name)
        tc = ToolCall(name="get_weather", arguments={"city": "NYC", "units": "celsius"})
        formatted = parser.format_tool_call(tc)
        calls, remaining = parser.parse(formatted)
        assert len(calls) >= 1, f"{parser_name}: expected at least 1 parsed call"
        assert calls[0].name == "get_weather"
        assert calls[0].arguments["city"] == "NYC"
        assert calls[0].arguments["units"] == "celsius"

    @pytest.mark.parametrize("parser_name", _ROUNDTRIP_PARSERS)
    def test_roundtrip_with_nested_json(self, parser_name):
        parser = get_tool_call_parser(parser_name)
        tc = ToolCall(
            name="config",
            arguments={
                "settings": {"nested": {"deep": [1, 2, 3]}},
                "flag": True,
            },
        )
        formatted = parser.format_tool_call(tc)
        calls, _ = parser.parse(formatted)
        assert len(calls) >= 1
        assert calls[0].arguments["settings"] == {"nested": {"deep": [1, 2, 3]}}
        assert calls[0].arguments["flag"] is True

    @pytest.mark.parametrize("parser_name", _ROUNDTRIP_PARSERS)
    def test_roundtrip_unicode(self, parser_name):
        parser = get_tool_call_parser(parser_name)
        tc = ToolCall(name="search", arguments={"query": "\u4f60\u597d\u4e16\u754c \u00e9\u00e0"})
        formatted = parser.format_tool_call(tc)
        calls, _ = parser.parse(formatted)
        assert len(calls) >= 1
        assert calls[0].arguments["query"] == "\u4f60\u597d\u4e16\u754c \u00e9\u00e0"

    @pytest.mark.parametrize("parser_name", _ROUNDTRIP_PARSERS)
    def test_empty_arguments_roundtrip(self, parser_name):
        parser = get_tool_call_parser(parser_name)
        tc = ToolCall(name="noop", arguments={})
        formatted = parser.format_tool_call(tc)
        calls, _ = parser.parse(formatted)
        assert len(calls) >= 1
        assert calls[0].name == "noop"
        assert calls[0].arguments == {}

    def test_all_parsers_handle_plain_text_gracefully(self):
        """No parser should crash on input without tool calls."""
        for name in TOOL_CALL_PARSERS:
            parser = get_tool_call_parser(name)
            calls, remaining = parser.parse("Just a normal response with no tool calls.")
            assert isinstance(calls, list)
            assert isinstance(remaining, str)

    def test_all_parsers_handle_empty_input(self):
        for name in TOOL_CALL_PARSERS:
            parser = get_tool_call_parser(name)
            calls, remaining = parser.parse("")
            assert calls == []
