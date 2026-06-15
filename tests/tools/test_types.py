"""Tests for axon.tools.types module."""

import json

import pytest

from axon.tools.types import ToolCall, ToolOutput, ToolResult

# =============================================================================
# ToolCall
# =============================================================================


class TestToolCallFromRawToolCall:
    def test_with_toolcall_instance_returns_same_object(self):
        original = ToolCall(name="foo", arguments={"a": 1})
        result = ToolCall.from_raw_tool_call(original)
        assert result is original

    def test_with_flat_dict(self):
        d = {"name": "search", "arguments": {"query": "hi"}}
        tc = ToolCall.from_raw_tool_call(d)
        assert tc.name == "search"
        assert tc.arguments == {"query": "hi"}

    def test_with_openai_dict_json_string_arguments(self):
        d = {
            "id": "call_xyz",
            "function": {
                "name": "calculator",
                "arguments": '{"x": 5}',
            },
        }
        tc = ToolCall.from_raw_tool_call(d)
        assert tc.name == "calculator"
        assert tc.arguments == {"x": 5}
        assert tc.id == "call_xyz"

    def test_with_openai_dict_dict_arguments(self):
        d = {
            "id": "call_999",
            "function": {
                "name": "tool1",
                "arguments": {"key": "value"},
            },
        }
        tc = ToolCall.from_raw_tool_call(d)
        assert tc.name == "tool1"
        assert tc.arguments == {"key": "value"}
        assert tc.id == "call_999"

    def test_with_parameters_key_instead_of_arguments(self):
        d = {"name": "tool2", "parameters": {"p": 42}}
        tc = ToolCall.from_raw_tool_call(d)
        assert tc.arguments == {"p": 42}

    def test_invalid_type_raises_type_error(self):
        with pytest.raises(TypeError):
            ToolCall.from_raw_tool_call(42)

    def test_stores_raw_tool_call(self):
        d = {"name": "t", "arguments": {"a": 1}}
        tc = ToolCall.from_raw_tool_call(d)
        assert tc.raw_tool_call is not None

    def test_dict_missing_name_key_raises_value_error(self):
        """A flat dict without 'name' or 'function' should raise ValueError."""
        with pytest.raises(ValueError, match="must contain 'name' or 'function'"):
            ToolCall.from_raw_tool_call({"arguments": {"x": 1}})

    def test_empty_dict_raises_value_error(self):
        """An empty dict has neither 'name' nor 'function'."""
        with pytest.raises(ValueError, match="must contain 'name' or 'function'"):
            ToolCall.from_raw_tool_call({})

    def test_flat_dict_with_json_string_arguments_stored_as_parsed(self):
        """In flat dict format, string arguments are JSON-parsed when valid JSON."""
        d = {"name": "search", "arguments": '{"query": "hello"}'}
        tc = ToolCall.from_raw_tool_call(d)
        # The from_raw_tool_call normalizes string arguments via json.loads
        assert tc.arguments == {"query": "hello"}

    def test_flat_dict_with_non_json_string_arguments_wrapped(self):
        """Non-JSON string arguments get wrapped in {'raw': ...}."""
        d = {"name": "search", "arguments": "not-valid-json"}
        tc = ToolCall.from_raw_tool_call(d)
        assert tc.arguments == {"raw": "not-valid-json"}

    # -- hardened edge cases --

    def test_openai_dict_with_none_arguments(self):
        """OpenAI dict with arguments=None should normalize to empty dict, not None."""
        d = {
            "id": "call_1",
            "function": {
                "name": "tool",
                "arguments": None,
            },
        }
        tc = ToolCall.from_raw_tool_call(d)
        assert tc.name == "tool"
        assert isinstance(tc.arguments, dict), (
            f"Expected dict arguments but got {type(tc.arguments).__name__}: {tc.arguments}"
        )

    def test_flat_dict_with_integer_arguments(self):
        """Integer arguments should be normalized to a dict, not left as int."""
        d = {"name": "tool", "arguments": 42}
        tc = ToolCall.from_raw_tool_call(d)
        assert isinstance(tc.arguments, dict), (
            f"Expected dict arguments but got {type(tc.arguments).__name__}: {tc.arguments}"
        )

    def test_flat_dict_with_list_arguments(self):
        """List arguments should be normalized to a dict."""
        d = {"name": "tool", "arguments": [1, 2, 3]}
        tc = ToolCall.from_raw_tool_call(d)
        assert isinstance(tc.arguments, dict), (
            f"Expected dict arguments but got {type(tc.arguments).__name__}: {tc.arguments}"
        )

    def test_to_openai_dict_roundtrip(self):
        """Converting to OpenAI dict and back should preserve name and arguments."""
        original = ToolCall(name="search", arguments={"q": "test", "n": 5}, id="call_rt")
        openai_dict = original.to_openai_dict()
        restored = ToolCall.from_raw_tool_call(openai_dict)
        assert restored.name == original.name
        assert restored.arguments == original.arguments
        assert restored.id == original.id

    def test_to_dict_does_not_include_id(self):
        """to_dict should only have name and arguments, not id."""
        tc = ToolCall(name="foo", arguments={"a": 1}, id="call_123")
        d = tc.to_dict()
        assert set(d.keys()) == {"name", "arguments"}

    def test_deeply_nested_arguments_serialize(self):
        """Deeply nested arguments should survive to_openai_dict serialization."""
        deep = {"a": {"b": {"c": {"d": {"e": [1, 2, {"f": True}]}}}}}
        tc = ToolCall(name="deep", arguments=deep)
        d = tc.to_openai_dict()
        restored_args = json.loads(d["function"]["arguments"])
        assert restored_args == deep


class TestToolCallToOpenaiDict:
    def test_structure(self):
        tc = ToolCall(name="search", arguments={"q": "test"}, id="call_abc")
        d = tc.to_openai_dict()
        assert d["id"] == "call_abc"
        assert d["type"] == "function"
        assert d["function"]["name"] == "search"
        assert json.loads(d["function"]["arguments"]) == {"q": "test"}

    def test_arguments_serialized_as_json_string(self):
        tc = ToolCall(name="calc", arguments={"x": 1, "y": 2}, id="call_1")
        d = tc.to_openai_dict()
        assert isinstance(d["function"]["arguments"], str)
        assert json.loads(d["function"]["arguments"]) == {"x": 1, "y": 2}

    def test_non_json_serializable_arguments_raises(self):
        """Arguments containing non-serializable types should raise TypeError
        when to_openai_dict tries json.dumps."""
        tc = ToolCall(name="bad", arguments={"fn": lambda x: x}, id="call_bad")
        with pytest.raises(TypeError):
            tc.to_openai_dict()


class TestToolCallIdUniqueness:
    def test_many_instances_have_unique_ids(self):
        """Generate 500 ToolCalls and verify all auto-generated ids are unique."""
        ids = {ToolCall(name="t", arguments={}).id for _ in range(500)}
        assert len(ids) == 500


# =============================================================================
# ToolResult
# =============================================================================


class TestToolResult:
    def test_to_openai_dict(self):
        tr = ToolResult(content="result text", name="search", tool_call_id="call_abc")
        d = tr.to_openai_dict()
        assert d == {
            "role": "tool",
            "tool_call_id": "call_abc",
            "content": "result text",
        }


# =============================================================================
# ToolOutput
# =============================================================================


class TestToolOutput:
    def test_to_content_string_with_error(self):
        to = ToolOutput(name="t", error="something went wrong")
        assert to.to_content_string() == "Error: something went wrong"

    def test_to_content_string_error_takes_precedence_over_output(self):
        to = ToolOutput(name="t", output="data", error="fail")
        assert to.to_content_string() == "Error: fail"

    def test_to_content_string_with_dict_output(self):
        data = {"key": "value", "num": 42}
        to = ToolOutput(name="t", output=data)
        result = to.to_content_string()
        assert json.loads(result) == data

    def test_to_content_string_with_list_output(self):
        data = [1, 2, 3]
        to = ToolOutput(name="t", output=data)
        result = to.to_content_string()
        assert json.loads(result) == data

    def test_to_content_string_with_none_output(self):
        to = ToolOutput(name="t", output=None)
        assert to.to_content_string() == ""

    def test_to_content_string_with_integer_output(self):
        """Integer output should be converted via str()."""
        to = ToolOutput(name="t", output=42)
        assert to.to_content_string() == "42"

    def test_to_content_string_with_bool_output(self):
        """Bool output should be converted via str()."""
        to = ToolOutput(name="t", output=True)
        assert to.to_content_string() == "True"

    def test_to_content_string_with_deeply_nested_dict(self):
        """Deeply nested dict should be JSON-serialized properly."""
        data = {"a": {"b": {"c": {"d": {"e": [1, 2, {"f": "deep"}]}}}}}
        to = ToolOutput(name="t", output=data)
        result = to.to_content_string()
        assert json.loads(result) == data

    def test_str_dunder_calls_to_content_string(self):
        to = ToolOutput(name="t", output="hello")
        assert str(to) == to.to_content_string()

    # -- hardened edge cases --

    def test_empty_string_error_bypasses_error_branch(self):
        """Empty string error is falsy — the output is returned instead of the error."""
        to = ToolOutput(name="t", error="", output="data")
        result = to.to_content_string()
        # BUG: error="" is set but ignored because "" is falsy in Python
        assert result != "data", (
            "Empty string error was silently ignored: output 'data' returned instead. "
            "An explicitly set error (even empty) should take precedence over output."
        )

    def test_error_with_whitespace_only(self):
        """Whitespace-only error string is truthy, so it should be treated as error."""
        to = ToolOutput(name="t", error="  ", output="data")
        assert to.to_content_string() == "Error:   "

    def test_output_with_empty_dict(self):
        """Empty dict output should serialize to '{}'."""
        to = ToolOutput(name="t", output={})
        assert to.to_content_string() == "{}"

    def test_output_with_empty_list(self):
        """Empty list output should serialize to '[]'."""
        to = ToolOutput(name="t", output=[])
        assert to.to_content_string() == "[]"
