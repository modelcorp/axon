"""Tests for axon.tools.tools module — Tool base class and _function_to_schema."""

from typing import Annotated

import pytest

from axon.tools.tools import Tool, _function_to_schema
from axon.tools.types import ToolOutput


# ---------------------------------------------------------------------------
# _function_to_schema
# ---------------------------------------------------------------------------
class TestFunctionToSchema:
    def test_basic_function(self):
        def greet(name: str) -> str:
            """Say hello to someone."""
            return f"Hello, {name}"

        schema = _function_to_schema(greet)
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "greet"
        assert schema["function"]["description"] == "Say hello to someone."
        props = schema["function"]["parameters"]["properties"]
        assert "name" in props
        assert props["name"]["type"] == "string"
        assert "name" in schema["function"]["parameters"]["required"]

    def test_multiple_params_with_types(self):
        def add(x: int, y: float) -> float:
            """Add two numbers."""
            return x + y

        schema = _function_to_schema(add)
        props = schema["function"]["parameters"]["properties"]
        assert props["x"]["type"] == "integer"
        assert props["y"]["type"] == "number"
        assert set(schema["function"]["parameters"]["required"]) == {"x", "y"}

    def test_optional_param_not_required(self):
        def search(query: str, limit: int = 10) -> list:
            """Search for items."""
            return []

        schema = _function_to_schema(search)
        required = schema["function"]["parameters"]["required"]
        assert "query" in required
        assert "limit" not in required

    def test_annotated_with_description(self):
        def lookup(
            key: Annotated[str, "The lookup key"],
            timeout: Annotated[int, "Timeout in seconds"] = 30,
        ) -> dict:
            """Look up a value."""
            return {}

        schema = _function_to_schema(lookup)
        props = schema["function"]["parameters"]["properties"]
        assert props["key"]["type"] == "string"
        assert props["key"]["description"] == "The lookup key"
        assert props["timeout"]["type"] == "integer"
        assert props["timeout"]["description"] == "Timeout in seconds"

    def test_no_type_annotations_defaults_to_string(self):
        def raw(x, y):
            """No annotations."""
            pass

        schema = _function_to_schema(raw)
        props = schema["function"]["parameters"]["properties"]
        assert props["x"]["type"] == "string"
        assert props["y"]["type"] == "string"

    def test_no_docstring(self):
        def silent(x: str):
            pass

        schema = _function_to_schema(silent)
        assert schema["function"]["description"] == ""

    def test_multiline_docstring_uses_first_line(self):
        def verbose(x: str):
            """First line summary.

            This is a longer description that should be ignored.
            """
            pass

        schema = _function_to_schema(verbose)
        assert schema["function"]["description"] == "First line summary."

    def test_bool_param_type(self):
        def toggle(flag: bool) -> None:
            """Toggle a flag."""
            pass

        schema = _function_to_schema(toggle)
        assert schema["function"]["parameters"]["properties"]["flag"]["type"] == "boolean"

    def test_dict_param_type(self):
        def process(data: dict) -> None:
            """Process data."""
            pass

        schema = _function_to_schema(process)
        assert schema["function"]["parameters"]["properties"]["data"]["type"] == "object"

    def test_list_param_type(self):
        def batch(items: list) -> None:
            """Process batch."""
            pass

        schema = _function_to_schema(batch)
        assert schema["function"]["parameters"]["properties"]["items"]["type"] == "array"

    def test_unknown_type_defaults_to_string(self):
        """Custom classes without mapping should default to 'string'."""

        class CustomType:
            pass

        def process(data: CustomType) -> None:
            """Process."""
            pass

        schema = _function_to_schema(process)
        assert schema["function"]["parameters"]["properties"]["data"]["type"] == "string"

    def test_no_params(self):
        def noop():
            """Do nothing."""
            pass

        schema = _function_to_schema(noop)
        assert schema["function"]["parameters"]["properties"] == {}
        assert schema["function"]["parameters"]["required"] == []

    def test_self_param_included_for_unbound_methods(self):
        """_function_to_schema doesn't filter 'self' — it includes all params."""

        class MyClass:
            def method(self, x: str) -> str:
                """A method."""
                return x

        schema = _function_to_schema(MyClass.method)
        props = schema["function"]["parameters"]["properties"]
        # 'self' is included as a parameter (potential bug — should be filtered)
        assert "self" in props, (
            "'self' should be in schema for unbound methods. If Tool wraps a method, this leaks implementation details."
        )


# ---------------------------------------------------------------------------
# Tool base class
# ---------------------------------------------------------------------------
class TestToolInit:
    def test_init_with_name_and_description(self):
        tool = Tool(name="calc", description="A calculator")
        assert tool.name == "calc"
        assert tool.description == "A calculator"

    def test_init_without_name_or_function_raises(self):
        """All three cases (no args, name-only, desc-only) hit the same check."""
        with pytest.raises(ValueError, match="requires.*name.*description.*function"):
            Tool()
        with pytest.raises(ValueError):
            Tool(name="calc")
        with pytest.raises(ValueError):
            Tool(description="A calculator")

    def test_init_with_function(self):
        def my_func(query: str) -> str:
            """Search the web."""
            return f"Results for {query}"

        tool = Tool(function=my_func)
        assert tool.name == "my_func"
        assert "Search the web" in tool.description

    def test_json_property_with_function(self):
        def add(x: int, y: int) -> int:
            """Add two numbers."""
            return x + y

        tool = Tool(function=add)
        schema = tool.json
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "add"

    def test_json_property_without_function_raises(self):
        tool = Tool(name="calc", description="calc")
        with pytest.raises(NotImplementedError, match="must implement.*json"):
            _ = tool.json


class TestToolForward:
    def test_forward_with_function(self):
        def greet(name: str) -> str:
            """Greet someone."""
            return f"Hello, {name}!"

        tool = Tool(function=greet)
        result = tool.forward(name="World")
        assert isinstance(result, ToolOutput)
        assert result.output == "Hello, World!"
        assert result.error is None

    def test_forward_with_function_that_raises(self):
        def failing(x: int) -> int:
            """Will fail."""
            raise ValueError("bad input")

        tool = Tool(function=failing)
        result = tool.forward(x=42)
        assert isinstance(result, ToolOutput)
        assert result.error is not None
        assert "ValueError" in result.error
        assert "bad input" in result.error

    def test_forward_without_function_raises(self):
        tool = Tool(name="calc", description="calc")
        with pytest.raises(NotImplementedError, match="must implement forward"):
            tool.forward()

    def test_forward_with_no_args(self):
        def ping() -> str:
            """Ping."""
            return "pong"

        tool = Tool(function=ping)
        result = tool.forward()
        assert result.output == "pong"

    def test_forward_return_none(self):
        def void_fn() -> None:
            """Returns nothing."""
            pass

        tool = Tool(function=void_fn)
        result = tool.forward()
        assert isinstance(result, ToolOutput)
        assert result.output is None


class TestToolAsyncForward:
    def test_async_forward_delegates_to_forward(self):
        import asyncio

        def greet(name: str) -> str:
            """Greet."""
            return f"Hi, {name}!"

        tool = Tool(function=greet)
        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(tool.async_forward(name="Test"))
        finally:
            loop.close()
        assert isinstance(result, ToolOutput)
        assert result.output == "Hi, Test!"


# ---------------------------------------------------------------------------
# Edge cases that break
# ---------------------------------------------------------------------------
class TestToolEdgeCases:
    def test_function_with_kwargs_not_in_schema(self):
        """Passing kwargs not declared in the function should raise TypeError."""

        def strict(x: int) -> int:
            """Strict."""
            return x

        tool = Tool(function=strict)
        result = tool.forward(x=1, y=2)  # y is unexpected
        # The function raises TypeError, which gets caught and returned as error
        assert result.error is not None
        assert "TypeError" in result.error

    def test_function_with_missing_required_arg(self):
        """Missing required arg should raise TypeError."""

        def required(x: int) -> int:
            """Needs x."""
            return x

        tool = Tool(function=required)
        result = tool.forward()  # missing x
        assert result.error is not None
        assert "TypeError" in result.error

    def test_empty_string_name_or_description_raises(self):
        """Empty strings are falsy — same check as missing args."""
        with pytest.raises(ValueError):
            Tool(name="", description="something")
        with pytest.raises(ValueError):
            Tool(name="something", description="")
