"""Tests for axon.tools.executors module."""

import copy
import json

import pytest

from axon.tools.executors import LocalToolExecutor
from axon.tools.tools import Tool
from axon.tools.types import ToolCall, ToolOutput, ToolResult

# =============================================================================
# Mock Tools
# =============================================================================


class MockTool(Tool):
    def __init__(self):
        super().__init__(name="mock", description="A mock tool")

    @property
    def json(self):
        return {
            "type": "function",
            "function": {
                "name": "mock",
                "description": "A mock tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }

    def forward(self, **kwargs) -> ToolOutput:
        return ToolOutput(name="mock", output=f"result: {kwargs}")


class FailingTool(Tool):
    def __init__(self):
        super().__init__(name="failing", description="A tool that always fails")

    @property
    def json(self):
        return {
            "type": "function",
            "function": {
                "name": "failing",
                "description": "A tool that always fails",
                "parameters": {"type": "object", "properties": {}},
            },
        }

    def forward(self, **kwargs) -> ToolOutput:
        raise RuntimeError("intentional error")


class MutatingTool(Tool):
    """Tool that mutates its kwargs dict -- should not leak to ToolCall.arguments."""

    def __init__(self):
        super().__init__(name="mutating", description="Mutates kwargs")

    @property
    def json(self):
        return {
            "type": "function",
            "function": {
                "name": "mutating",
                "description": "Mutates kwargs",
                "parameters": {"type": "object", "properties": {}},
            },
        }

    def forward(self, **kwargs) -> ToolOutput:
        kwargs["injected"] = True
        return ToolOutput(name="mutating", output="done")


class NonStandardReturnTool(Tool):
    """Tool whose forward returns a plain string instead of ToolOutput."""

    def __init__(self):
        super().__init__(name="nonstandard", description="Returns non-ToolOutput")

    @property
    def json(self):
        return {
            "type": "function",
            "function": {
                "name": "nonstandard",
                "description": "Returns non-ToolOutput",
                "parameters": {"type": "object", "properties": {}},
            },
        }

    def forward(self, **kwargs):
        return "just a string"


# =============================================================================
# Execution tests
# =============================================================================


class TestLocalToolExecutorExecuteSync:
    def test_execute_known_tool(self):
        executor = LocalToolExecutor(tools={"mock": MockTool()})
        tc = ToolCall(name="mock", arguments={"x": 1})
        result = executor.execute_sync(tc)
        assert isinstance(result, ToolResult)
        assert result.name == "mock"
        assert result.tool_call_id == tc.id
        assert "x" in result.content
        assert "1" in result.content

    def test_execute_unknown_tool(self):
        executor = LocalToolExecutor(tools={"mock": MockTool()})
        tc = ToolCall(name="unknown_tool", arguments={})
        result = executor.execute_sync(tc)
        assert isinstance(result, ToolResult)
        parsed = json.loads(result.content)
        assert "error" in parsed
        assert "Unknown tool" in parsed["error"]

    def test_execute_tool_that_raises(self):
        executor = LocalToolExecutor(tools={"failing": FailingTool()})
        tc = ToolCall(name="failing", arguments={})
        result = executor.execute_sync(tc)
        assert isinstance(result, ToolResult)
        parsed = json.loads(result.content)
        assert "error" in parsed
        assert "RuntimeError" in parsed["error"]

    def test_execute_with_non_standard_return_type(self):
        """Tool.forward returning a non-ToolOutput should cause an error
        (to_content_string call will fail on a plain string)."""
        executor = LocalToolExecutor(tools={"nonstandard": NonStandardReturnTool()})
        tc = ToolCall(name="nonstandard", arguments={})
        result = executor.execute_sync(tc)
        # The executor wraps the call in try/except, so we get an error result
        # because _success_result calls output.to_content_string() on a string.
        assert isinstance(result, ToolResult)
        parsed = json.loads(result.content)
        assert "error" in parsed


class TestLocalToolExecutorBatch:
    def test_sequential_batch(self):
        executor = LocalToolExecutor(tools={"mock": MockTool()})
        calls = [
            ToolCall(name="mock", arguments={"i": 0}),
            ToolCall(name="mock", arguments={"i": 1}),
            ToolCall(name="mock", arguments={"i": 2}),
        ]
        results = executor.execute_batch_sync(calls, parallel=False)
        assert len(results) == 3
        for i, r in enumerate(results):
            assert isinstance(r, ToolResult)
            assert str(i) in r.content

    def test_parallel_batch(self):
        executor = LocalToolExecutor(tools={"mock": MockTool()})
        calls = [
            ToolCall(name="mock", arguments={"i": 0}),
            ToolCall(name="mock", arguments={"i": 1}),
            ToolCall(name="mock", arguments={"i": 2}),
        ]
        results = executor.execute_batch_sync(calls, parallel=True)
        assert len(results) == 3
        for i, r in enumerate(results):
            assert isinstance(r, ToolResult)
            assert str(i) in r.content

    def test_parallel_batch_preserves_ordering_5_items(self):
        """Parallel execution with 5 items must return results in original order."""
        executor = LocalToolExecutor(tools={"mock": MockTool()})
        calls = [ToolCall(name="mock", arguments={"idx": i}) for i in range(5)]
        results = executor.execute_batch_sync(calls, parallel=True)
        assert len(results) == 5
        for i, r in enumerate(results):
            assert str(i) in r.content


class TestLocalToolExecutorRegistration:
    def test_register_class_auto_instantiated(self):
        executor = LocalToolExecutor()
        executor.register("mock", MockTool)
        assert len(executor) == 1
        assert isinstance(executor.get("mock"), MockTool)

    def test_register_invalid_type_raises(self):
        executor = LocalToolExecutor()
        with pytest.raises(TypeError):
            executor.register("bad", 42)

    def test_register_same_name_twice_overwrites_silently(self):
        """Registering a tool with the same name should overwrite the previous one."""
        executor = LocalToolExecutor()
        tool1 = MockTool()
        tool2 = MockTool()
        executor.register("shared_name", tool1)
        executor.register("shared_name", tool2)
        assert len(executor) == 1
        assert executor.get("shared_name") is tool2
        assert executor.get("shared_name") is not tool1

    def test_getitem_missing_key_raises_key_error(self):
        executor = LocalToolExecutor(tools={"mock": MockTool()})
        with pytest.raises(KeyError):
            _ = executor["nonexistent"]

    def test_tools_property_returns_copy_mutation_safe(self):
        """Mutating the returned tools dict must not affect the executor."""
        executor = LocalToolExecutor(tools={"mock": MockTool()})
        tools = executor.tools
        tools["injected"] = MockTool()
        assert "injected" not in executor

    def test_tool_mutating_kwargs_does_not_leak(self):
        """A tool that mutates kwargs in forward() should not corrupt ToolCall.arguments."""
        executor = LocalToolExecutor(tools={"mutating": MutatingTool()})
        tc = ToolCall(name="mutating", arguments={"key": "value"})
        original_args = copy.deepcopy(tc.arguments)
        executor.execute_sync(tc)
        # ToolCall.arguments should be unchanged because the executor unpacks
        # into **kwargs (a new dict), not the arguments dict itself.
        assert tc.arguments == original_args
        assert "injected" not in tc.arguments

    # -- hardened edge cases --

    def test_empty_executor_properties(self):
        """Empty executor should have sensible defaults for all properties."""
        executor = LocalToolExecutor()
        assert len(executor) == 0
        assert executor.names == []
        assert executor.schemas == []
        assert executor.tools == {}
        assert executor.get("anything") is None

    def test_contains_and_iter(self):
        """__contains__ and __iter__ should work correctly."""
        executor = LocalToolExecutor(tools={"mock": MockTool(), "failing": FailingTool()})
        assert "mock" in executor
        assert "nonexistent" not in executor
        assert set(executor) == {"mock", "failing"}


class TestLocalToolExecutorBatchEdgeCases:
    def test_execute_batch_sync_empty_list(self):
        """Batch with empty list should return empty list for both modes."""
        executor = LocalToolExecutor(tools={"mock": MockTool()})
        assert executor.execute_batch_sync([], parallel=False) == []
        assert executor.execute_batch_sync([], parallel=True) == []

    def test_parallel_batch_duplicate_ids_returns_correct_results(self):
        """Two tool calls with the same ID in parallel: each should get its own result."""
        executor = LocalToolExecutor(tools={"mock": MockTool()})
        tc1 = ToolCall(name="mock", arguments={"val": "first"}, id="same_id")
        tc2 = ToolCall(name="mock", arguments={"val": "second"}, id="same_id")
        results = executor.execute_batch_sync([tc1, tc2], parallel=True)
        assert len(results) == 2
        # Each result should reflect its OWN tool call arguments
        assert "first" in results[0].content, f"First result should contain 'first' but got: {results[0].content}"
        assert "second" in results[1].content, f"Second result should contain 'second' but got: {results[1].content}"

    def test_parallel_batch_mixed_success_and_failure(self):
        """Parallel batch with both succeeding and failing tools should handle both."""
        executor = LocalToolExecutor(tools={"mock": MockTool(), "failing": FailingTool()})
        calls = [
            ToolCall(name="mock", arguments={"i": 0}),
            ToolCall(name="failing", arguments={}),
            ToolCall(name="mock", arguments={"i": 2}),
        ]
        results = executor.execute_batch_sync(calls, parallel=True)
        assert len(results) == 3
        # First and third succeed
        assert "error" not in results[0].content.lower() or "result" in results[0].content.lower()
        # Second fails
        parsed = json.loads(results[1].content)
        assert "error" in parsed
        assert "RuntimeError" in parsed["error"]
