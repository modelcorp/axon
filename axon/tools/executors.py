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
Tool execution layer.

Decouples *parsing* (what tool to call) from *execution* (how to call it).
Workflow: ToolCall → execute → ToolResult.

------------------
LocalToolExecutor  : Holds and executes local ``Tool`` instances directly.
HTTPToolExecutor   : Calls tools via HTTP POST (NeMo Gym resource servers, etc.).


Usage
-----
::

    from axon.tools.executors import LocalToolExecutor

    # Accepts instances, classes, or import path strings
    executor = LocalToolExecutor({
        "calculator": CalculatorTool(),
        "search":     GoogleSearchTool,                   # class → auto-instantiated
        "browser":    "path/to/file.py:BrowserTool",      # string → loaded + instantiated
    })

    # Schemas for building system prompts
    schemas = executor.schemas

    # Async execution
    result = await executor.execute(tool_call)
    results = await executor.execute_batch(tool_calls)

    # Synchronous execution (agents, environments)
    result = executor.execute_sync(tool_call)
    results = executor.execute_batch_sync(tool_calls, parallel=True)
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import httpx

from axon.tools.tools import Tool
from axon.tools.types import ToolCall, ToolOutput, ToolResult

logger = logging.getLogger(__name__)


# =============================================================================
# Tool loading utility
# =============================================================================


def load_tool_class(path: str) -> type[Tool]:
    """
    Load a Tool class from a file path or module path.

    Args:
        path: Format ``'path/to/file.py:ClassName'`` or ``'module.path:ClassName'``
    """
    from axon.utils.module_loader import load_module

    if ":" not in path:
        raise ValueError(
            f"Invalid tool path format: {path!r}. Expected 'path/to/file.py:ClassName' or 'module.path:ClassName'"
        )

    module_path, class_name = path.rsplit(":", 1)
    module = load_module(module_path)
    if module is None:
        raise ImportError(f"Failed to load module from: {module_path}")
    if not hasattr(module, class_name):
        raise AttributeError(f"Module {module_path} does not have class '{class_name}'")
    return getattr(module, class_name)


# =============================================================================
# Helpers
# =============================================================================


def _error_result(tool_call: ToolCall, error: str) -> ToolResult:
    """Build an error ToolResult for a failed call."""
    return ToolResult(
        content=json.dumps({"error": error}),
        name=tool_call.name,
        tool_call_id=tool_call.id,
    )


def _success_result(tool_call: ToolCall, output: ToolOutput) -> ToolResult:
    """Build a success ToolResult from a ToolOutput."""
    return ToolResult(
        content=output.to_content_string(),
        name=tool_call.name,
        tool_call_id=tool_call.id,
    )


# =============================================================================
# Base class
# =============================================================================


class ToolExecutor(ABC):
    """Executes tool calls and produces results."""

    @abstractmethod
    async def execute(self, tool_call: ToolCall) -> ToolResult:
        """Execute a single tool call (async)."""
        ...

    async def execute_batch(self, tool_calls: list[ToolCall]) -> list[ToolResult]:
        """Execute multiple tool calls. Override for parallelism."""
        return [await self.execute(tc) for tc in tool_calls]

    def execute_sync(self, tool_call: ToolCall) -> ToolResult:
        """Synchronous single-call execution. Override if supported."""
        raise NotImplementedError(f"{self.__class__.__name__} does not support synchronous execution")

    def execute_batch_sync(
        self,
        tool_calls: list[ToolCall],
        parallel: bool = False,
    ) -> list[ToolResult]:
        """Synchronous batch execution. Override for parallel threading."""
        return [self.execute_sync(tc) for tc in tool_calls]


# =============================================================================
# Local executor — holds tools, provides schemas, executes calls
# =============================================================================


class LocalToolExecutor(ToolExecutor):
    """
    Holds and executes local ``Tool`` instances.

    This is the primary executor for Axon. It owns the tool collection,
    provides schema access for prompt building, and handles dispatch.
    Works with any Tool subclass including MCPTool (via async_forward).

    Accepts Tool instances, Tool classes (auto-instantiated), or string
    import paths (``'path/to/file.py:ClassName'``).
    """

    def __init__(self, tools: dict[str, Any] | None = None):
        self._tools: dict[str, Tool] = {}
        if tools:
            for name, t in tools.items():
                self.register(name, t)

    # -- Registration ---------------------------------------------------------

    def register(self, name: str, tool: Any) -> None:
        """
        Register a tool by name.

        Args:
            name: Tool name (used for dispatch and schema listing).
            tool: A Tool instance, Tool class, or string import path.
        """

        if isinstance(tool, str):
            tool = load_tool_class(tool)()
        elif isinstance(tool, type) and issubclass(tool, Tool):
            tool = tool()

        if not isinstance(tool, Tool):
            raise TypeError(
                f"Expected Tool instance, Tool class, or import path string. Got {type(tool).__name__} for '{name}'"
            )

        self._tools[name] = tool

    # -- Collection access ----------------------------------------------------

    def get(self, name: str) -> Tool | None:
        """Get a tool by name, or None if not found."""
        return self._tools.get(name)

    def __getitem__(self, name: str) -> Tool:
        return self._tools[name]

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    def __iter__(self):
        return iter(self._tools)

    @property
    def names(self) -> list[str]:
        """All registered tool names."""
        return list(self._tools)

    @property
    def tools(self) -> dict[str, Tool]:
        """The underlying name → Tool mapping (copy)."""
        return dict(self._tools)

    @property
    def schemas(self) -> list[dict[str, Any]]:
        """All tool schemas as a list (for system prompt building)."""
        return [t.json for t in self._tools.values()]

    # -- Execution ------------------------------------------------------------

    def _resolve(self, tool_call: ToolCall) -> Tool | None:
        return self._tools.get(tool_call.name)

    async def execute(self, tool_call: ToolCall) -> ToolResult:
        tool = self._resolve(tool_call)
        if tool is None:
            return _error_result(
                tool_call,
                f"Unknown tool: {tool_call.name}. Available: {self.names}",
            )
        try:
            output = await tool.async_forward(**tool_call.arguments)
            return _success_result(tool_call, output)
        except Exception as e:
            logger.warning("Tool %s raised %s: %s", tool_call.name, type(e).__name__, e)
            return _error_result(tool_call, f"{type(e).__name__}: {e}")

    def execute_sync(self, tool_call: ToolCall) -> ToolResult:
        tool = self._resolve(tool_call)
        if tool is None:
            return _error_result(
                tool_call,
                f"Unknown tool: {tool_call.name}. Available: {self.names}",
            )
        try:
            output = tool.forward(**tool_call.arguments)
            return _success_result(tool_call, output)
        except Exception as e:
            logger.warning("Tool %s raised %s: %s", tool_call.name, type(e).__name__, e)
            return _error_result(tool_call, f"{type(e).__name__}: {e}")

    def execute_batch_sync(
        self,
        tool_calls: list[ToolCall],
        parallel: bool = False,
    ) -> list[ToolResult]:
        """Execute tool calls, optionally in parallel threads."""
        if not parallel or len(tool_calls) <= 1:
            return [self.execute_sync(tc) for tc in tool_calls]

        results: list[ToolResult | None] = [None] * len(tool_calls)
        with ThreadPoolExecutor(max_workers=len(tool_calls)) as pool:
            futures = {pool.submit(self.execute_sync, tc): idx for idx, tc in enumerate(tool_calls)}
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    results[idx] = ToolResult(
                        content=json.dumps({"error": str(e)}),
                        name="unknown",
                        tool_call_id=tool_calls[idx].id,
                    )

        return results  # type: ignore[return-value]


# =============================================================================
# HTTP executor — calls tools via REST endpoints
# =============================================================================


class HTTPToolExecutor(ToolExecutor):
    """
    Executes tools via HTTP POST to ``{base_url}/{tool_name}``.

    Used by NeMo Gym resource servers and similar HTTP-based tool backends.
    Does not hold Tool instances — schemas are managed externally.
    """

    def __init__(self, http_client: httpx.AsyncClient, base_url: str):
        self.http = http_client
        self.base_url = base_url.rstrip("/")

    async def execute(self, tool_call: ToolCall) -> ToolResult:
        try:
            resp = await self.http.post(
                f"{self.base_url}/{tool_call.name}",
                json=tool_call.arguments,
            )
            resp.raise_for_status()
            raw = resp.json()
            content = json.dumps(raw) if isinstance(raw, (dict | list)) else str(raw)
        except Exception as exc:
            logger.warning("Tool %s failed: %s", tool_call.name, exc)
            content = json.dumps({"error": str(exc)})

        return ToolResult(
            content=content,
            name=tool_call.name,
            tool_call_id=tool_call.id,
        )
