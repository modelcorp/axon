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
MCP connection manager.

Manages MCP server connections in a dedicated thread to avoid asyncio
context issues. Creates ``MCPTool`` instances that work with the standard
``LocalToolExecutor``.

Usage
-----
::

    manager = MCPConnectionManager("npx", ["-y", "@mcp/server-filesystem"])
    manager.start()

    # Get tools for executor
    tool_map = manager.tool_map  # dict[str, MCPTool]
    executor = LocalToolExecutor(tool_map)

    # Or execute directly (thread-safe)
    results = manager.execute_tool_calls(raw_tool_calls)

    manager.stop()
"""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
from contextlib import AsyncExitStack
from typing import Any

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from axon.tools.tools import MCPTool
from axon.tools.types import ToolCall

logger = logging.getLogger(__name__)


class MCPConnectionManager:
    """
    Manages MCP connections in a dedicated thread.

    Provides thread-safe tool execution and exposes ``tool_map`` for
    building executors and registries.
    """

    def __init__(
        self,
        mcp_server_command: str,
        mcp_server_args: list[str] | None = None,
        mcp_server_env: dict[str, str] | None = None,
    ):
        if ClientSession is None:
            raise ImportError("mcp package not installed. Install with: pip install mcp")

        self.mcp_server_command = mcp_server_command
        self.mcp_server_args = mcp_server_args or []
        self.mcp_server_env = mcp_server_env

        self._request_queue: queue.Queue = queue.Queue()
        self._worker_thread: threading.Thread | None = None
        self._running = False
        self.tool_map: dict[str, MCPTool] = {}

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self, timeout: float = 30.0):
        """Start the connection manager thread and wait for initialization."""
        if self._running:
            return

        self._running = True
        self._worker_thread = threading.Thread(target=self._run_worker, daemon=True)
        self._worker_thread.start()

        response_q: queue.Queue = queue.Queue()
        self._request_queue.put(("init", None, response_q))
        status, result = response_q.get(timeout=timeout)
        if status == "error":
            raise RuntimeError(f"Failed to initialize MCP connection: {result}")
        self.tool_map = result

    def stop(self):
        """Stop the connection manager thread."""
        if not self._running:
            return
        self._running = False
        self._request_queue.put(("stop", None, None))
        if self._worker_thread:
            self._worker_thread.join(timeout=5)

    def execute_tool_calls(self, tool_calls: list[dict[str, Any]], timeout: float = 30.0) -> dict[str, str]:
        """Execute raw tool call dicts and return {tool_call_id: content_str}."""
        if not self._running:
            raise RuntimeError("Connection manager not running")
        response_q: queue.Queue = queue.Queue()
        self._request_queue.put(("execute", tool_calls, response_q))
        status, result = response_q.get(timeout=timeout)
        if status == "error":
            raise RuntimeError(f"Tool execution failed: {result}")
        return result

    # -- Internal: worker thread ----------------------------------------------

    def _run_worker(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._worker_loop())
        finally:
            loop.close()

    async def _worker_loop(self):
        exit_stack: AsyncExitStack | None = None
        session: ClientSession | None = None

        while self._running:
            try:
                request = self._request_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            command, data, response_q = request

            if command == "init":
                try:
                    exit_stack = AsyncExitStack()
                    server_params = StdioServerParameters(
                        command=self.mcp_server_command,
                        args=self.mcp_server_args,
                        env=self.mcp_server_env,
                    )
                    transport = await exit_stack.enter_async_context(stdio_client(server_params))
                    stdio, write = transport
                    session = await exit_stack.enter_async_context(ClientSession(stdio, write))
                    await session.initialize()

                    tools_response = await session.list_tools()
                    tool_map = {}
                    for tool in tools_response.tools:
                        tool_map[tool.name] = MCPTool(
                            session=session,
                            tool_name=tool.name,
                            tool_description=tool.description,
                            tool_schema=tool.inputSchema,
                        )
                    logger.info("Connected to MCP server with %d tools", len(tool_map))
                    response_q.put(("success", tool_map))
                except Exception as e:
                    response_q.put(("error", str(e)))

            elif command == "execute":
                try:
                    results = await self._execute_tools(data, session)
                    response_q.put(("success", results))
                except Exception as e:
                    response_q.put(("error", str(e)))

            elif command == "stop":
                break

        # Cleanup
        if exit_stack:
            try:
                await exit_stack.aclose()
            except Exception:
                pass

    async def _execute_tools(
        self,
        raw_tool_calls: list[dict[str, Any]],
        session: Any,
    ) -> dict[str, str]:
        outputs: dict[str, str] = {}
        for raw_call in raw_tool_calls:
            tc = ToolCall.from_raw_tool_call(raw_call)
            if tc.name in self.tool_map:
                result = await self.tool_map[tc.name].async_forward(**tc.arguments)
                outputs[tc.id] = result.to_content_string()
            else:
                outputs[tc.id] = f"Error: Tool {tc.name} not found"
        return outputs
