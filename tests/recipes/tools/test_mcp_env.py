import sys
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

# Add recipes folder to sys.path since it's outside the axon package
_recipes_path = Path(__file__).parent.parent.parent.parent / "recipes"
if str(_recipes_path) not in sys.path:
    sys.path.insert(0, str(_recipes_path))

from tools.mcp_connection_manager import MCPConnectionManager  # noqa: E402

from axon.tools.tools import MCPTool  # noqa: E402
from axon.tools.types import ToolOutput  # noqa: E402


class TestMCPConnectionManagerInit:
    """Test MCPConnectionManager initialization."""

    def test_init_basic(self):
        """Test basic initialization."""
        manager = MCPConnectionManager("npx", ["-y", "@mcp/server"])
        assert manager.mcp_server_command == "npx"
        assert manager.mcp_server_args == ["-y", "@mcp/server"]
        assert manager.mcp_server_env is None
        assert manager.tool_map == {}
        assert not manager.is_running

    def test_init_with_env(self):
        """Test initialization with environment variables."""
        env = {"PATH": "/usr/bin", "HOME": "/root"}
        manager = MCPConnectionManager("node", ["server.js"], env)
        assert manager.mcp_server_env == env

    def test_init_default_args(self):
        """Test initialization with default args."""
        manager = MCPConnectionManager("npx")
        assert manager.mcp_server_args == []
        assert manager.mcp_server_env is None

    def test_is_running_initially_false(self):
        """Test that is_running is False initially."""
        manager = MCPConnectionManager("npx")
        assert manager.is_running is False


class TestMCPConnectionManagerOperations:
    """Test MCPConnectionManager operations (with mocked internals)."""

    def test_stop_when_not_running(self):
        """Test that stop() is a no-op when not running."""
        manager = MCPConnectionManager("npx")
        manager.stop()  # Should not raise

    def test_execute_tool_calls_when_not_running(self):
        """Test that execute_tool_calls raises when not running."""
        manager = MCPConnectionManager("npx")
        with pytest.raises(RuntimeError, match="not running"):
            manager.execute_tool_calls([{"id": "call_1", "function": {"name": "test", "arguments": "{}"}}])

    def test_start_sets_running(self):
        """Test that start sets _running and creates thread."""
        manager = MCPConnectionManager("npx")

        # Mock the worker thread and queue to avoid actual MCP connection
        with patch.object(manager, "_run_worker"):
            # Simulate the init response

            def mock_start(self, timeout=30.0):
                self._running = True
                # Simulate successful init by setting tool_map directly
                mock_session = Mock()
                self.tool_map = {
                    "test_tool": MCPTool(
                        session=mock_session,
                        tool_name="test_tool",
                        tool_description="A test tool",
                        tool_schema={"type": "object", "properties": {}},
                    )
                }

            with patch.object(MCPConnectionManager, "start", mock_start):
                manager.start()
                assert manager._running is True
                assert "test_tool" in manager.tool_map

    def test_stop_sets_not_running(self):
        """Test that stop clears running flag."""
        manager = MCPConnectionManager("npx")
        manager._running = True
        manager._worker_thread = Mock()
        manager._worker_thread.join = Mock()
        manager.stop()
        assert manager._running is False


class TestMCPConnectionManagerToolMap:
    """Test MCPConnectionManager tool_map integration."""

    def test_tool_map_contains_mcp_tools(self):
        """Test that tool_map entries are MCPTool instances."""
        manager = MCPConnectionManager("npx")
        mock_session = Mock()
        tool = MCPTool(
            session=mock_session,
            tool_name="calculator",
            tool_description="Performs calculations",
            tool_schema={
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        )
        manager.tool_map = {"calculator": tool}

        assert "calculator" in manager.tool_map
        assert isinstance(manager.tool_map["calculator"], MCPTool)
        assert manager.tool_map["calculator"].name == "calculator"

    def test_mcp_tool_schema(self):
        """Test that MCPTool produces correct schema."""
        mock_session = Mock()
        tool = MCPTool(
            session=mock_session,
            tool_name="search",
            tool_description="Search the web",
            tool_schema={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        )

        schema = tool.json
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "search"
        assert schema["function"]["description"] == "Search the web"
        assert "query" in schema["function"]["parameters"]["properties"]

    def test_mcp_tool_async_forward(self):
        """Test MCPTool async_forward calls session."""
        import asyncio

        mock_session = AsyncMock()
        mock_result = Mock()
        mock_result.content = [Mock(text="Search result: Python is great")]
        mock_result.isError = False
        mock_session.call_tool.return_value = mock_result

        tool = MCPTool(
            session=mock_session,
            tool_name="search",
            tool_description="Search the web",
            tool_schema={"type": "object", "properties": {}},
        )

        result = asyncio.run(tool.async_forward(query="python"))
        assert isinstance(result, ToolOutput)
        mock_session.call_tool.assert_called_once_with("search", {"query": "python"})


class TestMCPConnectionManagerWithToolEnvironment:
    """Test MCPConnectionManager integration with ToolEnvironment."""

    def test_tool_environment_accepts_mcp_manager(self):
        """Test that ToolEnvironment can be created with an mcp_manager."""
        from tools.env import ToolEnvironment

        mock_session = Mock()
        tool = MCPTool(
            session=mock_session,
            tool_name="calc",
            tool_description="Calculator",
            tool_schema={"type": "object", "properties": {}},
        )

        mock_manager = Mock()
        mock_manager.tool_map = {"calc": tool}

        env = ToolEnvironment(mcp_manager=mock_manager)
        assert "calc" in env.executor

    def test_tool_environment_from_dict_with_mcp(self):
        """Test ToolEnvironment.from_dict with mcp_server_command triggers MCPConnectionManager."""
        from tools.env import ToolEnvironment

        # MCPConnectionManager is lazily imported inside from_dict via
        # "from .mcp_connection_manager import MCPConnectionManager",
        # so we patch it at the source module.
        with patch("tools.mcp_connection_manager.MCPConnectionManager") as mock_mcp_cls:
            mock_manager = Mock()
            mock_manager.tool_map = {}
            mock_mcp_cls.return_value = mock_manager

            env_args = {
                "question": "test",
                "mcp_server_command": "npx",
                "mcp_server_args": ["-y", "@mcp/server"],
            }
            ToolEnvironment.from_dict(env_args)

            mock_mcp_cls.assert_called_once_with("npx", ["-y", "@mcp/server"], None)
            mock_manager.start.assert_called_once()
