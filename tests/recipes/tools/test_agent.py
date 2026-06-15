import json
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

# Add recipes folder to sys.path since it's outside the axon package
_recipes_path = Path(__file__).parent.parent.parent.parent / "recipes"
if str(_recipes_path) not in sys.path:
    sys.path.insert(0, str(_recipes_path))

from tools.agent import ToolAgent  # noqa: E402

from axon.core import Action  # noqa: E402
from axon.tools.tools import Tool  # noqa: E402
from axon.tools.types import ToolOutput  # noqa: E402


class MockTool(Tool):
    """Mock tool for testing."""

    def __init__(self):
        super().__init__(name="mock_tool", description="A mock tool for testing")

    @property
    def json(self):
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string", "description": "Test query"}},
                    "required": ["query"],
                },
            },
        }

    def forward(self, **kwargs):
        return ToolOutput(name=self.name, output=f"Mock tool called with {kwargs}")


class TestToolAgent:
    """Test suite for ToolAgent core functionality."""

    def _make_agent(self, **kwargs):
        """Create a ToolAgent with a mock tool_map."""
        defaults = {"tool_map": {"mock_tool": MockTool()}}
        defaults.update(kwargs)
        return ToolAgent(**defaults)

    def test_init_with_tool_map(self):
        """Test ToolAgent initialization with tool_map."""
        agent = self._make_agent()
        assert agent.system_prompt is not None
        assert agent.tool_parser is not None

    def test_init_requires_tool_map(self):
        """Test that ToolAgent requires tool_map."""
        with pytest.raises(AssertionError):
            ToolAgent()

    def test_system_prompt_includes_tools(self):
        """Test that system prompt includes tool descriptions."""
        agent = self._make_agent()
        prompt = agent.system_prompt
        # The tool prompt should be appended to the base system prompt
        assert "mock_tool" in prompt or "tools" in prompt.lower()

    def test_custom_system_prompt(self):
        """Test ToolAgent with a custom system prompt."""
        agent = self._make_agent(system_prompt="Custom prompt")
        assert "Custom prompt" in agent.system_prompt

    def test_reset(self):
        """Test the reset method."""
        agent = self._make_agent()
        # Reset should not raise any errors
        agent.reset()

    def test_process_observation_question(self):
        """Test process_observation with a question dict."""
        agent = self._make_agent()
        obs = {"question": "What is the weather?"}
        processed = agent.process_observation(obs, 0.0, False, {})
        assert processed == "What is the weather?"

    def test_process_observation_string(self):
        """Test process_observation with a plain string."""
        agent = self._make_agent()
        obs = "Hello, how can I help?"
        processed = agent.process_observation(obs, 0.0, False, {})
        assert processed == "Hello, how can I help?"

    def test_process_observation_tool_outputs(self):
        """Test process_observation with tool_outputs dict."""
        agent = self._make_agent()
        obs = {"tool_outputs": {"call_1": "Weather is sunny, 25°C"}}
        processed = agent.process_observation(obs, 0.0, False, {})
        assert isinstance(processed, list)
        assert len(processed) == 1
        assert processed[0]["role"] == "tool"
        assert processed[0]["content"] == "Weather is sunny, 25°C"
        assert processed[0]["tool_call_id"] == "call_1"

    def test_process_observation_generic_dict(self):
        """Test process_observation with a generic dict (no question/tool_outputs key)."""
        agent = self._make_agent()
        obs = {"some_key": "some_value"}
        processed = agent.process_observation(obs, 0.0, False, {})
        assert isinstance(processed, str)

    def test_process_action_no_tool_calls(self):
        """Test process_action when no tool calls are found in text."""
        agent = self._make_agent()
        response = "This is a plain text response with no tool calls."
        action = agent.process_action(response)

        # Should create a finish tool call
        assert isinstance(action, Action)
        assert isinstance(action.action, list)
        assert len(action.action) == 1
        assert action.action[0]["function"]["name"] == "finish"
        args = json.loads(action.action[0]["function"]["arguments"])
        assert args["response"] == response

    def test_process_action_with_parsing_error(self):
        """Test process_action when tool parsing raises an exception."""
        agent = self._make_agent()
        # Patch the parser to raise an exception
        agent.tool_parser.parse = Mock(side_effect=Exception("Parsing failed"))

        response = "This will fail to parse."
        action = agent.process_action(response)

        # Should fall through to finish
        assert isinstance(action, Action)
        assert action.action[0]["function"]["name"] == "finish"

    def test_process_action_with_tool_calls(self):
        """Test process_action when the parser finds tool calls."""
        agent = self._make_agent()

        # Create a mock parsed tool call
        mock_tool_call = Mock()
        mock_tool_call.to_openai_dict.return_value = {
            "id": "call_123",
            "type": "function",
            "function": {"name": "mock_tool", "arguments": '{"query": "test"}'},
        }
        agent.tool_parser.parse = Mock(return_value=([mock_tool_call], "remaining"))

        response = "Let me search for that."
        action = agent.process_action(response)

        assert isinstance(action, Action)
        assert isinstance(action.action, list)
        assert len(action.action) == 1
        assert action.action[0]["function"]["name"] == "mock_tool"

    def test_basic_interaction_flow(self):
        """Test a basic complete interaction flow."""
        agent = self._make_agent()

        # Step 1: Initial environment observation
        obs = {"question": "What's the weather?"}
        processed_obs = agent.process_observation(obs, 0.0, False, {})
        assert processed_obs == "What's the weather?"

        # Step 2: Model response with no tool calls → finish
        response = "The weather is sunny."
        action = agent.process_action(response)
        assert isinstance(action, Action)
        assert action.action[0]["function"]["name"] == "finish"

    def test_tool_output_interaction_flow(self):
        """Test processing tool outputs from the environment."""
        agent = self._make_agent()

        # Simulate receiving tool outputs
        tool_outputs_obs = {"tool_outputs": {"call_1": "Result: 42", "call_2": "Result: hello"}}
        processed = agent.process_observation(tool_outputs_obs, 0.0, False, {})

        assert isinstance(processed, list)
        assert len(processed) == 2
        for msg in processed:
            assert msg["role"] == "tool"
            assert "tool_call_id" in msg
            assert "content" in msg
