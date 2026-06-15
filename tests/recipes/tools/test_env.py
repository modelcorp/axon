import json
import sys
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import Mock

# Add recipes folder to sys.path since it's outside the axon package
_recipes_path = Path(__file__).parent.parent.parent.parent / "recipes"
if str(_recipes_path) not in sys.path:
    sys.path.insert(0, str(_recipes_path))

from tools.env import ToolEnvironment  # noqa: E402

from axon.tools.executors import LocalToolExecutor  # noqa: E402
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
        return ToolOutput(name=self.name, output=f"Mock result for {kwargs}")


@dataclass
class RewardOutput:
    reward: float = 0.0
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class MockRewardFunction:
    """Mock reward function for testing."""

    def __call__(self, task_info, action, **kwargs):
        reward = 1.0 if "correct" in str(action).lower() else 0.0
        metadata = {"evaluated": True, "action": action}
        return RewardOutput(reward=reward, metadata=metadata)


class TestToolEnvironment:
    """Test suite for ToolEnvironment class."""

    def test_init_default(self):
        """Test ToolEnvironment initialization with default parameters."""
        env = ToolEnvironment()
        assert env.step_count == 0
        assert env.max_turns == 10
        assert env.task is None
        assert env.reward_fn is not None  # Should use _zero_reward_fn

    def test_init_with_tool_map(self):
        """Test ToolEnvironment initialization with tool_map."""
        tool_map = {"mock_tool": MockTool()}
        env = ToolEnvironment(tool_map=tool_map)
        assert isinstance(env.executor, LocalToolExecutor)
        assert "mock_tool" in env.executor

    def test_init_with_executor(self):
        """Test ToolEnvironment initialization with a pre-built executor."""
        executor = LocalToolExecutor({"mock_tool": MockTool()})
        env = ToolEnvironment(executor=executor)
        assert env.executor is executor

    def test_init_with_mcp_manager(self):
        """Test ToolEnvironment initialization with mcp_manager."""
        mock_manager = Mock()
        mock_manager.tool_map = {"mcp_tool": MockTool()}
        env = ToolEnvironment(mcp_manager=mock_manager)
        assert isinstance(env.executor, LocalToolExecutor)

    def test_init_with_custom_parameters(self):
        """Test ToolEnvironment initialization with custom parameters."""
        task = {"question": "Test question"}
        reward_fn = MockRewardFunction()
        max_steps = 5

        env = ToolEnvironment(task=task, reward_fn=reward_fn, max_turns=max_steps)

        assert env.task == task
        assert env.reward_fn == reward_fn
        assert env.max_turns == max_steps

    def test_init_no_reward_function_default(self):
        """Test that default reward function is used when none provided."""
        env = ToolEnvironment(reward_fn=None)
        assert env.reward_fn is not None

    def test_reset(self):
        """Test the reset method."""
        task = {"question": "Test question"}
        env = ToolEnvironment(task=task, max_turns=5)

        # Set some non-initial state
        env.step_count = 3

        obs, info = env.reset()

        assert env.step_count == 0
        assert obs == task
        assert isinstance(info, dict)

    def test_reset_empty_task(self):
        """Test reset when no task is set."""
        env = ToolEnvironment()
        obs, info = env.reset()
        assert obs == {}
        assert isinstance(info, dict)

    def test_step_with_string_action(self):
        """Test stepping with string action (should terminate)."""
        task = {"question": "Test question"}
        reward_fn = MockRewardFunction()
        env = ToolEnvironment(task=task, reward_fn=reward_fn)
        env.reset()

        action = "This is my final answer"
        obs, reward, done, info = env.step(action)

        assert obs == {}
        assert reward == 0.0  # MockRewardFunction returns 0 for non-"correct" answers
        assert done is True
        assert info["response"] == action
        assert "metadata" in info

    def test_step_with_string_action_correct(self):
        """Test stepping with string action containing 'correct'."""
        task = {"question": "Test question"}
        reward_fn = MockRewardFunction()
        env = ToolEnvironment(task=task, reward_fn=reward_fn)
        env.reset()

        action = "The correct answer is 42"
        obs, reward, done, info = env.step(action)

        assert reward == 1.0
        assert done is True

    def test_step_with_finish_tool_call(self):
        """Test stepping with finish tool call."""
        task = {"question": "Test question"}
        reward_fn = MockRewardFunction()
        env = ToolEnvironment(task=task, reward_fn=reward_fn)
        env.reset()

        action = [{"id": "call_1", "function": {"name": "finish", "arguments": '{"response": "Final answer"}'}}]

        obs, reward, done, info = env.step(action)

        assert obs == {}
        assert done is True
        assert info["response"] == action

    def test_step_with_regular_tool_calls(self):
        """Test stepping with regular tool calls."""
        tool_map = {"mock_tool": MockTool()}
        env = ToolEnvironment(tool_map=tool_map)
        env.reset()

        action = [{"id": "call_1", "function": {"name": "mock_tool", "arguments": '{"query": "test"}'}}]

        obs, reward, done, info = env.step(action)

        assert "tool_outputs" in obs
        assert "call_1" in obs["tool_outputs"]
        assert "Mock result" in obs["tool_outputs"]["call_1"]
        assert reward == 0
        assert done is False

    def test_step_max_steps_termination(self):
        """Test that environment terminates when max_steps is reached."""
        tool_map = {"mock_tool": MockTool()}
        env = ToolEnvironment(tool_map=tool_map, max_turns=2)
        env.reset()

        action = [{"id": "call_1", "function": {"name": "mock_tool", "arguments": '{"query": "test"}'}}]

        # Step 1: not done
        obs1, reward1, done1, info1 = env.step(action)
        assert done1 is False

        # Step 2: done due to max_steps
        obs2, reward2, done2, info2 = env.step(action)
        assert done2 is True

    def test_step_with_dict_action(self):
        """Test stepping with dict action (converted to list)."""
        tool_map = {"mock_tool": MockTool()}
        env = ToolEnvironment(tool_map=tool_map)
        env.reset()

        action = {"id": "call_1", "function": {"name": "mock_tool", "arguments": '{"query": "test"}'}}

        obs, reward, done, info = env.step(action)

        # Should convert dict to list and execute
        assert "tool_outputs" in obs
        assert "call_1" in obs["tool_outputs"]

    def test_step_with_multiple_tool_calls(self):
        """Test stepping with multiple tool calls."""
        tool_map = {"mock_tool": MockTool()}
        env = ToolEnvironment(tool_map=tool_map)
        env.reset()

        action = [
            {"id": "call_1", "function": {"name": "mock_tool", "arguments": '{"query": "test1"}'}},
            {"id": "call_2", "function": {"name": "mock_tool", "arguments": '{"query": "test2"}'}},
        ]

        obs, reward, done, info = env.step(action)

        assert len(obs["tool_outputs"]) == 2
        assert "call_1" in obs["tool_outputs"]
        assert "call_2" in obs["tool_outputs"]

    def test_step_with_unknown_tool(self):
        """Test stepping with an unknown tool name."""
        env = ToolEnvironment()
        env.reset()

        action = [{"id": "call_1", "function": {"name": "nonexistent_tool", "arguments": '{"query": "test"}'}}]

        obs, reward, done, info = env.step(action)

        # Should return error in tool output
        assert "tool_outputs" in obs
        assert "call_1" in obs["tool_outputs"]
        assert "error" in obs["tool_outputs"]["call_1"].lower() or "unknown" in obs["tool_outputs"]["call_1"].lower()

    def test_idx_property(self):
        """Test the idx property from BaseEnv."""
        env = ToolEnvironment()
        assert env.idx is None
        env.idx = 5
        assert env.idx == 5

    def test_is_multithread_safe(self):
        """Test the is_multithread_safe static method."""
        assert ToolEnvironment.is_multithread_safe() is True

    def test_from_dict(self):
        """Test creating environment from dictionary."""
        env_args = {
            "question": "Test question",
            "max_turns": 15,
            "reward_fn": MockRewardFunction(),
        }

        env = ToolEnvironment.from_dict(env_args)

        assert isinstance(env, ToolEnvironment)
        assert env.max_turns == 15
        assert env.task == {"question": "Test question"}

    def test_from_dict_with_tool_map(self):
        """Test creating environment from dictionary with tool_map."""
        tool_map = {"mock_tool": MockTool()}
        env_args = {"question": "Test question", "tool_map": tool_map, "max_turns": 20}

        env = ToolEnvironment.from_dict(env_args)

        assert isinstance(env, ToolEnvironment)
        assert env.max_turns == 20
        assert "mock_tool" in env.executor

    def test_close_method(self):
        """Test the close method (inherited from BaseEnv)."""
        env = ToolEnvironment()
        env.close()

    def test_full_interaction_flow(self):
        """Test a complete interaction flow."""
        task = {"question": "What is 2 + 2?"}
        reward_fn = MockRewardFunction()
        tool_map = {"mock_tool": MockTool()}
        env = ToolEnvironment(task=task, reward_fn=reward_fn, tool_map=tool_map, max_turns=3)

        # Reset environment
        obs, info = env.reset()
        assert obs == task
        assert env.step_count == 0

        # Step 1: Use a tool
        action1 = [{"id": "call_1", "function": {"name": "mock_tool", "arguments": '{"query": "2 + 2"}'}}]
        obs1, reward1, done1, info1 = env.step(action1)

        assert "tool_outputs" in obs1
        assert reward1 == 0
        assert done1 is False
        assert env.step_count == 1

        # Step 2: Finish with answer containing "correct"
        action2 = "The correct answer is 4"
        obs2, reward2, done2, info2 = env.step(action2)

        assert obs2 == {}
        assert reward2 == 1.0
        assert done2 is True
        assert info2["response"] == action2

    def test_finish_tool_call_extracts_response(self):
        """Test that finish tool call arguments are properly extracted for reward."""
        task = {"question": "Test"}
        reward_fn = MockRewardFunction()
        env = ToolEnvironment(task=task, reward_fn=reward_fn)
        env.reset()

        # JSON string arguments
        action = [
            {
                "id": "call_1",
                "function": {"name": "finish", "arguments": json.dumps({"response": "The correct answer"})},
            }
        ]
        obs, reward, done, info = env.step(action)

        assert done is True
        assert reward == 1.0  # "correct" is in the response

    def test_reward_function_integration(self):
        """Test integration with different reward functions."""

        class CustomReward:
            def __call__(self, task_info, action, **kwargs):
                score = len(str(action)) / 10.0
                return RewardOutput(reward=score, metadata={"length": len(str(action))})

        task = {"question": "Test"}
        reward_fn = CustomReward()
        env = ToolEnvironment(task=task, reward_fn=reward_fn)

        # Test with different length responses
        env.reset()
        short_action = "Short"
        _, reward1, _, info1 = env.step(short_action)

        env.reset()
        long_action = "This is a much longer response that should get a higher reward"
        _, reward2, _, info2 = env.step(long_action)

        assert reward2 > reward1
        assert info1["metadata"]["length"] < info2["metadata"]["length"]
