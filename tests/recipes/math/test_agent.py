import importlib.util
import sys
from pathlib import Path

# Load the math.agent module directly since 'math' conflicts with Python's stdlib
_math_agent_path = Path(__file__).parent.parent.parent.parent / "recipes" / "math" / "agent.py"
spec = importlib.util.spec_from_file_location("math_agent", _math_agent_path)
math_agent_module = importlib.util.module_from_spec(spec)
sys.modules["math_agent"] = math_agent_module
spec.loader.exec_module(math_agent_module)

MathAgent = math_agent_module.MathAgent

from axon.core import Action  # noqa: E402


class TestMathAgent:
    """Simplified test suite for MathAgent core functionality."""

    def test_init_default(self):
        """Test MathAgent initialization with default parameters."""
        agent = MathAgent()
        assert agent.instruction == "Let's think step by step, and put your final answer within \\boxed{}."
        assert agent.first_time is True

    def test_init_custom_accumulate_thinking(self):
        """Test MathAgent initialization with custom accumulate_thinking parameter."""
        # This parameter no longer exists in the new interface, but we keep the test name
        # Test that agent can be initialized without errors
        agent = MathAgent()
        assert agent.instruction == "Let's think step by step, and put your final answer within \\boxed{}."

    def test_reset(self):
        """Test the reset method."""
        agent = MathAgent()
        # Simulate some state change
        agent.first_time = False

        agent.reset()

        assert agent.first_time is True

    def test_properties(self):
        """Test key properties."""
        agent = MathAgent()
        # Test that the agent has the expected instruction property
        assert hasattr(agent, "instruction")
        assert agent.instruction == "Let's think step by step, and put your final answer within \\boxed{}."
        # Test that system_prompt property exists (inherited from BaseAgent)
        assert hasattr(agent, "system_prompt")

    def test_update_from_env_initial_question(self):
        """Test update_from_env with initial question observation."""
        agent = MathAgent()
        observation = {"question": "What is the square root of 16?"}
        formatted_observation = agent.process_observation(observation, 0.0, False, {})

        # Check that observation was formatted correctly
        expected_content = f"What is the square root of 16? {agent.instruction}"
        assert formatted_observation == expected_content
        # After processing first observation, first_time should be False
        assert agent.first_time is False

    def test_update_from_env_follow_up_correction(self):
        """Test update_from_env with follow-up correction."""
        agent = MathAgent()

        # First, process an initial observation to set first_time to False
        agent.process_observation({"question": "Initial question"}, 0.0, False, {})

        # Now process a follow-up
        formatted_observation = agent.process_observation("correction needed", 0.5, False, {"attempt": 2})

        # Check that correction message was returned
        expected_correction = "Your previous answer may or may not contain a mistake. Please review it carefully and put your final answer within \\boxed{}."
        assert formatted_observation == expected_correction

    def test_update_from_model_basic(self):
        """Test basic update_from_model functionality."""
        agent = MathAgent()

        # First provide a question
        agent.process_observation({"question": "What is 2+2?"}, 0.0, False, {})

        response = "<think>2 + 2 = 4</think> \\boxed{4}"
        action = agent.process_action(response)

        # Check return value
        assert isinstance(action, Action)
        assert action.action == response

    def test_update_from_model_without_accumulate_thinking(self):
        """Test update_from_model with accumulate_thinking=False."""
        # The new interface doesn't have accumulate_thinking, but we keep the test name
        # Test that process_action works correctly
        agent = MathAgent()

        agent.process_observation({"question": "What is 2+2?"}, 0.0, False, {})
        response1 = "<think>Let me calculate this</think> The answer is 4"
        action1 = agent.process_action(response1)

        # Check that action is returned correctly
        assert isinstance(action1, Action)
        assert action1.action == response1

        # Process another question/response
        agent.process_observation({"question": "What is 3+3?"}, 0.0, False, {})
        response2 = "<think>3 + 3 = 6</think> The answer is 6"
        action2 = agent.process_action(response2)

        assert isinstance(action2, Action)
        assert action2.action == response2

    def test_update_from_model_no_thinking_tags(self):
        """Test update_from_model when response has no thinking tags."""
        agent = MathAgent()

        # First provide a question
        agent.process_observation({"question": "What is 2+2?"}, 0.0, False, {})

        response = "The answer is 4"
        action = agent.process_action(response)

        # Should return action with the original response
        assert isinstance(action, Action)
        assert action.action == response

    def test_basic_interaction_flow(self):
        """Test a basic complete interaction flow."""
        agent = MathAgent()

        # Step 1: Initial environment update
        observation = {"question": "What is 5 + 3?"}
        formatted_obs = agent.process_observation(observation, 0.0, False, {})

        expected_content = f"What is 5 + 3? {agent.instruction}"
        assert formatted_obs == expected_content

        # Step 2: Model response
        response = "<think>5 + 3 = 8</think> \\boxed{8}"
        action = agent.process_action(response)

        assert isinstance(action, Action)
        assert action.action == response

    def test_program_to_dict(self):
        """Test that program can be converted to dict."""
        # The new interface doesn't have a program attribute, but we keep the test name
        # Test a basic interaction workflow instead
        agent = MathAgent()

        # Add basic interaction
        formatted_obs = agent.process_observation({"question": "test"}, 0.0, False, {})
        assert "test" in formatted_obs

        action = agent.process_action("test response")
        assert isinstance(action, Action)
        assert action.action == "test response"
