import sys
from pathlib import Path

# Add recipes/frozenlake folder to sys.path since agent.py imports from env directly
_frozenlake_path = Path(__file__).parent.parent.parent.parent / "recipes" / "frozenlake"
if str(_frozenlake_path) not in sys.path:
    sys.path.insert(0, str(_frozenlake_path))

from agent import FrozenLakeAgent  # noqa: E402

from axon.core import Action  # noqa: E402


class TestFrozenLakeAgent:
    """Simplified test suite for FrozenLakeAgent core functionality."""

    def test_init_default(self):
        """Test FrozenLakeAgent initialization with default parameters."""
        agent = FrozenLakeAgent()
        assert agent.step == 0
        assert agent.multistep_prompt is False
        assert "FrozenLake Quick Guide" in agent.system_prompt

    def test_init_with_params(self):
        """Test FrozenLakeAgent initialization with custom parameters."""
        agent = FrozenLakeAgent(use_multistep_prompt=True)
        assert agent.multistep_prompt is True
        assert "Example1:" in agent.system_prompt  # Multi-shot prompt has examples

    def test_reset(self):
        """Test the reset method."""
        agent = FrozenLakeAgent()

        # Add some state to reset
        agent.step = 5
        agent.last_observation = "some observation"

        agent.reset()

        assert agent.step == 0
        assert agent.last_observation is None

    def test_properties(self):
        """Test key properties."""
        agent = FrozenLakeAgent()
        assert hasattr(agent, "system_prompt")
        assert isinstance(agent.system_prompt, str)
        assert agent.step == 0
        assert isinstance(agent.multistep_prompt, bool)

    def test_update_from_env_basic(self):
        """Test basic process_observation functionality."""
        agent = FrozenLakeAgent()

        observation = "P _ _ G\n_ O _ _\n_ _ _ _\n_ _ _ _"
        processed_obs = agent.process_observation(observation, 0.0, False, {})

        # Check that observation was processed correctly - agent formats it
        assert "Current Observation:" in processed_obs
        assert observation in processed_obs
        assert "Please give the next action" in processed_obs
        assert agent.last_observation == processed_obs

    def test_update_from_model_basic(self):
        """Test basic process_action functionality."""
        agent = FrozenLakeAgent()

        response = "I need to move right. ```Right```"
        action = agent.process_action(response)

        # Check that action was parsed correctly
        assert isinstance(action, Action)
        assert action.action == "3"  # Right = 3
        assert action.thought == "I need to move right."

    def test_parse_model_response(self):
        """Test model response parsing for different directions."""
        agent = FrozenLakeAgent()

        test_cases = [
            ("I'll move ```Right```", "I'll move", "3"),
            ("Going up. ```Up```", "Going up.", "4"),
            ("Moving ```Down```", "Moving", "2"),
            ("Turn ```Left```", "Turn", "1"),
            ("Invalid move ```diagonal```", "Invalid move", "0"),  # Invalid action
            ("No action here", "No action here", "0"),  # No code block
        ]

        for response, expected_thought, expected_action in test_cases:
            action = agent.process_action(response)
            assert action.thought == expected_thought
            assert action.action == expected_action

    def test_basic_interaction_flow(self):
        """Test a basic complete interaction flow."""
        agent = FrozenLakeAgent()

        # Step 1: Environment provides observation
        observation = "P _ G"
        processed_obs = agent.process_observation(observation, 0.0, False, {})
        assert "Current Observation:" in processed_obs
        assert observation in processed_obs

        # Step 2: Model responds with action
        response = "I need to move right. ```Right```"
        action = agent.process_action(response)

        assert isinstance(action, Action)
        assert action.action == "3"  # Right
        assert action.thought == "I need to move right."

        # Step 3: Environment provides new observation (could be goal reached)
        new_observation = "_ P G"
        processed_obs = agent.process_observation(new_observation, 1.0, True, {"success": True})
        assert "Current Observation:" in processed_obs
        assert new_observation in processed_obs

    def test_program_to_dict(self):
        """Test that agent maintains state correctly across interactions."""
        agent = FrozenLakeAgent()

        # Add basic interaction
        obs1 = "P _ G"
        agent.process_observation(obs1, 0.0, False, {})
        action1 = agent.process_action("Moving right. ```Right```")

        # Verify state - agent formats observation
        assert obs1 in agent.last_observation
        assert action1.action == "3"

        # Add another interaction
        obs2 = "_ P G"
        agent.process_observation(obs2, 0.0, False, {})
        action2 = agent.process_action("Moving down. ```Down```")

        # Verify state updated
        assert obs2 in agent.last_observation
        assert action2.action == "2"
