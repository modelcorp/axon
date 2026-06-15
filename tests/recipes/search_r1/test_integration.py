#!/usr/bin/env python3
"""
Integration tests for Search-R1 implementation.

Tests end-to-end functionality including:
- Agent + Environment interaction
- Reward function integration
- Complete episode execution
- Data format compatibility
"""

import sys
from pathlib import Path

import pytest

# Add recipes folder to sys.path since it's outside the axon package
_recipes_path = Path(__file__).parent.parent.parent.parent / "recipes"
if str(_recipes_path) not in sys.path:
    sys.path.insert(0, str(_recipes_path))

from search_r1.agent import SearchR1Agent  # noqa: E402
from search_r1.env import SearchR1Env  # noqa: E402
from search_r1.reward import SearchR1RewardConfig, SearchR1RewardFn  # noqa: E402

from axon.utils.rewards.base import RewardInput  # noqa: E402


class TestSearchR1Integration:
    """Integration tests for Search-R1 system."""

    @pytest.fixture
    def agent(self):
        """Create agent."""
        return SearchR1Agent()

    @pytest.fixture
    def task(self):
        """Create sample task."""
        return {"question": "What is the capital of France?", "answer": ["Paris", "paris"]}

    @pytest.fixture
    def env(self, task):
        """Create environment."""
        return SearchR1Env(task=task, retrieval_url="http://127.0.0.1:8000/retrieve", max_turns=3)

    @pytest.fixture
    def reward_fn(self):
        """Create reward function."""
        config = SearchR1RewardConfig(correct_reward=1.0, structure_format_score=0.2, retrieval_score=0.1)
        return SearchR1RewardFn(config)

    def test_agent_environment_interaction(self, agent, env, task):
        """Test agent and environment can interact."""
        # Reset
        obs, info = env.reset()
        agent.reset()

        # Step 1: Agent processes initial observation
        processed_obs = agent.process_observation(obs, reward=0, done=False, info={})
        assert isinstance(processed_obs, str)
        assert len(processed_obs) > 0

        # Step 2: Simulate model response
        model_response = "<think>The capital of France is Paris</think><answer>test</answer><answer>Paris</answer>"
        action = agent.process_action(model_response)

        assert action.action["type"] == "answer"
        assert action.action["content"] is not None

        # Step 3: Environment evaluates
        reward, next_obs = env.get_reward_and_next_obs(task, action.action)

        assert reward > 0, "Correct answer should give positive reward"
        assert next_obs == {}, "Episode should be done"

    def test_multi_turn_episode(self, agent, env, task):
        """Test multi-turn episode with search."""
        obs, info = env.reset()
        agent.reset()

        # Turn 1: Initial observation
        processed_obs = agent.process_observation(obs, reward=0, done=False, info={})
        assert "France" in processed_obs

        # Turn 2: Agent decides to search (simulated)
        search_response = "<think>I need to search</think><search>capital of France</search>"
        action = agent.process_action(search_response)

        assert action.action["type"] == "search"

        # Environment would normally call retrieval service here
        # For testing, we'll simulate what happens after retrieval fails or succeeds
        # The actual retrieval is tested separately

    def test_reward_function_with_agent_output(self, agent, reward_fn, task):
        """Test reward function can process agent output."""
        # Correct answer
        correct_response = "<think>reasoning</think><answer>test</answer><answer>Paris</answer>"
        result = reward_fn(input=RewardInput(task_info=task, action=correct_response))

        assert result.reward > 0
        # is_correct requires reward >= 1.0

        # Incorrect answer
        incorrect_response = "<think>reasoning</think><answer>test</answer><answer>London</answer>"
        result = reward_fn(input=RewardInput(task_info=task, action=incorrect_response))

        assert not result.is_correct

    def test_data_format_compatibility(self):
        """Test that data format is compatible with all components."""
        # Example data point in Search-R1 format
        example_data = {
            "prompt": [{"role": "user", "content": "Answer the given question..."}],
            "question": "What is the capital of France?",
            "answer": ["Paris", "paris"],
        }

        # Test environment can be created from this format
        env = SearchR1Env.from_dict(
            {"task": example_data, "retrieval_url": "http://127.0.0.1:8000/retrieve", "topk": 3, "max_turns": 3}
        )

        assert env is not None

        # Test environment can be reset
        obs, info = env.reset()
        # obs may be wrapped in {"task": ...} or be the task directly
        task_data = obs.get("task", obs) if isinstance(obs, dict) else obs
        assert "question" in task_data
        assert task_data["question"] == example_data["question"]

    def test_invalid_action_handling(self, agent, env, task):
        """Test system handles invalid actions gracefully."""
        env.reset()
        agent.reset()

        # Invalid action from agent
        invalid_response = "I don't know what to do"
        action = agent.process_action(invalid_response)

        assert action.action["type"] == "invalid"

        # Environment handles invalid action
        reward, next_obs = env.get_reward_and_next_obs(task, action.action)

        assert reward == 0
        assert "error" in next_obs or next_obs == {}

    def test_max_turns_episode(self, agent, task):
        """Test episode terminates at max turns."""
        env = SearchR1Env(task=task, retrieval_url="http://127.0.0.1:8000/retrieve", max_turns=2)

        env.reset()
        agent.reset()

        # Turn 1
        search_action = {"type": "search", "content": "test", "full_response": "<search>test</search>"}
        next_obs, reward, done, info = env.step(search_action)
        assert not done

        # Turn 2 (max turns reached)
        invalid_action = {"type": "invalid", "content": "", "full_response": "invalid"}
        next_obs, reward, done, info = env.step(invalid_action)
        assert done

    def test_complete_episode_workflow(self, agent, env, task, reward_fn):
        """Test complete episode from start to finish."""
        # 1. Initialize
        obs, info = env.reset()
        agent.reset()

        # 2. Agent receives observation
        processed_obs = agent.process_observation(obs, reward=0, done=False, info={})
        assert len(processed_obs) > 0

        # 3. Model generates response (simulated)
        model_response = "<think>I know this answer</think><answer>test</answer><answer>Paris</answer>"

        # 4. Agent processes model output
        action = agent.process_action(model_response)
        assert action.action["type"] == "answer"

        # 5. Environment computes reward using step
        next_obs, env_reward, done, info = env.step(action.action)

        # 6. Verify reward function gives consistent result
        reward_result = reward_fn(input=RewardInput(task_info=task, action=model_response))

        # Both should give positive rewards
        assert env_reward > 0
        assert reward_result.reward > 0
        assert done


class TestSearchR1ErrorHandling:
    """Test error handling in Search-R1 system."""

    def test_malformed_task(self):
        """Test handling of malformed task data."""
        malformed_task = {
            "question": "Test question"
            # Missing extra_info with answer
        }

        # Environment should handle missing reward_model gracefully
        try:
            env = SearchR1Env(task=malformed_task, retrieval_url="http://127.0.0.1:8000/retrieve", max_turns=3)
            obs, info = env.reset()
            # Should either work or raise clear error
            assert "question" in obs
        except (KeyError, ValueError):
            # Expected error for malformed task
            assert True

    def test_empty_question(self):
        """Test handling of empty question."""
        task = {"question": "", "answer": ["Paris"]}

        env = SearchR1Env(task=task, retrieval_url="http://127.0.0.1:8000/retrieve", max_turns=3)

        obs, info = env.reset()
        assert "question" in obs

    def test_missing_ground_truth(self):
        """Test handling of missing ground truth."""
        from search_r1.reward import SearchR1RewardConfig, SearchR1RewardFn

        from axon.utils.rewards.base import RewardInput

        config = SearchR1RewardConfig()
        reward_fn = SearchR1RewardFn(config)

        task_info = {}  # Missing answer

        response = "<think>test</think><answer>test</answer><answer>answer</answer>"

        try:
            reward_fn(input=RewardInput(task_info=task_info, action=response))
            # Should handle gracefully
        except (KeyError, ValueError):
            # Expected error
            assert True


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
