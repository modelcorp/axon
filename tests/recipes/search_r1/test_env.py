#!/usr/bin/env python3
"""
Test suite for Search-R1 Environment.

Tests the SearchR1Env class including:
- Environment initialization
- Reset functionality
- Step execution
- Reward computation
- Terminal condition handling
"""

import sys
from pathlib import Path

import pytest

# Add recipes folder to sys.path since it's outside the axon package
_recipes_path = Path(__file__).parent.parent.parent.parent / "recipes"
if str(_recipes_path) not in sys.path:
    sys.path.insert(0, str(_recipes_path))

from search_r1.env import SearchR1Env  # noqa: E402


class TestSearchR1Env:
    """Test suite for SearchR1Env."""

    @pytest.fixture
    def task(self):
        """Create sample task."""
        return {"question": "What is the capital of France?", "answer": ["Paris"]}

    @pytest.fixture
    def env(self, task):
        """Create environment instance."""
        return SearchR1Env(task=task, retrieval_url="http://127.0.0.1:8000/retrieve", topk=3, max_turns=3)

    def test_environment_creation(self, env):
        """Test environment can be created."""
        assert env is not None
        assert env.max_turns == 3
        assert env.topk == 3

    def test_environment_reset(self, env, task):
        """Test environment reset."""
        obs, info = env.reset()

        assert "question" in obs, "Observation should contain question"
        assert obs["question"] == task["question"]
        assert isinstance(info, dict), "Info should be a dictionary"

    def test_invalid_action(self, task):
        """Test handling of invalid action."""
        # Create env with terminate_on_incorrect_action=False to get error message
        env = SearchR1Env(
            task=task,
            retrieval_url="http://127.0.0.1:8000/retrieve",
            topk=3,
            max_turns=3,
            terminate_on_incorrect_action=False,
        )
        env.reset()

        invalid_action = {"type": "invalid", "content": "", "full_response": "bad action"}

        reward, next_obs = env.get_reward_and_next_obs(task, invalid_action)

        assert reward == 0, "Invalid action should give zero reward"
        assert "error" in next_obs, "Next observation should contain error"

    def test_answer_action_step(self, env, task):
        """Test answer action using step() method."""
        env.reset()

        # Note: Need 2+ <answer> tags for Search-R1's extraction logic
        answer_action = {
            "type": "answer",
            "content": "Paris",
            "full_response": "<think>reasoning</think><answer>wrong</answer><answer>Paris</answer>",
        }

        next_obs, reward, done, info = env.step(answer_action)

        assert reward > 0, "Correct answer should give positive reward"
        assert done == True, "Episode should be done after answer"
        assert next_obs == {}, "Next obs should be empty when done"

    def test_max_turns_termination(self, task):
        """Test max_turns termination."""
        env = SearchR1Env(task=task, retrieval_url="http://127.0.0.1:8000/retrieve", max_turns=2)
        env.reset()

        # First turn: search (should not be done)
        search_action = {
            "type": "search",
            "content": "test query",
            "full_response": "<think>need info</think><search>test query</search>",
        }
        next_obs, reward, done, info = env.step(search_action)

        assert done == False, "Should not be done after first search"
        assert env.current_turn == 1

        # Second turn: invalid action (should be done due to max_turns)
        invalid_action = {"type": "invalid", "content": "", "full_response": "invalid response"}
        next_obs, reward, done, info = env.step(invalid_action)

        assert done == True, "Should be done after reaching max_turns"
        assert env.current_turn == 2

    def test_from_dict_creation(self, task):
        """Test environment creation from dict."""
        env_dict = {"task": task, "retrieval_url": "http://127.0.0.1:8000/retrieve", "topk": 3, "max_turns": 3}

        env = SearchR1Env.from_dict(env_dict)

        assert env is not None
        assert env.max_turns == 3
        assert env.topk == 3

    def test_search_action_without_retrieval(self, env, task):
        """Test search action (without actual retrieval server)."""
        env.reset()

        search_action = {
            "type": "search",
            "content": "capital of France",
            "full_response": "<think>need to search</think><search>capital of France</search>",
        }

        # This will fail to retrieve but should handle gracefully
        try:
            next_obs, reward, done, info = env.step(search_action)
            # If retrieval fails, it should still return some observation
            assert isinstance(next_obs, dict)
        except Exception as e:
            # Expected to fail without actual retrieval server
            assert "retrieval" in str(e).lower() or "connection" in str(e).lower()

    def test_turn_counter(self, env):
        """Test turn counter increments properly."""
        env.reset()
        assert env.current_turn == 0

        # Make a search action
        search_action = {"type": "search", "content": "test", "full_response": "<search>test</search>"}

        try:
            env.step(search_action)
            assert env.current_turn == 1
        except Exception:
            # May fail without retrieval server, but turn should still increment
            assert env.current_turn == 1


class TestSearchR1EnvIntegration:
    """Integration tests for SearchR1Env with agent."""

    def test_environment_agent_compatibility(self):
        """Test that environment works with agent."""
        from search_r1.agent import SearchR1Agent

        agent = SearchR1Agent()
        task = {"question": "What is the capital of France?", "answer": ["Paris"]}

        env = SearchR1Env(task=task, retrieval_url="http://127.0.0.1:8000/retrieve", max_turns=3)

        # Reset
        obs, info = env.reset()
        agent.reset()

        # Process observation
        processed_obs = agent.process_observation(obs, reward=0, done=False, info={})
        assert isinstance(processed_obs, str)

        # Simulate model response
        model_response = "<think>Paris is the capital</think><answer>test</answer><answer>Paris</answer>"
        action = agent.process_action(model_response)

        # Environment evaluates
        reward, next_obs = env.get_reward_and_next_obs(task, action.action)

        assert reward > 0, "Correct answer should give positive reward"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
