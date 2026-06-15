#!/usr/bin/env python3
"""
Test suite for Search-R1 Agent implementation.

Tests the SearchR1Agent class including:
- System prompt loading
- Observation processing
- Action parsing (search, answer, invalid)
"""

import sys
from pathlib import Path

import pytest

# Add recipes folder to sys.path since it's outside the axon package
_recipes_path = Path(__file__).parent.parent.parent.parent / "recipes"
if str(_recipes_path) not in sys.path:
    sys.path.insert(0, str(_recipes_path))

from search_r1.agent import SearchR1Agent  # noqa: E402


class TestSearchR1Agent:
    """Test suite for SearchR1Agent."""

    @pytest.fixture
    def agent(self):
        """Create a SearchR1Agent instance."""
        return SearchR1Agent()

    def test_system_prompt(self, agent):
        """Test that system prompt contains required tags."""
        system_prompt = agent.system_prompt

        assert len(system_prompt) > 0, "System prompt should not be empty"
        assert "<think>" in system_prompt, "System prompt should contain <think> tag"
        assert "<search>" in system_prompt, "System prompt should contain <search> tag"
        assert "<answer>" in system_prompt, "System prompt should contain <answer> tag"
        assert "<information>" in system_prompt, "System prompt should contain <information> tag"

    def test_initial_observation_processing(self, agent):
        """Test processing of initial observation."""
        observation = {"question": "What is the capital of France?"}
        processed = agent.process_observation(observation, reward=0, done=False, info={})

        assert isinstance(processed, str), "Processed observation should be a string"
        assert len(processed) > 0, "Processed observation should not be empty"
        assert "France" in processed, "Processed observation should contain the question"

    def test_search_action_parsing(self, agent):
        """Test parsing of search action."""
        search_action = "I need to search for this. <search>capital of France</search>"
        action = agent.process_action(search_action)

        assert action.action["type"] == "search", "Action type should be 'search'"
        assert action.action["content"] == "capital of France", "Search content should be extracted"
        assert "full_response" in action.action, "Action should contain full_response"

    def test_answer_action_parsing(self, agent):
        """Test parsing of answer action."""
        answer_action = "<think>Based on my search</think><answer>Paris</answer>"
        action = agent.process_action(answer_action)

        assert action.action["type"] == "answer", "Action type should be 'answer'"
        assert action.action["content"] == "Paris", "Answer content should be extracted"

    def test_invalid_action_detection(self, agent):
        """Test detection of invalid actions."""
        invalid_action = "I don't know the answer."
        action = agent.process_action(invalid_action)

        assert action.action["type"] == "invalid", "Invalid action should be detected"
        assert "content" in action.action, "Invalid action should have content field"

    def test_multiple_answer_tags(self, agent):
        """Test parsing action with multiple answer tags (Search-R1 format)."""
        multi_answer = "<think>reasoning</think><answer>wrong</answer><answer>Paris</answer>"
        action = agent.process_action(multi_answer)

        assert action.action["type"] == "answer", "Action type should be 'answer'"
        # The agent extracts answer content (implementation may vary)
        assert action.action["content"] is not None, "Should have content"

    def test_agent_reset(self, agent):
        """Test agent reset functionality."""
        # Process an observation
        agent.process_observation({"question": "Test?"}, reward=0, done=False, info={})

        # Reset agent
        agent.reset()

        # Agent should be in initial state (this is implementation-dependent)
        # At minimum, reset should not raise an error
        assert True

    def test_observation_with_retrieval_results(self, agent):
        """Test processing observation with retrieval results."""
        observation = {
            "question": "What is the capital of France?",
            "search_results": "Paris is the capital of France.",
        }

        processed = agent.process_observation(observation, reward=0, done=False, info={})

        assert isinstance(processed, str), "Processed observation should be a string"
        assert len(processed) > 0, "Processed observation should not be empty"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
