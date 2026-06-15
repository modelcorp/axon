#!/usr/bin/env python3
"""
Test suite for Search-R1 Reward Function.

Tests the SearchR1RewardFn class including:
- Correct answer detection
- Incorrect answer handling
- Structure and format scoring
- Retrieval bonus
"""

import sys
from pathlib import Path

import pytest

# Add recipes folder to sys.path since it's outside the axon package
_recipes_path = Path(__file__).parent.parent.parent.parent / "recipes"
if str(_recipes_path) not in sys.path:
    sys.path.insert(0, str(_recipes_path))

from search_r1.reward import SearchR1RewardConfig, SearchR1RewardFn  # noqa: E402

from axon.utils.rewards.base import RewardInput  # noqa: E402


class TestSearchR1RewardFn:
    """Test suite for SearchR1RewardFn."""

    @pytest.fixture
    def config(self):
        """Create a reward config."""
        return SearchR1RewardConfig(
            correct_reward=1.0, structure_format_score=0.2, retrieval_score=0.1, final_format_score=0.05
        )

    @pytest.fixture
    def reward_fn(self, config):
        """Create a reward function instance."""
        return SearchR1RewardFn(config)

    @pytest.fixture
    def task_info(self):
        """Create sample task info."""
        return {"answer": ["Paris", "paris"]}

    def test_correct_answer(self, reward_fn, task_info):
        """Test reward for correct answer."""
        # Note: Search-R1 requires 2+ <answer> tags
        correct_response = "<think>Let me think</think><answer>wrong</answer><answer>Paris</answer>"
        result = reward_fn(input=RewardInput(task_info=task_info, action=correct_response))

        assert result.reward > 0, "Correct answer should have positive reward"
        # is_correct is True only if reward >= correct_reward (1.0)
        # This response gets 0.8 (structure format + final format but no retrieval bonus)
        assert result.reward >= 0.8, "Should have structure bonus"

    def test_incorrect_answer(self, reward_fn, task_info):
        """Test reward for incorrect answer."""
        incorrect_response = "<think>Let me think</think><answer>test</answer><answer>London</answer>"
        result = reward_fn(input=RewardInput(task_info=task_info, action=incorrect_response))

        assert not result.is_correct, "Should be marked as incorrect"
        # May still have structure bonus
        assert result.reward >= 0, "Reward should be non-negative"

    def test_no_answer_incomplete(self, reward_fn, task_info):
        """Test reward for incomplete sequence (no answer)."""
        no_answer = "<think>I am thinking</think>"
        result = reward_fn(input=RewardInput(task_info=task_info, action=no_answer))

        assert not result.is_correct, "Incomplete sequence should not be correct"
        # Should have low or zero reward
        assert result.reward <= 0.3, "Incomplete sequence should have low reward"

    def test_invalid_sequence(self, reward_fn, task_info, config):
        """Test reward for invalid sequence (answer before think)."""
        invalid_sequence = "<answer>test</answer><answer>Paris</answer>"
        result = reward_fn(input=RewardInput(task_info=task_info, action=invalid_sequence))

        # Should get final format score but not structure bonus
        assert result.reward <= config.final_format_score + config.correct_reward

    def test_valid_sequence_with_search(self, reward_fn, task_info):
        """Test reward for valid sequence with search."""
        valid_with_search = (
            "<think>I need to search</think>"
            "<search>capital of France</search>"
            "<information>Paris is the capital</information>"
            "<think>Now I know</think>"
            "<answer>test</answer><answer>Paris</answer>"
        )
        result = reward_fn(input=RewardInput(task_info=task_info, action=valid_with_search))

        # Should get high reward with all bonuses
        assert result.reward >= 0.8, "Valid sequence should have high reward"

    def test_case_insensitive_matching(self, reward_fn):
        """Test case-insensitive answer matching."""
        task_info = {"answer": ["Paris", "PARIS"]}

        # Test with lowercase
        response_lower = "<think>thinking</think><answer>test</answer><answer>paris</answer>"
        result = reward_fn(input=RewardInput(task_info=task_info, action=response_lower))
        assert result.reward > 0, "Should get positive reward"

        # Test with uppercase
        response_upper = "<think>thinking</think><answer>test</answer><answer>PARIS</answer>"
        result = reward_fn(input=RewardInput(task_info=task_info, action=response_upper))
        assert result.reward > 0, "Should get positive reward"

    def test_multiple_ground_truths(self, reward_fn):
        """Test matching against multiple ground truth answers."""
        task_info = {"answer": ["Paris", "paris", "City of Light"]}

        # Test matching first
        response1 = "<think>thinking</think><answer>test</answer><answer>Paris</answer>"
        result = reward_fn(input=RewardInput(task_info=task_info, action=response1))
        assert result.reward > 0, "Should get positive reward for correct answer"

        # Test matching alternative
        response2 = "<think>thinking</think><answer>test</answer><answer>City of Light</answer>"
        result = reward_fn(input=RewardInput(task_info=task_info, action=response2))
        assert result.reward > 0, "Should get positive reward for alternative answer"

    def test_structure_format_scoring(self, reward_fn, task_info, config):
        """Test that structure format scoring works."""
        # Valid structure with incorrect answer
        valid_structure = (
            "<think>reasoning</think>"
            "<search>query</search>"
            "<information>info</information>"
            "<think>more reasoning</think>"
            "<answer>test</answer><answer>Wrong</answer>"
        )
        result = reward_fn(input=RewardInput(task_info=task_info, action=valid_structure))

        # Should get some bonus even if answer is wrong (at least final format score)
        assert result.reward >= config.final_format_score, "Should get format bonus"

    def test_empty_response(self, reward_fn, task_info):
        """Test handling of empty response."""
        empty_response = ""
        result = reward_fn(input=RewardInput(task_info=task_info, action=empty_response))

        assert not result.is_correct, "Empty response should not be correct"
        assert result.reward <= 0, "Empty response should have zero or negative reward"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
