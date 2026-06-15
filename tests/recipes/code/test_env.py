import importlib.util
import sys
from pathlib import Path

import pytest

# Load the code.env module directly since 'code' conflicts with Python's stdlib
_code_env_path = Path(__file__).parent.parent.parent.parent / "recipes" / "code" / "env.py"
spec = importlib.util.spec_from_file_location("code_env", _code_env_path)
code_env_module = importlib.util.module_from_spec(spec)
sys.modules["code_env"] = code_env_module
spec.loader.exec_module(code_env_module)

CompetitionCodingEnv = code_env_module.CompetitionCodingEnv


class TestCompetitionCodingEnv:
    """Test suite for CompetitionCodingEnv class."""

    @pytest.fixture
    def sample_task(self):
        """Create a sample task for testing."""
        return {
            "question": "Write a function that adds two numbers.",
            "problem_type": "CODE",
            "data_source": "code_contests",
            "ground_truth": {
                "inputs": ["2 3\n"],
                "outputs": ["5\n"],
            },
        }

    def test_init_default(self, sample_task):
        """Test CompetitionCodingEnv initialization with default parameters."""
        env = CompetitionCodingEnv(task=sample_task)
        assert env.task == sample_task
        assert env.max_turns == 1
        assert env.reward_bonus_coeff == 0.0
        assert env.reward_fn is not None

    def test_init_custom_parameters(self, sample_task):
        """Test CompetitionCodingEnv initialization with custom parameters."""
        env = CompetitionCodingEnv(task=sample_task, max_turns=5, reward_bonus_coeff=0.5)
        assert env.task == sample_task
        assert env.max_turns == 5
        assert env.reward_bonus_coeff == 0.5

    def test_reset(self, sample_task):
        """Test the reset method."""
        env = CompetitionCodingEnv(task=sample_task)
        obs, info = env.reset()

        assert "question" in obs
        assert obs["question"] == sample_task["question"]
        assert isinstance(info, dict)
        assert env.done is False
        assert env.current_turn == 0
        assert env.history == []
        assert env.prev_reward is None

    def test_reset_with_new_task(self, sample_task):
        """Test reset with a new task."""
        env = CompetitionCodingEnv(task=sample_task)
        env.reset()

        new_task = {
            "question": "Write a function that multiplies two numbers.",
            "problem_type": "CODE",
            "data_source": "code_contests",
            "ground_truth": {
                "inputs": ["2 3\n"],
                "outputs": ["6\n"],
            },
        }
        obs, info = env.reset(task=new_task)

        assert obs["question"] == new_task["question"]
        assert env.task == new_task

    def test_reset_with_seed(self, sample_task):
        """Test reset with a seed for reproducibility."""
        env = CompetitionCodingEnv(task=sample_task)
        obs1, _ = env.reset(seed=42)
        obs2, _ = env.reset(seed=42)

        assert obs1 == obs2

    def test_step_increments_turn(self, sample_task):
        """Test that step increments the turn counter."""
        env = CompetitionCodingEnv(task=sample_task, max_turns=3)
        env.reset()

        action = "```python\nprint(5)\n```"
        env.step(action)

        assert env.current_turn == 1
        assert action in env.history

    def test_step_returns_correct_format(self, sample_task):
        """Test that step returns the expected format."""
        env = CompetitionCodingEnv(task=sample_task, max_turns=3)
        env.reset()

        action = "```python\nprint(5)\n```"
        next_obs, reward, done, info = env.step(action)

        assert isinstance(next_obs, dict)
        assert isinstance(reward, int | float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)

    def test_step_terminates_at_max_turns(self, sample_task):
        """Test that environment terminates at max_turns."""
        env = CompetitionCodingEnv(task=sample_task, max_turns=2)
        env.reset()

        action = "```python\nprint(5)\n```"

        # First step
        _, _, done1, _ = env.step(action)
        assert done1 is False
        assert env.current_turn == 1

        # Second step (should terminate)
        _, _, done2, _ = env.step(action)
        assert done2 is True
        assert env.current_turn == 2

    def test_step_single_turn(self, sample_task):
        """Test environment with single turn (max_turns=1)."""
        env = CompetitionCodingEnv(task=sample_task, max_turns=1)
        env.reset()

        action = "```python\nprint(5)\n```"
        _, _, done, _ = env.step(action)

        assert done is True
        assert env.current_turn == 1

    def test_reward_shaping_with_bonus(self, sample_task):
        """Test reward shaping with bonus coefficient."""
        env = CompetitionCodingEnv(task=sample_task, max_turns=3, reward_bonus_coeff=0.5)
        env.reset()

        # First action - no previous reward, so no bonus
        action = "```python\nprint(5)\n```"
        _, reward1, _, _ = env.step(action)

        assert env.prev_reward is not None

    def test_from_dict(self, sample_task):
        """Test creating environment from dictionary."""
        env_args = {
            **sample_task,
            "max_turns": 3,
            "reward_bonus_coeff": 0.25,
        }

        env = CompetitionCodingEnv.from_dict(env_args)

        assert isinstance(env, CompetitionCodingEnv)
        assert env.task["question"] == sample_task["question"]
        assert env.max_turns == 3
        assert env.reward_bonus_coeff == 0.25

    def test_from_dict_defaults(self, sample_task):
        """Test creating environment from dictionary with defaults."""
        env_args = {"task": sample_task}

        env = CompetitionCodingEnv.from_dict(env_args)

        assert isinstance(env, CompetitionCodingEnv)
        assert env.max_turns == 1
        assert env.reward_bonus_coeff == 0.0

    def test_history_tracking(self, sample_task):
        """Test that history is properly tracked."""
        env = CompetitionCodingEnv(task=sample_task, max_turns=3)
        env.reset()

        action1 = "```python\nprint(1)\n```"
        action2 = "```python\nprint(2)\n```"

        env.step(action1)
        env.step(action2)

        assert len(env.history) == 2
        assert env.history[0] == action1
        assert env.history[1] == action2

    def test_reset_clears_history(self, sample_task):
        """Test that reset clears the history."""
        env = CompetitionCodingEnv(task=sample_task, max_turns=3)
        env.reset()

        action = "```python\nprint(5)\n```"
        env.step(action)

        assert len(env.history) == 1

        env.reset()

        assert len(env.history) == 0

    def test_get_reward_and_next_obs(self, sample_task):
        """Test the get_reward_and_next_obs method directly."""
        env = CompetitionCodingEnv(task=sample_task)
        env.reset()

        action = "```python\nprint(5)\n```"
        reward, metadata = env.get_reward_and_next_obs(sample_task, action)

        assert isinstance(reward, int | float)
        assert isinstance(metadata, dict)


class TestCompetitionCodingEnvIntegration:
    """Integration tests for CompetitionCodingEnv."""

    def test_correct_solution(self):
        """Test with a correct solution."""
        task = {
            "question": "Read two integers and print their sum.",
            "problem_type": "CODE",
            "data_source": "code_contests",
            "ground_truth": {
                "inputs": ["2 3\n", "10 20\n"],
                "outputs": ["5\n", "30\n"],
            },
        }
        env = CompetitionCodingEnv(task=task, max_turns=1)
        env.reset()

        correct_solution = """
```python
a, b = map(int, input().split())
print(a + b)
```
"""
        _, reward, done, _ = env.step(correct_solution)

        assert done is True
        assert reward > 0  # Should get positive reward for correct solution

    def test_incorrect_solution(self):
        """Test with an incorrect solution."""
        task = {
            "question": "Read two integers and print their sum.",
            "problem_type": "CODE",
            "data_source": "code_contests",
            "ground_truth": {
                "inputs": ["2 3\n"],
                "outputs": ["5\n"],
            },
        }
        env = CompetitionCodingEnv(task=task, max_turns=1)
        env.reset()

        incorrect_solution = """
```python
a, b = map(int, input().split())
print(a * b)  # Wrong - multiplies instead of adds
```
"""
        _, reward, done, _ = env.step(incorrect_solution)

        assert done is True
        assert reward == 0  # Should get 0 reward for incorrect solution

    def test_malformed_solution(self):
        """Test with a malformed solution (no code block)."""
        task = {
            "question": "Read two integers and print their sum.",
            "problem_type": "CODE",
            "data_source": "code_contests",
            "ground_truth": {
                "inputs": ["2 3\n"],
                "outputs": ["5\n"],
            },
        }
        env = CompetitionCodingEnv(task=task, max_turns=1)
        env.reset()

        malformed_solution = "Just print a + b"
        _, reward, done, _ = env.step(malformed_solution)

        assert done is True
        assert reward == 0  # Should get 0 reward for malformed solution

    def test_multi_turn_improvement(self):
        """Test multi-turn scenario where solution improves."""
        task = {
            "question": "Read two integers and print their sum.",
            "problem_type": "CODE",
            "data_source": "code_contests",
            "ground_truth": {
                "inputs": ["2 3\n", "10 20\n"],
                "outputs": ["5\n", "30\n"],
            },
        }
        env = CompetitionCodingEnv(task=task, max_turns=3, reward_bonus_coeff=0.5)
        env.reset()

        # First attempt - incorrect
        incorrect_solution = """
```python
a, b = map(int, input().split())
print(a * b)
```
"""
        _, reward1, done1, _ = env.step(incorrect_solution)
        assert done1 is False

        # Second attempt - correct
        correct_solution = """
```python
a, b = map(int, input().split())
print(a + b)
```
"""
        _, reward2, done2, _ = env.step(correct_solution)
        assert done2 is False

        # Reward should potentially include bonus for improvement
        assert reward2 >= reward1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
