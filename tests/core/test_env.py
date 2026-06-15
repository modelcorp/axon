"""Tests for axon.core.env module."""

import warnings

import pytest

from axon.core.env import (
    ENV_CLASS_MAPPING,
    MultiTurnEnvironment,
    SingleTurnEnvironment,
)
from axon.utils.rewards.base import RewardOutput, zero_reward_fn


# ---------------------------------------------------------------------------
# Concrete subclass of MultiTurnEnvironment for testing
# ---------------------------------------------------------------------------
class SimpleMultiTurnEnv(MultiTurnEnvironment):
    """Minimal concrete subclass that returns a fixed reward and observation."""

    def get_reward_and_next_obs(self, task, action):
        return 1.0, {"next": "obs"}


# ---------------------------------------------------------------------------
# MultiTurnEnvironment – kept + adversarial
# ---------------------------------------------------------------------------
class TestMultiTurnEnvironment:
    def test_step_increments_turn_and_appends_history(self):
        env = SimpleMultiTurnEnv(task={"q": "hi"}, max_turns=3)
        obs, reward, done, info = env.step("action_0")
        assert reward == 1.0
        assert done is False
        assert env.current_turn == 1
        assert env.history == ["action_0"]
        assert obs == {"next": "obs"}

    def test_step_at_max_turns_sets_done(self):
        env = SimpleMultiTurnEnv(task={"q": "hi"}, max_turns=2)
        env.step("a1")
        obs, reward, done, info = env.step("a2")
        assert done is True
        assert env.current_turn == 2
        assert obs == {}  # last turn returns empty obs

    def test_full_episode(self):
        env = SimpleMultiTurnEnv(task={"q": "hi"}, max_turns=3)
        env.reset()
        results = []
        for i in range(3):
            results.append(env.step(f"action_{i}"))
        for obs, reward, done, info in results[:2]:
            assert done is False
            assert reward == 1.0
        _, _, done, _ = results[2]
        assert done is True
        assert env.history == ["action_0", "action_1", "action_2"]

    def test_reset_clears_state(self):
        env = SimpleMultiTurnEnv(task={"q": "hi"}, max_turns=3)
        env.current_turn = 2
        env.done = True
        env.history = ["a", "b"]
        obs, info = env.reset()
        assert obs == {"q": "hi"}
        assert info == {}
        assert env.current_turn == 0
        assert env.done is False
        assert env.history == []

    # -- adversarial --

    def test_step_after_done_keeps_going(self):
        """Stepping after done does not raise; it keeps incrementing."""
        env = SimpleMultiTurnEnv(task={"q": "hi"}, max_turns=1)
        _, _, done, _ = env.step("a1")
        assert done is True
        # Stepping again: current_turn goes past max_turns, done stays True
        _, _, done2, _ = env.step("a2")
        assert done2 is True
        assert env.current_turn == 2
        assert env.history == ["a1", "a2"]

    def test_max_turns_zero_first_step_is_done(self):
        env = SimpleMultiTurnEnv(task={"q": "hi"}, max_turns=0)
        _, _, done, _ = env.step("a")
        assert done is True

    def test_step_without_reset_works(self):
        """task is set in __init__, so step works without calling reset first."""
        env = SimpleMultiTurnEnv(task={"q": "hi"}, max_turns=3)
        obs, reward, done, info = env.step("first")
        assert reward == 1.0
        assert done is False

    def test_step_with_task_none_raises(self):
        env = SimpleMultiTurnEnv(task=None, max_turns=3)
        with pytest.raises(AssertionError, match="Task is not set"):
            env.step("action")

    def test_reset_mid_episode_clears_state(self):
        env = SimpleMultiTurnEnv(task={"q": "hi"}, max_turns=5)
        env.step("a1")
        env.step("a2")
        assert env.current_turn == 2
        env.reset()
        assert env.current_turn == 0
        assert env.history == []
        assert env.done is False


# ---------------------------------------------------------------------------
# SingleTurnEnvironment – kept + adversarial
# ---------------------------------------------------------------------------
class TestSingleTurnEnvironment:
    def test_step_returns_reward_from_reward_fn(self):
        def my_reward(task_info, action):
            return RewardOutput(reward=5.0, metadata={"custom": True})

        env = SingleTurnEnvironment(task={"q": "hi"}, reward_fn=my_reward)
        obs, reward, done, info = env.step("answer")
        assert reward == 5.0
        assert done is True

    def test_no_reward_fn_warns_and_uses_zero(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            env = SingleTurnEnvironment(task={"q": "hi"}, reward_fn=None)
            assert len(w) == 1
            assert "zero reward" in str(w[0].message).lower()
        assert env.reward_fn is zero_reward_fn

    def test_from_dict_does_not_mutate_input(self):
        original = {"task": {"q": "hi"}, "reward_fn": None}
        original_copy = dict(original)
        SingleTurnEnvironment.from_dict(original)
        assert original == original_copy, "from_dict must not mutate the input dict"

    def test_from_dict_without_task_key_uses_whole_dict(self):
        env_args = {"question": "What?", "answer": "42"}
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            env = SingleTurnEnvironment.from_dict(env_args)
        # The whole dict (minus reward_fn) becomes the task
        assert env.task == {"question": "What?", "answer": "42"}

    def test_reward_fn_invalid_string_raises_key_error(self):
        with pytest.raises(KeyError, match="Unknown reward_fn"):
            SingleTurnEnvironment(task={"q": "hi"}, reward_fn="nonexistent_reward_xyz")

    def test_reward_fn_returns_wrong_type_raises(self):
        """If reward_fn returns a float instead of RewardOutput, accessing .reward fails."""

        def bad_reward(task_info, action):
            return 3.14  # float, not RewardOutput

        env = SingleTurnEnvironment(task={"q": "hi"}, reward_fn=bad_reward)
        with pytest.raises(AttributeError):
            env.step("answer")

    # -- hardened edge cases --

    def test_from_dict_with_none_task_value(self):
        """from_dict with task=None creates env that fails on step."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            env = SingleTurnEnvironment.from_dict({"task": None})
        with pytest.raises(AssertionError, match="Task is not set"):
            env.step("action")

    def test_reward_fn_returning_none_raises(self):
        """reward_fn returning None should fail when accessing .reward."""

        def nil_fn(task_info, action):
            return None

        env = SingleTurnEnvironment(task={"q": "hi"}, reward_fn=nil_fn)
        with pytest.raises(AttributeError):
            env.step("answer")

    def test_from_dict_with_nested_task(self):
        """from_dict with nested dict task should preserve structure."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            env = SingleTurnEnvironment.from_dict({"task": {"q": "hi", "nested": {"a": 1}}})
        assert env.task == {"q": "hi", "nested": {"a": 1}}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
class TestEnvRegistry:
    def test_single_turn_registered(self):
        assert "single_turn" in ENV_CLASS_MAPPING
        assert ENV_CLASS_MAPPING["single_turn"] is SingleTurnEnvironment

    def test_multi_turn_registered(self):
        assert "multi_turn" in ENV_CLASS_MAPPING
        assert ENV_CLASS_MAPPING["multi_turn"] is MultiTurnEnvironment


class TestMultiTurnEnvironmentEdgeCases:
    def test_negative_max_turns(self):
        """Negative max_turns: 0 >= -1, so first step is always done."""
        env = SimpleMultiTurnEnv(task={"q": "hi"}, max_turns=-1)
        obs, reward, done, info = env.step("a")
        assert done is True

    def test_very_large_max_turns(self):
        """Large max_turns should not cause issues."""
        env = SimpleMultiTurnEnv(task={"q": "hi"}, max_turns=10**9)
        obs, reward, done, info = env.step("a")
        assert done is False
        assert env.current_turn == 1

    def test_idx_property_default_none(self):
        """BaseEnv.idx should default to None."""
        env = SimpleMultiTurnEnv(task={"q": "hi"}, max_turns=3)
        assert env.idx is None

    def test_idx_property_set_and_get(self):
        """BaseEnv.idx setter and getter should work."""
        env = SimpleMultiTurnEnv(task={"q": "hi"}, max_turns=3)
        env.idx = 42
        assert env.idx == 42
        env.idx = "batch_0"
        assert env.idx == "batch_0"

    def test_close_does_not_raise(self):
        """close() should be a no-op that doesn't raise."""
        env = SimpleMultiTurnEnv(task={"q": "hi"}, max_turns=3)
        env.close()

    def test_is_multithread_safe_default(self):
        """Default is_multithread_safe should return True."""
        env = SimpleMultiTurnEnv(task={"q": "hi"}, max_turns=3)
        assert env.is_multithread_safe() is True

    def test_from_dict_raises_on_abstract(self):
        """MultiTurnEnvironment.from_dict should raise NotImplementedError."""
        import pytest

        with pytest.raises(NotImplementedError):
            MultiTurnEnvironment.from_dict({})
