"""Tests for axon.core.agent module."""

import pytest

from axon.core.agent import AGENT_CLASS_MAPPING, Action, BaseAgent, DefaultAgent


# ---------------------------------------------------------------------------
# BaseAgent abstract contract
# ---------------------------------------------------------------------------
class TestBaseAgent:
    def test_process_observation_raises(self):
        class StubAgent(BaseAgent):
            def process_action(self, action):
                return Action()

        agent = StubAgent()
        with pytest.raises(NotImplementedError, match="process_observation"):
            agent.process_observation("obs", 0.0, False, {})

    def test_process_action_raises(self):
        class StubAgent(BaseAgent):
            def process_observation(self, observation, reward, done, info, **kwargs):
                return observation

        agent = StubAgent()
        with pytest.raises(NotImplementedError, match="process_action"):
            agent.process_action("act")


# ---------------------------------------------------------------------------
# Agent registry
# ---------------------------------------------------------------------------
class TestAgentRegistry:
    def test_default_is_registered(self):
        assert "default" in AGENT_CLASS_MAPPING

    def test_lookup_returns_default_agent_class(self):
        assert AGENT_CLASS_MAPPING["default"] is DefaultAgent

    def test_lookup_nonexistent_raises(self):
        with pytest.raises(ValueError, match="Unknown agent"):
            AGENT_CLASS_MAPPING["does_not_exist_xyz"]


# ---------------------------------------------------------------------------
# DefaultAgent.process_observation – kept + adversarial
# ---------------------------------------------------------------------------
class TestDefaultAgentProcessObservation:
    def setup_method(self):
        self.agent = DefaultAgent()

    def test_dict_with_question_key(self):
        obs = {"question": "What is 2+2?", "extra": "data"}
        result = self.agent.process_observation(obs, 0.0, False, {})
        assert result == "What is 2+2?"

    def test_dict_without_question_key(self):
        obs = {"context": "some context", "id": 42}
        result = self.agent.process_observation(obs, 0.0, False, {})
        assert result == str(obs)

    def test_string_observation(self):
        result = self.agent.process_observation("plain text", 0.0, False, {})
        assert result == "plain text"

    # -- adversarial inputs --

    def test_none_observation_returns_none_string(self):
        result = self.agent.process_observation(None, 0.0, False, {})
        assert result == "None"

    def test_empty_dict_returns_str_of_empty_dict(self):
        result = self.agent.process_observation({}, 0.0, False, {})
        # {} has no "question" key, falls through to str({})
        assert result == str({})

    def test_dict_with_question_none_returns_none(self):
        """dict.get('question') returns None when value is None."""
        result = self.agent.process_observation({"question": None}, 0.0, False, {})
        assert result is None

    def test_dict_with_nested_question_value(self):
        nested = {"nested": "val"}
        result = self.agent.process_observation({"question": nested}, 0.0, False, {})
        assert result == nested
        assert result is nested  # same dict object from .get()

    def test_list_observation_returns_str_of_list(self):
        obs = [1, 2, 3]
        result = self.agent.process_observation(obs, 0.0, False, {})
        assert result == str([1, 2, 3])

    def test_int_observation_returns_str(self):
        result = self.agent.process_observation(42, 0.0, False, {})
        assert result == "42"

    # -- hardened edge cases --

    def test_dict_with_question_key_false_value(self):
        """dict.get('question') returns False — falsy but valid value."""
        result = self.agent.process_observation({"question": False}, 0.0, False, {})
        assert result is False

    def test_dict_with_question_key_zero(self):
        """dict.get('question') returns 0 — falsy but valid value."""
        result = self.agent.process_observation({"question": 0}, 0.0, False, {})
        assert result == 0

    def test_dict_with_question_key_empty_string(self):
        """dict.get('question') returns '' — falsy but valid value."""
        result = self.agent.process_observation({"question": ""}, 0.0, False, {})
        assert result == ""

    def test_dict_with_question_key_empty_list(self):
        """dict.get('question') returns [] — falsy but valid value."""
        result = self.agent.process_observation({"question": []}, 0.0, False, {})
        assert result == []

    def test_float_observation_returns_str(self):
        result = self.agent.process_observation(3.14, 0.0, False, {})
        assert result == "3.14"

    def test_bool_observation_returns_str(self):
        result = self.agent.process_observation(True, 0.0, False, {})
        assert result == "True"


# ---------------------------------------------------------------------------
# DefaultAgent.process_action – kept + adversarial
# ---------------------------------------------------------------------------
class TestDefaultAgentProcessAction:
    def setup_method(self):
        self.agent = DefaultAgent()

    def test_returns_action_with_value(self):
        action = self.agent.process_action("my_action")
        assert isinstance(action, Action)
        assert action.action == "my_action"
        assert action.thought == ""

    def test_process_action_with_none(self):
        action = self.agent.process_action(None)
        assert isinstance(action, Action)
        assert action.action is None
        assert action.thought == ""


# ---------------------------------------------------------------------------
# Action dataclass – adversarial usage
# ---------------------------------------------------------------------------
class TestActionEdgeCases:
    def test_action_as_dict_value(self):
        a = Action(thought="t", action="a")
        d = {"key": a}
        assert d["key"].action == "a"
        assert d["key"].thought == "t"

    def test_action_in_list(self):
        actions = [Action(action=i) for i in range(3)]
        assert [a.action for a in actions] == [0, 1, 2]

    def test_action_equality(self):
        """dataclass default __eq__ compares field values."""
        a1 = Action(thought="t", action="a")
        a2 = Action(thought="t", action="a")
        assert a1 == a2

    def test_action_with_complex_payload(self):
        payload = {"tool": "search", "args": [1, 2, 3]}
        a = Action(action=payload)
        assert a.action["tool"] == "search"
        assert a.action is payload

    def test_action_with_large_payload(self):
        """Action with very large payload should work."""
        payload = {"data": list(range(10000))}
        a = Action(action=payload)
        assert len(a.action["data"]) == 10000


class TestBaseAgentEdgeCases:
    def test_reset_returns_none(self):
        """BaseAgent.reset() should return None."""

        class StubAgent(BaseAgent):
            def process_observation(self, observation, reward, done, info, **kwargs):
                return observation

            def process_action(self, action):
                return Action()

        agent = StubAgent()
        assert agent.reset() is None

    def test_system_prompt_default_empty_string(self):
        """BaseAgent.system_prompt should default to empty string."""

        class StubAgent(BaseAgent):
            def process_observation(self, observation, reward, done, info, **kwargs):
                return observation

            def process_action(self, action):
                return Action()

        agent = StubAgent()
        assert agent.system_prompt == ""

    def test_kwargs_forwarded_to_process_observation(self):
        """Extra kwargs should be forwarded to process_observation."""
        received_kwargs = {}

        class KwargsAgent(BaseAgent):
            def process_observation(self, observation, reward, done, info, **kwargs):
                received_kwargs.update(kwargs)
                return observation

            def process_action(self, action):
                return Action()

        agent = KwargsAgent()
        agent.process_observation("obs", 0.0, False, {}, extra="value", debug=True)
        assert received_kwargs == {"extra": "value", "debug": True}
