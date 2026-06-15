"""Tests for axon.programs.react_program module."""

from collections import OrderedDict

import pytest

from axon.programs.react_program import ReactProgram, _broadcast

# ---------------------------------------------------------------------------
# _broadcast helper
# ---------------------------------------------------------------------------


class TestBroadcast:
    def test_none_returns_empty_dicts(self):
        result = _broadcast(None, 3)
        assert result == [{}, {}, {}]

    def test_dict_returns_copies(self):
        d = {"a": 1}
        result = _broadcast(d, 2)
        assert result == [{"a": 1}, {"a": 1}]

    def test_list_correct_length(self):
        lst = [{"x": 1}, {"x": 2}]
        result = _broadcast(lst, 2)
        assert result == [{"x": 1}, {"x": 2}]

    def test_list_wrong_length_raises(self):
        with pytest.raises(AssertionError, match="Expected list of length 3"):
            _broadcast([{"a": 1}], 3)

    def test_n_zero_returns_empty_list(self):
        """Broadcasting with n=0 should return an empty list."""
        result = _broadcast(None, 0)
        assert result == []

    def test_ordered_dict_mapping_subclass(self):
        """OrderedDict is a Mapping subclass and should be handled like a dict."""
        od = OrderedDict([("z", 26), ("a", 1)])
        result = _broadcast(od, 3)
        assert len(result) == 3
        for item in result:
            assert isinstance(item, dict)
            assert item == {"z": 26, "a": 1}

    def test_dict_broadcast_returns_independent_copies(self):
        d = {"key": "value"}
        result = _broadcast(d, 3)
        assert result[0] is not result[1]
        assert result[1] is not result[2]

        result[0]["new_key"] = "oops"
        assert "new_key" not in result[1]


# ---------------------------------------------------------------------------
# ReactProgram construction
# ---------------------------------------------------------------------------


class TestReactProgramConstruction:
    def test_single_env_agent(self):
        prog = ReactProgram(agent_name="default", env_name="single_turn")
        assert prog.agent_name == "default"
        assert prog.env_name == "single_turn"
        assert prog.accumulate_thinking is True
        assert prog.accumulate_history is True


# ---------------------------------------------------------------------------
# format_messages
# ---------------------------------------------------------------------------


class TestFormatMessages:
    def _make_program(self, accumulate_thinking=True, accumulate_history=True):
        return ReactProgram(
            agent_name="default",
            env_name="single_turn",
            accumulate_thinking=accumulate_thinking,
            accumulate_history=accumulate_history,
        )

    def test_accumulate_thinking_false_strips_before_think_close(self):
        prog = self._make_program(accumulate_thinking=False)
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "<think>internal reasoning</think>The answer is 42."},
        ]
        result = prog.format_messages(messages)
        assert result[2]["content"] == "The answer is 42."
        assert result[0]["content"] == "You are helpful."
        assert result[1]["content"] == "Hello"

    def test_accumulate_history_false_keeps_first_and_last(self):
        prog = self._make_program(accumulate_history=False)
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "msg1"},
            {"role": "assistant", "content": "resp1"},
            {"role": "user", "content": "msg2"},
        ]
        result = prog.format_messages(messages)
        assert len(result) == 2
        assert result[0]["content"] == "sys"
        assert result[1]["content"] == "msg2"

    def test_both_true_returns_deepcopy(self):
        prog = self._make_program(accumulate_thinking=True, accumulate_history=True)
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "<think>think</think>answer"},
        ]
        result = prog.format_messages(messages)
        assert len(result) == 3
        assert result[2]["content"] == "<think>think</think>answer"
        assert result is not messages
        assert result[0] is not messages[0]

    def test_multiple_think_tags_only_first_split(self):
        """partition() splits on the FIRST occurrence of </think>,
        so content after the first </think> (including later tags) is kept."""
        prog = self._make_program(accumulate_thinking=False)
        messages = [
            {"role": "assistant", "content": "<think>first</think>middle<think>second</think>end"},
        ]
        result = prog.format_messages(messages)
        assert result[0]["content"] == "middle<think>second</think>end"

    def test_think_close_at_very_start_of_content(self):
        """If </think> appears at the very start, everything before it (empty string)
        is stripped, leaving the rest."""
        prog = self._make_program(accumulate_thinking=False)
        messages = [
            {"role": "assistant", "content": "</think>All visible content"},
        ]
        result = prog.format_messages(messages)
        assert result[0]["content"] == "All visible content"

    def test_empty_messages_list(self):
        """format_messages with an empty list should return an empty list."""
        prog = self._make_program(accumulate_thinking=False, accumulate_history=False)
        result = prog.format_messages([])
        assert result == []

    def test_both_flags_false(self):
        """With both accumulate_thinking=False and accumulate_history=False,
        thinking should be stripped AND only first+last messages kept."""
        prog = self._make_program(accumulate_thinking=False, accumulate_history=False)
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "<think>hidden</think>answer1"},
            {"role": "user", "content": "q2"},
        ]
        result = prog.format_messages(messages)
        # accumulate_history=False keeps first and last
        assert len(result) == 2
        assert result[0]["content"] == "sys"
        assert result[1]["content"] == "q2"


# ---------------------------------------------------------------------------
# add_observation_to_messages / add_action_to_messages
# ---------------------------------------------------------------------------


class TestMessageHelpers:
    def _make_program(self):
        prog = ReactProgram(agent_name="default", env_name="single_turn")
        prog.messages = [{"role": "system", "content": "sys"}]
        return prog

    def test_add_observation_string(self):
        prog = self._make_program()
        prog.add_observation_to_messages("observation text")
        assert prog.messages[-1] == {"role": "user", "content": "observation text"}

    def test_add_observation_list_of_dicts(self):
        prog = self._make_program()
        obs = [{"role": "user", "content": "a"}, {"role": "user", "content": "b"}]
        prog.add_observation_to_messages(obs)
        assert prog.messages[-2] == {"role": "user", "content": "a"}
        assert prog.messages[-1] == {"role": "user", "content": "b"}

    def test_add_observation_with_non_list_non_string(self):
        """An integer observation gets wrapped in a user message dict.
        The content field will store the integer as-is (not stringified by this method)."""
        prog = self._make_program()
        prog.add_observation_to_messages(42)
        assert prog.messages[-1] == {"role": "user", "content": 42}

    def test_add_action_to_messages(self):
        prog = self._make_program()
        prog.add_action_to_messages("action text")
        assert prog.messages[-1] == {"role": "assistant", "content": "action text"}
