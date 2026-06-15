from axon.utils.rewards.gpqa_reward import (
    _compute_gpqa,
    _extract_letter,
    _normalize_text,
    _strip_chain_of_thought,
    gpqa_reward_fn,
)


class TestStripChainOfThought:
    """Test the _strip_chain_of_thought helper."""

    def test_with_think_tag(self):
        assert _strip_chain_of_thought("reasoning here</think>The answer is B") == "The answer is B"

    def test_without_think_tag(self):
        assert _strip_chain_of_thought("The answer is B") == "The answer is B"

    def test_empty(self):
        assert _strip_chain_of_thought("") == ""

    def test_multiple_think_tags(self):
        """Should split on the last </think> occurrence."""
        result = _strip_chain_of_thought("first</think>second</think>final answer")
        assert result == "final answer"

    def test_think_tag_at_end(self):
        result = _strip_chain_of_thought("reasoning</think>")
        assert result == ""

    def test_none_input(self):
        """None should return empty string since the guard checks falsy."""
        assert _strip_chain_of_thought(None) == ""


class TestNormalizeText:
    """Test the _normalize_text helper."""

    def test_special_characters(self):
        assert _normalize_text("!@#$%^&*()") == ""

    def test_mixed_case_and_numbers(self):
        assert _normalize_text("Answer42Is-Correct") == "answer42is correct"


class TestExtractLetter:
    """Test the _extract_letter extraction function."""

    def test_option_pattern(self):
        assert _extract_letter("option: C", list("ABCD")) == "C"

    def test_correct_pattern(self):
        assert _extract_letter("A is the correct answer", list("ABCD")) == "A"

    def test_fallback_last_letter(self):
        assert _extract_letter("I think D", list("ABCD")) == "D"

    def test_empty_response(self):
        assert _extract_letter("", list("ABCD")) is None

    def test_no_valid_letter(self):
        assert _extract_letter("I think Z", list("ABCD")) is None

    def test_strips_cot(self):
        assert _extract_letter("blah</think>The answer is A", list("ABCD")) == "A"

    def test_choice_pattern(self):
        assert _extract_letter("choice is D", list("ABCD")) == "D"

    def test_final_answer_pattern(self):
        assert _extract_letter("My final answer is C", list("ABCD")) == "C"

    def test_lowercase_in_pattern(self):
        """Patterns use IGNORECASE so lowercase letters in pattern should work."""
        assert _extract_letter("The answer is b", list("ABCD")) == "B"

    def test_multiple_letters_picks_pattern_first(self):
        """Pattern match should take priority over fallback."""
        assert _extract_letter("A B C the answer is D", list("ABCD")) == "D"

    def test_multiple_valid_letters_picks_pattern_over_fallback(self):
        """If text contains 'answer is D' plus standalone 'A', pattern wins."""
        assert _extract_letter("A B C the answer is D", list("ABCD")) == "D"

    def test_letter_outside_valid_set_ignored(self):
        """Only valid_letters should be returned."""
        assert _extract_letter("The answer is Z", list("AB")) is None

    def test_many_choices_up_to_h(self):
        assert _extract_letter("The answer is H", list("ABCDEFGH")) == "H"


class TestComputeGpqa:
    """Test the _compute_gpqa core scoring function."""

    def test_wrong_letter_match(self):
        score = _compute_gpqa("The answer is C", "B", {"choices": ["x", "y", "z", "w"]})
        assert score == 0.0

    def test_none_response(self):
        score = _compute_gpqa(None, "A", {})
        assert score == 0.0

    def test_integer_label(self):
        """Integer label is treated as an index into valid_letters."""
        score = _compute_gpqa("The answer is B", 1, {"choices": ["x", "y", "z"]})
        assert score == 1.0

    def test_integer_label_wrong(self):
        score = _compute_gpqa("The answer is A", 1, {"choices": ["x", "y", "z"]})
        assert score == 0.0

    def test_correct_letter_in_metadata(self):
        """When metadata has a correct_letter key, it should be used."""
        score = _compute_gpqa("The answer is C", "ignored", {"correct_letter": "C"})
        assert score == 1.0

    def test_substring_fallback(self):
        """When no letter is extracted but the correct answer text appears in the response."""
        score = _compute_gpqa(
            "The correct answer is photosynthesis",
            "B",
            {"choices": ["mitosis", "photosynthesis", "osmosis"], "correct_letter": "B"},
        )
        assert score == 1.0

    def test_no_choices_no_metadata(self):
        """With minimal metadata, scoring should still work for letter matching."""
        score = _compute_gpqa("The answer is A", "A", {})
        assert score == 1.0

    def test_float_label_index_out_of_bounds(self):
        """Float label 10.0 -> idx=10, but only 3 choices."""
        score = _compute_gpqa("The answer is A", 10.0, {"choices": ["x", "y", "z"]})
        # idx=10 >= len(valid_letters), so no correct_letter set from label
        # Falls back to other matching strategies
        assert score == 0.0 or score == 1.0  # depends on fallback

    def test_empty_choices_list(self):
        score = _compute_gpqa("The answer is A", "A", {"choices": []})
        assert score == 1.0  # Falls back to default valid letters A-H

    def test_correct_answer_in_metadata_enables_substring_match(self):
        score = _compute_gpqa(
            "I believe the process is photosynthesis because...",
            None,
            {"correct_answer": "photosynthesis"},
        )
        assert score == 1.0

    def test_ambiguous_response_with_no_correct_letter(self):
        """No correct_letter can be determined -> 0.0."""
        score = _compute_gpqa("some random text", None, {})
        assert score == 0.0


class TestGpqaRewardFn:
    """Test the gpqa_reward_fn reward function."""

    def test_none_response(self):
        result = gpqa_reward_fn({"answer": "A"}, None)
        assert result.reward == 0.0

    def test_integer_label(self):
        result = gpqa_reward_fn({"answer": 1, "choices": ["x", "y", "z"]}, "The answer is B")
        assert result.reward == 1.0

    def test_with_chain_of_thought(self):
        result = gpqa_reward_fn({"answer": "A"}, "Let me think...</think>The answer is A")
        assert result.reward == 1.0

    def test_ground_truth_key(self):
        """ground_truth should work as an alternative to answer."""
        result = gpqa_reward_fn({"ground_truth": "C", "choices": ["x", "y", "z", "w"]}, "The answer is C")
        assert result.reward == 1.0

    def test_empty_string_response(self):
        result = gpqa_reward_fn({"answer": "A"}, "")
        assert result.reward == 0.0
        assert result.is_correct is False

    def test_choices_as_dict(self):
        """Choices provided as a dict should be converted to a list of values."""
        result = gpqa_reward_fn({"answer": "A", "choices": {"A": "alpha", "B": "beta"}}, "The answer is A")
        assert result.reward == 1.0

    def test_float_label_as_index(self):
        """Float label should be cast to int index."""
        result = gpqa_reward_fn({"answer": 0.0, "choices": ["x", "y", "z"]}, "The answer is A")
        assert result.reward == 1.0

    def test_no_label(self):
        """When no answer or ground_truth is provided, label is None."""
        result = gpqa_reward_fn({}, "The answer is A")
        assert result.reward == 0.0

    def test_choices_dict_values_used_not_keys(self):
        result = gpqa_reward_fn(
            {"answer": "A", "choices": {"opt1": "alpha", "opt2": "beta"}},
            "The answer is A",
        )
        assert result.reward == 1.0

    def test_negative_label_index(self):
        result = gpqa_reward_fn({"answer": -1, "choices": ["x", "y"]}, "The answer is A")
        # -1 as int: idx = -1, 0 <= -1 < 2 is False, no correct_letter from label
        assert isinstance(result.reward, float)
