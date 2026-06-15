import sys
from pathlib import Path

import pytest

mathruler = pytest.importorskip("mathruler")

# Add recipes folder to sys.path since it's outside the axon package
_recipes_path = Path(__file__).parent.parent.parent.parent / "recipes"
if str(_recipes_path) not in sys.path:
    sys.path.insert(0, str(_recipes_path))

from geo3k.env import (  # noqa: E402
    RewardGeo3kFn,
    acc_reward,
    compute_score,
    format_reward,
    geo3k_reward_fn,
)

from axon.utils.rewards.base import RewardConfig, RewardOutput  # noqa: E402

# ---------------------------------------------------------------------------
# format_reward
# ---------------------------------------------------------------------------


class TestFormatReward:
    """Tests for the format_reward helper."""

    def test_valid_format(self):
        """A response with <think>...</think> and \\boxed{} should score 1.0."""
        text = r"<think>some reasoning</think> The answer is \boxed{42}"
        assert format_reward(text) == 1.0

    def test_valid_format_multiline(self):
        """Multi-line think block with boxed should score 1.0."""
        text = "<think>\nline1\nline2\n</think>\n\\boxed{7}"
        assert format_reward(text) == 1.0

    def test_missing_think_tags(self):
        """Missing <think> tags should score 0.0."""
        text = r"The answer is \boxed{42}"
        assert format_reward(text) == 0.0

    def test_missing_boxed(self):
        """Missing \\boxed{} should score 0.0."""
        text = "<think>reasoning</think> The answer is 42"
        assert format_reward(text) == 0.0

    def test_empty_boxed_content(self):
        r"""\\boxed{} with nothing inside the braces should still match the regex."""
        text = r"<think>x</think> \boxed{}"
        assert format_reward(text) == 1.0

    # -- edge cases ---------------------------------------------------------

    def test_only_opening_think_tag(self):
        """Only opening <think> tag (no closing) should score 0.0."""
        text = r"<think>some reasoning \boxed{42}"
        assert format_reward(text) == 0.0

    def test_think_tags_wrong_order(self):
        """Think tags in wrong order with boxed content should score 0.0.

        The regex requires <think>.*</think> in that order via fullmatch.
        </think> before <think> won't match.
        """
        text = r"</think>some reasoning<think> \boxed{42}"
        assert format_reward(text) == 0.0


# ---------------------------------------------------------------------------
# acc_reward
# ---------------------------------------------------------------------------


class TestAccReward:
    """Tests for the acc_reward helper."""

    def test_correct_answer(self):
        """Matching answer should return 1.0."""
        assert acc_reward("42", "42") == 1.0

    def test_incorrect_answer(self):
        """Non-matching answer should return 0.0."""
        assert acc_reward("99", "42") == 0.0

    def test_use_boxed_extracts_content(self):
        """With use_boxed=True the answer is extracted from \\boxed{}."""
        assert acc_reward(r"\boxed{42}", "42", use_boxed=True) == 1.0

    def test_use_boxed_wrong_answer(self):
        """With use_boxed=True and wrong answer should return 0.0."""
        assert acc_reward(r"\boxed{99}", "42", use_boxed=True) == 0.0

    def test_use_boxed_empty_boxed_content(self):
        r"""With use_boxed=True and empty \\boxed{}, extracted content is empty string."""
        result = acc_reward(r"\boxed{}", "42", use_boxed=True)
        # Empty string won't match "42"
        assert result == 0.0

    # -- edge cases: numeric equivalence ------------------------------------

    def test_equivalent_numeric_forms_integer_float(self):
        """Test '3.0' vs '3' -- mathruler's grade_answer handles numeric equivalence."""
        result = acc_reward("3.0", "3")
        # grade_answer should recognise these as equivalent
        assert result == 1.0

    def test_equivalent_numeric_fraction_decimal(self):
        """Test '0.5' vs '1/2' via mathruler grading."""
        result = acc_reward("0.5", "1/2")
        # mathruler should handle simple fraction / decimal equivalence
        assert result == 1.0

    def test_latex_fraction_vs_decimal(self):
        r"""Test LaTeX \frac{1}{2} vs '0.5'."""
        result = acc_reward(r"\frac{1}{2}", "0.5")
        assert result == 1.0


# ---------------------------------------------------------------------------
# compute_score
# ---------------------------------------------------------------------------


class TestComputeScore:
    """Tests for the compute_score blending function."""

    def test_format_score_zero_uses_acc_only(self):
        """format_score=0 means the score comes entirely from accuracy."""
        # Correct answer inside boxed (use_boxed defaults to True)
        assert compute_score(r"\boxed{42}", "42", format_score=0.0) == 1.0
        # Wrong answer
        assert compute_score(r"\boxed{99}", "42", format_score=0.0) == 0.0

    def test_format_score_one_uses_format_only(self):
        """format_score=1.0 means the score comes entirely from format."""
        # Correct format but we don't care about answer correctness
        text = r"<think>ok</think> \boxed{99}"
        assert compute_score(text, "42", format_score=1.0) == 1.0

        # Bad format
        text_bad = r"\boxed{42}"
        assert compute_score(text_bad, "42", format_score=1.0) == 0.0

    def test_format_score_half_blends(self):
        """format_score=0.5 blends accuracy and format equally."""
        # Correct answer + correct format -> 0.5 * 1.0 + 0.5 * 1.0 = 1.0
        text = r"<think>reasoning</think> \boxed{42}"
        assert compute_score(text, "42", format_score=0.5) == pytest.approx(1.0)

        # Correct answer but bad format -> 0.5 * 1.0 + 0.5 * 0.0 = 0.5
        # (use_boxed=True by default, so acc_reward extracts from boxed)
        text2 = r"\boxed{42}"
        assert compute_score(text2, "42", format_score=0.5) == pytest.approx(0.5)

        # Wrong answer but correct format -> 0.5 * 0.0 + 0.5 * 1.0 = 0.5
        text3 = r"<think>reasoning</think> \boxed{99}"
        assert compute_score(text3, "42", format_score=0.5) == pytest.approx(0.5)

    def test_use_boxed_false(self):
        """With use_boxed=False, acc_reward uses predict_str directly (no extraction)."""
        # The raw string "42" compared against ground_truth "42" -> 1.0
        assert compute_score("42", "42", use_boxed=False, format_score=0.0) == 1.0
        # With boxed wrapper but use_boxed=False, the raw string r"\boxed{42}" is compared
        # directly -- mathruler may or may not handle this, but it's a different code path
        score = compute_score(r"\boxed{42}", "42", use_boxed=False, format_score=0.0)
        # The raw string contains \boxed{42}, which grade_answer may still parse
        assert isinstance(score, float)


# ---------------------------------------------------------------------------
# RewardGeo3kFn
# ---------------------------------------------------------------------------


class TestRewardGeo3kFn:
    """Tests for the RewardGeo3kFn callable class."""

    def _make_fn(self, **overrides) -> RewardGeo3kFn:
        config = RewardConfig(**overrides)
        return RewardGeo3kFn(config)

    # -- empty / None response --

    def test_empty_response(self):
        fn = self._make_fn(format_error_reward=-1.0)
        result = fn({"ground_truth": "42"}, "")
        assert result.reward == -1.0
        assert result.is_correct is False

    def test_none_response(self):
        fn = self._make_fn(format_error_reward=-2.0)
        result = fn({"ground_truth": "42"}, None)
        assert result.reward == -2.0
        assert result.is_correct is False

    # -- no boxed content --

    def test_no_boxed_content(self):
        """Response without \\boxed{} should get format_error_reward.

        NOTE: mathruler's extract_boxed_content returns the string 'None'
        rather than actual None for inputs lacking \\boxed{}.  The env code
        checks ``model_answer is None`` which will be False, so the answer
        proceeds to the grading path.  This test documents that behaviour.
        """
        fn = self._make_fn(format_error_reward=-1.0, incorrect_reward=-0.5)
        result = fn({"ground_truth": "42"}, "just plain text without boxed")
        # Because extract_boxed_content returns "None" (a string), the is-None
        # guard is bypassed and the answer "None" is graded against "42" which
        # is incorrect -> incorrect_reward.
        assert result.reward == -0.5
        assert result.is_correct is False

    # -- correct answer --

    def test_correct_answer(self):
        fn = self._make_fn(correct_reward=1.0)
        result = fn({"ground_truth": "42"}, r"\boxed{42}")
        assert result.reward == 1.0
        assert result.is_correct is True

    # -- incorrect answer --

    def test_incorrect_answer(self):
        fn = self._make_fn(incorrect_reward=-0.5)
        result = fn({"ground_truth": "42"}, r"\boxed{99}")
        assert result.reward == -0.5
        assert result.is_correct is False

    # -- missing ground truth --

    def test_missing_ground_truth(self):
        fn = self._make_fn(unk_error_reward=-3.0)
        result = fn({}, r"\boxed{42}")
        assert result.reward == -3.0
        assert result.is_correct is False

    def test_ground_truth_from_answer_key(self):
        """Falls back to 'answer' key when 'ground_truth' is absent."""
        fn = self._make_fn(correct_reward=1.0)
        result = fn({"answer": "42"}, r"\boxed{42}")
        assert result.reward == 1.0
        assert result.is_correct is True

    # -- toolcall bonus --

    def test_toolcall_bonus(self):
        fn = self._make_fn(correct_reward=1.0, toolcall_bonus=0.5)
        result = fn({"ground_truth": "42", "has_toolcall": True}, r"\boxed{42}")
        assert result.reward == 1.5
        assert result.is_correct is True

    def test_no_toolcall_bonus_when_incorrect(self):
        fn = self._make_fn(incorrect_reward=0.0, toolcall_bonus=0.5)
        result = fn({"ground_truth": "42", "has_toolcall": True}, r"\boxed{99}")
        assert result.reward == 0.0
        assert result.is_correct is False

    # -- multiple ground truths --

    def test_multiple_ground_truths_first_matches(self):
        fn = self._make_fn(correct_reward=1.0)
        result = fn({"ground_truth": ["42", "forty-two"]}, r"\boxed{42}")
        assert result.reward == 1.0
        assert result.is_correct is True

    def test_multiple_ground_truths_none_match(self):
        fn = self._make_fn(incorrect_reward=-0.5)
        result = fn({"ground_truth": ["100", "200"]}, r"\boxed{42}")
        assert result.reward == -0.5
        assert result.is_correct is False

    def test_empty_ground_truth_list(self):
        """Empty ground_truth list means the for-loop body never executes -> incorrect."""
        fn = self._make_fn(incorrect_reward=-0.5)
        result = fn({"ground_truth": []}, r"\boxed{42}")
        assert result.reward == -0.5
        assert result.is_correct is False

    # -- edge cases ---------------------------------------------------------

    def test_ground_truth_as_single_float(self):
        """ground_truth as a float exercises the isinstance(str|float|int) branch.

        NOTE: The code wraps float/int in a list but does NOT convert to str
        before passing to grade_answer, which expects strings. This causes an
        AttributeError inside mathruler. This test documents that bug.
        """
        fn = self._make_fn(correct_reward=1.0)
        with pytest.raises(AttributeError):
            fn({"ground_truth": 3.14}, r"\boxed{3.14}")

    def test_ground_truth_as_single_int(self):
        """ground_truth as an int exercises the isinstance(str|float|int) branch.

        Same bug as with float -- grade_answer expects strings.
        """
        fn = self._make_fn(correct_reward=1.0)
        with pytest.raises(AttributeError):
            fn({"ground_truth": 7}, r"\boxed{7}")

    def test_has_toolcall_false_no_bonus(self):
        """Explicit has_toolcall=False should NOT add bonus."""
        fn = self._make_fn(correct_reward=1.0, toolcall_bonus=0.5)
        result = fn({"ground_truth": "42", "has_toolcall": False}, r"\boxed{42}")
        assert result.reward == 1.0
        assert result.is_correct is True


# ---------------------------------------------------------------------------
# geo3k_reward_fn  (convenience wrapper)
# ---------------------------------------------------------------------------


class TestGeo3kRewardFn:
    """Sanity checks for the module-level geo3k_reward_fn helper."""

    def test_correct_answer(self):
        result = geo3k_reward_fn({"ground_truth": "42"}, r"\boxed{42}")
        assert isinstance(result, RewardOutput)
        assert result.is_correct is True
        assert result.reward == RewardConfig().correct_reward

    def test_incorrect_answer(self):
        result = geo3k_reward_fn({"ground_truth": "42"}, r"\boxed{99}")
        assert isinstance(result, RewardOutput)
        assert result.is_correct is False
        assert result.reward == RewardConfig().incorrect_reward
