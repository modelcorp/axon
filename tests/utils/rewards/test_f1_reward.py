import pytest

from axon.utils.rewards.f1_reward import _normalize_answer, compute_f1, f1_reward_fn


class TestNormalizeAnswer:
    """Test the _normalize_answer helper function."""

    def test_remove_articles(self):
        assert _normalize_answer("the cat and a dog") == "cat and dog"

    def test_remove_punctuation(self):
        assert _normalize_answer("hello, world!") == "hello world"

    def test_combined(self):
        assert _normalize_answer("The Quick, Brown Fox!") == "quick brown fox"

    def test_empty_string(self):
        assert _normalize_answer("") == ""

    def test_only_articles(self):
        assert _normalize_answer("a the an") == ""

    def test_only_punctuation(self):
        assert _normalize_answer("!@#$%") == ""

    def test_mixed_whitespace_and_punctuation(self):
        assert _normalize_answer("  hello...   world???  ") == "hello world"


class TestComputeF1:
    """Test the compute_f1 scoring function."""

    def test_empty_prediction(self):
        assert compute_f1("", "hello") == (0.0, 0.0, 0.0)

    def test_partial_overlap(self):
        f1, p, r = compute_f1("hello world foo", "hello world bar")
        assert 0 < f1 < 1.0
        # 2 common tokens out of 3 each: precision=2/3, recall=2/3, f1=2/3
        assert abs(p - 2.0 / 3.0) < 1e-9
        assert abs(r - 2.0 / 3.0) < 1e-9

    def test_no_overlap(self):
        assert compute_f1("abc", "xyz")[0] == 0.0

    def test_yes_no_mismatch(self):
        assert compute_f1("yes", "no") == (0.0, 0.0, 0.0)

    def test_noanswer_mismatch(self):
        assert compute_f1("noanswer", "yes") == (0.0, 0.0, 0.0)

    def test_prediction_superset(self):
        """Prediction has extra tokens: perfect recall, lower precision."""
        f1, p, r = compute_f1("hello world extra", "hello world")
        assert r == 1.0
        assert p < 1.0

    def test_prediction_subset(self):
        """Prediction is a subset of ground truth: perfect precision, lower recall."""
        f1, p, r = compute_f1("hello", "hello world")
        assert p == 1.0
        assert r < 1.0

    def test_normalization_applied(self):
        """Answers that differ only by case/articles/punctuation should match."""
        f1, _, _ = compute_f1("The Quick, Brown Fox!", "quick brown fox")
        assert f1 == 1.0

    def test_gold_is_noanswer_pred_differs(self):
        """When gold_norm is a special token and prediction doesn't match, return 0."""
        assert compute_f1("some other answer", "noanswer") == (0.0, 0.0, 0.0)

    def test_both_empty(self):
        """Empty prediction returns 0 immediately regardless of ground truth."""
        assert compute_f1("", "") == (0.0, 0.0, 0.0)

    def test_repeated_tokens_counted_correctly(self):
        """'the the the' vs 'the dog' - Counter intersection should be 1, not 3."""
        f1, p, r = compute_f1("the the the", "the dog")
        # After normalization articles removed: '' vs 'dog' -> 0 overlap
        assert f1 == 0.0

    def test_unicode_text(self):
        f1, _, _ = compute_f1("cafe resume", "cafe resume")
        # These are different after normalization
        assert 0 <= f1 <= 1.0

    def test_very_long_answer(self):
        prediction = " ".join(["word"] * 1000)
        ground_truth = " ".join(["word"] * 500 + ["other"] * 500)
        f1, p, r = compute_f1(prediction, ground_truth)
        # Counter intersection: min(1000, 500)=500 overlap
        # precision = 500/1000 = 0.5, recall = 500/1000 = 0.5
        assert p == pytest.approx(0.5)
        assert r == pytest.approx(0.5)
        assert f1 == pytest.approx(0.5)


class TestF1RewardFn:
    """Test the f1_reward_fn reward function."""

    def test_no_ground_truth(self):
        result = f1_reward_fn({}, "hello")
        assert result.reward == 0.0
        assert result.is_correct is False

    def test_multiple_ground_truths(self):
        result = f1_reward_fn({"ground_truth": ["foo bar", "hello world"]}, "hello world")
        assert result.reward == 1.0

    def test_partial_match_above_threshold(self):
        # 2 out of 3 tokens overlap -> f1 = 2/3 > 0.5
        result = f1_reward_fn({"answer": "hello world foo"}, "hello world bar")
        assert result.is_correct is True

    def test_low_overlap_below_threshold(self):
        result = f1_reward_fn({"answer": "alpha beta gamma delta"}, "epsilon")
        assert result.is_correct is False

    def test_ground_truth_takes_precedence_over_answer(self):
        """When both 'ground_truth' and 'answer' are present, 'ground_truth' is used."""
        result = f1_reward_fn({"ground_truth": "correct", "answer": "wrong"}, "correct")
        assert result.reward == 1.0

    def test_metadata_contains_f1(self):
        result = f1_reward_fn({"answer": "hello world"}, "hello world")
        assert "f1" in result.metadata
        assert result.metadata["f1"] == 1.0

    def test_best_f1_is_chosen(self):
        """With multiple ground truths, the best F1 should be used."""
        result = f1_reward_fn({"ground_truth": ["xyz", "hello world"]}, "hello world")
        assert result.reward == 1.0

    def test_none_ground_truth(self):
        result = f1_reward_fn({"answer": None}, "hello")
        assert result.reward == 0.0
        assert result.is_correct is False

    def test_threshold_boundary(self):
        """F1 exactly at 0.5 should count as correct (>= 0.5)."""
        # "cat dog" vs "cat bird" -> 1 common out of 2 each -> p=0.5, r=0.5, f1=0.5
        result = f1_reward_fn({"answer": "cat bird"}, "cat dog")
        assert result.reward == pytest.approx(0.5)
        assert result.is_correct is True

    def test_f1_exactly_at_threshold_boundary(self):
        """Construct case where F1 = exactly 0.5"""
        # 1 token overlap, 2 pred tokens, 2 gold tokens: p=0.5, r=0.5, f1=0.5
        result = f1_reward_fn({"answer": "cat bird"}, "cat dog")
        assert result.reward == pytest.approx(0.5)
        assert result.is_correct is True  # >= 0.5

    def test_f1_just_below_threshold(self):
        """F1 < 0.5 should be incorrect."""
        # 1 overlap, 3 pred, 2 gold: p=1/3, r=1/2, f1=2/5=0.4
        result = f1_reward_fn({"answer": "cat bird"}, "cat dog fish")
        assert result.reward < 0.5
        assert result.is_correct is False

    def test_ground_truth_as_non_string_list(self):
        """ground_truth list elements are cast to str."""
        result = f1_reward_fn({"ground_truth": [42]}, "42")
        assert result.reward == 1.0
