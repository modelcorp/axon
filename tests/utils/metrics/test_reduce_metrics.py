"""Tests for axon.utils.metrics.reduce_metrics module."""

import numpy as np
import pytest

from axon.utils.metrics.reduce_metrics import reduce_metrics


class TestReduceMetrics:
    # -- kept tests --

    def test_docstring_example(self):
        metrics = {
            "loss": [1.0, 2.0, 3.0],
            "accuracy": [0.8, 0.9, 0.7],
            "max_reward": [5.0, 8.0, 6.0],
            "min_error": [0.1, 0.05, 0.2],
        }
        result = reduce_metrics(metrics)
        assert np.isclose(result["loss"], 2.0)
        assert np.isclose(result["accuracy"], 0.8)
        assert np.isclose(result["max_reward"], 8.0)
        assert np.isclose(result["min_error"], 0.05)

    def test_negative_values(self):
        result = reduce_metrics(
            {
                "score": [-1.0, -2.0, -3.0],
                "max_score": [-1.0, -2.0, -3.0],
                "min_score": [-1.0, -2.0, -3.0],
            }
        )
        assert np.isclose(result["score"], -2.0)
        assert np.isclose(result["max_score"], -1.0)
        assert np.isclose(result["min_score"], -3.0)

    # -- adversarial tests --

    def test_empty_list_value_crashes(self):
        """np.concatenate on an empty sequence raises ValueError."""
        with pytest.raises(ValueError):
            reduce_metrics({"loss": []})

    def test_nan_propagation_in_mean(self):
        result = reduce_metrics({"loss": [1.0, float("nan"), 3.0]})
        assert np.isnan(result["loss"])

    def test_inf_values(self):
        result = reduce_metrics(
            {
                "loss": [1.0, float("inf")],
                "max_val": [float("-inf"), 5.0],
                "min_val": [float("inf"), 5.0],
            }
        )
        assert result["loss"] == float("inf")  # mean of [1, inf] = inf
        assert result["max_val"] == 5.0
        assert result["min_val"] == 5.0

    def test_key_with_both_max_and_min_uses_max(self):
        """'max' is checked first, so 'max_min_score' uses np.max."""
        result = reduce_metrics({"max_min_score": [1.0, 5.0, 3.0]})
        assert np.isclose(result["max_min_score"], 5.0)

    def test_nested_arrays_different_shapes_flattened(self):
        """Values can be arrays of different lengths; they get flattened via concatenate."""
        result = reduce_metrics(
            {
                "loss": [np.array([1.0, 2.0]), np.array([3.0])],
            }
        )
        # mean of [1, 2, 3] = 2.0
        assert np.isclose(result["loss"], 2.0)

    def test_substring_maximum_triggers_max(self):
        """'maximum' contains 'max', so np.max is used."""
        result = reduce_metrics({"maximum_reward": [3.0, 9.0, 1.0]})
        assert np.isclose(result["maximum_reward"], 9.0)

    def test_substring_minimize_triggers_min(self):
        """'minimize' contains 'min', so np.min is used."""
        result = reduce_metrics({"minimize_loss": [3.0, 1.0, 5.0]})
        assert np.isclose(result["minimize_loss"], 1.0)

    # -- hardened edge cases: false-positive substring matches --

    def test_admin_key_should_use_mean_not_min(self):
        """'admin_score' contains 'min' as substring of 'admin', but semantically
        this key has nothing to do with min — it should use mean."""
        result = reduce_metrics({"admin_score": [1.0, 5.0, 3.0]})
        assert np.isclose(result["admin_score"], 3.0), (
            f"'admin_score' should use mean (3.0), got {result['admin_score']}. "
            f"False positive: 'min' substring in 'admin' triggered np.min."
        )

    def test_terminal_key_should_use_mean_not_min(self):
        """'terminal_status' contains 'min' in 'terminal', should use mean."""
        result = reduce_metrics({"terminal_status": [2.0, 4.0, 6.0]})
        assert np.isclose(result["terminal_status"], 4.0), (
            f"'terminal_status' should use mean (4.0), got {result['terminal_status']}"
        )

    def test_diminish_key_should_use_mean_not_min(self):
        """'diminish_rate' contains 'min' in 'diminish', should use mean."""
        result = reduce_metrics({"diminish_rate": [1.0, 5.0, 3.0]})
        assert np.isclose(result["diminish_rate"], 3.0), (
            f"'diminish_rate' should use mean (3.0), got {result['diminish_rate']}"
        )

    def test_single_element_lists(self):
        """Single-element lists should return the element itself for all agg types."""
        result = reduce_metrics(
            {
                "loss": [5.0],
                "max_val": [3.0],
                "min_val": [7.0],
            }
        )
        assert np.isclose(result["loss"], 5.0)
        assert np.isclose(result["max_val"], 3.0)
        assert np.isclose(result["min_val"], 7.0)

    def test_mixed_int_and_float_values(self):
        """Mixed int and float values should be reduced correctly."""
        result = reduce_metrics({"score": [1, 2.0, 3]})
        assert np.isclose(result["score"], 2.0)
