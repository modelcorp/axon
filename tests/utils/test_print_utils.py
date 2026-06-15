"""Tests for axon.utils.print_utils module."""

import warnings

from axon.utils.print_utils import (
    append_to_dict,
    colorful_print,
    colorful_warning,
    log_metrics,
    merge_dicts,
)


# ---------------------------------------------------------------------------
# colorful_print
# ---------------------------------------------------------------------------
class TestColorfulPrint:
    def test_output_contains_string(self, capsys):
        colorful_print("hello world")
        captured = capsys.readouterr()
        assert "hello world" in captured.out

    def test_custom_end_kwarg(self, capsys):
        colorful_print("no newline", end="")
        captured = capsys.readouterr()
        assert not captured.out.endswith("\n")
        assert "no newline" in captured.out


# ---------------------------------------------------------------------------
# colorful_warning
# ---------------------------------------------------------------------------
class TestColorfulWarning:
    def test_warning_is_raised(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            colorful_warning("danger ahead")
            assert len(w) == 1
            assert "danger ahead" in str(w[0].message)

    def test_warning_with_color_kwarg(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            colorful_warning("red alert", fg="red")
            assert len(w) == 1
            assert "red alert" in str(w[0].message)


# ---------------------------------------------------------------------------
# log_metrics
# ---------------------------------------------------------------------------
class TestLogMetrics:
    def test_empty_dict_produces_no_output(self, capsys):
        log_metrics({})
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_title_appears_in_output(self, capsys):
        log_metrics({"acc": 0.95}, title="Train")
        captured = capsys.readouterr()
        assert "Train" in captured.out

    def test_keys_are_sorted(self, capsys):
        metrics = {"z_metric": 1, "a_metric": 2, "m_metric": 3}
        log_metrics(metrics)
        captured = capsys.readouterr()
        # Check relative ordering via index in raw output
        pos_a = captured.out.index("a_metric")
        pos_m = captured.out.index("m_metric")
        pos_z = captured.out.index("z_metric")
        assert pos_a < pos_m < pos_z

    def test_float_values_formatted_with_4f(self, capsys):
        log_metrics({"loss": 0.123456789})
        captured = capsys.readouterr()
        assert "0.1235" in captured.out
        # Should NOT show the full unformatted float
        assert "0.123456789" not in captured.out

    def test_int_values_not_formatted_as_float(self, capsys):
        log_metrics({"epoch": 5})
        captured = capsys.readouterr()
        assert "epoch: 5" in captured.out
        # Should not contain decimal formatting
        assert "5.0000" not in captured.out

    def test_single_metric_with_very_long_key(self, capsys):
        """A very long key name should not crash the formatter."""
        long_key = "a" * 300
        log_metrics({long_key: 1.0})
        captured = capsys.readouterr()
        assert long_key in captured.out

    def test_metric_value_is_string(self, capsys):
        """String values (not int/float) should be printed without .4f formatting."""
        log_metrics({"status": "converged"})
        captured = capsys.readouterr()
        assert "status: converged" in captured.out


# ---------------------------------------------------------------------------
# merge_dicts
# ---------------------------------------------------------------------------
class TestMergeDicts:
    def test_empty_list(self):
        result = merge_dicts([])
        assert result == {}

    def test_single_dict(self):
        result = merge_dicts([{"a": 1, "b": 2}])
        assert result == {"a": [1], "b": [2]}

    def test_multiple_dicts_overlapping_keys(self):
        result = merge_dicts([{"a": 1}, {"a": 2}, {"a": 3}])
        assert result == {"a": [1, 2, 3]}

    def test_multiple_dicts_disjoint_keys(self):
        result = merge_dicts([{"a": 1}, {"b": 2}, {"c": 3}])
        assert result == {"a": [1], "b": [2], "c": [3]}

    def test_mixed_overlapping_and_disjoint(self):
        result = merge_dicts([{"a": 1, "b": 10}, {"a": 2, "c": 20}])
        assert result == {"a": [1, 2], "b": [10], "c": [20]}

    def test_values_preserve_order(self):
        result = merge_dicts([{"k": "first"}, {"k": "second"}, {"k": "third"}])
        assert result["k"] == ["first", "second", "third"]

    def test_large_merge_100_dicts(self):
        """Merging 100 dicts with the same key should produce a list of length 100."""
        dicts = [{"x": i} for i in range(100)]
        result = merge_dicts(dicts)
        assert len(result["x"]) == 100
        assert result["x"] == list(range(100))

    def test_values_are_dicts_not_flattened(self):
        """Dict values should be collected as-is, not flattened."""
        result = merge_dicts([{"k": {"nested": 1}}, {"k": {"nested": 2}}])
        assert result == {"k": [{"nested": 1}, {"nested": 2}]}

    def test_values_are_lists_not_flattened(self):
        """List values should be collected as-is, not flattened."""
        result = merge_dicts([{"k": [1, 2]}, {"k": [3, 4]}])
        assert result == {"k": [[1, 2], [3, 4]]}

    def test_none_values(self):
        """None values should be collected like any other value."""
        result = merge_dicts([{"a": None}, {"a": None}, {"a": 1}])
        assert result == {"a": [None, None, 1]}

    def test_empty_dicts_in_list(self):
        """Empty dicts interspersed should be ignored, producing only keys from non-empty dicts."""
        result = merge_dicts([{}, {"a": 1}, {}])
        assert result == {"a": [1]}


# ---------------------------------------------------------------------------
# append_to_dict
# ---------------------------------------------------------------------------
class TestAppendToDict:
    def test_basic_append_scalar(self):
        data = {}
        append_to_dict(data, {"loss": 0.5})
        assert data == {"loss": [0.5]}

    def test_append_accumulates(self):
        data = {"loss": [0.5]}
        append_to_dict(data, {"loss": 0.3})
        assert data == {"loss": [0.5, 0.3]}

    def test_with_prefix(self):
        data = {}
        append_to_dict(data, {"loss": 0.5}, prefix="train/")
        assert data == {"train/loss": [0.5]}

    def test_prefix_already_present_not_doubled(self):
        data = {}
        append_to_dict(data, {"train/loss": 0.5}, prefix="train/")
        assert data == {"train/loss": [0.5]}
        assert "train/train/loss" not in data

    def test_scalar_values_wrapped_in_list(self):
        data = {}
        append_to_dict(data, {"a": 42, "b": "hello"})
        assert data == {"a": [42], "b": ["hello"]}

    def test_list_values_are_extended(self):
        data = {"a": [1]}
        append_to_dict(data, {"a": [2, 3]})
        assert data == {"a": [1, 2, 3]}

    def test_nested_list_values_are_extended(self):
        """A nested list [[1,2], [3]] should extend, producing those sublists as elements."""
        data = {}
        append_to_dict(data, {"a": [[1, 2], [3]]})
        assert data == {"a": [[1, 2], [3]]}

    def test_multiple_prefixes_in_sequence(self):
        """Calling twice with different prefixes should create separate keys."""
        data = {}
        append_to_dict(data, {"loss": 0.5}, prefix="train/")
        append_to_dict(data, {"loss": 0.3}, prefix="eval/")
        assert data == {"train/loss": [0.5], "eval/loss": [0.3]}

    def test_key_collision_between_prefix_and_non_prefix(self):
        """A prefixed key and a non-prefixed key that match should accumulate together."""
        data = {}
        append_to_dict(data, {"train/loss": 0.5})
        append_to_dict(data, {"loss": 0.3}, prefix="train/")
        assert data == {"train/loss": [0.5, 0.3]}

    def test_key_starts_with_prefix_but_is_not_prefix_plus_key(self):
        """A key like 'train/eval/loss' already starts with 'train/' so the prefix
        should NOT be doubled -- the code checks key.startswith(prefix)."""
        data = {}
        append_to_dict(data, {"train/eval/loss": 0.7}, prefix="train/")
        assert data == {"train/eval/loss": [0.7]}
        assert "train/train/eval/loss" not in data
