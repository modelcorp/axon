from axon.utils.rewards.ifbench_reward import _coerce_kwargs_list, _normalize_instruction_ids


class TestNormalizeInstructionIds:
    """Test the _normalize_instruction_ids helper function."""

    def test_filters_none(self):
        assert _normalize_instruction_ids(["id1", None, "id2"]) == ["id1", "id2"]

    def test_filters_empty(self):
        assert _normalize_instruction_ids(["id1", "", "id2"]) == ["id1", "id2"]

    def test_converts_to_string(self):
        assert _normalize_instruction_ids([1, 2, 3]) == ["1", "2", "3"]

    def test_strips_whitespace(self):
        assert _normalize_instruction_ids(["  id1  "]) == ["id1"]

    def test_all_none_and_empty(self):
        assert _normalize_instruction_ids([None, "", "  ", None]) == []

    def test_mixed_types(self):
        result = _normalize_instruction_ids(["abc", 42, None, "", "def"])
        assert result == ["abc", "42", "def"]

    def test_whitespace_only_filtered(self):
        """Strings that are only whitespace should be stripped to empty and filtered out."""
        assert _normalize_instruction_ids(["   "]) == []

    def test_boolean_values_converted(self):
        assert _normalize_instruction_ids([True, False]) == ["True", "False"]

    def test_float_values_converted(self):
        assert _normalize_instruction_ids([1.5, 2.7]) == ["1.5", "2.7"]


class TestCoerceKwargsList:
    """Test the _coerce_kwargs_list helper function."""

    def test_single_dict_repeated(self):
        result = _coerce_kwargs_list({"key": "val"}, 2)
        assert len(result) == 2
        assert all(d == {"key": "val"} for d in result)

    def test_list_of_dicts(self):
        result = _coerce_kwargs_list([{"a": 1}, {"b": 2}], 2)
        assert len(result) == 2
        assert result[0] == {"a": 1}
        assert result[1] == {"b": 2}

    def test_pads_short_list(self):
        result = _coerce_kwargs_list([{"a": 1}], 3)
        assert len(result) == 3
        # First element is the original, padding copies the last element
        assert result[0] == {"a": 1}
        assert result[1] == {"a": 1}
        assert result[2] == {"a": 1}

    def test_trims_long_list(self):
        result = _coerce_kwargs_list([{"a": 1}, {"b": 2}, {"c": 3}], 2)
        assert len(result) == 2
        assert result[0] == {"a": 1}
        assert result[1] == {"b": 2}

    def test_filters_none_values(self):
        """None values within kwargs dicts should be filtered out."""
        result = _coerce_kwargs_list([{"a": 1, "b": None}], 1)
        assert result == [{"a": 1}]

    def test_empty_list_pads(self):
        result = _coerce_kwargs_list([], 2)
        assert result == [{}, {}]

    def test_non_dict_elements_in_list(self):
        """Non-dict elements in a list should become empty dicts."""
        result = _coerce_kwargs_list(["not_a_dict", 42], 2)
        assert result == [{}, {}]

    def test_single_dict_copies_are_independent(self):
        """Each repeated dict should be an independent copy."""
        result = _coerce_kwargs_list({"key": "val"}, 2)
        result[0]["key"] = "modified"
        assert result[1]["key"] == "val"

    def test_zero_length(self):
        result = _coerce_kwargs_list([{"a": 1}], 0)
        assert result == []

    def test_deeply_nested_dict_preserved(self):
        result = _coerce_kwargs_list([{"a": {"nested": "val"}}], 1)
        assert result[0]["a"] == {"nested": "val"}

    def test_large_n(self):
        result = _coerce_kwargs_list({"k": "v"}, 100)
        assert len(result) == 100
        assert all(d == {"k": "v"} for d in result)
