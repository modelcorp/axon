"""Tests for axon.utils.py_utils module."""

import os
import warnings

import pytest
from omegaconf import DictConfig, ListConfig

from axon.utils.py_utils import (
    DynamicEnum,
    convert_to_regular_types,
    deprecated,
    hash_string_to_int,
    temp_env_var,
    union_two_dict,
)


# ---------------------------------------------------------------------------
# hash_string_to_int – kept + adversarial
# ---------------------------------------------------------------------------
class TestHashStringToInt:
    def test_deterministic(self):
        assert hash_string_to_int("hello") == hash_string_to_int("hello")

    def test_different_strings_differ(self):
        assert hash_string_to_int("hello") != hash_string_to_int("world")

    def test_returns_int_in_range(self):
        h = hash_string_to_int("test_string")
        assert isinstance(h, int)
        assert 0 <= h < 2**31 - 1

    def test_empty_string(self):
        h = hash_string_to_int("")
        assert isinstance(h, int)
        assert 0 <= h < 2**31 - 1

    def test_unicode_string(self):
        h = hash_string_to_int("\u00e9\u00e0\u00fc\u2603\U0001f600")
        assert isinstance(h, int)
        assert 0 <= h < 2**31 - 1
        # Deterministic for same unicode
        assert h == hash_string_to_int("\u00e9\u00e0\u00fc\u2603\U0001f600")

    def test_very_long_string(self):
        s = "a" * 1_000_000
        h = hash_string_to_int(s)
        assert isinstance(h, int)
        assert 0 <= h < 2**31 - 1

    def test_bytes_input_raises(self):
        """Passing bytes instead of str should raise a clear error."""
        with pytest.raises((AttributeError, TypeError)):
            hash_string_to_int(b"hello")

    def test_whitespace_strings_differ(self):
        """Strings differing only in whitespace should produce different hashes."""
        h1 = hash_string_to_int("hello world")
        h2 = hash_string_to_int("hello  world")
        h3 = hash_string_to_int("hello\tworld")
        assert h1 != h2
        assert h1 != h3


# ---------------------------------------------------------------------------
# union_two_dict – kept + adversarial
# ---------------------------------------------------------------------------
class TestUnionTwoDict:
    def test_merges_disjoint_dicts(self):
        d1 = {"a": 1}
        d2 = {"b": 2}
        result = union_two_dict(d1, d2)
        assert result == {"a": 1, "b": 2}
        assert result is d1  # modifies d1 in-place

    def test_same_key_same_value_ok(self):
        d1 = {"a": 1}
        d2 = {"a": 1}
        result = union_two_dict(d1, d2)
        assert result == {"a": 1}

    def test_same_key_different_value_raises(self):
        d1 = {"a": 1}
        d2 = {"a": 2}
        with pytest.raises(AssertionError, match="not the same object"):
            union_two_dict(d1, d2)

    def test_both_empty(self):
        d1, d2 = {}, {}
        result = union_two_dict(d1, d2)
        assert result == {}
        assert result is d1

    def test_mutable_values_equal_but_not_identical(self):
        """Two lists that are == but not 'is' should pass (uses == check)."""
        d1 = {"a": [1, 2, 3]}
        d2 = {"a": [1, 2, 3]}
        assert d1["a"] is not d2["a"]
        result = union_two_dict(d1, d2)
        assert result["a"] == [1, 2, 3]

    def test_nan_values_same_key_should_not_raise(self):
        """NaN == NaN is False in Python; both dicts have NaN so merge should succeed."""
        import math

        d1 = {"a": float("nan")}
        d2 = {"a": float("nan")}
        # Both dicts have NaN for "a" — logically the same, should merge without error
        result = union_two_dict(d1, d2)
        assert "a" in result
        assert math.isnan(result["a"])

    def test_none_values_same_key_ok(self):
        """None == None is True, so this should merge fine."""
        d1 = {"a": None}
        d2 = {"a": None}
        result = union_two_dict(d1, d2)
        assert result == {"a": None}

    def test_large_dict_merge(self):
        """Merging two large dicts with disjoint keys."""
        d1 = {f"key_{i}": i for i in range(1000)}
        d2 = {f"other_{i}": i for i in range(1000)}
        result = union_two_dict(d1, d2)
        assert len(result) == 2000


# ---------------------------------------------------------------------------
# convert_to_regular_types – kept + adversarial
# ---------------------------------------------------------------------------
class TestConvertToRegularTypes:
    def test_dictconfig_to_dict(self):
        cfg = DictConfig({"key": "value"})
        result = convert_to_regular_types(cfg)
        assert isinstance(result, dict)
        assert result == {"key": "value"}

    def test_listconfig_to_list(self):
        cfg = ListConfig([1, 2, 3])
        result = convert_to_regular_types(cfg)
        assert isinstance(result, list)
        assert result == [1, 2, 3]

    def test_nested_structure(self):
        cfg = DictConfig({"items": [1, 2], "sub": {"x": 10}})
        result = convert_to_regular_types(cfg)
        assert isinstance(result, dict)
        assert isinstance(result["items"], list)
        assert isinstance(result["sub"], dict)

    def test_passthrough_plain_types(self):
        assert convert_to_regular_types(42) == 42
        assert convert_to_regular_types("hello") == "hello"
        assert convert_to_regular_types(None) is None

    def test_deeply_nested_four_levels(self):
        cfg = DictConfig({"a": {"b": {"c": {"d": "deep"}}}})
        result = convert_to_regular_types(cfg)
        assert isinstance(result, dict)
        assert isinstance(result["a"], dict)
        assert isinstance(result["a"]["b"], dict)
        assert isinstance(result["a"]["b"]["c"], dict)
        assert result["a"]["b"]["c"]["d"] == "deep"

    def test_tuple_type_preserved(self):
        """convert_to_regular_types should preserve tuple type, not silently convert to list."""
        result = convert_to_regular_types((1, 2, 3))
        assert isinstance(result, tuple), f"Expected tuple but got {type(result).__name__}"

    def test_tuple_containing_dictconfig(self):
        """Tuple containing DictConfig items should have inner items converted to dict."""
        cfg = DictConfig({"a": 1})
        result = convert_to_regular_types((cfg, cfg))
        # Inner DictConfig should be converted to dict
        for item in result:
            assert isinstance(item, dict)
            assert item == {"a": 1}


# ---------------------------------------------------------------------------
# temp_env_var – kept + adversarial
# ---------------------------------------------------------------------------
class TestTempEnvVar:
    def test_sets_var_inside_context(self):
        key = "_AXON_TEST_TEMP_VAR"
        os.environ.pop(key, None)
        with temp_env_var(key, "test_value"):
            assert os.environ[key] == "test_value"

    def test_restores_original_after(self):
        key = "_AXON_TEST_TEMP_VAR2"
        os.environ[key] = "original"
        try:
            with temp_env_var(key, "temp"):
                assert os.environ[key] == "temp"
            assert os.environ[key] == "original"
        finally:
            os.environ.pop(key, None)

    def test_removes_if_did_not_exist_before(self):
        key = "_AXON_TEST_TEMP_VAR3"
        os.environ.pop(key, None)
        with temp_env_var(key, "temp"):
            pass
        assert key not in os.environ

    def test_restores_even_when_exception_raised(self):
        key = "_AXON_TEST_TEMP_VAR4"
        os.environ[key] = "original"
        try:
            with pytest.raises(RuntimeError):
                with temp_env_var(key, "temp"):
                    assert os.environ[key] == "temp"
                    raise RuntimeError("boom")
            # Must still restore after the exception
            assert os.environ[key] == "original"
        finally:
            os.environ.pop(key, None)

    def test_nested_temp_env_var(self):
        """Nested temp_env_var contexts should restore correctly at each level."""
        key = "_AXON_TEST_NESTED"
        os.environ.pop(key, None)
        with temp_env_var(key, "outer"):
            assert os.environ[key] == "outer"
            with temp_env_var(key, "inner"):
                assert os.environ[key] == "inner"
            assert os.environ[key] == "outer"
        assert key not in os.environ

    def test_temp_env_var_with_empty_value(self):
        """Setting env var to empty string should work."""
        key = "_AXON_TEST_EMPTY_VAL"
        os.environ.pop(key, None)
        with temp_env_var(key, ""):
            assert os.environ[key] == ""
        assert key not in os.environ


# ---------------------------------------------------------------------------
# deprecated – kept
# ---------------------------------------------------------------------------
class TestDeprecated:
    def test_function_decorator_emits_future_warning(self):
        @deprecated(replacement="new_func")
        def old_func():
            return 42

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = old_func()
            assert result == 42
            assert len(w) == 1
            assert issubclass(w[0].category, FutureWarning)
            assert "deprecated" in str(w[0].message).lower()
            assert "new_func" in str(w[0].message)

    def test_class_decorator_emits_future_warning_on_init(self):
        @deprecated(replacement="NewClass")
        class OldClass:
            def __init__(self):
                self.x = 1

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            obj = OldClass()
            assert obj.x == 1
            assert len(w) == 1
            assert issubclass(w[0].category, FutureWarning)
            assert "NewClass" in str(w[0].message)


# ---------------------------------------------------------------------------
# DynamicEnum – kept + adversarial
# ---------------------------------------------------------------------------
class TestDynamicEnum:
    """Each test uses a fresh subclass to avoid polluting global state."""

    def _make_enum(self):
        class Color(DynamicEnum):
            _registry = {}
            _next_value = 0

        return Color

    def test_register(self):
        Color = self._make_enum()
        red = Color.register("red")
        assert red.name == "RED"
        assert red.value == 0
        assert hasattr(Color, "RED")

    def test_names_and_values(self):
        Color = self._make_enum()
        Color.register("red")
        Color.register("blue")
        assert set(Color.names()) == {"RED", "BLUE"}
        assert len(Color.values()) == 2

    def test_from_name(self):
        Color = self._make_enum()
        red = Color.register("red")
        assert Color.from_name("red") is red
        assert Color.from_name("RED") is red
        assert Color.from_name("nonexistent") is None

    def test_remove(self):
        Color = self._make_enum()
        Color.register("green")
        removed = Color.remove("green")
        assert removed.name == "GREEN"
        assert "GREEN" not in Color
        assert not hasattr(Color, "GREEN")

    def test_contains_name(self):
        Color = self._make_enum()
        Color.register("red")
        assert "RED" in Color
        assert "BLUE" not in Color

    def test_contains_member(self):
        Color = self._make_enum()
        red = Color.register("red")
        assert red in Color

    def test_iter(self):
        Color = self._make_enum()
        Color.register("red")
        Color.register("blue")
        assert len(list(Color)) == 2

    def test_duplicate_raises_value_error(self):
        Color = self._make_enum()
        Color.register("red")
        with pytest.raises(ValueError, match="already registered"):
            Color.register("red")

    # -- adversarial --

    def test_remove_nonexistent_raises_key_error(self):
        Color = self._make_enum()
        with pytest.raises(KeyError):
            Color.remove("nonexistent")

    def test_getitem_nonexistent_raises_key_error(self):
        Color = self._make_enum()
        with pytest.raises(KeyError):
            Color["NONEXISTENT"]

    def test_case_insensitive_register(self):
        """register('foo') stores under 'FOO'."""
        Color = self._make_enum()
        member = Color.register("foo")
        assert member.name == "FOO"
        assert "FOO" in Color
        assert hasattr(Color, "FOO")

    def test_value_counter_does_not_reset_after_remove(self):
        Color = self._make_enum()
        Color.register("red")  # value=0
        Color.register("green")  # value=1
        Color.remove("red")
        blue = Color.register("blue")  # value=2, NOT 0
        assert blue.value == 2

    def test_register_empty_string_name(self):
        """Registering with empty string should not corrupt the enum."""
        Color = self._make_enum()
        member = Color.register("")
        assert member.name == ""
        assert "" in Color

    def test_register_special_characters(self):
        """Special characters in name should be stored uppercased."""
        Color = self._make_enum()
        member = Color.register("hello-world")
        assert member.name == "HELLO-WORLD"
        assert "HELLO-WORLD" in Color

    def test_subclass_registry_isolation(self):
        """Two DynamicEnum subclasses with separate _registry should not interfere."""
        Color = self._make_enum()
        Shape = self._make_enum()
        Color.register("red")
        Shape.register("circle")
        assert "RED" in Color
        assert "CIRCLE" not in Color
        assert "CIRCLE" in Shape
        assert "RED" not in Shape
