"""
Tests for FunctionRegistry (axon.utils.registry.function_registry).
"""

from enum import Enum

import pytest

from axon.utils.registry import FunctionRegistry, FunctionRegistryEntry


class DummyFn(str, Enum):
    ADD = "add"
    MUL = "mul"


class TestRegistration:
    def test_register_creates_entry(self):
        reg = FunctionRegistry[int](name="test")

        @reg.register("my_fn")
        def my_fn(data, config):
            return 1

        assert "my_fn" in reg
        entry = reg.get_entry("my_fn")
        assert isinstance(entry, FunctionRegistryEntry)
        assert entry.fn is my_fn

    def test_register_with_enum(self):
        reg = FunctionRegistry[int](name="test")

        @reg.register(DummyFn.ADD)
        def add_fn(data, config):
            return 1

        assert "add" in reg
        assert reg.get_fn(DummyFn.ADD) is add_fn
        assert reg.get_fn("add") is add_fn

    def test_get_entry_nonexistent_raises(self):
        reg = FunctionRegistry[int](name="test")
        with pytest.raises(ValueError, match="Unknown test"):
            reg.get_entry("nope")

    def test_duplicate_registration_same_fn_ok(self):
        reg = FunctionRegistry[int](name="test")

        def my_fn(data, config):
            return 1

        reg.register("dup")(my_fn)
        reg.register("dup")(my_fn)
        assert reg.get_fn("dup") is my_fn

    def test_duplicate_registration_different_fn_raises(self):
        reg = FunctionRegistry[int](name="test")

        @reg.register("conflict")
        def fn1(data, config):
            return 1

        with pytest.raises(ValueError, match="already registered"):

            @reg.register("conflict")
            def fn2(data, config):
                return 2

    def test_keys(self):
        reg = FunctionRegistry[int](name="test")

        @reg.register("a")
        def fn_a(data, config):
            return 1

        @reg.register("b")
        def fn_b(data, config):
            return 2

        assert set(reg.keys()) == {"a", "b"}

    def test_contains_str(self):
        reg = FunctionRegistry[int](name="test")

        @reg.register("present")
        def fn(data, config):
            return 1

        assert "present" in reg
        assert "absent" not in reg

    def test_contains_enum(self):
        reg = FunctionRegistry[int](name="test")

        @reg.register(DummyFn.MUL)
        def fn(data, config):
            return 1

        assert DummyFn.MUL in reg
        assert DummyFn.ADD not in reg


class TestCompute:
    def test_compute_calls_fn(self):
        reg = FunctionRegistry[int](name="test")

        @reg.register("inc")
        def inc_fn(data, config):
            return 42

        result = reg.compute("inc", data=None, config=None)
        assert result == 42

    def test_compute_passes_args(self):
        reg = FunctionRegistry[str](name="test")

        @reg.register("echo")
        def echo_fn(data, config):
            return f"{data}-{config}"

        result = reg.compute("echo", data="hello", config="world")
        assert result == "hello-world"

    def test_compute_with_enum(self):
        reg = FunctionRegistry[int](name="test")

        @reg.register(DummyFn.ADD)
        def add_fn(data, config):
            return 99

        result = reg.compute(DummyFn.ADD, data=None, config=None)
        assert result == 99

    def test_compute_nonexistent_raises(self):
        reg = FunctionRegistry[int](name="test")
        with pytest.raises(ValueError, match="Unknown test"):
            reg.compute("nope", data=None, config=None)


# ---------------------------------------------------------------------------
# Hardened edge cases
# ---------------------------------------------------------------------------
class TestFunctionRegistryEdgeCases:
    def test_register_with_empty_string_name(self):
        """Registering with empty string name should work (or raise clearly)."""
        reg = FunctionRegistry[int](name="test")

        @reg.register("")
        def fn(data, config):
            return 1

        assert "" in reg
        assert reg.get_fn("") is fn

    def test_register_with_none_enum_value(self):
        """Enum with None value should work if used as string."""

        class NullEnum(str, Enum):
            NONE = "none"

        reg = FunctionRegistry[int](name="test")

        @reg.register(NullEnum.NONE)
        def fn(data, config):
            return 0

        assert "none" in reg

    def test_compute_propagates_exceptions(self):
        """If the registered function raises, compute should propagate it."""
        reg = FunctionRegistry[int](name="test")

        @reg.register("boom")
        def boom_fn(data, config):
            raise RuntimeError("intentional")

        with pytest.raises(RuntimeError, match="intentional"):
            reg.compute("boom", data=None, config=None)

    def test_get_fn_returns_callable(self):
        reg = FunctionRegistry[int](name="test")

        @reg.register("fn")
        def my_fn(data, config):
            return 42

        fn = reg.get_fn("fn")
        assert callable(fn)
        assert fn(None, None) == 42

    def test_register_lambda(self):
        """Lambdas should be registerable."""
        reg = FunctionRegistry[int](name="test")
        reg.register("lam")(lambda data, config: 99)
        assert reg.compute("lam", data=None, config=None) == 99

    def test_keys_empty_registry(self):
        reg = FunctionRegistry[int](name="test")
        assert reg.keys() == []

    def test_contains_with_string_on_enum_registered(self):
        """String lookup should work for enum-registered functions."""

        class MyFn(str, Enum):
            FOO = "foo"

        reg = FunctionRegistry[int](name="test")

        @reg.register(MyFn.FOO)
        def fn(data, config):
            return 1

        assert "foo" in reg
        assert MyFn.FOO in reg
