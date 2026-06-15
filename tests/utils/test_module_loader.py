"""Tests for axon.utils.module_loader module."""

import sys

import pytest

from axon.utils.module_loader import get_random_string, load_module


# ---------------------------------------------------------------------------
# get_random_string
# ---------------------------------------------------------------------------
class TestGetRandomString:
    def test_correct_length_and_charset(self):
        for length in [0, 1, 10, 100]:
            result = get_random_string(length)
            assert len(result) == length
            assert result == "" or result.isalnum()

    def test_nondeterministic(self):
        assert get_random_string(32) != get_random_string(32)


# ---------------------------------------------------------------------------
# load_module – pkg:// protocol
# ---------------------------------------------------------------------------
class TestLoadModulePkg:
    def test_empty_returns_none(self):
        assert load_module("") is None

    def test_loads_stdlib(self):
        import json

        assert load_module("pkg://json") is json

    def test_slash_and_dot_notation_equivalent(self):
        import os.path

        assert load_module("pkg://os/path") is os.path
        assert load_module("pkg://os.path") is os.path


# ---------------------------------------------------------------------------
# load_module – file:// and bare paths
# ---------------------------------------------------------------------------
class TestLoadModuleFile:
    def test_file_prefix(self, tmp_path):
        f = tmp_path / "mod.py"
        f.write_text("X = 'hello'\n")
        assert load_module(f"file://{f}").X == "hello"

    def test_bare_path(self, tmp_path):
        f = tmp_path / "bare.py"
        f.write_text("VALUE = 42\n")
        assert load_module(str(f)).VALUE == 42

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Custom module file not found"):
            load_module(str(tmp_path / "nonexistent.py"))

    def test_syntax_error_raises_runtime(self, tmp_path):
        f = tmp_path / "bad.py"
        f.write_text("def x(\n")
        with pytest.raises(RuntimeError, match="Error loading module"):
            load_module(str(f))

    def test_runtime_error_in_module(self, tmp_path):
        f = tmp_path / "boom.py"
        f.write_text("x = 1 / 0\n")
        with pytest.raises(RuntimeError, match="Error loading module"):
            load_module(str(f))

    def test_module_with_stdlib_import(self, tmp_path):
        f = tmp_path / "imp.py"
        f.write_text("import math\nPI = math.pi\n")
        import math

        assert load_module(str(f)).PI == math.pi


# ---------------------------------------------------------------------------
# module_name registration and conflicts
# ---------------------------------------------------------------------------
class TestModuleNameRegistration:
    def test_registers_in_sys_modules(self, tmp_path):
        f = tmp_path / "named.py"
        f.write_text("Y = 99\n")
        name = "_test_ml_named"
        try:
            mod = load_module(str(f), module_name=name)
            assert sys.modules[name] is mod
            assert mod.Y == 99
        finally:
            sys.modules.pop(name, None)

    def test_same_name_different_file_raises(self, tmp_path):
        a = tmp_path / "a.py"
        a.write_text("A = 1\n")
        b = tmp_path / "b.py"
        b.write_text("B = 2\n")
        name = "_test_ml_conflict"
        try:
            load_module(str(a), module_name=name)
            with pytest.raises(RuntimeError, match="already in `sys.modules`"):
                load_module(str(b), module_name=name)
        finally:
            sys.modules.pop(name, None)

    def test_no_name_creates_independent_objects(self, tmp_path):
        """Without module_name, each load creates a fresh module object."""
        f = tmp_path / "fresh.py"
        f.write_text("V = []\n")
        mod1 = load_module(str(f))
        mod2 = load_module(str(f))
        assert mod1 is not mod2
        mod1.V.append(1)
        assert mod2.V == []  # mutation isolation
