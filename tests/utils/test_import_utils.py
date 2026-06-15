"""Tests for axon.utils.import_utils module."""

import os
import sys

import pytest

from axon.utils.import_utils import (
    import_external_libs,
    is_megatron_core_available,
    is_nvtx_available,
    is_sglang_available,
    is_trl_available,
    is_vllm_available,
    load_class_from_fqn,
    load_extern_object,
    load_module,
)


# ---------------------------------------------------------------------------
# is_*_available -- returns bool
# ---------------------------------------------------------------------------
class TestIsAvailableFunctions:
    """Each is_*_available helper should return a bool regardless of what is installed."""

    @pytest.mark.parametrize(
        "func",
        [
            is_megatron_core_available,
            is_vllm_available,
            is_sglang_available,
            is_nvtx_available,
            is_trl_available,
        ],
        ids=[
            "megatron_core",
            "vllm",
            "sglang",
            "nvtx",
            "trl",
        ],
    )
    def test_is_available_returns_bool(self, func):
        assert isinstance(func(), bool)


# ---------------------------------------------------------------------------
# import_external_libs
# ---------------------------------------------------------------------------
class TestImportExternalLibs:
    """Tests for import_external_libs."""

    def test_none_is_noop(self):
        """Passing None should return immediately without error."""
        assert import_external_libs(None) is None

    def test_single_string_imports_module(self):
        """A single string (not wrapped in a list) should be imported."""
        import_external_libs("json")
        assert "json" in sys.modules

    def test_list_of_strings_imports_all(self):
        """A list of module names should import every one of them."""
        import_external_libs(["json", "os", "sys"])
        assert "json" in sys.modules
        assert "os" in sys.modules
        assert "sys" in sys.modules

    def test_bad_module_raises(self):
        """An unknown module name should propagate a ModuleNotFoundError."""
        with pytest.raises(ModuleNotFoundError):
            import_external_libs("absolutely_nonexistent_module_xyz_12345")

    def test_bad_module_in_list_raises(self):
        """If any module in the list is bad the error should propagate."""
        with pytest.raises(ModuleNotFoundError):
            import_external_libs(["json", "absolutely_nonexistent_module_xyz_12345"])

    def test_dotted_submodule_string_imports(self):
        """A single dotted submodule string like 'os.path' should be imported."""
        import_external_libs("os.path")
        assert "os.path" in sys.modules

    def test_empty_list_is_noop(self):
        """An empty list should be a no-op and not raise."""
        import_external_libs([])


# ---------------------------------------------------------------------------
# load_module
# ---------------------------------------------------------------------------
class TestLoadModule:
    """Tests for load_module."""

    def test_empty_path_returns_none(self):
        assert load_module("") is None

    def test_pkg_prefix_loads_stdlib_module(self):
        """pkg://json should load the json module."""
        mod = load_module("pkg://json")
        import json

        assert mod is json

    def test_pkg_prefix_with_slashes(self):
        """pkg://os/path should be translated to os.path and loaded."""
        mod = load_module("pkg://os/path")
        import os.path

        assert mod is os.path

    def test_pkg_prefix_with_dots_loads_submodule(self):
        """pkg://os.path (using dots) should load the os.path submodule."""
        mod = load_module("pkg://os.path")
        import os.path

        assert mod is os.path

    def test_file_prefix_loads_temp_file(self, tmp_path):
        """file:// prefix with a real .py file should load the module."""
        py_file = tmp_path / "hello_mod.py"
        py_file.write_text("GREETING = 'hello'\n")

        mod = load_module(f"file://{py_file}")
        assert hasattr(mod, "GREETING")
        assert mod.GREETING == "hello"

    def test_bare_file_path_loads_module(self, tmp_path):
        """A bare file path (no prefix) should also load the module."""
        py_file = tmp_path / "bare_mod.py"
        py_file.write_text("VALUE = 42\n")

        mod = load_module(str(py_file))
        assert mod.VALUE == 42

    def test_file_that_imports_stdlib(self, tmp_path):
        """A loaded file that imports other stdlib modules should work."""
        py_file = tmp_path / "importing_mod.py"
        py_file.write_text("import math\nPI = math.pi\n")

        mod = load_module(str(py_file))
        import math

        assert mod.PI == math.pi

    def test_missing_file_raises_file_not_found(self, tmp_path):
        missing = str(tmp_path / "does_not_exist.py")
        with pytest.raises(FileNotFoundError, match="Custom module file not found"):
            load_module(missing)

    def test_module_name_registers_in_sys_modules(self, tmp_path):
        """When module_name is provided the module should be added to sys.modules."""
        py_file = tmp_path / "named_mod.py"
        py_file.write_text("X = 99\n")
        mod_name = "_test_import_utils_named_mod"

        try:
            mod = load_module(str(py_file), module_name=mod_name)
            assert mod_name in sys.modules
            assert sys.modules[mod_name] is mod
            assert mod.X == 99
        finally:
            sys.modules.pop(mod_name, None)

    def test_module_name_conflict_raises_runtime_error(self, tmp_path):
        """Loading a second file under the same module_name should raise RuntimeError."""
        file_a = tmp_path / "mod_a.py"
        file_a.write_text("A = 1\n")
        file_b = tmp_path / "mod_b.py"
        file_b.write_text("B = 2\n")
        mod_name = "_test_import_utils_conflict_mod"

        try:
            load_module(str(file_a), module_name=mod_name)
            with pytest.raises(RuntimeError, match="already in `sys.modules`"):
                load_module(str(file_b), module_name=mod_name)
        finally:
            sys.modules.pop(mod_name, None)

    def test_file_with_syntax_error_raises_runtime_error(self, tmp_path):
        """A .py file with a syntax error should raise RuntimeError."""
        bad_file = tmp_path / "bad_syntax.py"
        bad_file.write_text("def oops(\n")  # intentional syntax error

        with pytest.raises(RuntimeError, match="Error loading module"):
            load_module(str(bad_file))

    def test_file_with_runtime_error_raises_runtime_error(self, tmp_path):
        """A .py file that raises at import time (not syntax) should also raise RuntimeError."""
        bad_file = tmp_path / "bad_runtime.py"
        bad_file.write_text("x = 1 / 0\n")  # ZeroDivisionError at module level

        with pytest.raises(RuntimeError, match="Error loading module"):
            load_module(str(bad_file))


# ---------------------------------------------------------------------------
# load_extern_object
# ---------------------------------------------------------------------------
class TestLoadExternObject:
    """Tests for load_extern_object."""

    def test_loads_known_object_from_package(self):
        """Should be able to load json.loads via pkg:// path."""
        import json

        obj = load_extern_object("pkg://json", "loads")
        assert obj is json.loads

    def test_missing_attribute_raises(self):
        with pytest.raises(AttributeError, match="Object not found in module"):
            load_extern_object("pkg://json", "absolutely_nonexistent_attr_xyz")

    def test_loads_class_from_package(self):
        """Should be able to load a class object."""
        import collections

        obj = load_extern_object("pkg://collections", "OrderedDict")
        assert obj is collections.OrderedDict

    def test_loads_object_from_temp_file(self, tmp_path):
        """Should work with file:// paths as well."""
        py_file = tmp_path / "ext_obj_mod.py"
        py_file.write_text("MY_CONST = 'found_it'\n")

        result = load_extern_object(f"file://{py_file}", "MY_CONST")
        assert result == "found_it"

    def test_loads_callable_function(self):
        """Loading a function should return something callable."""
        obj = load_extern_object("pkg://os.path", "join")
        assert callable(obj)
        # Verify it actually works
        assert obj("a", "b") == os.path.join("a", "b")


# ---------------------------------------------------------------------------
# load_class_from_fqn
# ---------------------------------------------------------------------------
class TestLoadClassFromFqn:
    """Tests for load_class_from_fqn."""

    def test_loads_real_class(self):
        """Should load pathlib.Path successfully."""
        import pathlib

        cls = load_class_from_fqn("pathlib.Path")
        assert cls is pathlib.Path

    def test_loads_nested_class(self):
        """Should handle deeper module paths like collections.OrderedDict."""
        import collections

        cls = load_class_from_fqn("collections.OrderedDict")
        assert cls is collections.OrderedDict

    def test_no_dot_raises_value_error(self):
        with pytest.raises(ValueError, match="Invalid"):
            load_class_from_fqn("NoDotHere")

    def test_bad_module_raises_import_error(self):
        with pytest.raises(ImportError, match="Failed to import module"):
            load_class_from_fqn("nonexistent_package_xyz_999.SomeClass")

    def test_bad_class_raises_attribute_error(self):
        with pytest.raises(AttributeError, match="not found in module"):
            load_class_from_fqn("pathlib.CompletelyMadeUpClassName")

    def test_description_appears_in_error(self):
        """The custom description should appear in ValueError messages."""
        with pytest.raises(ValueError, match="my_custom_desc"):
            load_class_from_fqn("NoDot", description="my_custom_desc")

    def test_loads_non_class_attribute(self):
        """load_class_from_fqn with os.path.join -- not a class, but still an attribute.
        The function uses getattr so it should work for any attribute."""
        import os.path

        result = load_class_from_fqn("os.path.join")
        assert result is os.path.join
