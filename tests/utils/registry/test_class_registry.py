import os
import sys
import tempfile
import threading

import pytest

from axon.utils.registry.class_registry import ClassRegistry


class TestClassRegistryInit:
    def test_empty_registry(self):
        reg = ClassRegistry("test")
        assert reg.name == "test"
        assert reg._registry == {}
        assert reg._discovered is False

    def test_distinct_registries_are_independent(self):
        reg1 = ClassRegistry("alpha")
        reg2 = ClassRegistry("beta")
        assert reg1.name == "alpha"
        assert reg2.name == "beta"
        assert reg1._registry is not reg2._registry


class TestClassRegistryRegister:
    def test_register_class(self):
        reg = ClassRegistry("test")

        @reg.register("foo")
        class Foo:
            pass

        assert reg._registry["foo"] is Foo

    def test_register_multiple_classes(self):
        reg = ClassRegistry("test")

        @reg.register("a")
        class A:
            pass

        @reg.register("b")
        class B:
            pass

        assert reg._registry["a"] is A
        assert reg._registry["b"] is B
        assert len(reg._registry) == 2

    def test_idempotent_registration_returns_first_class(self):
        reg = ClassRegistry("test")

        @reg.register("foo")
        class Foo1:
            pass

        @reg.register("foo")
        class Foo2:
            pass

        assert reg._registry["foo"] is Foo1

    def test_idempotent_registration_decorator_returns_first_class(self):
        reg = ClassRegistry("test")

        @reg.register("dup")
        class First:
            VALUE = 1

        # The decorator should return the already-registered class
        result = reg.register("dup")(type("Second", (), {"VALUE": 2}))
        assert result is First
        assert result.VALUE == 1

    def test_register_non_class_object(self):
        reg = ClassRegistry("test")

        @reg.register("func")
        def some_function():
            return 42

        assert reg._registry["func"] is some_function


class TestClassRegistryGet:
    def test_get_registered(self):
        reg = ClassRegistry("test")

        @reg.register("bar")
        class Bar:
            pass

        reg._discovered = True
        assert reg.get("bar") is Bar

    def test_getitem_syntax(self):
        reg = ClassRegistry("test")

        @reg.register("bar")
        class Bar:
            pass

        reg._discovered = True
        assert reg["bar"] is Bar

    def test_get_unknown_raises_valueerror(self):
        reg = ClassRegistry("test")
        reg._discovered = True
        with pytest.raises(ValueError, match="Unknown test"):
            reg.get("nonexistent")

    def test_getitem_unknown_raises_valueerror(self):
        reg = ClassRegistry("test")
        reg._discovered = True
        with pytest.raises(ValueError, match="Unknown test"):
            reg["nonexistent"]

    def test_get_error_message_lists_available(self):
        reg = ClassRegistry("test")

        @reg.register("alpha")
        class Alpha:
            pass

        @reg.register("beta")
        class Beta:
            pass

        reg._discovered = True
        with pytest.raises(ValueError, match="Available:.*alpha.*beta"):
            reg.get("gamma")

    def test_get_error_message_suggests_module_syntax(self):
        reg = ClassRegistry("test")
        reg._discovered = True
        with pytest.raises(ValueError, match="module:ClassName"):
            reg.get("nonexistent")

    def test_get_error_message_suggests_file_syntax(self):
        reg = ClassRegistry("test")
        reg._discovered = True
        with pytest.raises(ValueError, match="/path/to/file.py:ClassName"):
            reg.get("nonexistent")


class TestClassRegistryContains:
    def test_contains_registered(self):
        reg = ClassRegistry("test")

        @reg.register("x")
        class X:
            pass

        reg._discovered = True
        assert "x" in reg

    def test_not_contains_unregistered(self):
        reg = ClassRegistry("test")
        reg._discovered = True
        assert "missing" not in reg


class TestClassRegistryKeysValuesItems:
    def test_keys(self):
        reg = ClassRegistry("test")

        @reg.register("b")
        class B:
            pass

        @reg.register("a")
        class A:
            pass

        reg._discovered = True
        # keys() delegates to list() which returns sorted keys
        assert reg.keys() == ["a", "b"]

    def test_values(self):
        reg = ClassRegistry("test")

        @reg.register("one")
        class One:
            pass

        reg._discovered = True
        vals = list(reg.values())
        assert len(vals) == 1
        assert vals[0] is One

    def test_items(self):
        reg = ClassRegistry("test")

        @reg.register("cls")
        class Cls:
            pass

        reg._discovered = True
        items_list = list(reg.items())
        assert len(items_list) == 1
        assert items_list[0] == ("cls", Cls)

    def test_empty_keys_values_items(self):
        reg = ClassRegistry("test")
        reg._discovered = True
        assert reg.keys() == []
        assert list(reg.values()) == []
        assert list(reg.items()) == []


class TestClassRegistrySetitemUpdate:
    def test_setitem(self):
        reg = ClassRegistry("test")
        reg._discovered = True

        class MyClass:
            pass

        reg["myclass"] = MyClass
        assert reg._registry["myclass"] is MyClass
        assert reg.get("myclass") is MyClass

    def test_setitem_overwrite(self):
        reg = ClassRegistry("test")
        reg._discovered = True

        class First:
            pass

        class Second:
            pass

        reg["key"] = First
        reg["key"] = Second
        assert reg._registry["key"] is Second

    def test_update_with_mapping(self):
        reg = ClassRegistry("test")
        reg._discovered = True

        class A:
            pass

        class B:
            pass

        reg.update({"a": A, "b": B})
        assert reg._registry["a"] is A
        assert reg._registry["b"] is B

    def test_update_merges_with_existing(self):
        reg = ClassRegistry("test")

        @reg.register("existing")
        class Existing:
            pass

        reg._discovered = True

        class New:
            pass

        reg.update({"new": New})
        assert "existing" in reg._registry
        assert "new" in reg._registry


class TestClassRegistryList:
    def test_list_sorted(self):
        reg = ClassRegistry("test")

        @reg.register("zebra")
        class Zebra:
            pass

        @reg.register("apple")
        class Apple:
            pass

        @reg.register("mango")
        class Mango:
            pass

        reg._discovered = True
        assert reg.list() == ["apple", "mango", "zebra"]

    def test_list_empty(self):
        reg = ClassRegistry("test")
        reg._discovered = True
        assert reg.list() == []


class TestClassRegistryDynamicImport:
    def test_import_from_module_path(self):
        reg = ClassRegistry("test")
        reg._discovered = True
        cls = reg.get("collections:OrderedDict")
        from collections import OrderedDict

        assert cls is OrderedDict

    def test_import_from_module_path_cached(self):
        reg = ClassRegistry("test")
        reg._discovered = True
        cls1 = reg.get("collections:OrderedDict")
        cls2 = reg.get("collections:OrderedDict")
        assert cls1 is cls2
        assert "collections:OrderedDict" in reg._registry

    def test_import_stdlib_defaultdict(self):
        reg = ClassRegistry("test")
        reg._discovered = True
        cls = reg.get("collections:defaultdict")
        from collections import defaultdict

        assert cls is defaultdict

    def test_import_nonexistent_module_raises(self):
        reg = ClassRegistry("test")
        reg._discovered = True
        with pytest.raises(ValueError, match="Failed to import"):
            reg.get("nonexistent_module_xyz:SomeClass")

    def test_import_nonexistent_attribute_raises(self):
        reg = ClassRegistry("test")
        reg._discovered = True
        with pytest.raises(ValueError, match="Failed to import"):
            reg.get("collections:NonExistentClass999")

    def test_import_from_file(self):
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write("class MyClass:\n    VALUE = 42\n")
            f.flush()
            filepath = f.name
        try:
            reg = ClassRegistry("test")
            reg._discovered = True
            cls = reg.get(f"{filepath}:MyClass")
            assert cls.VALUE == 42
        finally:
            os.unlink(filepath)

    def test_import_from_file_instance_creation(self):
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write(
                "class Greeter:\n"
                "    def __init__(self, name):\n"
                "        self.name = name\n"
                "    def greet(self):\n"
                "        return f'Hello, {self.name}'\n"
            )
            f.flush()
            filepath = f.name
        try:
            reg = ClassRegistry("test")
            reg._discovered = True
            cls = reg.get(f"{filepath}:Greeter")
            obj = cls("World")
            assert obj.greet() == "Hello, World"
        finally:
            os.unlink(filepath)

    def test_import_from_file_is_cached(self):
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write("class Cached:\n    pass\n")
            f.flush()
            filepath = f.name
        try:
            reg = ClassRegistry("test")
            reg._discovered = True
            cls1 = reg.get(f"{filepath}:Cached")
            cls2 = reg.get(f"{filepath}:Cached")
            assert cls1 is cls2
        finally:
            os.unlink(filepath)

    def test_import_nonexistent_file_raises(self):
        reg = ClassRegistry("test")
        reg._discovered = True
        with pytest.raises(ValueError, match="Failed to import"):
            reg.get("/nonexistent/path.py:Cls")

    def test_import_file_with_missing_class_raises(self):
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write("class Exists:\n    pass\n")
            f.flush()
            filepath = f.name
        try:
            reg = ClassRegistry("test")
            reg._discovered = True
            with pytest.raises((ValueError, AttributeError)):
                reg.get(f"{filepath}:DoesNotExist")
        finally:
            os.unlink(filepath)

    def test_import_file_with_syntax_error_raises(self):
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write("class Broken\n    pass\n")  # missing colon
            f.flush()
            filepath = f.name
        try:
            reg = ClassRegistry("test")
            reg._discovered = True
            with pytest.raises((ValueError, SyntaxError)):
                reg.get(f"{filepath}:Broken")
        finally:
            # Clean up any cached module
            keys_to_remove = [k for k in sys.modules if k.startswith("_axon_dynamic_")]
            for k in keys_to_remove:
                sys.modules.pop(k, None)
            os.unlink(filepath)

    def test_import_from_file_adds_parent_to_sys_path(self):
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False, dir=tempfile.gettempdir()) as f:
            f.write("class PathTest:\n    pass\n")
            f.flush()
            filepath = f.name
        try:
            parent = str(os.path.dirname(filepath))
            reg = ClassRegistry("test")
            reg._discovered = True
            reg.get(f"{filepath}:PathTest")
            assert parent in sys.path
        finally:
            os.unlink(filepath)


class TestClassRegistryImportFromFile:
    def test_import_from_file_direct(self):
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write("class Direct:\n    VALUE = 99\n")
            f.flush()
            filepath = f.name
        try:
            reg = ClassRegistry("test")
            cls = reg._import_from_file(filepath, "Direct")
            assert cls.VALUE == 99
        finally:
            os.unlink(filepath)

    def test_import_from_file_nonexistent(self):
        reg = ClassRegistry("test")
        with pytest.raises(ImportError, match="File not found"):
            reg._import_from_file("/no/such/file.py", "Cls")

    def test_import_from_file_creates_unique_module_name(self):
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write("class UniqueModule:\n    pass\n")
            f.flush()
            filepath = f.name
        try:
            reg = ClassRegistry("test")
            reg._import_from_file(filepath, "UniqueModule")
            stem = os.path.basename(filepath).replace(".py", "")
            matching = [k for k in sys.modules if k.startswith(f"_axon_dynamic_{stem}")]
            assert len(matching) == 1
        finally:
            os.unlink(filepath)


class TestClassRegistryDynamicImportMethod:
    def test_dynamic_import_module_colon_class(self):
        reg = ClassRegistry("test")
        cls = reg._dynamic_import("collections:OrderedDict")
        from collections import OrderedDict

        assert cls is OrderedDict

    def test_dynamic_import_file_colon_class(self):
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write("class DynTest:\n    pass\n")
            f.flush()
            filepath = f.name
        try:
            reg = ClassRegistry("test")
            cls = reg._dynamic_import(f"{filepath}:DynTest")
            assert cls.__name__ == "DynTest"
        finally:
            os.unlink(filepath)

    def test_dynamic_import_nonexistent_module_raises(self):
        reg = ClassRegistry("test")
        with pytest.raises(ModuleNotFoundError):
            reg._dynamic_import("totally_fake_module:FakeClass")


class TestClassRegistryDiscovery:
    def test_ensure_discovered_sets_flag(self):
        reg = ClassRegistry("test")
        assert reg._discovered is False
        reg._ensure_discovered()
        assert reg._discovered is True

    def test_get_triggers_discovery(self):
        reg = ClassRegistry("test")

        @reg.register("pre")
        class Pre:
            pass

        assert reg._discovered is False
        reg.get("pre")
        assert reg._discovered is True

    def test_contains_triggers_discovery(self):
        reg = ClassRegistry("test")
        assert reg._discovered is False
        assert "anything" not in reg
        assert reg._discovered is True

    def test_list_triggers_discovery(self):
        reg = ClassRegistry("test")
        assert reg._discovered is False
        reg.list()
        assert reg._discovered is True

    def test_keys_triggers_discovery(self):
        reg = ClassRegistry("test")
        assert reg._discovered is False
        reg.keys()
        assert reg._discovered is True

    def test_values_triggers_discovery(self):
        reg = ClassRegistry("test")
        assert reg._discovered is False
        reg.values()
        assert reg._discovered is True

    def test_items_triggers_discovery(self):
        reg = ClassRegistry("test")
        assert reg._discovered is False
        reg.items()
        assert reg._discovered is True

    def test_setitem_triggers_discovery(self):
        reg = ClassRegistry("test")
        assert reg._discovered is False

        class X:
            pass

        reg["x"] = X
        assert reg._discovered is True

    def test_update_triggers_discovery(self):
        reg = ClassRegistry("test")
        assert reg._discovered is False
        reg.update({})
        assert reg._discovered is True

    def test_discover_directory(self, tmp_path):
        plugin_dir = tmp_path / "plugins"
        plugin_dir.mkdir()
        plugin_file = plugin_dir / "my_plugin.py"
        plugin_file.write_text("# A plugin file\nPLUGIN_LOADED = True\n")
        reg = ClassRegistry("test")
        reg._discover_directory(plugin_dir)
        assert str(plugin_dir) in sys.path

    def test_discover_directory_skips_underscored_files(self, tmp_path):
        plugin_dir = tmp_path / "plugins"
        plugin_dir.mkdir()
        (plugin_dir / "_private.py").write_text("PRIVATE = True\n")
        (plugin_dir / "__init__.py").write_text("")
        reg = ClassRegistry("test")
        reg._discover_directory(plugin_dir)
        # Should not crash; underscored files are skipped

    def test_try_import_bad_module_does_not_raise(self):
        reg = ClassRegistry("test")
        # Should log a warning but not raise
        reg._try_import("completely_nonexistent_module_xyz")

    def test_plugin_env_var(self, monkeypatch, tmp_path):
        plugin_file = tmp_path / "env_plugin.py"
        plugin_file.write_text("ENV_LOADED = True\n")
        monkeypatch.setenv("AXON_TEST_PLUGINS", "")
        reg = ClassRegistry("test")
        reg._ensure_discovered()
        # Should not crash with empty plugin env var


class TestClassRegistryThreadSafety:
    def test_concurrent_registration(self):
        reg = ClassRegistry("test")
        errors = []

        def register_class(i):
            try:

                @reg.register(f"cls_{i}")
                class C:
                    pass
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=register_class, args=(i,)) for i in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(reg._registry) == 50

    def test_concurrent_get(self):
        reg = ClassRegistry("test")

        @reg.register("shared")
        class Shared:
            pass

        reg._discovered = True
        results = []
        errors = []

        def get_class():
            try:
                cls = reg.get("shared")
                results.append(cls)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=get_class) for _ in range(30)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 30
        assert all(r is Shared for r in results)

    def test_concurrent_registration_same_name(self):
        reg = ClassRegistry("test")
        results = []
        errors = []

        def register_and_collect(i):
            try:
                cls = reg.register("same")(type(f"Cls{i}", (), {"index": i}))
                results.append(cls)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=register_and_collect, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 20
        # All threads should get back the same class (the first one registered)
        first = results[0]
        assert all(r is first for r in results)
        assert reg._registry["same"] is first

    def test_concurrent_dynamic_import(self):
        reg = ClassRegistry("test")
        reg._discovered = True
        results = []
        errors = []

        def import_class():
            try:
                cls = reg.get("collections:OrderedDict")
                results.append(cls)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=import_class) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        from collections import OrderedDict

        assert len(errors) == 0
        assert len(results) == 20
        assert all(r is OrderedDict for r in results)

    def test_concurrent_file_import(self):
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write("class ThreadSafe:\n    VALUE = 'ok'\n")
            f.flush()
            filepath = f.name

        try:
            reg = ClassRegistry("test")
            reg._discovered = True
            results = []
            errors = []

            def import_from_file():
                try:
                    cls = reg.get(f"{filepath}:ThreadSafe")
                    results.append(cls)
                except Exception as e:
                    errors.append(e)

            threads = [threading.Thread(target=import_from_file) for _ in range(15)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert len(errors) == 0
            assert len(results) == 15
            assert all(r.VALUE == "ok" for r in results)
            # All should be the same class object
            assert all(r is results[0] for r in results)
        finally:
            os.unlink(filepath)


class TestClassRegistryEdgeCases:
    def test_register_then_get_without_discovery_skip(self):
        reg = ClassRegistry("test")

        @reg.register("early")
        class Early:
            pass

        # get() should work even without manually setting _discovered
        # because _ensure_discovered will run and then find the class
        result = reg.get("early")
        assert result is Early

    def test_get_with_colon_no_registration(self):
        reg = ClassRegistry("test")
        reg._discovered = True
        cls = reg.get("json:JSONEncoder")
        import json

        assert cls is json.JSONEncoder

    def test_setitem_then_getitem(self):
        reg = ClassRegistry("test")
        reg._discovered = True

        class Custom:
            pass

        reg["custom"] = Custom
        assert reg["custom"] is Custom

    def test_update_then_list(self):
        reg = ClassRegistry("test")
        reg._discovered = True

        class X:
            pass

        class Y:
            pass

        reg.update({"x": X, "y": Y})
        assert reg.list() == ["x", "y"]

    def test_register_preserves_class_identity(self):
        reg = ClassRegistry("test")

        @reg.register("identity")
        class Original:
            ATTR = "original"

        reg._discovered = True
        retrieved = reg.get("identity")
        assert retrieved is Original
        assert retrieved.ATTR == "original"

    def test_multiple_registries_isolated(self):
        reg1 = ClassRegistry("type_a")
        reg2 = ClassRegistry("type_b")

        @reg1.register("shared_name")
        class InReg1:
            pass

        @reg2.register("shared_name")
        class InReg2:
            pass

        reg1._discovered = True
        reg2._discovered = True

        assert reg1.get("shared_name") is InReg1
        assert reg2.get("shared_name") is InReg2
        assert reg1.get("shared_name") is not reg2.get("shared_name")

    def test_class_import_lock_is_reentrant(self):
        assert isinstance(ClassRegistry._import_lock, type(threading.RLock()))


# ---------------------------------------------------------------------------
# Hardened edge cases
# ---------------------------------------------------------------------------
class TestClassRegistryHardenedEdgeCases:
    def test_register_empty_string_name(self):
        """Empty string as registry name should work."""
        reg = ClassRegistry("test")

        @reg.register("")
        class Empty:
            pass

        assert reg._registry[""] is Empty

    def test_register_with_special_characters(self):
        """Special characters in name should be stored as-is."""
        reg = ClassRegistry("test")

        @reg.register("my/tool:v2")
        class Versioned:
            pass

        assert reg._registry["my/tool:v2"] is Versioned

    def test_get_with_colon_in_registered_name_vs_dynamic_import(self):
        """A name with ':' that's already registered should return the registered class,
        not attempt dynamic import."""
        reg = ClassRegistry("test")

        @reg.register("my_module:MyClass")
        class Registered:
            pass

        reg._discovered = True
        result = reg.get("my_module:MyClass")
        assert result is Registered

    def test_register_overwrites_on_setitem_but_not_on_decorator(self):
        """register() is idempotent (keeps first), but __setitem__ overwrites."""
        reg = ClassRegistry("test")

        @reg.register("name")
        class First:
            pass

        @reg.register("name")
        class Second:
            pass

        # Decorator is idempotent — keeps First
        assert reg._registry["name"] is First

        # But setitem should overwrite
        reg._discovered = True
        reg["name"] = Second
        assert reg._registry["name"] is Second

    def test_import_from_file_with_import_error_in_module(self):
        """File that raises ImportError during exec should propagate cleanly."""
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write("import nonexistent_module_xyz_123\nclass Broken:\n    pass\n")
            f.flush()
            filepath = f.name
        try:
            reg = ClassRegistry("test")
            with pytest.raises((ValueError, ImportError, ModuleNotFoundError)):
                reg._import_from_file(filepath, "Broken")
            # Module should not remain in sys.modules after failure
            import hashlib

            h = hashlib.blake2b(str(filepath).encode(), digest_size=8).hexdigest()
            stem = os.path.basename(filepath).replace(".py", "")
            module_name = f"_axon_dynamic_{stem}_{h}"
            assert module_name not in sys.modules, "Failed module should be cleaned from sys.modules"
        finally:
            os.unlink(filepath)

    def test_list_returns_sorted_keys(self):
        reg = ClassRegistry("test")
        for name in ["zebra", "apple", "mango", "banana"]:
            reg.register(name)(type(name.title(), (), {}))
        reg._discovered = True
        assert reg.list() == ["apple", "banana", "mango", "zebra"]

    def test_empty_registry_operations(self):
        """All operations on empty registry should work."""
        reg = ClassRegistry("test")
        reg._discovered = True
        assert reg.list() == []
        assert reg.keys() == []
        assert list(reg.values()) == []
        assert list(reg.items()) == []
        assert "anything" not in reg
