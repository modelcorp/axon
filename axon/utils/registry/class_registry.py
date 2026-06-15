# Copyright 2025 Model AI Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# axon/utils/registry/class_registry.py
import importlib
import logging
import os
import threading
from pathlib import Path

logger = logging.getLogger(__name__)


class ClassRegistry:
    # Class-level reentrant lock for thread-safe dynamic imports
    # RLock allows same thread to acquire multiple times (needed for nested imports)
    _import_lock = threading.RLock()

    def __init__(self, name: str):
        self.name = name
        self._registry: dict[str, type] = {}
        self._discovered = False

    # ============ Decorator API ============
    def register(self, name: str):
        """Decorator to register a class.

        Thread-safe and idempotent - if the same name is registered multiple times
        (e.g., due to module reimport), returns the first registered class.
        """

        def decorator(cls):
            with ClassRegistry._import_lock:
                if name in self._registry:
                    # Idempotent: return existing class for consistency
                    # This handles module reimport scenarios where the same file
                    # is imported multiple times, creating new class objects
                    return self._registry[name]
                self._registry[name] = cls
            return cls

        return decorator

    # ============ Dict-like API ============
    def __getitem__(self, name: str) -> type:
        return self.get(name)

    def __contains__(self, name: str) -> bool:
        self._ensure_discovered()
        return name in self._registry

    def keys(self) -> list[str]:
        return self.list()

    def values(self):
        self._ensure_discovered()
        return self._registry.values()

    def items(self):
        self._ensure_discovered()
        return self._registry.items()

    def update(self, mapping: dict[str, type]):
        """Update registry with multiple entries (dict-like)."""
        self._ensure_discovered()
        self._registry.update(mapping)

    def __setitem__(self, name: str, cls: type):
        """Allow registry[name] = cls syntax."""
        self._ensure_discovered()
        self._registry[name] = cls

    # ============ Core API ============
    def get(self, name: str) -> type:
        """Get a class by name, module path, or file path."""
        self._ensure_discovered()

        if name in self._registry:
            return self._registry[name]

        # Dynamic import: module path or file path
        if ":" in name:
            try:
                cls = self._dynamic_import(name)
                self._registry[name] = cls
                return cls
            except (ImportError, AttributeError) as e:
                raise ValueError(f"Failed to import {self.name} '{name}': {e}") from e

        raise ValueError(
            f"Unknown {self.name} '{name}'.\n"
            f"Available: {sorted(self._registry.keys())}\n"
            f"For custom classes, use:\n"
            f"  - Module: 'my_package.module:ClassName'\n"
            f"  - File:   '/path/to/file.py:ClassName'"
        )

    def list(self) -> list[str]:
        self._ensure_discovered()
        return sorted(self._registry.keys())

    # ============ Discovery ============
    def _ensure_discovered(self):
        if self._discovered:
            return
        self._discovered = True
        # 1. Build-in classes are registered via decorators
        # 2. User plugin directory (~/.axon/agents/, ~/.axon/envs/, etc.)
        user_dir = Path.home() / ".axon" / f"{self.name}s"
        if user_dir.exists():
            self._discover_directory(user_dir)

        # 3. Optional Environment variable plugins
        # Set the environment variable to a comma-separated list of Python module paths
        # On first registry access, Axon imports each module
        # Any @register_* decorators in those modules are executed
        # Registered classes become available by their short names
        plugins = os.environ.get(f"AXON_{self.name.upper()}_PLUGINS", "")
        for module in plugins.split(","):
            if module.strip():
                self._try_import(module.strip())

    def _discover_directory(self, directory: Path):
        import sys

        if str(directory) not in sys.path:
            sys.path.insert(0, str(directory))
        for f in directory.rglob("*.py"):
            if f.name.startswith("_"):
                continue
            module_name = str(f.relative_to(directory).with_suffix("")).replace("/", ".")
            self._try_import(module_name)

    def _try_import(self, module_path: str):
        try:
            importlib.import_module(module_path)
        except Exception as e:
            logger.warning("Failed import %s: %s", module_path, e)
            pass

    def _dynamic_import(self, path: str) -> type:
        """
        Import from:
        - Module path: 'my_package.module:ClassName'
        - File path:   '/path/to/file.py:ClassName'
        """
        if ":" in path:
            source, class_name = path.rsplit(":", 1)
        else:
            source, class_name = path.rsplit(".", 1)

        # Check if it's a file path
        if source.endswith(".py") or "/" in source or "\\" in source:
            return self._import_from_file(source, class_name)

        # Module path
        module = importlib.import_module(source)
        return getattr(module, class_name)

    def _import_from_file(self, filepath: str, class_name: str) -> type:
        """Import a class from a .py file path."""
        import hashlib
        import sys
        from importlib.util import module_from_spec, spec_from_file_location

        filepath = Path(filepath).expanduser().resolve()
        if not filepath.exists():
            raise ImportError(f"File not found: {filepath}")

        parent = str(filepath.parent)
        if parent not in sys.path:
            sys.path.insert(0, parent)

        # Create a unique module name
        h = hashlib.blake2b(str(filepath).encode(), digest_size=8).hexdigest()
        module_name = f"_axon_dynamic_{filepath.stem}_{h}"

        # Thread-safe module loading to prevent duplicate registration
        with ClassRegistry._import_lock:
            # Check if module is already loaded and valid
            if module_name in sys.modules:
                module = sys.modules[module_name]
                # Verify the module has the requested class (wasn't a failed partial load)
                if hasattr(module, class_name):
                    return getattr(module, class_name)
                # Module exists but is incomplete - remove it and reimport
                del sys.modules[module_name]

            spec = spec_from_file_location(module_name, filepath)
            if spec is None or spec.loader is None:
                raise ImportError(f"Cannot load spec for {filepath}")

            module = module_from_spec(spec)
            sys.modules[module_name] = module
            try:
                spec.loader.exec_module(module)
            except Exception:
                # Remove incomplete module from sys.modules on failure
                sys.modules.pop(module_name, None)
                raise

        return getattr(module, class_name)
