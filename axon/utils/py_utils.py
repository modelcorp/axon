# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""
Contain small python utility functions
"""

import hashlib
import importlib
import os
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from functools import wraps
from typing import Any, Optional


def hash_string_to_int(s: str) -> int:
    """Convert string to a consistent integer hash across processes."""
    full_hash = int(hashlib.md5(s.encode(), usedforsecurity=False).hexdigest(), 16)
    return full_hash % (2**31 - 1)  # Keep it within 31-bit positive range


def union_two_dict(dict1: dict, dict2: dict):
    """Union two dict. Will throw an error if there is an item not the same object with the same key.

    Args:
        dict1:
        dict2:

    Returns:

    """
    for key, val in dict2.items():
        if key in dict1:
            v1, v2 = dict1[key], dict2[key]
            # NaN != NaN in Python, so handle float NaN explicitly
            if isinstance(v1, float) and isinstance(v2, float):
                import math

                if math.isnan(v1) and math.isnan(v2):
                    continue
            assert v2 == v1, f"{key} in meta_dict1 and meta_dict2 are not the same object"
        dict1[key] = val

    return dict1


class DynamicEnumMeta(type):
    def __iter__(cls) -> Iterator[Any]:
        return iter(cls._registry.values())

    def __contains__(cls, item: Any) -> bool:
        # allow `name in EnumClass` or `member in EnumClass`
        if isinstance(item, str):
            return item in cls._registry
        return item in cls._registry.values()

    def __getitem__(cls, name: str) -> Any:
        return cls._registry[name]

    def __reduce_ex__(cls, protocol):
        # Always load the existing module and grab the class
        return getattr, (importlib.import_module(cls.__module__), cls.__name__)

    def names(cls):
        return list(cls._registry.keys())

    def values(cls):
        return list(cls._registry.values())


class DynamicEnum(metaclass=DynamicEnumMeta):
    _registry: dict[str, "DynamicEnum"] = {}
    _next_value: int = 0

    def __init__(self, name: str, value: int):
        self.name = name
        self.value = value

    def __repr__(self):
        return f"<{self.__class__.__name__}.{self.name}: {self.value}>"

    def __reduce_ex__(self, protocol):
        """
        Unpickle via: getattr(import_module(module).Dispatch, 'ONE_TO_ALL')
        so the existing class is reused instead of re-executed.
        """
        module = importlib.import_module(self.__class__.__module__)
        enum_cls = getattr(module, self.__class__.__name__)
        return getattr, (enum_cls, self.name)

    @classmethod
    def register(cls, name: str) -> "DynamicEnum":
        key = name.upper()
        if key in cls._registry:
            raise ValueError(f"{key} already registered")
        member = cls(key, cls._next_value)
        cls._registry[key] = member
        setattr(cls, key, member)
        cls._next_value += 1
        return member

    @classmethod
    def remove(cls, name: str):
        key = name.upper()
        member = cls._registry.pop(key)
        delattr(cls, key)
        return member

    @classmethod
    def from_name(cls, name: str) -> Optional["DynamicEnum"]:
        return cls._registry.get(name.upper())


@contextmanager
def temp_env_var(key: str, value: str):
    """Context manager for temporarily setting an environment variable.

    This context manager ensures that environment variables are properly set and restored,
    even if an exception occurs during the execution of the code block.

    Args:
        key: Environment variable name to set
        value: Value to set the environment variable to

    Yields:
        None

    Example:
        >>> with temp_env_var("MY_VAR", "test_value"):
        ...     # MY_VAR is set to "test_value"
        ...     do_something()
        ... # MY_VAR is restored to its original value or removed if it didn't exist
    """
    original = os.environ.get(key)
    os.environ[key] = value
    try:
        yield
    finally:
        if original is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = original


def convert_to_regular_types(obj):
    """Convert Hydra configs and other special types to regular Python types."""
    from omegaconf import DictConfig, ListConfig

    if isinstance(obj, ListConfig | DictConfig):
        return {k: convert_to_regular_types(v) for k, v in obj.items()} if isinstance(obj, DictConfig) else list(obj)
    elif isinstance(obj, tuple):
        return tuple(convert_to_regular_types(x) for x in obj)
    elif isinstance(obj, list):
        return [convert_to_regular_types(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: convert_to_regular_types(v) for k, v in obj.items()}
    return obj


def _get_qualified_name(func):
    """Get full qualified name including module and class (if any)."""
    module = func.__module__
    qualname = func.__qualname__
    return f"{module}.{qualname}"


def deprecated(replacement: str = ""):
    """Decorator to mark functions or classes as deprecated."""

    def decorator(obj):
        qualified_name = _get_qualified_name(obj)

        if isinstance(obj, type):
            original_init = obj.__init__

            @wraps(original_init)
            def wrapped_init(self, *args, **kwargs):
                msg = f"Warning: Class '{qualified_name}' is deprecated."
                if replacement:
                    msg += f" Please use '{replacement}' instead."
                warnings.warn(msg, category=FutureWarning, stacklevel=2)
                return original_init(self, *args, **kwargs)

            obj.__init__ = wrapped_init
            return obj

        else:

            @wraps(obj)
            def wrapped(*args, **kwargs):
                msg = f"Warning: Function '{qualified_name}' is deprecated."
                if replacement:
                    msg += f" Please use '{replacement}' instead."
                warnings.warn(msg, category=FutureWarning, stacklevel=2)
                return obj(*args, **kwargs)

            return wrapped

    return decorator
