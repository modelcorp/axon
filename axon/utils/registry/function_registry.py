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
"""
Unified algorithm registry for reinforcement learning algorithms.

This module provides a generic registry system for algorithm functions
with a uniform (data: DataProto, config: DictConfig) interface.

Each registered function is responsible for extracting its own data
and config arguments from the DataProto and DictConfig objects.

Usage:
    # Create a registry for a specific function type
    _my_registry = FunctionRegistry[ReturnType](name="my function type")

    # Register functions
    @_my_registry.register("my_fn")
    def my_fn(data: DataProto, config: DictConfig) -> ReturnType:
        x = data.batch["x"]
        lr = config.my_args.get("lr", 0.01)
        ...
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Generic, TypeVar

from omegaconf import DictConfig

from axon.protocol import DataProto

__all__ = [
    "FunctionRegistryEntry",
    "FunctionRegistry",
]

# Generic return type for registry functions
T = TypeVar("T")


@dataclass
class FunctionRegistryEntry:
    """Registry entry for an algorithm function."""

    fn: Callable


class FunctionRegistry(Generic[T]):
    """Generic function registry for algorithm functions.

    Registered functions must have the uniform signature:
        fn(data: DataProto, config: DictConfig) -> T

    Each function is responsible for extracting its own arguments
    from data and config.

    Type parameter T is the return type of registered functions.

    Example:
        # For advantage estimators returning (advantages, returns)
        adv_registry = FunctionRegistry[tuple[torch.Tensor, torch.Tensor]](
            name="advantage estimator",
        )

        @adv_registry.register("gae")
        def gae_fn(data: DataProto, config: DictConfig) -> tuple[Tensor, Tensor]:
            rewards = data.batch["token_level_rewards"]
            gamma = config.advantage_args.get("gamma", 0.99)
            ...
    """

    def __init__(self, name: str):
        """Initialize the registry.

        Args:
            name: Human-readable name for error messages (e.g., "advantage estimator")
        """
        self.name = name
        self._registry: dict[str, FunctionRegistryEntry] = {}

    def register(
        self,
        name_or_enum: str | Enum,
    ) -> Callable[[Callable[..., T]], Callable[..., T]]:
        """Register a function.

        Args:
            name_or_enum: The name (str) or enum value to register under.

        Returns:
            Decorator function
        """

        def decorator(fn: Callable[..., T]) -> Callable[..., T]:
            reg_name = name_or_enum.value if isinstance(name_or_enum, Enum) else name_or_enum

            # Check for conflicts
            if reg_name in self._registry and self._registry[reg_name].fn != fn:
                raise ValueError(f"{self.name.title()} '{reg_name}' already registered with different function")

            self._registry[reg_name] = FunctionRegistryEntry(fn=fn)

            return fn

        return decorator

    def get_entry(self, name_or_enum: str | Enum) -> FunctionRegistryEntry:
        """Get the full registry entry."""
        name = name_or_enum.value if isinstance(name_or_enum, Enum) else name_or_enum
        if name not in self._registry:
            raise ValueError(f"Unknown {self.name}: '{name}'. Available: {list(self._registry.keys())}")
        return self._registry[name]

    def get_fn(self, name_or_enum: str | Enum) -> Callable[..., T]:
        """Get the registered function."""
        return self.get_entry(name_or_enum).fn

    def compute(
        self,
        name_or_enum: str | Enum,
        data: DataProto,
        config: DictConfig,
    ) -> T:
        """Compute using the registered function.

        Args:
            name_or_enum: Name or enum of the registered function
            data: DataProto containing batch data
            config: DictConfig containing configuration

        Returns:
            The result of calling the registered function
        """
        entry = self.get_entry(name_or_enum)
        return entry.fn(data, config)

    def keys(self) -> list[str]:
        """Return all registered names."""
        return list(self._registry.keys())

    def __contains__(self, name_or_enum: str | Enum) -> bool:
        """Check if a name is registered."""
        name = name_or_enum.value if isinstance(name_or_enum, Enum) else name_or_enum
        return name in self._registry
