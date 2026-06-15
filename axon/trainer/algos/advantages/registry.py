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
Advantage estimator registry for reinforcement learning algorithms.

This module provides a registry system for advantage estimation functions
using the unified FunctionRegistry infrastructure.

All registered functions have the uniform signature:
    fn(data: DataProto, config: DictConfig) -> tuple[torch.Tensor, torch.Tensor]
"""

from enum import Enum
from typing import Any

import torch

from axon.utils.registry import FunctionRegistry, FunctionRegistryEntry

__all__ = [
    "AdvantageFn",
    "AdvantageRegistryEntry",
    "ADV_REGISTRY",
    "register_advantage",
    "get_advantage_entry",
    "get_advantage_fn",
]

# Re-export FunctionRegistryEntry with advantage-specific name for backwards compatibility
AdvantageRegistryEntry = FunctionRegistryEntry


class AdvantageFn(str, Enum):
    """Enumeration of available advantage estimators for policy optimization."""

    GAE = "gae"
    GRPO = "grpo"
    RLOO = "rloo"
    LOOP = "loop"  # alias for RLOO (same algorithm)
    REINFORCE_PLUS_PLUS = "reinforce_plus_plus"
    REINFORCE_PLUS_PLUS_BASELINE = "reinforce_plus_plus_baseline"
    REMAX = "remax"
    OPO = "opo"
    GRPO_PASSK = "grpo_passk"
    GPG = "gpg"
    CHUNKED_GAE = "chunked_gae"
    IDENTITY = "identity"
    KIMI_K1_5 = "kimi_k1_5"


# Create the advantage registry
_advantage_registry = FunctionRegistry[tuple[torch.Tensor, torch.Tensor]](
    name="advantage estimator",
)

# Expose internal registry dict for backwards compatibility (used in tests)
ADV_REGISTRY = _advantage_registry._registry


# Public API functions that delegate to the registry
def register_advantage(name_or_enum: str | AdvantageFn):
    """Register an advantage estimator function.

    Registered functions must have the signature:
        fn(data: DataProto, config: DictConfig) -> tuple[torch.Tensor, torch.Tensor]
    """
    return _advantage_registry.register(name_or_enum)


def get_advantage_entry(name_or_enum: str | AdvantageFn) -> AdvantageRegistryEntry:
    """Get the full registry entry for an advantage estimator."""
    return _advantage_registry.get_entry(name_or_enum)


def get_advantage_fn(name_or_enum: str | AdvantageFn) -> Any:
    """Get the advantage estimator function."""
    return _advantage_registry.get_fn(name_or_enum)
