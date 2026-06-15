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
Policy loss registry for reinforcement learning algorithms.

This module provides a registry system for policy loss functions
using the unified FunctionRegistry infrastructure.

All registered functions have the uniform signature:
    fn(data: DataProto, config: DictConfig) -> tuple[torch.Tensor, dict[str, Any]]
"""

from enum import Enum
from typing import Any

import torch

from axon.protocol import DataProto
from axon.utils.registry import FunctionRegistry, FunctionRegistryEntry

# Re-export FunctionRegistryEntry with loss-specific name for backwards compatibility
LossRegistryEntry = FunctionRegistryEntry


class LossFn(str, Enum):
    """Enumeration of available policy loss functions for policy optimization."""

    PPO = "ppo"
    GSPO = "gspo"
    GPG = "gpg"
    CLIP_COV = "clip_cov"
    KL_COV = "kl_cov"
    GEO_MEAN = "geo_mean"
    CISPO = "cispo"
    REINFORCE = "reinforce"
    VALUE = "value"


# Create the loss registry
_loss_registry = FunctionRegistry[tuple[torch.Tensor, dict[str, Any]]](
    name="policy loss",
)


# Public API functions that delegate to the registry
def register_loss(name_or_enum: str | LossFn):
    """Register a policy loss function.

    Registered functions must have the signature:
        fn(data: DataProto, config: DictConfig) -> tuple[torch.Tensor, dict[str, Any]]
    """
    return _loss_registry.register(name_or_enum)


def get_loss_entry(name_or_enum: str | LossFn) -> LossRegistryEntry:
    """Get the full registry entry for a policy loss function."""
    return _loss_registry.get_entry(name_or_enum)


def get_loss_fn(name_or_enum: str | LossFn) -> Any:
    """Get the policy loss function."""
    return _loss_registry.get_fn(name_or_enum)


def compute_loss_fn(
    data: DataProto,
    loss_fn: str,
    loss_fn_args: dict | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute policy loss using a registered loss function.

    Args:
        data: DataProto containing batch data.
        loss_fn: Name of the loss function (e.g., "ppo", "gspo", "gpg")
        loss_fn_args: Dict of config overrides for the loss function
            (e.g., clip_ratio, token_reduce, batch_reduce, sampler_is, sampler_rs, etc.)

    Returns:
        tuple[torch.Tensor, dict[str, Any]]: (loss, metrics) where loss is a scalar
            tensor and metrics is a flat dictionary of training metrics (no prefix).
    """
    if loss_fn_args is None:
        loss_fn_args = {}
    fn = _loss_registry.get_fn(loss_fn)

    # Apply sampler correction (IS weights + rejection sampling) if configured.
    # Checks if any sampler correction mechanism is enabled in loss_fn_args.
    sampler_corr_metrics = {}
    sampler_is = loss_fn_args.get("sampler_is", None)
    sampler_rs = loss_fn_args.get("sampler_rs", None)
    sampler_token_veto = loss_fn_args.get("sampler_token_veto_threshold", None)
    if not all(v is None for v in [sampler_is, sampler_rs, sampler_token_veto]) and "sampler_log_probs" in data.batch:
        from axon.utils.rl.sampler import compute_sampler_correction_and_add_to_batch

        data, sampler_corr_metrics = compute_sampler_correction_and_add_to_batch(data, loss_fn_args)

    loss, loss_metrics = fn(data, loss_fn_args)
    loss_metrics.update(sampler_corr_metrics)

    return loss, loss_metrics
