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
Advantage estimation module for reinforcement learning algorithms.
"""

# Registry
from enum import Enum

import torch
from omegaconf import DictConfig

from axon.protocol import DataProto

# Import advantage functions to register them
from axon.trainer.algos.advantages import advantage as _advantage  # noqa: F401
from axon.trainer.algos.advantages.registry import (
    AdvantageFn,
    AdvantageRegistryEntry,
    _advantage_registry,
    get_advantage_entry,
    get_advantage_fn,
    register_advantage,
)


def compute_advantage(
    data: DataProto,
    config: DictConfig = None,
) -> DataProto:
    """Compute advantage estimates using the registered advantage function.

    Looks up the registered advantage function by name from config.advantage,
    then calls it with (data, config).

    Args:
        data: DataProto containing batch data with keys like:
            - token_level_rewards: (batch_size, response_length)
            - response_mask: (batch_size, response_length)
            - values: (batch_size, response_length) [optional, for GAE]
            - uid: Group IDs in non_tensor_batch [optional, for group-based]
            - reward_baselines: (batch_size,) [optional, for ReMax]
        config: Config object with:
            - advantage: str name of the advantage estimator
            - advantage_args: dict of config overrides (optional)

    Returns:
        DataProto: The input data with 'advantages' and 'returns' added to batch.
    """
    # Get estimator name from config
    adv_estimator = config.advantage
    if adv_estimator is None:
        raise ValueError("Config must have 'advantage' field specifying the estimator name")

    name = adv_estimator.value if isinstance(adv_estimator, Enum) else adv_estimator

    # Pass advantage_args directly as the config to the function
    adv_config = getattr(config, "advantage_args", {}) or {}
    advantages, returns = _advantage_registry.compute(name, data, adv_config)

    # Set results on batch
    if adv_config.get("clip_advantages", False):
        advantages = torch.clamp(advantages, -1, 1)

    data.batch["advantages"] = advantages
    data.batch["returns"] = returns

    return data


__all__ = [
    # Registry
    "AdvantageFn",
    "AdvantageRegistryEntry",
    "register_advantage",
    "get_advantage_entry",
    "get_advantage_fn",
    "compute_advantage",
]
