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
Axon Configuration Module.

This module provides unified configuration management for Axon training,
including PPO trainer configuration with support for both FSDP and Megatron strategies.

Example usage:
    from axon.config import validate_axon_config

    # Load configuration (agent configs are self-contained)
    config = load_ppo_config("axon/config/config.yaml")

    # Validate configuration
    validate_axon_config(config)

    # Check configuration properties
    from axon.config import needs_critic, needs_reference_policy
    if needs_critic(config):
        print("Critic is required")
"""

from pathlib import Path

from .validate_config import (
    validate_axon_config,
)

DEFAULT_AGENT_PPO_CONFIG_PATH = Path(__file__).parent / "config.yaml"

__all__ = [
    # Configuration loading and validation
    "validate_axon_config",
    "DEFAULT_AGENT_PPO_CONFIG_PATH",
]
