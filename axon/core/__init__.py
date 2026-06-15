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
from typing import TYPE_CHECKING

from axon.core.class_init import ClassWithInitArgs
from axon.core.resource_pool import ResourcePool

if TYPE_CHECKING:
    from axon.core.agent import AGENT_CLASS_MAPPING, Action, BaseAgent, register_agent
    from axon.core.env import ENV_CLASS_MAPPING, BaseEnv, MultiTurnEnvironment, SingleTurnEnvironment, register_env

__all__ = [
    "Action",
    "AGENT_CLASS_MAPPING",
    "BaseAgent",
    "BaseEnv",
    "ClassWithInitArgs",
    "ENV_CLASS_MAPPING",
    "MultiTurnEnvironment",
    "ResourcePool",
    "SingleTurnEnvironment",
    "register_agent",
    "register_env",
]

_LAZY_EXPORTS = {
    "AGENT_CLASS_MAPPING": ("axon.core.agent", "AGENT_CLASS_MAPPING"),
    "Action": ("axon.core.agent", "Action"),
    "BaseAgent": ("axon.core.agent", "BaseAgent"),
    "register_agent": ("axon.core.agent", "register_agent"),
    "ENV_CLASS_MAPPING": ("axon.core.env", "ENV_CLASS_MAPPING"),
    "BaseEnv": ("axon.core.env", "BaseEnv"),
    "MultiTurnEnvironment": ("axon.core.env", "MultiTurnEnvironment"),
    "SingleTurnEnvironment": ("axon.core.env", "SingleTurnEnvironment"),
    "register_env": ("axon.core.env", "register_env"),
}


def __getattr__(name: str):
    if name in _LAZY_EXPORTS:
        module_name, attr_name = _LAZY_EXPORTS[name]
        module = __import__(module_name, fromlist=[attr_name])
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
