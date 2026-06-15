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
import importlib

from axon.sampler.base.engine import Engine
from axon.sampler.base.server import SERVER_REGISTRY, Server

__all__ = ["Engine", "SERVER_REGISTRY", "get_engine_class", "get_server_class"]

_ENGINE_REGISTRY = {
    "vllm": "axon.sampler.vllm.engine.vLLMEngine",
    "sglang": "axon.sampler.sglang.engine.SGLangEngine",
}

# Modules that contain @register_server decorators for built-in backends.
# These must be imported before SERVER_REGISTRY is queried by short name.
_SERVER_MODULES = {
    "vllm": "axon.sampler.vllm.server",
    "sglang": "axon.sampler.sglang.server",
}


def get_engine_class(sampler_name: str) -> type[Engine]:
    """Get the sampler class by name.

    Args:
        sampler_name: The name of the sampler.

    Returns:
        The sampler class.
    """
    assert sampler_name in _ENGINE_REGISTRY, f"Sampler {sampler_name} not found"
    fqdn = _ENGINE_REGISTRY[sampler_name]
    module_name, class_name = fqdn.rsplit(".", 1)
    sampler_module = importlib.import_module(module_name)
    return getattr(sampler_module, class_name)


def get_server_class(sampler: str) -> type[Server]:
    """Get the server class by name.

    Args:
        sampler: The name of the sampler backend (e.g. "vllm", "sglang").

    Returns:
        The server class.
    """
    if sampler in _SERVER_MODULES:
        importlib.import_module(_SERVER_MODULES[sampler])
    return SERVER_REGISTRY.get(sampler)
