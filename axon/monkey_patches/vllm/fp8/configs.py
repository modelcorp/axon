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
#
# Config layer adapted from FlashRL (github.com/LLM360/Flash-RL), Apache-2.0; Int4 / MxFP8 configs are Axon additions.
from dataclasses import dataclass, field


@dataclass
class FP8TensorConfig:
    fn: str = "fp8_tensor"
    load_format: str = "dummy"
    distributed_executor_backend: str = "external_launcher"
    module_attribute_to_preserve: list[str] = field(default_factory=lambda: ["workspace"])


@dataclass
class FP8ChannelConfig:
    fn: str = "fp8_channel"
    load_format: str = "dummy"
    distributed_executor_backend: str = "external_launcher"
    module_attribute_to_preserve: list[str] = field(default_factory=lambda: ["workspace"])


@dataclass
class FP8vLLMConfig:
    fn: str = "fp8_vllm"
    load_format: str = "auto"
    distributed_executor_backend: str = "external_launcher"
    module_attribute_to_preserve: list[str] = field(default_factory=lambda: ["workspace"])
    quantization: str = "fp8"


@dataclass
class FP8vLLMFastConfig:
    fn: str = "fp8_vllm_fast"
    load_format: str = "auto"
    distributed_executor_backend: str = "external_launcher"
    module_attribute_to_preserve: list[str] = field(default_factory=lambda: ["workspace"])
    quantization: str = "fp8"


@dataclass
class BF16Config:
    fn: str = "bf16"
    load_format: str = "dummy"
    distributed_executor_backend: str = "external_launcher"


@dataclass
class Int8FastConfig:
    fn: str = "int8_fast"
    load_format: str = "auto"
    distributed_executor_backend: str = "external_launcher"


@dataclass
class Int8Config:
    fn: str = "int8"
    load_format: str = "auto"
    distributed_executor_backend: str = "external_launcher"


@dataclass
class Int8PruneConfig:
    fn: str = "int8_prune"
    load_format: str = "auto"
    distributed_executor_backend: str = "external_launcher"


@dataclass
class Int4Config:
    """INT4 compressed-tensors quantization config for vLLM."""

    fn: str = "int4"
    load_format: str = "auto"
    distributed_executor_backend: str = "external_launcher"
    quantization: str = "compressed-tensors"
    group_size: int = 128
    symmetric: bool = True


@dataclass
class MxFP8Config:
    """MxFP8 (mixed-format FP8 with UE8M0 scales) config for vLLM."""

    fn: str = "mxfp8"
    load_format: str = "auto"
    distributed_executor_backend: str = "external_launcher"
    quantization: str = "fp8"
    weight_block_size: list = field(default_factory=lambda: [1, 32])
    scale_fmt: str = "ue8m0"


def get_default_config(fn):
    return {
        "fp8": FP8vLLMConfig(),
        "fp8_vllm": FP8vLLMConfig(),
        "fp8_vllm_fast": FP8vLLMFastConfig(),
        "fp8_fast": FP8vLLMFastConfig(),
        "fp8_channel": FP8ChannelConfig(),
        "fp8_tensor": FP8TensorConfig(),
        "int4": Int4Config(),
        "int8": Int8Config(),
        "int8_fast": Int8FastConfig(),
        "int8_wo_prune": Int8Config(),
        "int8_prune": Int8PruneConfig(),
        "mxfp8": MxFP8Config(),
        "bf16": BF16Config(),
    }[fn]
