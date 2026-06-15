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
Megatron-Bridge package integration.

Wraps the `megatron-bridge` pip package, providing AutoBridge, LoRA adapters,
and model utility functions (make_value_model, freeze_moe_router).
"""

from .bridge import (
    AutoBridge,
    CanonicalLoRA,
    DoRA,
    LinearForLastLayer,
    LoRA,
    VLMLoRA,
    freeze_moe_router,
    make_value_model,
)

__all__ = [
    "AutoBridge",
    "make_value_model",
    "freeze_moe_router",
    "LoRA",
    "VLMLoRA",
    "DoRA",
    "CanonicalLoRA",
    "LinearForLastLayer",
]
