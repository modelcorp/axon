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
MBridge model bridge patches for custom model architectures.

These patches override and extend the default mbridge model implementations
with custom configurations and weight loading logic.
"""

# Gemma4 requires transformers with gemma4 support (google/gemma-4-*); skip if unavailable.
try:
    from .gemma4 import Gemma4Bridge
except ImportError as _gemma4_err:
    print(f"[mbridge] Gemma4Bridge unavailable: {_gemma4_err}. Install a transformers version with gemma4 support to enable it.")
    Gemma4Bridge = None

from .glm4 import GLM4Bridge
from .glm4_moe_lite import GLM4MoELiteBridge
from .glm5 import GLM5Bridge
from .gpt_oss import GPTOSSBridge
from .qwen3_5 import Qwen3_5Bridge, Qwen3_5MoEBridge
from .qwen3_next import Qwen3NextBridge

# Re-exports from the mbridge pip package
try:
    from mbridge import AutoBridge
    from mbridge.utils.post_creation_callbacks import freeze_moe_router, make_value_model
except ImportError:
    print("mbridge package not found. Please install mbridge with `pip install mbridge`")
    raise

__all__ = [
    "Gemma4Bridge",
    "GLM4Bridge",
    "GLM4MoELiteBridge",
    "GLM5Bridge",
    "Qwen3NextBridge",
    "Qwen3_5Bridge",
    "Qwen3_5MoEBridge",
    "GPTOSSBridge",
    "AutoBridge",
    "make_value_model",
    "freeze_moe_router",
]
