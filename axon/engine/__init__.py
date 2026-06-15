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
"""Engine module for Axon.

This module contains the core execution infrastructure for agent program sampler.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .engine import Engine

__all__ = ["Engine"]


def __getattr__(name: str):
    if name == "Engine":
        from .engine import Engine

        return Engine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
