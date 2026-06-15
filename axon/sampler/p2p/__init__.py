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
"""Sampler P2P hook registry.

A model whose parameter layout doesn't match the sampler mixin's generic
enumeration registers per-model callables (``override_param`` and/or
``extra_buffers``) here. The mixin looks them up by the vLLM model's
class name via ``get_hooks(vllm_model)``. Hook files are imported at
package load for side-effect registration; each one must be
self-contained and import only vLLM-safe dependencies because the
sampler runs in a Ray worker that doesn't have Megatron / mbridge
available.
"""

from collections.abc import Callable

# vLLM class name → {"override_param": fn, "extra_buffers": fn} (both optional).
_HOOKS: dict[str, dict[str, Callable]] = {}


def _register(class_names, **hooks):
    for name in class_names:
        _HOOKS[name] = hooks


def get_hooks(vllm_model) -> dict[str, Callable] | None:
    """Return the registered hooks for ``vllm_model``, or None.

    vLLM v1 wraps the real model in a ``CUDAGraphWrapper`` (or similar)
    whose MRO doesn't include the underlying class. The wrapper stores
    the wrapped model on ``.runnable`` (or ``.model``) and delegates
    attribute access via ``__getattr__``, so we walk that chain here to
    detect the real class. ``named_modules`` / ``named_parameters`` are
    unaffected because they also flow through the same ``__getattr__``.
    """
    seen = set()
    candidate = vllm_model
    while candidate is not None and id(candidate) not in seen:
        seen.add(id(candidate))
        for cls in type(candidate).__mro__:
            hit = _HOOKS.get(cls.__name__)
            if hit is not None:
                return hit
        # Common wrapper field names: ``runnable`` (CUDAGraphWrapper),
        # ``model`` (generic nn.Module wrappers).
        candidate = getattr(candidate, "runnable", None) or getattr(candidate, "model", None)
    return None


# Trigger built-in hook registrations.
from . import gemma4 as _gemma4  # noqa: E402, F401
