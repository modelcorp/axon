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

"""Keep flashinfer bound to the real libcudart (tilelang stub collision workaround).

GDN models (Qwen3.5, Qwen3-Next, GLM) use ``tilelang``, which ships an INCOMPLETE
``libcudart_stub.so`` and loads it during kernel compilation/model build. flashinfer's
``comm/cuda_ipc.py`` resolves libcudart via ``find_loaded_library("libcudart")``, a
substring scan of ``/proc/self/maps`` that also matches ``libcudart_stub.so``. Once
tilelang has loaded its stub (lower VA), that scan returns the stub instead of the real
libcudart, and flashinfer's module-level ``cudart = CudaRTLibrary()`` (run at import,
during vLLM ``init_device``) crashes with ``undefined symbol: cudaDeviceReset``.

Importing flashinfer's ``cuda_ipc`` *before* tilelang loads its stub binds flashinfer's
module-level ``cudart`` to the real libcudart and caches it for the process, so the later
import during ``init_device`` reuses the good binding. We do this once at worker startup,
before any model (and therefore tilelang) is built.
"""

import logging

logger = logging.getLogger(__name__)

_PINNED = False


def ensure_flashinfer_real_libcudart() -> None:
    """Pre-bind flashinfer's libcudart to the real one, before tilelang's stub loads.

    Idempotent and best-effort: a no-op if flashinfer/libcudart are unavailable.
    """
    global _PINNED
    if _PINNED:
        return
    _PINNED = True
    try:
        # Importing this module runs ``cudart = CudaRTLibrary()`` which binds to the
        # real libcudart as long as tilelang's stub is not yet loaded. The binding is
        # cached on the module, so later imports (vLLM init_device) reuse it.
        import flashinfer.comm.cuda_ipc  # noqa: F401

        logger.debug("Pre-bound flashinfer cudart to the real libcudart.")
    except Exception as e:  # best-effort; never block worker startup
        logger.debug("flashinfer cudart pre-bind skipped: %s", e)
