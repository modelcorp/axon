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
# MemPoolContext fallback adapted from FlashRL (github.com/LLM360/Flash-RL), Apache-2.0.
from contextlib import contextmanager

import torch
from vllm.device_allocator.cumem import CuMemAllocator

# Prefer torch 2.8+ MemPool API; fall back to older MemPoolContext if available
try:
    _HAVE_NEW_MEMPOOL_API = True
except Exception:
    _HAVE_NEW_MEMPOOL_API = False
    try:
        from torch._C import (
            _cuda_beginAllocateToPool,  # type: ignore
            _cuda_endAllocateCurrentStreamToPool,  # type: ignore
        )
        from torch.cuda.memory import MemPoolContext  # type: ignore
    except Exception:
        MemPoolContext = None  # type: ignore
        _cuda_beginAllocateToPool = None  # type: ignore
        _cuda_endAllocateCurrentStreamToPool = None  # type: ignore


@contextmanager
def disable_mem_pool(disable=False):
    """Temporarily route allocations away from vLLM's weights pool.

    On torch >= 2.8, uses torch.cuda.memory.MemPool/use_mem_pool to override
    the current thread's pool with a temporary pool. On older versions, falls
    back to MemPoolContext if available. Otherwise, no-ops.
    """
    if not disable:
        yield
        return

    if _HAVE_NEW_MEMPOOL_API:
        # On torch >= 2.8, avoid interfering with vLLM's managed pools.
        # No-op to maintain allocator semantics and prevent potential stalls.
        yield
        return

    # Fallback for older torch versions that still have MemPoolContext
    if MemPoolContext is None:
        # No available API to switch pools; best-effort no-op
        yield
        return

    allocator = CuMemAllocator.get_instance()
    need_restart = False
    try:
        if (
            "weights" in allocator.allocator_and_pools
            and hasattr(MemPoolContext, "active_pool")
            and MemPoolContext.active_pool() == allocator.allocator_and_pools["weights"][0]
        ):
            pool = MemPoolContext.active_pool()
            ctx = MemPoolContext(None)
            device_index = torch.cuda.current_device()
            _cuda_endAllocateCurrentStreamToPool(device_index, pool.id)
            need_restart = True
    except Exception:
        need_restart = False

    try:
        yield
    finally:
        if need_restart:
            _cuda_beginAllocateToPool(device_index, pool.id)
            del ctx
