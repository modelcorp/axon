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
Fix for NCCL corruption in batch_isend_irecv with custom process groups.

The issue: _coalescing_manager manipulates pg_coalesce_state which is also
used by collective operations, causing state corruption.

The solution: Use NCCL grouping directly without touching pg_coalesce_state.
"""

import torch
import torch.distributed as dist
from torch.distributed.distributed_c10d import (
    P2POp,
    _check_p2p_op_list,
    _get_default_group,
)

_ORIGINAL_BATCH_ISEND_IRECV = None


def apply_batch_isend_irecv_patch():
    """
    Apply the patched batch_isend_irecv to torch.distributed.
    This should be called once during initialization.
    """
    global _ORIGINAL_BATCH_ISEND_IRECV
    if _ORIGINAL_BATCH_ISEND_IRECV is None:
        import torch.distributed as dist

        _ORIGINAL_BATCH_ISEND_IRECV = dist.batch_isend_irecv
        dist.batch_isend_irecv = patched_batch_isend_irecv
        print("[P2P Fix] Successfully patched torch.distributed.batch_isend_irecv")


def patched_batch_isend_irecv(p2p_op_list):
    """
    Fixed batch_isend_irecv that uses NCCL grouping without state corruption.

    Key fix: We directly use NCCL's ncclGroupStart/End without going through
    _coalescing_manager, which prevents interference with collective operations.
    """
    _check_p2p_op_list(p2p_op_list)
    group = p2p_op_list[0].group
    if group is None:
        group = _get_default_group()
    device = p2p_op_list[0].tensor.device

    def peer_kwarg(op: P2POp) -> dict:
        key = "group_dst" if op.op == dist.isend else "group_src"
        return {key: op.group_peer}

    # Reuse a cached stream per device to avoid leaking CUDA stream objects.
    # Each CUDA stream allocates device memory for internal state; creating
    # a new one per call accumulates ~67 MB/step of unreclaimable device memory.
    if not hasattr(patched_batch_isend_irecv, "_stream_cache"):
        patched_batch_isend_irecv._stream_cache = {}
    cache_key = device
    stream = patched_batch_isend_irecv._stream_cache.get(cache_key)
    if stream is None and device.type == "cuda":
        stream = torch.cuda.Stream(device=device)
        patched_batch_isend_irecv._stream_cache[cache_key] = stream

    reqs = []

    if device.type == "cuda" and stream is not None:
        default_stream = torch.cuda.current_stream(device)
        stream.wait_stream(default_stream)
        with torch.cuda.stream(stream):
            for p2p_op in p2p_op_list:
                work = p2p_op.op(
                    p2p_op.tensor,
                    group=p2p_op.group,
                    tag=p2p_op.tag,
                    **peer_kwarg(p2p_op),
                )
                if work:
                    reqs.append(work)
        # Make the default stream wait for NCCL to finish, so subsequent
        # reads of recv tensors on the default stream see the received data.
        default_stream.wait_stream(stream)
    else:
        # Non-CUDA or no stream: sequential execution
        for p2p_op in p2p_op_list:
            work = p2p_op.op(
                p2p_op.tensor,
                group=p2p_op.group,
                tag=p2p_op.tag,
                **peer_kwarg(p2p_op),
            )
            if work:
                reqs.append(work)

    return reqs
