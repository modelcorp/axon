# Copyright 2025 Model AI Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import torch
from megatron.core import dist_checkpointing
from megatron.core.dist_checkpointing.serialization import (
    get_default_load_sharded_strategy,
)

# np.product was removed in NumPy 2.0 but megatron still uses it in validation.py
if not hasattr(np, "product"):
    np.product = np.prod


def save_dist_checkpointing(sharded_state_dict, ckpt_path, async_save=False):
    # For sync saves, patch the checkpoint writer to clone CPU view tensors
    # one-at-a-time instead of all-at-once (~29 GB/rank OOM with full CPU
    # optimizer offloading).  Async saves need the upfront clone so the
    # background writer has independent copies while training continues.

    from axon.monkey_patches.megatron.streaming_checkpointing import (
        apply_streaming_checkpointing_patch,
        revert_streaming_checkpointing_patch,
    )

    if async_save:
        revert_streaming_checkpointing_patch()
    else:
        apply_streaming_checkpointing_patch()

    validate_sharding_integrity = True  # False to make it much faster.
    # Get checkpointing strategies.  Default thread_count=2 means only 2 forked
    # writer processes per rank.  With fast local NVMe, more writers let us
    # saturate I/O by parallelizing torch.save serialization across processes.
    from megatron.core.dist_checkpointing.strategies.torch import TorchDistSaveShardedStrategy

    save_strategy = TorchDistSaveShardedStrategy("torch_dist", 1, thread_count=8)

    # Save model sharded state dicts
    async_save_request = dist_checkpointing.save(
        sharded_state_dict,
        ckpt_path,
        sharded_strategy=save_strategy,
        async_sharded_save=async_save,
        validate_access_integrity=validate_sharding_integrity,
    )

    return async_save_request


def load_dist_checkpointing(sharded_state_dict, ckpt_dir):
    # Get checkpointing strategies
    load_strategy = get_default_load_sharded_strategy(ckpt_dir)

    # Fix torch.load weights only error
    try:
        import transformer_engine as te

        torch.serialization.add_safe_globals([torch.optim.AdamW])
        torch.serialization.add_safe_globals([te.pytorch.optimizers.fused_adam.FusedAdam])
    except Exception:
        pass

    # Load model sharded state dicts
    state_dict = dist_checkpointing.load(sharded_state_dict, ckpt_dir, sharded_strategy=load_strategy)

    return state_dict
