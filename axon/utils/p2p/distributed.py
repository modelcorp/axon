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
# init_trainer_sampler_process_group adapted from PyTorch torch.distributed.init_process_group (github.com/pytorch/pytorch), BSD-3-Clause.
from datetime import timedelta
from typing import Any

import torch
from torch._C._distributed_c10d import Store
from torch.distributed.distributed_c10d import (
    Backend,
    PrefixStore,
    _check_valid_timeout,
    _get_default_timeout,
    _new_process_group_helper,
    _world,
    rendezvous,
)
from torch.utils._typing_utils import not_none


def init_trainer_sampler_process_group(
    backend: str | None = None,
    init_method: str | None = None,
    timeout: timedelta | None = None,
    world_size: int = -1,
    rank: int = -1,
    store: Store | None = None,
    group_name: str = "",
    pg_options: Any | None = None,
    device_id: torch.device | None = None,
):
    assert backend != Backend.MPI, "MPI backend is not supported for custom process groups"
    assert (store is None) or (init_method is None), "Cannot specify both init_method and store."

    if store is not None:
        assert world_size > 0, "world_size must be positive if using store"
        assert rank >= 0, "rank must be non-negative if using store"
    elif init_method is None:
        init_method = "env://"

    # If user did not provide a backend string but provided a device id, e.g.
    # >>> init_process_group(device_id=device)
    # we try to figure out the backend name based on the device type.
    if backend is None and device_id is not None:
        # Note: 3rd-party devices can register default backend through the
        # default map below.
        backend = Backend.default_device_backend_map.get(device_id.type)

    # If we still cannot figure it out, e.g.
    # >>> init_process_group()
    # we set it to `undefined` and rely on lazy init.
    if backend is None:
        backend = "undefined"

    # Convert string into `Backend` type
    backend = Backend(backend)

    if timeout is None:
        timeout = _get_default_timeout(backend)

    _check_valid_timeout(timeout)

    # backward compatible API
    if store is None:
        rendezvous_iterator = rendezvous(not_none(init_method), rank, world_size, timeout=timeout)
        store, rank, world_size = next(rendezvous_iterator)
        store.set_timeout(timeout)

        # Use a PrefixStore to avoid accidental overrides of keys used by
        # different systems (e.g. RPC) in case the store is multi-tenant.
        # CRITICAL: Use group_name as prefix to avoid collision with default PG
        prefix = group_name if group_name else "custom_pg"
        store = PrefixStore(prefix, store)

    # For cross-worker process groups (actors + samplers), we pass empty ranks list
    # to avoid the default PG rank checks in _new_process_group_helper.
    # This treats it like a "default" group but for a custom set of processes.
    pg, _ = _new_process_group_helper(
        world_size,
        rank,
        [],  # Empty list to bypass default PG rank validation
        backend,
        store,
        group_name=group_name,
        backend_options=pg_options,
        timeout=timeout,
        device_id=device_id,
        group_desc="custom_pg",
    )

    _world.pg_group_ranks[pg] = {i: i for i in range(world_size)}
    return pg
