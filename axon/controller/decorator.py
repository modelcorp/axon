# Copyright 2025 Model AI Corp.
# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
# Adapted from verl single_controller/base/decorator.py (github.com/volcengine/verl), Apache-2.0.
import inspect
import os
from functools import wraps
from types import FunctionType

from tensordict import TensorDict

from axon.protocol import DataProtoFuture
from axon.utils.py_utils import DynamicEnum
from axon.utils.ray.collective import (
    collect_one_to_all_via_cc,
    dispatch_all_to_all_via_cc,
    dispatch_one_to_all_via_cc,
)
from axon.utils.tensordict_utils import chunk_tensordict, concat_tensordict

MAGIC_ATTR = "mluo_loves_teddy_good_doggy"


class Dispatch(DynamicEnum):
    """Dispatch modes for distributed computation."""

    _registry = {}
    _next_value = 0


class Execute(DynamicEnum):
    """Execution modes for distributed computation."""

    _registry = {}
    _next_value = 0


def init_predefined_dispatch_mode():
    Dispatch.register("RANK_ZERO")
    Dispatch.register("ONE_TO_ALL")
    Dispatch.register("ALL_TO_ALL")
    Dispatch.register("DP_COMPUTE_PROTO_WITH_FUNC")
    Dispatch.register("DIRECT_SAMPLER_METHOD")


def init_predefined_execute_mode():
    Execute.register("ALL")
    Execute.register("RANK_ZERO")


init_predefined_dispatch_mode()
init_predefined_execute_mode()


# =============================================================================
# Core helpers
# =============================================================================


def _chunk_item(item, chunks):
    """Split a single item into chunks."""
    if isinstance(item, TensorDict):
        return chunk_tensordict(item, chunks)
    if hasattr(item, "chunk"):
        chunked = item.chunk(chunks=chunks)
        assert len(chunked) == chunks
        return chunked
    return [item] * chunks


def _broadcast_item(item, world_size, collective=False):
    """Broadcast item to all workers."""
    if collective:
        return [item] + [None] * (world_size - 1)
    return [item] * world_size


def _remap(items, mapping, world_size):
    """Remap chunked items according to rank mapping."""
    return [[item[mapping[i]] for i in range(world_size)] for item in items]


# =============================================================================
# Unified dispatch/collect
# =============================================================================


def dispatch(
    worker_group,
    *args,
    replicate: bool = True,
    split_first_arg: bool = False,
    rank_mapping: list[int] | None = None,
    use_ray_object_store: bool = True,
    **kwargs,
):
    """Dispatch data to workers."""
    world_size = worker_group.world_size
    is_collective = getattr(worker_group, "_ray_collective_initialized", False)

    # Mesh-based dispatch: split by dp_size, then remap to workers
    if rank_mapping is not None:
        dp_size = max(rank_mapping) + 1
        split_args = [_chunk_item(arg, dp_size) for arg in args]
        split_kwargs = {k: _chunk_item(v, dp_size) for k, v in kwargs.items()}

        if use_ray_object_store and not is_collective:
            from axon.utils.ray.utils import parallel_put

            max_workers = max(1, min(len(split_args[0]) if split_args else 1, os.cpu_count()))
            split_args = [parallel_put(arg, max_workers=max_workers) for arg in split_args]
            split_kwargs = {k: parallel_put(v, max_workers=max_workers) for k, v in split_kwargs.items()}

        return (
            tuple(_remap(split_args, rank_mapping, world_size)),
            {k: _remap([v], rank_mapping, world_size)[0] for k, v in split_kwargs.items()},
        )

    # Broadcast: replicate same data to all workers
    if replicate:
        return (
            tuple(_broadcast_item(arg, world_size, is_collective) for arg in args),
            {k: _broadcast_item(v, world_size, is_collective) for k, v in kwargs.items()},
        )

    # Scatter: split data across workers
    if split_first_arg and args and isinstance(args[0], FunctionType):
        # First arg is a function - replicate it, split the rest
        return (
            tuple([[args[0]] * world_size] + [_chunk_item(arg, world_size) for arg in args[1:]]),
            {k: _chunk_item(v, world_size) for k, v in kwargs.items()},
        )

    return (
        tuple(_chunk_item(arg, world_size) for arg in args),
        {k: _chunk_item(v, world_size) for k, v in kwargs.items()},
    )


def collect(worker_group, output, concat: bool = False, collect_mask: list[bool] | None = None):
    """Collect results from workers."""
    # Unwrap collective results (rank 0 has gathered list, others are None)
    if getattr(worker_group, "_ray_collective_initialized", False) or getattr(
        worker_group, "_enable_ray_collective", False
    ):
        non_none = [o for o in output if o is not None]
        if len(non_none) == 1 and isinstance(non_none[0], list):
            output = non_none[0]

    # Filter by mask (mesh-based collect)
    if collect_mask is not None:
        output = [output[i] for i in range(len(output)) if collect_mask[i]]

    return _concat_outputs(output) if concat else output


def _concat_outputs(output: list):
    """Concatenate list of DataProto/TensorDict/BatchMeta into one."""
    if not output:
        return output

    import ray

    from axon.protocol import DataProto, DataProtoFuture

    first = output[0]
    concat_fn = {
        DataProto: DataProto.concat,
        ray.ObjectRef: DataProtoFuture.concat,
        # BatchMeta: BatchMeta.concat,
        TensorDict: concat_tensordict,
    }.get(type(first))

    if concat_fn:
        return concat_fn(output)
    raise NotImplementedError(f"Cannot concat type {type(first)}")


# =============================================================================
# Registry wrappers
# =============================================================================


def _passthrough_dispatch(worker_group, *args, **kwargs):
    return args, kwargs


def _passthrough_collect(worker_group, output):
    return output


def _forbidden(*args, **kwargs):
    raise NotImplementedError("Direct sampler call is forbidden.")


# =============================================================================
# Mesh-based dispatch (nd_compute) - uses unified dispatch/collect
# =============================================================================


def make_nd_compute_dataproto_dispatch_fn(mesh_name):
    """Create dispatch/collect functions for a mesh."""

    def dispatch_fn(worker_group, *args, **kwargs):
        from axon.controller.worker_group import WorkerGroup

        assert isinstance(worker_group, WorkerGroup)

        if mesh_name not in worker_group._dispatch_info:
            worker_group._dispatch_info[mesh_name] = worker_group._query_dispatch_info(mesh_name)
            assert len(worker_group._dispatch_info[mesh_name]) == worker_group.world_size

        rank_mapping = worker_group._dispatch_info[mesh_name]
        use_ray = not getattr(worker_group, "_ray_collective_initialized", False)
        return dispatch(worker_group, *args, rank_mapping=rank_mapping, use_ray_object_store=use_ray, **kwargs)

    def collect_fn(worker_group, output):
        from axon.controller.worker_group import WorkerGroup

        assert isinstance(worker_group, WorkerGroup)
        assert mesh_name in worker_group._dispatch_info

        if mesh_name not in worker_group._collect_info:
            worker_group._collect_info[mesh_name] = worker_group._query_collect_info(mesh_name)
            assert len(worker_group._collect_info[mesh_name]) == worker_group.world_size

        collect_mask = worker_group._collect_info[mesh_name]
        return collect(worker_group, output, collect_mask=collect_mask, concat=True)

    return {"dispatch_fn": dispatch_fn, "collect_fn": collect_fn, "mesh_name": mesh_name}


# =============================================================================
# Registry
# =============================================================================


def _make_dispatch_fn(replicate=True, split_first_arg=False):
    """Create a dispatch function with preset parameters."""

    def fn(worker_group, *args, **kwargs):
        return dispatch(worker_group, *args, replicate=replicate, split_first_arg=split_first_arg, **kwargs)

    return fn


def _make_collect_fn(concat=False):
    """Create a collect function with preset parameters."""

    def fn(worker_group, output):
        return collect(worker_group, output, concat=concat)

    return fn


def get_predefined_dispatch_fn(dispatch_mode):
    return {
        Dispatch.RANK_ZERO: {"dispatch_fn": _passthrough_dispatch, "collect_fn": _passthrough_collect},
        Dispatch.ONE_TO_ALL: {
            "dispatch_fn": _make_dispatch_fn(replicate=True),
            "collect_fn": _make_collect_fn(concat=False),
        },
        Dispatch.ALL_TO_ALL: {"dispatch_fn": _passthrough_dispatch, "collect_fn": _make_collect_fn(concat=False)},
        Dispatch.DP_COMPUTE_PROTO_WITH_FUNC: {
            "dispatch_fn": _make_dispatch_fn(replicate=False, split_first_arg=True),
            "collect_fn": _make_collect_fn(concat=True),
        },
        Dispatch.DIRECT_SAMPLER_METHOD: {"dispatch_fn": _forbidden, "collect_fn": _forbidden},
    }[dispatch_mode]


def get_predefined_execute_fn(execute_mode):
    return {
        Execute.ALL: {"execute_fn_name": "execute_all"},
        Execute.RANK_ZERO: {"execute_fn_name": "execute_rank_zero"},
    }[execute_mode]


# =============================================================================
# Helpers
# =============================================================================


def _check_dispatch_mode(dispatch_mode):
    if isinstance(dispatch_mode, Dispatch):
        return
    if isinstance(dispatch_mode, dict):
        assert "dispatch_fn" in dispatch_mode and "collect_fn" in dispatch_mode
        return
    raise ValueError(f"dispatch_mode must be Dispatch enum or dict, got {type(dispatch_mode)}")


def _check_execute_mode(execute_mode):
    assert isinstance(execute_mode, Execute), f"execute_mode must be Execute. Got {execute_mode}"


def _materialize_futures(*args, **kwargs):
    new_args = tuple(arg.get() if isinstance(arg, DataProtoFuture) else arg for arg in args)
    new_kwargs = {k: v.get() if isinstance(v, DataProtoFuture) else v for k, v in kwargs.items()}
    return new_args, new_kwargs


def _dispatch_with_collective(group_name, args, kwargs, dispatch_mode):
    if dispatch_mode == Dispatch.ONE_TO_ALL:
        return dispatch_one_to_all_via_cc(group_name, args, kwargs)
    elif dispatch_mode == Dispatch.ALL_TO_ALL:
        return dispatch_all_to_all_via_cc(group_name, args, kwargs)
    elif dispatch_mode == Dispatch.DIRECT_SAMPLER_METHOD:
        return args, kwargs
    elif isinstance(dispatch_mode, dict):
        return dispatch_one_to_all_via_cc(group_name, args, kwargs)
    raise NotImplementedError(f"Collective not supported for {dispatch_mode}")


def _collect_with_collective(group_name, result, dispatch_mode):
    if dispatch_mode in (Dispatch.ONE_TO_ALL, Dispatch.ALL_TO_ALL):
        return collect_one_to_all_via_cc(group_name, result)
    elif dispatch_mode == Dispatch.DIRECT_SAMPLER_METHOD:
        return result
    elif isinstance(dispatch_mode, dict):
        return collect_one_to_all_via_cc(group_name, result)
    raise NotImplementedError(f"Collective not supported for {dispatch_mode}")


# =============================================================================
# Decorator
# =============================================================================


def register(
    dispatch_mode=Dispatch.ALL_TO_ALL,
    execute_mode=Execute.ALL,
    blocking: bool = True,
    materialize_futures: bool = True,
    disable_collective: bool = False,
):
    """Register a function with distributed execution configuration."""
    _check_dispatch_mode(dispatch_mode)
    _check_execute_mode(execute_mode)

    def decorator(func):
        @wraps(func)
        def inner(*args, **kwargs):
            worker_group = args[0]
            is_collective = worker_group._enable_ray_collective and not disable_collective

            if materialize_futures:
                args, kwargs = _materialize_futures(*args, **kwargs)

            if is_collective:
                args, kwargs = _dispatch_with_collective(
                    worker_group._ray_collective_group_name, args, kwargs, dispatch_mode
                )

            result = func(*args, **kwargs)

            if is_collective:
                result = _collect_with_collective(worker_group._ray_collective_group_name, result, dispatch_mode)

            return result

        @wraps(func)
        async def async_inner(*args, **kwargs):
            worker_group = args[0]
            is_collective = worker_group._enable_ray_collective and not disable_collective

            if materialize_futures:
                args, kwargs = _materialize_futures(*args, **kwargs)

            if is_collective:
                args, kwargs = _dispatch_with_collective(
                    worker_group._ray_collective_group_name, args, kwargs, dispatch_mode
                )

            result = await func(*args, **kwargs)

            if is_collective:
                result = _collect_with_collective(worker_group._ray_collective_group_name, result, dispatch_mode)

            return result

        wrapper = async_inner if inspect.iscoroutinefunction(func) else inner
        setattr(
            wrapper,
            MAGIC_ATTR,
            {
                "dispatch_mode": dispatch_mode,
                "execute_mode": execute_mode,
                "blocking": blocking,
                "disable_collective": disable_collective,
            },
        )
        return wrapper

    return decorator
