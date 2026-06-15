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
# Adapted from verl single_controller/ray/base.py (github.com/volcengine/verl), Apache-2.0.
import logging
import os
import re
import socket

import ray
from ray.experimental.state.api import get_actor
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from axon.controller.ray.class_init import RayActorWithInitArgs
from axon.controller.ray.fused_worker_group import RoleProxy, fuse_worker_cls
from axon.controller.worker_group import WorkerGroup
from axon.core import ResourcePool
from axon.protocol import DataProto, _padding_size_key
from axon.utils.module_loader import get_random_string
from axon.utils.ray.placement_group import create_placement_groups, sort_placement_group_by_node_ip

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("AXON_LOGGING_LEVEL", "WARN"))


@ray.remote
def get_master_addr_port() -> tuple[str, str]:
    addr = ray.util.get_node_ip_address().strip("[]")
    with socket.socket() as sock:
        sock.bind(("", 0))
        port = sock.getsockname()[1]
    return addr, str(port)


def _make_wg_view(base, **overrides):
    # dynamic subclass so isinstance(view, WorkerGroup) is True
    WG = type(base)

    class WGView(WG):
        __slots__ = ("_base", "_overrides")

        def __getattribute__(self, name):
            if name in ("_base", "_overrides", "__class__", "__dict__", "__slots__", "__weakref__"):
                return object.__getattribute__(self, name)
            ov = object.__getattribute__(self, "_overrides")
            if name in ov:
                return ov[name]
            return getattr(object.__getattribute__(self, "_base"), name)

        def __setattr__(self, name, value):
            ov = object.__getattribute__(self, "_overrides")
            if name in ov:
                ov[name] = value
            else:
                setattr(object.__getattribute__(self, "_base"), name, value)

    view = object.__new__(WGView)  # avoid calling WG.__init__
    object.__setattr__(view, "_base", base)
    object.__setattr__(view, "_overrides", dict(overrides))
    return view


def func_generator(self, method_name, dispatch_fn, collect_fn, execute_fn, blocking, disable_collective):
    class Functor:
        def __call__(this, *args, **kwargs):
            wg_view = _make_wg_view(self, _ray_collective_initialized=False) if disable_collective else self
            args, kwargs = dispatch_fn(wg_view, *args, **kwargs)
            padding_count = kwargs.pop(_padding_size_key, 0)
            output = execute_fn(method_name, *args, **kwargs)
            if blocking:
                output = ray.get(output)
            output = collect_fn(wg_view, output)
            if padding_count > 0:
                if isinstance(output, DataProto):
                    indices = [i for i in range(len(output))][:-padding_count]
                    output = output.select_idxs(indices)
                elif isinstance(output, list):
                    output = output[:-padding_count]

            # Edge case: First time, init_ray_collective  is called, treat it as non-collective call.
            if self._enable_ray_collective and "init_ray_collective" in method_name:
                self._ray_collective_initialized = True
            return output

    # use class type to pass the method_name to get a better observability
    return type(method_name, (Functor,), {})()


class RayWorkerGroup(WorkerGroup):
    """A group of Ray workers that can be managed collectively.

    This class extends WorkerGroup to provide Ray-specific functionality for
    creating and managing groups of Ray actors with specific resource requirements
    and scheduling strategies.
    """

    def __init__(
        self,
        resource_pool: ResourcePool = None,
        ray_cls_with_init: RayActorWithInitArgs = None,
        bin_pack: bool = True,
        name_prefix: str = None,
        worker_names=None,
        worker_handles: list[ray.actor.ActorHandle] = None,
        ray_wait_register_center_timeout: int = 300,
        enable_ray_collective: bool = False,
        **kwargs,
    ) -> None:
        """Initialize a RayWorkerGroup.

        Args:
            resource_pool: Resource pool for worker allocation
            ray_cls_with_init: Class with initialization arguments for workers
            bin_pack: Whether to use strict bin packing for resource allocation
            name_prefix: Prefix for worker names
            worker_names: Names of existing workers to attach to
            worker_handles: Handles of existing workers to attach to
            ray_wait_register_center_timeout: Timeout for waiting on register center
            **kwargs: Additional keyword arguments
        """
        super().__init__(resource_pool=resource_pool, **kwargs)
        self.resource_pool = resource_pool
        self._skip_init_workers = worker_names is not None
        self._worker_names = worker_names
        self.ray_cls_with_init = ray_cls_with_init
        # Optional Ray Collective initialization
        self._enable_ray_collective = enable_ray_collective
        self.name_prefix = get_random_string(length=6) if name_prefix is None else name_prefix
        self._ray_wait_register_center_timeout = ray_wait_register_center_timeout
        self.device_name = kwargs.get("device_name", "cuda")
        self.profile_steps = kwargs.get("profile_steps", None)

        self.wg_dict = None
        self.method_names = []

        # Select backend based on device type
        # Use NCCL for CUDA GPUs, otherwise default to Gloo for CPU/NPU
        self._ray_collective_backend = "nccl" if self.device_name == "cuda" else "gloo"
        self._ray_collective_group_name = f"{self.name_prefix}_ray_cc"
        self._ray_collective_initialized = False

        self._master_addr = kwargs.pop("master_addr", None)
        self._master_port = kwargs.pop("master_port", None)

        if self._skip_init_workers:
            # Workers are already created, so we just need to attach them to the worker group.
            self._init_with_existing_workers(worker_names=worker_names, worker_handles=worker_handles)
        else:
            # Create new workers from a resource pool.
            self._init_with_resource_pool(
                resource_pool=resource_pool,
                ray_cls_with_init=ray_cls_with_init,
                bin_pack=bin_pack,
            )

        if ray_cls_with_init is not None:
            self._bind_worker_method(self.ray_cls_with_init.cls, func_generator)

    def _is_worker_alive(self, worker: ray.actor.ActorHandle):
        """Check if a worker actor is still alive.

        Args:
            worker: Ray actor handle to check

        Returns:
            bool: True if the worker is alive, False otherwise
        """
        worker_state_dict = get_actor(worker._actor_id.hex())
        return worker_state_dict.get("state", "undefined") == "ALIVE" if worker_state_dict is not None else False

    def _init_with_existing_workers(self, worker_names, worker_handles):
        # ray.get_actor holds a weak reference to the actor, which can let actors be collected unexpectedly
        # if we only hold a spawned RayWorkerGroup. Passing actor handles explicitly
        # gives the spawned RayWorkerGroup strong references to these actors.
        # https://github.com/ray-project/ray/pull/45699
        workers = worker_handles if worker_handles else [ray.get_actor(name=name) for name in worker_names]
        self._workers = workers
        self._world_size = len(workers)

    def _get_master_addr_port(self, pg):
        """Get master addr and port for this worker group"""

        if self._master_addr is None and self._master_port is None:
            self._master_addr, self._master_port = ray.get(
                get_master_addr_port.options(
                    scheduling_strategy=PlacementGroupSchedulingStrategy(
                        placement_group=pg, placement_group_bundle_index=0
                    ),
                ).remote()
            )
        elif self._master_addr is not None and self._master_port is not None:
            logger.debug(f"{self._master_addr=} {self._master_port=}")
        else:
            raise ValueError(
                "Both 'master_addr' and 'master_port' must be provided if you intend to manually specify them, "
                "or neither should be provided to use Ray's default assignment."
            )

    def _init_with_resource_pool(self, resource_pool, ray_cls_with_init, bin_pack):
        """Initialize the worker group by creating new workers from a resource pool.

        Args:
            resource_pool: Resource pool for worker allocation
            ray_cls_with_init: Class with initialization arguments for workers
            bin_pack: Whether to use strict bin packing for resource allocation
        """
        strategy = "PACK" if not bin_pack else "STRICT_PACK"
        pgs, bundles_list = create_placement_groups(
            resource_pool=resource_pool, strategy=strategy, device_name=self.device_name
        )
        self._workers = []
        self._worker_names = []
        self._world_size = resource_pool.world_size
        self._bundles_list = bundles_list
        rank = 0
        for pg_idx, pg in enumerate(sort_placement_group_by_node_ip(pgs)):
            if pg_idx == 0:
                self._get_master_addr_port(pg)
            for local_rank in range(pg.bundle_count):
                self._init_worker(
                    rank=rank,
                    pg_idx=pg_idx,
                    pg=pg,
                    local_rank=local_rank,
                    resource_pool=resource_pool,
                    ray_cls_with_init=ray_cls_with_init,
                )
                rank += 1

    def _init_worker(self, rank, pg_idx, pg, local_rank, resource_pool, ray_cls_with_init):
        # we pass in environment variable at option so that Worker can use environment variable to set
        env_vars = {
            "WORLD_SIZE": str(self._world_size),
            "RANK": str(rank),
            "WG_PREFIX": self.name_prefix,
            "WG_BACKEND": "ray",
            "RAY_LOCAL_WORLD_SIZE": str(pg.bundle_count),
            "LOCAL_WORLD_SIZE": str(pg.bundle_count),
            "MASTER_ADDR": self._master_addr,
            "MASTER_PORT": self._master_port,
        }
        cia_name = type(ray_cls_with_init.cls).__name__
        match = re.search(r"ActorClass\(([^)]+)\)", cia_name)  # ray.remote(Obj) -> "ActorClass(Obj)"
        cia_name = match.group(1) if match else cia_name  # "ActorClass(Obj)" -> "Obj"
        name = f"{cia_name}_{self.name_prefix}:{pg_idx}_{local_rank}"  # e.g. Worker_2:5
        ray_cls_with_init.update_options({"runtime_env": {"env_vars": env_vars}, "name": name})
        resource_dict = self._bundles_list[pg_idx][local_rank]
        # resource_dict = {k: float(v)/resource_pool.max_colocate_count for k, v in resource_dict.items()}
        worker = ray_cls_with_init(
            placement_group=pg,
            placement_group_bundle_idx=local_rank,
            resource_dict=resource_dict,
        )
        self._workers.append(worker)
        self._worker_names.append(name)

    @property
    def worker_names(self):
        return self._worker_names

    def _execute_remote_single_worker(self, worker, method_name: str, *args, **kwargs):
        """Execute a method on a single worker remotely.

        Args:
            worker: The worker actor handle
            method_name: Name of the method to execute
            *args: Positional arguments for the method
            **kwargs: Keyword arguments for the method

        Returns:
            Remote object reference to the method execution
        """
        remote_call = getattr(worker, method_name)
        return remote_call.remote(*args, **kwargs)

    def execute(
        self,
        method_name: str,
        *args,
        ranks: list[int] | None = None,
        blocking: bool = False,
        **kwargs,
    ):
        """Execute a method on specified worker ranks.

        Args:
            method_name: Name of the method to execute
            *args: Positional arguments for the method
            ranks: List of rank indices to execute on. None means all workers.
            blocking: If True, wait for results (sync). If False, return futures (async).
            **kwargs: Keyword arguments for the method

        Returns:
            Results (if blocking) or remote object references (if not blocking)
        """
        if ranks is None:
            workers = self._workers
        else:
            workers = [self._workers[r] for r in ranks]

        length = len(workers)

        # If all args/kwargs are lists matching worker count, distribute them
        # Note: all() on empty sequence returns True (vacuous truth), which correctly
        # handles kwargs-only or args-only cases
        if all(isinstance(arg, list) for arg in args) and all(isinstance(v, list) for v in kwargs.values()):
            if all(len(arg) == length for arg in args) and all(len(v) == length for v in kwargs.values()):
                results = [
                    self._execute_remote_single_worker(
                        workers[i],
                        method_name,
                        *(arg[i] for arg in args),
                        **{k: v[i] for k, v in kwargs.items()},
                    )
                    for i in range(length)
                ]
                # Unwrap single-element list for single rank execution
                if ranks is not None and len(ranks) == 1:
                    results = results[0]
                return ray.get(results) if blocking else results

        # Fallback: broadcast same args/kwargs to all workers
        results = [self._execute_remote_single_worker(w, method_name, *args, **kwargs) for w in workers]

        # Unwrap single-element list for single rank execution
        if ranks is not None and len(ranks) == 1:
            results = results[0]

        return ray.get(results) if blocking else results

    # Backward-compatible aliases (used by decorator system)
    def execute_all(self, method_name: str, *args, **kwargs):
        """Execute on all workers (async). Alias for decorator compatibility."""
        return self.execute(method_name, *args, **kwargs)

    def execute_rank_zero(self, method_name: str, *args, **kwargs):
        """Execute on rank 0 (async). Alias for decorator compatibility."""
        return self.execute(method_name, *args, ranks=[0], **kwargs)

    @property
    def master_address(self):
        return self._master_addr

    @property
    def master_port(self):
        return self._master_port

    @property
    def workers(self):
        return self._workers

    @property
    def world_size(self):
        return self._world_size


def init_worker_group(class_dict, resource_pool, **kwargs) -> dict:
    """
    Create colocated workers and return role views.

    Usage:
        views = colocate({"actor": actor_cls, "critic": critic_cls}, resource_pool)
        views["actor"].init_model()
    """
    if len(class_dict) == 1:
        role, ray_cls = next(iter(class_dict.items()))
        wg = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=ray_cls, **kwargs)
        return {role: wg}

    # Need to merge Ray actor class (i.e. actor & critic on same GPU) into one actor class.
    worker_cls_with_init = fuse_worker_cls(class_dict)
    wg = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=worker_cls_with_init, **kwargs)
    return {r: RoleProxy(wg, str(r)) for r in class_dict}
