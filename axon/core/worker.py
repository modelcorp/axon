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
# Adapted from verl single_controller/base/worker.py (github.com/volcengine/verl), Apache-2.0.
"""
Base Worker Class.
"""

import os
import uuid

import ray.util.collective as collective

from axon.controller.decorator import Dispatch, register


# we assume that in each WorkerGroup, there is a Master Worker
class Worker:
    """A distributed worker that handles initialization and configuration for distributed training.

    This class manages worker initialization, configuration, and provides methods for executing
    distributed operations. It handles communication settings, device configuration, and worker
    metadata management.
    """

    def _register_dispatch_collect_info(self, mesh_name: str, dp_rank: int, is_collect: bool):
        """Register the dp_rank for a given mesh name. This function is meant to be called by the worker

        Args:
            mesh_name (str):
                Name of the mesh to register dp_rank for.
            dp_rank (int):
                dp_rank to register for the given mesh name.
            is_collect (bool):
                Whether the dp_rank is used for collect.
        """
        if mesh_name in self.__dispatch_dp_rank or mesh_name in self.__collect_dp_rank:
            raise ValueError(f"mesh_name {mesh_name} has been registered")
        self.__dispatch_dp_rank[mesh_name] = dp_rank
        self.__collect_dp_rank[mesh_name] = is_collect

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def _query_dispatch_info(self, mesh_name: str):
        """Query the dispatch info for a given mesh name.

        Args:
            mesh_name (str):
                Name of the mesh to query dispatch info for.

        Returns:
            int:
                The dp_rank for the given mesh name.
        """
        assert mesh_name in self.__dispatch_dp_rank, f"{mesh_name} is not registered in {self.__class__.__name__}"
        # note that each rank store its own dp_rank
        return self.__dispatch_dp_rank[mesh_name]

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def _query_collect_info(self, mesh_name: str):
        """Query the collect info for a given mesh name.

        Args:
            mesh_name (str):
                Name of the mesh to query collect info for.

        Returns:
            bool:
                Whether the dp_rank is used for collect.
        """
        assert mesh_name in self.__collect_dp_rank, f"{mesh_name} is not registered in {self.__class__.__name__}"
        return self.__collect_dp_rank[mesh_name]

    @classmethod
    def env_keys(cls):
        """The keys of the environment variables that are used to configure the Worker."""
        return [
            "WORLD_SIZE",
            "RANK",
            "LOCAL_WORLD_SIZE",
            "LOCAL_RANK",
            "MASTER_ADDR",
            "MASTER_PORT",
            "CUDA_VISIBLE_DEVICES",
            "HIP_VISIBLE_DEVICES",
            "ROCR_VISIBLE_DEVICES",
        ]

    def __init__(self, cuda_visible_devices=None) -> None:
        """Initialize the worker with environment settings and device configuration.

        Args:
            cuda_visible_devices (str, optional):
                CUDA visible devices configuration. Defaults to None.
        """
        self._rank = int(os.environ["RANK"])
        self._world_size = int(os.environ["WORLD_SIZE"])

        master_addr = os.environ["MASTER_ADDR"]
        master_port = os.environ["MASTER_PORT"]
        self._local_world_size = int(os.getenv("LOCAL_WORLD_SIZE", "1"))
        self._local_rank = int(os.getenv("LOCAL_RANK", "0"))

        store = {
            "_world_size": self._world_size,
            "_rank": self._rank,
            "_local_world_size": self._local_world_size,
            "_local_rank": self._local_rank,
            "_master_addr": master_addr,
            "_master_port": master_port,
        }
        if cuda_visible_devices is not None:
            store["_cuda_visible_devices"] = cuda_visible_devices

        self._configure_with_store(store=store)

        self.__dispatch_dp_rank = {}
        self.__collect_dp_rank = {}

        # Collective Parameters
        self._enable_ray_collective = False
        self._ray_collective_group_name = None

        # Bind flashinfer to the real libcudart before any GDN model loads tilelang's
        # libcudart_stub.so (which would otherwise shadow it and crash vLLM
        # init_device on `undefined symbol: cudaDeviceReset`). See flashinfer_compat.
        from axon.utils.flashinfer_compat import ensure_flashinfer_real_libcudart

        ensure_flashinfer_real_libcudart()

    def _configure_with_store(self, store: dict):
        """
        This function should only be called inside by WorkerGroup
        """
        store_env_dict = {f"_{key.lower()}": store.get(f"_{key.lower()}", None) for key in type(self).env_keys()}
        self.__dict__.update(store_env_dict)  # this is hacky
        # print(f"__dict__: {self.__dict__}")
        for key in type(self).env_keys():
            val = self.__dict__.get(f"_{key.lower()}", None)
            if val is not None:
                # print(f"set {key} to {val}")
                os.environ[key] = str(val)
        os.environ["REDIS_STORE_SERVER_HOST"] = (
            str(self._master_addr).replace("[", "").replace("]", "") if self._master_addr else ""
        )

    @property
    def world_size(self):
        """Get the total number of workers in the distributed setup."""
        return self._world_size

    @property
    def rank(self):
        """Get the rank of this worker in the distributed setup."""
        return self._rank

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO_WITH_FUNC)
    def execute_with_func_generator(self, func, *args, **kwargs):
        """Execute a function with function generator dispatch mode.

        Args:
            func:
                Function to execute
            *args:
                Positional arguments for the function
            **kwargs:
                Keyword arguments for the function
        """
        ret_proto = func(self, *args, **kwargs)
        return ret_proto

    # ============================ Ray Collective (CCL) ============================#

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_ray_collective(self, backend: str | None = None, group_name: str | None = None):
        """Initialize Ray Collective group on all workers.

        Args:
            backend: Collective backend. Defaults to "nccl" when CUDA GPUs are used, otherwise "gloo".
            group_name: Unique collective group name shared by all workers in the group.
        """

        if group_name is None:
            group_name = str(uuid.uuid4())[:16]

        if self._ray_collective_group_name is None:
            self._enable_ray_collective = True
            self._ray_collective_group_name = group_name
        else:
            return
        if backend is None:
            # Default backend selection — ROCm uses RCCL which exposes as "nccl" in PyTorch
            _not_set = (None, "", "not set")
            has_gpu = (
                os.environ.get("CUDA_VISIBLE_DEVICES") not in _not_set
                or os.environ.get("HIP_VISIBLE_DEVICES") not in _not_set
                or os.environ.get("ROCR_VISIBLE_DEVICES") not in _not_set
            )
            backend = "nccl" if has_gpu else "gloo"

        collective.init_collective_group(
            world_size=self._world_size, rank=self._rank, backend=backend, group_name=group_name
        )
        # Ensure all ranks have finished initialization before proceeding
        collective.barrier(group_name=group_name)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def destroy_ray_collective(self, group_name: str | None = None):
        """Destroy the specified Ray Collective group on all workers."""
        if group_name is None:
            group_name = self._ray_collective_group_name
        # Barrier before destruction to avoid races
        try:
            collective.barrier(group_name=group_name)
        except Exception:
            pass
        try:
            collective.destroy_collective_group(group_name=group_name)
        except Exception:
            # Allow idempotent destroy across retries or partially created groups
            pass
