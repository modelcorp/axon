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
"""Unified SamplerWorker base class.

Framework-specific behaviour (FSDP hybrid-engine sync, Megatron hybrid-engine
sync, P2P weight transfer) is injected via mixins composed in
``train_agent_ppo.py``.
"""

import datetime
import logging
import os

import torch
from omegaconf import DictConfig
from torch.distributed.device_mesh import init_device_mesh

from axon.controller.decorator import Dispatch, register
from axon.core.worker import Worker
from axon.sampler import get_engine_class
from axon.utils.import_utils import import_external_libs
from axon.utils.profiler import DistProfilerExtension, init_profiler_on_worker, log_gpu_memory_usage
from axon.utils.torch import (
    get_device_name,
    get_nccl_backend,
    get_torch_device,
    set_expandable_segments,
)
from axon.utils.torch.distributed import set_numa_affinity

device_name = get_device_name()

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("AXON_LOGGING_LEVEL", "WARN"))


class SamplerWorker(Worker, DistProfilerExtension):
    """Unified sampler worker for disaggregated or hybrid-engine inference.

    In **standalone** (disaggregated) mode this worker runs on its own GPU(s)
    and receives weights from a TrainerWorker via P2P (torch.distributed).

    In **fused** (hybrid-engine) mode it is co-located with a TrainerWorker via
    ``fuse_worker_cls()`` and accesses actor weights directly through
    ``self._colocated["actor"]``.

    Framework-specific methods (hybrid-engine weight sync, P2P transfer) are
    provided by mixins; see ``axon.sampler.mixins``.
    """

    def __init__(self, config: DictConfig, **kwargs):
        Worker.__init__(self)
        self.config = config

        if not torch.distributed.is_initialized():
            set_numa_affinity()
            rank = int(os.environ.get("RANK", 0))
            world_size = int(os.environ.get("WORLD_SIZE", 1))
            torch.distributed.init_process_group(
                backend=f"cpu:gloo,{get_device_name()}:{get_nccl_backend()}",
                rank=rank,
                world_size=world_size,
                timeout=datetime.timedelta(seconds=600),
                init_method=os.environ.get("DIST_INIT_METHOD", None),
            )

        # Ensure CUDA device is set to local rank
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        get_torch_device().set_device(local_rank)

        # Profiler from sampler config
        init_profiler_on_worker(self, config.get("profiler", {}))

        set_expandable_segments(False)

        # P2P state (used by P2P mixins)
        self.offload_p2p_buffer = self.config.get("offload_p2p_buffer", False)
        self.ops = []
        self.buffers = []
        self.routing_table = None

    def _build_sampler(self):
        sampler_config = self.config

        infer_tp = self.config.tensor_model_parallel_size * self.config.data_parallel_size
        infer_pp = self.config.pipeline_model_parallel_size
        infer_world_size = infer_tp * infer_pp
        dp = self.world_size // infer_world_size
        assert self.world_size % infer_world_size == 0, (
            f"sampler world_size: {self.world_size} is not divisible by infer_world_size: {infer_world_size}"
        )
        sampler_device_mesh = init_device_mesh(
            device_name, mesh_shape=(dp, infer_pp, infer_tp), mesh_dim_names=["dp", "infer_pp", "infer_tp"]
        )
        sampler_name = self.config.name
        self.sampler_device_mesh = sampler_device_mesh

        if sampler_name == "hf":
            self._register_dispatch_collect_info("sampler", dp_rank=self.rank, is_collect=True)
        else:
            is_collect = (
                sampler_device_mesh["infer_tp"].get_local_rank() == 0
                and sampler_device_mesh["infer_pp"].get_local_rank() == 0
            )
            self._register_dispatch_collect_info(
                "sampler", dp_rank=sampler_device_mesh["dp"].get_local_rank(), is_collect=is_collect
            )

        gen_dp_rank = sampler_device_mesh["dp"].get_local_rank()
        get_torch_device().manual_seed(gen_dp_rank + 1000)
        self.gen_random_states = get_torch_device().get_rng_state()

        log_gpu_memory_usage(f"Before building {self.config.name} sampler", logger=logger)
        self.sampler = get_engine_class(sampler_config.name)(config=sampler_config, device_mesh=sampler_device_mesh)
        log_gpu_memory_usage(f"After building {self.config.name} sampler", logger=logger)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        import_external_libs(self.config.get("external_lib", None))

        self._build_sampler()
        log_gpu_memory_usage("After sampler init", logger=logger)

        get_torch_device().empty_cache()
        log_gpu_memory_usage("After init_model finish", logger=logger)

    # ---- Hybrid engine dispatch methods ----

    @register(dispatch_mode=Dispatch.DIRECT_SAMPLER_METHOD)
    async def wake_up(self):
        if hasattr(self, "sampler_mode"):
            await self.sampler_mode()
        return True

    @register(dispatch_mode=Dispatch.DIRECT_SAMPLER_METHOD)
    async def sleep(self):
        if self.config.offload_sampler:
            log_gpu_memory_usage("Before sampler offload", logger=logger)
            await self.sampler.release()
            log_gpu_memory_usage("After sampler offload", logger=logger)

        actor = getattr(self, "_colocated", {}).get("actor")
        if actor is not None:
            actor.trainer_mode()

        self.gen_random_states = get_torch_device().get_rng_state()
        if hasattr(self, "_colocated") and "actor" in self._colocated:
            get_torch_device().set_rng_state(self._colocated["actor"].torch_random_states)
        return True

    @register(dispatch_mode=Dispatch.DIRECT_SAMPLER_METHOD)
    def get_zeromq_address(self):
        return self.sampler.get_zeromq_address()
