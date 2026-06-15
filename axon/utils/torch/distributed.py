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
"""Utilities for distributed training."""

import ctypes
import logging
import os
from datetime import timedelta

import ray
import torch.distributed

from axon.utils.torch import get_nccl_backend, get_torch_device

logger = logging.getLogger(__name__)


def set_numa_affinity():
    # ROCm/HIP: pynvml is NVIDIA-only, skip NUMA affinity setup
    import torch

    if hasattr(torch.version, "hip") and torch.version.hip is not None:
        logger.info("Detected ROCm/HIP environment, skipping pynvml NUMA affinity setup")
        return

    initialized = False
    try:
        libnuma = ctypes.CDLL("libnuma.so")
        if libnuma.numa_available() < 0:
            return

        import pynvml

        pynvml.nvmlInit()
        initialized = True
        device_name = "GPU"
        local_rank = int(ray.get_runtime_context().get_accelerator_ids()[device_name][0])
        handle = pynvml.nvmlDeviceGetHandleByIndex(local_rank)
        pynvml.nvmlDeviceSetCpuAffinity(handle)
    except ImportError:
        logger.warning("pynvml not available, skipping NUMA affinity setup")
    except Exception as e:
        logger.warning("Failed to set NUMA affinity: %s", e)
    finally:
        if initialized:
            pynvml.nvmlShutdown()


def initialize_global_process_group(timeout_second=36000):
    torch.distributed.init_process_group(
        get_nccl_backend(),
        timeout=timedelta(seconds=timeout_second),
        init_method=os.environ.get("DIST_INIT_METHOD", None),
    )
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    if torch.distributed.is_initialized():
        get_torch_device().set_device(local_rank)
    return local_rank, rank, world_size


def destroy_global_process_group():
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
