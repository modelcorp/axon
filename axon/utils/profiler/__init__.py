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

from ..import_utils import is_nvtx_available
from ..metrics.timing import marked_timer
from .performance import GPUMemoryLogger, log_gpu_memory_usage
from .profile import DistProfiler, DistProfilerExtension, ProfilerConfig

# NVTX-aware range markers: use NVTX when available, otherwise fall back to no-ops.
if is_nvtx_available():
    from .nvtx_profile import mark_annotate, mark_end_range, mark_start_range
else:
    from .profile import mark_annotate, mark_end_range, mark_start_range


def init_profiler_on_worker(worker, omega_config):
    """Initialize DistProfilerExtension on a worker from an OmegaConf profiler config."""
    from axon.utils.config import get_profiler_tool_config, omega_conf_to_dataclass

    profiler_config = omega_conf_to_dataclass(omega_config, dataclass_type=ProfilerConfig)
    tool_config = get_profiler_tool_config(omega_config)
    DistProfilerExtension.__init__(
        worker, DistProfiler(rank=worker.rank, config=profiler_config, tool_config=tool_config)
    )


__all__ = [
    "GPUMemoryLogger",
    "log_gpu_memory_usage",
    "mark_start_range",
    "mark_end_range",
    "mark_annotate",
    "DistProfiler",
    "DistProfilerExtension",
    "ProfilerConfig",
    "marked_timer",
    "init_profiler_on_worker",
]
