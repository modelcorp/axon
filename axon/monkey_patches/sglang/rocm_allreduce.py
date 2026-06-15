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
"""ROCm monkey-patch for SGLang's CustomAllreduce.

On HIP, when torch_memory_saver CUDA graph mode is active, the registered
allreduce path causes memory pressure.  This patch makes SGLang use the
unregistered allreduce path instead, matching the fix in the miles framework
(docker/amd_patch/latest/sglang.patch).

The patch modifies CustomAllreduce.custom_all_reduce():

  BEFORE (line 413 in sglang-miles):
      if _is_hip:
          return self.all_reduce_reg(input)

  AFTER:
      if _is_hip:
          if self.tms_cudagraph:
              return self.all_reduce_unreg(input)
          return self.all_reduce_reg(input)

Note: self.tms_cudagraph is already set by CustomAllreduce.__init__ (line 212)
from envs.SGLANG_MEMORY_SAVER_CUDA_GRAPH, so no GroupCoordinator patch is
needed.
"""

import logging

from axon.utils.rocm_utils import is_rocm

logger = logging.getLogger(__name__)

_PATCH_APPLIED = False


def apply_sglang_rocm_allreduce_patch() -> None:
    """Monkey-patch SGLang's CustomAllreduce for ROCm HIP compatibility.

    No-op on CUDA or if SGLang is not installed.  Safe to call multiple times.
    """
    global _PATCH_APPLIED
    if _PATCH_APPLIED or not is_rocm():
        return

    try:
        import sglang.srt.distributed.device_communicators.custom_all_reduce as car_mod
    except ImportError:
        return

    _patch_custom_allreduce(car_mod)

    _PATCH_APPLIED = True
    logger.info("[ROCm] Applied SGLang CustomAllreduce monkey-patch for HIP")


def _patch_custom_allreduce(car_mod) -> None:
    """Patch CustomAllreduce.custom_all_reduce to use all_reduce_unreg on HIP
    when tms_cudagraph is set.

    The original code (line 411-414 in custom_all_reduce.py):
        if self._IS_CAPTURING:
            if torch.cuda.is_current_stream_capturing():
                if _is_hip:
                    return self.all_reduce_reg(input)   # <-- always reg

    After patching:
        if self._IS_CAPTURING:
            if torch.cuda.is_current_stream_capturing():
                if _is_hip:
                    if self.tms_cudagraph:
                        return self.all_reduce_unreg(input)  # <-- unreg when tms
                    return self.all_reduce_reg(input)
    """
    import torch

    CAClass = car_mod.CustomAllreduce
    _orig_custom_all_reduce = CAClass.custom_all_reduce

    def _patched_custom_all_reduce(self, input: torch.Tensor):
        # Intercept the specific code path: _IS_CAPTURING + stream capturing + HIP + tms_cudagraph
        if (
            self._IS_CAPTURING
            and torch.cuda.is_current_stream_capturing()
            and getattr(car_mod, "_is_hip", False)
            and getattr(self, "tms_cudagraph", False)
        ):
            # Must still check disabled/should_custom_ar (same as original line 409-410)
            if self.disabled or not self.should_custom_ar(input):
                return None
            return self.all_reduce_unreg(input)
        return _orig_custom_all_reduce(self, input)

    CAClass.custom_all_reduce = _patched_custom_all_reduce
