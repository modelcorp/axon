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
Monkey patches for PyTorch FSDP.
"""

from typing import Any

import torch.distributed as dist


def apply_fsdp_monkey_patch():
    try:
        from torch.distributed._state_dict_utils import _gather_state_dict, _offload_state_dict_to_cpu
        from torch.distributed.checkpoint import state_dict as state_dict_module

        def patched_maybe_full_or_cpu_state_dict(
            state_dict: dict[str, Any], info: state_dict_module._StateDictInfo
        ) -> dict[str, Any]:
            """
            Patched version that forces ranks_only=(0,) for full state dict.

            This allows rank0_only behavior without cpu_offload=True,
            enabling much faster state dict retrieval by keeping tensors on GPU.
            """
            if info.full_state_dict:
                # PATCHED: Always use ranks_only=(0,) instead of tying it to cpu_offload
                # This ensures only rank 0 gets the full state dict
                ranks_only = (0,) if dist.is_initialized() else ()

                return _gather_state_dict(state_dict, cpu_offload=info.cpu_offload, ranks_only=ranks_only)
            elif info.cpu_offload:
                return _offload_state_dict_to_cpu(state_dict)
            else:
                return state_dict

        # Apply the patch
        state_dict_module._maybe_full_or_cpu_state_dict = patched_maybe_full_or_cpu_state_dict

        return True

    except Exception as e:
        import warnings

        warnings.warn(
            f"Failed to apply FSDP state dict monkey patch: {e}. "
            "State dict operations may be slower or fail with rank0_only=True and cpu_offload=False.",
            RuntimeWarning,
            stacklevel=2,
        )
        return False
