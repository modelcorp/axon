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

"""Sync trainer mixins for switching between trainer and sampler modes."""

import torch

from axon.utils.memory_utils import aggressive_empty_cache
from axon.utils.torch import set_expandable_segments


class SyncTrainerMixin:
    """Provides trainer_mode() for TrainerWorker for hybrid engine."""

    def trainer_mode(self):
        """Switch actor back to training mode. Called by colocated SamplerWorker.sleep()."""
        raise NotImplementedError


class FSDPSyncTrainerMixin(SyncTrainerMixin):
    """Provides trainer_mode() for FSDP TrainerWorker."""

    def trainer_mode(self):
        """Switch actor back to training mode. Called by colocated SamplerWorker.sleep()."""
        self.module_fsdp.train()
        # NOTE: do NOT call torch._C._cuda_clearCublasWorkspaces() here. When the
        # colocated vLLM sampler captures CUDA graphs (the default, performant path),
        # the cuBLAS workspace pointers used by the model's linear layers (qkv/mlp
        # GEMMs) are baked into the captured PIECEWISE graphs at sampler-init time.
        # Those graphs persist for the whole run, so freeing the workspaces frees
        # memory the next graph replay (the first rollout after this sleep) still
        # references -> cudaErrorIllegalAddress. Reclaiming the (tens of MB) cuBLAS
        # workspace is not worth corrupting the sampler's CUDA graphs.
        aggressive_empty_cache(force_sync=True)
        set_expandable_segments(True)


class MegatronSyncTrainerMixin(SyncTrainerMixin):
    """Provides trainer_mode() for Megatron TrainerWorker."""

    def trainer_mode(self):
        """Switch actor back to training mode. Called by colocated SamplerWorker.sleep()."""
        for model in self.module:
            model.train()
        aggressive_empty_cache(force_sync=True)
        # Do NOT enable expandable_segments for Megatron.
        #  Megatron-core officially does not support expandable_segments
        # (see megatron.core.distributed.distributed_data_parallel_config and
        # megatron.core.transformer.cuda_graphs for explicit warnings).
        # set_expandable_segments(True)
