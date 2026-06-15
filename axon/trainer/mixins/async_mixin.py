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

"""AsyncTrainerMixin: "send data once, then commands" pattern for actor workers.

This mixin is added to actor worker classes (FSDP or Megatron) to enable an
efficient training protocol where:

1. The controller (or sampler via P2P) sends a DataProto batch ONCE
2. The controller issues a sequence of compute commands (log probs, training)
3. Each command operates on the stored batch and returns metrics only
4. After all commands, the batch is cleared from GPU memory

Protocol::

    # Controller side:
    actor_wg.set_batch(batch)                    # send data once
    actor_wg.compute_log_prob_on_batch(temp)     # command → metrics
    actor_wg.train_on_batch(loss, args, 4, 32)   # command → metrics
    actor_wg.get_batch_metrics_and_clear(timing)  # metrics + cleanup
"""

import copy
import logging

import torch
import torch.distributed as dist
from torch.distributed.distributed_c10d import P2POp, batch_isend_irecv

from axon.controller.decorator import Dispatch, register
from axon.driver.driver_utils import process_mini_batch, split_mini_batches_by_programs
from axon.protocol import DataProto
from axon.utils.metrics import (
    compute_data_metrics,
    compute_timing_metrics,
    compute_trainer_sampler_mismatch_metrics,
    reduce_metrics,
)
from axon.utils.print_utils import append_to_dict
from axon.utils.ray.collective import decode_cuda_tensor_to_object
from axon.utils.torch import get_device_id, get_device_name

logger = logging.getLogger(__name__)


class AsyncTrainerMixin:
    """Mixin providing the trainer protocol for actor workers in async mode.

    Expects the host class to provide:
    - self.forward(data: DataProto) -> DataProto  (forward-only pass)
    - self.forward_backward(data, loss_fn, loss_fn_args) -> DataProto  (forward+backward)
    - self.optim_step(step_lr=bool) -> DataProto  (optimizer step)
    - self.load_model(include_model, include_optimizer)
    - self.offload_model(include_model, include_optimizer)
    - self.config  (worker config with actor.* settings)
    """

    # ------------------------------------------------------------------
    # Data transfer methods
    # ------------------------------------------------------------------

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=True, disable_collective=True)
    def set_batch(self, batch: DataProto):
        """Store a DataProto batch for subsequent training operations.

        Called by the controller to send training data. After this call,
        compute_log_prob_on_batch() and train_on_batch() operate on the
        stored batch without additional data transfers.

        Args:
            batch: DataProto containing the training batch.

        Returns:
            True on success.
        """
        self.active_batch = batch
        if hasattr(self.active_batch.batch, "unlock_") and self.active_batch.batch.is_locked:
            self.active_batch.batch.unlock_()
        return True

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def receive_batch_from_sampler(self, batch_uid):
        """Receive a DataProto batch from the sampler worker via NCCL P2P.

        Used in worker-resident mode where the sampler worker sends the
        prepared training batch directly to actor workers via GPU-to-GPU
        transfer, bypassing the controller entirely.

        The transfer uses a two-phase protocol:
        1. Receive the byte count (int64 tensor)
        2. Receive the serialized DataProto (uint8 tensor)

        Args:
            batch_uid: Unique identifier for this batch transfer.
        """
        assert hasattr(self, "sampler_bridge_pg"), "Bridge process group not initialized for P2P"

        device = torch.device(get_device_name(), get_device_id())

        # Sampler rank in the bridge process group is after all actor ranks
        actor_offset = self.world_size
        sampler_rank = actor_offset

        # Phase 1: receive size
        size_gpu = torch.zeros(1, dtype=torch.int64, device=device)
        size_op = P2POp(op=dist.irecv, tensor=size_gpu, peer=sampler_rank, group=self.sampler_bridge_pg)
        size_reqs = batch_isend_irecv([size_op])
        for req in size_reqs:
            req.wait()

        # Phase 2: receive serialized DataProto
        num_bytes = int(size_gpu.item())
        recv_buffer = torch.empty(num_bytes, dtype=torch.uint8, device=device)
        data_op = P2POp(op=dist.irecv, tensor=recv_buffer, peer=sampler_rank, group=self.sampler_bridge_pg)
        torch.cuda.synchronize()
        data_reqs = batch_isend_irecv([data_op])
        for req in data_reqs:
            req.wait()

        # Decode and store in the batch buffer
        if not hasattr(self, "_batch_buffer_dict"):
            self._batch_buffer_dict = {}
        self._batch_buffer_dict[batch_uid] = decode_cuda_tensor_to_object(recv_buffer).to("cpu")
        del recv_buffer

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=True, disable_collective=True)
    def set_batch_from_uid(self, batch_uid):
        """Activate a previously received P2P batch by its UID.

        After receive_batch_from_sampler() stores a batch, this method
        moves it into the active_batch slot.

        Args:
            batch_uid: UID of the batch to activate.

        Returns:
            True on success, False if UID not found.
        """
        if not hasattr(self, "_batch_buffer_dict") or batch_uid not in self._batch_buffer_dict:
            logger.error(f"Batch UID not found in buffer: {batch_uid}")
            return False
        self._active_batch_uid = batch_uid
        self.active_batch = self._batch_buffer_dict[batch_uid]
        if hasattr(self.active_batch.batch, "unlock_") and self.active_batch.batch.is_locked:
            self.active_batch.batch.unlock_()
        return True

    # ------------------------------------------------------------------
    # Compute methods (operate on stored active_batch)
    # ------------------------------------------------------------------

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=True, disable_collective=True)
    def use_sampler_log_probs(self, temperature):
        """Use sampler-computed log probs as old_log_probs (skip recomputation).

        When the sampler already computed log probs during generation,
        this method copies them directly instead of running a forward pass.
        This is valid when sampler and actor share the same weights.

        Args:
            temperature: Sampling temperature used during generation.

        Returns:
            Dict of sampler-actor mismatch metrics.
        """
        self.active_batch.batch["old_log_probs"] = self.active_batch.batch["sampler_log_probs"]
        self.active_batch.meta_info["temperature"] = temperature
        self.active_batch.meta_info["global_token_num"] = torch.sum(
            self.active_batch.batch["attention_mask"], dim=-1
        ).tolist()
        return compute_trainer_sampler_mismatch_metrics(self.active_batch)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=True, disable_collective=True)
    def compute_log_prob_on_batch(self, temperature):
        """Run a forward-only pass to compute log probs on the stored batch.

        Computes old_log_probs and entropy via the worker's forward() method,
        then merges them into the stored active_batch. Returns mismatch
        metrics comparing sampler vs actor log probs.

        Args:
            temperature: Sampling temperature for the forward pass.

        Returns:
            Dict of sampler-actor mismatch metrics.
        """
        assert isinstance(self.active_batch, DataProto)
        self.active_batch.meta_info["temperature"] = temperature
        self.active_batch.meta_info["micro_batch_size"] = self.config.forward_micro_batch_size_per_gpu
        self.active_batch.meta_info["max_token_len"] = self.config.forward_max_token_len_per_gpu
        self.active_batch.meta_info["use_dynamic_bsz"] = self.config.forward_use_dynamic_bsz

        output = self.forward(copy.deepcopy(self.active_batch))
        assert isinstance(output, DataProto)

        old_log_prob = DataProto.from_dict(
            tensors={"old_log_probs": output.batch["log_probs"]},
        )
        del output
        self.active_batch = self.active_batch.union(old_log_prob)
        del old_log_prob
        self.active_batch.meta_info["global_token_num"] = torch.sum(
            self.active_batch.batch["attention_mask"], dim=-1
        ).tolist()
        return compute_trainer_sampler_mismatch_metrics(self.active_batch)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False, disable_collective=True)
    def train_on_batch(
        self,
        loss_fn: str,
        loss_fn_args: dict,
        epochs: int = 1,
        mini_batch_size: int | None = None,
        mini_batch_shuffle: bool = False,
        mini_batch_seed: int | None = None,
    ):
        """Run the full PPO training loop on the stored batch.

        **Online SGD mode**: ``optim_step`` runs after each mini-batch.
        Each mini-batch uses global mini-batch counts (local × D) so the
        gradient matches what a single GPU would compute on all D workers'
        data combined. See ``update_trainer`` docstring for a blueprint on
        how to switch to batch-SGD (gradient accumulation) mode.

        Works with both FSDP and Megatron actor workers.

        Args:
            loss_fn: Name of the registered loss function (e.g. "ppo", "grpo").
            loss_fn_args: Dict of loss function arguments.
            epochs: Number of PPO epochs.
            mini_batch_size: Global mini-batch size (will be divided by DP size).
                If None or <= 0, uses the full batch.
            mini_batch_shuffle: Shuffle samples before splitting each epoch.
            mini_batch_seed: Random seed for reproducible shuffling.

        Returns:
            DataProto with meta_info["metrics"] containing aggregated training metrics.
        """
        self.load_model(include_model=True, include_optimizer=True)

        try:
            data = copy.deepcopy(self.active_batch)

            # mini_batch_size is global; normalize to per-worker size
            if hasattr(self, "device_mesh"):
                dp_size = self.device_mesh.size() // getattr(self, "ulysses_sequence_parallel_size", 1)
            else:
                from megatron.core import parallel_state as mpu

                dp_size = mpu.get_data_parallel_world_size()
            if mini_batch_size and mini_batch_size > 0:
                mini_batch_size = mini_batch_size // dp_size
            if not mini_batch_size or mini_batch_size <= 0:
                mini_batch_size = len(data)

            generator = torch.Generator().manual_seed(mini_batch_seed) if mini_batch_seed is not None else None

            metrics = {}
            for epoch in range(epochs):
                mini_batches = split_mini_batches_by_programs(
                    data,
                    mini_batch_size,
                    world_size=1,
                    shuffle=mini_batch_shuffle,
                    generator=generator,
                )
                last_epoch = epoch == epochs - 1

                for i, mini_batch in enumerate(mini_batches):
                    # Compute per-mini-batch counts then scale to GLOBAL
                    # mini-batch counts (×D). This matches the controller path
                    # where process_mini_batch sees the full mini-batch (all D
                    # workers' data) before DP dispatch.
                    # Local: ~B/(K*D).  × D → B/K = global mini-batch count.
                    process_mini_batch(mini_batch)
                    for key in ("valid_batch_size", "valid_token_count", "valid_program_count"):
                        if key in mini_batch.batch:
                            mini_batch.batch[key] = mini_batch.batch[key] * dp_size

                    output = self.forward_backward(mini_batch, loss_fn=loss_fn, loss_fn_args=loss_fn_args)
                    append_to_dict(metrics, output.meta_info["metrics"])

                    last_mini_batch = last_epoch and i == len(mini_batches) - 1
                    optim_output = self.optim_step(step_lr=last_mini_batch)
                    append_to_dict(metrics, optim_output.meta_info["metrics"])

            output = DataProto(meta_info={"metrics": reduce_metrics(metrics)}).to("cpu")
        finally:
            self.offload_model(include_model=True, include_optimizer=True)

        return output

    # ------------------------------------------------------------------
    # Metrics and cleanup
    # ------------------------------------------------------------------

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=True, disable_collective=True)
    def get_batch_metrics_and_clear(self, timing_raw):
        """Compute data/timing metrics from the stored batch, then release it.

        This should be the LAST call in a training step. After this call,
        active_batch is deleted to free GPU/CPU memory.

        Args:
            timing_raw: Dict of raw timing measurements from the controller.

        Returns:
            Dict of data metrics (reward stats, token counts) and timing metrics.
        """
        metrics = {}
        if hasattr(self, "active_batch") and self.active_batch is not None:
            data_metrics = compute_data_metrics(batch=self.active_batch, use_critic=False, include_detail=True)
            timing_metrics = compute_timing_metrics(batch=self.active_batch, timing_raw=timing_raw, include_detail=True)
            metrics.update(data_metrics)
            metrics.update(timing_metrics)

        # Clean up stored batch and any P2P buffer
        if hasattr(self, "_active_batch_uid") and self._active_batch_uid is not None:
            if hasattr(self, "_batch_buffer_dict") and self._active_batch_uid in self._batch_buffer_dict:
                del self._batch_buffer_dict[self._active_batch_uid]
            self._active_batch_uid = None

        if hasattr(self, "active_batch"):
            del self.active_batch

        return metrics
