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

"""AsyncSamplerMixin: run sampling + transformation on sampler workers.

In worker-resident mode, the full process — program execution, transformation
to training data, advantage computation, and transfer to actor workers — runs
on the sampler worker. The controller only sends lightweight commands.

This mixin is added to sampler worker classes (FSDP or Megatron).

The P2P transfer uses a two-phase protocol:
1. Send byte count (int64 tensor) so receiver can allocate buffer
2. Send serialized DataProto (uint8 tensor)
"""

import logging
import os

import ray
import torch
from torch.distributed import P2POp, batch_isend_irecv
from transformers import AutoProcessor, AutoTokenizer, ProcessorMixin

from axon.controller.decorator import Dispatch, Execute, register
from axon.driver.components.advantage_component import compute_advantage_component
from axon.driver.components.program_component import create_program_components
from axon.driver.driver_utils import balance_batch, pad_dataproto_to_world_size
from axon.protocol import DataProto
from axon.utils.memory_utils import aggressive_empty_cache
from axon.utils.ray.collective import encode_object_to_cuda_tensor
from axon.utils.torch import get_device_id, get_device_name

logger = logging.getLogger(__name__)


class AsyncSamplerMixin:
    """Mixin providing sampling + transformation on sampler workers in async mode.

    Used in worker-resident mode to run the entire data preparation process
    on the sampler worker, avoiding controller-mediated data transfers.

    Expects the host class to provide:
    - self.world_size  (total world size of the sampler worker group)
    - self.rank  (rank of this worker)
    - self.sampler_bridge_pg  (NCCL process group for P2P to actors)
    - self.actor_world_size  (number of actor workers for P2P)
    - self.actor_workers  (list of Ray actor handles for triggering receives)
    """

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    @register(dispatch_mode=Dispatch.RANK_ZERO, execute_mode=Execute.RANK_ZERO, blocking=True, disable_collective=True)
    def initialize_sampler(
        self,
        sampler_servers,
        server_addresses,
        config,
    ):
        """Initialize the sampler mixin on this worker.

        Loads tokenizer/processor, then uses the shared
        ``create_program_components()`` factory to create all
        components (SamplingClient, Engine, ProgramProcessor,
        ProgramRunner, event loop).

        Note:
            The ``config`` parameter is the **full** training config, stored
            as ``self.full_config``. The host worker's ``self.config`` is the
            sampler sub-config (e.g. ``config.sampler``). Both coexist on
            the same instance: ``full_config`` for training parameters
            (loss, advantage, model path, etc.) and ``self.config`` for
            sampler-specific settings (TP size, engine, decoding, etc.).

        Args:
            sampler_servers: List of sampler server handles for the SamplingClient.
            server_addresses: List of sampler server HTTP addresses.
            config: Full training config (Hydra/OmegaConf).
        """
        self.full_config = config

        if self.full_config.moe_replay:
            actor_config = self.full_config.actor
            assert actor_config.strategy == "megatron", "MOE replay is only supported with megatron actor"
            assert actor_config.megatron.virtual_pipeline_model_parallel_size is None, (
                "VPP not supported with MOE replay"
            )

        # Load tokenizer and optional multimodal processor
        tokenizer_path = self.full_config.model_path
        trust_remote_code = config.get("trust_remote_code", True)
        self.tokenizer = AutoTokenizer.from_pretrained(  # nosec B615
            tokenizer_path,
            revision=os.environ.get("HF_HUB_REVISION"),
            trust_remote_code=trust_remote_code,
        )
        try:
            self.processor = AutoProcessor.from_pretrained(  # nosec B615
                tokenizer_path, trust_remote_code=trust_remote_code, use_fast=True
            )
            if "Processor" not in self.processor.__class__.__name__ or not isinstance(self.processor, ProcessorMixin):
                logger.info(
                    "Setting processor to None because it looks like a tokenizer. "
                    "This is normal for LLM but an error if multimodal processing is intended."
                )
                self.processor = None
        except Exception as e:
            logger.warning(f"Failed to create processor: {e}. This may affect multimodal processing.")
            self.processor = None

        # Create all pipeline components via the shared factory
        components = create_program_components(
            config=config,
            tokenizer=self.tokenizer,
            processor=self.processor,
            server_addresses=server_addresses,
            sampler_servers=sampler_servers,
            thread_name_prefix="SamplerPipeline",
        )
        self.sampling_client = components.sampling_client
        self.engine = components.engine
        self.program_processor = components.program_processor
        self.program_runner = components.program_runner
        self._tasks_loop = components.tasks_loop

        # Experiment directory for saving program outputs
        output_dir = self.full_config.output_dir
        if not os.path.isabs(output_dir):
            output_dir = os.path.join(os.getcwd(), output_dir)
        self.experiment_dir = os.path.join(
            output_dir, self.full_config.project_name, self.full_config.experiment_name, "samplers"
        )
        os.makedirs(self.experiment_dir, exist_ok=True)

        self.replay_buffer = None

    # ------------------------------------------------------------------
    # Mode switching
    # ------------------------------------------------------------------

    @register(dispatch_mode=Dispatch.RANK_ZERO, execute_mode=Execute.RANK_ZERO, blocking=True, disable_collective=True)
    def train(self):
        """Switch the execution engine to training mode."""
        self.engine.train()

    @register(dispatch_mode=Dispatch.RANK_ZERO, execute_mode=Execute.RANK_ZERO, blocking=True, disable_collective=True)
    def eval(self):
        """Switch the execution engine to evaluation mode."""
        self.engine.eval()

    @register(dispatch_mode=Dispatch.RANK_ZERO, execute_mode=Execute.RANK_ZERO, blocking=True, disable_collective=True)
    async def pause_all(self):
        """Pause all sampler servers (for weight sync)."""
        await self.sampling_client.pause_all()

    @register(dispatch_mode=Dispatch.RANK_ZERO, execute_mode=Execute.RANK_ZERO, blocking=True, disable_collective=True)
    async def continue_all(self):
        """Resume all sampler servers after weight sync."""
        await self.sampling_client.continue_all()

    # ------------------------------------------------------------------
    # Program creation and execution
    # ------------------------------------------------------------------

    @register(dispatch_mode=Dispatch.RANK_ZERO, execute_mode=Execute.RANK_ZERO, blocking=True, disable_collective=True)
    def add_batch_to_engine(self, batch: DataProto, global_steps: int):
        """Queue programs for execution from a DataProto batch.

        Creates program instances via ProgramRunner and stores them for
        later execution in generate_programs(). Checks sampler capacity
        before accepting.

        Args:
            batch: DataProto with non_tensor_batch containing "env_args" and "uid".
            global_steps: Current training step.

        Returns:
            True if the batch was accepted, False if sampler is at capacity.
        """
        max_queue = getattr(self.full_config, "max_sampler_queue_size", 1e9)
        if sum(self.sampling_client._usage.values()) < max_queue:
            _, self._uid_to_index = self.program_runner.create_programs(batch, global_steps)
            self.original_batch = batch
            return True
        else:
            return False

    @register(dispatch_mode=Dispatch.RANK_ZERO, execute_mode=Execute.RANK_ZERO, blocking=False, disable_collective=True)
    async def generate_programs(self, global_steps: int = -1):
        """Run programs, transform results, and store as replay_buffer.

        Executes all queued programs, waits for completion, transforms
        finished programs into a DataProto training batch via ProgramProcessor,
        and stores the result as ``self.replay_buffer``.

        Args:
            global_steps: Current training step.

        Returns:
            Dict of engine and transformation metrics. Contains "skip": True
            if no trainable programs were produced.
        """
        if self.full_config.use_dummy_batch:
            batch = self.program_processor.create_dummy_batch(self.original_batch, global_steps=global_steps)
            self.replay_buffer = batch
            return {}

        finished_programs, engine_metrics = await self.program_runner.run_and_collect()

        if not finished_programs:
            self.replay_buffer = None
            return {"skip": True}

        filter_errors = self.full_config.filter_program_errors
        programs = [p for p in finished_programs if p.is_trainable(strict=filter_errors)]
        engine_metrics["engine/training_programs"] = len(programs)

        if not programs:
            self.replay_buffer = None
            return {"skip": True}

        batch, transform_metrics = self.program_processor.transform_programs(
            programs,
            experiment_dir=self.experiment_dir,
            global_steps=global_steps,
            uid_to_index=getattr(self, "_uid_to_index", None),
        )

        metrics = {}
        metrics.update(engine_metrics)
        metrics.update(transform_metrics)

        self.replay_buffer = batch
        metrics["skip"] = self.replay_buffer is None
        return metrics

    @register(dispatch_mode=Dispatch.RANK_ZERO, execute_mode=Execute.RANK_ZERO, blocking=False, disable_collective=True)
    async def evaluate_programs(self, global_steps: int = 0):
        """Run validation programs and return evaluation results.

        Args:
            global_steps: Current training step, supplied by the driver (the owner
                of the step) and used to tag saved validation programs. 0 for
                before-train validation (validation.before_train=True).

        Returns:
            List of dicts with "reward", "data_source", and "group_id" per program.
        """
        finished_programs, _ = await self.program_runner.run_and_collect(val=True)

        eval_results = []
        for p in finished_programs:
            r = p.reward
            metadata = p.metadata
            if "raw_score" in metadata:
                r = metadata["raw_score"]
            eval_results.append(
                {
                    "reward": r,
                    "data_source": getattr(p, "data_source", "unknown"),
                    "group_id": p.group_id,
                }
            )

        if self.full_config.save_programs_flag:
            transformed = [self.program_processor.transform_single_program(p) for p in finished_programs]
            self.program_processor.save_programs(transformed, self.experiment_dir, global_steps, validation=True)

        return eval_results

    # ------------------------------------------------------------------
    # Advantage computation and data preparation
    # ------------------------------------------------------------------

    @register(dispatch_mode=Dispatch.RANK_ZERO, execute_mode=Execute.RANK_ZERO, blocking=True, disable_collective=True)
    def compute_advantage_on_replay_buffer(self):
        """Compute advantages, pad, and balance for DP distribution.

        After this call, the replay buffer is ready for P2P transfer to
        actor workers.

        Returns:
            Dict of metrics (pass rates, reward stats, balance stats).
        """
        metrics = {}

        # 1. Compute advantages (pure: no padding/balancing).
        self.replay_buffer = compute_advantage_component(
            batch=self.replay_buffer,
            config=self.full_config,
            metrics=metrics,
        )

        # 2. Pad and balance for DP distribution (program-aware).
        world_sizes = [self.world_size, self.actor_world_size]
        self.replay_buffer = pad_dataproto_to_world_size(self.replay_buffer, world_sizes)
        self.replay_buffer = balance_batch(
            self.replay_buffer,
            world_size=self.actor_world_size,
            metrics=metrics,
        )

        return metrics

    # ------------------------------------------------------------------
    # Data retrieval
    # ------------------------------------------------------------------

    @register(dispatch_mode=Dispatch.RANK_ZERO, execute_mode=Execute.RANK_ZERO, blocking=True, disable_collective=True)
    def get_replay_buffer(self):
        """Return the latest replay buffer.

        Used by the controller to pull the batch for inspection or when
        using a non-P2P transfer mode.

        Returns:
            The DataProto replay buffer.
        """
        assert hasattr(self, "replay_buffer") and self.replay_buffer is not None, (
            "Replay buffer not set; generate_programs() not called yet."
        )
        return self.replay_buffer

    # ------------------------------------------------------------------
    # P2P transfer to actor workers
    # ------------------------------------------------------------------

    @register(dispatch_mode=Dispatch.RANK_ZERO, execute_mode=Execute.RANK_ZERO, blocking=True, disable_collective=True)
    def send_batch_to_actor(self, dp_rank_mapping, batch_uid):
        """Send the replay buffer to actor workers via NCCL P2P.

        Uses a two-phase protocol per actor rank:
        1. Send the byte count of the serialized payload (int64)
        2. Send the serialized DataProto payload (uint8)

        The replay buffer is split by data-parallel rank so each actor
        gets its correct shard.

        Args:
            dp_rank_mapping: List mapping actor_rank -> dp_rank (for chunking).
            batch_uid: Unique identifier for this batch transfer, used by the
                actor's receive_batch_from_sampler to tag the received data.
        """
        assert hasattr(self, "sampler_bridge_pg"), "Bridge process group not initialized for P2P"

        aggressive_empty_cache(force_sync=True)

        # Launch async receives on all actor workers
        recv_refs = None
        if self.rank == 0:
            recv_refs = [worker.receive_batch_from_sampler.remote(batch_uid) for worker in self.actor_workers]

        # Split replay buffer into chunks, one per DP rank
        dp_size = max(dp_rank_mapping) + 1
        all_data = self.replay_buffer.chunk(chunks=dp_size)

        for data in all_data:
            for key in data.batch.keys():
                if isinstance(data.batch[key], torch.Tensor):
                    data.batch[key] = data.batch[key].contiguous()

        device = torch.device(get_device_name(), get_device_id())

        # Serialize each DP chunk ONCE and let actor ranks that share a DP
        # rank reuse the same GPU payload tensor. NCCL isend reads the
        # buffer read-only, so multiple sends off the same tensor are safe
        # and avoid an O(actor_world_size) blow-up on the sampler GPU —
        # which otherwise OOMs when actor_world_size > dp_size (common
        # when actor TP > actor DP).
        dp_to_payload: dict[int, torch.Tensor] = {}
        dp_to_size: dict[int, torch.Tensor] = {}
        rank_to_data = {}
        rank_to_data_size = {}
        for actor_rank in range(self.actor_world_size):
            local_dp_rank = dp_rank_mapping[actor_rank]
            if local_dp_rank not in dp_to_payload:
                payload_gpu = encode_object_to_cuda_tensor(all_data[local_dp_rank], device)
                dp_to_payload[local_dp_rank] = payload_gpu
                dp_to_size[local_dp_rank] = torch.tensor([payload_gpu.numel()], dtype=torch.int64, device=device)
            rank_to_data[actor_rank] = dp_to_payload[local_dp_rank]
            rank_to_data_size[actor_rank] = dp_to_size[local_dp_rank]

        # Phase 1: Send byte counts to all actor ranks
        size_ops = []
        for actor_rank, size_tensor in rank_to_data_size.items():
            size_ops.append(
                P2POp(
                    op=torch.distributed.isend,
                    tensor=size_tensor,
                    peer=actor_rank,
                    group=self.sampler_bridge_pg,
                )
            )
        if size_ops:
            torch.cuda.synchronize()
            size_reqs = batch_isend_irecv(size_ops)
            for req in size_reqs:
                req.wait()

        # Phase 2: Send serialized payloads to all actor ranks
        data_ops = []
        for actor_rank, data_tensor in rank_to_data.items():
            data_ops.append(
                P2POp(
                    op=torch.distributed.isend,
                    tensor=data_tensor.contiguous(),
                    peer=actor_rank,
                    group=self.sampler_bridge_pg,
                )
            )
        if data_ops:
            torch.cuda.synchronize()
            data_reqs = batch_isend_irecv(data_ops)
            for req in data_reqs:
                req.wait()

        # Wait for all actor workers to finish receiving
        if self.rank == 0 and recv_refs is not None:
            ray.get(recv_refs)
        del rank_to_data
        del rank_to_data_size
        del dp_to_payload
        del dp_to_size

        aggressive_empty_cache(force_sync=True)
