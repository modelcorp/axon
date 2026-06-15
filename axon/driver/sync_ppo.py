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

"""Synchronous PPO trainer.

Programs run on the controller process. Training data is visible between
steps, enabling critic value computation, reference policy log probs,
reward model scoring, KL penalty shaping, and any custom enrichment
before the batch is sent to the actor.

This trainer demonstrates the Tinker :class:`~axon.tinker.TrainingClient`
API for the actor training step, showing how ``forward()``,
``forward_backward()``, and ``optim_step()`` compose into a PPO update.

Works with both hybrid engine (actor + sampler on the same GPUs) and
disaggregated engine (separate GPU pools).

Example usage::

    from axon.driver.sync_ppo import SyncPPO

    driver = SyncPPO(config, tokenizer, processor, role_worker_mapping)
    driver.init_workers()
    driver.train_rl()
"""

import asyncio
import logging
import threading
from queue import Queue

import torch

from axon.data.data_sampler.abstract_sampler import AbstractSampler
from axon.driver.base import PPODriverBase
from axon.driver.components.advantage_component import compute_advantage_component
from axon.driver.components.program_component import create_program_components
from axon.driver.driver_utils import (
    ValidationResult,
    balance_batch,
    convert_batch_dict_to_dataproto,
    pad_dataproto_to_world_size,
    update_trainer,
)
from axon.protocol import DataProto
from axon.tinker import TrainingClient
from axon.utils.metrics import (
    aggregate_trainer_sampler_metrics,
    compute_trainer_sampler_mismatch_metrics,
    marked_timer,
)
from axon.utils.metrics.data_metrics import compute_data_metrics
from axon.utils.metrics.repetition import compute_repetition_metrics
from axon.utils.metrics.timing import compute_timing_metrics
from axon.utils.print_utils import colorful_print

logger = logging.getLogger(__name__)


class SyncPPO(PPODriverBase):
    """PPO driver where programs run on the controller process.

    Data is visible between steps, supporting critic value computation,
    reference policy log probs, reward model scoring, and custom reward
    shaping — all before sending the enriched batch to the actor for a
    single training transfer.

    Best for:
        - Hybrid engine (shared actor + sampler GPUs)
        - Workflows that inspect or modify data between stages
        - Debugging and development (data is locally accessible)
    """

    # ------------------------------------------------------------------
    # Mode initialization
    # ------------------------------------------------------------------

    def _init_mode(self):
        """Set up program components and TrainingClient for local program execution."""

        if self._use_dummy_batch and not self._memory_stress_test:
            # Actor-only mode: only need ProgramProcessor for create_dummy_batch()
            from axon.driver.components.program_component import ProgramProcessor, ProgramTransformConfig

            transform_config = ProgramTransformConfig.from_config(self.config)
            self.program_processor = ProgramProcessor(transform_config, self.tokenizer, self.processor)
        else:
            components = create_program_components(
                config=self.config,
                tokenizer=self.tokenizer,
                processor=self.processor,
                server_addresses=self.server_addresses,
                sampler_servers=self.sampler_servers,
                hybrid_engine=self.hybrid_engine,
                thread_name_prefix="ControllerResident",
            )
            self.sampling_client = components.sampling_client
            self.engine = components.engine
            self.program_processor = components.program_processor
            self.program_runner = components.program_runner
            self._tasks_loop = components.tasks_loop

        if self._eval_only:
            return

        # Initially put replicas to sleep if configured
        if not self._use_dummy_batch and self.hybrid_engine and self.config.sampler.offload_sampler:
            asyncio.run(self.sampling_client.sleep())

        # Tinker TrainingClient wraps the actor worker group with a
        # higher-level API: forward(), forward_backward(), optim_step().
        self.actor_client = TrainingClient(
            worker_group=self.actor_wg,
            config=self.config,
            tokenizer=self.tokenizer,
            processor=self.processor,
        )
        self.actor_world_size = self.actor_wg.world_size

        self.critic_client = None if not self.use_critic else TrainingClient(worker_group=self.critic_wg)
        self.rm_client = None if not self.use_rm else TrainingClient(worker_group=self.rm_wg)

        # P2P weight transfer for disaggregated mode (actor → sampler).
        if self.disaggregated and not self._use_dummy_batch:
            self._init_p2p_weight_sync()

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def _run_training_loop(self):
        """Select overlapped or sequential loop based on engine mode and config."""
        if self.disaggregated and self.config.get("enable_one_off_pipeline", False):
            self._run_training_loop_overlapped()
        else:
            self._run_training_loop_sequential()

    def _run_training_loop_sequential(self):
        """Sequential training loop: sample → enrich → train → sync."""
        self.global_steps += 1

        for step_idx in range(self.total_training_steps - self.global_steps + 1):
            if self.steps_per_epoch is not None and step_idx % self.steps_per_epoch == 0:
                epoch = step_idx // self.steps_per_epoch
                colorful_print(f"[Epoch {epoch}, Step {self.global_steps}]", "green")

            batch_dict = self.train_dataloader.next(batch_size=self.train_batch_size)
            batch = self._prepare_batch(batch_dict, self.global_steps)
            self._current_temperature = batch.meta_info.get("sample_params", {}).get(
                "temperature", self.config.decoding.temperature
            )

            metrics, timing_raw = {}, {}

            with marked_timer("step", timing_raw):
                train_batch = self._sample_and_prepare(batch, timing_raw, metrics)
                if train_batch is None:
                    self.global_steps += 1
                    continue

                self._train_on_actor(train_batch, timing_raw, metrics)

                if self.disaggregated and not self._use_dummy_batch:
                    with marked_timer("broadcast_weights", timing_raw):
                        self._broadcast_weights()

            self._collect_batch_metrics(train_batch, metrics, timing_raw)
            self._post_step(metrics, timing_raw)

    def _run_training_loop_overlapped(self):
        """Overlapped training loop: pre-fetch next batch while training current.

        In disaggregated mode, overlaps sampling of the next batch with
        training on the current batch. Uses a background thread and a
        queue of size 1 for overlap.

        Timing::

            Thread   Step N              Step N+1
            ------   ------------------  ------------------
            BG       [sample & prepare]
            Main     wait...             [train on actor]
            BG                           [sample & prepare]
            Main                         wait...
        """
        self.global_steps += 1
        pipeline_queue: Queue = Queue(maxsize=1)

        # Pre-fetch first batch
        first_batch_dict = self.train_dataloader.next(batch_size=self.train_batch_size)
        first_batch = self._prepare_batch(first_batch_dict, self.global_steps)

        warmup_thread = threading.Thread(
            target=self._sample_and_prepare_into_queue,
            args=(first_batch, pipeline_queue),
            name="ControllerResident-Warmup",
            daemon=True,
        )
        warmup_thread.start()

        for step_idx in range(self.total_training_steps - self.global_steps + 1):
            if self.steps_per_epoch is not None and step_idx % self.steps_per_epoch == 0:
                epoch = step_idx // self.steps_per_epoch
                colorful_print(f"[Epoch {epoch}, Step {self.global_steps}]", "green")

            # Spawn thread for NEXT batch (overlaps with training current)
            next_step = self.global_steps + 1
            if next_step <= self.total_training_steps:
                next_batch_dict = self.train_dataloader.next(batch_size=self.train_batch_size)
                next_batch = self._prepare_batch(next_batch_dict, next_step)
                next_thread = threading.Thread(
                    target=self._sample_and_prepare_into_queue,
                    args=(next_batch, pipeline_queue),
                    name="ControllerResident-Pipeline",
                    daemon=True,
                )
            else:
                next_thread = None

            # Get current batch from queue (blocks until warmup/previous finishes)
            item = pipeline_queue.get()
            if isinstance(item, Exception):
                raise item
            if item is None:
                # Skipped batch (no trainable programs)
                if next_thread is not None:
                    next_thread.start()
                self.global_steps += 1
                continue

            train_batch, sample_metrics, sample_timing = item
            metrics = dict(sample_metrics)
            timing_raw = dict(sample_timing)

            # Start next batch's sampling in the background
            if next_thread is not None:
                next_thread.start()

            self._train_on_actor(train_batch, timing_raw, metrics)

            # Wait for sampling to finish before weight sync — the sampler
            # must be idle during P2P weight transfer (NCCL collective).
            if next_thread is not None:
                next_thread.join()

            if not self._use_dummy_batch:
                with marked_timer("broadcast_weights", timing_raw):
                    self._broadcast_weights()

            self._collect_batch_metrics(train_batch, metrics, timing_raw)
            self._post_step(metrics, timing_raw)

    def _sample_and_prepare_into_queue(self, batch: DataProto, queue: Queue):
        """Run _sample_and_prepare and put the result into the queue.

        Used by the overlapped training loop to run sampling in a
        background thread. Puts None if the batch should be skipped.
        """
        try:
            metrics, timing_raw = {}, {}
            train_batch = self._sample_and_prepare(batch, timing_raw, metrics)
            if train_batch is None:
                queue.put(None)
            else:
                queue.put((train_batch, metrics, timing_raw))
        except Exception as e:
            logger.exception("Background sample_and_prepare failed")
            queue.put(e)

    # ------------------------------------------------------------------
    # Sampling and data preparation
    # ------------------------------------------------------------------

    def _sample_and_prepare(self, batch: DataProto, timing_raw: dict, metrics: dict) -> DataProto | None:
        """Run programs locally, enrich data, compute advantages.

        Steps:
            1. Execute agent programs on the controller
            2. Transform programs into a tokenized DataProto batch
            3. Compute critic values (if enabled)
            4. Compute reference policy log probs (if enabled)
            5. Compute reward model scores (if enabled)
            6. Apply KL penalty to rewards (if configured)
            7. Run through the advantage computation components
            8. Update critic (before actor, since actor may offload)

        Args:
            batch: DataProto batch from the dataloader.
            timing_raw: Dict to collect timing measurements.
            metrics: Dict to collect metrics.

        Returns:
            The enriched training batch, or None if this step should be
            skipped (no trainable programs).
        """
        # Use dummy batch if configured (for testing)
        if self.config.use_dummy_batch:
            train_batch = self.program_processor.create_dummy_batch(batch, global_steps=self.global_steps)
        else:
            with marked_timer("collect_programs", timing_raw):
                _, uid_to_index = self.program_runner.create_programs(batch, self.global_steps)
                if self.hybrid_engine:
                    self.engine.train()
                finished_programs, engine_metrics = asyncio.run(self.program_runner.run_and_collect())
                if not finished_programs:
                    return None
                programs = [p for p in finished_programs if p.is_trainable(strict=self.config.filter_program_errors)]
                engine_metrics["engine/training_programs"] = len(programs)
                if not programs:
                    return None
                metrics.update(engine_metrics)

            with marked_timer("transform_programs", timing_raw):
                train_batch, transform_metrics = self.program_processor.transform_programs(
                    programs,
                    experiment_dir=self.experiment_dir,
                    global_steps=self.global_steps,
                    uid_to_index=uid_to_index,
                )
                metrics.update(transform_metrics)

        # Prepare batch metadata
        train_batch.meta_info["temperature"] = getattr(self, "_current_temperature", self.config.decoding.temperature)
        train_batch.meta_info["global_token_num"] = torch.sum(train_batch.batch["attention_mask"], dim=-1).tolist()

        # Critic values
        if self.use_critic:
            with marked_timer("values", timing_raw):
                values = self.critic_client.forward(train_batch)
                train_batch = train_batch.union(values)

        if self.config.use_sampler_logprobs or self.use_kl_in_reward:
            train_batch.batch["old_log_probs"] = train_batch.batch["sampler_log_probs"]

        # Advantage computation
        with marked_timer("advantage", timing_raw):
            # Reward model scores
            if self.use_rm:
                reward_tensor = self.rm_client.forward(train_batch)
                train_batch = train_batch.union(reward_tensor)

            if self.use_kl_in_reward:
                raise NotImplementedError(
                    "kl_reward (KL penalty applied to the token-level reward before advantage "
                    "computation) is not implemented. Use loss_args.kl_coef for a KL term in the "
                    "policy loss instead."
                )

            # Advantage computation (pass rates, stepwise broadcast, filtering).
            train_batch = compute_advantage_component(train_batch, self.config, metrics)

            # Pad and balance for DP distribution.
            world_sizes = self._collect_world_sizes()
            train_batch = pad_dataproto_to_world_size(train_batch, world_sizes)
            train_batch = balance_batch(
                train_batch,
                world_size=self.actor_world_size,
                metrics=metrics,
            )

        # Update critic before sending to actor (actor may offload model after set_batch)
        if self.use_critic:
            with marked_timer("update_critic", timing_raw):
                with self.critic_client.load_model_context():
                    critic_output_metrics = update_trainer(
                        batch=train_batch,
                        training_client=self.critic_client,
                        epochs=self.config.ppo_epochs,
                        mini_batch_size=self.config.mini_batch_size * self.config.decoding.n,
                        loss_fn="value",
                        loss_fn_args={
                            "cliprange_value": self.config.loss_args.cliprange_value,
                            "token_reduce": self.config.loss_args.get("token_reduce", "sum"),
                            "batch_reduce": self.config.loss_args.get("batch_reduce", "token-mean"),
                        },
                        mini_batch_shuffle=self.config.get("mini_batch_shuffle", False),
                        mini_batch_seed=self.config.get("mini_batch_seed", None),
                        world_size=self.critic_wg.world_size,
                    )
            metrics.update(critic_output_metrics)

        # Reference policy log probs
        if self.use_ref:
            with marked_timer("ref_log_prob", timing_raw):
                if self._ref_uses_actor:
                    # FSDP LoRA: disable adapter via flag
                    train_batch.meta_info["is_lora"] = True
                result = self.ref_policy_wg.forward(train_batch)
                ref_log_prob = DataProto.from_dict(
                    tensors={"ref_log_prob": result.batch["log_probs"]},
                )
                train_batch = train_batch.union(ref_log_prob)

        return train_batch

    # ------------------------------------------------------------------
    # Actor training step (TrainingClient API)
    # ------------------------------------------------------------------

    def _train_on_actor(self, train_batch: DataProto, timing_raw: dict, metrics: dict):
        """Training step using the Tinker TrainingClient API.

        Computes old log probs, runs PPO update via ``update_trainer()``,
        and collects batch metrics.

        Args:
            train_batch: The enriched training batch (used for forward pass
                and PPO update via TrainingClient).
            timing_raw: Dict to collect timing measurements.
            metrics: Dict to collect metrics.
        """

        # 1. Old log probs
        if not self.config.use_sampler_logprobs:
            with marked_timer("old_log_prob", timing_raw):
                result = self.actor_client.forward(train_batch)
                old_log_prob = DataProto.from_dict(
                    tensors={"old_log_probs": result.batch["log_probs"]},
                )
                train_batch = train_batch.union(old_log_prob)
                metrics.update(
                    aggregate_trainer_sampler_metrics([compute_trainer_sampler_mismatch_metrics(train_batch)])
                )

        # 2. PPO update via update_trainer()
        if self.config.critic_warmup <= self.global_steps:
            with marked_timer("forward_backward", timing_raw):
                with self.actor_client.load_model_context():
                    actor_metrics = update_trainer(
                        batch=train_batch,
                        training_client=self.actor_client,
                        loss_fn=self.config.loss,
                        loss_fn_args=self.config.loss_args,
                        epochs=self.config.ppo_epochs,
                        mini_batch_size=self.config.mini_batch_size * self.config.decoding.n,
                        mini_batch_shuffle=self.config.get("mini_batch_shuffle", False),
                        mini_batch_seed=self.config.get("mini_batch_seed", None),
                        world_size=self.actor_world_size,
                    )
            metrics.update(actor_metrics)
            # Free GPU memory after actor update — the batch is no longer
            # needed on device (metrics and curriculum use CPU-side fields).
            train_batch = train_batch.to("cpu")

    # ------------------------------------------------------------------
    # Batch metrics and curriculum sampler (controller has the batch)
    # ------------------------------------------------------------------

    def _collect_batch_metrics(self, batch: DataProto, metrics: dict, timing_raw: dict):
        """Collect data/timing metrics from the training batch and update the curriculum sampler.

        Called by the training loop after training, before ``_post_step``.
        Only the controller-resident path has the batch on the controller;
        the worker-resident path collects metrics on the workers instead.
        """
        metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
        metrics.update(compute_repetition_metrics(batch=batch, tokenizer=self.tokenizer))
        metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))

        if isinstance(getattr(self.train_dataloader, "sampler", None), AbstractSampler):
            self.train_dataloader.sampler.update(batch=batch)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validation_step(self) -> dict:
        """Run validation by executing programs locally on the controller."""
        if self.val_dataloader is None or self._use_dummy_batch:
            return {}

        result = ValidationResult(n_samples=self.config.validation.decoding.n)

        self.engine.eval()

        for test_data in self.val_dataloader:
            test_batch = convert_batch_dict_to_dataproto(
                test_data,
                global_steps=self.global_steps,
                val=True,
                val_n_samples=self.config.validation.decoding.n,
            )
            test_batch.meta_info.update(
                {
                    "sample_params": {"validate": True},
                    "global_steps": self.global_steps,
                }
            )

            self.program_runner.create_programs(test_batch, self.global_steps)
            programs, _ = asyncio.run(self.program_runner.run_and_collect(val=True))

            if not programs:
                continue

            for program in programs:
                reward = program.metadata.get("raw_score", program.reward)
                result.add(
                    reward=reward,
                    uid=program.group_id,
                    data_source=getattr(program, "data_source", "unknown"),
                )

            if self.config.save_programs_flag:
                transformed = [self.program_processor.transform_single_program(p) for p in programs]
                self.program_processor.save_programs(
                    transformed,
                    self.experiment_dir,
                    self.global_steps,
                    validation=True,
                )

        return result.compute_metrics()
