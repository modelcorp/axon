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

"""Asynchronous PPO driver.

Programs run on the sampler worker via ``AsyncSamplerMixin``. The
full process — program execution, transformation, advantage computation,
and NCCL P2P transfer to actor workers — happens on the sampler. The
controller only sends lightweight commands and coordinates training steps.

A background producer thread continuously generates data on the sampler
while the main thread runs PPO updates on the actor, overlapping sampling
and training for maximum throughput.

Requires a disaggregated engine (separate actor and sampler GPU pools).

Example usage::

    from axon.driver.async_ppo import AsyncPPO

    driver = AsyncPPO(config, tokenizer, processor, role_worker_mapping)
    driver.init_workers()
    driver.train_rl()
"""

import logging
import threading
import time
import uuid
from queue import Queue

import ray

from axon.data.data_sampler.abstract_sampler import AbstractSampler
from axon.driver.base import PPODriverBase
from axon.driver.driver_utils import ValidationResult, convert_batch_dict_to_dataproto
from axon.protocol import DataProto
from axon.utils.metrics import (
    marked_timer,
    reduce_data_metrics,
    reduce_metrics,
    reduce_timing_metrics,
)

logger = logging.getLogger(__name__)

_ADD_BATCH_MAX_RETRIES = 300  # ~30s at 0.1s interval


def _retry_add_batch(sampler_wg, batch, step, max_retries=_ADD_BATCH_MAX_RETRIES):
    """Try ``add_batch_to_engine`` with polling retry. Raises on timeout."""
    success = sampler_wg.add_batch_to_engine(batch, step)
    retries = 0
    while not success:
        retries += 1
        if retries > max_retries:
            raise RuntimeError(f"add_batch_to_engine failed after {max_retries} retries (step {step})")
        time.sleep(0.1)
        success = sampler_wg.add_batch_to_engine(batch, step)


class AsyncPPO(PPODriverBase):
    """PPO driver where programs run on the sampler worker.

    The sampler worker handles the full process: program execution,
    transformation, advantage computation, and NCCL P2P transfer to actor
    workers. The controller sends lightweight commands and coordinates
    training steps. Maximum throughput for large-scale disaggregated setups.
    """

    # ------------------------------------------------------------------
    # Mode initialization
    # ------------------------------------------------------------------

    def _init_mode(self):
        """Set up P2P channels, initialize sampler mixin, create locks."""
        assert self.disaggregated, "AsyncPPO requires disaggregated (non-hybrid) engine"
        # Worker-resident mode runs the full process on the sampler worker.
        # Critic values, ref log probs, and KL penalty require controller-side
        # enrichment and are not yet supported in this mode.
        if self.use_critic:
            raise ValueError(
                "AsyncPPO does not compute critic values on the sampler. "
                "Advantage will be computed without a critic baseline (no GAE). "
                "Use SyncPPO if critic is needed."
            )
        if self.use_ref:
            raise ValueError(
                "AsyncPPO does not compute ref log probs on the sampler. "
                "KL divergence will not be available. "
                "Use SyncPPO if ref policy is needed."
            )
        if self.use_kl_in_reward:
            raise ValueError(
                "AsyncPPO does not support kl_reward (requires ref log probs "
                "and controller-side reward shaping). Use SyncPPO instead."
            )

        # Curriculum sampler requires the training batch on the controller for
        # pass-rate updates. Worker-resident keeps the batch on the workers.
        if isinstance(getattr(self.train_dataloader, "sampler", None), AbstractSampler):
            raise ValueError(
                "AsyncPPO does not support curriculum samplers (the training batch "
                "is not available on the controller). Use SyncPPO instead."
            )

        # P2P weight transfer: actor → sampler (shared helper)
        self._init_p2p_weight_sync()

        # P2P data transfer: sampler → actor (bridge process group)
        master_address, master_port = self.actor_wg.get_node_ip_and_free_port()[0]
        tcp_address = f"tcp://{master_address}:{master_port}"
        self.actor_wg.connect_trainer_to_sampler(
            self.sampler_wg,
            init_method=tcp_address,
            group_name="sampler_bridge_pg",
            group_attribute_name="sampler_bridge_pg",
        )

        # Sampler workers need actor worker references for P2P data transfer
        self.sampler_wg.set_actor_wg(self.actor_wg)

        # Initialize the sampler mixin on the sampler worker
        self.sampler_wg.initialize_sampler(
            sampler_servers=self.sampler_servers,
            server_addresses=self.server_addresses,
            config=self.config,
        )

        # Locks for producer/consumer overlap.
        # sampler_wg_lock: mutual exclusion between producer work (train →
        #   generate → advantage) and validation (eval → evaluate).
        # sampler_wg_broadcast_lock: protects P2P data transfers and
        #   checkpoint saves from overlapping.
        self.sampler_wg_lock = threading.Lock()
        self.sampler_wg_broadcast_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def _run_training_loop(self):
        """Training loop with background producer thread.

        The producer thread runs on the sampler worker and pushes
        completed batches (or NCCL P2P batch UIDs) into a queue.
        The main thread consumes from the queue, sets the batch on
        the actor, runs PPO updates, and syncs weights.

        Synchronization with validation: ``_validation_step`` acquires
        ``sampler_wg_lock``, which the producer also holds during its
        sampler work section. This provides mutual exclusion without
        extra events — if validation starts mid-iteration, it simply
        waits for the producer to finish the current iteration.
        """
        sampler_queue: Queue = Queue(maxsize=2)
        shutdown_event = threading.Event()
        start_step = self.global_steps + 1

        def producer_loop():
            step = start_step
            try:
                for _epoch in range(self.config.get("total_epochs", None) or int(1e9)):
                    for batch_dict in iter(self.train_dataloader):
                        if shutdown_event.is_set() or step > self.total_training_steps:
                            sampler_queue.put(None)
                            return

                        sample_batch = self._prepare_batch(batch_dict, step)

                        metrics, timing_raw = {}, {}

                        with self.sampler_wg_lock:
                            self.sampler_wg.train()

                            with marked_timer("gen", timing_raw):
                                with marked_timer("gen/add_batch", timing_raw):
                                    _retry_add_batch(self.sampler_wg, sample_batch, step)

                                with marked_timer("gen/programs", timing_raw):
                                    gen_metrics_ref = self.sampler_wg.generate_programs(global_steps=step)
                                    gen_metrics = ray.get(gen_metrics_ref)

                            skip = gen_metrics.pop("skip", False)
                            if skip:
                                step += 1
                                continue
                            metrics.update(gen_metrics)

                            with marked_timer("advantage", timing_raw):
                                adv_metrics = self.sampler_wg.compute_advantage_on_replay_buffer()
                                metrics.update(adv_metrics)

                        # Transfer data to actor via NCCL P2P or Ray
                        batch_uid = None
                        replay_buffer = None
                        channel = self.config.get("sampler_channel", "nccl")

                        with self.sampler_wg_broadcast_lock:
                            if channel == "ray":
                                with marked_timer("p2p_transfer", timing_raw):
                                    replay_buffer = self.sampler_wg.get_replay_buffer()
                            elif channel == "nccl":
                                dispatch_info = self.actor_wg._query_dispatch_info("trainer")
                                if "trainer" not in self.actor_wg._dispatch_info:
                                    self.actor_wg._dispatch_info["trainer"] = dispatch_info
                                batch_uid = str(uuid.uuid4())
                                with marked_timer("p2p_transfer", timing_raw):
                                    self.sampler_wg.send_batch_to_actor(dispatch_info, batch_uid)
                            else:
                                raise Exception(f"Unsupported channel: {channel}")

                        scheduled_temp = sample_batch.meta_info.get("sample_params", {}).get("temperature")
                        sampler_queue.put(
                            {
                                "step": step,
                                "batch": replay_buffer,
                                "batch_uid": batch_uid,
                                "metrics": metrics,
                                "timing_raw": timing_raw,
                                "temperature": scheduled_temp
                                if scheduled_temp is not None
                                else self.config.decoding.temperature,
                            }
                        )
                        step += 1
            except Exception as e:
                logger.exception("AsyncPPO producer thread failed")
                sampler_queue.put(e)
            else:
                sampler_queue.put(None)

        producer_thread = threading.Thread(
            target=producer_loop,
            name="AsyncPPO-Producer",
            daemon=False,
        )
        producer_thread.start()

        # Main thread: consume from queue and run training
        consumer_step_start = time.time()
        while True:
            queue_wait_start = time.time()
            item = sampler_queue.get()
            if item is None:
                break
            if isinstance(item, Exception):
                raise item

            self.global_steps = item["step"]
            if self.global_steps > self.total_training_steps:
                shutdown_event.set()
                break

            metrics = item["metrics"]
            timing_raw = item["timing_raw"]

            # Overlap timing
            timing_raw["queue_wait"] = time.time() - queue_wait_start
            timing_raw["step_interval"] = time.time() - consumer_step_start
            consumer_step_start = time.time()
            self._current_temperature = item["temperature"]
            channel = self.config.get("sampler_channel", "nccl")

            # Set batch on actor
            with marked_timer("set_batch", timing_raw):
                if channel == "ray":
                    self.actor_wg.set_batch(item["batch"])
                elif channel == "nccl":
                    self.actor_wg.set_batch_from_uid(item["batch_uid"])

            # Train on actor
            self._train_on_actor(timing_raw, metrics)

            # Weight sync
            with marked_timer("weight_sync", timing_raw):
                with marked_timer("weight_sync/pause", timing_raw):
                    self.sampler_wg.pause_all()
                with marked_timer("weight_sync/broadcast", timing_raw):
                    self._broadcast_weights()
                with marked_timer("weight_sync/resume", timing_raw):
                    self.sampler_wg.continue_all()

            self._post_step(metrics, timing_raw)

        shutdown_event.set()
        producer_thread.join(timeout=10.0)
        if producer_thread.is_alive():
            logger.warning("Producer thread did not shut down within timeout")

    def _train_on_actor(self, timing_raw: dict, metrics: dict):
        """Issue training commands to the actor worker.

        Operates on whatever data is in the actor's ``active_batch``.

        Steps:
            1. Load model, compute old log probs, offload
            2. Run PPO update (load model+optimizer, forward-backward + optim step, offload)
            3. Collect batch metrics and clear the active batch
        """
        temperature = getattr(self, "_current_temperature", self.config.decoding.temperature)

        # Compute old log probs
        with marked_timer("old_log_prob", timing_raw):
            self.actor_wg.load_model(include_model=True, include_optimizer=False)
            if self.config.use_sampler_logprobs:
                logprob_result = self.actor_wg.use_sampler_log_probs(temperature)
                metrics.update(self._collect_metrics(logprob_result))
            else:
                logprob_result = self.actor_wg.compute_log_prob_on_batch(temperature)
                metrics.update(self._collect_metrics(logprob_result))
            self.actor_wg.offload_model(include_model=True, include_optimizer=False)

        # PPO update (forward-backward + optim step, includes load/offload)
        if self.config.critic_warmup <= self.global_steps:
            with marked_timer("forward_backward", timing_raw):
                actor_output_ref = self.actor_wg.train_on_batch(
                    loss_fn=self.config.loss,
                    loss_fn_args=self.config.loss_args,
                    epochs=self.config.ppo_epochs,
                    mini_batch_size=self.config.mini_batch_size * self.config.decoding.n,
                    mini_batch_shuffle=self.config.get("mini_batch_shuffle", False),
                    mini_batch_seed=self.config.get("mini_batch_seed", None),
                )
                if hasattr(actor_output_ref, "__iter__") and not isinstance(actor_output_ref, DataProto):
                    actor_output_list = ray.get(actor_output_ref)
                    actor_output = DataProto.concat(actor_output_list)
                else:
                    actor_output = actor_output_ref
                if hasattr(actor_output, "meta_info") and "metrics" in actor_output.meta_info:
                    actor_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                    metrics.update(actor_metrics)

        # Collect batch metrics and clear active_batch
        batch_metrics = self.actor_wg.get_batch_metrics_and_clear(timing_raw)
        metrics.update(self._collect_metrics(batch_metrics, reduce_data_metrics, reduce_timing_metrics))

    # ------------------------------------------------------------------
    # Validation and checkpointing hooks
    # ------------------------------------------------------------------

    def _save_state(self):
        """Save state with broadcast lock to avoid P2P conflicts."""
        with self.sampler_wg_broadcast_lock:
            super()._save_state()

    def _validation_step(self) -> dict:
        """Run validation by delegating to the sampler worker.

        Acquires ``sampler_wg_lock`` to ensure the producer thread is not
        mid-iteration on the sampler. If the producer holds the lock,
        validation blocks until the current iteration completes.
        """
        if self.val_dataloader is None:
            return {}

        result = ValidationResult(n_samples=self.config.validation.decoding.n)

        with self.sampler_wg_lock:
            self.sampler_wg.eval()

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

                _retry_add_batch(self.sampler_wg, test_batch, self.global_steps)

                eval_results_ref = self.sampler_wg.evaluate_programs()
                eval_results = ray.get(eval_results_ref)

                for r in eval_results:
                    result.add(
                        reward=r["reward"],
                        uid=r["group_id"],
                        data_source=r.get("data_source", "unknown"),
                    )

        return result.compute_metrics()
