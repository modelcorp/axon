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

"""Abstract base class for PPO drivers.

``PPODriverBase`` provides the shared infrastructure that every PPO
trainer needs: dataloader management, worker group creation, the common
actor training step, weight synchronization, checkpointing, and logging.

Subclasses implement three methods to define how data flows through the
system:

- :meth:`_init_mode` — initialize mode-specific components
- :meth:`_run_training_loop` — the outer training loop, can call `_post_step` for utility
- :meth:`_validation_step` — run validation

Two concrete implementations are provided:

- :class:`~axon.driver.sync_ppo.SyncPPO`:
  Programs run on the controller process. Data is visible between steps.

- :class:`~axon.driver.async_ppo.AsyncPPO`:
  Programs run on the sampler worker. Data transfers via NCCL P2P.

Usage::

    from axon.driver.sync_ppo import SyncPPO

    driver = SyncPPO(config, tokenizer, processor, role_worker_mapping)
    driver.init_workers()
    driver.train_rl()
"""

import asyncio
import logging
import os
import re
import time
import warnings
from abc import ABC, abstractmethod

import ray
import torch
from omegaconf import OmegaConf, open_dict

from axon.controller.ray import RayActorWithInitArgs, init_worker_group
from axon.core.role import Role
from axon.data import DynamicDataLoader, RLDataset, collate_rl_dataset, create_data_sampler
from axon.driver.driver_utils import RoleWorkerConfig, convert_batch_dict_to_dataproto
from axon.protocol import DataProto
from axon.sampler import get_server_class
from axon.trainer.algos.advantages import AdvantageFn
from axon.utils.metrics import aggregate_trainer_sampler_metrics
from axon.utils.p2p.routing_table import RoutingTable
from axon.utils.state import find_latest_ckpt_path
from axon.utils.temperature_scheduler import TemperatureScheduler
from axon.utils.torch import get_device_name
from axon.utils.tracking import Tracking

logger = logging.getLogger(__name__)


class PPODriverBase(ABC):
    """Abstract base for PPO training controllers.

    Provides shared infrastructure for all PPO training modes:

    - Dataloader initialization and batching
    - Worker group creation and model initialization
    - Sampler server initialization (vLLM/SGLang)
    - The common actor training step (log probs → PPO update → metrics)
    - Weight synchronization (actor → sampler)
    - Checkpointing (save/load state)
    - Post-step logging, validation dispatch, and checkpointing

    Subclass this to implement a custom data residency mode. You must
    implement three abstract methods:

    - :meth:`_init_mode` — set up mode-specific components
    - :meth:`_run_training_loop` — define the training loop structure
    - :meth:`_validation_step` — run validation

    Args:
        config: Hydra/OmegaConf training configuration.
        tokenizer: HuggingFace tokenizer.
        processor: Optional HuggingFace processor (for multimodal).
        role_worker_mapping: Dict mapping Role enums to RoleWorkerConfig.
    """

    def __init__(
        self,
        config,
        tokenizer,
        processor,
        role_worker_mapping: dict[Role, RoleWorkerConfig],
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.processor = processor
        self.role_worker_mapping = role_worker_mapping

        self.hybrid_engine = config.hybrid_engine
        self._eval_only = config.get("mode") == "eval"
        self._use_dummy_batch = config.get("use_dummy_batch", False)
        self._memory_stress_test = config.get("memory_stress_test", False)
        if not self._use_dummy_batch and not self._memory_stress_test:
            assert Role.Sampler in role_worker_mapping, f"Sampler role required, got {role_worker_mapping.keys()=}"
        if not self._eval_only:
            assert Role.Actor in role_worker_mapping, f"Actor role required, got {role_worker_mapping.keys()=}"

        self.use_ref = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.use_critic = bool(self.config.critic.enable) or self.config.advantage == AdvantageFn.GAE

        self.device_name = get_device_name()

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        self.ref_in_actor = (
            config.actor.get("lora", {}).get("rank", 0) > 0
            or config.actor.get("lora", {}).get("adapter_path") is not None
        )

        # Disaggregated mode: separate actor and sampler GPU pools
        self.disaggregated = not self.hybrid_engine

        self._init_dataloader()

        # Experiment directory
        output_dir = self.config.output_dir
        if not os.path.isabs(output_dir):
            output_dir = os.path.join(os.getcwd(), output_dir)
        self.experiment_dir = os.path.join(output_dir, self.config.project_name, self.config.experiment_name)
        os.makedirs(self.experiment_dir, exist_ok=True)

        self.tracking = Tracking(
            project_name=self.config.project_name,
            experiment_name=self.config.experiment_name,
            default_backend=self.config.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

    # ------------------------------------------------------------------
    # Abstract methods — subclasses must implement these
    # ------------------------------------------------------------------

    @abstractmethod
    def _init_mode(self) -> None:
        """Initialize mode-specific components after workers are ready.

        Called at the end of :meth:`init_workers`. Set up whatever the
        mode needs: components, P2P channels, threading, etc.
        """

    @abstractmethod
    def _run_training_loop(self) -> None:
        """Run the training loop."""

    @abstractmethod
    def _validation_step(self) -> dict:
        """Run validation and return metrics.

        Returns:
            Dict of validation metrics (pass@k, reward stats, etc.).
        """

    # ------------------------------------------------------------------
    # Dataloader initialization
    # ------------------------------------------------------------------

    def _init_dataloader(self):
        """Create train and validation dataloaders with dynamic batching."""
        self.train_dataset = RLDataset(self.config.train_files, getattr(self.config, "data_mix", None))
        self.val_dataset = RLDataset(self.config.val_files)
        self.train_sampler = create_data_sampler(self.config, self.train_dataset)

        self.train_batch_size = self.config.train_batch_size

        self.train_dataloader = DynamicDataLoader(
            dataset=self.train_dataset,
            sampler=self.train_sampler,
            num_workers=32,
            collate_fn=collate_rl_dataset,
            batch_size=self.train_batch_size,
            infinite=True,
        )
        val_len = len(self.val_dataset)
        if val_len > 0:
            self.val_dataloader = DynamicDataLoader(
                dataset=self.val_dataset,
                batch_size=val_len,
                infinite=False,
                num_workers=32,
                shuffle=True,
                collate_fn=collate_rl_dataset,
            )
        else:
            self.val_dataloader = None

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"

        print(
            f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: "
            f"{len(self.val_dataloader) if self.val_dataloader else 0}"
        )

        # Compute total training steps
        self.steps_per_epoch = None
        if self.config.total_epochs is not None:
            self.steps_per_epoch = len(self.train_dataset) // self.config.train_batch_size
            self.total_training_steps = self.steps_per_epoch * self.config.total_epochs
        elif self.config.total_training_steps is not None:
            self.total_training_steps = self.config.total_training_steps
        else:
            warnings.warn("Training steps/epochs not set, defaulting to 1e9", stacklevel=2)
            self.total_training_steps = int(1e9)
        logger.info(f"Total training steps: {self.total_training_steps}")

        # Propagate total steps to LR scheduler configs
        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor.lr_scheduler_args"):
                    self.config.actor.lr_scheduler_args.total_training_steps = self.total_training_steps
                if OmegaConf.select(self.config, "critic.lr_scheduler_args"):
                    self.config.critic.lr_scheduler_args.total_training_steps = self.total_training_steps
        except Exception as e:
            logger.warning(f"Could not set total_training_steps in config: {e}")

    def _prepare_batch(self, batch_dict: dict, global_steps: int) -> DataProto:
        """Convert a dataloader dict to DataProto with temperature scheduling.

        Shared helper used by both controller-resident and worker-resident
        trainers. Returns the scheduled temperature alongside the batch so
        callers can thread it to the right place without mutating config.

        Args:
            batch_dict: Raw dict from the dataloader.
            global_steps: Current training step (may differ from
                ``self.global_steps`` in producer-thread modes).

        Returns:
            DataProto ready for sampling. The scheduled temperature (if any)
            is available in ``batch.meta_info["sample_params"]["temperature"]``.
        """
        return convert_batch_dict_to_dataproto(
            batch_dict,
            global_steps=global_steps,
            n_samples=self.config.decoding.n,
            temperature_scheduler=self.temperature_scheduler,
            temperature_config=self.config.decoding.temperature_schedule,
        )

    @staticmethod
    def _collect_metrics(result, *aggregators) -> dict:
        """Normalize a worker group call result into a flat metrics dict.

        Worker group methods return either a list of per-rank dicts (when
        ray collective is disabled) or a single aggregated dict (when
        enabled). This helper handles both cases.

        Args:
            result: Worker group return value (list or dict).
            *aggregators: Functions to apply when result is a list.
                Each ``aggregator(result) -> dict`` is called and the
                results are merged.  When no aggregators are given,
                ``aggregate_trainer_sampler_metrics`` is used as default.
        """
        if isinstance(result, list):
            if not aggregators:
                aggregators = (aggregate_trainer_sampler_metrics,)
            merged = {}
            for agg in aggregators:
                merged.update(agg(result))
            return merged
        elif isinstance(result, dict):
            return result
        return {}

    # ------------------------------------------------------------------
    # Worker initialization
    # ------------------------------------------------------------------

    def init_workers(self):
        """Create and initialize all worker groups.

        Creates actor, sampler, critic, ref policy, and reward model worker
        groups as specified by the role_worker_mapping. After models are
        initialized, calls :meth:`_init_mode` for mode-specific setup.
        """
        unique_pools = set(cfg.resource_pool for cfg in self.role_worker_mapping.values())
        resource_pool_to_cls = {pool: {} for pool in unique_pools}

        # Register worker classes with their resource pools.
        # In hybrid-engine mode Actor and Sampler share a pool and get fused
        # automatically by init_worker_group / fuse_worker_cls.
        if Role.Actor in self.role_worker_mapping:
            actor_cfg = self.role_worker_mapping[Role.Actor]
            _actor_kwargs = dict(actor_cfg.init_kwargs)
            if actor_cfg.max_concurrency is not None:
                _actor_kwargs["max_concurrency"] = actor_cfg.max_concurrency
            resource_pool_to_cls[actor_cfg.resource_pool][Role.Actor] = RayActorWithInitArgs(
                cls=actor_cfg.cls,
                config=self.config.actor,
                **_actor_kwargs,
            )

        if Role.Sampler in self.role_worker_mapping:
            sampler_cfg = self.role_worker_mapping[Role.Sampler]
            resource_pool_to_cls[sampler_cfg.resource_pool][Role.Sampler] = RayActorWithInitArgs(
                cls=sampler_cfg.cls,
                config=self.config.sampler,
            )

        if self.use_critic:
            critic_cfg = self.role_worker_mapping[Role.Critic]
            resource_pool_to_cls[critic_cfg.resource_pool][Role.Critic] = RayActorWithInitArgs(
                cls=critic_cfg.cls,
                config=self.config.critic,
                **critic_cfg.init_kwargs,
            )

        if self.use_ref and Role.RefPolicy in self.role_worker_mapping:
            ref_cfg = self.role_worker_mapping[Role.RefPolicy]
            resource_pool_to_cls[ref_cfg.resource_pool][Role.RefPolicy] = RayActorWithInitArgs(
                ref_cfg.cls,
                config=self.config.ref,
                **ref_cfg.init_kwargs,
            )

        if self.use_rm:
            rm_cfg = self.role_worker_mapping[Role.RewardModel]
            resource_pool_to_cls[rm_cfg.resource_pool][Role.RewardModel] = RayActorWithInitArgs(
                rm_cfg.cls,
                config=self.config.reward_model,
                **rm_cfg.init_kwargs,
            )

        # Instantiate all worker groups
        all_wg = {}
        for resource_pool, class_dict in resource_pool_to_cls.items():
            wg_dict = init_worker_group(
                class_dict,
                resource_pool,
                enable_ray_collective=self.config.enable_ray_collective,
                device_name=self.device_name,
            )
            all_wg.update(wg_dict)

        if self.config.enable_ray_collective:
            for wg in all_wg.values():
                wg.init_ray_collective(group_name=wg._ray_collective_group_name)

        # Initialize models
        if self.use_critic:
            self.critic_wg = all_wg[Role.Critic]
            self.critic_wg.init_model()

        strategy = self.config.actor.strategy
        self._ref_uses_actor = self.ref_in_actor and strategy in ("fsdp", "fsdp2")

        if self.use_ref and not self._ref_uses_actor:
            assert Role.RefPolicy in all_wg, f"RefPolicy role missing: {all_wg.keys()=}"
            self.ref_policy_wg = all_wg[Role.RefPolicy]
            self.ref_policy_wg.init_model()

        self.rm_wg = None
        if self.use_rm:
            self.rm_wg = all_wg[Role.RewardModel]
            self.rm_wg.init_model()

        self.actor_wg = all_wg.get(Role.Actor, None)
        self.sampler_wg = all_wg.get(Role.Sampler, None)
        # Eval mode: only sampler, then early return.
        if self._eval_only:
            self.sampler_wg.init_model()
        elif self._use_dummy_batch:
            # Actor-only mode: skip sampler entirely
            self.actor_wg.init_model()
        elif self.hybrid_engine:
            self.actor_wg.init_model()
            self.sampler_wg.init_model()
        else:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=2) as pool:
                actor_future = pool.submit(self.actor_wg.init_model)
                sampler_future = pool.submit(self.sampler_wg.init_model)
                actor_future.result()
                sampler_future.result()

        if self._ref_uses_actor:
            self.ref_policy_wg = self.actor_wg

        # Initialize sampler servers (vLLM/SGLang) — skip when using dummy batches
        if self.sampler_wg is not None and not self._use_dummy_batch:
            self.sampler_servers, self.server_addresses = self._initialize_sampler_servers(self.sampler_wg)
        else:
            self.sampler_servers, self.server_addresses = [], []

        # Temperature scheduler
        self.temperature_scheduler = TemperatureScheduler(self.config.decoding.temperature_schedule)

        # KL penalty controller
        self.use_kl_in_reward = self.config.kl_reward is not None
        if self.use_kl_in_reward:
            from axon.utils.rl.kl import get_kl_controller

            self.kl_ctrl_in_reward = get_kl_controller(self.config.kl_reward_args)

        # MoE replay validation (common to all modes)
        if self.config.moe_replay:
            assert self.config.actor.strategy == "megatron", "MOE replay requires megatron actor"
            assert self.config.actor.megatron.virtual_pipeline_model_parallel_size is None, (
                "VPP not supported with MOE replay"
            )

        # Delegate to subclass for mode-specific setup
        self._init_mode()

    # ------------------------------------------------------------------
    # Shared helpers for subclass _init_mode()
    # ------------------------------------------------------------------

    def _init_p2p_weight_sync(self):
        """Set up P2P weight transfer channel (actor → sampler) for disaggregated mode.

        Creates a NCCL process group between actor and sampler workers and
        builds a routing table mapping actor parameters to sampler parameters.
        Subclasses call this from ``_init_mode()`` when ``self.disaggregated``
        is True.
        """
        master_address, master_port = self.actor_wg.get_node_ip_and_free_port()[0]
        tcp_address = f"tcp://{master_address}:{master_port}"
        self.actor_wg.connect_trainer_to_sampler(self.sampler_wg, init_method=tcp_address)

        actor_params = self.actor_wg.get_parameter_mapping()
        sampler_params = self.sampler_wg.get_parameter_mapping()
        self.routing_table = RoutingTable(
            actor_rank_mapping=actor_params,
            sampler_rank_mapping=sampler_params,
        )

    def _initialize_sampler_servers(self, worker_group):
        """Initialize sampler servers and return their server addresses.

        This method creates Server instances (vllm or sglang based on config)
        and initializes them in hybrid mode with the provided worker group.

        Args:
            worker_group: The Ray worker group for the sampler/actor_sampler workers.

        Returns:
            tuple: (sampler_servers, server_addresses)
                - sampler_servers: List of Server instances
                - server_addresses: List of server address strings for LLM generation
        """
        sampler_world_size = (
            self.config.sampler.tensor_model_parallel_size
            * self.config.sampler.data_parallel_size
            * self.config.sampler.pipeline_model_parallel_size
        )
        world_size = worker_group.world_size
        num_replicas = world_size // sampler_world_size

        sampler_config = self.config.sampler
        sampler_server_class = get_server_class(sampler_config.name)

        sampler_servers = [
            sampler_server_class(
                replica_rank=replica_rank,
                config=sampler_config,
                gpus_per_node=self.config.num_gpus_per_node,
                decoding_config=self.config.decoding,
            )
            for replica_rank in range(num_replicas)
        ]

        async def init_all_replicas():
            await asyncio.gather(*[replica.init_hybrid(worker_group) for replica in sampler_servers])

        asyncio.run(init_all_replicas())

        server_addresses = [replica.server_address for replica in sampler_servers]
        logger.info(f"Initialized {len(sampler_servers)} sampler servers: {server_addresses}")
        return sampler_servers, server_addresses

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train_rl(self):
        """Run the PPO training loop.

        Sets up tracking, loads checkpoints, performs initial weight sync,
        optional pre-training validation, and then delegates to the
        subclass's :meth:`_run_training_loop`. Tracking backends are torn
        down deterministically on exit via ``Tracking.finish()``.
        """
        try:
            if not self._eval_only:
                self._load_state()

                # Initial weight sync: actor → sampler
                if self.disaggregated and not self._use_dummy_batch:
                    self._broadcast_weights()

            # Optional pre-training validation
            if getattr(self.config, "validation", None) and (
                self.config.validation.get("before_train", False) or self.config.get("mode") == "eval"
            ):
                val_metrics = self._validation_step()
                self.tracking.log(
                    data=val_metrics, step=self.global_steps, title=f"Validation Step: {self.global_steps}"
                )
                if self.config.get("mode") == "eval":
                    return

            if self._memory_stress_test:
                self._run_memory_stress_test()
                return

            self._run_training_loop()
            self._final_validation()
        finally:
            self.tracking.finish()

    # ------------------------------------------------------------------
    # Memory stress test
    # ------------------------------------------------------------------

    def _run_memory_stress_test(self):
        """Run worst-case memory stress test and exit.

        Exercises the full training pipeline with max-length dummy data:
        1. vLLM wake → sleep cycle (establishes real memory baseline)
        2. Old log prob forward (fills PyTorch cache pool)
        3. forward_backward + optim_step (peak memory)

        If both steps complete without OOM, the config is viable.
        """
        from axon.driver.components.advantage_component import compute_advantage_component
        from axon.driver.components.program_component import ProgramProcessor, ProgramTransformConfig
        from axon.driver.driver_utils import balance_batch, pad_dataproto_to_world_size, update_trainer
        from axon.protocol import DataProto
        from axon.utils.print_utils import colorful_print

        colorful_print("=" * 60, "cyan")
        colorful_print("  MEMORY STRESS TEST — worst-case validation", "cyan")
        colorful_print("=" * 60, "cyan")

        transform_config = ProgramTransformConfig.from_config(self.config)
        pp = ProgramProcessor(transform_config, self.tokenizer, self.processor)
        has_sampling_client = self.hybrid_engine and hasattr(self, "sampling_client")
        n_samples = self.config.decoding.n
        use_sampler_logprobs = self.config.get("use_sampler_logprobs", False)

        for step in range(1, self.total_training_steps + 1):
            colorful_print(f"[stress] Step {step} — wake/sleep cycle...", "yellow")

            # 1. vLLM wake → sleep (same as real training's sampling → train transition)
            if has_sampling_client:
                asyncio.run(self.sampling_client.wake_up())
                asyncio.run(self.sampling_client.sleep())

            # 2. Create worst-case dummy batch
            batch_dict = self.train_dataloader.next(batch_size=self.train_batch_size)
            batch = self._prepare_batch(batch_dict, global_steps=step)
            train_batch = pp.create_dummy_batch(batch, global_steps=step)

            # Alternating rewards so advantages survive filtering
            scores = train_batch.batch["token_level_scores"]
            for i in range(len(scores)):
                scores[i, -1] = float(i % n_samples < n_samples // 2)

            train_batch.meta_info["temperature"] = self.config.decoding.temperature
            train_batch.meta_info["global_token_num"] = train_batch.batch["attention_mask"].sum(dim=-1).tolist()

            metrics = {}
            train_batch = compute_advantage_component(train_batch, self.config, metrics)
            world_sizes = self._collect_world_sizes()
            train_batch = pad_dataproto_to_world_size(train_batch, world_sizes)
            train_batch = balance_batch(train_batch, world_size=self.actor_wg.world_size, metrics=metrics)

            # 3. Old log prob forward
            if not use_sampler_logprobs:
                result = self.actor_client.forward(train_batch)
                train_batch = train_batch.union(
                    DataProto.from_dict(tensors={"old_log_probs": result.batch["log_probs"]})
                )

            # 4. forward_backward + optim_step
            with self.actor_client.load_model_context():
                actor_metrics = update_trainer(
                    batch=train_batch,
                    training_client=self.actor_client,
                    loss_fn=self.config.loss,
                    loss_fn_args=self.config.loss_args,
                    epochs=1,
                    mini_batch_size=self.config.mini_batch_size * self.config.decoding.n,
                    world_size=self.actor_wg.world_size,
                )

            # Report per-step metrics
            perf = {k: v for k, v in sorted(actor_metrics.items()) if "perf/" in k}
            perf_str = ", ".join(f"{k.split('/')[-1]}={v:.2f}" for k, v in perf.items() if "gb" in k.lower())
            colorful_print(f"[stress] Step {step} PASSED — {perf_str}", "green")
            self.tracking.log(data=actor_metrics, step=step, title=f"Stress Step: {step}")

        colorful_print("=" * 60, "green")
        colorful_print(f"  MEMORY STRESS TEST PASSED — {self.total_training_steps} steps", "green")
        colorful_print("=" * 60, "green")

    # ------------------------------------------------------------------
    # Weight synchronization
    # ------------------------------------------------------------------

    def _broadcast_weights(self):
        """Synchronize model weights from actor to sampler workers.

        Uses concurrent send/recv P2P operations. The routing table is
        cleared after the first transfer since workers store it internally.
        """
        receive_refs = self.sampler_wg.receive_sampler_to_trainer_weights(routing_table=self.routing_table)
        self.actor_wg.send_trainer_to_sampler_weights(routing_table=self.routing_table)
        ray.get(receive_refs)
        self.routing_table = None

    # ------------------------------------------------------------------
    # Post-step: metrics, validation, checkpoint
    # ------------------------------------------------------------------

    def _post_step(self, metrics: dict, timing_raw: dict):
        """Run post-step operations: validation, checkpoint, then single log.

        Args:
            metrics: Dict of metrics collected during the step. Subclasses
                should merge data/timing metrics before calling this.
            timing_raw: Dict of raw timing measurements.
        """
        # Periodic validation (results merged into step metrics)
        validation_config = getattr(self.config, "validation", None)
        if validation_config and getattr(validation_config, "steps", 0) > 0:
            if self.global_steps % validation_config.steps == 0:
                val_start = time.time()
                val_metrics = self._validation_step()
                timing_raw["validation"] = time.time() - val_start
                metrics.update(val_metrics)

        # Periodic checkpoint
        if self.config.save_steps > 0 and self.global_steps % self.config.save_steps == 0:
            save_start = time.time()
            self._save_state()
            timing_raw["save_state"] = time.time() - save_start

        # Add post-step timing
        for key in ("validation", "save_state"):
            if key in timing_raw:
                metrics[f"timing_s/{key}"] = timing_raw[key]

        # Single combined log per step
        self.tracking.log(data=metrics, step=self.global_steps, title=f"Training Step: {self.global_steps}")

        self.temperature_scheduler.step()
        self.global_steps += 1

    def _final_validation(self):
        """Run final validation at the end of training."""
        val_metrics = self._validation_step()
        self.tracking.log(data=val_metrics, step=self.global_steps, title=f"Final Validation Step: {self.global_steps}")

    def _collect_world_sizes(self) -> list[int]:
        """Collect world sizes from all active worker groups for padding.

        Returns a list of non-zero world sizes from critic, ref policy,
        reward model, actor/sampler worker groups. Used by
        pad_dataproto_to_world_size to ensure batch size is divisible by
        the LCM of all worker group sizes.
        """
        world_sizes = []
        if self.use_critic and self.critic_wg.world_size != 0:
            world_sizes.append(self.critic_wg.world_size)
        if self.use_ref and self.ref_policy_wg.world_size != 0:
            world_sizes.append(self.ref_policy_wg.world_size)
        if self.use_rm and self.rm_wg is not None and self.rm_wg.world_size != 0:
            world_sizes.append(self.rm_wg.world_size)
        world_sizes.append(self.actor_wg.world_size)
        if self.disaggregated and self.sampler_wg is not None:
            world_sizes.append(self.sampler_wg.world_size)
        return world_sizes

    # ------------------------------------------------------------------
    # Checkpoint management
    # ------------------------------------------------------------------

    def _save_state(self):
        """Save actor, critic, and dataloader state to checkpoint."""
        checkpoint_base = os.path.join(self.experiment_dir, "checkpoints")
        step_folder = os.path.join(checkpoint_base, f"step_{self.global_steps}")
        os.makedirs(step_folder, exist_ok=True)
        logger.info(f"Saving state to: {step_folder}")

        actor_path = os.path.join(step_folder, str(Role.Actor))
        max_ckpt = self.config.max_checkpoints_to_keep
        save_mode = getattr(self.config, "checkpoint_format", None)
        async_save = getattr(self.config, "async_save", False)

        self.actor_wg.save_state(
            actor_path,
            self.global_steps,
            max_ckpt_to_keep=max_ckpt,
            save_mode=save_mode,
            async_save=async_save,
        )

        if self.use_critic:
            critic_path = os.path.join(step_folder, str(Role.Critic))
            self.critic_wg.save_state(
                critic_path,
                self.global_steps,
                max_ckpt_to_keep=max_ckpt,
                save_mode=save_mode,
                async_save=async_save,
            )

        dataloader_path = os.path.join(step_folder, "data.pt")
        torch.save(self.train_dataloader.state_dict(), dataloader_path)

    def _load_state(self):
        """Load actor, critic, and dataloader state from latest checkpoint."""
        resume_config = self.config.resume_from_checkpoint
        if not resume_config:
            print("[load_state] Training from scratch (resume_from_checkpoint is disabled)", flush=True)
            return

        checkpoint_folder = os.path.join(self.experiment_dir, "checkpoints")
        global_step_folder = find_latest_ckpt_path(checkpoint_folder)
        if global_step_folder is None:
            print(f"[load_state] No checkpoint under {checkpoint_folder}, training from scratch", flush=True)
            return

        print(f"[load_state] Loading checkpoint from: {global_step_folder}", flush=True)

        folder_name = os.path.basename(global_step_folder)
        match = re.search(r"(\d+)$", folder_name)
        if match:
            self.global_steps = int(match.group(1))
        else:
            logger.warning(f"Could not parse step from {folder_name}, starting from step 0")
            self.global_steps = 0

        actor_path = os.path.join(global_step_folder, str(Role.Actor))
        save_mode = getattr(self.config, "checkpoint_format", None)
        print(f"[load_state] actor <- {actor_path} (save_mode={save_mode})", flush=True)
        self.actor_wg.load_state(actor_path, save_mode=save_mode)

        if self.use_critic:
            critic_path = os.path.join(global_step_folder, str(Role.Critic))
            print(f"[load_state] critic <- {critic_path} (save_mode={save_mode})", flush=True)
            self.critic_wg.load_state(critic_path, save_mode=save_mode)

        dataloader_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_path):
            dataloader_state_dict = torch.load(dataloader_path, weights_only=False)  # nosec B614
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            logger.warning(f"No dataloader state at {dataloader_path}, starting from scratch")

        logger.info(f"Resumed from step {self.global_steps}")
