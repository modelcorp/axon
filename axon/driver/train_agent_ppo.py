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
"""Entry point for Agent PPO training.

Two controller modes:
- **SyncPPO** (hybrid engine default): Data flows through the controller
  for inspection, reward shaping, and critic/ref enrichment.
- **AsyncPPO** (disaggregated default): Data transfers via NCCL P2P,
  bypassing the controller for maximum throughput.

Select via ``driver_mode`` (``"sync"``/``"async"``) or auto-detect from
``hybrid_engine``.
"""

import os
from pprint import pprint

import hydra
import ray
import torch
from omegaconf import DictConfig, OmegaConf

from axon.config import validate_axon_config
from axon.core import ResourcePool
from axon.core.role import Role
from axon.utils.ray import kill_ray_actors
from axon.utils.tokenizer import hf_processor, hf_tokenizer

# Static env vars propagated to all Ray workers via runtime_env.
_ENV_VARS: dict[str, str] = {
    # Base
    "PYTHONUNBUFFERED": "1",
    "TOKENIZERS_PARALLELISM": "true",
    "NCCL_DEBUG": "WARN",
    "HYDRA_FULL_ERROR": "1",
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    # NOTE: Do NOT set PYTORCH_CUDA_ALLOC_CONF here — each worker configures
    # expandable_segments in init_model() based on whether it is colocated with
    # vLLM (False) or disaggregated (True). Setting it globally would override
    # the per-worker configuration and cause memory fragmentation OOMs on
    # disaggregated trainers.
    "VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1",
    "VLLM_ENGINE_ITERATION_TIMEOUT_S": "100000000000",
    "VLLM_ALLREDUCE_USE_SYMM_MEM": "0",  # Breaks SPMD mode
    "VLLM_LOGGING_LEVEL": "WARN",
    "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "true",
    # SGLang
    "SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK": "True",
    # NCCL — transport and memory
    "NCCL_CUMEM_ENABLE": "0",  # Disable cuMem API; avoids hangs during actor<->sampler weight sync
    "NCCL_CUMEM_HOST_ENABLE": "0",  # Disable host-side cuMem; reduces device memory baseline
    "NCCL_NVLS_ENABLE": "0",  # Disable NVLink SHARP; not needed and avoids extra memory
    "NCCL_P2P_LEVEL": "NVL",  # Use NVLink for P2P transfers
    "NCCL_P2P_DISABLE": "0",
    # NCCL — memory optimization (reduces device memory baseline significantly)
    # Tradeoff: fewer CTAs may reduce bandwidth for small allreduces (Megatron TP),
    # but negligible for large shards (FSDP reduce-scatter).
    "NCCL_MAX_CTAS": "2",
    "NCCL_MIN_CTAS": "1",
    # NCCL RAS subsystem leaks device memory on every P2P communicator usage
    # (github.com/NVIDIA/nccl/issues/1762). Fixed in NCCL 2.29.2+.
    # Remove this once NCCL is upgraded past 2.29.2.
    "NCCL_RAS_ENABLE": "0",
    "ROCR_VISIBLE_DEVICES": "",
    "TORCH_NCCL_AVOID_RECORD_STREAMS": "True",
    "NVTE_FP8_BLOCK_SCALING_FP32_SCALES": "1",
    # Uncomment to diagnose NCCL hangs (adds overhead, triggers failures faster):
    # "TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC": "60",
    # "TORCH_NCCL_TRACE_BUFFER_SIZE": "1000",
    # "TORCH_NCCL_DESYNC_DEBUG": "1",
    # "TORCH_SHOW_CPP_STACKTRACES": "1",
}

# Ray actor concurrency limits.
#
# cuBLAS allocates ~67 MB of internal state per thread-local handle.
# Ray async actors dispatch sync method calls to an internal thread pool;
# with high max_concurrency new threads keep being created, each leaking
# a cuBLAS handle that is never freed.
_RAY_MAX_CONCURRENCY_HYBRID = 2
_RAY_MAX_CONCURRENCY_DISAGG = 4


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(config: DictConfig) -> None:
    run_ppo_agent(config)


def _get_dynamic_env_vars(config: DictConfig) -> dict[str, str]:
    """Return runtime-dependent env vars derived from *config* and the GPU."""
    from axon.utils.rocm_utils import get_rocm_env_vars, is_rocm

    env = {
        "AXON_MOE_REPLAY": str(int(config.moe_replay)),
        "AXON_ENABLE_MTP": os.environ.get("AXON_ENABLE_MTP", "0"),
        "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH", ""),
    }

    # Merge ROCm-specific performance env vars (no-op on CUDA)
    env.update(get_rocm_env_vars())

    if is_rocm():
        # ROCm: no nvidia-smi / compute_capability queries
        env.setdefault("RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES", "1")
        return env

    major, _ = torch.cuda.get_device_capability(0)
    gpu = torch.cuda.get_device_name(0).lower()

    if "h100" in gpu or "h200" in gpu:
        env["TORCH_CUDA_ARCH_LIST"] = "9.0"
    elif "b100" in gpu or "b200" in gpu or "blackwell" in gpu:
        env["TORCH_CUDA_ARCH_LIST"] = "10.0"

    return env


def _preprocess_role_configs(config: DictConfig) -> None:
    """Propagate shared fields to role sub-configs and resolve interpolations
    so they remain self-contained after extraction to Ray workers."""
    from omegaconf import open_dict

    with open_dict(config):
        for role in ("actor", "ref", "sampler"):
            config[role].model_path = config.model_path
            config[role].trust_remote_code = config.get("trust_remote_code", False)
            config[role].external_lib = config.get("external_lib", None)

        config.sampler.lora_rank = config.actor.get("lora", {}).get("rank", 0)

        # Resolve while parent context is still available.
        config.ref = OmegaConf.create(OmegaConf.to_container(config.ref, resolve=True))
        if config.get("reward_model") is not None:
            config.reward_model = OmegaConf.create(OmegaConf.to_container(config.reward_model, resolve=True))

        # Copy framework config from actor -> sampler.
        strategy = config.actor.strategy
        if strategy == "megatron" and OmegaConf.select(config.actor, "megatron") is not None:
            config.sampler.megatron = config.actor.megatron
        elif strategy in ("fsdp", "fsdp2") and OmegaConf.select(config.actor, "fsdp") is not None:
            config.sampler.fsdp = config.actor.fsdp


def _build_resource_pools(config: DictConfig, is_async: bool):
    """Build GPU resource pools. Returns ``(actor_pool, sampler_pool)``."""
    gpus = config.num_gpus_per_node

    if not is_async:
        pool = ResourcePool(process_on_nodes=[gpus] * config.num_nodes, max_colocate_count=3, name_prefix="global_pool")
        return pool, pool

    total = gpus * config.num_nodes
    num_train = int(total // (config.sampler_trainer_gpu_ratio + 1))
    num_infer = total - num_train

    def _split(n):
        nodes, rem = divmod(n, gpus)
        return [gpus] * nodes + ([rem] if rem else [])

    return (
        ResourcePool(process_on_nodes=_split(num_train), max_colocate_count=3, name_prefix="actor_pool"),
        ResourcePool(process_on_nodes=_split(num_infer), max_colocate_count=3, name_prefix="sampler_pool"),
    )


def run_ppo_agent(config: DictConfig) -> None:
    """Initialize Ray and dispatch training."""
    env_vars = {**_ENV_VARS, **_get_dynamic_env_vars(config)}

    if not ray.is_initialized():
        ray.init(runtime_env={"env_vars": env_vars})
    else:
        kill_ray_actors()

    if config.get("launch_on_head", False):
        os.environ.update(env_vars)
        train_agent(config)
    else:
        ray.get(train_agent_ray_task.remote(config))


def train_agent(config: DictConfig) -> None:
    """Validate config, build workers and resource pools, and run PPO training."""
    config_dict = OmegaConf.to_container(config, resolve=True)
    validate_axon_config(config_dict)
    pprint(config_dict)

    local_path = config.model_path
    trust_remote_code = config.get("trust_remote_code", True)
    tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
    processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

    if config.get("mode") == "eval":
        return eval_agent(config, tokenizer, processor)

    # Determine async vs sync mode
    driver_mode = getattr(config, "driver_mode", None)
    use_async = not config.hybrid_engine
    if driver_mode == "async":
        use_async = True
    elif driver_mode == "sync":
        use_async = False

    # Resolve worker classes based on parallelism strategy
    strategy = config.actor.strategy
    if strategy in ("fsdp", "fsdp2"):
        assert config.critic.strategy in ("fsdp", "fsdp2")
        from axon.models.fsdp_models import RewardModel, ValueModel
        from axon.trainer.fsdp_trainer import TrainerWorker as _TrainerWorker
    elif strategy == "megatron":
        assert strategy == config.critic.strategy
        from axon.models.megatron_models import RewardModel, ValueModel
        from axon.trainer.megatron_trainer import TrainerWorker as _TrainerWorker
    else:
        raise NotImplementedError(f"Unsupported actor strategy: {strategy!r}")

    from axon.sampler.sampler import SamplerWorker as _SamplerWorker

    # Compose worker classes with mode-appropriate mixins
    is_fsdp = strategy in ("fsdp", "fsdp2")
    hybrid = config.hybrid_engine
    trainer_mixins, sampler_mixins = [], []

    if hybrid:
        from axon.sampler.mixins.sync_mixin import FSDPSyncSamplerMixin, MegatronSyncSamplerMixin
        from axon.trainer.mixins.sync_mixin import FSDPSyncTrainerMixin, MegatronSyncTrainerMixin

        trainer_mixins.append(FSDPSyncTrainerMixin if is_fsdp else MegatronSyncTrainerMixin)
        sampler_mixins.append(FSDPSyncSamplerMixin if is_fsdp else MegatronSyncSamplerMixin)
    if not hybrid or use_async:
        from axon.sampler.mixins.p2p_mixin import FSDPSamplerP2PMixin, MegatronSamplerP2PMixin
        from axon.trainer.mixins.p2p_mixin import FSDPTrainerP2PMixin, MegatronTrainerP2PMixin

        trainer_mixins.append(FSDPTrainerP2PMixin if is_fsdp else MegatronTrainerP2PMixin)
        sampler_mixins.append(FSDPSamplerP2PMixin if is_fsdp else MegatronSamplerP2PMixin)
    if use_async:
        from axon.sampler.mixins.async_mixin import AsyncSamplerMixin
        from axon.trainer.mixins import AsyncTrainerMixin

        trainer_mixins.append(AsyncTrainerMixin)
        sampler_mixins.append(AsyncSamplerMixin)

    def _with_mixins(base, mixins):
        return type(base.__name__, (*mixins, base), {}) if mixins else base

    TrainerWorker = _with_mixins(_TrainerWorker, trainer_mixins)
    SamplerWorker = _with_mixins(_SamplerWorker, sampler_mixins)

    _preprocess_role_configs(config)

    # One-off-pipeline uses SyncPPO (not async) but with disaggregated GPU pools.
    needs_separate_pools = use_async or (not config.hybrid_engine and config.get("enable_one_off_pipeline", False))
    actor_pool, sampler_pool = _build_resource_pools(config, needs_separate_pools)

    # Build role -> worker mapping
    from axon.driver.driver_utils import RoleWorkerConfig

    _max_conc = _RAY_MAX_CONCURRENCY_HYBRID if hybrid else _RAY_MAX_CONCURRENCY_DISAGG
    _remote = lambda cls: ray.remote(max_concurrency=_max_conc)(cls)

    role_worker_mapping = {
        Role.Actor: RoleWorkerConfig(
            cls=_remote(TrainerWorker),
            resource_pool=actor_pool,
            init_kwargs={"name": "actor"},
            max_concurrency=_max_conc,
        ),
        Role.Critic: RoleWorkerConfig(
            cls=ray.remote(_TrainerWorker),
            resource_pool=actor_pool if use_async else sampler_pool,
            init_kwargs={"model": ValueModel, "name": "critic"},
        ),
    }

    # Skip sampler entirely when using dummy batches (actor-only testing)
    if not config.use_dummy_batch:
        role_worker_mapping[Role.Sampler] = RoleWorkerConfig(cls=_remote(SamplerWorker), resource_pool=sampler_pool)

    ref_in_actor = (
        config.actor.get("lora", {}).get("rank", 0) > 0 or config.actor.get("lora", {}).get("adapter_path") is not None
    )
    if not ref_in_actor and (
        config.get("kl_reward", None) is not None or config.get("loss_args", {}).get("kl_coef", 0) != 0
    ):
        role_worker_mapping[Role.RefPolicy] = RoleWorkerConfig(
            cls=ray.remote(TrainerWorker), resource_pool=actor_pool, init_kwargs={"name": "ref"}
        )

    if config.reward_model.get("enable", False):
        role_worker_mapping[Role.RewardModel] = RoleWorkerConfig(
            cls=ray.remote(TrainerWorker),
            resource_pool=actor_pool,
            init_kwargs={"model": RewardModel, "name": "reward"},
        )

    # Select and run controller
    if use_async:
        from axon.driver.async_ppo import AsyncPPO as controller_cls
    else:
        from axon.driver.sync_ppo import SyncPPO as controller_cls

    controller = controller_cls(
        config=config,
        tokenizer=tokenizer,
        processor=processor,
        role_worker_mapping=role_worker_mapping,
    )
    controller.init_workers()
    controller.train_rl()


def eval_agent(config, tokenizer, processor):
    """Eval-only: sampler workers only, run validation, exit."""
    from axon.driver.driver_utils import RoleWorkerConfig
    from axon.driver.sync_ppo import SyncPPO
    from axon.sampler.sampler import SamplerWorker

    _preprocess_role_configs(config)
    pool = ResourcePool(
        process_on_nodes=[config.num_gpus_per_node] * config.num_nodes,
        max_colocate_count=1,
        name_prefix="eval",
    )
    controller = SyncPPO(
        config=config,
        tokenizer=tokenizer,
        processor=processor,
        role_worker_mapping={
            Role.Sampler: RoleWorkerConfig(
                cls=ray.remote(max_concurrency=_RAY_MAX_CONCURRENCY_HYBRID)(SamplerWorker),
                resource_pool=pool,
            ),
        },
    )
    controller.init_workers()
    controller.train_rl()


@ray.remote(num_cpus=1)
def train_agent_ray_task(config: DictConfig) -> None:
    """Ray remote wrapper that runs :func:`train_agent` on a worker node."""
    return train_agent(config)


if __name__ == "__main__":
    main()
