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
import asyncio
import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

from omegaconf import DictConfig
from pydantic import BaseModel
from ray.actor import ActorHandle

from axon.controller.ray import RayActorWithInitArgs, RayWorkerGroup
from axon.core import ResourcePool
from axon.utils.registry import ClassRegistry

logger = logging.getLogger(__file__)

# Server registry
SERVER_REGISTRY = ClassRegistry("server")
register_server = SERVER_REGISTRY.register


class SpecDecodeMetrics(BaseModel):
    """Per-request speculative decoding metrics (EAGLE / MTP / n-gram)."""

    num_draft_tokens: int = 0
    """Total tokens proposed by the draft model."""
    num_accepted_tokens: int = 0
    """Tokens accepted after verification by the target model."""
    num_completions: int = 0
    """Number of verification rounds (each round accepts 1+ tokens)."""

    @property
    def accept_rate(self) -> float:
        """Fraction of drafted tokens that were accepted."""
        return self.num_accepted_tokens / self.num_draft_tokens if self.num_draft_tokens > 0 else 0.0

    @property
    def accept_length(self) -> float:
        """Average accepted tokens per verification round (1.0 = no speedup)."""
        return (
            (self.num_accepted_tokens + self.num_completions) / self.num_completions
            if self.num_completions > 0
            else 0.0
        )


class TokenOutput(BaseModel):
    token_ids: list[int]
    """response token ids"""
    log_probs: list[float] | None = None
    """logprobs of response token ids"""
    routed_experts: Any | None = None
    """routed experts of response token ids"""
    stop_reason: str | None = None
    """stop reason: 'completed', 'aborted', or None for unknown"""
    spec_decode_metrics: SpecDecodeMetrics | None = None
    """speculative decoding metrics (EAGLE / MTP / n-gram), None when disabled"""


class SamplerMode(Enum):
    # Sampler engine and training engine(fsdp/megatron) fused in same process
    # Sampler and trainer share GPUs, switch context with weight synchronization.
    # Usage scenarios: on-policy training.
    HYBRID = "hybrid"

    # Sampler engine colocated with hybrid engine in same ray placement group but in separate process.
    # Sampler and hybrid processes share GPUs, switch context without weight synchronization.
    # Usage scenarios: GRM (LLM as a judge).
    COLOCATED = "colocated"

    # Standalone sampler server with separate GPU resource, disaggregated architecture.
    # Usage scenarios: off-policy training.
    STANDALONE = "standalone"


class Server(ABC):
    """An individual server may be deployed on single or multiple nodes.
    It is almost equivalent to launch server in each node with command line:

    SGLang:
    ```
    python -m sglang.launch_server --node-rank 0 --nnode 2 ...
    python -m sglang.launch_server --node-rank 1 --nnode 2 ...
    ```

    vLLM:
    ```
    vllm serve --data-parallel-size 16 --data-parallel-size-local 8 --data-parallel-start-rank 0 ...
    vllm serve --data-parallel-size 16 --data-parallel-size-local 8 --data-parallel-start-rank 8 ...
    ```

    Args:
        replica_rank: int, rank of this sampler replica.
        config: Sampler config dict.
        gpus_per_node: int, number of gpus per node.
    """

    def __init__(
        self,
        replica_rank: int,
        config: DictConfig,
        decoding_config: DictConfig,
        gpus_per_node: int = 8,
        is_reward_model: bool = False,
    ) -> None:
        self.replica_rank = replica_rank
        self.config = config
        self.decoding_config = decoding_config

        self.world_size = (
            self.config.tensor_model_parallel_size
            * self.config.data_parallel_size
            * self.config.pipeline_model_parallel_size
        )
        self.gpus_per_node = min(gpus_per_node, self.world_size)
        assert self.world_size % self.gpus_per_node == 0, (
            f"world_size {self.world_size} must be divisible by gpus_per_node {self.gpus_per_node}"
        )
        self.nnodes = self.world_size // self.gpus_per_node
        self.is_reward_model = is_reward_model

        self.sampler_mode: SamplerMode = None
        self.workers: list[ActorHandle] = []
        self.resource_pool: ResourcePool = None

        self.servers: list[ActorHandle] = []
        self._server_address: str = None
        self._server_handle: ActorHandle = None

    async def init_hybrid(self, worker_group: RayWorkerGroup):
        """Init hybrid sampler server, sampler engine and training engine(fsdp/megatron) fused in same process.

        Args:
            worker_group: RayWorkerGroup, fused workers where training engine(fsdp/megatron) have been initialized.
        """
        self.sampler_mode = SamplerMode.HYBRID
        self.workers = worker_group.workers[
            self.world_size * self.replica_rank : self.world_size * (self.replica_rank + 1)
        ]
        await self.launch_servers()

    @abstractmethod
    def get_ray_class_with_init_args(self) -> RayActorWithInitArgs:
        """Get sampler worker actor class for colocated and standalone mode."""
        raise NotImplementedError

    @abstractmethod
    async def launch_servers(self):
        """Launch http server in each node."""
        raise NotImplementedError

    @property
    def server_address(self) -> str:
        """Get sampler server address for OpenAI chat completion."""
        return self._server_address

    @property
    def server_handle(self) -> ActorHandle:
        """Get sampler server handle for Token-in-token-out generation."""
        return self._server_handle

    async def wake_up(self):
        """Wake up each sampler server."""
        await asyncio.gather(*[server.wake_up.remote() for server in self.servers])

    async def sleep(self):
        """Sleep each sampler server."""
        await asyncio.gather(*[server.sleep.remote() for server in self.servers])
