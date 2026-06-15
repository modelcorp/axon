# Copyright 2023-2024 SGLang Team
# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
import dataclasses
import json
import logging
import os
from typing import Any

import axon.sampler.sglang._compat  # noqa: F401  isort:skip
import ray
import sglang
import sglang.srt.entrypoints.engine
import torch
from omegaconf import DictConfig
from ray.actor import ActorHandle
from sglang.srt.entrypoints.http_server import (
    ServerArgs,
    _GlobalState,
    _launch_subprocesses,
    app,
    set_global_state,
)
from sglang.srt.managers.io_struct import (
    GenerateReqInput,
    ReleaseMemoryOccupationReqInput,
    ResumeMemoryOccupationReqInput,
)
from sglang.srt.managers.tokenizer_manager import ServerStatus

from axon.controller.ray import RayActorWithInitArgs
from axon.sampler.base.server import SamplerMode, Server, TokenOutput, register_server
from axon.sampler.sglang.engine import SGLangEngine, _set_envs_and_config
from axon.utils.networking_utils import get_free_port, is_valid_ipv6_address, run_unvicorn

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@ray.remote(num_cpus=1)
class SGLangHttpClient:
    """SGLang http server in single node, this is equivalent to launch server with command line:
    ```
    python -m sglang.launch_server --node-rank 0 --nnode 1 ...
    ```

    Args:
        config (DictConfig): full config.
        sampler_mode (SamplerMode): sampler mode.
        replica_rank (int): replica rank, a replica may contain multiple nodes.
        node_rank (int): node rank.
        nnodes (int): number of nodes.
        cuda_visible_devices (str): cuda visible devices.
    """

    def __init__(
        self,
        config: DictConfig,
        sampler_mode: SamplerMode,
        workers: list[ActorHandle],
        replica_rank: int,
        node_rank: int,
        nnodes: int,
        cuda_visible_devices: str,
    ):
        logger.info(
            f"SGLang server: {sampler_mode=} {replica_rank=} {node_rank=} {nnodes=} gpus={cuda_visible_devices}"
        )
        from axon.utils.rocm_utils import get_visible_devices_env_key

        vis_key = get_visible_devices_env_key()
        os.environ[vis_key] = cuda_visible_devices
        # SGLang/vLLM also reads CUDA_VISIBLE_DEVICES on ROCm
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
        assert torch.cuda.is_available(), "SGLang http server should run on GPU node"

        self.config = config
        self.sampler_mode = sampler_mode
        self.workers = workers

        self.replica_rank = replica_rank
        self.node_rank = node_rank
        self.nnodes = nnodes

        self.max_model_len = int(self.config.max_model_len) if self.config.max_model_len else 32768

        if self.sampler_mode != SamplerMode.HYBRID and self.config.load_format == "dummy":
            logger.warning(f"sampler mode is {self.sampler_mode}, load_format is dummy, set to auto")
            self.config.load_format = "auto"

        self._server_address = ray.util.get_node_ip_address().strip("[]")
        self._server_port = None  # set once launch_server() completes

        if self.node_rank == 0:
            self._master_address = self._server_address
            # _master_sock is kept alive to reserve the port for NCCL process group init
            self._master_port, self._master_sock = get_free_port(self._server_address)
            logger.info(f"SGLang master: replica={self.replica_rank} addr={self._master_address}:{self._master_port}")
        else:
            self._master_address = None
            self._master_port = None

    def get_master_address(self) -> tuple[str | None, int | None]:
        """Return (address, port) for NCCL process group init."""
        return self._master_address, self._master_port

    def get_server_address(self) -> tuple[str, int]:
        """Return (address, port) of the running HTTP server."""
        assert self._server_port is not None, "http server not launched yet"
        return self._server_address, self._server_port

    async def launch_server(self, master_address: str = None, master_port: int = None):
        if self.node_rank != 0:
            assert master_address and master_port, "non-master node should provide master address and port"
            self._master_address = master_address
            self._master_port = master_port

        engine_kwargs = self.config.get("engine_kwargs", {}).get("sglang", {}) or {}
        attention_backend = engine_kwargs.pop("attention_backend", None)
        if self.config.get("limit_images", None):
            engine_kwargs["limit_mm_data_per_request"] = {"image": self.config.get("limit_images")}
        quantization = self.config.get("quantization")
        quantization_model_override = {}
        if quantization is not None:
            if quantization == "fp8":
                from packaging.version import Version

                assert Version(sglang.__version__) >= Version("0.5.5"), "sglang>=0.5.5 is required for FP8 quantization"
                quantization_model_override = {
                    "activation_scheme": "dynamic",
                    "fmt": "e4m3",
                    "quant_method": "fp8",
                    "weight_block_size": [128, 128],
                }
            elif quantization == "mxfp8":
                quantization_model_override = {
                    "activation_scheme": "dynamic",
                    "fmt": "e4m3",
                    "quant_method": "mxfp8",
                    "weight_block_size": [1, 32],
                    "scale_fmt": "ue8m0",
                }
            elif quantization == "int4":
                quantization_model_override = {
                    "quant_method": "compressed-tensors",
                    "format": "pack-quantized",
                    "config_groups": {
                        "group_0": {
                            "weights": {
                                "num_bits": 4,
                                "group_size": self.config.get("int4_group_size", 128),
                                "symmetric": self.config.get("int4_symmetric", True),
                                "strategy": "group",
                                "observer": "minmax",
                            }
                        }
                    },
                }
                # INT4 compressed-tensors: SGLang loads via its native quantization support
                quantization = "compressed-tensors"
            else:
                raise ValueError(
                    f"SGLang supports fp8, mxfp8, int4 quantization, got: {quantization}. "
                    f"fp8_fast is only available with the vLLM backend."
                )
        dist_init_addr = (
            f"[{self._master_address}]:{self._master_port}"
            if is_valid_ipv6_address(self._master_address)
            else f"{self._master_address}:{self._master_port}"
        )

        args = {
            "model_path": self.config.model_path,
            "dtype": self.config.dtype,
            "mem_fraction_static": self.config.gpu_memory_utilization,
            "disable_cuda_graph": self.config.enforce_eager,
            "enable_memory_saver": True,
            "base_gpu_id": 0,
            "gpu_id_step": 1,
            "tp_size": self.config.tensor_model_parallel_size,
            "pp_size": self.config.pipeline_model_parallel_size,
            "dp_size": self.config.data_parallel_size,
            "ep_size": self.config.expert_parallel_size,
            "node_rank": self.node_rank,
            "load_format": self.config.load_format,
            "dist_init_addr": dist_init_addr,
            "nnodes": self.nnodes,
            "trust_remote_code": self.config.trust_remote_code,
            "max_running_requests": self.config.get("max_num_seqs"),
            "log_level": "error" if self.config.get("disable_log_stats", True) else "info",
            "mm_attention_backend": "fa3",
            "attention_backend": attention_backend or "fa3",
            "skip_tokenizer_init": self.config.skip_tokenizer_init,
            "skip_server_warmup": True,
            "quantization": quantization,
            "json_model_override_args": json.dumps({"quantization_config": quantization_model_override})
            if quantization_model_override
            else json.dumps({}),
            **engine_kwargs,
        }

        if self.config.cudagraph_capture_sizes:
            args["cuda_graph_bs"] = self.config.cudagraph_capture_sizes

        if not self.config.get("enable_chunked_prefill", True):
            args["chunked_prefill_size"] = -1
        else:
            max_batched = self.config.get("max_num_batched_tokens")
            if max_batched:
                args["chunked_prefill_size"] = max(int(max_batched), self.max_model_len)

        if not self.config.get("enable_prefix_caching", False):
            args["disable_radix_cache"] = True

        if self.config.get("speculative_config"):
            # sglang uses individual speculative_* fields (not a JSON blob like vLLM)
            spec_cfg = dict(self.config.speculative_config)
            args.update(spec_cfg)

        if self.config.expert_parallel_size > 1:
            args["moe_a2a_backend"] = self.config.get("all2all_backend", "deepep")
            args["enable_eplb"] = self.config.get("enable_eplb", False)

        if self.config.prometheus.enable:
            if self.config.prometheus.served_model_name:
                args["served_model_name"] = self.config.prometheus.served_model_name.rsplit("/", 1)[-1]
            args["enable_metrics"] = True

        if any(f.name == "enable_weights_cpu_backup" for f in dataclasses.fields(ServerArgs)):
            # Enable CPU backup for colocated mode (weight swapping) and for any
            # speculative decoding config (draft model weights need CPU backup so
            # they can be synced during RL training without holding GPU memory).
            args["enable_weights_cpu_backup"] = self.sampler_mode == SamplerMode.COLOCATED or bool(
                self.config.get("speculative_config")
            )

        # We can't call SGLang's launch_server directly because it's synchronous;
        # instead we call _launch_subprocesses and wire up the ASGI app ourselves.
        sglang.srt.entrypoints.engine._set_envs_and_config = _set_envs_and_config
        os.environ["SGLANG_BLOCK_NONZERO_RANK_CHILDREN"] = "0"
        server_args = ServerArgs(**args)
        self.tokenizer_manager, self.template_manager, self.scheduler_info, *_ = _launch_subprocesses(server_args)

        if self.node_rank > 0:  # only rank-0 node runs the http server
            return

        set_global_state(
            _GlobalState(
                tokenizer_manager=self.tokenizer_manager,
                template_manager=self.template_manager,
                scheduler_info=self.scheduler_info,
            )
        )
        app.is_single_tokenizer_mode = True

        app.warmup_thread_args = (server_args, None, None)  # avoid AttributeError in lifespan

        # Add Prometheus middleware before server start so /metrics is available immediately
        if server_args.enable_metrics:
            from sglang.srt.utils.common import add_prometheus_middleware

            add_prometheus_middleware(app)

        self._server_port, self._server_task = await run_unvicorn(app, server_args, self._server_address)
        self.tokenizer_manager.server_status = ServerStatus.Up

    async def wake_up(self):
        if self.sampler_mode == SamplerMode.HYBRID:
            # Workers switch from trainer mode to sampler mode (includes weight sync)
            await asyncio.gather(*[worker.wake_up.remote() for worker in self.workers])
        elif self.sampler_mode == SamplerMode.COLOCATED:
            # Resume GPU memory directly, no weight sync needed
            obj = ResumeMemoryOccupationReqInput(tags=["kv_cache", "weights"])
            await self.tokenizer_manager.resume_memory_occupation(obj, None)
            await self.tokenizer_manager.flush_cache()
        elif self.sampler_mode == SamplerMode.STANDALONE:
            logger.info("skip wake_up in standalone mode")

    async def sleep(self):
        if self.sampler_mode == SamplerMode.HYBRID:
            await asyncio.gather(*[worker.sleep.remote() for worker in self.workers])
        elif self.sampler_mode == SamplerMode.COLOCATED:
            obj = ReleaseMemoryOccupationReqInput(tags=["kv_cache", "weights"])
            await self.tokenizer_manager.release_memory_occupation(obj, None)
        elif self.sampler_mode == SamplerMode.STANDALONE:
            logger.info("skip sleep in standalone mode")

    async def generate(
        self,
        prompt_ids: torch.Tensor,
        sampling_params: dict[str, Any],
        request_id: str,
        image_data: list[Any] | None = None,
    ) -> TokenOutput:
        """Generate tokens via the local tokenizer_manager (token-in-token-out)."""
        max_new_tokens = max(1, self.max_model_len - len(prompt_ids) - 1)
        sampling_params["max_new_tokens"] = max_new_tokens
        return_logprob = sampling_params.pop("logprobs", False)

        request = GenerateReqInput(
            rid=request_id,
            input_ids=prompt_ids,
            sampling_params=sampling_params,
            return_logprob=return_logprob,
            image_data=image_data,
        )
        output = await self.tokenizer_manager.generate_request(request, None).__anext__()
        if return_logprob:
            log_probs, token_ids = zip(
                *[(lp, tid) for lp, tid, _ in output["meta_info"]["output_token_logprobs"]], strict=True
            )
        else:
            token_ids = output["output_ids"]
            log_probs = None

        # Extract speculative decoding metrics from SGLang meta_info if available
        spec_metrics = None
        meta = output.get("meta_info", {})
        draft_count = meta.get("spec_draft_token_num", 0)
        if draft_count > 0:
            from axon.sampler.base.server import SpecDecodeMetrics

            spec_metrics = SpecDecodeMetrics(
                num_draft_tokens=draft_count,
                num_accepted_tokens=meta.get("spec_accept_token_num", 0),
                num_completions=meta.get("spec_verify_ct", 0),
            )

        return TokenOutput(token_ids=token_ids, log_probs=log_probs, spec_decode_metrics=spec_metrics)


_sampler_worker_actor_cls = ray.remote(SGLangEngine)


@register_server("sglang")
class SGLangServer(Server):
    def get_ray_class_with_init_args(self) -> RayActorWithInitArgs:
        """Get sampler worker actor class for colocated and standalone mode."""
        return RayActorWithInitArgs(
            cls=_sampler_worker_actor_cls,
            config=self.config,
            device_mesh=None,
        )

    async def launch_servers(self):
        """Launch http server in each node."""
        assert len(self.workers) == self.world_size, (
            f"worker number {len(self.workers)} not equal to world size {self.world_size}"
        )

        worker_infos = await asyncio.gather(
            *[
                worker.__ray_call__.remote(
                    lambda self: (ray.get_runtime_context().get_node_id(), os.environ["CUDA_VISIBLE_DEVICES"])
                )
                for worker in self.workers
            ]
        )
        worker_node_ids, worker_cuda_devices = zip(*worker_infos, strict=False) if worker_infos else ([], [])

        for node_rank in range(self.nnodes):
            workers = self.workers[node_rank * self.gpus_per_node : (node_rank + 1) * self.gpus_per_node]
            node_cuda_visible_devices = ",".join(
                worker_cuda_devices[node_rank * self.gpus_per_node : (node_rank + 1) * self.gpus_per_node]
            )
            node_id = worker_node_ids[node_rank * self.gpus_per_node]
            prefix = "sglang_server_reward" if self.is_reward_model else "sglang_server"
            name = f"{prefix}_{self.replica_rank}_{node_rank}"
            server = SGLangHttpClient.options(
                scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                    node_id=node_id,
                    soft=False,
                ),
                runtime_env={"env_vars": {"RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1"}},
                name=name,
            ).remote(
                config=self.config,
                sampler_mode=self.sampler_mode,
                workers=workers,
                replica_rank=self.replica_rank,
                node_rank=node_rank,
                nnodes=self.nnodes,
                cuda_visible_devices=node_cuda_visible_devices,
            )
            self.servers.append(server)

        master_address, master_port = await self.servers[0].get_master_address.remote()
        await asyncio.gather(
            *[
                server.launch_server.remote(master_address=master_address, master_port=master_port)
                for server in self.servers
            ]
        )

        server_address, server_port = await self.servers[0].get_server_address.remote()
        self._server_handle = self.servers[0]
        self._server_address = (
            f"[{server_address}]:{server_port}"
            if is_valid_ipv6_address(server_address)
            else f"{server_address}:{server_port}"
        )
