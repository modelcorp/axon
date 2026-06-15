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
import argparse
import asyncio
import inspect
import json
import logging
import os
import queue
import threading
import uuid
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from pprint import pprint
from typing import Any

import cloudpickle as pickle
import ray
import vllm.entrypoints.cli.serve  # noqa: E402
import zmq
from omegaconf import DictConfig
from ray.actor import ActorHandle
from starlette.requests import Request
from starlette.responses import JSONResponse
from vllm.engine.arg_utils import AsyncEngineArgs  # noqa: E402
from vllm.entrypoints.openai.api_server import (  # noqa: E402
    build_app,
    init_app_state,
)
from vllm.usage.usage_lib import UsageContext
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.utils.network_utils import get_tcp_uri
from vllm.v1.engine.async_llm import AsyncLLM  # noqa: E402
from vllm.v1.engine.core import EngineCoreProc  # noqa: E402
from vllm.v1.engine.utils import CoreEngineProcManager  # noqa: E402
from vllm.v1.executor.abstract import Executor  # noqa: E402
from vllm.v1.outputs import ModelRunnerOutput  # noqa: E402

from axon.controller.ray import RayActorWithInitArgs

# =============================================================================
# vLLM patches (sampler logprobs, pause/continue, multimodal, MTP, MOE replay,
# PP sync, FlashInfer fixes) are all in the vLLM fork — no monkey patches needed.
# axon-side patches still active: batch_invariant_ops, fp8/
# =============================================================================
from axon.sampler.base.server import SamplerMode, Server, register_server

# =============================================================================
# vLLM patches (pause/continue, multimodal, routed_experts API) are in the
# vllm fork — no monkey patches needed here.
# axon-side patches: FP8 (fp8/), compute_logits (engine.py), batch_invariant_ops
# =============================================================================
from axon.sampler.vllm import vLLMEngine  # noqa: E402
from axon.utils.networking_utils import get_free_port, is_valid_ipv6_address, run_unvicorn
from axon.utils.vllm import (
    ZMQFutureWrapper,
    get_vllm_max_lora_rank,
)

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


def _check_rpc_result(result):
    """Raise RuntimeError if the RPC result wraps a remote exception."""
    if isinstance(result, dict) and result.get("__exception__"):
        error_msg = f"RPC exception: {result.get('error')}\n{result.get('traceback', '')}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)
    return result


def recv_worker(socket, pool, expected_request_id):
    try:
        frames = socket.recv_multipart()
        assert frames[0] == expected_request_id
        return _check_rpc_result(pickle.loads(frames[1]))
    finally:
        pool.put(socket)


class ExternalZeroMQDistributedExecutor(Executor):
    """An executor that engines are launched by external ray actors."""

    uses_ray: bool = False
    # Prevent assertion error from EngineArgs
    supports_pp: bool = True

    def _init_executor(self) -> None:
        dp_rank_local = self.vllm_config.parallel_config.data_parallel_rank_local
        self.dp_size = self.vllm_config.parallel_config.data_parallel_size
        self.pp_size = self.vllm_config.parallel_config.pipeline_parallel_size
        self.tp_size = self.vllm_config.parallel_config.tensor_parallel_size
        self.world_size = self.tp_size * self.pp_size * self.dp_size

        addresses = os.environ["AXON_VLLM_ZMQ_ADDRESSES"].split(",")
        addresses = addresses[
            dp_rank_local * self.tp_size * self.pp_size : (dp_rank_local + 1) * self.tp_size * self.pp_size
        ]

        self.context = zmq.Context()
        self.num_workers = len(addresses)

        # Socket pool: max_concurrent_batches sockets per worker
        self.socket_pools = []
        for address in addresses:
            pool = queue.Queue()
            for _ in range(self.max_concurrent_batches):
                socket = self.context.socket(zmq.DEALER)
                if address.startswith("tcp://["):
                    socket.setsockopt(zmq.IPV6, 1)
                socket.connect(address)
                pool.put(socket)
            self.socket_pools.append(pool)

        self.thread_pool = ThreadPoolExecutor(max_workers=self.num_workers * self.max_concurrent_batches)
        # First TP worker of the last PP stage (the rank that produces final output)
        self.output_rank = self.parallel_config.world_size - self.parallel_config.tensor_parallel_size
        # Monotonic step id for execute_model ordering across PP ranks
        self.step_id = -1
        self.step_lock = threading.Lock()

        kwargs = dict(
            vllm_config=self.vllm_config,
            local_rank=None,
            rank=None,
            distributed_init_method="env://",
            is_driver_worker=True,
        )
        self.collective_rpc("init_worker", args=([kwargs],))
        self.collective_rpc("init_device")
        self.collective_rpc("load_model")

    @property
    def max_concurrent_batches(self) -> int:
        """Ray distributed executor supports pipeline parallelism,
        meaning that it allows PP size batches to be executed concurrently.
        """
        if self.scheduler_config.async_scheduling:
            return 2
        return self.parallel_config.pipeline_parallel_size

    def shutdown(self) -> None:
        super().shutdown()
        self.thread_pool.shutdown(wait=False, cancel_futures=True)
        for pool in self.socket_pools:
            while not pool.empty():
                socket = pool.get_nowait()
                socket.close()
        self.context.destroy()

    def __del__(self):
        self.shutdown()

    def collective_rpc_serial(self, message: bytes, non_block: bool = False) -> list[Any]:
        """Serial RPC - send to all, then recv from all sequentially."""
        request_id = uuid.uuid4().bytes

        sockets = []
        for pool in self.socket_pools:
            socket = pool.get()
            socket.send_multipart([request_id, message], zmq.DONTWAIT)
            sockets.append((socket, pool))

        outputs = []
        try:
            for i, (socket, pool) in enumerate(sockets):
                frames = socket.recv_multipart()
                assert frames[0] == request_id
                outputs.append(_check_rpc_result(pickle.loads(frames[1])))
                pool.put(socket)
        except Exception:
            # Return the current (failed) and all remaining sockets to their pools
            for socket, pool in sockets[i:]:
                pool.put(socket)
            raise
        return outputs

    def collective_rpc_pipeline_parallel(
        self,
        method: str | Callable,
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict[str, Any] | None = None,
        non_block: bool = False,
    ):
        assert method == "execute_model", "Only execute_model is supported for pipeline parallel"
        if isinstance(method, str):
            sent_method = method
        else:
            sent_method = pickle.dumps(method)
        del method

        # Monotonic step_id guarantees in-order execution across PP ranks
        with self.step_lock:
            self.step_id += 1
            current_step_id = self.step_id

        message = pickle.dumps((sent_method, args, kwargs or {}, current_step_id))
        request_id = uuid.uuid4().bytes

        sockets = []
        for pool in self.socket_pools:
            socket = pool.get()
            socket.send_multipart([request_id, message], zmq.DONTWAIT)
            sockets.append((socket, pool))

        recv_futures = [self.thread_pool.submit(recv_worker, socket, pool, request_id) for socket, pool in sockets]

        if non_block:
            return recv_futures
        return [f.result() for f in recv_futures]

    def collective_rpc_threaded(self, message: bytes, non_block: bool = False) -> list[Any]:
        """Threaded RPC - send to all workers, recv in parallel via thread pool."""
        request_ids = [uuid.uuid4().bytes for _ in range(self.num_workers)]

        sockets = []
        for i, pool in enumerate(self.socket_pools):
            socket = pool.get()
            socket.send_multipart([request_ids[i], message], zmq.DONTWAIT)
            sockets.append((socket, pool, request_ids[i]))

        recv_futures = [self.thread_pool.submit(recv_worker, socket, pool, req_id) for socket, pool, req_id in sockets]

        if non_block:
            return recv_futures
        return [f.result() for f in recv_futures]

    def collective_rpc(
        self,
        method: str | Callable,
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict[str, Any] | None = None,
        non_block: bool = False,
        **kwargs_extra: Any,
    ) -> list[Any]:
        if isinstance(method, str):
            sent_method = method
        else:
            sent_method = pickle.dumps(method)
        del method

        message = pickle.dumps((sent_method, args, kwargs or {}))
        if self.num_workers == 1:
            return self.collective_rpc_serial(message, non_block=non_block)
        return self.collective_rpc_threaded(message, non_block=non_block)

    def execute_model(
        self,
        scheduler_output,
        non_block: bool = True,
    ) -> ModelRunnerOutput | Future[ModelRunnerOutput]:
        use_non_block = self.max_concurrent_batches > 1
        outputs = self.collective_rpc_pipeline_parallel(
            "execute_model", args=(scheduler_output,), non_block=use_non_block
        )
        if use_non_block:
            # output_rank indexes the first TP worker of the last PP stage
            return ZMQFutureWrapper(outputs, result_index=self.output_rank)
        # vllm v0.16+ always calls future.result() — wrap sync result in a Future
        from concurrent.futures import Future

        future: Future[ModelRunnerOutput] = Future()
        future.set_result(outputs[self.output_rank])
        return future

    def check_health(self):
        return


class vLLMHttpServerBase:
    """vLLM http server in single node, this is equivalent to launch server with command line:
    ```
    vllm serve --tensor-parallel-size=8 ...
    ```
    """

    def __init__(
        self,
        config: DictConfig,
        decoding_config: DictConfig,
        sampler_mode: SamplerMode,
        workers: list[ActorHandle],
        replica_rank: int,
        gpus_per_node: int,
        nnodes: int,
        dp_idx: int = 0,
    ):
        """
        Args:
            config: Sampler config DictConfig.
            decoding_config: Decoding config DictConfig (temperature, top_k, top_p, etc.).
            sampler_mode (SamplerMode): sampler mode.
            replica_rank (int): replica rank, a replica may contain multiple nodes.
            gpus_per_node (int): number of gpus per node.
            nnodes (int): number of nodes per DP replica (may span multiple physical nodes).
            dp_idx (int): index of this DP replica (0 = master HTTP server, >0 = headless for DP LB).
        """
        super().__init__()

        self.config = config
        self.decoding_config = decoding_config
        self.sampler_mode = sampler_mode
        self.workers = workers

        self.replica_rank = replica_rank
        self.gpus_per_node = gpus_per_node
        self.nnodes = nnodes
        self.dp_idx = dp_idx
        self.num_dp_groups = self.config.data_parallel_size

        self.max_model_len = int(self.config.max_model_len) if self.config.max_model_len else 32768

        if self.sampler_mode != SamplerMode.HYBRID and self.config.load_format == "dummy":
            logger.warning(f"Sampler mode is `{self.sampler_mode}`, load_format is dummy.")

        self._server_address = ray.util.get_node_ip_address().strip("[]")
        self._server_port = None

        # dp_idx == 0 is the master server that handles HTTP requests
        if self.dp_idx == 0:
            self._master_address = self._server_address
            self._master_port, self._master_sock = get_free_port(self._server_address)
            self._dp_master_port, self._dp_master_sock = get_free_port(self._server_address)
            logger.info(
                f"vLLMHttpServer, replica_rank: {self.replica_rank}, master address: {self._master_address}, "
                f"master port: {self._master_port}, data parallel master port: {self._dp_master_port}"
            )
        else:
            self._master_address = None
            self._master_port = None

    def get_master_address(self):
        """Get master address and port for data parallel."""
        return self._master_address, self._master_port

    def get_server_address(self):
        """Get http server address and port."""
        assert self._server_port is not None, "http server is not launched, port is None"
        return self._server_address, self._server_port

    def _build_attention_config(self) -> dict:
        # import torch
        """Build vLLM attention_config based on GPU type and model."""
        attn_config = {}
        # NOTE: this code below won't work correctly for now as cuda.is_available returns False on ray worker.
        # Need to properly fix when this becomes necessary. Currently all attention backend enforcements are handled in vllm.
        # if torch.cuda.is_available():
        #     major, _ = torch.cuda.get_device_capability(0)
        #     is_blackwell = major >= 10
        #     is_vl = any(t in self.config.model_path.lower() for t in ("vl", "vision", "visual"))

        #     if not is_blackwell:
        #         # Pre-Blackwell: force FLASH_ATTN (stable; FlashInfer may not be)
        #         attn_config["backend"] = "FLASH_ATTN"
        #     elif is_blackwell and is_vl:
        #         # Blackwell VL: enable TRTLLM (better prob_diff: 0.006 vs 0.03)
        #         attn_config["use_trtllm_attention"] = True
        #     else:
        #         # Blackwell non-VL: disable TRTLLM (repeated tokens ~2%)
        #         attn_config["use_trtllm_attention"] = False
        # else:
        #     attn_config["use_trtllm_attention"] = False
        return attn_config

    async def launch_server(self, master_address: str = None, master_port: int = None):
        if self.dp_idx != 0:
            assert master_address and master_port, "non-master DP replica should provide master address and port"
            self._master_address = master_address
            self._master_port = master_port

        # 1. setup vllm serve cli args
        engine_kwargs = self.config.get("engine_kwargs", {}).get("vllm", {}) or {}
        engine_kwargs = {key: val for key, val in engine_kwargs.items() if val is not None}
        if self.config.get("limit_images", None):  # support for multi-image data
            engine_kwargs["limit_mm_per_prompt"] = {"image": self.config.get("limit_images")}
        if self.config.cudagraph_capture_sizes:
            engine_kwargs["cuda_graph_sizes"] = self.config.cudagraph_capture_sizes

        # Override HF model generation_config defaults to match training config.
        override_generation_config = dict(
            max_new_tokens=self.config.max_model_len,
        )
        # Critical: some models ship with non-trivial defaults (e.g. Qwen2.5-VL has
        # repetition_penalty=1.05) that would silently distort vLLM log-probs vs the
        # FSDP forward pass which applies no such penalty.
        override_generation_config.update(
            temperature=self.decoding_config.temperature,
            top_k=self.decoding_config.top_k,
            top_p=self.decoding_config.top_p,
            repetition_penalty=self.decoding_config.repetition_penalty,
        )
        logger.info(f"override_generation_config: {override_generation_config}")

        quantization = self.config.quantization
        if quantization is not None:
            if quantization in ("fp8", "fp8_fast"):
                from axon.monkey_patches.vllm.fp8.vllm_fp8_patch import FP8_BLOCK_QUANT_KWARGS, patch_vllm_fp8

                fp8_block_quant_kwargs = dict(FP8_BLOCK_QUANT_KWARGS)
                patch_vllm_fp8(quantization)
            elif quantization == "int4":
                # INT4 uses vLLM's compressed-tensors quantization natively
                # for loading pre-converted INT4 checkpoints (load_format=auto).
                # Online weight sync is NOT supported with vLLM due to Marlin
                # kernel repacking.  Use SGLang backend for online INT4 sync.
                pass
            elif quantization == "mxfp8":
                # MxFP8 (group-wise FP8 with UE8M0 scales) is not natively
                # supported by vLLM.  Fall back to standard FP8 blockwise
                # quantization.  True MxFP8 is only available on SGLang.
                from axon.monkey_patches.vllm.fp8.vllm_fp8_patch import FP8_BLOCK_QUANT_KWARGS, patch_vllm_fp8

                logger.warning(
                    "MxFP8 not supported by vLLM; using standard FP8 blockwise [128,128]. "
                    "Use the SGLang backend for true MxFP8 with UE8M0 scales."
                )
                fp8_block_quant_kwargs = dict(FP8_BLOCK_QUANT_KWARGS)
                patch_vllm_fp8("fp8")
            else:
                raise ValueError(f"Unsupported quantization method: {quantization}")
        max_num_batched_tokens = max(self.config.get("max_num_batched_tokens", 32768), self.max_model_len)

        hf_overrides = {
            "max_position_embeddings": self.max_model_len,
        }
        if quantization in ("fp8", "fp8_fast", "mxfp8"):
            hf_overrides["quantization_config"] = fp8_block_quant_kwargs
        elif quantization == "int4":
            hf_overrides["quantization_config"] = {
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

        args = {
            "dtype": self.config.dtype,
            "load_format": "dummy" if self.config.load_format.startswith("dummy") else self.config.load_format,
            "skip_tokenizer_init": self.config.skip_tokenizer_init,
            "trust_remote_code": self.config.trust_remote_code,
            "max_model_len": self.max_model_len,
            "max_num_seqs": self.config.max_num_seqs,
            "enable_chunked_prefill": self.config.enable_chunked_prefill,
            "max_num_batched_tokens": max_num_batched_tokens,
            "enable_prefix_caching": self.config.enable_prefix_caching,
            # Use "processor_only" mm cache — axon's external executor can't provide
            # IPC for the default "lru" sender/receiver cache split.
            "mm_processor_cache_type": "processor_only",
            "enable_sleep_mode": True,
            "disable_custom_all_reduce": True,
            # Attention backend configuration:
            # - Pre-Blackwell: force FLASH_ATTN (FA2 is stable, FlashInfer may not be)
            # - Blackwell: auto backend (FlashInfer default). TRTLLM attention enabled
            #   only for VL models (better prob_diff: 0.006 vs 0.03 for Qwen2.5-VL).
            "attention_config": json.dumps(self._build_attention_config()),
            # Disable FlashInfer allreduce fusion — requires IPC between GPUs
            # which doesn't work with Ray's per-actor GPU isolation.
            "compilation_config": json.dumps({"pass_config": {"fuse_allreduce_rms": False}}),
            "enforce_eager": self.config.enforce_eager,
            "gpu_memory_utilization": self.config.gpu_memory_utilization,
            "disable_log_stats": self.config.disable_log_stats,
            "tensor_parallel_size": self.config.tensor_model_parallel_size,
            "pipeline_parallel_size": self.config.pipeline_model_parallel_size,
            "seed": self.config.get("seed", 0),
            "override_generation_config": json.dumps(override_generation_config),
            "quantization": {
                "fp8": "fp8",
                "fp8_fast": "fp8",
                "mxfp8": "fp8",  # MxFP8 loads as FP8 in vLLM with custom block config
                "int4": "compressed-tensors",  # INT4 loads as compressed-tensors in vLLM
            }.get(quantization, quantization),
            "hf_overrides": hf_overrides,
            **engine_kwargs,
        }

        if self.config.speculative_config:
            spec_cfg = dict(self.config.speculative_config)
            spec_cfg.setdefault("disable_logprobs", False)
            args["speculative_config"] = json.dumps(spec_cfg)

        # Data parallel: pass DP args when data_parallel_size > 1
        # See: https://docs.vllm.ai/en/latest/serving/data_parallel_deployment/
        if self.config.data_parallel_size > 1:
            tp_size = self.config.tensor_model_parallel_size
            pp_size = self.config.pipeline_model_parallel_size
            model_parallel_size = tp_size * pp_size

            assert len(self.workers) % model_parallel_size == 0, (
                f"num workers ({len(self.workers)}) should be divisible by "
                f"model_parallel_size (tp={tp_size} * pp={pp_size} = {model_parallel_size})"
            )
            data_parallel_size_local = len(self.workers) // model_parallel_size

            args.update(
                {
                    "data_parallel_size": self.config.data_parallel_size,
                    "data_parallel_size_local": data_parallel_size_local,
                    "data_parallel_start_rank": self.dp_idx * data_parallel_size_local,
                    "data_parallel_address": self._master_address,
                    "data_parallel_rpc_port": self._master_port,
                }
            )

        if self.config.expert_parallel_size > 1:
            assert self.config.pipeline_model_parallel_size == 1, (
                "pipeline_model_parallel_size must be 1 when expert_parallel_size > 1"
            )
            args.update(
                {
                    "enable_expert_parallel": True,
                    "all2all_backend": self.config.all2all_backend,
                    "enable_eplb": self.config.enable_eplb,
                }
            )

        if self.config.lora_rank > 0:
            args.update(
                {
                    "enable_lora": True,
                    "max_loras": 1,
                    "max_lora_rank": get_vllm_max_lora_rank(self.config.lora_rank),
                }
            )

        # MOE routing replay: use vllm's native routed_experts capture.
        if os.environ.get("AXON_MOE_REPLAY") == "1":
            from transformers import AutoConfig

            # model_path is operator-supplied (same model the run trains); HF revision pinning N/A.
            _hf_cfg = AutoConfig.from_pretrained(self.config.model_path, trust_remote_code=False)  # nosec B615
            _text_cfg = getattr(_hf_cfg, "text_config", _hf_cfg)
            _has_moe = bool(
                getattr(_text_cfg, "num_experts", None)
                or getattr(_text_cfg, "num_local_experts", None)
                or getattr(_text_cfg, "n_routed_experts", None)
            )
            if _has_moe:
                args["enable_return_routed_experts"] = True

        server_args = ["serve", self.config.model_path]
        # vLLM BooleanOptionalAction args that support --no-{k} to explicitly set False.
        # Other bool args use store_true and should simply be omitted when False.
        _BOOL_OPTIONAL_ARGS = {"enable_prefix_caching", "enable_chunked_prefill"}
        for k, v in args.items():
            if isinstance(v, bool):
                if v:
                    server_args.append(f"--{k}")
                elif k in _BOOL_OPTIONAL_ARGS:
                    server_args.append(f"--no-{k}")
            elif v is not None:
                server_args.append(f"--{k}")
                # Use json.dumps for dict to ensure valid JSON format
                server_args.append(json.dumps(v) if isinstance(v, dict) else str(v))

        if self.replica_rank == 0:
            pprint(server_args)

        CMD_MODULES = [vllm.entrypoints.cli.serve]
        parser = FlexibleArgumentParser(description="vLLM CLI")
        subparsers = parser.add_subparsers(required=False, dest="subparser")
        cmds = {}
        for cmd_module in CMD_MODULES:
            new_cmds = cmd_module.cmd_init()
            for cmd in new_cmds:
                cmd.subparser_init(subparsers).set_defaults(dispatch_function=cmd.cmd)
                cmds[cmd.name] = cmd
        server_args = parser.parse_args(args=server_args)
        server_args.model = server_args.model_tag
        if server_args.subparser in cmds:
            cmds[server_args.subparser].validate(server_args)

        # 2. setup distributed executor backend
        server_args.distributed_executor_backend = ExternalZeroMQDistributedExecutor if self.workers else None

        zmq_addresses = ray.get([worker.get_zeromq_address.remote() for worker in self.workers])
        logger.info(
            f"replica_rank={self.replica_rank}, dp_idx={self.dp_idx}, nnodes={self.nnodes}, "
            f"get worker zmq addresses: {zmq_addresses}"
        )
        os.environ["AXON_VLLM_ZMQ_ADDRESSES"] = ",".join(zmq_addresses)

        # vLLM expects CUDA_VISIBLE_DEVICES to have at least world_size entries for its internal checks.
        # The actual GPU work is done by external workers via ZMQ, so we just need enough entries
        # to pass vLLM's validation. Use modulo to map to available GPUs.
        # On ROCm, also set HIP_VISIBLE_DEVICES.
        from axon.utils.rocm_utils import get_visible_devices_env_key

        tp_size = self.config.tensor_model_parallel_size
        pp_size = self.config.pipeline_model_parallel_size
        world_size = tp_size * pp_size
        vis_key = get_visible_devices_env_key()
        if vis_key not in os.environ or not os.environ.get(vis_key):
            os.environ[vis_key] = ",".join(str(i % self.gpus_per_node) for i in range(world_size))
        # vLLM always checks CUDA_VISIBLE_DEVICES even on ROCm, so mirror the value
        if "CUDA_VISIBLE_DEVICES" not in os.environ or not os.environ["CUDA_VISIBLE_DEVICES"]:
            os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get(vis_key, os.environ.get("CUDA_VISIBLE_DEVICES", ""))

        # 3. launch server: dp_idx=0 runs HTTP server, others run headless for DP load balancing
        if self.dp_idx == 0:
            await self.run_server(server_args)
        else:
            await self.run_headless(server_args)

    async def run_server(self, args: argparse.Namespace):
        engine_args = AsyncEngineArgs.from_cli_args(args)
        usage_context = UsageContext.OPENAI_API_SERVER
        vllm_config = engine_args.create_engine_config(usage_context=usage_context)
        vllm_config.parallel_config.data_parallel_master_port = self._dp_master_port

        fn_args = set(inspect.signature(AsyncLLM.from_vllm_config).parameters)
        kwargs = {}
        if "enable_log_requests" in fn_args:
            kwargs["enable_log_requests"] = engine_args.enable_log_requests
        if "disable_log_stats" in fn_args:
            kwargs["disable_log_stats"] = engine_args.disable_log_stats

        engine_client = AsyncLLM.from_vllm_config(vllm_config=vllm_config, usage_context=usage_context, **kwargs)

        # Don't keep the dummy data in memory
        await engine_client.reset_mm_cache()

        app = build_app(args)
        await init_app_state(engine_client, app.state, args)
        if self.replica_rank == 0 and self.dp_idx == 0:
            logger.info(f"Initializing a V1 LLM engine with config: {vllm_config}")

        self.engine = engine_client
        self._server_port, self._server_task = await run_unvicorn(app, args, self._server_address)

        @app.post("/pause_generation")
        async def pause_generation(raw_request: Request):
            return await self.pause_generation(raw_request)

        @app.post("/continue_generation")
        async def continue_generation(raw_request: Request):
            return await self.continue_generation(raw_request)

    async def run_headless(self, args: argparse.Namespace):
        engine_args = vllm.AsyncEngineArgs.from_cli_args(args)
        usage_context = UsageContext.OPENAI_API_SERVER
        vllm_config = engine_args.create_engine_config(usage_context=usage_context, headless=True)

        parallel_config = vllm_config.parallel_config
        local_engine_count = parallel_config.data_parallel_size_local

        host = parallel_config.data_parallel_master_ip
        port = engine_args.data_parallel_rpc_port
        handshake_address = get_tcp_uri(host, port)

        self.engine_manager = CoreEngineProcManager(
            target_fn=EngineCoreProc.run_engine_core,
            local_engine_count=local_engine_count,
            start_index=vllm_config.parallel_config.data_parallel_rank,
            local_start_index=0,
            vllm_config=vllm_config,
            local_client=False,
            handshake_address=handshake_address,
            executor_class=Executor.get_class(vllm_config),
            log_stats=not engine_args.disable_log_stats,
        )

    async def wake_up(self):
        if self.sampler_mode == SamplerMode.HYBRID:
            await asyncio.gather(*[worker.wake_up.remote() for worker in self.workers])
        elif self.sampler_mode == SamplerMode.COLOCATED:
            if self.dp_idx == 0:
                await self.engine.wake_up(tags=["kv_cache", "weights"])
        elif self.sampler_mode == SamplerMode.STANDALONE:
            logger.info("skip wake_up in standalone mode")

    async def sleep(self):
        if self.sampler_mode == SamplerMode.HYBRID:
            if self.dp_idx == 0:
                await self.engine.reset_prefix_cache()
            await asyncio.gather(*[worker.sleep.remote() for worker in self.workers])
        elif self.sampler_mode == SamplerMode.COLOCATED:
            if self.dp_idx == 0:
                await self.engine.reset_prefix_cache()
                await self.engine.sleep(level=1)
        elif self.sampler_mode == SamplerMode.STANDALONE:
            logger.info("skip sleep in standalone mode")

    async def wait_for_requests_to_drain(self):
        await self.engine.wait_for_requests_to_drain()

    async def pause_generation(self, raw_request: Request):
        logger.info("vLLM server received pause_generation")
        request_json = await raw_request.json()
        strategy = request_json.get("strategy", "empty")
        if strategy not in ["drain", "hold", "continue", "reset"]:
            return JSONResponse({"success": False, "content": f"{strategy} not supported"}, status_code=400)

        engine_client = self.engine
        match strategy:
            case "drain":
                # Wait for all in-flight requests to finish, then abort + clear caches
                while True:
                    try:
                        await engine_client.wait_for_requests_to_drain(drain_timeout=300)
                        break
                    except Exception:
                        print("wait_for_requests_to_drain times out, retrying")
                await engine_client.pause_generation(mode="abort", clear_cache=True)
            case "hold":
                # Freeze all requests in place, keep KV cache intact for fast resume
                await engine_client.pause_generation(mode="keep", clear_cache=False)
            case "continue":
                # Freeze all requests, preempt running→waiting, clear caches
                # Requests survive (not aborted) but need re-prefill on resume
                await engine_client.pause_generation(mode="keep", clear_cache=True)
            case "reset":
                # Abort all requests (clients notified), clear caches
                await engine_client.pause_generation(mode="abort", clear_cache=True)
        return JSONResponse({"success": True}, status_code=200)

    async def continue_generation(self, raw_request: Request):
        logger.info("vLLM server received continue_generation")
        await self.engine.resume_generation()
        return JSONResponse({"success": True}, status_code=200)


@ray.remote(num_cpus=1)
class vLLMHttpServer(vLLMHttpServerBase):
    """Ray actor wrapper for vLLMHttpServerBase."""

    pass


_sampler_worker_actor_cls = ray.remote(vLLMEngine)


@register_server("vllm")
class vLLMServer(Server):
    def __init__(
        self,
        replica_rank: int,
        config: DictConfig,
        decoding_config: DictConfig,
        gpus_per_node: int = 8,
        is_reward_model: bool = False,
    ):
        super().__init__(replica_rank, config, decoding_config, gpus_per_node, is_reward_model)
        self.server_class = vLLMHttpServer

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

        worker_node_ids = await asyncio.gather(
            *[
                worker.__ray_call__.remote(lambda self: ray.get_runtime_context().get_node_id())
                for worker in self.workers
            ]
        )

        # Calculate how many nodes each DP replica spans
        tp_size = self.config.tensor_model_parallel_size
        pp_size = self.config.pipeline_model_parallel_size
        model_parallel_size = tp_size * pp_size
        nodes_per_dp_replica = (model_parallel_size + self.gpus_per_node - 1) // self.gpus_per_node

        # Launch a server on the first node of each DP replica (workers connect via ZMQ)
        for dp_idx in range(self.config.data_parallel_size):
            start_worker = dp_idx * model_parallel_size
            end_worker = (dp_idx + 1) * model_parallel_size
            workers = self.workers[start_worker:end_worker]
            node_id = worker_node_ids[start_worker]
            name = (
                f"vllm_server_{self.replica_rank}_{dp_idx}"
                if not self.is_reward_model
                else f"vllm_server_reward_{self.replica_rank}_{dp_idx}"
            )

            server = self.server_class.options(
                scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                    node_id=node_id,
                    soft=False,
                ),
                name=name,
            ).remote(
                config=self.config,
                decoding_config=self.decoding_config,
                sampler_mode=self.sampler_mode,
                workers=workers,
                replica_rank=self.replica_rank,
                gpus_per_node=self.gpus_per_node,
                nnodes=nodes_per_dp_replica,  # Nodes per DP replica
                dp_idx=dp_idx,  # 0=master HTTP server, >0=headless for DP LB
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

    async def sleep(self):
        """Sleep each sampler server."""
        # Drain DP engines for safe sleep.
        await self.servers[0].wait_for_requests_to_drain.remote()
        await asyncio.gather(*[server.sleep.remote() for server in self.servers])
