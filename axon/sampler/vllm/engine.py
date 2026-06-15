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
"""vLLM engine wrapper for a single worker process.

Supports FSDP (DTensor/HF weight loader) and Megatron backends.
In Megatron mode, PP-rank parameters are broadcast before inference and freed after.
"""

import asyncio
import getpass
import logging
import os
from collections.abc import Generator
from dataclasses import asdict
from typing import Any

import cloudpickle as pickle
import ray
import torch
import torch.distributed
import zmq
import zmq.asyncio
from filelock import FileLock
from omegaconf import DictConfig
from torch.distributed.device_mesh import DeviceMesh
from vllm.config import LoRAConfig
from vllm.v1.worker.worker_base import WorkerWrapperBase

from axon.monkey_patches.vllm.fp8.vllm_fp8_patch import apply_vllm_fp8_patches, is_fp8_model, load_quanted_weights
from axon.sampler.base.engine import Engine
from axon.utils.networking_utils import get_free_port, is_valid_ipv6_address
from axon.utils.ray.utils import ray_noset_visible_devices
from axon.utils.tokenizer import hf_tokenizer
from axon.utils.vllm import TensorLoRARequest, VLLMHijack
from axon.utils.vllm.lora_utils import (
    VLLM_LORA_INT_ID,
    VLLM_LORA_NAME,
    VLLM_LORA_PATH,
    get_vllm_max_lora_rank,
)

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("AXON_LOGGING_LEVEL", "WARN"))

VLLMHijack.hijack()


class vLLMEngine(Engine):
    """vLLMEngine is a thin wrapper of WorkerWrapperBase, which is engine in single worker process."""

    def __init__(
        self,
        config: DictConfig,
        device_mesh: DeviceMesh,
    ):
        super().__init__(config, device_mesh)
        # Load tokenizer from config.model_path for vocab_size used in _monkey_patch_compute_logits
        local_path = self.config.model_path
        trust_remote_code = self.config.get("trust_remote_code", False)
        self.tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)

        self.inference_engine: WorkerWrapperBase = None

        self.tensor_parallel_size = self.config.tensor_model_parallel_size
        self.pipeline_parallel_size = self.config.pipeline_model_parallel_size
        self.model_parallel_size = self.tensor_parallel_size * self.pipeline_parallel_size
        self.rank = int(os.environ["RANK"])
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.world_size = int(os.environ.get("WORLD_SIZE", self.model_parallel_size))
        self.rpc_rank = self.rank % self.model_parallel_size
        # Execute_model output rank is the last TP rank of the last PP stage.
        self.output_rank = self.model_parallel_size - self.tensor_parallel_size
        # Enforce in-order execute_model across PP ranks using a step_id.
        # Each worker expects step_id to increase monotonically starting at 0.
        self.step_id: int = 0
        # Buffer of out-of-order execute_model calls: step_id -> (client_id, request_id, method, args, kwargs)
        self.pending_requests: dict[int, Any] = {}

        self.address = self._init_zeromq()
        self.lora_config = (
            {"max_loras": 1, "max_lora_rank": get_vllm_max_lora_rank(self.config.lora_rank)}
            if self.config.lora_rank > 0
            else {}
        )

        self.sleep_level = 1

    def _init_zeromq(self) -> str:
        # single node: ipc, multi nodes: tcp
        local_world_size = int(os.environ["RAY_LOCAL_WORLD_SIZE"])
        socket_type = "ipc" if self.world_size <= local_world_size else "tcp"

        # File lock to prevent multiple workers from binding to the same port
        with FileLock(f"/tmp/axon_vllm_zmq_{getpass.getuser()}.lock"):  # nosec B108
            context = zmq.asyncio.Context()
            self.socket = context.socket(zmq.ROUTER)
            if socket_type == "ipc":
                pid = os.getpid()
                address = f"ipc:///tmp/axon_vllm_zmq_{pid}_{getpass.getuser()}.ipc"
            else:
                ip = ray.util.get_node_ip_address().strip("[]")
                port, self._tcp_sock = get_free_port(ip)
                if is_valid_ipv6_address(ip):
                    address = f"tcp://[{ip}]:{port}"
                    self.socket.setsockopt(zmq.IPV6, 1)
                else:
                    address = f"tcp://{ip}:{port}"
            self.socket.bind(address)

        loop = asyncio.get_running_loop()
        self.zmq_loop_task = loop.create_task(self._loop_forever())

        return address

    async def _execute_method_and_send_response(
        self, client_id: bytes, request_id: bytes, method: str | bytes, *args, **kwargs
    ):
        try:
            result = await self._execute_method(method, *args, **kwargs)
        except Exception as e:
            import traceback

            tb = traceback.format_exc()
            logger.error(f"Error executing method {method}: {e}\n{tb}")
            print(f"[vLLMEngine rank={self.rank}] Error executing method {method}: {e}\n{tb}", flush=True)
            result = {"__exception__": True, "error": str(e), "traceback": tb}
        # For execute_model with PP, only the first TP rank per PP stage sends the real result;
        # redundant TP ranks send None (they hold identical data).
        is_redundant_tp = method == "execute_model" and (self.rpc_rank % self.tensor_parallel_size) != 0
        response = pickle.dumps(None if is_redundant_tp else result)
        await self.socket.send_multipart([client_id, request_id, response])

    async def _loop_forever(self):
        while True:
            # ROUTER receives: [client_id, request_id, payload]
            frames = await self.socket.recv_multipart()
            client_id, request_id, payload = frames
            decoded = pickle.loads(payload)

            if len(decoded) == 4:  # execute_model method
                method, args, kwargs, step_id = decoded
            else:
                method, args, kwargs = decoded
                step_id = None

            # execute_model for PP is a special case. Requests must be executed in order.
            if method == "execute_model":
                assert step_id is not None, "step_id is required for execute_model"
                step_id = int(step_id)
                if step_id == self.step_id:
                    self.step_id += 1
                    await self._execute_method_and_send_response(client_id, request_id, method, *args, **kwargs)
                else:
                    self.pending_requests[step_id] = (client_id, request_id, method, args, kwargs)

                while self.step_id in self.pending_requests:
                    q_client_id, q_request_id, q_method, q_args, q_kwargs = self.pending_requests.pop(self.step_id)
                    await self._execute_method_and_send_response(
                        q_client_id, q_request_id, q_method, *q_args, **q_kwargs
                    )
                    self.step_id += 1
                continue

            # Default: immediate execution for all other methods
            await self._execute_method_and_send_response(client_id, request_id, method, *args, **kwargs)

    def _init_worker(self, all_kwargs: list[dict[str, Any]]):
        """Initialize worker engine."""
        # WorkerWrapperBase.init_worker expects all_kwargs indexed by rpc_rank
        indexed_kwargs = all_kwargs[0]
        all_kwargs = [None] * self.model_parallel_size
        all_kwargs[self.rpc_rank] = indexed_kwargs

        all_kwargs[self.rpc_rank]["rank"] = int(os.environ["RANK"])
        # set local_rank to 0 as ray sets CUDA_VISIBLE_DEVICES to a single device for each rank
        all_kwargs[self.rpc_rank]["local_rank"] = (
            0 if not ray_noset_visible_devices() else int(os.environ.get("RAY_LOCAL_RANK", 0))
        )
        self.vllm_config = all_kwargs[self.rpc_rank]["vllm_config"]
        if self.lora_config:
            lora_dtype = getattr(torch, self.config.dtype)
            self.vllm_config.lora_config = LoRAConfig(lora_dtype=lora_dtype, **self.lora_config)
        if self.config.quantization is not None:
            if self.config.quantization in ("fp8", "fp8_fast"):
                apply_vllm_fp8_patches()
            elif self.config.quantization in ("int4", "mxfp8"):
                # INT4 and MxFP8 quantization is handled at weight-sync time
                # (in the SGLang engine path or via pre-converted checkpoints).
                # For vLLM, the model loads with compressed-tensors / fp8 format
                # natively, so no extra patches are needed here.
                pass
            else:
                raise ValueError(f"Unsupported quantization method: {self.config.quantization}")
        self.inference_engine = WorkerWrapperBase(rpc_rank=self.rpc_rank)
        self.inference_engine.init_worker(all_kwargs)

        # Reset PP ordering state on (re)initialization
        self.step_id = 0
        self.pending_requests.clear()

    def _load_model(self, *args, **kwargs):
        self.inference_engine.load_model(*args, **kwargs)

    async def _execute_method(self, method, *args, **kwargs):
        if method == "init_worker":
            return self._init_worker(*args, **kwargs)
        elif method == "load_model":
            return self._load_model(*args, **kwargs)
        elif callable(method):
            # Callable RPC — direct function dispatch
            return method(self.inference_engine, *args, **kwargs)
        elif isinstance(method, bytes):
            # Pickled callable — deserialize and invoke with worker wrapper as first arg
            import cloudpickle

            func = cloudpickle.loads(method)
            return func(self.inference_engine, *args, **kwargs)
        else:
            # vLLM v0.19.0 removed execute_method from WorkerWrapperBase.
            # Resolve the method directly on the worker or wrapper.
            target = self.inference_engine
            fn = getattr(target, method, None)
            if fn is None and hasattr(target, "worker"):
                fn = getattr(target.worker, method, None)
            if fn is None:
                raise AttributeError(f"Neither WorkerWrapperBase nor Worker has method '{method}'")
            return fn(*args, **kwargs)

    async def resume(self, tags: list[str]):
        """Resume sampler memory by tag in GPU memory.

        Args:
            tags: one or more of "weights", "kv_cache", "default".
        """
        if self.config.offload_sampler:
            self.inference_engine.wake_up(tags=tags)
            # Synchronize to surface any async CUDA errors from create_and_map
            # (e.g., OOM during kv_cache re-mapping) immediately here rather than
            # silently propagating to a later kernel launch in sample_tokens.
            torch.cuda.synchronize()

    async def zero_mamba_cache(self):
        """Zero Mamba/GDN state caches after KV cache resume.

        After sleep→wake_up, KV cache memory is re-mapped but NOT zeroed.
        For standard attention this is fine (KV entries overwritten during prefill).
        For Mamba/GDN models (e.g. Qwen3.5), conv_state and ssm_state are part
        of the KV cache and may be read before being fully overwritten.
        Zeroing ensures deterministic initial state across all TP ranks.
        """
        model_runner = getattr(self.inference_engine, "worker", self.inference_engine)
        if hasattr(model_runner, "model_runner"):
            model_runner = model_runner.model_runner
        kv_caches = getattr(model_runner, "kv_caches", None)
        if not kv_caches:
            return

        # Also clear the mamba_state_idx mapping to prevent stale entries
        # from previous sessions pointing to stale cache blocks.
        mamba_state_idx = getattr(model_runner, "mamba_state_idx", None)
        if mamba_state_idx is None:
            # Standard attention model (no Mamba/GDN states) — KV entries are
            # overwritten during prefill so no zeroing is needed or safe here.
            # Calling c.zero_() on re-mapped KV cache tensors can silently swallow
            # an async CUDA error (AcceleratorError IS-A RuntimeError) from a failed
            # create_and_map, masking the root cause until sample_tokens crashes.
            return
        mamba_state_idx.clear()
        logger.info("Cleared mamba_state_idx after resume")

        num_zeroed = 0
        for cache in kv_caches:
            tensors = cache if isinstance(cache, (list | tuple)) else [cache]
            for c in tensors:
                if not isinstance(c, torch.Tensor):
                    continue
                # Safety: check tensor has valid CUDA storage before zeroing.
                # During sleep/wake transitions, storage may be temporarily freed.
                try:
                    if c.is_cuda and c.numel() > 0 and c.storage().size() > 0:
                        c.zero_()
                        num_zeroed += 1
                except RuntimeError:
                    # Storage not available yet — skip, will be initialized on first use
                    pass
        if num_zeroed > 0:
            logger.info(f"Zeroed {num_zeroed} KV/Mamba state cache tensors after resume")

    async def release(self):
        """Release weights and kv cache in GPU memory."""
        if self.config.offload_sampler:
            self.inference_engine.sleep(level=self.sleep_level)

    async def update_weights(self, weights: Generator[tuple[str, torch.Tensor], None, None], **kwargs):
        """Update the weights of the sampler model.

        Args:
            weights: A generator that yields the name of the weight tensor and the tensor itself.
        """
        peft_config = kwargs.get("peft_config")
        base_sync_done = kwargs.get("base_sync_done", False)
        if peft_config and base_sync_done:
            # Remove old LoRA before adding new one (required in async mode)
            self.inference_engine.worker.remove_lora(VLLM_LORA_INT_ID)
            weights = dict(weights)
            lora_request = TensorLoRARequest(
                lora_name=VLLM_LORA_NAME,
                lora_int_id=VLLM_LORA_INT_ID,
                lora_path=VLLM_LORA_PATH,
                peft_config=asdict(peft_config),
                lora_tensors=weights,
            )
            self.inference_engine.worker.add_lora(lora_request)
            logger.info(f"vLLM load weights, loaded_params: {len(weights)}")
        else:
            # moe_weight_loader patch removed — vllm v0.16+ sets weight_loader
            # natively via FusedMoE.create_weights() → set_weight_attrs()
            model_runner = self.inference_engine.worker.model_runner
            model = model_runner.model

            quantization = self.config.get("quantization")

            if quantization == "int4":
                # INT4 online weight sync is NOT supported on vLLM because the
                # Marlin kernel does destructive repacking (gptq_marlin_repack)
                # during process_weights_after_loading.  Raw packed INT4 weights
                # cannot be re-loaded on top of Marlin-repacked parameters.
                #
                # For INT4 with vLLM:
                #   - Use a pre-converted INT4 checkpoint (scripts/convert_hf_to_int4.py)
                #   - Set load_format="auto" instead of "dummy"
                #   - Weight sync is skipped — the initial checkpoint weights are used
                #
                # For INT4 with online weight sync during RL training, use the
                # SGLang backend which handles compressed-tensors natively.
                raise NotImplementedError(
                    "INT4 online weight sync is not supported with vLLM because "
                    "Marlin's kernel repacking is not reversible. Use the SGLang "
                    "backend (sampler.name=sglang) for INT4 online weight sync, "
                    "or use a pre-converted INT4 checkpoint with load_format=auto."
                )

            elif quantization == "mxfp8":
                # MxFP8 (group-wise FP8 with UE8M0 uint8 scales, [1,32] blocks)
                # is NOT supported by vLLM's FP8 path because:
                # 1. scaled_fp8_blockwise asserts block_size0 == block_size1
                #    (fails for [1, 32])
                # 2. vLLM 0.10 has no native MxFP8 handler
                #
                # Fall back to standard FP8 blockwise [128, 128] quantization.
                # For true MxFP8, use the SGLang backend.
                if is_fp8_model(model_runner.vllm_config):
                    logger.warning(
                        "MxFP8 [1,32] blocks not supported by vLLM; "
                        "falling back to standard FP8 blockwise [128,128]. "
                        "For true MxFP8, use sampler.name=sglang."
                    )
                    loaded_params = load_quanted_weights(weights, model_runner)
                    logger.info(f"FP8 (fallback) weights loaded, loaded_params: {len(loaded_params)}")
                else:
                    logger.warning("MxFP8 requested but model not loaded as FP8; loading raw weights")
                    model.load_weights(weights)

            elif is_fp8_model(model_runner.vllm_config):
                logger.info(f"FP8 model detected: {model_runner.vllm_config.quant_config}")
                loaded_params = load_quanted_weights(weights, model_runner)
                logger.info(f"FP8 weights loaded, loaded_params: {len(loaded_params)}")
            else:
                # VL models (e.g. Qwen3.5) have two sub-models in vLLM:
                #   self.visual       -> expects keys like "visual.*"
                #   self.language_model -> expects keys like "language_model.model.layers.*"
                #
                # HF VL models export state dict keys as:
                #   model.visual.*              (vision encoder)
                #   model.language_model.*      (text model)
                #   lm_head.weight              (output head)
                #
                # Megatron (via mbridge) exports text-only keys as:
                #   model.layers.*              (text model)
                #   model.embed_tokens.weight
                #   lm_head.weight
                #
                # Remap both formats to match vLLM's expected structure.
                if hasattr(model, "language_model"):
                    _hf_vl_lm_prefix = "model.language_model."
                    _hf_vl_visual_prefix = "model.visual."

                    def _remap_vl_keys(ws):
                        for name, tensor in ws:
                            if name.startswith(_hf_vl_lm_prefix):
                                # HF VL text key: model.language_model.X -> language_model.model.X
                                yield f"language_model.model.{name[len(_hf_vl_lm_prefix) :]}", tensor
                            elif name.startswith(_hf_vl_visual_prefix):
                                # HF VL visual key: model.visual.X -> visual.X
                                yield name[len("model.") :], tensor
                            elif name.startswith("visual."):
                                # Megatron/mbridge VL visual key already without "model." prefix:
                                # visual.X -> visual.X (pass through unchanged)
                                yield name, tensor
                            else:
                                # Text-only keys (Megatron) or top-level keys (lm_head):
                                # model.layers.X -> language_model.model.layers.X
                                # lm_head.weight -> language_model.lm_head.weight
                                yield f"language_model.{name}", tensor

                    weights = _remap_vl_keys(weights)

                # vLLM's load_weights() handles packing individual HF projections
                # into fused modules (e.g. q_proj+k_proj+v_proj -> qkv_proj)
                # and both old per-expert (experts.N.gate_proj) and transformers 5.x
                # stacked expert format (experts.gate_up_proj) for MoE models.
                logger.info("Loading standard weights (non-FP8)")
                model.load_weights(weights)

    def get_zeromq_address(self):
        return self.address
