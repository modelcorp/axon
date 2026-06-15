# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
from __future__ import annotations

import logging
import multiprocessing as mp
import os
from collections.abc import Generator

import ray
import sglang.srt.entrypoints.engine
import torch
from omegaconf import DictConfig
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import (
    assert_pkg_version,
    is_cuda,
    set_prometheus_multiproc_dir,
    set_ulimit,
)
from sglang.srt.weight_sync.utils import update_weights as sgl_update_weights
from torch.distributed.device_mesh import DeviceMesh

from axon.sampler.base.engine import Engine
from axon.utils.networking_utils import is_valid_ipv6_address
from axon.utils.sglang.http_server_engine import AsyncHttpServerAdapter
from axon.utils.sglang.sampler_utils import get_named_tensor_buckets

_UPDATE_WEIGHTS_BUCKET_BYTES = 512 << 20  # 512 MB

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("AXON_LOGGING_LEVEL", "WARN"))


# Patch to avoid issue https://github.com/sgl-project/sglang/issues/6723
def _set_envs_and_config(server_args: ServerArgs):
    env = {
        "TF_CPP_MIN_LOG_LEVEL": "3",
        "NCCL_CUMEM_ENABLE": "0",
        "NCCL_NVLS_ENABLE": str(int(server_args.enable_nccl_nvls)),
        "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
        "CUDA_DEVICE_MAX_CONNECTIONS": "4",
        "CUDA_MODULE_LOADING": "AUTO",
    }
    # Merge ROCm-specific performance env vars when running on HIP
    from axon.utils.rocm_utils import get_rocm_env_vars

    env.update(get_rocm_env_vars())
    os.environ.update(env)

    # Monkey-patch SGLang's CustomAllreduce for ROCm (no-op on CUDA)
    from axon.monkey_patches.sglang.rocm_allreduce import apply_sglang_rocm_allreduce_patch

    apply_sglang_rocm_allreduce_patch()

    if server_args.enable_metrics:
        set_prometheus_multiproc_dir()

    set_ulimit()

    if server_args.attention_backend == "flashinfer":
        assert_pkg_version(
            "flashinfer_python",
            "0.2.5",
            "Please uninstall the old version and reinstall the latest version by following the instructions at https://docs.flashinfer.ai/installation.html.",
        )
    if is_cuda():
        assert_pkg_version(
            "sgl-kernel",
            "0.1.1",
            "Please reinstall the latest version with `pip install sgl-kernel --force-reinstall`",
        )

    mp.set_start_method("spawn", force=True)


sglang.srt.entrypoints.engine._set_envs_and_config = _set_envs_and_config


class SGLangEngine(Engine):
    """SGLang engine: HTTP client for weight sync and memory management against the SGLang server.

    In hybrid mode, resides in each worker to sync weights between trainer and SGLang.
    In standalone/colocated mode, acts as a GPU placeholder to prevent ray from scheduling over it.
    """

    def __init__(
        self,
        config: DictConfig,
        device_mesh: DeviceMesh,
    ):
        self._quant_method = config.get("quantization")  # "fp8", "int4", "mxfp8", or None
        self._quant_config = None
        self._quant_source_dtype = None

        if self._quant_method in ("fp8", "int4", "mxfp8"):
            from transformers import AutoConfig

            hf_config = AutoConfig.from_pretrained(  # nosec B615
                config.model_path, trust_remote_code=config.get("trust_remote_code", False)
            )
            self._quant_source_dtype = getattr(hf_config, "torch_dtype", torch.bfloat16)

        if self._quant_method == "fp8":
            import sglang
            from packaging.version import Version

            assert Version(sglang.__version__) >= Version("0.5.5"), "sglang>=0.5.5 is required for FP8 quantization"
            self._quant_config = {
                "activation_scheme": "dynamic",
                "fmt": "e4m3",
                "quant_method": "fp8",
                "weight_block_size": [128, 128],
            }
        elif self._quant_method == "mxfp8":
            self._quant_config = {
                "activation_scheme": "dynamic",
                "fmt": "e4m3",
                "quant_method": "mxfp8",
                "weight_block_size": [1, 32],
                "scale_fmt": "ue8m0",
            }
        elif self._quant_method == "int4":
            # Read ignore rules from the model's quantization_config if available
            # (set during conversion by convert_hf_to_int4.py). This ensures the
            # runtime quantizes exactly the same layers as the offline conversion.
            int4_ignore = ["re:.*lm_head.*", "re:.*norm.*", "re:.*embed.*"]
            try:
                from transformers import AutoConfig as _AC

                _hf_cfg = _AC.from_pretrained(  # nosec B615
                    config.model_path, trust_remote_code=config.get("trust_remote_code", False)
                )
                _qcfg = getattr(_hf_cfg, "quantization_config", None)
                if isinstance(_qcfg, dict) and "ignore" in _qcfg:
                    int4_ignore = _qcfg["ignore"]
                    logger.info(f"INT4: loaded ignore rules from model config: {int4_ignore}")
            except Exception:
                pass  # fall back to defaults

            self._quant_config = {
                "quant_method": "compressed-tensors",
                "group_size": config.get("int4_group_size", 128),
                "symmetric": config.get("int4_symmetric", True),
                "ignore": int4_ignore,
            }

        super().__init__(config, device_mesh)
        self._engine: AsyncHttpServerAdapter | None = None

        rank = int(os.environ["RANK"])
        local_world_size = int(os.environ["RAY_LOCAL_WORLD_SIZE"])
        sampler_world_size = (
            self.config.tensor_model_parallel_size
            * self.config.data_parallel_size
            * self.config.pipeline_model_parallel_size
        )
        self.replica_rank, self.sampler_rank = divmod(rank, sampler_world_size)
        self.node_rank, self.local_rank = divmod(self.sampler_rank, local_world_size)

    async def _init_server_adapter(self):
        """Lazy init: the HTTP server is only available after the hybrid engine has launched."""
        if self._engine is not None:
            return

        server_actor = ray.get_actor(f"sglang_server_{self.replica_rank}_{self.node_rank}")
        server_address, server_port = await server_actor.get_server_address.remote()
        logger.debug(f"replica={self.replica_rank} node={self.node_rank} server={server_address}:{server_port}")
        host = f"[{server_address}]" if is_valid_ipv6_address(server_address) else server_address
        self._engine = AsyncHttpServerAdapter(
            model_path=self.config.model_path, host=host, port=server_port, launch_server=False
        )

    async def resume(self, tags: list[str]):
        """Resume sampler weights or kv cache in GPU memory."""
        if self.device_mesh["infer_tp"].get_local_rank() == 0 and self.config.offload_sampler:
            await self._init_server_adapter()
            await self._engine.resume_memory_occupation(tags=tags)

    async def release(self):
        """Release weights and kv cache in GPU memory."""
        if self.device_mesh["infer_tp"].get_local_rank() == 0 and self.config.offload_sampler:
            await self._init_server_adapter()
            await self._engine.release_memory_occupation(tags=["kv_cache", "weights"])

    async def update_weights(self, weights: Generator[tuple[str, torch.Tensor], None, None], **kwargs):
        """Update model weights using tensor buckets."""
        if self.device_mesh["infer_tp"].get_local_rank() == 0:
            await self._init_server_adapter()

        if self._quant_method == "fp8":
            from axon.utils.sglang.fp8 import fp8_quantize_weight_generator

            logger.info("Converting weights to FP8 before loading")
            weights = fp8_quantize_weight_generator(
                weights,
                self._quant_config,
                dtype=self._quant_source_dtype,
            )
        elif self._quant_method == "mxfp8":
            from axon.utils.sglang.mxfp8 import mxfp8_quantize_weight_generator

            logger.info("Converting weights to MxFP8 before loading")
            weights = mxfp8_quantize_weight_generator(
                weights,
                self._quant_config,
                dtype=self._quant_source_dtype,
            )
        elif self._quant_method == "int4":
            from axon.utils.sglang.int4 import int4_quantize_weight_generator

            logger.info("Converting weights to INT4 before loading")
            weights = int4_quantize_weight_generator(
                weights,
                self._quant_config,
                dtype=self._quant_source_dtype,
            )

        for params_batch in get_named_tensor_buckets(weights, _UPDATE_WEIGHTS_BUCKET_BYTES):
            await sgl_update_weights(
                engine=self._engine,
                params_batch=params_batch,
                device_mesh_key="infer_tp",
                device_mesh=self.device_mesh,
            )

        if self.device_mesh["infer_tp"].get_local_rank() == 0:
            await self._engine.flush_cache()
