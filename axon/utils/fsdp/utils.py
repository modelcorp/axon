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

import functools
import logging
from abc import ABC
from collections import OrderedDict
from contextlib import contextmanager, nullcontext

import torch
import torch.distributed as dist
import torch.nn as nn
from packaging import version
from torch.distributed import DeviceMesh
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp._runtime_utils import _lazy_init
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy, transformer_auto_wrap_policy
from transformers.trainer_pt_utils import get_module_class_from_name

from axon.utils.hf_model import check_exclude_modules, check_target_modules
from axon.utils.torch import get_device_id, get_device_name, get_torch_device

if version.parse(torch.__version__) >= version.parse("2.6"):
    from torch.distributed.fsdp import CPUOffloadPolicy, FSDPModule, MixedPrecisionPolicy, fully_shard
    from torch.distributed.tensor import Shard

    fully_shard_module = torch.distributed.fsdp._fully_shard._fully_shard
elif version.parse(torch.__version__) >= version.parse("2.4"):
    from torch.distributed._composable.fsdp import CPUOffloadPolicy, FSDPModule, MixedPrecisionPolicy, fully_shard

    fully_shard_module = torch.distributed._composable.fsdp.fully_shard
else:
    fully_shard, MixedPrecisionPolicy, FSDPModule, CPUOffloadPolicy, fully_shard_module = None, None, None, None, None

logger = logging.getLogger(__name__)


def init_fn(x: torch.nn.Module):
    if torch.distributed.get_rank() != 0:
        x = x.to_empty(device=get_device_id(), recurse=False)
        get_torch_device().empty_cache()
    return x


def get_init_weight_context_manager(use_meta_tensor=True, mesh: DeviceMesh = None):
    from accelerate import init_empty_weights

    cpu_init_weights = lambda: torch.device("cpu")
    if use_meta_tensor:
        if mesh is None:
            init_context = init_empty_weights if torch.distributed.get_rank() != 0 else cpu_init_weights
        else:
            init_context = init_empty_weights if mesh.get_coordinate()[-1] != 0 else cpu_init_weights
    else:
        init_context = cpu_init_weights
    return init_context


# Copyright 2020-present the HuggingFace Inc. team.
# Adapted from https://github.com/huggingface/transformers/src/transformers/trainer.py
def get_fsdp_wrap_policy(module, config=None, is_lora=False):
    """Get FSDP wrap policy for the module.

    Args:
        module: The module to get wrap policy for
        config: Configuration for wrap policy
        is_lora: Whether to enable lambda policy for LoRA modules
    """
    if config is None:
        config = {}

    # NOTE: This is a temporary workaround to be compatible with the OmegaConf & dataclass. We will remove this
    # once we have make all config from OmegaConf to data class.
    def _get_attr(attr_name, default_value=None):
        if hasattr(config, "get"):
            return config.get(attr_name, default_value)
        else:
            return config.__getattribute__(attr_name)

    if _get_attr("disable", False):
        return None

    default_transformer_cls_names_to_wrap = getattr(module, "_no_split_modules", None)
    fsdp_transformer_layer_cls_to_wrap = _get_attr(
        "transformer_layer_cls_to_wrap", default_transformer_cls_names_to_wrap
    )
    min_num_params = _get_attr("min_num_params", 0)
    auto_wrap_policy = None

    policies = []

    from torch.distributed.fsdp.wrap import _or_policy, lambda_auto_wrap_policy

    # Add lambda policy for LoRA modules if is_lora is True
    if is_lora:

        def lambda_policy_fn(module):
            return bool(
                len(list(module.named_children())) == 0
                and getattr(module, "weight", None) is not None
                and module.weight.requires_grad
            )

        lambda_policy = functools.partial(lambda_auto_wrap_policy, lambda_fn=lambda_policy_fn)
        policies.append(lambda_policy)

    if min_num_params > 0:
        size_policy = functools.partial(size_based_auto_wrap_policy, min_num_params=min_num_params)
        policies.append(size_policy)
    elif fsdp_transformer_layer_cls_to_wrap is not None:
        transformer_cls_to_wrap = set()
        for layer_class in fsdp_transformer_layer_cls_to_wrap:
            transformer_cls = get_module_class_from_name(module, layer_class)
            if transformer_cls is None:
                # Some _no_split_modules entries may not exist in every model variant
                # (e.g. Gemma4AudioLayer absent in text-only Gemma4 31B).
                logger.warning(f"Could not find layer class '{layer_class}' in the model, skipping.")
            else:
                transformer_cls_to_wrap.add(transformer_cls)
        if not transformer_cls_to_wrap:
            raise Exception(
                "Could not find any transformer layer class to wrap in the model. "
                f"Searched for: {fsdp_transformer_layer_cls_to_wrap}"
            )

        transformer_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls=transformer_cls_to_wrap,
        )
        policies.append(transformer_policy)

    if len(policies) > 0:
        auto_wrap_policy = functools.partial(_or_policy, policies=policies)

    return auto_wrap_policy


@torch.no_grad()
def offload_fsdp_model_to_cpu(model: FSDP, empty_cache: bool = True):
    if fsdp_version(model) == 2:
        offload_fsdp2_model_to_cpu(model, empty_cache)
        return

    assert isinstance(model, FSDP)
    # lazy init FSDP model
    _lazy_init(model, model)
    assert model._is_root, "Only support root model offloading to CPU"
    for handle in model._all_handles:
        if handle._offload_params:
            continue
        flat_param = handle.flat_param
        # After optim_step, flat_param.data may still be the full unsharded
        # tensor (size != _local_shard). Reshard before offloading.
        if flat_param.data.size() != flat_param._local_shard.size():
            handle.reshard(True)
        handle.flat_param_to(torch.device("cpu"), non_blocking=True)
        # the following still keeps id(._local_shard) != id(.data)
        flat_param._local_shard = flat_param.data
    if empty_cache:
        get_torch_device().empty_cache()


@torch.no_grad()
def offload_fsdp2_model_to_cpu(model, empty_cache: bool = True):
    # The root module (and any reshard_after_forward=False module) leaves its params
    # unsharded as plain tensors after forward. model.cpu() -> FSDP2 reset_sharded_param
    # then reads new_param._local_tensor and raises AttributeError on those plain params
    # (e.g. Qwen2.5-VL, whose vision tower sits in the root's unsharded group). Reshard
    # every FSDP module first so all params are sharded DTensors; no-op if already sharded.
    if FSDPModule is not None:
        for m in model.modules():
            if isinstance(m, FSDPModule) and hasattr(m, "reshard"):
                m.reshard()
    model.cpu()
    if empty_cache:
        get_torch_device().empty_cache()


@torch.no_grad()
def load_fsdp_model_to_gpu(model: FSDP):
    if fsdp_version(model) == 2:
        load_fsdp2_model_to_gpu(model)
        return

    assert isinstance(model, FSDP)
    # lazy init FSDP model
    _lazy_init(model, model)
    assert model._is_root, "Only support root model loading to GPU"
    device_id = get_device_id()
    for handle in model._all_handles:
        if handle._offload_params:
            continue
        flat_param = handle.flat_param
        handle.flat_param_to(torch.device(f"{get_device_name()}:{device_id}"), non_blocking=True)
        # the following still keeps id(._local_shard) != id(.data)
        flat_param._local_shard = flat_param.data


@torch.no_grad()
def load_fsdp2_model_to_gpu(model):
    device = get_device_id()
    model.to(device)


@torch.no_grad()
def offload_fsdp_optimizer(optimizer):
    if not optimizer.state:
        return
    for param_group in optimizer.param_groups:
        for param in param_group["params"]:
            state = optimizer.state[param]
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to("cpu", non_blocking=True)


@torch.no_grad()
def load_fsdp_optimizer(optimizer, device_id):
    if not optimizer.state:
        return
    for param_group in optimizer.param_groups:
        for param in param_group["params"]:
            state = optimizer.state[param]
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(device_id, non_blocking=True)


def fsdp_version(model):
    if isinstance(model, FSDP):
        return 1
    elif isinstance(model, FSDPModule):
        return 2
    else:
        return 0


def get_fsdp_state_ctx(model, state_type, state_cfg, optim_cfg):
    if fsdp_version(model) == 1:
        return FSDP.state_dict_type(model, state_type, state_cfg, optim_cfg)
    else:
        return nullcontext()


def get_fsdp_full_state_dict(model: torch.nn.Module, offload_to_cpu: bool = True, rank0_only: bool = True):
    """
    Get the full state dict from an FSDP model.

    Args:
        model (torch.nn.Module): The FSDP model to get state dict from
        offload_to_cpu (bool, optional): Whether to offload the state dict to CPU. Defaults to True.
        rank0_only (bool, optional): Whether to only get state dict on rank 0. Defaults to True.

    Returns:
        dict: The full state dict of the model

    Raises:
        NotImplementedError: If the FSDP version is unknown
    """
    if fsdp_version(model) == 1:
        from torch.distributed.fsdp import FullStateDictConfig, StateDictType

        state_dict_config = FullStateDictConfig(offload_to_cpu=offload_to_cpu, rank0_only=rank0_only)
        with get_fsdp_state_ctx(
            model, state_type=StateDictType.FULL_STATE_DICT, state_cfg=state_dict_config, optim_cfg=None
        ):
            state_dict = model.state_dict()
        return state_dict
    elif fsdp_version(model) == 2:
        from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict

        # FSDP2 state dict with rank0_only=True and cpu_offload=False
        # Requires monkey patch (see axon.monkey_patches.fsdp_monkey_patch) that forces ranks_only=(0,)
        # Without the patch, PyTorch would require cpu_offload=True to get rank0_only behavior
        state_dict_config = StateDictOptions(
            full_state_dict=True,
            cpu_offload=offload_to_cpu,
            broadcast_from_rank0=not rank0_only,
            strict=False if (rank0_only and not offload_to_cpu) else True,
        )
        state_dict = get_model_state_dict(model, options=state_dict_config)
        return state_dict
    else:
        raise NotImplementedError(f"Unknown FSDP version {fsdp_version}")


def fsdp2_load_full_state_dict(
    model: torch.nn.Module,
    full_state: dict,
    device_mesh=None,
    cpu_offload=None,
    all_ranks_have_state: bool = False,
):
    """
    Loads the full state dict into the FSDP2-sharded model.

    When only rank 0 has the state (meta-tensor init), broadcasts from rank 0.
    When all ranks have the state (cpu init for tied-embedding models), each
    rank loads its own copy — no broadcast needed.

    Args:
        model: The FSDP2-wrapped model to load the state dict into.
        full_state: The full state dict. On rank 0 always; on other ranks only
            when ``all_ranks_have_state=True``.
        device_mesh: Optional device mesh for FSDP.
        cpu_offload: Optional CPUOffloadPolicy; if not None, model is moved
            back to CPU after loading.
        all_ranks_have_state: If True, all ranks already have the full state
            dict (e.g. because tie_word_embeddings forced CPU init on all ranks).
            Skips the broadcast path entirely.
    """
    from torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict

    cpu_offload = cpu_offload is not None

    if all_ranks_have_state:
        # All ranks already have the full model on CPU — just move to GPU and load.
        # No broadcast needed; avoids NCCL collective mismatches on buffers.
        model = model.to(device=get_device_id(), non_blocking=True)
        options = StateDictOptions(full_state_dict=True, cpu_offload=cpu_offload, broadcast_from_rank0=False)
        set_model_state_dict(model, full_state, options=options)
    else:
        # Only rank 0 has the state — broadcast to other ranks.
        if dist.get_rank() == 0:
            model = model.to(device=get_device_id(), non_blocking=True)
        else:
            model = model.to_empty(device=get_device_id())

        options = StateDictOptions(full_state_dict=True, cpu_offload=cpu_offload, broadcast_from_rank0=True)
        set_model_state_dict(model, full_state, options=options)

        # Buffers not in state_dict (e.g. rotary_emb inv_freq) need manual broadcast.
        # Use rank 0's buffer list as the canonical source to ensure all ranks
        # perform the same number of broadcast ops in the same order.
        rank = dist.get_rank()
        buffer_dict = dict(model.named_buffers())
        if rank == 0:
            buffer_names = list(buffer_dict.keys())
        else:
            buffer_names = None
        buffer_names_list = [buffer_names]
        dist.broadcast_object_list(buffer_names_list, src=0)
        buffer_names = buffer_names_list[0]

        for name in buffer_names:
            if name in buffer_dict:
                dist.broadcast(buffer_dict[name], src=0)

    if cpu_offload:
        model.to("cpu", non_blocking=True)
        for buf in model.buffers():
            buf.data = buf.data.to(get_device_id())


@contextmanager
def maybe_patch_fsdp_module(model):
    if fully_shard_module is None:
        yield
        return

    orig_fsdp_module = fully_shard_module.FSDPModule

    class FSDPModuleABC(ABC, orig_fsdp_module):
        pass

    try:
        if isinstance(model, ABC):
            fully_shard_module.FSDPModule = FSDPModuleABC
        yield
    finally:
        fully_shard_module.FSDPModule = orig_fsdp_module


def apply_fsdp2(model, fsdp_kwargs, config):
    """model: AutoModelForCausalLM"""
    assert CPUOffloadPolicy is not None, "PyTorch version >= 2.4 is required for using fully_shard API (FSDP2)"

    default_transformer_cls_names_to_wrap = getattr(model, "_no_split_modules", None)
    fsdp_transformer_layer_cls_to_wrap = config.get("wrap_policy", {}).get(
        "transformer_layer_cls_to_wrap", default_transformer_cls_names_to_wrap
    )

    if isinstance(fsdp_transformer_layer_cls_to_wrap, str):
        fsdp_transformer_layer_cls_to_wrap = [fsdp_transformer_layer_cls_to_wrap]
    elif isinstance(fsdp_transformer_layer_cls_to_wrap, set):
        fsdp_transformer_layer_cls_to_wrap = list(fsdp_transformer_layer_cls_to_wrap)

    assert len(fsdp_transformer_layer_cls_to_wrap) > 0 and fsdp_transformer_layer_cls_to_wrap[0] is not None

    modules = []
    for name, module in model.named_modules():
        if module.__class__.__name__ in fsdp_transformer_layer_cls_to_wrap or (
            isinstance(module, nn.Embedding) and not model.config.tie_word_embeddings
        ):
            modules.append(module)

    for idx, module in enumerate(modules):
        # if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
        #     print(f"wrap module {module.__class__.__name__}")
        with maybe_patch_fsdp_module(module):
            fully_shard(module, **fsdp_kwargs)

    # if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
    #     print(f"wrap module {model.__class__.__name__}")
    with maybe_patch_fsdp_module(model):
        fully_shard(model, **fsdp_kwargs)  # fsdp2 will not reshard_after_forward for root module


def get_shard_placement_fn(fsdp_size):
    """Choose the dimension that can divide fsdp_size to avoid padding"""

    def shard_placement_fn(param):
        shape = list(param.shape)
        for i in range(len(shape)):
            if shape[i] % fsdp_size == 0:
                return Shard(i)
        return Shard(0)

    return shard_placement_fn


def fsdp2_clip_grad_norm_(parameters, max_norm, norm_type=2.0, error_if_nonfinite=False, foreach=None):
    """torch.nn.utils.clip_grad_norm_ cann't run on cpu parameter DTensor"""
    from torch.nn.utils.clip_grad import _clip_grads_with_norm_, _get_total_norm

    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    else:
        # prevent generators from being exhausted
        parameters = list(parameters)
    grads = [p.grad for p in parameters if p.grad is not None]
    total_norm = _get_total_norm(grads, norm_type, error_if_nonfinite, foreach)
    total_norm = total_norm.to(get_device_id(), non_blocking=True)
    _clip_grads_with_norm_(parameters, max_norm, total_norm, foreach)
    return total_norm


def layered_summon_lora_params(fsdp_module) -> OrderedDict:
    from peft.utils.save_and_load import get_peft_model_state_dict

    def __prefix_submodules(module, prefix):
        for name, submodule in module.named_modules():
            if name.startswith(prefix) and "." not in name[len(prefix) :]:
                yield name, submodule

    lora_params = OrderedDict()
    prefix_list = [
        # fsdp
        "_fsdp_wrapped_module.base_model.model.",
        "_fsdp_wrapped_module.base_model.model.model.",
        "_fsdp_wrapped_module.base_model.model.model.layers.",
        "_fsdp_wrapped_module.base_model.model.model.language_model.layers.",
        # fsdp2
        "base_model.model.",
        "base_model.model.model.",
        "base_model.model.model.layers.",
        "base_model.model.model.language_model.layers.",
    ]
    peft_model = getattr(fsdp_module, "_fsdp_wrapped_module", fsdp_module)
    for prefix in prefix_list:
        for name, submodule in __prefix_submodules(fsdp_module, prefix):
            prefix = name.replace("_fsdp_wrapped_module.base_model.model.", "base_model.model.")
            if name.endswith(".model") or name.endswith(".layers"):
                continue
            if fsdp_version(submodule) > 0:
                with FSDP.summon_full_params(submodule, writeback=False):
                    sub_lora_params = get_peft_model_state_dict(peft_model, state_dict=submodule.state_dict())
                    sub_lora_params = {
                        f"{prefix}.{name}": param.full_tensor().detach().cpu()
                        if hasattr(param, "full_tensor")
                        else param.detach().cpu()
                        for name, param in sub_lora_params.items()
                    }
                    lora_params.update(sub_lora_params)
                    submodule._is_root = False
                get_torch_device().empty_cache()
    return lora_params


def collect_lora_params(module: FSDP, layered_summon: bool, base_sync_done: bool) -> OrderedDict:
    """
    collect lora params or full params if base model is not ready in vllm
    work with if isinstance(self.module._fsdp_wrapped_module, PeftModel)
    """
    from peft.utils.save_and_load import get_peft_model_state_dict

    lora_params = OrderedDict()
    peft_model = getattr(module, "_fsdp_wrapped_module", module)
    if fsdp_version(module) > 0:
        if layered_summon:
            if not base_sync_done:
                raise ValueError(
                    "To use layered_summon, you must make sure base-model is preloaded in vllm, e.g. let "
                    "sampler.load_format=safetensors"
                )
            lora_params = layered_summon_lora_params(module)
        else:
            with FSDP.summon_full_params(module, writeback=False):
                if base_sync_done:
                    lora_params = get_peft_model_state_dict(peft_model)
                    lora_params = {
                        name: param.full_tensor().detach().cpu()
                        if hasattr(param, "full_tensor")
                        else param.detach().cpu()
                        for name, param in lora_params.items()
                    }
                else:
                    model = peft_model.base_model.model
                    orig_dev = "cpu" if "cpu" in str(next(model.parameters()).device) else get_device_name()
                    model = model.to("cpu")
                    for name, param in model.state_dict().items():
                        if any(x in name for x in ["_flat_param", "lora_"]):
                            continue
                        name = name.replace("_fsdp_wrapped_module.", "").replace(".base_layer", "")
                        lora_params[name] = (
                            param.full_tensor().detach().cpu()
                            if hasattr(param, "full_tensor")
                            else param.detach().cpu()
                        )
                    model = model.to(orig_dev)
            get_torch_device().empty_cache()
    else:
        if base_sync_done:
            lora_params = get_peft_model_state_dict(peft_model)
        else:
            model = peft_model.base_model.model
            orig_dev = "cpu" if "cpu" in str(next(model.parameters()).device) else get_device_name()
            model = model.to("cpu")
            for name, param in model.state_dict().items():
                if any(x in name for x in ["_flat_param", "lora_"]):
                    continue
                name = name.replace("_fsdp_wrapped_module.", "").replace(".base_layer", "")
                lora_params[name] = param.detach().cpu()
            model = model.to(orig_dev)
    return lora_params


def create_device_mesh(world_size, fsdp_size):
    """Create a 1D or 2D device mesh for FSDP sharding."""
    device_name = get_device_name()
    if fsdp_size < 0 or fsdp_size >= world_size:
        device_mesh = init_device_mesh(device_name, mesh_shape=(world_size,), mesh_dim_names=["fsdp"])
    else:
        device_mesh = init_device_mesh(
            device_name, mesh_shape=(world_size // fsdp_size, fsdp_size), mesh_dim_names=["ddp", "fsdp"]
        )
    return device_mesh


def get_sharding_strategy(device_mesh):
    """Return the appropriate FSDP ShardingStrategy for the given device mesh."""
    from torch.distributed.fsdp import ShardingStrategy

    if device_mesh.ndim == 1:
        sharding_strategy = ShardingStrategy.FULL_SHARD
    elif device_mesh.ndim == 2:
        sharding_strategy = ShardingStrategy.HYBRID_SHARD
    else:
        raise NotImplementedError(f"Get device mesh ndim={device_mesh.ndim}, but only support 1 or 2")
    return sharding_strategy


def parse_mixed_precision(fsdp_config, default_param_dtype=None):
    """Parse mixed precision settings from fsdp_config.

    Returns (param_dtype, reduce_dtype, buffer_dtype).
    When no mixed_precision sub-config exists, falls back to *default_param_dtype*
    (or ``fsdp_config.dtype`` when that is ``None``).
    """
    from axon.utils.torch.dtypes import PrecisionType

    mixed_precision_config = fsdp_config.get("mixed_precision", None)
    if mixed_precision_config is not None:
        param_dtype = PrecisionType.to_dtype(mixed_precision_config.get("param_dtype", "bf16"))
        reduce_dtype = PrecisionType.to_dtype(mixed_precision_config.get("reduce_dtype", "fp32"))
        buffer_dtype = PrecisionType.to_dtype(mixed_precision_config.get("buffer_dtype", "fp32"))
    else:
        if default_param_dtype is not None:
            param_dtype = default_param_dtype
        else:
            param_dtype = PrecisionType.to_dtype(fsdp_config.dtype)
        reduce_dtype = torch.float32
        buffer_dtype = torch.float32
    return param_dtype, reduce_dtype, buffer_dtype


def replace_lora_wrapper(k, peft_config):
    """Replace LoRA parameter keys with base layer equivalents.

    Transforms LoRA parameter names to their corresponding base layer
    names for proper weight loading in vLLM when base model sync is not done.

    Args:
        k (str): Original parameter key name.

    Returns:
        str: Transformed parameter key for base layer.
    """
    stacked_params = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    if k.endswith(".weight"):
        module_k = k[: -len(".weight")]
        if check_exclude_modules(peft_config, module_k):
            return k
        elif any([module_k.endswith(s) for s in stacked_params]) or check_target_modules(peft_config, module_k):
            return f"{module_k}.base_layer.weight"
    if k.endswith(".bias"):
        module_k = k[: -len(".bias")]
        if check_exclude_modules(peft_config, module_k):
            return k
        elif any([module_k.endswith(s) for s in stacked_params]) or check_target_modules(peft_config, module_k):
            return f"{module_k}.base_layer.bias"
    return k
