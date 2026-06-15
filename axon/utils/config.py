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

import logging
from dataclasses import is_dataclass
from typing import Any

from omegaconf import DictConfig, ListConfig, OmegaConf

logger = logging.getLogger(__name__)

__all__ = ["omega_conf_to_dataclass", "get_profiler_tool_config", "validate_config"]


def omega_conf_to_dataclass(config: DictConfig | dict, dataclass_type: type[Any] | None = None) -> Any:
    """
    Convert an OmegaConf DictConfig to a dataclass.

    Args:
        config: The OmegaConf DictConfig or dict to convert.
        dataclass_type: The dataclass type to convert to. When dataclass_type is None,
            the DictConfig must contain _target_ to be instantiated via hydra.instantiate API.

    Returns:
        The dataclass instance.
    """
    # Got an empty config
    if not config:
        return dataclass_type if dataclass_type is None else dataclass_type()
    # Got an object
    if not isinstance(config, DictConfig | ListConfig | dict | list):
        return config

    if dataclass_type is None:
        assert "_target_" in config, (
            "When dataclass_type is not provided, config must contain _target_. "
            "See axon/config/config.yaml algorithm section for an example. "
            f"Got config: {config}"
        )
        from hydra.utils import instantiate

        return instantiate(config, _convert_="partial")

    if not is_dataclass(dataclass_type):
        raise ValueError(f"{dataclass_type} must be a dataclass")

    # Convert config to plain dict if needed
    if isinstance(config, DictConfig):
        cfg_dict = OmegaConf.to_container(config, resolve=True)
    else:
        cfg_dict = dict(config)

    # Remove _target_ if present (not needed when dataclass_type is provided)
    cfg_dict.pop("_target_", None)

    # Use OmegaConf with struct mode disabled to allow flexible merging
    cfg = OmegaConf.create(cfg_dict)
    try:
        cfg_from_dataclass = OmegaConf.structured(dataclass_type)
    except Exception:
        # If structured config creation fails (due to type conflicts in nested configs),
        # fall back to direct dataclass instantiation
        return _create_dataclass_from_dict(dataclass_type, cfg_dict)

    # let cfg override the existing vals in `cfg_from_dataclass`
    cfg_merged = OmegaConf.merge(cfg_from_dataclass, cfg)
    # now convert to `dataclass_type`
    config_object = OmegaConf.to_object(cfg_merged)
    return config_object


def _create_dataclass_from_dict(dataclass_type: type, data: dict) -> Any:
    """Create a dataclass instance from a dictionary, handling nested dataclasses."""
    from dataclasses import fields as dataclass_fields

    field_values = {}
    for f in dataclass_fields(dataclass_type):
        if f.name in data:
            value = data[f.name]
            # If field type is a dataclass and value is a dict, recurse
            if is_dataclass(f.type) and isinstance(value, dict):
                value = _create_dataclass_from_dict(f.type, value)
            field_values[f.name] = value

    return dataclass_type(**field_values)


def get_profiler_tool_config(omega_profiler_config: DictConfig | dict) -> Any:
    """
    Get the profiler tool config dataclass from an OmegaConf profiler config.

    Args:
        omega_profiler_config: The profiler config (typically config.get("profiler", {}))

    Returns:
        The tool config dataclass instance, or None if no tool is specified.
    """
    tool_name = omega_profiler_config.get("tool", None) if omega_profiler_config else None
    if tool_name not in ["nsys", "torch", "torch_memory"]:
        return None

    from axon.utils.profiler.config import (
        NsightToolConfig,
        TorchMemoryToolConfig,
        TorchProfilerToolConfig,
    )

    tool_dataclass_map = {
        "nsys": NsightToolConfig,
        "torch": TorchProfilerToolConfig,
        "torch_memory": TorchMemoryToolConfig,
    }
    tool_config_data = omega_profiler_config.get("tool_config", {}).get(tool_name)
    if tool_config_data is None:
        return None
    return omega_conf_to_dataclass(tool_config_data, dataclass_type=tool_dataclass_map[tool_name])


def update_dict_with_config(dictionary: dict, config: DictConfig):
    for key in dictionary:
        if hasattr(config, key):
            dictionary[key] = getattr(config, key)


def validate_config(
    config: DictConfig,
    use_reference_policy: bool,
    use_critic: bool,
) -> None:
    """Validate an OmegaConf DictConfig.

    Args:
        config (DictConfig): The OmegaConf DictConfig to validate.
        use_reference_policy (bool): is ref policy needed
        use_critic (bool): is critic needed
    """
    # number of GPUs total
    n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes

    if not config.actor.use_dynamic_bsz:
        if config.actor.strategy == "megatron":
            model_parallel_size = (
                config.actor.megatron.tensor_model_parallel_size * config.actor.megatron.pipeline_model_parallel_size
            )
            assert n_gpus % (model_parallel_size * config.actor.megatron.context_parallel_size) == 0, (
                f"n_gpus ({n_gpus}) must be divisible by model_parallel_size ({model_parallel_size}) times "
                f"context_parallel_size ({config.actor.megatron.context_parallel_size})"
            )
            megatron_dp = n_gpus // (model_parallel_size * config.actor.megatron.context_parallel_size)
            minimal_bsz = megatron_dp * config.actor.micro_batch_size_per_gpu
        else:
            minimal_bsz = n_gpus

        # 1. Check total batch size for data correctness
        real_train_batch_size = config.data.train_batch_size * config.decoding.n
        assert real_train_batch_size % minimal_bsz == 0, (
            f"real_train_batch_size ({real_train_batch_size}) must be divisible by minimal possible batch size "
            f"({minimal_bsz})"
        )

    # A helper function to check "micro_batch_size" vs "micro_batch_size_per_gpu"
    # We throw an error if the user sets both. The new convention is "..._micro_batch_size_per_gpu".
    def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
        """Validate mutually exclusive micro batch size configuration options.

        Ensures that users don't set both deprecated micro_batch_size and
        the new micro_batch_size_per_gpu parameters simultaneously.

        Args:
            mbs: Deprecated micro batch size parameter value.
            mbs_per_gpu: New micro batch size per GPU parameter value.
            name (str): Configuration section name for error messages.

        Raises:
            ValueError: If both parameters are set or neither is set.
        """
        settings = {
            "reward_model": "micro_batch_size",
            "ref": "forward_micro_batch_size",
            "actor": "forward_micro_batch_size",
        }

        if name in settings:
            param = settings[name]
            param_per_gpu = f"{param}_per_gpu"

            if mbs is None and mbs_per_gpu is None:
                raise ValueError(f"[{name}] Please set at least one of '{name}.{param}' or '{name}.{param_per_gpu}'.")

            if mbs is not None and mbs_per_gpu is not None:
                raise ValueError(
                    f"[{name}] You have set both '{name}.{param}' AND '{name}.{param_per_gpu}'. Please remove "
                    f"'{name}.{param}' because only '*_{param_per_gpu}' is supported (the former is deprecated)."
                )

    if not config.actor.use_dynamic_bsz:
        if use_reference_policy:
            # reference: forward_micro_batch_size vs. forward_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.ref.forward_micro_batch_size,
                config.ref.forward_micro_batch_size_per_gpu,
                "ref",
            )

        #  The actor section also has forward_micro_batch_size vs. forward_micro_batch_size_per_gpu
        check_mutually_exclusive(
            config.actor.forward_micro_batch_size,
            config.actor.forward_micro_batch_size_per_gpu,
            "actor",
        )

    # Check for reward model micro-batch size conflicts
    if config.reward_model.enable and not config.reward_model.use_dynamic_bsz:
        check_mutually_exclusive(
            config.reward_model.micro_batch_size, config.reward_model.micro_batch_size_per_gpu, "reward_model"
        )

    kl_coef = OmegaConf.select(config, "loss_args.kl_coef", default=0)
    use_kl_in_reward = OmegaConf.select(config, "kl_reward", default=None) is not None or (
        OmegaConf.select(config, "algorithm.kl_reward", default=None) is not None
    )
    if use_kl_in_reward and kl_coef:
        logger.warning("You have both enabled in-reward kl and kl loss.")

    if config.data.get("val_batch_size", None) is not None:
        logger.warning(
            "val_batch_size is deprecated."
            " Validation datasets are sent to inference engines as a whole batch,"
            " which will schedule the memory themselves."
        )

    # check LoRA rank in vLLM
    lora_rank = config.actor.get("lora", {}).get("rank", 0)
    if lora_rank > 0 and config.sampler.name == "vllm":
        assert lora_rank <= 512, "LoRA rank in vLLM must be less than or equal to 512"

    logger.info("All configuration checks passed successfully.")
