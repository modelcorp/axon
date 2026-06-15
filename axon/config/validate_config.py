# Copyright 2025 Model AI Corp.
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
"""
Axon Unified Configuration Validation Module.

Configuration loading and validation functions for general training.
Validation structure mirrors config.yaml sections:
  Data | Algorithm | Loss | Actor | Ref | Critic | Sampler | Reward Model | Cross-section
"""

import warnings
from pathlib import Path
from typing import Any

import yaml

from axon.trainer.algos.constants import VALID_BATCH_REDUCE, VALID_TOKEN_REDUCE

# =============================================================================
# Configuration Loading
# =============================================================================


def load_ppo_config(config_path: str | Path, overrides: dict | None = None) -> dict:
    """Load PPO configuration from a YAML file."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    if overrides:
        config = _deep_merge(config, overrides)
    return config


def load_agent_ppo_config(
    strategy: str = "fsdp",
    overrides: dict | None = None,
    config_dir: str | Path | None = None,
) -> dict:
    """Load agent PPO configuration for the specified strategy.

    Args:
        strategy: Training strategy - ``"fsdp"`` or ``"megatron"``.
        overrides: Optional dictionary of additional configuration overrides.
        config_dir: Optional path to configuration directory.
            Defaults to the directory containing this module.
    """
    if strategy not in {"fsdp", "megatron"}:
        raise ValueError(f"Invalid strategy: {strategy}. Must be 'fsdp' or 'megatron'.")

    config_dir = Path(config_dir) if config_dir is not None else Path(__file__).parent
    config = load_ppo_config(config_dir / "config.yaml")

    if strategy == "megatron":
        config["strategy"] = "megatron"
    if overrides:
        config = _deep_merge(config, overrides)
    return config


def load_config_with_inheritance(
    config_path: str | Path,
    base_config_path: str | Path | None = None,
    overrides: dict | None = None,
) -> dict:
    """Load a configuration file with optional inheritance from a base config.

    The child config can specify a ``_base_`` key to indicate inheritance.
    If *base_config_path* is not provided, the ``_base_`` key is used instead.
    """
    config_path = Path(config_path)
    config = load_ppo_config(config_path)

    base_ref = config.pop("_base_", None)
    if base_config_path is not None:
        base_config_path = Path(base_config_path)
    elif base_ref is not None:
        base_config_path = config_path.parent / base_ref

    if base_config_path is not None and base_config_path.exists():
        config = _deep_merge(load_ppo_config(base_config_path), config)
    if overrides:
        config = _deep_merge(config, overrides)
    return config


def _deep_merge(base: dict, override: dict) -> dict:
    """Deep-merge *override* into *base*, returning a new dict."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _get_nested(config: dict, path: str, default: Any = None) -> Any:
    """Get a nested value using dot notation (e.g. ``"actor.fsdp.use_remove_padding"``)."""
    value = config
    for key in path.split("."):
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    return value


# =============================================================================
# Validation Helpers
# =============================================================================

_VALID_ADV_ESTIMATORS = {
    "gae",
    "grpo",
    "rloo",
    "loop",
    "reinforce_plus_plus",
    "reinforce_plus_plus_baseline",
    "remax",
    "opo",
    "grpo_passk",
    "gpg",
    "chunked_gae",
    "identity",
    "kimi_k1_5",
}
_VALID_KL_PENALTIES = {"kl", "abs", "mse", "low_var_kl", "full"}
_VALID_KL_TYPES = {"fixed", "adaptive"}

_VALID_SAMPLER_IS = {None, "token", "sequence"}
_VALID_SAMPLER_RS = {None, "token", "sequence", "geometric"}


def _check_mutually_exclusive(
    mbs: int | None,
    mbs_per_gpu: int | None,
    section: str,
    param: str = "micro_batch_size",
) -> None:
    """Ensure exactly one of *mbs* or *mbs_per_gpu* is set."""
    ppg = f"{param}_per_gpu"
    if mbs is None and mbs_per_gpu is None:
        raise ValueError(f"[{section}] Please set at least one of '{section}.{param}' or '{section}.{ppg}'.")
    if mbs is not None and mbs_per_gpu is not None:
        raise ValueError(
            f"[{section}] You have set both '{section}.{param}' AND '{section}.{ppg}'. Please remove "
            f"'{section}.{param}' because only '*_{ppg}' is supported (the former is deprecated)."
        )


# =============================================================================
# Section Validators (ordered to mirror config.yaml)
# =============================================================================

# --- Data ---


def validate_data_config(config: dict) -> None:
    """Validate train_files presence and sequence length constraints."""
    if config.get("train_files") is None:
        warnings.warn("data.train_files is not specified. Make sure to set it before training.", stacklevel=2)

    max_prompt = config.get("max_prompt_length", 512)
    max_seq = config.get("max_seq_length", 8192)
    if max_prompt <= 0:
        raise ValueError("max_prompt_length must be positive.")
    if max_seq <= 0:
        raise ValueError("max_seq_length must be positive.")
    if max_prompt > max_seq:
        raise ValueError(
            f"max_prompt_length ({max_prompt}) must not exceed max_seq_length ({max_seq}). "
            "max_prompt_length is the initial prompt truncation threshold; "
            "max_seq_length is the total token budget."
        )


# --- Algorithm (Advantage & KL Reward) ---


def validate_algorithm_config(config: dict) -> None:
    """Validate advantage estimator, kl_reward, and kl_reward_args."""
    adv = config.get("advantage", "grpo")
    if adv not in _VALID_ADV_ESTIMATORS:
        raise ValueError(f"Invalid adv_estimator: {adv}. Must be one of {sorted(_VALID_ADV_ESTIMATORS)}")

    kl_reward = config.get("kl_reward")
    if kl_reward is not None and kl_reward not in _VALID_KL_PENALTIES:
        raise ValueError(f"Invalid kl_reward: {kl_reward}. Must be null or one of {sorted(_VALID_KL_PENALTIES)}")

    kl_reward_args = config.get("kl_reward_args", {})
    if kl_reward_args:
        kl_type = kl_reward_args.get("type", "fixed")
        if kl_type not in _VALID_KL_TYPES:
            raise ValueError(f"Invalid kl_reward_args.type: {kl_type}. Must be one of {sorted(_VALID_KL_TYPES)}")


# --- Loss ---


def validate_loss_config(config: dict) -> None:
    """Validate loss aggregation and sampler correction fields in loss_args."""
    loss_args = config.get("loss_args", {}) or {}

    token_reduce = loss_args.get("token_reduce", None)
    batch_reduce = loss_args.get("batch_reduce", None)

    if token_reduce is not None and token_reduce not in VALID_TOKEN_REDUCE:
        raise ValueError(f"Invalid token_reduce: {token_reduce}")
    if batch_reduce is not None and batch_reduce not in VALID_BATCH_REDUCE:
        raise ValueError(f"Invalid batch_reduce: {batch_reduce}")

    sampler_is = loss_args.get("sampler_is")
    if sampler_is not in _VALID_SAMPLER_IS:
        raise ValueError(f"Invalid loss_args.sampler_is: {sampler_is}. Must be one of {_VALID_SAMPLER_IS}")

    sampler_rs = loss_args.get("sampler_rs")
    if sampler_rs not in _VALID_SAMPLER_RS:
        raise ValueError(f"Invalid loss_args.sampler_rs: {sampler_rs}. Must be one of {_VALID_SAMPLER_RS}")


validate_loss_args_config = validate_loss_config  # backwards-compatible alias


# --- Actor ---


def validate_actor_config(config: dict, n_gpus: int, strategy: str | None = None) -> None:
    """Validate actor batch sizes, GPU divisibility, and FSDP sequence parallelism."""
    actor = _get_nested(config, "actor", {})
    strategy = strategy or config.get("strategy", "fsdp")
    use_dynamic_bsz = actor.get("use_dynamic_bsz", False)

    if not use_dynamic_bsz:
        micro_batch_size = actor.get("micro_batch_size")
        micro_batch_size_per_gpu = actor.get("micro_batch_size_per_gpu")
        _check_mutually_exclusive(micro_batch_size, micro_batch_size_per_gpu, "actor")
        _check_mutually_exclusive(
            actor.get("forward_micro_batch_size"),
            actor.get("forward_micro_batch_size_per_gpu"),
            "actor",
            "forward_micro_batch_size",
        )

        mini_batch_size = config.get("mini_batch_size", 256)
        train_batch_size = config.get("train_batch_size", 64)
        real_train_batch_size = train_batch_size * _get_nested(config, "decoding.n", 1)

        # Compute minimal batch size based on strategy
        if strategy == "megatron":
            meg = actor.get("megatron", {})
            tp = meg.get("tensor_model_parallel_size", 1)
            pp = meg.get("pipeline_model_parallel_size", 1)
            cp = meg.get("context_parallel_size", 1)
            model_parallel = tp * pp

            if n_gpus % (model_parallel * cp) != 0:
                raise ValueError(
                    f"n_gpus ({n_gpus}) must be divisible by model_parallel_size ({model_parallel}) "
                    f"times context_parallel_size ({cp})"
                )
            minimal_bsz = (n_gpus // (model_parallel * cp)) * (micro_batch_size_per_gpu or 1)
        else:
            minimal_bsz = n_gpus

        if real_train_batch_size % minimal_bsz != 0:
            raise ValueError(
                f"real_train_batch_size ({real_train_batch_size}) must be divisible by "
                f"minimal possible batch size ({minimal_bsz})"
            )
        if train_batch_size < mini_batch_size:
            raise ValueError(f"train_batch_size ({train_batch_size}) must be >= mini_batch_size ({mini_batch_size})")

        sp_size = _get_nested(actor, "fsdp.ulysses_sequence_parallel_size", 1)
        if micro_batch_size is not None:
            if mini_batch_size % micro_batch_size != 0:
                raise ValueError(
                    f"mini_batch_size ({mini_batch_size}) must be divisible by micro_batch_size ({micro_batch_size})"
                )
            if micro_batch_size * sp_size < n_gpus:
                raise ValueError(
                    f"micro_batch_size ({micro_batch_size}) * "
                    f"ulysses_sequence_parallel_size ({sp_size}) must be >= n_gpus ({n_gpus})"
                )

    # FSDP sequence parallelism requires remove_padding
    if strategy in {"fsdp", "fsdp2"}:
        sp_size = _get_nested(actor, "fsdp.ulysses_sequence_parallel_size", 1)
        if sp_size > 1 and not _get_nested(actor, "fsdp.use_remove_padding", False):
            raise ValueError(
                "When using sequence parallelism for actor/ref policy, you must enable `use_remove_padding`."
            )


# --- Reference Policy ---


def validate_ref_config(config: dict, n_gpus: int) -> None:
    """Validate reference model forward_micro_batch_size mutual exclusivity."""
    if not _get_nested(config, "actor.use_dynamic_bsz", False):
        ref = _get_nested(config, "ref", {})
        _check_mutually_exclusive(
            ref.get("forward_micro_batch_size"),
            ref.get("forward_micro_batch_size_per_gpu"),
            "ref",
            "forward_micro_batch_size",
        )


# --- Critic ---


def validate_critic_config(config: dict, n_gpus: int) -> None:
    """Validate critic batch sizes, divisibility, and FSDP sequence parallelism."""
    critic = config.get("critic", {})
    strategy = config.get("strategy", "fsdp")
    use_dynamic_bsz = critic.get("use_dynamic_bsz", False)

    micro_batch_size = critic.get("micro_batch_size")
    mini_batch_size = config.get("mini_batch_size", 256)
    train_batch_size = config.get("train_batch_size", 64)

    if not use_dynamic_bsz:
        _check_mutually_exclusive(micro_batch_size, critic.get("micro_batch_size_per_gpu"), "critic")

        if micro_batch_size is not None and mini_batch_size % micro_batch_size != 0:
            raise ValueError(
                f"[critic] mini_batch_size ({mini_batch_size}) must be divisible by "
                f"micro_batch_size ({micro_batch_size})"
            )
        if train_batch_size < mini_batch_size:
            raise ValueError(f"train_batch_size ({train_batch_size}) must be >= mini_batch_size ({mini_batch_size})")

    if strategy in {"fsdp", "fsdp2"}:
        sp_size = _get_nested(critic, "fsdp.ulysses_sequence_parallel_size", 1)

        if not use_dynamic_bsz and micro_batch_size is not None and micro_batch_size * sp_size < n_gpus:
            raise ValueError(
                f"critic.micro_batch_size ({micro_batch_size}) * "
                f"ulysses_sequence_parallel_size ({sp_size}) must be >= n_gpus ({n_gpus})"
            )
        if sp_size > 1 and not _get_nested(critic, "fsdp.use_remove_padding", False):
            raise ValueError("When using sequence parallelism for critic, you must enable `use_remove_padding`.")


# --- Sampler ---


def validate_sampler_config(config: dict) -> None:
    """Validate LoRA rank for vLLM and moe_replay/prefix_caching compatibility."""
    sampler = _get_nested(config, "sampler", {})
    actor = _get_nested(config, "actor", {})

    lora_rank = _get_nested(actor, "lora.rank", 0)
    if sampler.get("name", "vllm") == "vllm" and lora_rank > 512:
        raise ValueError("LoRA rank in vLLM must be less than or equal to 512")

    if config.get("moe_replay", False) and sampler.get("enable_prefix_caching", False):
        raise ValueError(
            "moe_replay is incompatible with enable_prefix_caching. "
            "Prefix-cached tokens skip the vLLM forward pass, so their MoE routing data "
            "cannot be captured. Set sampler.enable_prefix_caching=false "
            "when using moe_replay=true."
        )

    # Validate speculative decoding config
    spec_cfg = sampler.get("speculative_config", {})
    if spec_cfg:
        backend = sampler.get("name", "vllm")
        method = spec_cfg.get("method") or spec_cfg.get("speculative_algorithm", "")
        if backend == "sglang" and method.upper() == "EAGLE":
            # EAGLE on SGLang uses MTP layers as the draft model — remind user to
            # ensure the model has MTP layers and AXON_ENABLE_MTP is set.
            import os

            if os.environ.get("AXON_ENABLE_MTP", "0") != "1":
                print(
                    "NOTICE: EAGLE speculative decoding on SGLang typically requires MTP layers. "
                    "Set AXON_ENABLE_MTP=1 if using vLLM backend with MTP drafter."
                )


# --- Reward Model ---


def validate_reward_model_config(config: dict) -> None:
    """Validate reward model micro_batch_size mutual exclusivity."""
    rm = config.get("reward_model", {})
    if not rm.get("use_dynamic_bsz", False):
        _check_mutually_exclusive(rm.get("micro_batch_size"), rm.get("micro_batch_size_per_gpu"), "reward_model")


# --- Cross-section Consistency ---


def _validate_cross_section_consistency(config: dict) -> None:
    """Warn if both in-reward KL and KL loss are enabled simultaneously."""
    kl_reward = config.get("kl_reward", _get_nested(config, "algorithm.kl_reward", None))
    kl_coef = _get_nested(config, "loss_args.kl_coef", 0)
    if kl_reward is not None and kl_coef:
        print("NOTICE: You have both enabled in-reward kl and kl loss.")


# =============================================================================
# Main Validator
# =============================================================================


def validate_axon_config(config: dict) -> None:
    """Validate the entire Axon configuration.

    Orchestrates all section validators in order matching config.yaml.
    """
    use_ref = needs_reference_policy(config)
    use_critic = needs_critic(config)
    strategy = config.get("strategy", "fsdp")
    n_gpus = config.get("num_gpus_per_node", 8) * config.get("num_nodes", 1)

    validate_data_config(config)
    validate_sampler_config(config)
    if config.get("mode") == "eval":
        return

    validate_algorithm_config(config)
    validate_loss_config(config)
    validate_actor_config(config, n_gpus, strategy)

    if use_ref:
        validate_ref_config(config, n_gpus)
    if use_critic:
        validate_critic_config(config, n_gpus)

    if _get_nested(config, "reward_model.enable", False):
        validate_reward_model_config(config)

    _validate_cross_section_consistency(config)
    print("[validate_ppo_config] All configuration checks passed successfully!")


# =============================================================================
# Utility Functions
# =============================================================================


def get_effective_strategy(config: dict) -> str:
    """Return the training strategy (``"fsdp"`` or ``"megatron"``)."""
    return config.get("strategy", "fsdp")


def is_megatron_strategy(config: dict) -> bool:
    return get_effective_strategy(config) == "megatron"


def is_fsdp_strategy(config: dict) -> bool:
    return get_effective_strategy(config) in {"fsdp", "fsdp2"}


def needs_reference_policy(config: dict) -> bool:
    """Reference policy is needed when using KL loss or in-reward KL penalty."""
    kl_coef = _get_nested(config, "loss_args.kl_coef", 0)
    kl_reward = config.get("kl_reward", _get_nested(config, "algorithm.kl_reward", None))
    return bool(kl_coef) or kl_reward is not None


def needs_critic(config: dict) -> bool:
    """Critic is needed for GAE advantage estimation (or when explicitly enabled)."""
    critic_enable = _get_nested(config, "critic.enable")
    if critic_enable is not None:
        return critic_enable
    adv = config.get("advantage", _get_nested(config, "algorithm.adv_estimator", "gae"))
    return adv == "gae"


def print_config_summary(config: dict) -> None:
    """Print a summary of the configuration."""
    strategy = get_effective_strategy(config)
    n_gpus = _get_nested(config, "num_gpus_per_node", 8) * _get_nested(config, "num_nodes", 1)
    print("=" * 60)
    print("PPO Configuration Summary")
    print("=" * 60)
    print(f"Strategy:           {strategy}")
    print(f"Total GPUs:         {n_gpus}")
    print(f"Train batch size:   {_get_nested(config, 'data.train_batch_size', 1024)}")
    print(f"Sampler n:          {_get_nested(config, 'decoding.n', 1)}")
    print(f"Advantage estimator: {_get_nested(config, 'algorithm.adv_estimator', 'gae')}")
    print(f"Needs reference:    {needs_reference_policy(config)}")
    print(f"Needs critic:       {needs_critic(config)}")
    print("=" * 60)
