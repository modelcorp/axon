# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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

import torch
from megatron.core.optimizer import OptimizerConfig
from megatron.core.optimizer import get_megatron_optimizer as get_megatron_optimizer_native
from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler

from axon.utils.logger import print_rank_0


def _get(cfg, key, default=None):
    if cfg is None:
        return default
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def init_megatron_optim_config(
    optimizer_name,
    optimizer_args_config,
    grad_clip,
    lr_scheduler_args,
    use_distributed_optimizer: bool = True,
    fp16: bool = False,
) -> OptimizerConfig:
    # Megatron-core only accepts 'adam' or 'sgd'; auto-convert AdamW -> adam
    if optimizer_name and optimizer_name.lower() in ("adamw",):
        print_rank_0("Warning: converting AdamW to Adam for Megatron-core")
        optimizer_name = "adam"

    lr = _get(optimizer_args_config, "lr", 1e-3)
    weight_decay = _get(optimizer_args_config, "weight_decay", 0.1)
    override_optimizer_args = _get(optimizer_args_config, "override_optimizer_args", {})

    # Resolve min_lr: use explicit min_lr if set, otherwise derive from min_lr_ratio
    min_lr = _get(lr_scheduler_args, "min_lr", 0.0) or 0.0
    min_lr_ratio = _get(lr_scheduler_args, "min_lr_ratio", 0.0) or 0.0
    if min_lr_ratio > 0 and lr > 0:
        min_lr = max(min_lr, min_lr_ratio * lr)

    optim_args = {
        "optimizer": optimizer_name,
        "lr": lr,
        "min_lr": min_lr,
        "clip_grad": grad_clip,
        "weight_decay": weight_decay,
        "use_distributed_optimizer": use_distributed_optimizer,
    }
    if fp16:
        optim_args.update(
            {
                "bf16": False,
                "fp16": True,
                "params_dtype": torch.float16,
                "initial_loss_scale": 32768,
                "min_loss_scale": 1,
                "use_precision_aware_optimizer": True,
                "store_param_remainders": False,
            }
        )
    else:  # bf16 mode
        optim_args.update(
            {
                "bf16": True,
                "params_dtype": torch.bfloat16,
            }
        )
    if override_optimizer_args:
        for k, v in override_optimizer_args.items():
            optim_args[k] = v

    print_rank_0(f"optimizer config after override: {optim_args}")

    config = OptimizerConfig(**optim_args)
    return config


def get_megatron_optimizer(
    model,
    config: OptimizerConfig,
):
    # Base optimizer.
    return get_megatron_optimizer_native(
        config=config,
        model_chunks=model,
    )


def get_megatron_optimizer_param_scheduler(
    optimizer,
    lr_scheduler_type,
    lr_scheduler_args,
    optimizer_args_config,
):
    """
    Get the optimizer parameter scheduler for Megatron.

    Args:
        optimizer: The Megatron optimizer.
        lr_scheduler_type: LR decay style (e.g. "constant", "cosine").
        lr_scheduler_args: Dict/config with scheduler fields (total_training_steps,
            lr_warmup_steps, lr_warmup_steps_ratio, min_lr, min_lr_ratio,
            lr_warmup_init, lr_decay_steps, weight_decay_incr_style,
            lr_wsd_decay_style, lr_wsd_decay_steps).
        optimizer_args_config: Dict/config with optimizer fields (lr, weight_decay).
    """
    lr = _get(optimizer_args_config, "lr", 1e-3)
    weight_decay = _get(optimizer_args_config, "weight_decay", 0.1)

    lr_decay_steps = _get(lr_scheduler_args, "lr_decay_steps", None)
    lr_warmup_steps = _get(lr_scheduler_args, "lr_warmup_steps", -1)
    total_training_steps = _get(lr_scheduler_args, "total_training_steps", 0)
    if lr_decay_steps is None:
        lr_decay_steps = total_training_steps
    wsd_decay_steps = None
    if _get(lr_scheduler_args, "lr_wsd_decay_steps", None) is not None:
        wsd_decay_steps = _get(lr_scheduler_args, "lr_wsd_decay_steps")
    if _get(lr_scheduler_args, "lr_warmup_steps_ratio", None) is not None and (
        lr_warmup_steps is None or lr_warmup_steps <= 0
    ):
        lr_warmup_steps = int(_get(lr_scheduler_args, "lr_warmup_steps_ratio", 0.0) * lr_decay_steps)

    # Resolve min_lr: use explicit min_lr if set, otherwise derive from min_lr_ratio
    min_lr = _get(lr_scheduler_args, "min_lr", 0.0)
    min_lr_ratio = _get(lr_scheduler_args, "min_lr_ratio", 0.0)
    if min_lr_ratio > 0 and lr > 0:
        min_lr = max(min_lr, min_lr_ratio * lr)

    opt_param_scheduler = OptimizerParamScheduler(
        optimizer,
        init_lr=_get(lr_scheduler_args, "lr_warmup_init", 0.0),
        max_lr=lr,
        min_lr=min_lr,
        lr_warmup_steps=lr_warmup_steps,
        lr_decay_steps=lr_decay_steps,
        lr_decay_style=lr_scheduler_type,
        start_wd=weight_decay,
        end_wd=weight_decay,
        wd_incr_steps=total_training_steps,
        wd_incr_style=_get(lr_scheduler_args, "weight_decay_incr_style", "constant"),
        override_opt_param_scheduler=False,
        wsd_decay_steps=wsd_decay_steps,
        lr_wsd_decay_style=_get(lr_scheduler_args, "lr_wsd_decay_style", "exponential"),
    )

    return opt_param_scheduler


def get_megatron_last_lr(optimizer):
    """
    Get the last learning rate from the optimizer parameter scheduler.
    """
    return optimizer.param_groups[0]["lr"]
