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
"""
Learning rate scheduler utilities.
"""

import logging
import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

logger = logging.getLogger(__name__)


def get_cosine_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.0,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
    init_lr_ratio: float = None,
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.
    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
            The total number of training steps.
        min_lr_ratio (:obj:`float`, `optional`, defaults to 0.0):
            The minimum lr ratio w.r.t the maximum.
        num_cycles (:obj:`float`, `optional`, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
            following a half-cosine).
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.
        init_lr_ratio (:obj:`float`, `optional`, defaults to None):
            The initial lr ratio w.r.t the maximum.
    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """
    min_lr_ratio = 0.0 if min_lr_ratio is None else min_lr_ratio
    assert min_lr_ratio >= 0 and min_lr_ratio <= 1.0
    coef = (1 - min_lr_ratio) * 0.5
    intercept = (1 + min_lr_ratio) * 0.5

    init_lr_ratio = 0.0 if init_lr_ratio is None else init_lr_ratio
    assert init_lr_ratio >= 0 and init_lr_ratio <= 1.0

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return init_lr_ratio + (1.0 - init_lr_ratio) * (float(current_step) / float(max(1, num_warmup_steps)))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        x = math.cos(math.pi * float(num_cycles) * 2.0 * progress)
        return max(min_lr_ratio, x * coef + intercept)

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def build_lr_scheduler(optimizer, lr_scheduler_type, lr_scheduler_args, lr, rank):
    """Build an LR scheduler (constant or cosine with warmup) from lr_scheduler_args."""
    total_steps = lr_scheduler_args.get("total_training_steps", 0)
    num_warmup_steps = int(lr_scheduler_args.get("lr_warmup_steps", -1))
    min_lr_ratio = lr_scheduler_args.get("min_lr_ratio", 0.0)
    min_lr = lr_scheduler_args.get("min_lr", 0.0)
    num_cycles = lr_scheduler_args.get("num_cycles", 0.5)
    if num_warmup_steps < 0:
        num_warmup_steps_ratio = lr_scheduler_args.get("lr_warmup_steps_ratio", 0.0)
        num_warmup_steps = int(num_warmup_steps_ratio * total_steps)
    # Resolve min_lr_ratio: min_lr takes precedence if set, otherwise use min_lr_ratio
    if min_lr > 0 and lr > 0:
        min_lr_ratio = max(min_lr_ratio, min_lr / lr)

    if rank == 0:
        logger.info("Total steps: %d, num_warmup_steps: %d", total_steps, num_warmup_steps)

    if lr_scheduler_type == "constant":
        return get_constant_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=num_warmup_steps)
    elif lr_scheduler_type == "cosine":
        return get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=total_steps,
            min_lr_ratio=min_lr_ratio,
            num_cycles=num_cycles,
        )
    else:
        raise NotImplementedError(f"LR scheduler type {lr_scheduler_type} is not supported")


def get_constant_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    last_epoch: int = -1,
):
    """
    Create a constant LR schedule with a linear warmup phase.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        num_warmup_steps (int): Number of steps to ramp up the LR from 0 to initial value.
        last_epoch (int, optional): The index of the last epoch when resuming training. Defaults to -1.

    Returns:
        LambdaLR: Scheduler that increases LR linearly during warmup, then holds it constant.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1.0, num_warmup_steps))
        return 1.0

    return LambdaLR(optimizer, lr_lambda, last_epoch)
