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
"""Reward function hub — import reward-related classes and types."""

from .base import RewardConfig, RewardFunction, RewardInput, RewardOutput, RewardType, zero_reward_fn
from .code_reward import code_reward_fn
from .f1_reward import f1_reward_fn
from .gpqa_reward import gpqa_reward_fn
from .math_reward import math_reward_fn
from .remote_reward import remote_reward_fn


# ifbench requires external repo clone — lazy import to avoid hard dep
def ifbench_reward_fn(task_info: dict, action: str) -> RewardOutput:
    """Lazy wrapper so IFBench is only cloned/imported when actually used."""
    from .ifbench_reward import ifbench_reward_fn as _impl

    return _impl(task_info, action)


REWARD_FN_REGISTRY: dict[str, RewardFunction] = {
    "math": math_reward_fn,
    "code": code_reward_fn,
    "f1": f1_reward_fn,
    "gpqa": gpqa_reward_fn,
    "ifbench": ifbench_reward_fn,
    "remote": remote_reward_fn,
    "zero": zero_reward_fn,
}


def get_reward_fn(name: str) -> RewardFunction:
    """Look up a reward function by short name."""
    if name not in REWARD_FN_REGISTRY:
        raise KeyError(f"Unknown reward function '{name}'. Available: {list(REWARD_FN_REGISTRY)}")
    return REWARD_FN_REGISTRY[name]


__all__ = [
    "RewardInput",
    "RewardOutput",
    "RewardType",
    "RewardConfig",
    "RewardFunction",
    "REWARD_FN_REGISTRY",
    "get_reward_fn",
    "zero_reward_fn",
    "math_reward_fn",
    "code_reward_fn",
    "f1_reward_fn",
    "gpqa_reward_fn",
    "ifbench_reward_fn",
    "remote_reward_fn",
]
