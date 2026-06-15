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
from axon.core import SingleTurnEnvironment, register_env
from axon.utils.rewards import math_reward_fn


@register_env("math")
class MathEnvironment(SingleTurnEnvironment):
    @staticmethod
    def from_dict(env_args: dict) -> "SingleTurnEnvironment":
        reward_fn = env_args.pop("reward_fn", math_reward_fn)
        task = env_args
        return SingleTurnEnvironment(task=task, reward_fn=reward_fn)
