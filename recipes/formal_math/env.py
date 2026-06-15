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
import importlib.util
import os
import sys

from axon.core import SingleTurnEnvironment, register_env

# Load the reward module from the same directory.
# We can't use relative imports since axon loads env.py via spec_from_file_location.
_reward_module_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reward.py")
_spec = importlib.util.spec_from_file_location("formal_math_reward", _reward_module_path)
_reward_module = importlib.util.module_from_spec(_spec)
sys.modules["formal_math_reward"] = _reward_module
_spec.loader.exec_module(_reward_module)
formal_math_reward_fn = _reward_module.formal_math_reward_fn


@register_env("formal_math")
class FormalMathEnvironment(SingleTurnEnvironment):
    """Single-turn environment for Lean 4 formal proof generation."""

    @staticmethod
    def from_dict(env_args: dict) -> "SingleTurnEnvironment":
        reward_fn = env_args.pop("reward_fn", formal_math_reward_fn)
        task = env_args
        return SingleTurnEnvironment(task=task, reward_fn=reward_fn)
