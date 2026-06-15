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
from typing import Any

from axon.core import Action, BaseAgent, register_agent

INSTRUCTION = (
    "Before producing the Lean 4 code to formally prove the given theorem, "
    "provide a detailed proof plan outlining the main proof steps and strategies.\n"
    "The plan should highlight key ideas, intermediate lemmas, and proof structures "
    "that will guide the construction of the final formal proof."
)


@register_agent("formal_math")
class FormalMathAgent(BaseAgent):
    """Agent for generating Lean 4 formal proofs."""

    def __init__(self):
        self.reset()

    def process_observation(self, observation: Any, reward: float, done: bool, info: dict, **kwargs):
        if self.first_time:
            self.first_time = False
            assert isinstance(observation, dict) and "question" in observation
            question = observation["question"]
            return f"{question}\n\n{INSTRUCTION}"
        else:
            return (
                "Your previous proof attempt was invalid. "
                "Please review the Lean 4 error and try again with a corrected proof."
            )

    def process_action(self, action: str) -> Action:
        return Action(action=action)

    def reset(self):
        self.first_time = True
