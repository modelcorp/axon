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


@register_agent("math")
class MathAgent(BaseAgent):
    """
    A math agent that solves mathematical problems step by step, following the BaseAgent interface.
    """

    def __init__(self):
        """
        Initialize the MathAgent.
        """
        self.instruction = "Let's think step by step, and put your final answer within \\boxed{}."
        self.reset()

    def process_observation(self, observation: Any, reward: float, done: bool, info: dict, **kwargs):
        """Process environment feedback and update internal state."""

        # Format observation based on whether it's the initial problem or subsequent feedback
        if self.first_time:
            self.first_time = False
            assert isinstance(observation, dict) and "question" in observation
            question = observation["question"]
            formatted_observation = f"{question} {self.instruction}"
        else:
            # Follow-up correction prompt
            formatted_observation = "Your previous answer may or may not contain a mistake. Please review it carefully and put your final answer within \\boxed{}."
        return formatted_observation

    def process_action(self, action: str) -> Action:
        return Action(action=action)

    def reset(self):
        """Reset agent state for new episode."""
        self.first_time = True
