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


@register_agent("geo3k")
class Geo3kAgent(BaseAgent):
    def __init__(self):
        """
        Initialize the MathAgent.
        """
        self.instruction = "Let's think step by step, and put your final answer within \\boxed{}."
        self.reset()

    def process_observation(self, observation: Any, reward: float, done: bool, info: dict, **kwargs):
        """Process environment feedback and update internal state."""
        # Initial observation should be of structure:
        # {'answer': '', 'images': [{'bytes': 'b', 'path': None}], 'problem': ''}}

        # Format observation based on whether it's the initial problem or subsequent feedback
        if self.first_time:
            self.first_time = False
            # Initial problem presentation
            assert isinstance(observation, dict) and "problem" in observation, (
                f"Expected initial dict observation with problem key, but received {observation}"
            )
            problem = observation["problem"]
            formatted_observation = f"{problem} {self.instruction}"

            images = observation.get("images", [])
            assert images and len(images) == 1, (
                f"Expected initial dict observation with exactly 1 image but received {observation}"
            )

            return [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": formatted_observation,
                        },
                        {
                            "type": "image",
                            "image": images[0],
                        },
                    ],
                },
            ]

        formatted_observation = "Your previous answer may or may not contain a mistake. Please review it carefully and put your final answer within \\boxed{}."

        return [
            {
                "role": "user",
                "content": formatted_observation,
            }
        ]

    def process_action(self, action: str) -> Action:
        return Action(action=action)

    def reset(self):
        """Reset agent state for new episode."""
        self.first_time = True
