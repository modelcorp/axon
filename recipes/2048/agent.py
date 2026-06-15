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
import logging
import re
from typing import Any

from env import Game2048Env  # noqa: E402

from axon.core import Action, BaseAgent, register_agent

logger = logging.getLogger(__name__)

SYSTEM_PROMPT: str = """You are an excellent 2048 player.

2048 Quick Guide
Goal: Combine tiles with the same value by sliding them together to reach the target tile (2048).
When two tiles with the same number touch, they merge into one tile with double the value.
After every move, a new tile (usually 2, sometimes 4) appears on a random empty cell.

Board:
- Each cell is shown as a number. Empty cells are shown as '.'.
- Cells are separated by TABs. Rows are separated by newlines.

Valid Actions (separated by | ):
Left | Down | Right | Up

Rewards:
- Invalid/ineffective move: small negative penalty
- Game over: fraction of log2(max_tile) / log2(2048)
- Reach 2048: +1.0

Strategy tips:
- Keep the largest tile in a corner.
- Build a monotonic chain along one edge so merges cascade.
- Avoid moves that break your chain unless you have no choice.

You will be shown the current board. Decide the next action.
Show your thinking briefly, then output the final action in ``` ``` fences.
Example: ```Left```. The final action MUST be one of Left, Down, Right, Up.
"""


@register_agent("2048")
class Game2048Agent(BaseAgent):
    def __init__(self, **kwargs: Any):
        self.step: int = 0
        self.last_observation: Any = None
        self.reset()

    @property
    def system_prompt(self):
        return SYSTEM_PROMPT

    def process_observation(self, observation: Any, reward: float, done: bool, info: dict, **kwargs):
        new_obs_str = "Current Board:\n" + str(observation)
        if not done:
            new_obs_str += "\nYou have not reached 2048 yet. Please give the next action."

        if self.last_observation and self.last_observation == new_obs_str:
            new_obs_str += (
                "\nYour last action was invalid or ineffective (the board did not change)."
                " You must choose a different direction. Remember to put the final action in ``` ``` fences,"
                " and it MUST be one of Left, Down, Right, Up."
            )
        self.last_observation = new_obs_str
        return new_obs_str

    def process_action(self, action: str):
        DIRECTION_MAP = {"left": 1, "down": 2, "right": 3, "up": 4}

        thought = action
        action_str = str(Game2048Env.INVALID_ACTION)

        matches = re.findall(r"```(.*?)```", action, re.DOTALL)

        if matches:
            last_match_content = matches[-1].strip()
            last_match_index = action.rfind(f"```{last_match_content}```")
            if last_match_index != -1:
                thought = action[:last_match_index].strip()

            extracted_text = last_match_content.lower()

            if extracted_text in DIRECTION_MAP:
                action_str = str(DIRECTION_MAP[extracted_text])
            elif extracted_text.isdigit() and int(extracted_text) in DIRECTION_MAP.values():
                action_str = str(int(extracted_text))

        return Action(thought=thought, action=action_str)

    def reset(self) -> None:
        self.step = 0
        self.last_observation = None
