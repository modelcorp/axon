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

from env import SudokuEnv  # noqa: E402

from axon.core import Action, BaseAgent, register_agent

logger = logging.getLogger(__name__)

SYSTEM_PROMPT: str = """You are an expert Sudoku solver.

Sudoku Rules
- The board is an N x N grid (typically 9 x 9), divided into sqrt(N) x sqrt(N) boxes.
- Each row, column, and box must contain every digit from 1 to N exactly once.
- Some cells are pre-filled clues (immutable). The rest are empty (shown as '.').

Board format
- Rows and columns are 1-indexed and printed in the header / left margin.
- Empty cells are shown as '.'. Boxes are separated by '|' and dashed lines.

Action format
- On each turn, place ONE digit by responding with a single placement of the form:
      R<row>C<col>=<value>
  where row, col are 1-indexed and value is a digit 1..N.
- To remove a digit you previously placed (NOT a clue), use:
      R<row>C<col>=.
- Wrap your final action in triple backticks. Examples:
      ```R3C5=7```
      ```R1C1=.```

Rewards
- Valid placement: 0
- Invalid action / breaks Sudoku rules / overwrites a clue: small negative penalty
- Solve the entire puzzle: large positive reward
- Run out of turns: partial reward = fraction of blanks filled correctly

Strategy tips
- Look for cells with only one legal candidate (naked singles).
- Look for digits with only one legal cell in a row, column, or box (hidden singles).
- Don't guess randomly; if uncertain, place the move you're most confident in.

You will be shown the current board. Show your reasoning briefly, then output the
final action in ``` ``` fences. The action MUST match: R<row>C<col>=<value>.
"""

# Tolerate small format drift: optional spaces / commas / dashes / lowercase.
# Capture row, col, value (value may be a digit or '.').
_ACTION_RE = re.compile(
    r"r\s*(\d+)\s*[,\s\-x]*\s*c\s*(\d+)\s*[=,\s\-:]+\s*(\d+|\.)",
    re.IGNORECASE,
)


@register_agent("sudoku")
class SudokuAgent(BaseAgent):
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
            new_obs_str += (
                "\nPlace your next digit. Respond with the action wrapped in"
                " triple backticks, e.g. ```R3C5=7``` (or ```R3C5=.``` to clear)."
            )

        if self.last_observation and self.last_observation == new_obs_str:
            new_obs_str += (
                "\nYour last action was invalid or had no effect (the board did not change)."
                " Either it was unparseable, out of range, modified a clue, or violated"
                " Sudoku rules. Please try a different placement."
            )
        self.last_observation = new_obs_str
        return new_obs_str

    def process_action(self, action: str):
        thought = action
        action_str = SudokuEnv.INVALID_ACTION

        # Prefer a backtick-fenced match — consistent with the 2048 agent.
        fenced = re.findall(r"```(.*?)```", action, re.DOTALL)
        candidate: str | None = None
        if fenced:
            last_fence = fenced[-1].strip()
            last_fence_idx = action.rfind(f"```{last_fence}```")
            if last_fence_idx != -1:
                thought = action[:last_fence_idx].strip()
            candidate = last_fence

        # If the fenced text matches, use it. Otherwise fall back to scanning the
        # whole response for the first action-shaped pattern.
        if candidate:
            m = _ACTION_RE.search(candidate)
        else:
            m = _ACTION_RE.search(action)

        if m:
            r, c, v = m.group(1), m.group(2), m.group(3)
            action_str = f"R{int(r)}C{int(c)}={'.' if v == '.' else int(v)}"

        return Action(thought=thought, action=action_str)

    def reset(self) -> None:
        self.step = 0
        self.last_observation = None
