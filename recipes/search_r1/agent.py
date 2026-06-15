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
import re
from typing import Any

from recipes.search_r1.prompts import SEARCH_R1_SYSTEM_PROMPT
from axon.core import Action, BaseAgent, register_agent


@register_agent("search_r1")
class SearchR1Agent(BaseAgent):
    """
    Agent that performs interleaved reasoning and search using Search-R1's tag format.

    The agent learns to:
    1. Think inside <think></think> tags
    2. Search using <search>query</search> when it needs information
    3. Receive search results in <information></information> tags
    4. Provide final answer in <answer></answer> tags
    """

    def __init__(self):
        self.reset()

    @property
    def system_prompt(self) -> str:
        return SEARCH_R1_SYSTEM_PROMPT

    def process_observation(self, observation: Any, reward: float, done: bool, info: dict, **kwargs) -> str:
        """
        First turn: Just the question
        Later turns: <information>results</information> or error messages
        """
        if self.first_time:
            self.first_time = False

            # Extract question from observation
            if isinstance(observation, dict):
                question = observation.get("question", "")
            else:
                question = str(observation)

            # Ensure question ends with '?'
            question = question.strip()
            if question and not question.endswith("?"):
                question += "?"

            # Return just the question (instruction is in system prompt)
            return f"Question: {question}"
        else:
            # Subsequent turns: search results or error messages
            if isinstance(observation, dict):
                if "search_results" in observation:
                    results = observation["search_results"]
                    return f"\n\n<information>{results}</information>\n\n"
                elif "error" in observation:
                    return observation["error"]

            # Fallback for other observation types
            return str(observation) if observation else ""

    def process_action(self, action: str) -> Action:
        """Parse Search-R1 format: <search>...</search> or <answer>...</answer>"""
        pattern = r"<(search|answer)>(.*?)</\1>"
        matches = list(re.finditer(pattern, action, re.DOTALL))

        if matches:
            last_match = matches[-1]
            action_type = last_match.group(1)  # 'search' or 'answer'
            content = last_match.group(2).strip()

            return Action(action={"type": action_type, "content": content, "full_response": action})
        else:
            # Invalid format - no valid search or answer tag found
            return Action(action={"type": "invalid", "content": "", "full_response": action})

    def reset(self):
        self.first_time = True
