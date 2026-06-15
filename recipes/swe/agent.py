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
import json
import logging
import re
from typing import Any

try:
    from r2egym.agenthub.action import Action as SWEAction
except ImportError:
    SWEAction = None

from recipes.swe.prompts import (
    SWE_SYSTEM_PROMPT,
    SWE_SYSTEM_PROMPT_FN_CALL,
    SWE_USER_PROMPT,
    SWE_USER_PROMPT_FN_CALL,
    SWEAGENT_SYSTEM_PROMPT,
    SWEAGENT_USER_PROMPT,
)
from axon.core import Action, BaseAgent, register_agent

TOKEN_WARNING_THRESHOLD = 28000


def parse_oai_response(response):
    thought = response.choices[0].message.content
    if not thought:
        thought = ""
    try:
        function_name = response.choices[0].message.tool_calls[0].function.name
        parameters = json.loads(response.choices[0].message.tool_calls[0].function.arguments)
        action = SWEAction(function_name, parameters)
    except Exception:
        action = SWEAction(function_name="", parameters={})
    return thought, action


def parse_xml_response(response_text: str) -> tuple[str, "SWEAction"]:
    """
    Extracts:
    - thought: everything before the first <function=...> block
    - action: the entire first <function=...></function> block
    Returns (thought, action).
    """
    # Regex to match (non-greedily) from `<function=` up to the first `</function>`
    pattern = re.compile(r"(?s)(<function=.*?</function>)")
    match = pattern.search(response_text)

    if match:
        action = match.group(1)  # The entire <function=...></function> block
        thought = response_text[: match.start()]  # Everything before the block
    else:
        # If no match, treat entire text as "thought"
        thought = response_text
        action = ""

    # Strip leading/trailing whitespace
    thought = thought.strip()
    action = action.strip()

    # convert action to Action object
    action = SWEAction.from_string(action)

    return thought, action


logger = logging.getLogger(__name__)


@register_agent("swe")
class SWEAgent(BaseAgent):
    def __init__(self, use_fn_calling: bool = False, scaffold: str = "r2egym"):
        self.use_fn_calling = use_fn_calling
        self.scaffold = scaffold
        assert scaffold in [
            "r2egym",
            "sweagent",
        ], f"Invalid scaffold: {scaffold}, must be one of ['r2egym', 'sweagent']"
        self.user_prompt_template = SWE_USER_PROMPT_FN_CALL if use_fn_calling else SWE_USER_PROMPT
        if scaffold == "sweagent":
            self.user_prompt_template = SWEAGENT_USER_PROMPT
        self.reset()

    @property
    def system_prompt(self):
        self._system_prompt = SWE_SYSTEM_PROMPT_FN_CALL if self.use_fn_calling else SWE_SYSTEM_PROMPT
        if self.scaffold == "sweagent":
            self._system_prompt = SWEAGENT_SYSTEM_PROMPT
        return self._system_prompt

    def process_action(self, action: str) -> Action:
        """
        Processes the model's response to extract thought and action components.

        Parses the response using either function calling or XML parsing based on agent configuration.

        Args:
            response (str): The raw text response from the model.

        Returns:
            Action: An Action object containing the thought and action.
        """
        if self.use_fn_calling:
            thought, action = parse_oai_response(action)
            return Action(thought=thought, action=action)
        thought, action = parse_xml_response(action)
        self.step += 1
        return Action(thought=thought, action=action.to_xml_string())

    def process_observation(self, observation: Any, reward: float, done: bool, info: dict, **kwargs):
        # If the first step in environment, we need to update the state from the environment
        if not self.first_time:
            observation = str(observation)
        else:
            observation = str(observation)
            observation = self.user_prompt_template.format(problem_statement=observation)
            self.first_time = False

        max_turns = info.get("max_turns", None)
        if max_turns:
            remaining_steps = max_turns - self.step - 1
            if remaining_steps > 0:
                observation += f"\nSteps Remaining: {remaining_steps}"
            else:
                observation += "\nYou have reached the maximum number of steps. Please submit your answer NOW."

        cur_tokens = info.get("cur_tokens", None)
        if cur_tokens is not None and cur_tokens >= TOKEN_WARNING_THRESHOLD:
            observation += "\nYou are running out of tokens. Please submit your answer NOW."
        return observation

    def reset(self):
        self.first_time = True
        self.step = 0
