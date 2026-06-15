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

from axon.core import Action, BaseAgent, register_agent
from axon.tools.parsers.xml_parser import XMLToolCallParser
from axon.tools.types import ToolCall

logger = logging.getLogger(__name__)

ROUTER_SYSTEM_PROMPT = """You are an agent who is provided am input question and tasked to solve the problem. You have access to an expert who can help you solve the problem. Call the expert when you think you need help.

We have access to the following functions:

---- BEGIN FUNCTION #1: call_expert ----
Description: Calls an external expert for help. No parameters are required for this function.
---- END FUNCTION #1 ----

---- BEGIN FUNCTION #2: finish ----
Description: Finish the interaction once the task is complete or if no further progress can be made.

Behavior notes:
  •	The finish command finalizes your output.

Parameters:
  (2) answer (string, required): The answer to the input question within \\boxed{}.
---- END FUNCTION #2 ----

If you choose to call a function ONLY reply in the following format with NO suffix:

<function=example_function_name>
<parameter=example_parameter_1>value_1</parameter>
<parameter=example_parameter_2>
This is the value for the second parameter
that can span
multiple lines
</parameter>
</function>

<IMPORTANT>
Reminder:
- Function calls MUST follow the specified format, start with <function= and end with </function>
- Required parameters MUST be specified
- Only call one function at a time. Do not output multiple function calls in one response.
- VERY IMPORTANT: Each response must include both reasoning (as natural text) and function call (in above format) to solve the task.
"""


def parse_xml_response(response_text: str) -> tuple[str, ToolCall | None]:
    """
    Extracts:
    - thought: everything before the first <function=...> block
    - action: the parsed ToolCall from the first <function=...></function> block
    Returns (thought, tool_call).
    """

    # Regex to match (non-greedily) from `<function=` up to the first `</function>`
    pattern = re.compile(r"(?s)(<function=.*?</function>)")
    match = pattern.search(response_text)

    if match:
        action_str = match.group(1)  # The entire <function=...></function> block
        thought = response_text[: match.start()]  # Everything before the block
    else:
        # If no match, treat entire text as "thought"
        thought = response_text
        action_str = ""

    # Strip leading/trailing whitespace
    thought = thought.strip()
    action_str = action_str.strip()

    # convert action to ToolCall object
    tool_calls, _ = XMLToolCallParser().parse(action_str)
    assert len(tool_calls) <= 1, f"Should only be 1 function in action {action_str}"

    return thought, tool_calls[0] if len(tool_calls) == 1 else None


@register_agent("router")
class RouterAgent(BaseAgent):
    """
    An router agent that routes the user to the appropriate expert model if needed.
    """

    def __init__(
        self,
    ):
        self.user_prompt = "Let's think step by step, and put your final answer within \\boxed{}."
        self.reset()

    @property
    def system_prompt(self):
        return ROUTER_SYSTEM_PROMPT

    def process_observation(self, observation: Any, reward: float, done: bool, info: dict, **kwargs):
        """Process environment feedback and update internal state."""

        # From the initial user prompt
        if self.first_time:
            formatted_observation = f"{observation} {self.user_prompt}"
            self.first_time = False
        else:
            formatted_observation = observation

        max_turns = info.get("max_turns", None)
        if max_turns:
            remaining_steps = max_turns - self.step
            if remaining_steps > 0:
                formatted_observation += f"\nSteps Remaining: {remaining_steps}."
            else:
                formatted_observation += (
                    "\nYou have reached the maximum number of steps. Please submit your answer NOW."
                )
        return formatted_observation

    def process_action(self, action: str) -> Action:
        """
        Updates the agent's internal state based on the model's response.
        """
        thought, _ = parse_xml_response(action)
        self.step += 1
        return Action(thought=thought, action=action)

    def reset(self):
        """Reset agent state for new episode."""
        self.step = 0
        self.first_time = True
