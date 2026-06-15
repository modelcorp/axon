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
"""Router environment for routing the user to the appropriate expert model if needed."""

import re

from axon.core import MultiTurnEnvironment, register_env
from axon.tools.parsers.xml_parser import XMLToolCallParser
from axon.tools.types import ToolCall
from axon.utils.rewards import math_reward_fn
from axon.utils.rewards.base import RewardFunction

FORMAT_REWARD_PENALTY = 0.0


def _validate_action_format(action: str) -> tuple[bool, str | None]:
    """
    Returns (True, None) if action strictly follows:
    1) Exactly one <think>…</think>, in that order
    2) After </think>, exactly one <function=NAME>…</function> (any NAME), no trailing text
    3) Inside that function block, only matching <parameter=…>…</parameter> tags — if any
    Otherwise returns (False, error_message).
    """
    # 1) <think>…</think>
    opens = action.count("<think>")
    closes = action.count("</think>")
    if opens == 0 or closes == 0:
        return False, "Error: must include both <think>…</think> tags."
    if opens > 1 or closes > 1:
        return False, "Error: only one <think>…</think> block is allowed."
    start_think = action.find("<think>")
    end_think = action.find("</think>")
    if start_think > end_think:
        return False, "Error: <think> must come before </think>."

    # only validate suffix for the function
    suffix = action[end_think + len("</think>") :]

    # 2) one <function=…>…</function>, no trailing text
    func_openings = re.findall(r"<function=[^>]+>", suffix)
    func_closings = suffix.count("</function>")
    if len(func_openings) == 0:
        return False, "Error: missing <function=NAME>…</function> call after </think>."
    if len(func_openings) > 1 or func_closings > 1:
        return False, f"Error: only one function call allowed; found {len(func_openings)}."
    idx_end = suffix.rfind("</function>")
    if suffix[idx_end + len("</function>") :].strip():
        return False, "Error: unexpected text after </function>."

    # 3) extract inner function content
    m = re.search(r"<function=[^>]+>", suffix)
    if not m:
        return False, "Error: malformed <function> tag."
    inner = suffix[m.end() : idx_end]

    # 4) If there's any non‑whitespace inside, it must be only parameter tags
    if inner.strip():
        # matching opens vs closes
        opens_p = re.findall(r"<parameter=[^>]+>", inner)
        closes_p = re.findall(r"</parameter>", inner)
        if len(opens_p) != len(closes_p):
            return False, (
                f"Error: mismatched <parameter> tags inside function; found {len(opens_p)} opens but {len(closes_p)} closes."
            )
        # disallow *any* other tags
        if re.search(r"<(?!parameter=[^>]+>|/parameter>)", inner):
            return False, "Error: function block may only contain <parameter=…>…</parameter> tags."

    return True, None


@register_env("router")
class RouterEnv(MultiTurnEnvironment):
    """
    A router environment for routing the user to the appropriate expert model.

    This environment allows agents to either finish with an answer or call an expert
    for help. The interaction continues until the agent finishes or reaches the
    maximum number of expert calls.
    """

    def __init__(
        self,
        task: dict | None = None,
        reward_fn: RewardFunction | None = None,
        max_expert_calls: int = 1,
        max_turns: int = int(1e9),
        **kwargs,
    ):
        """
        Initialize the router environment.

        Args:
            task: Dictionary containing the task information, including at least a "question" field
            reward_fn: Function to compute rewards. Defaults to math_reward_fn if None
            max_expert_calls: Maximum number of expert calls allowed
            max_turns: Maximum agent steps (inherited, passed to agent via info dict)
            **kwargs: Additional arguments passed to parent class
        """
        super().__init__(task=task, max_turns=max_turns, **kwargs)
        if reward_fn is None:
            reward_fn = math_reward_fn
        self.reward_fn = reward_fn
        self.max_expert_calls = max_expert_calls
        self.num_expert_calls = 0
        self.tool_parser = XMLToolCallParser()

    def reset(self):
        super().reset()
        return self.task["question"], {"max_turns": self.max_turns}

    def step(self, action: str):
        """
                Take a step in the environment based on the action.
        []
                Args:
                    action: Response string from the LLM

                Returns:
                    tuple: (next_observation, reward, terminated, info)
        """

        is_valid, error_message = _validate_action_format(action)
        if not is_valid:
            return "Response format is invalid.", FORMAT_REWARD_PENALTY, True, {}

        end_think = action.find("</think>")
        # only validate suffix for the function
        action = action[end_think + len("</think>") :]
        # Parse the action
        tool_calls, _ = self.tool_parser.parse(action)
        action_obj: ToolCall = tool_calls[0]

        # Check if action is valid
        if not action_obj and action_obj.name not in ["finish", "call_expert"]:
            if not action_obj.name:
                return (
                    "No function was called, allowed functions: [`finish`, `call_expert`]",
                    FORMAT_REWARD_PENALTY,
                    True,
                    {},
                )
            return (
                "Invalid function name, allowed functions: [`finish`, `call_expert`]",
                FORMAT_REWARD_PENALTY,
                True,
                {},
            )

        # Handle finish action
        if action_obj.name == "finish":
            reward = self.reward_fn(task_info=self.task, action=action_obj.arguments.get("answer", ""))
            return "Finished.", reward.reward, True, {}

        # Handle call_expert action
        elif action_obj.name == "call_expert":
            if self.num_expert_calls >= self.max_expert_calls:
                return (
                    "You have reached the maximum number of expert calls. You may no longer call the expert.",
                    0,
                    False,
                    {},
                )
            self.num_expert_calls += 1
            obs = f"Expert calls remaining: {self.max_expert_calls - self.num_expert_calls}. Expert says the answer is: {self.task['answer']}."
            return obs, 0.0, False, {}

    @staticmethod
    def from_dict(env_args: dict) -> "RouterEnv":
        """
        Create a RouterEnv instance from a dictionary of arguments.

        Args:
            env_args: Dictionary containing environment configuration

        Returns:
            RouterEnv instance
        """
        reward_fn = env_args.pop("reward_fn", None)
        max_turns = env_args.pop("max_turns", int(1e9))
        task = env_args
        return RouterEnv(task=task, reward_fn=reward_fn, max_turns=max_turns)


if __name__ == "__main__":
    action = """
    <think>
    I need to solve the problem.
    </function>
    </think>
    </function></function></function></function>
    
    <function=call_expert>
    <parameter=question>
    What is the answer to 1+1?
    </parameter>
    </function>
    """
    is_valid, error_message = _validate_action_format(action)
    print(is_valid, error_message)
