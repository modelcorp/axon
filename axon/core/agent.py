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

from abc import ABC
from dataclasses import dataclass
from typing import Any

from axon.utils.registry import ClassRegistry

# Agent registry
AGENT_CLASS_MAPPING = ClassRegistry("agent")
register_agent = AGENT_CLASS_MAPPING.register


@dataclass
class Action:
    """A parsed agent response.

    Attributes:
        thought: The agent's reasoning, extracted from the model response. Often the
            content inside ``<think>`` tags or before the action block. Empty string
            when the model didn't produce a thought trace.
        action: The action to apply to the environment. Type depends on the
            environment — string for chat-style envs, dict for tool calls,
            structured object for typed actions.
    """

    thought: str = ""
    action: Any = None


class BaseAgent(ABC):
    """Helper class for agents used by :class:`~axon.programs.react_program.ReactProgram`.

    An agent owns prompt construction (via :attr:`system_prompt`) and response
    parsing (via :meth:`process_observation` and :meth:`process_action`). It
    pairs with a :class:`~axon.core.env.BaseEnv` inside a ReactProgram-style
    rollout. The rollout abstraction itself is
    :class:`~axon.programs.base_program.BaseProgram`; custom programs may bypass
    ``BaseAgent`` entirely.

    Subclasses register themselves through the ``@register_agent("name")``
    decorator so recipes can refer to them by string name in yaml.

    Subclasses must implement:
        * :meth:`process_observation` — turn an env observation into a chat message.
        * :meth:`process_action` — turn the LLM's text response into an :class:`Action`.

    Subclasses may override:
        * :attr:`system_prompt` — return a per-agent system prompt.
        * :meth:`reset` — clear per-episode state.
    """

    @property
    def system_prompt(self) -> str:
        """System prompt prepended to every conversation. Override to customise."""
        return ""

    def reset(self) -> None:
        """Clear per-episode state. Called at the start of every new rollout."""
        return

    def process_observation(self, observation: Any, reward: float, done: bool, info: dict, **kwargs) -> Any:
        """Turn an environment observation into a chat-message-shaped input.

        Args:
            observation: The environment's observation. Type depends on the env.
            reward: The reward returned by the previous environment step. ``0`` on
                the first turn.
            done: Whether the previous step terminated the episode.
            info: Environment metadata passed alongside the observation.
            **kwargs: Recipe-specific extras forwarded by the program.

        Returns:
            A chat message (string or dict) suitable for appending to the
            conversation history.
        """
        raise NotImplementedError("BaseAgent.process_observation must be implemented")

    def process_action(self, action: str) -> Action:
        """Parse the LLM's text response into an :class:`Action`.

        Args:
            action: The raw LLM response.

        Returns:
            An :class:`Action` whose ``action`` field is what the environment will
            ``step`` on, and whose ``thought`` field captures any reasoning trace.
        """
        raise NotImplementedError("BaseAgent.process_action must be implemented")


@register_agent("default")
class DefaultAgent(BaseAgent):
    """Pass-through agent. Returns the observation unchanged and treats the LLM
    response as the action verbatim. Useful as a smoke-test or as a fallback
    when the agent doesn't need to transform observations or actions."""

    def process_observation(self, observation, reward, done, info, **kwargs):
        if isinstance(observation, dict):
            return observation.get("question", str(observation))
        return str(observation)

    def process_action(self, action: str) -> Action:
        return Action(action=action)
