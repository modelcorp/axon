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
"""
Dispatches tool calls through the standard ``LocalToolExecutor``.

The executor can be backed by:
- Local Tool instances (via ``tool_map``)
- MCP tools (via ``mcp_manager``)
- A pre-built executor (via ``executor``)

Usage
-----
::

    # Local tools
    env = ToolEnvironment(task=task, tool_map={"calc": CalculatorTool})

    # MCP tools
    manager = MCPConnectionManager("npx", ["-y", "@mcp/server"])
    manager.start()
    env = ToolEnvironment(task=task, mcp_manager=manager)

    # Custom executor
    env = ToolEnvironment(task=task, executor=my_executor)
"""

import json
import logging
import warnings
from typing import Any

from axon.core.env import BaseEnv, register_env
from axon.tools.executors import LocalToolExecutor
from axon.tools.types import ToolCall

logger = logging.getLogger(__name__)


def _zero_reward_fn(task_info, action):
    """Fallback reward function that always returns 0."""
    from dataclasses import dataclass

    @dataclass
    class RewardOutput:
        reward: float = 0.0
        metadata: dict = None  # type: ignore

        def __post_init__(self):
            if self.metadata is None:
                self.metadata = {}

    return RewardOutput()


@register_env("tool")
class ToolEnvironment(BaseEnv):
    """
    Unified tool environment for RL training.

    Supports local tools, MCP tools, or a pre-built executor.
    Tool execution is delegated entirely to ``LocalToolExecutor``,
    eliminating ad-hoc threading and dispatch logic.
    """

    def __init__(
        self,
        task: dict | None = None,
        tool_map: dict[str, Any] | None = None,
        mcp_manager: Any | None = None,
        executor: LocalToolExecutor | None = None,
        reward_fn: Any | None = None,
        max_turns: int = 10,
    ):
        """
        Args:
            task: Task dict (passed to reward function).
            tool_map: Dict of tool name → Tool class/instance/path (builds executor).
            mcp_manager: MCPConnectionManager instance (uses its tool_map).
            executor: Pre-built LocalToolExecutor (overrides tool_map/mcp_manager).
            reward_fn: Callable(task_info, action) → RewardOutput.
            max_turns: Maximum steps before forced termination.
        """
        self.task = task
        self.max_turns = max_turns
        self.step_count = 0

        # Build executor from whatever source
        if executor is not None:
            self.executor = executor
        elif mcp_manager is not None:
            self.executor = LocalToolExecutor(mcp_manager.tool_map)
        elif tool_map is not None:
            self.executor = LocalToolExecutor(tool_map)
        else:
            self.executor = LocalToolExecutor()

        # Reward function
        if reward_fn is None:
            warnings.warn("No reward function specified, will get 0 reward.", stacklevel=2)
            self.reward_fn = _zero_reward_fn
        else:
            self.reward_fn = reward_fn

    # -- RL interface ---------------------------------------------------------

    def reset(self):
        """Reset the environment and return initial observations."""
        self.step_count = 0
        return self.task or {}, {}

    def step(self, action: list[dict] | str | dict):
        """
        Take a step.

        Args:
            action: Either a string (final response), a single tool call dict,
                    or a list of tool call dicts.

        Returns:
            (next_obs, reward, done, info)
        """
        if isinstance(action, dict):
            action = [action]
        self.step_count += 1

        # Check termination conditions
        done = self.step_count >= self.max_turns or isinstance(action, str)
        if isinstance(action, list):
            for tc in action:
                if tc.get("function", {}).get("name") == "finish":
                    done = True
                    break

        if done:
            return self._handle_done(action)

        # Execute tool calls
        tool_outputs = self._execute_tool_calls(action)
        return {"tool_outputs": tool_outputs}, 0, False, {"response": action, "metadata": {}}

    # -- Internal -------------------------------------------------------------

    def _handle_done(self, action):
        """Extract final response and compute reward."""
        if isinstance(action, str):
            llm_response = action
        elif isinstance(action, list):
            # Find finish tool call
            finish = next(
                (tc for tc in action if tc.get("function", {}).get("name") == "finish"),
                None,
            )
            if finish:
                arguments = finish.get("function", {}).get("arguments", {})
                if isinstance(arguments, str):
                    arguments = json.loads(arguments)
                llm_response = arguments.get("response", "") if isinstance(arguments, dict) else str(arguments)
            else:
                llm_response = str(action)
        else:
            llm_response = str(action)

        reward_output = self.reward_fn(task_info=self.task or {}, action=llm_response)
        return {}, reward_output.reward, True, {"response": action, "metadata": reward_output.metadata}

    def _execute_tool_calls(self, raw_tool_calls: list[dict]) -> dict[str, str]:
        """
        Execute tool calls via the executor. Returns {tool_call_id: content_str}.
        """
        tool_calls = [ToolCall.from_raw_tool_call(tc) for tc in raw_tool_calls]
        results = self.executor.execute_batch_sync(tool_calls, parallel=True)
        return {r.tool_call_id: r.content for r in results}

    # -- Factory --------------------------------------------------------------

    @staticmethod
    def from_dict(env_args: dict) -> "ToolEnvironment":
        """Build from a config dict (Hydra / command-line)."""
        tool_map = env_args.pop("tool_map", None)
        reward_fn = env_args.pop("reward_fn", None)
        max_turns = env_args.pop("max_turns", 10)

        # Hydra DictConfig → dict
        if tool_map is not None and not isinstance(tool_map, dict):
            tool_map = dict(tool_map)

        # Handle reward_fn as string import path
        if isinstance(reward_fn, str):
            reward_fn = _load_callable(reward_fn)

        # MCP support
        mcp_server_command = env_args.pop("mcp_server_command", None)
        mcp_manager = None
        if mcp_server_command:
            from .mcp_connection_manager import MCPConnectionManager

            mcp_manager = MCPConnectionManager(
                mcp_server_command,
                env_args.pop("mcp_server_args", None),
                env_args.pop("mcp_server_env", None),
            )
            mcp_manager.start()

        return ToolEnvironment(
            task=env_args,
            tool_map=tool_map,
            mcp_manager=mcp_manager,
            max_turns=max_turns,
            reward_fn=reward_fn,
        )


def _load_callable(path: str):
    """Load a callable from 'module.path:function_name' or '/path/to/file.py:fn'."""
    from axon.utils.module_loader import load_module

    if ":" not in path:
        raise ValueError(f"Invalid format: {path!r}. Expected 'module.path:function_name'")

    module_path, fn_name = path.rsplit(":", 1)
    if module_path.endswith(".py") or "/" in module_path or "\\" in module_path:
        module = load_module(module_path)
    else:
        module = load_module(f"pkg://{module_path}")

    if module is None:
        raise ImportError(f"Failed to load module from: {module_path}")
    return getattr(module, fn_name)
