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
Unified tool agent for RL training.

- Uses the unified ``ToolCallParser``
- Uses ``LocalToolExecutor`` for tool collection and schema access
- Accepts local tools, MCP tools, or a pre-built executor

Usage
-----
::

    # Local tools
    agent = ToolAgent(parser_name="qwen", tool_map={"calc": CalculatorTool})

    # MCP tools (just pass the tool_map from MCPConnectionManager)
    agent = ToolAgent(parser_name="qwen", tool_map=mcp_manager.tool_map)

"""

import json
import logging
import sys
import uuid
from pathlib import Path
from typing import Any

from axon.core.agent import Action, BaseAgent, register_agent
from axon.tools.executors import LocalToolExecutor
from axon.tools.parsers.base_parser import get_tool_call_parser

# Add recipes folder to sys.path for dynamic imports
_recipes_path = Path(__file__).parent.parent
if str(_recipes_path) not in sys.path:
    sys.path.insert(0, str(_recipes_path))

logger = logging.getLogger(__name__)

TOOL_SYSTEM_PROMPT = """You are a helpful assistant with access to tools. Your goal is to solve the given task accurately by reasoning step-by-step and using the available tools when needed.

## How to Approach Problems

1. **Understand the Task**: Read the problem carefully. Identify what is being asked and what information you need.

2. **Plan Your Approach**: Before using any tools, think about what steps are needed to solve the problem. Break down complex problems into smaller parts.

3. **Use Tools Strategically**: Only call tools when they provide value. For calculations, data lookups, or operations you cannot do reliably in your head, use the appropriate tool.

4. **Verify Results**: After receiving tool output, check if the result makes sense. If something seems wrong, reconsider your approach.

5. **Provide Clear Answers**: When you have the final answer, state it clearly. For math problems, put your final numerical answer in \\boxed{} format.

## Tool Usage Format

When you need to use a tool, output your reasoning first, then make the tool call. You can call multiple tools if needed.

## Important Guidelines

- Think before you act. Always explain your reasoning before making tool calls.
- Use tools for computations rather than doing complex math in your head.
- If a tool returns an error, analyze what went wrong and try a different approach.
- Be precise with tool arguments - use exact values, not approximations.
- After getting tool results, interpret them in context of the original problem."""


logger = logging.getLogger(__name__)


@register_agent("tool")
class ToolAgent(BaseAgent):
    """
    Tool-using agent for RL training.

    Parses model output into tool calls using the unified ``ToolCallParser``,
    and formats tool schemas using ``ToolCallParser.get_tool_system_prompt()``.

    Works with any tool source: local Tool instances, MCPTool instances,
    a pre-built LocalToolExecutor, or raw schema dicts.
    """

    def __init__(
        self,
        system_prompt: str = TOOL_SYSTEM_PROMPT,
        parser_name: str = "qwen",
        tool_map: dict[str, Any] | None = None,
    ) -> None:
        """
        Args:
            system_prompt: Base system prompt.
            parser_name: Registered parser name ("qwen", "r1", "glm", "llama", etc.)
            tool_map: Dict of tool name → Tool instance/class/path (builds an executor).
        """
        self._system_prompt = system_prompt
        assert tool_map is not None, f"Tool map must be provided for tool agent: {tool_map}"
        # Resolve schemas
        self._schemas = LocalToolExecutor(tool_map).schemas

        # Parser from unified system
        self.tool_parser = get_tool_call_parser(parser_name)
        self.tools_prompt = self.tool_parser.get_tool_system_prompt(self._schemas)

        self.reset()

    @property
    def system_prompt(self) -> str:
        """System prompt with tool descriptions included."""
        if self.tools_prompt:
            return f"{self._system_prompt}\n\n{self.tools_prompt}"
        return self._system_prompt

    def process_observation(self, observation: Any, reward: float, done: bool, info: dict) -> str | list[dict]:
        """
        Format environment observation for the next model call.

        Handles:
        - ``{"question": "..."}`` → string prompt
        - ``{"tool_outputs": {"call_id": "result"}}`` → list of tool role messages
        - Plain string → pass through
        """
        if isinstance(observation, dict):
            if "question" in observation:
                return observation["question"]
            if "tool_outputs" in observation:
                return [
                    {"role": "tool", "content": output, "tool_call_id": call_id}
                    for call_id, output in observation["tool_outputs"].items()
                ]
            return str(observation)
        return str(observation) if not isinstance(observation, str) else observation

    def process_action(self, action: str) -> Action:
        """
        Parse model output into structured tool calls.

        If tool calls are found → return them in OpenAI format.
        If no tool calls → wrap as a "finish" call.
        """
        try:
            tool_calls, remaining = self.tool_parser.parse(action)
        except Exception as e:
            logger.error("Failed to parse tool calls: %s", e)
            tool_calls = []

        if tool_calls:
            return Action(action=[tc.to_openai_dict() for tc in tool_calls])

        # No tool calls parsed → agent is done
        return Action(
            action=[
                {
                    "id": f"call_{uuid.uuid4().hex[:12]}",
                    "type": "function",
                    "function": {
                        "name": "finish",
                        "arguments": json.dumps({"response": action}),
                    },
                }
            ]
        )

    def reset(self) -> None:
        """Reset agent state for a new episode."""
        pass
