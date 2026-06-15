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
Unified tool-call parser interface and registry.

Registering a new parser
------------------------
::

    from axon.core.tool_parser import ToolCallParser, register_parser

    @register_parser("my_format")
    class MyParser(ToolCallParser):
        ...
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from axon.tools.types import ToolCall, ToolResult

logger = logging.getLogger(__name__)


# =============================================================================
# Base class
# =============================================================================


class ToolCallParser(ABC):
    """
    Base class for model-specific tool-call parsing.

    Model caller facing:
        **parse()**                 → model text → list[ToolCall] + remaining text

    Inference/training facing:
        **format_tool_call()**     → ToolCall → model-format text (inverse of parse)
        **format_tool_result()**   → tool result content + name → model-format text for next turn
        **get_tool_system_prompt()** → list[dict] → system-prompt section describing tools

    The ChatTemplateParser handles chat-level wrapping (role tokens, grouping).
    The ToolCallParser handles format-specific content (the body inside those tokens).
    """

    @abstractmethod
    def parse(self, response: str) -> tuple[list[ToolCall], str]:
        """
        Extract tool calls from raw model output.

        Returns
        -------
        tool_calls : list[ToolCall]
            Parsed tool invocations.
        remaining_text : str
            The response with tool-call markup stripped out.
        """
        ...

    @abstractmethod
    def format_tool_call(self, tool_call: ToolCall) -> str:
        """
        Render ToolCall object as model-native text (inverse of parse).

        Used by ChatTemplateParser.parse_assistant() to render tool calls.

        Output does NOT include chat template tokens — those are added
        by the ChatTemplateParser.
        """
        ...

    def format_tool_calls(self, tool_calls: list[ToolCall]) -> str:
        """Render multiple ToolCalls."""
        return "\n".join(self.format_tool_call(tc) for tc in tool_calls)

    @abstractmethod
    def format_tool_result(self, content: Any, name: str = "") -> str:
        """
        Render tool execution result as model-specific text for the
        next conversation turn.

        Output does NOT include chat template tokens.
        """
        ...

    def format_tool_results(self, results: list[ToolResult]) -> str:
        """Render multiple ToolResults for the next conversation turn."""
        return "\n".join(self.format_tool_result(r.content, r.name) for r in results)

    def get_tool_system_prompt(self, tools_json: list[dict[str, Any]]) -> str:
        """
        Generate the system-prompt section that describes available tools.

        Default returns empty string — override for formats that need
        an explicit tool prompt.
        """
        return ""


# =============================================================================
# Registry
# =============================================================================

TOOL_CALL_PARSERS: dict[str, type[ToolCallParser]] = {}


def register_parser(name: str):
    """
    Class decorator to register a parser.
    """

    def decorator(cls: type[ToolCallParser]) -> type[ToolCallParser]:
        if name in TOOL_CALL_PARSERS:
            logger.warning("Overwriting parser registration for '%s'", name)
        TOOL_CALL_PARSERS[name] = cls
        return cls

    return decorator


def get_tool_call_parser(name: str, **kwargs) -> ToolCallParser:
    """
    Instantiate a parser by its registered name.

    Raises ``ValueError`` if the name is unknown.
    """
    cls = TOOL_CALL_PARSERS.get(name)
    if cls is None:
        raise ValueError(f"Unknown tool_call_parser '{name}'. Available: {sorted(TOOL_CALL_PARSERS)}")
    return cls(**kwargs)


# =============================================================================
# Convenience: ensure all built-in parsers are registered on import
# =============================================================================
# Each parser file uses @register_parser(...) so we just need to import them.
# This is done at the bottom to avoid circular imports.


def _register_builtin_parsers():
    """Import all built-in parser modules to trigger registration."""
    # These imports have side effects (registration via decorator)
    import axon.tools.parsers.gemma4_parser  # noqa: F401
    import axon.tools.parsers.glm_parser  # noqa: F401
    import axon.tools.parsers.json_parser  # noqa: F401
    import axon.tools.parsers.openai_harmony_parser  # noqa: F401
    import axon.tools.parsers.qwen_parser  # noqa: F401
    import axon.tools.parsers.r1_parser  # noqa: F401
    import axon.tools.parsers.xml_parser  # noqa: F401


_register_builtin_parsers()
