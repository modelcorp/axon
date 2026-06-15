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
Tool base classes for Axon.

Defines the base ``Tool`` class and its specialisations.

Classes
-------
Tool       : Base class for all tools (sync + async).
MCPTool    : Wrapper for MCP (Model Context Protocol) tools.
"""

from __future__ import annotations

import inspect
import logging
import traceback
import typing
from typing import Any, cast

from axon.tools.types import ToolOutput

logger = logging.getLogger(__name__)


def _function_to_schema(func):
    """
    Converts a function into a OpenAI JSON Schema representation.

    Parameters:
        func (function): The function to convert.

    Returns:
        dict: A dictionary representing the function in JSON Schema format.
    """
    # Get the function name
    func_name = func.__name__

    # Get the docstring
    docstring = func.__doc__ or ""

    # Get the function signature
    sig = inspect.signature(func)

    # Initialize the parameters dictionary
    params = {"type": "object", "properties": {}, "required": []}
    required_list = cast(list[str], params["required"])  # Type hint for mypy
    properties_dict = cast(dict[str, Any], params["properties"])  # Type hint for mypy

    # Map Python types to JSON Schema types
    type_mapping = {
        int: "integer",
        float: "number",
        bool: "boolean",
        str: "string",
        dict: "object",
        list: "array",
    }

    for param_name, param in sig.parameters.items():
        # Get the type annotation
        annotation = param.annotation

        param_type = "string"  # Default type
        param_description = ""

        # Determine the JSON Schema type and description
        if annotation != inspect.Parameter.empty:
            # Handle Annotated types
            origin = typing.get_origin(annotation)
            if origin is typing.Annotated:
                args = typing.get_args(annotation)
                base_type = args[0]
                metadata = args[1:]
                param_type = type_mapping.get(base_type, "string")
                # Assuming the first metadata argument is the description
                if metadata:
                    param_description = metadata[0]
            else:
                param_type = type_mapping.get(annotation, "string")
        # Add the parameter to properties
        param_schema = {"type": param_type}
        if param_description:
            param_schema["description"] = param_description
        properties_dict[param_name] = param_schema

        # Add to required if there's no default value
        if param.default == inspect.Parameter.empty:
            required_list.append(param_name)

    # Build the final dictionary
    function_dict = {
        "type": "function",
        "function": {
            "name": func_name,
            "description": docstring.strip().split("\n")[0],  # First line of docstring
            "parameters": params,
        },
    }

    return function_dict


# =============================================================================
# Base Tool Class
# =============================================================================


class Tool:
    """Base class for tools that an agent can call during a rollout.

    A tool is callable code (synchronous via :meth:`forward` or asynchronous via
    :meth:`async_forward`) plus a JSON-schema description (:attr:`json`) used to
    build the chat-template tool-call surface.

    Three ways to define a tool:

    1. **Subclass and override** ``forward`` / ``async_forward`` and the ``json``
       property::

            class WeatherTool(Tool):
                name = "weather"
                json = {"type": "function", "function": {...}}
                def forward(self, location: str) -> str: ...

    2. **Wrap a plain callable** by passing it as ``function=``. The schema is
       auto-generated from the callable's signature::

            tool = Tool(function=my_function)

    3. **MCP tool** — see :class:`MCPTool`.

    Tools are discovered through :class:`~axon.tools.executors.LocalToolExecutor`
    /
    :class:`~axon.tools.executors.HTTPToolExecutor` and dispatched by the agent
    based on parsed tool calls (see ``axon/tools/parsers/``).
    """

    def __init__(
        self,
        name: str | None = None,
        description: str | None = None,
        function: Any | None = None,
    ):
        self._function = function

        if function is not None:
            self._json = _function_to_schema(function)
            self.name = self._json["function"]["name"]
            self.description = self._json["function"]["description"]
        else:
            if not name or not description:
                raise ValueError("Tool requires (name, description) or a function")
            self.name = name
            self.description = description
            self._json = None  # subclass must define .json property

    @property
    def json(self) -> dict[str, Any]:
        """
        Tool schema in OpenAI function-calling format::

            {
                "type": "function",
                "function": {
                    "name": "...",
                    "description": "...",
                    "parameters": { ... }
                }
            }
        """
        if self._json is not None:
            return self._json
        raise NotImplementedError(f"{self.__class__.__name__} must implement the `json` property")

    def forward(self, **kwargs) -> ToolOutput:
        """Synchronous execution. Override in subclasses."""
        if self._function is not None:
            try:
                output = self._function(**kwargs)
                return ToolOutput(name=self.name, output=output)
            except Exception as e:
                return ToolOutput(name=self.name, error=f"{type(e).__name__}: {e}")
        raise NotImplementedError(f"{self.__class__.__name__} must implement forward()")

    async def async_forward(self, **kwargs) -> ToolOutput:
        """Asynchronous execution. Default delegates to forward()."""
        return self.forward(**kwargs)


# =============================================================================
# MCP Tool
# =============================================================================


class MCPTool(Tool):
    """Async-first wrapper for a tool exposed by an MCP (Model Context Protocol) server.

    Construct via :class:`~axon.tools.executors.MCPConnectionManager` or by
    passing a live MCP client session and the remote tool's metadata.
    Synchronous :meth:`forward` is intentionally not implemented — MCP calls are
    network-bound and should always go through :meth:`async_forward`.
    """

    def __init__(self, session, tool_name: str, tool_description: str, tool_schema: dict):
        self._tool_schema = tool_schema
        self.session = session
        super().__init__(name=tool_name, description=tool_description)

    @property
    def json(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self._tool_schema,
            },
        }

    async def async_forward(self, **kwargs) -> ToolOutput:
        try:
            logger.debug("Calling MCP tool: %s with args: %s", self.name, kwargs)
            result = await self.session.call_tool(self.name, kwargs)

            # Extract text from MCP result (various shapes)
            if hasattr(result, "content"):
                content = result.content
                if hasattr(content, "text"):
                    content_str = content.text
                elif isinstance(content, list) and content and hasattr(content[0], "text"):
                    content_str = content[0].text
                else:
                    content_str = str(content)
            else:
                content_str = str(result)

            return ToolOutput(name=self.name, output=content_str)
        except Exception as e:
            logger.debug("Error executing MCP tool %s: %s", self.name, e)
            traceback.print_exc()
            return ToolOutput(name=self.name, error=f"MCP error: {e}")
