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
from abc import abstractmethod
from dataclasses import dataclass
from typing import Any

from axon.tools.tools import Tool, ToolOutput


@dataclass
class CodeToolOutput(ToolOutput):
    """Extended output for code-execution tools (E2B, LCB, local)."""

    stdout: str | None = None
    stderr: str | None = None

    def to_content_string(self) -> str:
        if self.error:
            return f"Error: {self.error}"
        if self.output is not None:
            return str(self.output)
        if self.stdout:
            return str(self.stdout)
        if self.stderr:
            return str(self.stderr)
        return ""

    def __str__(self) -> str:
        return self.to_content_string()

    def to_string(self) -> str:
        return self.to_content_string()


class CodeTool(Tool):
    """
    Base class for Python code execution tools.

    Subclasses implement ``forward(code, timeout)`` with their specific
    sandbox backend (LCB, E2B, local, etc.).
    """

    def __init__(self, name: str, description: str, n_sandboxes: int = 1):
        self.n_sandboxes = n_sandboxes
        super().__init__(name=name, description=description)

    @property
    def json(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "Python code to execute in the sandbox environment.",
                        },
                        "timeout": {
                            "type": "integer",
                            "description": "Maximum execution time in seconds before timing out",
                            "default": 12,
                        },
                    },
                    "required": ["code"],
                },
            },
        }

    @abstractmethod
    def forward(self, code: str, timeout: int = 12, **kwargs) -> CodeToolOutput: ...

    # Lifecycle hooks — override in subclasses as needed
    def init_sandbox(self):
        pass

    def kill_sandbox(self):
        pass

    def restart_sandbox(self):
        pass

    def __del__(self):
        self.kill_sandbox()
