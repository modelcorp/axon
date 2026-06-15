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
Calculator tool — minimal example of a concrete Tool implementation.

Shows the pattern: subclass Tool, define ``json``, implement ``forward()``.
"""

from axon.tools.tools import Tool
from axon.tools.types import ToolOutput


class CalculatorTool(Tool):
    """Perform basic arithmetic calculations."""

    def __init__(self):
        super().__init__(
            name="calculator",
            description="Perform basic arithmetic calculations. Supports add, subtract, multiply, divide.",
        )

    @property
    def json(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "operation": {
                            "type": "string",
                            "enum": ["add", "subtract", "multiply", "divide"],
                            "description": "The arithmetic operation to perform",
                        },
                        "a": {"type": "number", "description": "First operand"},
                        "b": {"type": "number", "description": "Second operand"},
                    },
                    "required": ["operation", "a", "b"],
                },
            },
        }

    def forward(self, operation: str, a: float, b: float, **kwargs) -> ToolOutput:
        ops = {
            "add": lambda: a + b,
            "subtract": lambda: a - b,
            "multiply": lambda: a * b,
            "divide": lambda: a / b if b != 0 else None,
        }
        if operation not in ops:
            return ToolOutput(name=self.name, error=f"Unknown operation: {operation}")
        if operation == "divide" and b == 0:
            return ToolOutput(name=self.name, error="Division by zero")
        return ToolOutput(name=self.name, output=f"Result: {ops[operation]()}")
