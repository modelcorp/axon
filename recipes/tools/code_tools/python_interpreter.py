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
import sys
from pathlib import Path
from typing import Any, Literal

# Add current directory to sys.path to support dynamic loading
_current_dir = Path(__file__).parent
if str(_current_dir) not in sys.path:
    sys.path.insert(0, str(_current_dir))

from base_code_tool import CodeTool, CodeToolOutput  # noqa: E402
from e2b_tool import E2BPythonInterpreter  # noqa: E402
from lcb_tool import LCBPythonInterpreter  # noqa: E402

# Backend types
BackendType = Literal["local", "e2b", "lcb"]


class PythonInterpreter(CodeTool):
    """
    A unified Python interpreter tool that supports multiple backends.

    This class provides a common interface for executing Python code using different
    backend implementations, including local execution, E2B sandbox,
    and LiveCodeBench environment.
    """

    def __init__(
        self,
        backend: BackendType = "local",
        n_sandboxes: int = 1,
        api_key: str | None = None,
        name: str = "python",
        description: str = "Execute Python code in a sandboxed environment. Returns results and standard output/error.",
    ):
        """
        Initialize the unified Python interpreter with the specified backend.

        Args:
            backend: The backend to use ("local", "e2b", or "lcb")
            n_sandboxes: Number of concurrent sandboxes/workers to use (for applicable backends)
            api_key: API key for cloud-based backends (e2b)
            name: The name of the tool
            description: Description of what the tool does
        """
        self.backend_type = backend
        self.n_sandboxes = n_sandboxes
        self.api_key = api_key

        # Initialize the appropriate backend
        self._init_backend()

        super().__init__(name=name, description=description, n_sandboxes=n_sandboxes)

    def _init_backend(self):
        """Initialize the selected backend interpreter."""
        if self.backend_type == "local" or self.backend_type == "lcb":
            self.backend: LCBPythonInterpreter | E2BPythonInterpreter = LCBPythonInterpreter()
        elif self.backend_type == "e2b":
            self.backend = E2BPythonInterpreter(n_sandboxes=self.n_sandboxes, api_key=self.api_key)
        else:
            raise ValueError(f"Unsupported backend type: {self.backend_type}")

    def forward(self, code: str, timeout: int = 12, **kwargs) -> CodeToolOutput:
        """
        Execute Python code using the selected backend.

        Args:
            code: Python code to execute
            timeout: Maximum execution time in seconds
            **kwargs: Additional parameters specific to the backend implementation

        Returns:
            CodeToolOutput containing execution results, stdout, and stderr
        """
        return self.backend.forward(code=code, timeout=timeout, **kwargs)

    def _init_sandbox(self):
        """Initialize the sandbox environment."""
        if hasattr(self, "backend") and hasattr(self.backend, "_init_sandbox"):
            self.backend._init_sandbox()

    def _kill_sandbox(self):
        """Clean up all sandbox resources."""
        if hasattr(self, "backend") and hasattr(self.backend, "_kill_sandbox"):
            self.backend._kill_sandbox()

    def _restart_sandbox(self):
        """Restart the sandbox environment."""
        if hasattr(self, "backend") and hasattr(self.backend, "_restart_sandbox"):
            self.backend._restart_sandbox()
        else:
            self._kill_sandbox()
            self._init_backend()

    @property
    def json(self) -> dict[str, Any]:
        """Return the tool's information in the required format."""
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
                            "description": "Execute Python code in a sandboxed environment. Returns results and standard output/error.",
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
