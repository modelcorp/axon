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
E2B cloud sandbox Python interpreter tool.

Executes Python code in E2B's managed sandbox environments with
round-robin distribution across multiple sandboxes.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

# Add current directory to sys.path to support dynamic loading
_current_dir = Path(__file__).parent
if str(_current_dir) not in sys.path:
    sys.path.insert(0, str(_current_dir))

from base_code_tool import CodeTool, CodeToolOutput  # noqa: E402

try:
    from e2b_code_interpreter import Sandbox
except ImportError:
    Sandbox = None

E2B_API_KEY = os.environ.get("E2B_API_KEY", None)


class E2BPythonInterpreter(CodeTool):
    """Execute Python code in E2B cloud sandboxes."""

    def __init__(self, n_sandboxes: int = 1, api_key: str | None = E2B_API_KEY):
        if Sandbox is None:
            raise ImportError("e2b_code_interpreter is not installed. Install with: pip install e2b-code-interpreter")
        if n_sandboxes < 1:
            raise ValueError("n_sandboxes must be >= 1")
        self.api_key = api_key
        self._cur_idx = 0
        self.sandboxes: list[Any] = []
        super().__init__(
            name="e2b_python",
            description="Execute python code in an E2B cloud sandbox environment.",
            n_sandboxes=n_sandboxes,
        )
        self.init_sandbox()

    def init_sandbox(self):
        self.sandboxes = [Sandbox(api_key=self.api_key, timeout=3600) for _ in range(self.n_sandboxes)]
        self._cur_idx = 0

    def kill_sandbox(self):
        for sb in self.sandboxes:
            try:
                sb.kill()
            except Exception:
                pass
        self.sandboxes = []

    def restart_sandbox(self, idx: int = 0):
        if idx < len(self.sandboxes):
            try:
                self.sandboxes[idx].kill()
            except Exception:
                pass
        self.sandboxes[idx] = Sandbox(api_key=self.api_key, timeout=3600)

    def forward(self, code: str, timeout: int = 20, **kwargs) -> CodeToolOutput:
        idx = kwargs.get("id", None)
        max_retries = kwargs.get("max_retries", 3)

        if idx is not None:
            self._cur_idx = idx % self.n_sandboxes
        else:
            self._cur_idx = (self._cur_idx + 1) % self.n_sandboxes
        sandbox = self.sandboxes[self._cur_idx]

        for attempt in range(max_retries):
            try:
                execution = sandbox.run_code(code, timeout=timeout)
                break
            except Exception:
                if attempt == max_retries - 1:
                    self.restart_sandbox(self._cur_idx)
                    return CodeToolOutput(name=self.name, error="Sandbox error, please try again.")

        result = execution.results[0].text if execution.results else None
        stdout = execution.logs.stdout[0] if execution.logs and execution.logs.stdout else None
        stderr = f"{execution.error.traceback}" if execution.error else None

        return CodeToolOutput(
            name=self.name,
            stdout=stdout or None,
            stderr=stderr or None,
            output=result or None,
        )


if __name__ == "__main__":
    interpreter = E2BPythonInterpreter()
    print(interpreter.forward(code="print('Hello'); import math; math.sqrt(4)"))
