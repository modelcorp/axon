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
LiveCodeBench Python interpreter tool.

Executes Python code in a sandboxed subprocess with timeout protection,
using the LiveCodeBench execution environment.
"""

from __future__ import annotations

import ast
import faulthandler
import multiprocessing
import queue
import signal
import sys
import traceback
from pathlib import Path

from axon.utils.rewards.code_utils.livecodebench import (
    Capturing,
    clean_if_name,
    compile_code,
    get_function,
    make_function,
    reliability_guard,
    timeout_handler,
)

# Add current directory to sys.path to support dynamic loading
_current_dir = Path(__file__).parent
if str(_current_dir) not in sys.path:
    sys.path.insert(0, str(_current_dir))

from base_code_tool import CodeTool, CodeToolOutput  # noqa: E402

# =============================================================================
# Helpers
# =============================================================================


def _ensure_return_value(code: str) -> str:
    """Convert the last expression statement into a return statement."""
    if not code.strip():
        return code
    try:
        tree = ast.parse(code)
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            tree.body[-1] = ast.Return(value=tree.body[-1].value)
            ast.fix_missing_locations(tree)
        return ast.unparse(tree)
    except (SyntaxError, Exception):
        return code


def _execute_code(code: str, timeout: int):
    """Execute code with safety measures and timeout. Returns (stdout, stderr, result)."""
    signal.signal(signal.SIGALRM, timeout_handler)
    stdout, stderr, result = None, None, None
    reliability_guard()
    signal.alarm(timeout)
    try:
        code = clean_if_name(code)
        code = make_function(code)
        compiled_sol = compile_code(code, timeout)
        if compiled_sol is None:
            return stdout, "Failed to compile code", result
        method = get_function(compiled_sol, "wrapped_function")
        if method is None:
            return stdout, "Failed to get function 'wrapped_function'", result

        signal.alarm(timeout)
        faulthandler.enable()
        with Capturing() as captured_output:
            try:
                result = method()
                signal.alarm(0)
            except SystemExit as e:
                signal.alarm(0)
                stderr = f"SystemExit: {e}"
            except Exception as e:
                signal.alarm(0)
                if "timeoutexception" in repr(e).lower():
                    stderr = "Time Limit Exceeded."
                else:
                    stderr = traceback.format_exc()
            finally:
                signal.alarm(0)
                faulthandler.disable()
        stdout = captured_output[0] if captured_output else ""
        # Coerce result to string-safe type before crossing process boundary.
        # Catches complex numbers, numpy arrays, custom objects, etc.
        if result is not None:
            try:
                result = str(result)
            except Exception:
                result = repr(result)
        return stdout, stderr, result
    except Exception:
        return stdout, stderr, result
    finally:
        signal.alarm(0)


def _wrapper_exec_fn(sample, timeout, result_queue):
    try:
        res = _execute_code(sample, timeout=timeout)
        result_queue.put(res)
    except BaseException as e:
        # Ensure we ALWAYS put a result — a bare crash here deadlocks the parent.
        try:
            result_queue.put((None, f"Fatal: {type(e).__name__}: {e}", None))
        except Exception:
            pass  # Queue itself broken — parent will hit timeout


def _lcb_sandbox(code: str, timeout: int):
    """Execute Python code in a sandboxed subprocess."""
    code = _ensure_return_value(code)
    manager = multiprocessing.Manager()
    result_queue = manager.Queue()

    p = multiprocessing.Process(target=_wrapper_exec_fn, args=(code, timeout, result_queue))
    p.start()
    p.join(timeout=timeout + 6)

    try:
        return result_queue.get(timeout=2)
    except (queue.Empty, Exception):
        return "Timeout", "", ""
    finally:
        if p.is_alive():
            p.terminate()
            p.join(timeout=1)
            if p.is_alive():
                p.kill()


# =============================================================================
# Tool class
# =============================================================================


class LCBPythonInterpreter(CodeTool):
    """Execute Python code in a LiveCodeBench sandbox environment."""

    def __init__(self):
        super().__init__(
            name="python",
            description="Execute python code in the same environment as the LiveCodeBench benchmark.",
            n_sandboxes=-1,
        )

    def forward(self, code: str, timeout: int = 12, **kwargs) -> CodeToolOutput:
        try:
            stdout, stderr, result = _lcb_sandbox(code, timeout=timeout)
            return CodeToolOutput(name=self.name, stdout=stdout, stderr=stderr, output=result)
        except Exception as e:
            return CodeToolOutput(name=self.name, error=f"Sandbox Error: {type(e).__name__}: {e}")


if __name__ == "__main__":
    # Create a Python interpreter instance
    interpreter = LCBPythonInterpreter()

    # Example code to execute
    test_code = """
# Generate a large amount of code
result = 0
for i in range(1000):
    exec(f"var_{i} = {i}")
    result += i

# Final expression after lots of code
result  # Should be converted to return
"""
    result = interpreter.forward(code=test_code)
    print(result)
