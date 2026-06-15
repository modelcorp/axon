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
Reward function for formal math (Lean 4) proof verification.

Uses the Kimina lean-server to check whether a generated proof is valid.
Returns 1.0 for valid proofs, 0.0 otherwise.
"""

import asyncio
import importlib.util
import logging
import os
import re
import sys
import threading

from axon.utils.rewards.base import RewardOutput

logger = logging.getLogger(__name__)

_VERIFICATION_TIMEOUT = 60


class _PersistentLoop:
    """A dedicated background thread running a persistent event loop.

    AsyncKiminaClient caches aiohttp sessions tied to an event loop.
    Using asyncio.run() per call creates/destroys loops, breaking cached sessions.
    This class keeps one loop alive for the lifetime of the process.
    """

    def __init__(self):
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def run_coroutine(self, coro):
        """Submit a coroutine to the persistent loop and block until done."""
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()


_PERSISTENT_LOOP: _PersistentLoop | None = None
_PERSISTENT_LOOP_LOCK = threading.Lock()


def _load_kimina_wrapper():
    """Load kimina_wrapper from the same directory, avoiding relative imports."""
    module_name = "formal_math_kimina_wrapper"
    if module_name in sys.modules:
        return sys.modules[module_name]
    wrapper_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kimina_wrapper.py")
    spec = importlib.util.spec_from_file_location(module_name, wrapper_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _extract_last_code_block(text: str) -> str | None:
    """Extract the last markdown code block from text."""
    pattern = r"```(?:\w+)?\n(.*?)```"
    matches = re.findall(pattern, text, flags=re.DOTALL)
    return matches[-1] if matches else None


def _extract_answer_code(response_code_block: str) -> str:
    """Extract the proof portion from a response code block."""
    haystack = ":="
    if haystack in response_code_block:
        return response_code_block[response_code_block.index(haystack) :]
    else:
        # Leanabell prover style: only the proof body
        return ":= by\n" + response_code_block


def _assemble_code(question: str, response: str) -> tuple[str | None, str | None]:
    """Combine the theorem statement from the question with the proof from the response."""
    prompt_code = _extract_last_code_block(question)
    if prompt_code is None:
        return None, "no_prompt_code"

    response_code = _extract_last_code_block(response)
    if response_code is None:
        return None, "no_response_code"

    if ":=" not in prompt_code:
        return None, "no_theorem_marker_in_prompt"

    theorem_statement = prompt_code[: prompt_code.index(":=")]
    proof_body = _extract_answer_code(response_code)

    return theorem_statement + proof_body, None


class _FormalMathRewardFn:
    """Stateful reward function that manages a Kimina verification cluster."""

    def __init__(self):
        self._verifier = None

    def _ensure_verifier(self):
        if self._verifier is None:
            kimina_wrapper = _load_kimina_wrapper()
            self._verifier = kimina_wrapper.KiminaServerAndClientCluster()

    async def _check_proof(self, question: str, response: str) -> RewardOutput:
        from kimina_client import SnippetStatus

        self._ensure_verifier()

        try:
            code, error_cat = _assemble_code(question, response)
            if code is None:
                return RewardOutput(
                    reward=0.0,
                    metadata={"reward_cat": error_cat},
                    is_correct=False,
                )

            resp = await self._verifier.check(snips=code, timeout=_VERIFICATION_TIMEOUT, show_progress=False)
            assert len(resp.results) == 1, f"Expected 1 result, got {len(resp.results)}"
            result = resp.results[0]
            analysis = result.analyze()
            is_valid = analysis.status == SnippetStatus.valid

            return RewardOutput(
                reward=float(is_valid),
                metadata={
                    "reward_cat": "success" if is_valid else f"lean_{analysis.status.value}",
                    "lean_result": result.model_dump(),
                    "extracted_code": code,
                },
                is_correct=is_valid,
            )
        except Exception as e:
            logger.warning(f"Error in formal math reward: {e}")
            return RewardOutput(
                reward=0.0,
                metadata={"reward_cat": "python_error", "error_details": str(e)},
                is_correct=False,
            )


_REWARD_FN: _FormalMathRewardFn | None = None


def _get_persistent_loop() -> _PersistentLoop:
    global _PERSISTENT_LOOP
    if _PERSISTENT_LOOP is None:
        with _PERSISTENT_LOOP_LOCK:
            if _PERSISTENT_LOOP is None:
                _PERSISTENT_LOOP = _PersistentLoop()
    return _PERSISTENT_LOOP


def formal_math_reward_fn(task_info: dict, action: str) -> RewardOutput:
    """
    Reward function for Lean 4 proof verification.

    Follows axon's RewardFunction protocol: (task_info, action) -> RewardOutput.
    The question (containing the Lean 4 theorem statement) comes from task_info["question"].
    The action is the model's generated proof attempt.
    """
    global _REWARD_FN
    if _REWARD_FN is None:
        _REWARD_FN = _FormalMathRewardFn()

    question = task_info.get("question", "")

    # Run the async Kimina verification on a persistent background event loop.
    # We use a persistent loop because AsyncKiminaClient caches aiohttp sessions
    # tied to a specific loop. Creating/destroying loops per call (asyncio.run)
    # would break these cached sessions on subsequent calls.
    loop = _get_persistent_loop()
    return loop.run_coroutine(_REWARD_FN._check_proof(question, action))
