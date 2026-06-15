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
VerifiersProgram - Axon Program for Prime Intellect Verifiers environments.
- Verifiers controls the rollout loop (system prompt, multi-turn logic, tool handling)
- Axon controls the training data collection
- Bridge via an AsyncOpenAI-compatible adapter

Usage:
    # In config
    program:
      name: verifiers
      env_module: wordle
      sampling_params:
        temperature: 0.7
        max_tokens: 1024

    # Data prep creates parquet with env_args
    python prepare_verifiers_data.py --env-module wordle
"""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING

from axon.programs.base_program import BaseProgram, ProgramResult, register_program

if TYPE_CHECKING:
    from verifiers import Environment
    from verifiers.types import State

from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.completion_usage import CompletionUsage

# =============================================================================
# AsyncOpenAI Adapter
# =============================================================================
# Wraps BaseProgram.generate() to look like AsyncOpenAI for Verifiers.
# Verifiers' env.rollout() expects client.chat.completions.create().
# =============================================================================


class _CompletionsNamespace:
    """Implements client.chat.completions interface."""

    def __init__(self, program: VerifiersProgram):
        self._program = program

    async def create(
        self,
        messages: list[dict] | None = None,
        model: str | None = None,
        **kwargs,
    ) -> ChatCompletion:
        messages = messages or []

        # Build sampling params
        params = dict(self._program.sampling_params)

        try:
            response_text, stop_program, _ = await self._program.generate(
                messages=messages,
                sample_params=params,
            )
        except Exception:
            import traceback

            traceback.print_exc()
            raise

        choice = Choice(
            index=0,
            message=ChatCompletionMessage(
                role="assistant",
                content=response_text,
            ),
            finish_reason="stop" if not stop_program else "length",
            logprobs=None,
        )

        result = ChatCompletion(
            id=f"chatcmpl-{int(time.time() * 1000)}",
            model=model or "axon",
            object="chat.completion",
            created=int(time.time()),
            choices=[choice],
            usage=CompletionUsage(
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
            ),
        )

        return result


class _ChatNamespace:
    """Implements client.chat interface."""

    def __init__(self, program: VerifiersProgram):
        self.completions = _CompletionsNamespace(program)


class OpenAIClientAdapter:
    """
    Makes VerifiersProgram look like AsyncOpenAI.

    Verifiers expects:
        response = await client.chat.completions.create(messages=..., model=...)

    We route this through Axon's generate() method.
    """

    def __init__(self, program: VerifiersProgram):
        self._program = program
        self._model = program.env_module
        self.chat = _ChatNamespace(program)

        # Verifiers' Environment may read these attributes off what it thinks is
        # an AsyncOpenAI client. Real calls go through `_program.generate()`
        # above; the values here are unused placeholders.
        self.base_url = "in-process"
        self.api_key = "unused"


# =============================================================================
# Environment Loading
# =============================================================================

# Module-level cache with thread safety
_env_cache: dict[str, Environment] = {}
_cache_lock = threading.Lock()


def load_verifiers_environment(env_module: str, env_kwargs: dict | None = None) -> Environment:
    """
    Load a Verifiers environment with caching and thread-safety fixes.

    Uses vf.load_environment() internally but:
    1. Caches by (env_module, env_kwargs) - load once, reuse many times
    2. Patches signal.signal for non-main thread compatibility
    """
    import json

    try:
        import verifiers as vf
    except ImportError as e:
        raise ImportError("verifiers not installed. Run: pip install verifiers") from e

    env_kwargs = env_kwargs or {}
    cache_key = f"{env_module}::{json.dumps(env_kwargs, sort_keys=True)}"

    # Check cache first (fast path)
    if cache_key in _env_cache:
        return _env_cache[cache_key]

    # Load with lock (slow path)
    with _cache_lock:
        # Double-check after acquiring lock
        if cache_key in _env_cache:
            return _env_cache[cache_key]

        # Workaround: Verifiers sets signal handlers in __post_init__,
        # which fails in non-main threads. Patch signal.signal temporarily.
        import signal

        original_signal = signal.signal

        def safe_signal(signalnum, handler):
            if threading.current_thread() is threading.main_thread():
                return original_signal(signalnum, handler)
            return signal.SIG_DFL  # No-op in worker threads

        try:
            signal.signal = safe_signal
            env = vf.load_environment(env_module, **env_kwargs)
        finally:
            signal.signal = original_signal

        _env_cache[cache_key] = env
        return env


# =============================================================================
# VerifiersProgram
# =============================================================================


@register_program("verifiers")
class VerifiersProgram(BaseProgram):
    def __init__(
        self,
        # flat dict (from parquet, takes precedence)
        env_args: dict,
        # Generation (from config)
        sampling_params: dict | None = None,
        env_module: str | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        env_module = env_args.get("env_module", None) or env_module
        env_kwargs = env_args.get("env_kwargs", None)
        task = env_args.get("task", None)
        task_idx = env_args.get("task_idx", None)
        eval = env_args.get("eval", False)

        # Two preparation modes: reference (`task_idx` indexes the environment's dataset)
        # or embed (`task` carries the full, self-contained task dict — no dataset lookup).
        if task is None and task_idx is None:
            raise ValueError(
                f"env_args must contain either 'task_idx' (reference mode) or 'task' (embed mode), but got: {str(env_args)}"
            )

        if not env_module:
            raise ValueError("env_module is required (via param or env_args)")

        self.env_module = env_module
        self.env_kwargs = env_kwargs or {}
        # In embed mode there is no dataset index; task_idx is only an example-id fallback / metadata, so default it.
        self.task_idx = task_idx if task_idx is not None else 0
        self.task = task
        self.sampling_params = sampling_params or {}
        self.eval = eval

    @property
    def vf_env(self):
        return load_verifiers_environment(self.env_module, self.env_kwargs)

    def _get_task(self) -> dict:
        """
        Get task data for this rollout.

        Priority:
        1. Explicit task dict from config
        2. Index into environment's dataset

        Returns:
            Task dict with at least 'prompt' or 'question' key
        """
        if self.task is not None:
            return dict(self.task)

        dataset = self.vf_env.dataset if not self.eval else self.vf_env.eval_dataset
        if not dataset:
            raise ValueError(
                f"Environment '{self.env_module}' has no dataset for {'eval' if self.eval else 'train'} and no explicit task provided. "
                f"Either pass task= in config or ensure the environment has a dataset."
            )

        idx = self.task_idx % len(dataset)
        return dict(dataset[idx])

    def _build_rollout_input(self, task: dict) -> dict:
        """
        Build Verifiers RolloutInput from task dict.

        Maps Axon task format to what Verifiers expects.
        """
        # Get prompt - Verifiers uses 'prompt', some datasets use 'question'
        prompt = task.get("prompt") or task.get("question") or task.get("messages", [])

        rollout_input = {
            "prompt": prompt,
            "example_id": task.get("example_id", self.task_idx),
            "task": task.get("task", "default"),
        }

        # Pass through answer for rubric scoring
        if "answer" in task:
            rollout_input["answer"] = task["answer"]

        # Pass through any extra info
        if "info" in task:
            rollout_input["info"] = task["info"]

        return rollout_input

    async def run(self) -> ProgramResult:
        """
        Run one episode using Verifiers' env.rollout().

        Flow:
        1. Get task from config or dataset
        2. Create AsyncOpenAI adapter wrapping self.generate()
        3. Call env.rollout() - Verifiers controls the conversation
        4. Score with rubric
        5. Return ProgramResult

        Returns:
            ProgramResult with reward and metadata
        """
        task = self._get_task()
        rollout_input = self._build_rollout_input(task)

        # Create AsyncOpenAI-compatible adapter
        client = OpenAIClientAdapter(self)

        # Let Verifiers handle the rollout
        state: State = await self.vf_env.rollout(
            input=rollout_input,
            client=client,
            model="axon",
            sampling_args=self.sampling_params,
        )
        # Check for stored error
        if "error" in state and state["error"] is not None:
            print(f"Rollout error: {state['error']}")
            print(f"Error type: {type(state['error'])}")
            # If it wraps another exception:
            if hasattr(state["error"], "__cause__"):
                print(f"Caused by: {state['error'].__cause__}")
            import traceback

            if hasattr(state["error"], "__traceback__"):
                traceback.print_tb(state["error"].__traceback__)

        # Score with rubric if available
        # Some environments compute reward during rollout, others need explicit scoring
        if hasattr(self.vf_env, "rubric") and self.vf_env.rubric is not None:
            try:
                from verifiers.utils.async_utils import maybe_semaphore

                score_sem = await maybe_semaphore(-1)  # No concurrency limit
                await self.vf_env.rubric.score_rollout(state, score_sem=score_sem)
            except Exception as e:
                # Rubric scoring failed - log but don't crash
                # Reward may already be set during rollout
                print(f"Warning: Rubric scoring failed: {e}")

        # Extract reward and metadata
        reward = self._extract_reward(state)
        metadata = self._extract_metadata(state, task)

        return ProgramResult(
            reward=reward,
            done=True,
            metadata=metadata,
        )

    def _extract_reward(self, state: State) -> float:
        """
        Extract reward from Verifiers state.

        Handles different reward formats:
        - state["reward"]: float (most common)
        - state["rewards"]: dict with component scores
        """
        # Primary: single reward value
        reward = state.get("reward")
        if reward is not None:
            return float(reward)

        # Fallback: rewards dict
        rewards = state.get("rewards", {})
        if isinstance(rewards, (int | float)):
            return float(rewards)

        if isinstance(rewards, dict):
            # Try 'total' key first
            if "total" in rewards:
                return float(rewards["total"])
            # Sum numeric values
            total = sum(v for v in rewards.values() if isinstance(v, (int | float)))
            return float(total)

        # No reward found
        return 0.0

    def _extract_metadata(self, state: State, task: dict) -> dict:
        """Extract useful metadata from state for logging."""
        metadata = {}

        # Basic info
        metadata["task_idx"] = self.task_idx
        metadata["env_module"] = self.env_module

        # Reward components if available
        if "rewards" in state and isinstance(state["rewards"], dict):
            metadata["reward_components"] = state["rewards"]

        # Turn count from Verifiers state ("trajectory" is a Verifiers API key).
        verifiers_steps = state.get("trajectory", [])
        metadata["num_turns"] = len(verifiers_steps)

        # Completion text for debugging
        if "completion" in state:
            completion = state["completion"]
            if isinstance(completion, list) and completion:
                # Get last assistant message
                for msg in reversed(completion):
                    if isinstance(msg, dict) and msg.get("role") == "assistant":
                        content = msg.get("content", "")
                        metadata["final_response_preview"] = content[:200]
                        break

        # Any metrics from Verifiers
        if "metrics" in state:
            metadata["verifiers_metrics"] = state["metrics"]

        return metadata
