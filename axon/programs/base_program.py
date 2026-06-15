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
import asyncio
from dataclasses import dataclass, field

import httpx

from axon.engine.engine import Engine
from axon.utils.registry import ClassRegistry

# Program registry
PROGRAM_CLASS_MAPPING = ClassRegistry("program")
register_program = PROGRAM_CLASS_MAPPING.register


@dataclass
class ProgramResult:
    """Final outcome of one program rollout.

    Attributes:
        reward: Terminal scalar reward for the rollout.
        done: ``True`` if the program terminated cleanly. ``False`` indicates a
            timeout, retry-limit-exceeded, or other abort that should be filtered
            out before training.
        metadata: Free-form dict for downstream consumers (env extras, debugging).
        step_rewards: Optional per-step rewards keyed by step index. Used by
            recipes that want step-wise advantage attribution.
    """

    reward: float
    done: bool
    metadata: dict = field(default_factory=dict)
    step_rewards: dict[int, float] = field(default_factory=dict)


class BaseProgram:
    """Base class for *programs* — the Axon rollout abstraction.

    A program defines what one rollout is. It owns:

    * the workflow (when to call the LLM, what to do with each response, when
      to terminate);
    * tokenisation and chat-template rendering for the underlying LLM;
    * partial-rollout suspend/resume across weight updates;
    * emission of per-token training signals (response masks, sampler logprobs,
      MoE routing decisions) that the trainer consumes.

    The shipped :class:`~axon.programs.react_program.ReactProgram` is the
    concrete program for ReAct-style multi-turn loops (math, code, FrozenLake,
    SWE, search-r1, tool use). It pairs with :class:`~axon.core.agent.BaseAgent`
    (prompt construction, response parsing) and :class:`~axon.core.env.BaseEnv`
    (world state, reward) as helpers. Custom programs whose rollout shape isn't
    "agent → action → env → obs → repeat" subclass ``BaseProgram`` directly and
    don't need an Agent or Environment — parallel solvers, multi-agent, search
    trees. See ``recipes/parallel_thinker/`` for a worked example.

    Subclasses register themselves through ``@register_program("name")`` so
    recipes can refer to them in yaml as ``program.name: <name>``. Subclass and
    override the async ``run`` method (and any other extension points) to define
    a new workflow.

    Args:
        group_id: Group identifier shared across rollouts that need to be
            advantage-normalised together (e.g., GRPO groups).
        sample_params: Per-rollout overrides forwarded to the sampling client.
        endpoint_url: Optional engine HTTP endpoint URL when running in
            HTTP-driven mode.
        retry_limit: How many times to retry the rollout on transient failure.
        program_timeout: Wall-clock timeout in seconds before the rollout is
            forcibly aborted.
    """

    def __init__(
        self,
        group_id: str = "",
        sample_params: dict | None = None,
        endpoint_url: str = "",
        retry_limit: int = 1,
        program_timeout: int = 10800,
    ):
        self.group_id = group_id
        self.sample_params = sample_params or {}
        self.endpoint_url = endpoint_url
        self.retry_limit = retry_limit
        self.program_timeout = program_timeout

        # Internal state
        self.session_id = None
        self.engine = None
        self._http_client = None  # Persistent HTTP client for API requests

    def set_sample_params(self, sample_params: dict):
        self.sample_params.update(sample_params)

    def set_engine(self, engine: Engine):
        """Set the engine."""
        self.engine = engine

    def set_endpoint_url(self, endpoint_url: str):
        self.endpoint_url = endpoint_url

    def set_group_id(self, group_id: str):
        self.group_id = group_id

    @property
    def api_base_url(self) -> str:
        """Get the API base URL from the endpoint URL."""
        return self.endpoint_url

    def _get_http_client(self) -> httpx.AsyncClient:
        """
        Get or create a persistent HTTP client for API requests.

        This client is reused across all API calls to avoid the overhead of
        creating new connections for each request. It uses connection pooling
        and keep-alive for better performance.
        """
        if self._http_client is None:
            # Create a persistent client with optimized settings
            self._http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(3600.0, connect=5.0),  # Long timeout for generation, short for connect
                limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),  # Increased connection pooling
                http2=True,  # Enable HTTP/2 for better performance with multiplexing
            )
        return self._http_client

    async def _close_http_client(self):
        """Close the HTTP client if it exists."""
        if self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None

    async def init_session(self, group_id: str | None = None, sample_params: dict | None = None):
        """
        Initialize a new session with the execution engine.

        Creates a new session ID by calling the engine's init_session method (via API if enabled)
        and stores it in self.session_id for use in subsequent engine interactions.
        If a session already exists, this is a no-op and returns the existing session id.
        """
        # Idempotent: if already initialized (e.g., pre-assigned by engine), return it
        if self.session_id is not None:
            return self.session_id

        if self.engine:
            self.session_id = await self.engine.run_in_engine_loop_async(
                self.engine.init_session(group_id, sample_params)
            )
            return self.session_id

        # Use persistent HTTP client
        client = self._get_http_client()
        response = await client.post(
            f"{self.api_base_url}/init_session", json={"group_id": group_id, "sample_params": sample_params}
        )
        response.raise_for_status()
        result = response.json()
        self.session_id = result["session_id"]
        return self.session_id

    async def generate(
        self, messages: list[dict[str, str]], sample_params: dict | None = None, parser_kwargs: dict | None = None
    ):
        """
        Send messages to the agent execution engine to generate a response.

        Args:
            messages: List of message dictionaries containing chat completions.
                     Each message should have 'role' and 'content' keys following
                     the standard chat completion format (e.g., system, user, assistant).
            sample_params: Optional sampling parameters for generation.

        Returns:
            tuple: (response_text, stop_program, step_idx)
        """
        if sample_params is None:
            sample_params = {}

        if self.engine:
            return await self.engine.run_in_engine_loop_async(
                self.engine.generate(
                    messages, session_id=self.session_id, sample_params=sample_params, parser_kwargs=parser_kwargs
                )
            )
        # Use persistent HTTP client
        assert not parser_kwargs, f"parser_kwargs not supported in API server mode yet: {parser_kwargs}"
        client = self._get_http_client()
        response = await client.post(
            f"{self.api_base_url}/generate",
            json={"messages": messages, "session_id": self.session_id, "sample_params": sample_params},
        )
        response.raise_for_status()
        result = response.json()
        return result["response"], result["stop_program"], result.get("step_idx", -1)

    async def end_session(self, reward: float, step_rewards: dict[int, float] | None = None):
        """
        End the session with the execution engine.

        Args:
            reward: The final reward for the session.
        """
        # Use API if available, otherwise direct engine access
        if self.engine:
            return await self.engine.run_in_engine_loop_async(
                self.engine.end_session(self.session_id, reward=reward, step_rewards=step_rewards)
            )

        # Use persistent HTTP client
        client = self._get_http_client()
        payload = {"session_id": self.session_id, "reward": reward}
        if step_rewards:
            # JSON keys must be strings; convert int step_idx keys
            payload["step_rewards"] = {str(k): v for k, v in step_rewards.items()}
        response = await client.post(f"{self.api_base_url}/end_session", json=payload)
        response.raise_for_status()
        return response.json()

    async def run_program(self):
        self._http_client = None
        retry_count = 0
        last_exception = None
        try:
            for attempt in range(self.retry_limit):
                if attempt > 0:
                    self.session_id = None  # Reset for fresh session on retry
                try:
                    sample_params = getattr(self, "sample_params", None)
                    # Single timeout covers init + run + end
                    await asyncio.wait_for(
                        self._run_program_inner(sample_params),
                        timeout=self.program_timeout,
                    )
                    return
                except asyncio.TimeoutError:
                    last_exception = asyncio.TimeoutError(f"Program timed out after {self.program_timeout} seconds")
                    retry_count += 1
                except Exception as e:
                    import traceback

                    traceback.print_exc()
                    last_exception = e
                    retry_count += 1
            if self.session_id is not None:
                await self.end_session(reward=-1e99)
            raise last_exception
        finally:
            await self._close_http_client()

    async def _run_program_inner(self, sample_params):
        await self.init_session(group_id=self.group_id, sample_params=sample_params)
        result = await asyncio.wait_for(self.run(), timeout=self.program_timeout)
        await self.end_session(reward=result.reward, step_rewards=result.step_rewards or None)
        return result

    async def run(self):
        raise NotImplementedError("Subclasses must implement this method")
