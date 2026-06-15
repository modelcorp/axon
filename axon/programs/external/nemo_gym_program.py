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
NeMo Gym Program.

Two execution modes:

**builtin** (default)
    Re-implements the simple_agent tool-calling loop in-process.
    Every ``generate()`` call goes through the engine with full session
    tracking (token IDs, logprobs, prefix tree, MOE, etc.).
    Only needs a NeMo Gym resource server running.

**native**
    Delegates orchestration to NeMo Gym's own agent server.
    Axon exposes a session-aware ``/v1/chat/completions`` endpoint;
    NeMo Gym's agent server calls it as its model server.
    Session_id is threaded via the ``user`` field (``axon:{session_id}``)
    because NeMo Gym's model servers overwrite the ``model`` field.
    Supports NeMo Gym agent patterns that run through its agent server
    while keeping model calls inside Axon's session-aware endpoint.

Architecture (builtin)::

    ┌──────────────────────────────────────────────────────┐
    │  Axon Engine                           │
    │  (sessions, tokens, logprobs, prefix tree, etc.)     │
    └──────────────────────┬───────────────────────────────┘
                           │ generate()
    ┌──────────────────────┴───────────────────────────────┐
    │  NemoGymProgram(BaseProgram)                          │
    │  - cookie jar per episode for session tracking        │
    │  - tool call parsing (pluggable)                      │
    │  - agent loop: generate → parse → tool call → repeat  │
    └──────────────────────┬───────────────────────────────┘
                           │ HTTP (with cookies)
    ┌──────────────────────▼───────────────────────────────┐
    │  NeMo Gym Resource Server (ng_run, unchanged)         │
    │  /seed_session, /{tool}, /verify                      │
    └──────────────────────────────────────────────────────┘

Architecture (native)::

    ┌──────────────────────┐     ┌──────────────────────────┐
    │  Agent Server        │────▶│  Axon Engine              │
    │  (NeMo Gym's own)    │     │  /v1/chat/completions     │
    │  simple_agent,       │     │  parses user="axon:{sid}" │
    │  multi-turn, etc.    │     └──────────────────────────┘
    └────────┬─────────────┘
             │
    ┌────────▼─────────────┐
    │  Resource Server     │
    │  (tools + verify)    │
    └──────────────────────┘

Session threading (native mode)
-------------------------------
NeMo Gym's model servers (VLLMModel, SimpleModelServer) overwrite
the ``model`` field with their own config value.  The ``user`` field
passes through all layers untouched:

    NemoGymProgram sets task["responses_create_params"]["user"] = "axon:{session_id}"
    → Agent Server /run (preserves user)
    → Model Server /v1/chat/completions (preserves user, overwrites model)
    → Axon /v1/chat/completions (reads user, extracts session_id)
    → engine.generate(messages, session_id=session_id)
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import uuid
from time import time

import httpx
import yaml

from axon.engine.server.oai_router import SESSION_USER_PREFIX
from axon.programs.base_program import BaseProgram, ProgramResult, register_program
from axon.tools.executors import HTTPToolExecutor
from axon.tools.parsers.base_parser import ToolCallParser, get_tool_call_parser
from axon.tools.types import ToolCall, ToolResult

logger = logging.getLogger(__name__)

# NeMo Gym head server default port.
DEFAULT_HEAD_SERVER_PORT = 11000

# Top-level keys in NeMo Gym's global config that contain servers.
_SERVER_TYPE_KEYS = ("resources_servers", "responses_api_agents", "responses_api_models")


# ═══════════════════════════════════════════════════════════════════════════════
# Server Autodiscovery
# ═══════════════════════════════════════════════════════════════════════════════


async def discover_server_url(
    head_server_url: str,
    server_name: str,
    http: httpx.AsyncClient | None = None,
) -> str:
    """Discover a NeMo Gym server's actual URL from the head server.

    NeMo Gym's head server (port 11000) serves
    ``GET /global_config_dict_yaml`` which contains host:port for every
    server.  Individual servers get dynamically assigned ports.

    Parameters
    ----------
    head_server_url : str
        URL of the head server, e.g. ``http://localhost:11000``.
    server_name : str
        Name of the server to find, e.g. ``workplace_assistant``,
        ``simple_agent``.
    http : httpx.AsyncClient, optional
        Reuse an existing client.

    Returns
    -------
    str
        The server's base URL, e.g. ``http://0.0.0.0:52341``.

    Raises
    ------
    ValueError
        If the server is not found in the config.
    """
    close_after = http is None
    if http is None:
        http = httpx.AsyncClient(timeout=10)
    try:
        resp = await http.get(f"{head_server_url.rstrip('/')}/global_config_dict_yaml")
        resp.raise_for_status()
        # FastAPI returns the YAML string as a JSON-encoded string,
        # so we need json.loads first to unwrap the JSON quoting,
        # then yaml.safe_load to parse the YAML content.
        raw = resp.text
        try:
            raw = json.loads(raw)  # unwrap JSON string
        except (json.JSONDecodeError, TypeError):
            pass  # already plain text
        config = yaml.safe_load(raw)
    finally:
        if close_after:
            await http.aclose()

    # Walk the config looking for server_name.
    #
    # NeMo Gym's config nests servers under instance-named top-level
    # keys, NOT directly under type keys:
    #
    #   workplace_assistant:              # instance (top-level)
    #     resources_servers:              # type
    #       workplace_assistant:          # server name
    #         host: 127.0.0.1
    #         port: 60235
    #
    # We search:  config[*][type_key][server_name]
    # Fallback:   config[type_key][server_name]  (flat layout)

    available = []
    for top_key, top_val in config.items():
        if not isinstance(top_val, dict):
            continue
        for type_key in _SERVER_TYPE_KEYS:
            servers = top_val.get(type_key, {})
            if not isinstance(servers, dict):
                continue
            for sname, srv in servers.items():
                available.append(f"{type_key}/{sname}")
                if sname == server_name and isinstance(srv, dict):
                    host = srv.get("host", "0.0.0.0")  # nosec B104
                    port = srv.get("port")
                    if port:
                        url = f"http://{host}:{port}"
                        logger.info(
                            "Discovered %s/%s at %s (under '%s')",
                            type_key,
                            server_name,
                            url,
                            top_key,
                        )
                        return url

    # Fallback: flat layout (config[type_key][server_name])
    for type_key in _SERVER_TYPE_KEYS:
        servers = config.get(type_key, {})
        if isinstance(servers, dict) and server_name in servers:
            srv = servers[server_name]
            if isinstance(srv, dict):
                host = srv.get("host", "0.0.0.0")  # nosec B104
                port = srv.get("port")
                if port:
                    url = f"http://{host}:{port}"
                    logger.info(
                        "Discovered %s/%s at %s (flat layout)",
                        type_key,
                        server_name,
                        url,
                    )
                    return url
            available.append(f"{type_key}/{server_name}")
    raise ValueError(f"Server '{server_name}' not found in NeMo Gym config. Available servers: {available}")


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _extract_text_content(content: str | list | dict) -> str:
    """Normalise Responses API ``content`` to a plain string.

    ``NeMoGymEasyInputMessage.content`` is
    ``Union[str, List[ResponseInputTextParam | ...]]``.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(item.get("text", ""))
        return "\n".join(parts)
    return str(content)


def _clean_tool_params(params: dict) -> dict:
    """Strip null-valued properties from NeMo Gym tool parameter schemas.

    NeMo Gym uses a union of ALL parameters across ALL tools in the
    environment, with irrelevant params set to ``null``.  For example,
    ``company_directory_find_email_address`` only needs ``name`` but
    the schema includes ~40 other params like ``board``, ``visitor_id``,
    etc. all set to ``null``.

    This is fine for structured tool-calling APIs but catastrophic when
    injected into the prompt — the model can't distinguish relevant
    params from noise.  We strip nulls and keep only params that have
    actual type definitions.
    """
    if not isinstance(params, dict):
        return params

    cleaned = dict(params)
    props = params.get("properties", {})
    if props:
        # Keep only properties that have a real definition (not null)
        cleaned_props = {}
        for name, schema in props.items():
            if schema is not None:
                cleaned_props[name] = schema
        cleaned["properties"] = cleaned_props

        # Update required to only include props that still exist
        if "required" in cleaned:
            cleaned["required"] = [r for r in cleaned["required"] if r in cleaned_props]

    return cleaned


def _clean_tools(tools: list[dict]) -> list[dict]:
    """Clean null parameters from NeMo Gym's union tool schemas."""
    cleaned = []
    for tool in tools:
        tool = dict(tool)
        fn = tool.get("function", tool)
        if "parameters" in fn:
            fn = dict(fn)
            fn["parameters"] = _clean_tool_params(fn["parameters"])
            if "function" in tool:
                tool["function"] = fn
            else:
                tool = fn
        cleaned.append(tool)
    return cleaned


def nemo_gym_task_to_messages(
    task: dict,
) -> tuple[list[dict[str, str]], list[dict]]:
    """Convert a NeMo Gym task to ``(messages, tools)`` for chat completions.

    NeMo Gym datasets use the **Responses API** schema
    (``NeMoGymResponseCreateParamsNonStreaming``).  In native mode the
    ``responses_create_params`` is sent directly to ``/v1/responses``
    unchanged.  In builtin mode we need to convert to chat-completion
    messages for Axon's generation engine.

    Conversion rules (Responses API → Chat Completions):

    1. ``instructions``        → first ``system`` message
    2. ``input[].role=developer`` → ``system``
    3. ``input[].content``     → flattened to string
    4. ``input`` as string     → single ``user`` message
    5. ``tools``               → passed through (same format)

    Returns
    -------
    messages : list[dict]
        Chat-completion-style messages.
    tools : list[dict]
        Tool definitions from ``responses_create_params``.
    """
    params = task.get("responses_create_params", task)
    raw_input = params.get("input", [])
    tools = params.get("tools", [])

    messages: list[dict[str, str]] = []

    # ``instructions`` is the Responses API equivalent of a system prompt.
    instructions = params.get("instructions")
    if instructions:
        messages.append({"role": "system", "content": instructions})

    # Handle string input (single prompt, valid per Responses API spec)
    if isinstance(raw_input, str):
        messages.append({"role": "user", "content": raw_input})
        return messages, tools

    for item in raw_input:
        if isinstance(item, str):
            messages.append({"role": "user", "content": item})
            continue

        item_type = item.get("type", "message")

        if item_type == "message":
            role = item.get("role", "user")
            if role == "developer":
                role = "system"
            content = _extract_text_content(item.get("content", ""))
            messages.append({"role": role, "content": content})

        elif item_type == "function_call":
            # Prior tool call in the conversation history — include as
            # assistant message with tool_calls for models that support it.
            # For prompt-injection mode we skip these (they'll be in
            # the raw text already).
            pass

        elif item_type == "function_call_output":
            # Prior tool result — skip for same reason as above.
            pass

        # Other types (reasoning, etc.) are silently skipped.

    return messages, tools


def build_nemo_gym_response(
    output_items: list[dict],
    task: dict,
    tools: list[dict] | None = None,
) -> dict:
    """Build a ``NeMoGymResponse``-compatible dict from sampler output.

    ``NeMoGymResponse`` extends OpenAI's ``Response`` model.  We populate
    all required fields and carry through ``tools``, ``tool_choice``, and
    ``parallel_tool_calls`` from the task's ``responses_create_params``
    so that the verify endpoint has the full context.

    Parameters
    ----------
    output_items : list[dict]
        The agent loop output in NeMo Gym Responses API format.
    task : dict
        The original task dict (for responses_create_params).
    tools : list[dict], optional
        Tool definitions from ``/seed_session``.  Falls back to
        ``responses_create_params.tools`` if not provided.
    """
    rcp = task.get("responses_create_params", {})
    return {
        # Required by Response
        "id": f"resp_{uuid.uuid4().hex}",
        "created_at": int(time()),
        "model": rcp.get("model", "axon"),
        "object": "response",
        "status": "completed",
        "output": output_items,
        # Required by Response (no defaults)
        "parallel_tool_calls": rcp.get("parallel_tool_calls", True),
        "tool_choice": rcp.get("tool_choice", "auto"),
        "tools": tools if tools is not None else rcp.get("tools", []),
        # Optional fields expected by Response
        "error": None,
        "incomplete_details": None,
        "instructions": rcp.get("instructions"),
        "metadata": rcp.get("metadata", {}),
        "temperature": rcp.get("temperature", 1.0),
        "top_p": rcp.get("top_p", 1.0),
        "max_output_tokens": rcp.get("max_output_tokens"),
        "previous_response_id": None,
        "reasoning": rcp.get("reasoning"),
        "text": rcp.get("text", {"format": {"type": "text"}}),
        "truncation": rcp.get("truncation", "disabled"),
        "usage": None,
        "user": rcp.get("user"),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# NemoGymProgram
# ═══════════════════════════════════════════════════════════════════════════════


@register_program("nemo_gym")
class NemoGymProgram(BaseProgram):
    """
    Axon program for NeMo Gym environments.

    Parameters
    ----------
    head_server_url : str
        URL of the NeMo Gym head server (default ``http://localhost:11000``).
        Used to discover actual server ports via ``/global_config_dict_yaml``.
    resource_server_name : str
        Name of the resource server in NeMo Gym config, e.g.
        ``workplace_assistant``, ``calendar``.  Used for autodiscovery.
    resource_server_url : str
        Direct URL override — skips autodiscovery if set.
    agent_server_name : str
        Name of the agent server (native mode), e.g. ``simple_agent``.
    agent_server_url : str
        Direct URL override for the agent server.
    mode : ``"builtin"`` | ``"native"``
        ``builtin`` — Axon runs the tool-calling loop.
        ``native``  — NeMo Gym's agent server runs the loop.
    tool_call_format : str
        Parser name (builtin mode only): ``qwen``, ``json``.
    max_steps : int
        Max tool-calling turns (builtin mode only).
    """

    # Class-level cache: discovery results shared across all instances.
    # Ports don't change during a training run, so we resolve once.
    _discovery_cache: dict[str, str] = {}  # server_name -> url
    _discovery_done: bool = False
    _discovery_lock: asyncio.Lock | None = None  # created lazily (needs event loop)

    @classmethod
    def _get_discovery_lock(cls) -> asyncio.Lock:
        """Lazily create the lock (must be called inside a running event loop)."""
        if cls._discovery_lock is None:
            cls._discovery_lock = asyncio.Lock()
        return cls._discovery_lock

    def __init__(
        self,
        # ---- NeMo Gym ----
        env_args: dict,
        mode: str = "builtin",
        # Server discovery
        head_server_url: str = "http://localhost:11000",
        resource_server_name: str = "",
        resource_server_url: str = "",
        agent_server_name: str = "simple_agent",
        agent_server_url: str = "",
        # builtin mode
        tool_call_format: str = "qwen",
        max_steps: int = 10,
        # native mode
        native_timeout: int = 300,
        # ---- BaseProgram ----
        group_id: str = "",
        sample_params: dict | None = None,
        endpoint_url: str = "",
        retry_limit: int = 1,
        program_timeout: int = 10800,
    ):
        super().__init__(
            group_id=group_id,
            sample_params=sample_params,
            endpoint_url=endpoint_url,
            retry_limit=retry_limit,
            program_timeout=program_timeout,
        )
        if mode not in ("builtin", "native"):
            raise ValueError(f"mode must be 'builtin' or 'native', got '{mode}'")

        self.mode = mode
        self.task = env_args["task"]
        self.max_steps = max_steps
        self.native_timeout = native_timeout

        # Server URLs — may be resolved lazily via autodiscovery
        self.head_server_url = head_server_url.rstrip("/")
        self.resource_server_name = resource_server_name
        self._resource_server_url = resource_server_url.rstrip("/") if resource_server_url else ""
        self.agent_server_name = agent_server_name
        self._agent_server_url = agent_server_url.rstrip("/") if agent_server_url else ""

        # Builtin: tool-call parser
        self.parser: ToolCallParser | None = get_tool_call_parser(tool_call_format) if mode == "builtin" else None

        # Validation: need either a direct URL or a name for discovery
        if mode == "builtin" and not resource_server_url and not resource_server_name:
            raise ValueError(
                "builtin mode requires resource_server_url or resource_server_name (for autodiscovery from head server)"
            )
        if mode == "native" and not agent_server_url and not agent_server_name:
            raise ValueError(
                "native mode requires agent_server_url or agent_server_name (for autodiscovery from head server)"
            )

    # ------------------------------------------------------------------
    # Server autodiscovery (class-level cache)
    # ------------------------------------------------------------------

    @property
    def resource_server_url(self) -> str:
        """Resolved resource server URL (from cache, explicit, or not yet resolved)."""
        if self._resource_server_url:
            return self._resource_server_url
        return self._discovery_cache.get(self.resource_server_name, "")

    @property
    def agent_server_url(self) -> str:
        """Resolved agent server URL (from cache, explicit, or not yet resolved)."""
        if self._agent_server_url:
            return self._agent_server_url
        return self._discovery_cache.get(self.agent_server_name, "")

    async def _ensure_discovered(self) -> None:
        """Resolve server URLs from the head server once, then cache.

        NeMo Gym runs a head server (default port 11000) that coordinates
        sub-servers, each on a dynamically assigned port.  Discovery
        results are cached at the class level so only the first instance
        pays the HTTP cost.  An asyncio.Lock ensures concurrent instances
        don't stampede the head server.
        """
        # Fast path (no lock needed): already have what we need
        if self.resource_server_url and (self.mode != "native" or self.agent_server_url):
            return

        # Serialise discovery — only one coroutine hits the head server
        async with self._get_discovery_lock():
            # Re-check after acquiring lock (another coroutine may have finished)
            if self.resource_server_url and (self.mode != "native" or self.agent_server_url):
                return

            if not NemoGymProgram._discovery_done:
                names_to_find: list[str] = []
                if self.resource_server_name and not self._resource_server_url:
                    names_to_find.append(self.resource_server_name)
                if self.agent_server_name and not self._agent_server_url:
                    names_to_find.append(self.agent_server_name)

                for name in names_to_find:
                    if name in self._discovery_cache:
                        continue
                    try:
                        url = await discover_server_url(
                            self.head_server_url,
                            name,
                        )
                        NemoGymProgram._discovery_cache[name] = url
                        logger.info(
                            "Autodiscovered '%s' at %s (cached for all instances)",
                            name,
                            url,
                        )
                    except ValueError:
                        logger.warning("Could not discover server '%s'", name)

                NemoGymProgram._discovery_done = True

        # Validate we have what we need
        if self.mode == "builtin" and not self.resource_server_url:
            raise ValueError(f"Could not resolve resource server '{self.resource_server_name}' from head server")
        if self.mode == "native" and not self.agent_server_url:
            raise ValueError(f"Could not resolve agent server '{self.agent_server_name}' from head server")

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    async def run(self) -> ProgramResult:
        await self._ensure_discovered()
        if self.mode == "native":
            return await self._run_native()
        return await self._run_builtin()

    # ------------------------------------------------------------------
    # Builtin mode: we own the agent loop
    # ------------------------------------------------------------------

    async def _run_builtin(self) -> ProgramResult:
        """
        Simple tool-calling agent loop.

        Reimplements NeMo Gym's ``simple_agent`` pattern.  Covers
        single-step and multi-step tool-calling environments.

        Flow  (mirrors ``SimpleAgent.run()``):
        1. ``/seed_session`` — sets up environment state + cookies
        2. Generate → parse tool calls → execute on resource server → repeat
        3. ``/verify`` — compute reward

        Tools come from ``responses_create_params`` in the dataset.
        ``/seed_session`` initialises server-side state (and returns
        cookies for session tracking).

        Session tracking: one cookie jar per episode.  The resource
        server uses Starlette's ``SessionMiddleware`` (cookie-based).
        ``/seed_session`` sets the cookie; subsequent requests thread
        it automatically via the cookie jar.
        """
        assert self.parser is not None

        messages, tools = nemo_gym_task_to_messages(self.task)
        tools = _clean_tools(tools)
        parser_kwargs = {"tools": tools} if tools else {}

        # Cookie jar = NeMo Gym session management
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(None),
        ) as http:
            executor = HTTPToolExecutor(http, self.resource_server_url)
            # 1. Seed session — pass task data so the resource server
            #    sets up the right environment state.
            seed_resp = await http.post(
                f"{self.resource_server_url}/seed_session",
                json=self.task,
            )
            seed_resp.raise_for_status()

            output_items: list[dict] = []
            steps = 0
            done = False

            while not done and steps < self.max_steps:
                # 2. Generate via engine (session tracked automatically)
                response_text, stop_program, _ = await self.generate(
                    messages=messages,
                    sample_params=self.sample_params,
                    parser_kwargs=parser_kwargs,
                )
                if stop_program:
                    break

                # 3. Parse tool calls
                tool_calls, remaining_text = self.parser.parse(response_text)
                messages.append(
                    {"role": "assistant", "content": response_text}
                )  # use full text here to ensure 1 conversation thread
                output_items.append(
                    {
                        "type": "message",
                        "role": "assistant",
                        "id": f"msg_{uuid.uuid4().hex[:12]}",
                        "content": [{"type": "output_text", "text": response_text, "annotations": []}],
                        "status": "completed",
                    }
                )

                if not tool_calls:
                    done = True
                else:
                    # 4. Execute tools on resource server
                    results = await self._execute_tool_calls(executor, tool_calls, output_items)
                    messages.append(
                        {
                            "role": "tool",
                            "content": self.parser.format_tool_results(results),
                        }
                    )

                steps += 1

            # 5. Verify → get reward
            reward, verify_resp = await self._verify(http, output_items, tools)

        return ProgramResult(
            reward=reward,
            done=True,
            metadata={
                "mode": "builtin",
                "steps": steps,
                "resource_server": self.resource_server_name,
                "verify_response": verify_resp,
            },
        )

    async def _execute_tool_calls(
        self,
        executor: HTTPToolExecutor,
        tool_calls: list[ToolCall],
        output_items: list[dict],
    ) -> list[ToolResult]:
        """Execute tool calls and track output_items for verify."""
        results: list[ToolResult] = []
        for tc in tool_calls:
            # Record request (NeMo Gym Responses API format)
            output_items.append(
                {
                    "type": "function_call",
                    "name": tc.name,
                    "arguments": json.dumps(tc.arguments),
                    "call_id": tc.id,
                    "status": "completed",
                }
            )

            # Dispatch via framework executor
            result = await executor.execute(tc)
            results.append(result)

            # Record response
            output_items.append(
                {
                    "type": "function_call_output",
                    "call_id": tc.id,
                    "output": result.content,
                }
            )
        return results

    async def _verify(
        self,
        http: httpx.AsyncClient,
        output_items: list[dict],
        tools: list[dict] | None = None,
    ) -> tuple[float, dict]:
        """Call ``/verify`` on the resource server.

        Sends a ``BaseVerifyRequest``-compatible payload.  Environment-
        specific verify endpoints (e.g. workplace_assistant) extend
        ``BaseVerifyRequest`` with extra fields like ``ground_truth``,
        ``id``, ``category``, ``environment_name``.  These come from the
        original task data, so we pass through **all** task fields and
        then overlay the ``response`` we built from sampler output.
        """
        reward = 0.0
        resp_data: dict = {}
        try:
            nemo_response = build_nemo_gym_response(
                output_items,
                self.task,
                tools=tools,
            )

            # Start with all task fields (includes ground_truth, id,
            # category, environment_name, responses_create_params, etc.)
            payload = dict(self.task)
            # Overlay our constructed response
            payload["response"] = nemo_response

            resp = await http.post(
                f"{self.resource_server_url}/verify",
                json=payload,
            )
            if resp.status_code == 422:
                detail = resp.text
                logger.error(
                    "Verify schema rejected (session=%s, status=422): %s\nPayload keys: %s",
                    self.session_id,
                    detail,
                    list(payload.keys()),
                )
            else:
                resp.raise_for_status()
                resp_data = resp.json()
                reward = float(resp_data.get("reward", 0.0))
        except Exception as exc:
            logger.error(
                "Verify failed (session=%s): %s",
                self.session_id,
                exc,
            )
        return reward, resp_data

    # ------------------------------------------------------------------
    # Native mode: NeMo Gym's agent server owns the loop
    # ------------------------------------------------------------------

    async def _run_native(self) -> ProgramResult:
        """
        Delegate to NeMo Gym's agent server.

        The agent server calls Axon's ``/v1/chat/completions`` for
        generation.  Session_id is threaded via the ``user`` field
        (``axon:{session_id}``) because NeMo Gym's model servers
        overwrite the ``model`` field with their own config value.

        See ``openai_compat.py`` for the endpoint that receives
        these requests and routes them to the engine.

        This path is designed for NeMo Gym agent patterns that run
        through its agent server.  Axon keeps the model-call side
        session-aware through the OpenAI-compatible endpoint.
        """
        assert self.endpoint_url, "native mode requires endpoint_url (Axon's OpenAI-compat server)"

        # Deep copy to avoid mutating the original task
        task = copy.deepcopy(self.task)

        # Thread session_id via the ``user`` field.
        # This survives through: agent server → model server → Axon
        # because nobody overwrites ``user`` (unlike ``model``).
        task.setdefault("responses_create_params", {})
        task["responses_create_params"]["user"] = f"{SESSION_USER_PREFIX}{self.session_id}"
        rcp = task.get("responses_create_params", {})
        if rcp.get("tools"):
            rcp["tools"] = _clean_tools(rcp["tools"])

        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(self.native_timeout),
            ) as http:
                # Agent server's /run accepts SimpleAgentRunRequest
                # (BaseRunRequest + extra="allow").  It forwards the
                # full body to /seed_session and /verify, so we must
                # include ALL task fields (ground_truth, id, category,
                # environment_name, etc.) — not just responses_create_params.
                resp = await http.post(
                    f"{self.agent_server_url}/run",
                    json=task,
                )
                if resp.status_code >= 400:
                    logger.error(
                        "Native /run returned %s (session=%s): %s",
                        resp.status_code,
                        self.session_id,
                        resp.text[:500],
                    )
                resp.raise_for_status()
                result = resp.json()
        except httpx.TimeoutException:
            logger.warning("Native sampler timed out (session=%s)", self.session_id)
            return ProgramResult(
                reward=-1e99,
                done=True,
                metadata={"mode": "native", "error": "timeout"},
            )
        except Exception as exc:
            logger.error("Native sampler failed (session=%s): %s", self.session_id, exc)
            return ProgramResult(
                reward=-1e99,
                done=True,
                metadata={"mode": "native", "error": str(exc)},
            )

        reward = float(result.get("reward", 0.0))

        return ProgramResult(
            reward=reward,
            done=True,
            metadata={
                "mode": "native",
                "steps": result.get("num_steps", -1),
                "resource_server": self.resource_server_name,
                "agent_result": result,
            },
        )
