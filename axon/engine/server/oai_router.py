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
OpenAI-compatible chat completions router for Engine.

Provides ``POST /v1/chat/completions``, ``GET /v1/models``, and
``POST /tokenize`` so that external systems speaking the OpenAI
protocol (e.g. NeMo Gym agent/model servers) can drive Axon sessions.

Session routing
---------------
The ``user`` field in the OpenAI request carries the session_id::

    user="axon:{session_id}"

Any standard OpenAI client works::

    client = OpenAI(base_url="http://engine:8080/v1", api_key="x")
    client.chat.completions.create(
        model="axon",
        messages=[...],
        extra_body={"user": f"axon:{session_id}"},
    )

NeMo Gym compatibility
----------------------
NeMo Gym's ``vllm_model`` wrapper sits between the agent server and
Axon.  It calls ``/v1/chat/completions`` with ``logprobs=True`` and
``return_tokens_as_token_ids=True``, then calls ``/tokenize`` for
prompt token IDs.  We handle both:

- ``logprobs`` in the response (empty list — Axon tracks internally)
- ``/tokenize`` endpoint (returns empty tokens — Axon tracks internally)
"""

from __future__ import annotations

import json
import logging
import time
import uuid

from fastapi import APIRouter, HTTPException, Request

logger = logging.getLogger(__name__)

SESSION_USER_PREFIX = "axon:"


def build_openai_router(engine) -> APIRouter:
    """
    Build a FastAPI router with OpenAI-compatible endpoints.

    Parameters
    ----------
    engine : Engine
        Must expose ``session_state_map``, ``generate()``, and
        ``run_in_engine_loop_async()``.

    Returns
    -------
    APIRouter
        Mount with ``app.include_router(router)``.
    """
    router = APIRouter(tags=["openai-compat"])

    @router.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        body = await request.json()

        # ---- Extract session_id from user field ----
        user = body.get("user", "")
        if not user.startswith(SESSION_USER_PREFIX):
            raise HTTPException(
                status_code=400,
                detail=f"Expected user='axon:{{session_id}}' for session routing. Got: '{user}'",
            )

        session_id = user[len(SESSION_USER_PREFIX) :]

        if session_id not in engine.session_state_map:
            logger.warning("Session '%s' not found — returning stop", session_id)
            return {
                "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": body.get("model", "axon"),
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": ""},
                        "finish_reason": "length",
                        "logprobs": {"content": []},
                    }
                ],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            }

        messages = body.get("messages", [])
        tools = body.get("tools", [])

        # ---- Generate through normal engine path ----
        try:
            response_text, stop_program, _step_idx = await engine.run_in_engine_loop_async(
                engine.generate(messages=messages, session_id=session_id, tools_json=tools)
            )
        except Exception as exc:
            logger.exception("Engine generate failed (session=%s): %s", session_id, exc)
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        # Parse tool calls from raw text
        tool_calls = []
        tool_parser = engine.chat_parser.tool_parser
        if tool_parser:
            tool_calls, response_text = tool_parser.parse(response_text)

        # Build response with structured tool_calls
        message = {"role": "assistant", "content": response_text or None}
        if tool_calls:
            message["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments),
                    },
                }
                for tc in tool_calls
            ]

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": body.get("model", "axon"),
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": "length" if stop_program else "stop",
                    # NeMo Gym's vllm_model wrapper accesses
                    # choice["logprobs"]["content"] for token-level
                    # log probabilities.  Axon tracks logprobs
                    # internally per session, so we return an empty
                    # list to satisfy the schema.
                    "logprobs": {
                        "content": [],
                    },
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        }

    @router.get("/v1/models")
    async def list_models():
        return {
            "object": "list",
            "data": [
                {
                    "id": "axon",
                    "object": "model",
                    "owned_by": "axon",
                }
            ],
        }

    # ── /tokenize ─────────────────────────────────────────────
    # NeMo Gym's vllm_model calls base_url.removesuffix("/v1")/tokenize
    # when return_token_id_information=true.  Axon tracks token info
    # internally per session, so we return empty tokens.

    @router.post("/tokenize")
    async def tokenize(request: Request):
        return {
            "tokens": [],
            "count": 0,
        }

    return router
