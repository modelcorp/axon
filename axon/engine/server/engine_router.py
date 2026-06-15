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
Native Engine HTTP router.

Provides a small FastAPI surface for driving an :class:`Engine` over
HTTP — used when the rollout loop runs in a different process from the
sampler (e.g. an external agent server).

Endpoints
---------
- ``GET  /health``        — liveness probe.
- ``POST /init_session``  — start a new program session, returns ``session_id``.
- ``POST /generate``      — step one turn of generation for a session.
- ``POST /end_session``   — finalize a session with a terminal reward.

The OpenAI-compatible surface (``/v1/chat/completions``, ``/v1/models``,
``/tokenize``) lives in :mod:`axon.engine.server.oai_router`.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class InitSessionRequest(BaseModel):
    group_id: str | None = None
    sample_params: dict | None = None


class InitSessionResponse(BaseModel):
    session_id: str


class GenerateRequest(BaseModel):
    messages: list[dict[str, str]]
    session_id: str
    sample_params: dict | None = None


class GenerateResponse(BaseModel):
    response: str
    stop_program: bool
    step_idx: int = -1


class EndSessionRequest(BaseModel):
    session_id: str
    reward: float
    step_rewards: dict[str, float] | None = None


class EndSessionResponse(BaseModel):
    success: bool


# JSON does not support inf / -inf, so clients send these sentinels for
# rewards that should be treated as infinite. Anything past 1e98 wins.
_POS_INF_SENTINEL = 1e98
_NEG_INF_SENTINEL = -1e98


def build_engine_router(engine) -> APIRouter:
    """
    Build a FastAPI router with native Engine endpoints.

    Parameters
    ----------
    engine : Engine
        Must expose ``init_session``, ``generate``, ``end_session``, and
        ``run_in_engine_loop_async``.

    Returns
    -------
    APIRouter
        Mount with ``app.include_router(router)``.
    """
    router = APIRouter(tags=["agent-execution-engine"])

    @router.get("/health")
    async def health() -> JSONResponse:
        return JSONResponse(content={"status": "ok"})

    @router.post("/init_session", response_model=InitSessionResponse)
    async def api_init_session(request: InitSessionRequest):
        """Initialize a new session."""
        try:
            session_id = await engine.run_in_engine_loop_async(
                engine.init_session(request.group_id, request.sample_params)
            )
            return InitSessionResponse(session_id=session_id)
        except Exception as e:
            logger.error(f"Error in init_session: {e}")
            raise HTTPException(status_code=500, detail=str(e)) from e

    @router.post("/generate", response_model=GenerateResponse)
    async def api_generate(request: GenerateRequest):
        """Generate one turn of response for a session."""
        try:
            response, stop_program, step_idx = await engine.run_in_engine_loop_async(
                engine.generate(request.messages, request.session_id, request.sample_params)
            )
            return GenerateResponse(response=response, stop_program=stop_program, step_idx=step_idx)
        except Exception as e:
            logger.error(f"Error in generate: {e}")
            raise HTTPException(status_code=500, detail=str(e)) from e

    @router.post("/end_session", response_model=EndSessionResponse)
    async def api_end_session(request: EndSessionRequest):
        """End a session with a terminal reward."""
        try:
            reward = request.reward
            if reward <= _NEG_INF_SENTINEL:
                reward = float("-inf")
            elif reward >= _POS_INF_SENTINEL:
                reward = float("inf")

            # JSON keys are strings; the engine expects int step indices.
            step_rewards = {int(k): v for k, v in request.step_rewards.items()} if request.step_rewards else None

            await engine.run_in_engine_loop_async(
                engine.end_session(request.session_id, reward, step_rewards=step_rewards)
            )
            return EndSessionResponse(success=True)
        except Exception as e:
            logger.error(f"Error in end_session: {e}")
            raise HTTPException(status_code=500, detail=str(e)) from e

    return router
