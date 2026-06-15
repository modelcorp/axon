"""Tests for axon.engine.server.engine_router.

Tests non-obvious behavior: sentinel conversion, step_rewards key coercion,
error propagation, multi-step session flow, and boundary conditions.
"""

import math
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from axon.engine.server.engine_router import build_engine_router


def _make_engine():
    engine = MagicMock()

    async def fake_run(coro):
        return await coro

    engine.run_in_engine_loop_async = AsyncMock(side_effect=fake_run)
    engine.init_session = AsyncMock(return_value="sess-0")
    engine.generate = AsyncMock(return_value=("response text", False, 0))
    engine.end_session = AsyncMock()
    return engine


def _make_app(engine) -> FastAPI:
    app = FastAPI()
    app.include_router(build_engine_router(engine))
    return app


def _end_session_reward(engine):
    """Extract the reward value actually passed to engine.end_session."""
    return engine.end_session.await_args[0][1]


def _end_session_step_rewards(engine):
    """Extract step_rewards kwarg passed to engine.end_session."""
    return engine.end_session.await_args[1]["step_rewards"]


# =========================================================================
# Sentinel value conversion (the non-obvious part of end_session)
# =========================================================================


class TestSentinelConversion:
    @pytest.mark.parametrize(
        "json_val,check",
        [
            (-1e99, lambda r: math.isinf(r) and r < 0),
            (-2e99, lambda r: math.isinf(r) and r < 0),
            (1e99, lambda r: math.isinf(r) and r > 0),
            (1e100, lambda r: math.isinf(r) and r > 0),
        ],
    )
    def test_sentinel_values_become_inf(self, json_val, check):
        engine = _make_engine()
        TestClient(_make_app(engine)).post("/end_session", json={"session_id": "s", "reward": json_val})
        assert check(_end_session_reward(engine))

    @pytest.mark.parametrize("json_val", [0.0, -0.5, 0.75, 1.0, -1.0, 1e97, -1e97])
    def test_non_sentinel_rewards_pass_through(self, json_val):
        engine = _make_engine()
        TestClient(_make_app(engine)).post("/end_session", json={"session_id": "s", "reward": json_val})
        assert _end_session_reward(engine) == pytest.approx(json_val)


# =========================================================================
# step_rewards key coercion
# =========================================================================


class TestStepRewardsCoercion:
    def test_string_keys_to_int(self):
        engine = _make_engine()
        TestClient(_make_app(engine)).post(
            "/end_session",
            json={"session_id": "s", "reward": 1.0, "step_rewards": {"0": 0.2, "5": 0.8}},
        )
        sr = _end_session_step_rewards(engine)
        assert sr == {0: 0.2, 5: 0.8}
        assert all(isinstance(k, int) for k in sr)

    def test_null_step_rewards(self):
        engine = _make_engine()
        TestClient(_make_app(engine)).post("/end_session", json={"session_id": "s", "reward": 1.0})
        assert _end_session_step_rewards(engine) is None

    def test_empty_dict_is_falsy_becomes_none(self):
        engine = _make_engine()
        TestClient(_make_app(engine)).post(
            "/end_session",
            json={"session_id": "s", "reward": 1.0, "step_rewards": {}},
        )
        assert _end_session_step_rewards(engine) is None

    def test_many_steps(self):
        engine = _make_engine()
        step_rewards = {str(i): float(i) / 10 for i in range(20)}
        TestClient(_make_app(engine)).post(
            "/end_session",
            json={"session_id": "s", "reward": 1.0, "step_rewards": step_rewards},
        )
        sr = _end_session_step_rewards(engine)
        assert len(sr) == 20
        assert sr[0] == 0.0
        assert sr[19] == pytest.approx(1.9)


# =========================================================================
# Error propagation — each endpoint
# =========================================================================


class TestErrorPropagation:
    @pytest.mark.parametrize(
        "endpoint,payload",
        [
            ("/init_session", {}),
            ("/generate", {"messages": [{"role": "user", "content": "hi"}], "session_id": "s"}),
            ("/end_session", {"session_id": "s", "reward": 0.0}),
        ],
    )
    def test_engine_error_returns_500(self, endpoint, payload):
        engine = _make_engine()
        engine.run_in_engine_loop_async = AsyncMock(side_effect=RuntimeError("boom"))
        resp = TestClient(_make_app(engine)).post(endpoint, json=payload)
        assert resp.status_code == 500

    def test_error_detail_preserved(self):
        engine = _make_engine()
        engine.run_in_engine_loop_async = AsyncMock(side_effect=RuntimeError("CUDA OOM"))
        resp = TestClient(_make_app(engine)).post("/init_session", json={})
        assert "CUDA OOM" in resp.json()["detail"]


# =========================================================================
# Multi-step session lifecycle
# =========================================================================


class TestSessionLifecycle:
    def test_init_generate_generate_end(self):
        call_count = 0
        engine = _make_engine()

        async def gen(messages, session_id, sample_params=None):
            nonlocal call_count
            call_count += 1
            return (f"Response {call_count}", call_count >= 3, call_count - 1)

        engine.generate = AsyncMock(side_effect=gen)
        client = TestClient(_make_app(engine))

        r = client.post("/init_session", json={"group_id": "g"})
        assert r.status_code == 200
        sid = r.json()["session_id"]

        for i in range(3):
            r = client.post(
                "/generate",
                json={"messages": [{"role": "user", "content": f"turn {i}"}], "session_id": sid},
            )
            assert r.status_code == 200
            data = r.json()
            assert data["response"] == f"Response {i + 1}"
            assert data["step_idx"] == i
            if i < 2:
                assert data["stop_program"] is False
            else:
                assert data["stop_program"] is True

        r = client.post(
            "/end_session",
            json={"session_id": sid, "reward": 1.0, "step_rewards": {"0": 0.1, "1": 0.3, "2": 0.6}},
        )
        assert r.status_code == 200
        assert r.json()["success"] is True

        sr = _end_session_step_rewards(engine)
        assert sr == {0: 0.1, 1: 0.3, 2: 0.6}

    def test_sample_params_forwarded(self):
        engine = _make_engine()
        TestClient(_make_app(engine)).post(
            "/generate",
            json={
                "messages": [],
                "session_id": "s",
                "sample_params": {"temperature": 0.1, "top_k": 50},
            },
        )
        assert engine.generate.await_args[0][2] == {"temperature": 0.1, "top_k": 50}


# =========================================================================
# Pydantic validation
# =========================================================================


class TestValidation:
    def test_missing_session_id_422(self):
        resp = TestClient(_make_app(_make_engine())).post("/generate", json={"messages": []})
        assert resp.status_code == 422

    def test_missing_reward_422(self):
        resp = TestClient(_make_app(_make_engine())).post("/end_session", json={"session_id": "s"})
        assert resp.status_code == 422

    def test_health_always_works(self):
        resp = TestClient(_make_app(MagicMock())).get("/health")
        assert resp.json() == {"status": "ok"}
