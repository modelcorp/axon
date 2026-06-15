"""Tests for axon.engine.server.oai_router.

Focuses on session routing logic, OpenAI response schema compliance,
tool call serialization fidelity, and content/finish_reason semantics.
"""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from axon.engine.server.oai_router import build_openai_router


def _make_engine(sessions=None):
    engine = MagicMock()
    engine.session_state_map = sessions or {}

    async def fake_run(coro):
        return await coro

    engine.run_in_engine_loop_async = AsyncMock(side_effect=fake_run)
    engine.generate = AsyncMock(return_value=("Hello!", False, 0))
    engine.chat_parser = MagicMock()
    engine.chat_parser.tool_parser = None
    return engine


def _app(engine):
    app = FastAPI()
    app.include_router(build_openai_router(engine))
    return app


def _chat(client, user, messages=None, **extra):
    body = {"messages": messages or [], "user": user}
    body.update(extra)
    return client.post("/v1/chat/completions", json=body)


# =========================================================================
# Session routing — the critical user-field extraction logic
# =========================================================================


class TestSessionRouting:
    @pytest.mark.parametrize("user", ["", "bad_prefix", "openai:sess", "rll:sess"])
    def test_invalid_user_prefix_returns_400(self, user):
        resp = _chat(TestClient(_app(_make_engine())), user)
        assert resp.status_code == 400

    def test_missing_user_field_returns_400(self):
        resp = TestClient(_app(_make_engine())).post("/v1/chat/completions", json={"messages": []})
        assert resp.status_code == 400

    def test_unknown_session_returns_graceful_stop(self):
        """Missing session → 200 with empty content, finish_reason=length, no engine call."""
        engine = _make_engine(sessions={})
        resp = _chat(TestClient(_app(engine)), "axon:ghost")
        assert resp.status_code == 200
        assert resp.json()["choices"][0]["finish_reason"] == "length"
        assert resp.json()["choices"][0]["message"]["content"] == ""
        engine.generate.assert_not_called()

    def test_session_id_with_special_chars(self):
        sid = "sess-with.dots_and-dashes/slashes"
        engine = _make_engine(sessions={sid: {}})
        resp = _chat(TestClient(_app(engine)), f"axon:{sid}")
        assert resp.status_code == 200
        assert engine.generate.await_args[1]["session_id"] == sid

    def test_empty_session_id(self):
        engine = _make_engine(sessions={"": {}})
        resp = _chat(TestClient(_app(engine)), "axon:")
        assert resp.status_code == 200


# =========================================================================
# OpenAI response schema
# =========================================================================


class TestResponseSchema:
    def test_required_fields_present(self):
        engine = _make_engine(sessions={"s": {}})
        data = _chat(TestClient(_app(engine)), "axon:s", model="my-model").json()
        assert data["object"] == "chat.completion"
        assert data["id"].startswith("chatcmpl-")
        assert isinstance(data["created"], int)
        assert data["model"] == "my-model"
        assert len(data["choices"]) == 1
        assert "usage" in data

    def test_model_defaults_to_axon(self):
        engine = _make_engine(sessions={"s": {}})
        assert _chat(TestClient(_app(engine)), "axon:s").json()["model"] == "axon"

    def test_unique_ids(self):
        engine = _make_engine(sessions={"s": {}})
        client = TestClient(_app(engine))
        ids = {_chat(client, "axon:s").json()["id"] for _ in range(10)}
        assert len(ids) == 10

    def test_logprobs_content_present_for_nemogym_compat(self):
        """NeMo Gym's vllm_model accesses choice.logprobs.content — must exist."""
        engine = _make_engine(sessions={"s": {}})
        choice = _chat(TestClient(_app(engine)), "axon:s").json()["choices"][0]
        assert choice["logprobs"]["content"] == []


# =========================================================================
# Tool call serialization
# =========================================================================


def _make_tc(id, name, arguments):
    tc = MagicMock()
    tc.id, tc.name, tc.arguments = id, name, arguments
    return tc


class TestToolCalls:
    def _engine_with_tools(self, tool_calls, remaining=""):
        engine = _make_engine(sessions={"s": {}})
        engine.generate = AsyncMock(return_value=("raw", False, 0))
        tp = MagicMock()
        tp.parse.return_value = (tool_calls, remaining)
        engine.chat_parser.tool_parser = tp
        return engine

    def test_single_tool_call(self):
        tc = _make_tc("c1", "get_weather", {"city": "NYC"})
        engine = self._engine_with_tools([tc], "thinking")
        msg = _chat(TestClient(_app(engine)), "axon:s").json()["choices"][0]["message"]
        assert msg["content"] == "thinking"
        assert len(msg["tool_calls"]) == 1
        fc = msg["tool_calls"][0]
        assert fc["id"] == "c1"
        assert fc["type"] == "function"
        assert fc["function"]["name"] == "get_weather"
        assert json.loads(fc["function"]["arguments"]) == {"city": "NYC"}

    def test_multiple_tool_calls(self):
        tcs = [
            _make_tc("c1", "search", {"q": "news"}),
            _make_tc("c2", "calc", {"expr": "2+2"}),
        ]
        engine = self._engine_with_tools(tcs)
        msg = _chat(TestClient(_app(engine)), "axon:s").json()["choices"][0]["message"]
        assert len(msg["tool_calls"]) == 2
        assert msg["tool_calls"][0]["function"]["name"] == "search"
        assert msg["tool_calls"][1]["function"]["name"] == "calc"

    def test_arguments_are_json_string_not_dict(self):
        """OpenAI spec requires arguments as a JSON string."""
        tc = _make_tc("c1", "fn", {"key": "value"})
        engine = self._engine_with_tools([tc])
        msg = _chat(TestClient(_app(engine)), "axon:s").json()["choices"][0]["message"]
        raw_args = msg["tool_calls"][0]["function"]["arguments"]
        assert isinstance(raw_args, str)
        assert json.loads(raw_args) == {"key": "value"}

    def test_special_chars_in_arguments_roundtrip(self):
        tc = _make_tc(
            "c1",
            "write",
            {
                "text": 'He said "hello"\nNew line\t\u00e9\u00e0\u00fc',
                "code": "if x > 0:\n    return True",
            },
        )
        engine = self._engine_with_tools([tc])
        msg = _chat(TestClient(_app(engine)), "axon:s").json()["choices"][0]["message"]
        parsed = json.loads(msg["tool_calls"][0]["function"]["arguments"])
        assert parsed["text"] == tc.arguments["text"]
        assert parsed["code"] == tc.arguments["code"]

    def test_no_tool_parser_means_no_tool_calls_key(self):
        engine = _make_engine(sessions={"s": {}})
        engine.chat_parser.tool_parser = None
        msg = _chat(TestClient(_app(engine)), "axon:s").json()["choices"][0]["message"]
        assert "tool_calls" not in msg
        assert msg["content"] == "Hello!"

    def test_content_is_none_when_only_tool_calls(self):
        tc = _make_tc("c1", "fn", {})
        engine = self._engine_with_tools([tc], remaining="")
        msg = _chat(TestClient(_app(engine)), "axon:s").json()["choices"][0]["message"]
        # Empty remaining → content should be None per OpenAI convention
        assert msg["content"] is None or msg["content"] == ""


# =========================================================================
# finish_reason
# =========================================================================


class TestFinishReason:
    def test_stop_when_not_truncated(self):
        engine = _make_engine(sessions={"s": {}})
        engine.generate = AsyncMock(return_value=("done", False, 0))
        assert _chat(TestClient(_app(engine)), "axon:s").json()["choices"][0]["finish_reason"] == "stop"

    def test_length_when_truncated(self):
        engine = _make_engine(sessions={"s": {}})
        engine.generate = AsyncMock(return_value=("partial", True, 5))
        assert _chat(TestClient(_app(engine)), "axon:s").json()["choices"][0]["finish_reason"] == "length"


# =========================================================================
# Message/tool passthrough to engine
# =========================================================================


class TestPassthrough:
    def test_messages_forwarded(self):
        engine = _make_engine(sessions={"s": {}})
        msgs = [{"role": "system", "content": "Be helpful"}, {"role": "user", "content": "Hi"}]
        _chat(TestClient(_app(engine)), "axon:s", messages=msgs)
        assert engine.generate.await_args[1]["messages"] == msgs

    def test_tools_forwarded(self):
        engine = _make_engine(sessions={"s": {}})
        tools = [{"type": "function", "function": {"name": "search"}}]
        _chat(TestClient(_app(engine)), "axon:s", tools=tools)
        assert engine.generate.await_args[1]["tools_json"] == tools

    def test_engine_error_returns_500(self):
        engine = _make_engine(sessions={"s": {}})
        engine.run_in_engine_loop_async = AsyncMock(side_effect=RuntimeError("boom"))
        assert _chat(TestClient(_app(engine)), "axon:s").status_code == 500


# =========================================================================
# Multi-turn session
# =========================================================================


class TestMultiTurnSession:
    def test_three_turn_conversation(self):
        """Simulate user→assistant→user→assistant→user→assistant."""
        engine = _make_engine(sessions={"s": {}})
        turn = 0

        async def gen(messages, session_id, tools_json=None):
            nonlocal turn
            turn += 1
            stop = turn >= 3
            return (f"Reply {turn}", stop, turn - 1)

        engine.generate = AsyncMock(side_effect=gen)
        client = TestClient(_app(engine))

        history = []
        for i in range(3):
            history.append({"role": "user", "content": f"User msg {i}"})
            resp = _chat(client, "axon:s", messages=history)
            data = resp.json()
            assert data["choices"][0]["message"]["content"] == f"Reply {i + 1}"
            history.append({"role": "assistant", "content": data["choices"][0]["message"]["content"]})

        # Final turn should have stop
        assert data["choices"][0]["finish_reason"] == "length"
