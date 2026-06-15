"""Tests for axon.programs.proxy_program module."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from axon.programs.base_program import PROGRAM_CLASS_MAPPING, ProgramResult
from axon.programs.proxy_program import ProxyProgram

# ---------------------------------------------------------------------------
# Helper to build a ProxyProgram with sensible defaults
# ---------------------------------------------------------------------------


def _make_proxy_program(**overrides):
    """Create a ProxyProgram with sensible defaults, providing a mock engine."""
    defaults = dict(
        proxy_url="http://proxy.test/api",
        proxy_token="test-token",
        server_url="http://server.test",
        env_args={"orgId": "org-1", "key": "value"},
        group_id="group-1",
        sample_params={"temperature": 0.7},
        endpoint_url="http://endpoint.test",
        retry_limit=1,
        program_timeout=10800,
    )
    defaults.update(overrides)
    prog = ProxyProgram(**defaults)
    # Provide a mock engine so async engine calls don't fail
    prog.engine = MagicMock()
    prog.session_id = "session-test-abc"
    return prog


def _run(coro):
    """Run an async coroutine synchronously, matching the project convention."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestProxyProgramRegistry:
    def test_proxy_registered(self):
        cls = PROGRAM_CLASS_MAPPING["proxy"]
        assert cls is ProxyProgram


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestProxyProgramConstruction:
    def test_inherits_base_program_fields(self):
        prog = ProxyProgram(
            proxy_url="http://proxy.test",
            proxy_token="tok",
            server_url="http://server.test",
            group_id="grp-42",
            sample_params={"top_p": 0.9},
            retry_limit=5,
            program_timeout=600,
        )
        assert prog.group_id == "grp-42"
        assert prog.sample_params == {"top_p": 0.9}
        assert prog.retry_limit == 5
        assert prog.program_timeout == 600
        assert prog.session_id is None
        assert prog._http_client is None


# ---------------------------------------------------------------------------
# _prepare_reset_payload
# ---------------------------------------------------------------------------


class TestPrepareResetPayload:
    def test_preserves_env_args_keys(self):
        prog = _make_proxy_program(env_args={"orgId": "org-1", "custom": "val"})
        payload = prog._prepare_reset_payload()
        assert payload["orgId"] == "org-1"
        assert payload["custom"] == "val"

    def test_does_not_mutate_original_env_args(self):
        env_args = {"orgId": "org-1", "custom": "val"}
        prog = _make_proxy_program(env_args=env_args)
        prog._prepare_reset_payload()
        assert "callbackEndpoint" not in env_args
        assert "callbackAuthToken" not in env_args

    def test_empty_env_args(self):
        prog = _make_proxy_program(env_args={})
        payload = prog._prepare_reset_payload()
        assert payload == {
            "callbackEndpoint": "http://server.test",
            "callbackAuthToken": "test-token",
        }

    def test_env_args_with_callback_fields_are_overwritten(self):
        """If env_args already contains callbackEndpoint, the copy should overwrite it."""
        prog = _make_proxy_program(env_args={"callbackEndpoint": "old", "callbackAuthToken": "old-tok"})
        payload = prog._prepare_reset_payload()
        assert payload["callbackEndpoint"] == "http://server.test"
        assert payload["callbackAuthToken"] == "test-token"

    def test_multiple_calls_return_independent_payloads(self):
        prog = _make_proxy_program()
        p1 = prog._prepare_reset_payload()
        p2 = prog._prepare_reset_payload()
        assert p1 == p2
        assert p1 is not p2
        p1["extra"] = "modified"
        assert "extra" not in p2


# ---------------------------------------------------------------------------
# _extract_session_id
# ---------------------------------------------------------------------------


class TestExtractSessionId:
    def test_valid_response(self):
        prog = _make_proxy_program()
        mock_response = MagicMock()
        mock_response.json.return_value = {"sessionId": "sess-123"}
        assert prog._extract_session_id(mock_response) == "sess-123"

    def test_missing_session_id_raises(self):
        prog = _make_proxy_program()
        mock_response = MagicMock()
        mock_response.json.return_value = {"other": "data"}
        with pytest.raises(Exception, match="No sessionId in response"):
            prog._extract_session_id(mock_response)

    def test_empty_session_id_raises(self):
        """An empty string is falsy, so it should raise."""
        prog = _make_proxy_program()
        mock_response = MagicMock()
        mock_response.json.return_value = {"sessionId": ""}
        with pytest.raises(Exception, match="No sessionId in response"):
            prog._extract_session_id(mock_response)

    def test_none_session_id_raises(self):
        prog = _make_proxy_program()
        mock_response = MagicMock()
        mock_response.json.return_value = {"sessionId": None}
        with pytest.raises(Exception, match="No sessionId in response"):
            prog._extract_session_id(mock_response)

    def test_invalid_json_raises_value_error(self):
        prog = _make_proxy_program()
        mock_response = MagicMock()
        mock_response.json.side_effect = ValueError("No JSON")
        with pytest.raises(ValueError):
            prog._extract_session_id(mock_response)

    def test_json_decode_error_raises(self):
        prog = _make_proxy_program()
        mock_response = MagicMock()
        mock_response.json.side_effect = json.JSONDecodeError("err", "", 0)
        with pytest.raises((ValueError, json.JSONDecodeError)):
            prog._extract_session_id(mock_response)


# ---------------------------------------------------------------------------
# _start_program_with_retry
# ---------------------------------------------------------------------------


class TestStartProgramWithRetry:
    @patch("axon.programs.proxy_program.asyncio.sleep", new_callable=AsyncMock)
    @patch("axon.programs.proxy_program.asyncio.to_thread", new_callable=AsyncMock)
    def test_success_on_first_attempt(self, mock_to_thread, mock_sleep):
        prog = _make_proxy_program()
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_to_thread.return_value = mock_response

        payload = {"orgId": "org-1"}
        result = _run(prog._start_program_with_retry(payload))

        assert result is mock_response
        assert mock_to_thread.call_count == 1
        mock_response.raise_for_status.assert_called_once()

    @patch("axon.programs.proxy_program.asyncio.sleep", new_callable=AsyncMock)
    @patch("axon.programs.proxy_program.asyncio.to_thread", new_callable=AsyncMock)
    def test_retries_on_request_exception(self, mock_to_thread, mock_sleep):
        import requests as req_lib

        prog = _make_proxy_program()
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()

        # Fail twice, succeed on third attempt
        mock_to_thread.side_effect = [
            req_lib.RequestException("fail-1"),
            req_lib.RequestException("fail-2"),
            mock_response,
        ]

        payload = {"orgId": "org-1"}
        result = _run(prog._start_program_with_retry(payload))

        assert result is mock_response
        assert mock_to_thread.call_count == 3

    @patch("axon.programs.proxy_program.asyncio.sleep", new_callable=AsyncMock)
    @patch("axon.programs.proxy_program.asyncio.to_thread", new_callable=AsyncMock)
    def test_raises_after_all_retries_exhausted(self, mock_to_thread, mock_sleep):
        import requests as req_lib

        prog = _make_proxy_program()
        mock_to_thread.side_effect = req_lib.RequestException("persistent failure")

        payload = {"orgId": "org-1"}
        with pytest.raises(req_lib.RequestException, match="persistent failure"):
            _run(prog._start_program_with_retry(payload))

        # Internal retry limit is 3
        assert mock_to_thread.call_count == 3

    @patch("axon.programs.proxy_program.asyncio.sleep", new_callable=AsyncMock)
    @patch("axon.programs.proxy_program.asyncio.to_thread", new_callable=AsyncMock)
    def test_sends_correct_headers(self, mock_to_thread, mock_sleep):
        prog = _make_proxy_program(proxy_token="my-secret-token")
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_to_thread.return_value = mock_response

        _run(prog._start_program_with_retry({"orgId": "org-1"}))

        # asyncio.to_thread is called with (requests.post, url, headers=..., data=..., timeout=...)
        call_args, call_kwargs = mock_to_thread.call_args
        headers = call_kwargs.get("headers")
        assert headers["Content-Type"] == "application/json"
        assert headers["Authorization"] == "Bearer my-secret-token"

    @patch("axon.programs.proxy_program.asyncio.sleep", new_callable=AsyncMock)
    @patch("axon.programs.proxy_program.asyncio.to_thread", new_callable=AsyncMock)
    def test_sends_json_payload(self, mock_to_thread, mock_sleep):
        prog = _make_proxy_program()
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_to_thread.return_value = mock_response

        payload = {"orgId": "org-1", "callbackEndpoint": "http://server.test"}
        _run(prog._start_program_with_retry(payload))

        call_args, call_kwargs = mock_to_thread.call_args
        sent_data = call_kwargs.get("data")
        parsed = json.loads(sent_data)
        assert parsed["orgId"] == "org-1"
        assert parsed["callbackEndpoint"] == "http://server.test"

    @patch("axon.programs.proxy_program.asyncio.sleep", new_callable=AsyncMock)
    @patch("axon.programs.proxy_program.asyncio.to_thread", new_callable=AsyncMock)
    def test_posts_to_proxy_url(self, mock_to_thread, mock_sleep):
        prog = _make_proxy_program(proxy_url="http://custom-proxy.test/start")
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_to_thread.return_value = mock_response

        _run(prog._start_program_with_retry({"orgId": "org-1"}))

        call_args, call_kwargs = mock_to_thread.call_args
        # First positional arg after the function is the URL
        assert call_args[1] == "http://custom-proxy.test/start"

    @patch("axon.programs.proxy_program.asyncio.sleep", new_callable=AsyncMock)
    @patch("axon.programs.proxy_program.asyncio.to_thread", new_callable=AsyncMock)
    def test_raise_for_status_triggers_retry(self, mock_to_thread, mock_sleep):
        """raise_for_status is called on every response; an HTTPError counts as RequestException."""
        import requests as req_lib

        prog = _make_proxy_program()
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = req_lib.HTTPError("500 Server Error")
        mock_to_thread.return_value = mock_response

        with pytest.raises(req_lib.RequestException):
            _run(prog._start_program_with_retry({"orgId": "org-1"}))

    @patch("axon.programs.proxy_program.asyncio.sleep", new_callable=AsyncMock)
    @patch("axon.programs.proxy_program.asyncio.to_thread", new_callable=AsyncMock)
    def test_retry_count_is_exactly_three(self, mock_to_thread, mock_sleep):
        """Internal retry limit is hardcoded to 3, regardless of program's retry_limit."""
        import requests as req_lib

        prog = _make_proxy_program(retry_limit=10)  # program retry_limit is different
        mock_to_thread.side_effect = req_lib.RequestException("fail")
        with pytest.raises(req_lib.RequestException):
            _run(prog._start_program_with_retry({"orgId": "org-1"}))
        assert mock_to_thread.call_count == 3  # hardcoded, not retry_limit

    @patch("axon.programs.proxy_program.asyncio.sleep", new_callable=AsyncMock)
    @patch("axon.programs.proxy_program.asyncio.to_thread", new_callable=AsyncMock)
    def test_timeout_is_30_seconds(self, mock_to_thread, mock_sleep):
        """The POST request should use a 30-second timeout."""
        prog = _make_proxy_program()
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_to_thread.return_value = mock_response

        _run(prog._start_program_with_retry({"orgId": "org-1"}))

        call_args, call_kwargs = mock_to_thread.call_args
        assert call_kwargs.get("timeout") == 30


# ---------------------------------------------------------------------------
# _register_session_and_metadata
# ---------------------------------------------------------------------------


class TestRegisterSessionAndMetadata:
    def test_registers_external_session_id(self):
        prog = _make_proxy_program()
        prog.engine.register_external_session_id = AsyncMock()
        prog.engine.run_in_engine_loop_async = AsyncMock()

        payload = {"orgId": "org-1", "callbackEndpoint": "http://server.test"}
        _run(prog._register_session_and_metadata("user-sess-42", payload))

        prog.engine.register_external_session_id.assert_awaited_once_with("user-sess-42", "session-test-abc")

    def test_adds_metadata(self):
        prog = _make_proxy_program()
        prog.engine.register_external_session_id = AsyncMock()
        prog.engine.run_in_engine_loop_async = AsyncMock()

        payload = {"orgId": "org-1", "callbackEndpoint": "http://server.test"}
        _run(prog._register_session_and_metadata("user-sess-42", payload))

        prog.engine.run_in_engine_loop_async.assert_awaited_once()
        # Verify the argument passed to run_in_engine_loop_async was the result of
        # calling add_to_program_metadata with the expected arguments
        prog.engine.add_to_program_metadata.assert_called_once_with(
            session_id="session-test-abc",
            metadata_key="reset_payload",
            metadata_val=payload,
        )

    def test_uses_correct_session_id(self):
        prog = _make_proxy_program()
        prog.session_id = "custom-session-999"
        prog.engine.register_external_session_id = AsyncMock()
        prog.engine.run_in_engine_loop_async = AsyncMock()

        payload = {"orgId": "org-1"}
        _run(prog._register_session_and_metadata("ext-sess-42", payload))

        prog.engine.register_external_session_id.assert_awaited_once_with("ext-sess-42", "custom-session-999")
        prog.engine.add_to_program_metadata.assert_called_once_with(
            session_id="custom-session-999",
            metadata_key="reset_payload",
            metadata_val=payload,
        )


# ---------------------------------------------------------------------------
# _monitor_program_status
# ---------------------------------------------------------------------------


class TestMonitorProgramStatus:
    @patch("axon.programs.proxy_program.asyncio.sleep", new_callable=AsyncMock)
    def test_completes_immediately_when_done(self, mock_sleep):
        prog = _make_proxy_program()
        prog.engine.run_in_engine_loop_async = AsyncMock(return_value=True)

        _run(prog._monitor_program_status())

        # Initial sleep of 15 seconds, then check returns True immediately
        mock_sleep.assert_awaited_once_with(15)

    @patch("axon.programs.proxy_program.asyncio.sleep", new_callable=AsyncMock)
    def test_polls_until_done(self, mock_sleep):
        prog = _make_proxy_program()
        # Not done, not done, done
        prog.engine.run_in_engine_loop_async = AsyncMock(side_effect=[False, False, True])

        _run(prog._monitor_program_status())

        # 1 initial sleep + 2 polling sleeps = 3 total sleeps of 15 seconds
        assert mock_sleep.call_count == 3
        for call in mock_sleep.call_args_list:
            assert call[0][0] == 15

    @patch("axon.programs.proxy_program.asyncio.sleep", new_callable=AsyncMock)
    def test_calls_check_program_status_with_session_id(self, mock_sleep):
        prog = _make_proxy_program()
        prog.session_id = "my-session-123"
        prog.engine.run_in_engine_loop_async = AsyncMock(return_value=True)

        _run(prog._monitor_program_status())

        prog.engine.check_program_status.assert_called_with(session_id="my-session-123")

    @patch("axon.programs.proxy_program.asyncio.sleep", new_callable=AsyncMock)
    def test_single_poll_cycle(self, mock_sleep):
        """Program is not done on first check, done on second."""
        prog = _make_proxy_program()
        prog.engine.run_in_engine_loop_async = AsyncMock(side_effect=[False, True])

        _run(prog._monitor_program_status())

        # 1 initial sleep + 1 polling sleep = 2 total
        assert mock_sleep.call_count == 2
        assert prog.engine.run_in_engine_loop_async.call_count == 2


# ---------------------------------------------------------------------------
# run (full orchestration)
# ---------------------------------------------------------------------------


class TestRun:
    @patch("axon.programs.proxy_program.asyncio.sleep", new_callable=AsyncMock)
    @patch("axon.programs.proxy_program.asyncio.to_thread", new_callable=AsyncMock)
    def test_run_returns_program_result(self, mock_to_thread, mock_sleep):
        prog = _make_proxy_program()

        # Mock the POST response
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"sessionId": "ext-sess-1"}
        mock_to_thread.return_value = mock_response

        # Mock engine calls
        prog.engine.register_external_session_id = AsyncMock()
        prog.engine.run_in_engine_loop_async = AsyncMock(return_value=True)

        result = _run(prog.run())

        assert isinstance(result, ProgramResult)
        assert result.reward == 0
        assert result.done is True

    @patch("axon.programs.proxy_program.asyncio.sleep", new_callable=AsyncMock)
    @patch("axon.programs.proxy_program.asyncio.to_thread", new_callable=AsyncMock)
    def test_run_calls_steps_in_order(self, mock_to_thread, mock_sleep):
        """Verify that run() calls the methods in the expected order."""
        prog = _make_proxy_program()

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"sessionId": "ext-sess-1"}
        mock_to_thread.return_value = mock_response

        prog.engine.register_external_session_id = AsyncMock()
        prog.engine.run_in_engine_loop_async = AsyncMock(return_value=True)

        call_order = []

        original_prepare = prog._prepare_reset_payload
        original_extract = prog._extract_session_id

        def tracked_prepare():
            call_order.append("prepare")
            return original_prepare()

        def tracked_extract(resp):
            call_order.append("extract")
            return original_extract(resp)

        async def tracked_start(payload):
            call_order.append("start")
            return mock_response

        async def tracked_register(sid, payload):
            call_order.append("register")

        async def tracked_monitor():
            call_order.append("monitor")

        prog._prepare_reset_payload = tracked_prepare
        prog._start_program_with_retry = tracked_start
        prog._extract_session_id = tracked_extract
        prog._register_session_and_metadata = tracked_register
        prog._monitor_program_status = tracked_monitor

        _run(prog.run())

        assert call_order == ["prepare", "start", "extract", "register", "monitor"]

    @patch("axon.programs.proxy_program.asyncio.sleep", new_callable=AsyncMock)
    @patch("axon.programs.proxy_program.asyncio.to_thread", new_callable=AsyncMock)
    def test_run_passes_payload_through(self, mock_to_thread, mock_sleep):
        prog = _make_proxy_program(
            env_args={"orgId": "org-test", "taskId": "task-7"},
            server_url="http://callback.test",
            proxy_token="cb-token",
        )

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"sessionId": "ext-sess-1"}
        mock_to_thread.return_value = mock_response

        prog.engine.register_external_session_id = AsyncMock()
        prog.engine.run_in_engine_loop_async = AsyncMock(return_value=True)

        _run(prog.run())

        call_args, call_kwargs = mock_to_thread.call_args
        sent_data = call_kwargs.get("data")
        parsed = json.loads(sent_data)
        assert parsed["orgId"] == "org-test"
        assert parsed["taskId"] == "task-7"
        assert parsed["callbackEndpoint"] == "http://callback.test"
        assert parsed["callbackAuthToken"] == "cb-token"

    @patch("axon.programs.proxy_program.asyncio.sleep", new_callable=AsyncMock)
    @patch("axon.programs.proxy_program.asyncio.to_thread", new_callable=AsyncMock)
    def test_run_propagates_start_program_error(self, mock_to_thread, mock_sleep):
        import requests as req_lib

        prog = _make_proxy_program()
        mock_to_thread.side_effect = req_lib.RequestException("connection refused")

        with pytest.raises(req_lib.RequestException, match="connection refused"):
            _run(prog.run())

    @patch("axon.programs.proxy_program.asyncio.sleep", new_callable=AsyncMock)
    @patch("axon.programs.proxy_program.asyncio.to_thread", new_callable=AsyncMock)
    def test_run_propagates_extract_session_id_error(self, mock_to_thread, mock_sleep):
        prog = _make_proxy_program()

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"no_session": True}
        mock_to_thread.return_value = mock_response

        with pytest.raises(Exception, match="No sessionId in response"):
            _run(prog.run())

    @patch("axon.programs.proxy_program.asyncio.sleep", new_callable=AsyncMock)
    @patch("axon.programs.proxy_program.asyncio.to_thread", new_callable=AsyncMock)
    def test_run_result_metadata_defaults(self, mock_to_thread, mock_sleep):
        """The ProgramResult returned by run() should have default metadata and step_rewards."""
        prog = _make_proxy_program()

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"sessionId": "ext-sess-1"}
        mock_to_thread.return_value = mock_response

        prog.engine.register_external_session_id = AsyncMock()
        prog.engine.run_in_engine_loop_async = AsyncMock(return_value=True)

        result = _run(prog.run())

        assert result.metadata == {}
        assert result.step_rewards == {}

    @patch("axon.programs.proxy_program.asyncio.sleep", new_callable=AsyncMock)
    @patch("axon.programs.proxy_program.asyncio.to_thread", new_callable=AsyncMock)
    def test_run_with_empty_env_args(self, mock_to_thread, mock_sleep):
        """run() with empty env_args hits KeyError on orgId in register step."""
        prog = _make_proxy_program(env_args={})
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"sessionId": "s1"}
        mock_to_thread.return_value = mock_response
        prog.engine.register_external_session_id = AsyncMock()
        prog.engine.run_in_engine_loop_async = AsyncMock(return_value=True)
        # Production code references reset_payload['orgId'] which won't exist
        with pytest.raises(KeyError, match="orgId"):
            _run(prog.run())

    @patch("axon.programs.proxy_program.asyncio.sleep", new_callable=AsyncMock)
    @patch("axon.programs.proxy_program.asyncio.to_thread", new_callable=AsyncMock)
    def test_run_registers_correct_external_session_id(self, mock_to_thread, mock_sleep):
        """The session ID extracted from the response is passed to register_external_session_id."""
        prog = _make_proxy_program()
        prog.session_id = "internal-session-77"

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"sessionId": "external-sess-abc"}
        mock_to_thread.return_value = mock_response

        prog.engine.register_external_session_id = AsyncMock()
        prog.engine.run_in_engine_loop_async = AsyncMock(return_value=True)

        _run(prog.run())

        prog.engine.register_external_session_id.assert_awaited_once_with("external-sess-abc", "internal-session-77")
