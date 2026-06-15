"""Tests for axon.utils.openai_client module."""

import asyncio
import base64
from unittest import mock

import numpy as np
import pytest
from PIL import Image

from axon.utils.openai_client import (
    _call_api,
    _qwen2_5_vl_dedup_image_tokens,
    fetch_responses_from_addresses,
    pil_to_b64_png,
    poll_completions_openai,
    serialize_images_for_http,
)


# ---------------------------------------------------------------------------
# pil_to_b64_png
# ---------------------------------------------------------------------------
class TestPilToB64Png:
    def test_png_magic_bytes(self):
        decoded = base64.b64decode(pil_to_b64_png(Image.new("RGB", (4, 4))))
        assert decoded[:4] == b"\x89PNG"

    def test_rgba_image(self):
        decoded = base64.b64decode(pil_to_b64_png(Image.new("RGBA", (8, 8), (255, 0, 0, 128))))
        assert decoded[:4] == b"\x89PNG"

    def test_large_image_produces_large_output(self):
        result = pil_to_b64_png(Image.new("RGB", (1000, 1000)))
        assert len(base64.b64decode(result)) > 1000


# ---------------------------------------------------------------------------
# serialize_images_for_http
# ---------------------------------------------------------------------------
class TestSerializeImagesForHttp:
    def test_output_structure_exact_keys(self):
        result = serialize_images_for_http([Image.new("RGB", (3, 3))])
        assert len(result) == 1
        assert result[0] == {
            "type": "image_base64",
            "mime_type": "image/png",
            "data": result[0]["data"],
        }
        base64.b64decode(result[0]["data"])

    def test_unsupported_type_raises(self):
        with pytest.raises(TypeError, match="Unsupported image payload type"):
            serialize_images_for_http(["not_an_image"])

    def test_mixed_types_second_item_raises(self):
        with pytest.raises(TypeError):
            serialize_images_for_http([Image.new("RGB", (3, 3)), 42])

    def test_numpy_array_raises(self):
        with pytest.raises(TypeError, match="Unsupported"):
            serialize_images_for_http([np.zeros((3, 3, 3), dtype=np.uint8)])

    def test_empty_list(self):
        assert serialize_images_for_http([]) == []


# ---------------------------------------------------------------------------
# _qwen2_5_vl_dedup_image_tokens
# ---------------------------------------------------------------------------
class _FakeQwenProcessor:
    def __init__(self, token_id=99):
        class _ImgProc:
            pass

        _ImgProc.__name__ = "Qwen2VLImageProcessor"
        self.image_processor = _ImgProc()
        self.image_token_id = token_id


class TestQwen25VlDedupImageTokens:
    def test_no_processor_passthrough(self):
        assert _qwen2_5_vl_dedup_image_tokens([1, 2, 3], processor=None) == [1, 2, 3]

    def test_non_qwen_processor_passthrough(self):
        class CLIPImageProcessor:
            pass

        class Proc:
            image_processor = CLIPImageProcessor()

        assert _qwen2_5_vl_dedup_image_tokens([1, 2, 3], processor=Proc()) == [1, 2, 3]

    def test_dedup_run_of_four(self):
        proc = _FakeQwenProcessor(99)
        assert _qwen2_5_vl_dedup_image_tokens([10, 99, 99, 99, 99, 11], processor=proc) == [10, 99, 11]

    def test_separated_tokens_preserved(self):
        proc = _FakeQwenProcessor(99)
        assert _qwen2_5_vl_dedup_image_tokens([10, 99, 11, 99, 12], processor=proc) == [10, 99, 11, 99, 12]

    def test_all_image_collapses_to_one(self):
        proc = _FakeQwenProcessor(50)
        assert _qwen2_5_vl_dedup_image_tokens([50, 50, 50, 50], processor=proc) == [50]

    def test_two_blocks_independently_deduped(self):
        proc = _FakeQwenProcessor(99)
        assert _qwen2_5_vl_dedup_image_tokens([99, 99, 99, 1, 99, 99], processor=proc) == [99, 1, 99]

    def test_empty_prompt(self):
        proc = _FakeQwenProcessor(99)
        assert _qwen2_5_vl_dedup_image_tokens([], processor=proc) == []

    def test_long_run_100_tokens(self):
        proc = _FakeQwenProcessor(7)
        assert _qwen2_5_vl_dedup_image_tokens([7] * 100, processor=proc) == [7]


# ---------------------------------------------------------------------------
# Helpers for async tests
# ---------------------------------------------------------------------------
def _make_httpx_mock(response_json, status_code=200):
    """Create an httpx AsyncClient mock. httpx response.json() is sync, not async."""
    mock_response = mock.Mock()
    mock_response.status_code = status_code
    mock_response.json.return_value = response_json
    mock_response.text = str(response_json)
    mock_response.raise_for_status = mock.Mock()

    mock_client = mock.AsyncMock()
    mock_client.__aenter__ = mock.AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = mock.AsyncMock(return_value=False)
    mock_client.post.return_value = mock_response
    return mock_client


def _run(coro):
    """Run an async coroutine, compatible with all Python 3.10+ versions."""
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# poll_completions_openai
# ---------------------------------------------------------------------------
class TestPollCompletionsOpenai:
    def test_successful_completion(self):
        response_json = {
            "choices": [{"text": "hello", "finish_reason": "stop"}],
            "usage": {"total_tokens": 10},
        }
        mock_client = _make_httpx_mock(response_json)
        with mock.patch("axon.utils.openai_client.httpx.AsyncClient", return_value=mock_client):
            result = _run(
                poll_completions_openai(
                    "127.0.0.1:8000",
                    model="test",
                    prompt="hi",
                    max_retries=1,
                )
            )
        assert result == response_json

    def test_non_200_retries_and_raises(self):
        mock_client = _make_httpx_mock({}, status_code=500)
        with (
            mock.patch("axon.utils.openai_client.httpx.AsyncClient", return_value=mock_client),
            mock.patch("axon.utils.openai_client.asyncio.sleep", new_callable=mock.AsyncMock),
        ):
            with pytest.raises(Exception, match="API request failed"):
                _run(
                    poll_completions_openai(
                        "127.0.0.1:8000",
                        model="t",
                        prompt="p",
                        max_retries=2,
                    )
                )

    def test_meta_info_and_extra_headers_stripped(self):
        response_json = {"choices": [{"text": "ok", "finish_reason": "stop"}]}
        mock_client = _make_httpx_mock(response_json)
        with mock.patch("axon.utils.openai_client.httpx.AsyncClient", return_value=mock_client):
            _run(
                poll_completions_openai(
                    "127.0.0.1:8000",
                    model="t",
                    prompt="p",
                    meta_info={"x": 1},
                    extra_headers={"y": 2},
                    max_retries=1,
                )
            )
            posted_json = mock_client.post.call_args[1]["json"]
            assert "meta_info" not in posted_json
            assert "extra_headers" not in posted_json

    def test_abort_finish_reason_retries_without_counting(self):
        """Aborted responses retry without incrementing the retry counter."""
        abort_resp = mock.Mock()
        abort_resp.status_code = 200
        abort_resp.json.return_value = {"choices": [{"text": "", "finish_reason": "abort"}]}

        ok_resp = mock.Mock()
        ok_resp.status_code = 200
        ok_resp.json.return_value = {"choices": [{"text": "done", "finish_reason": "stop"}]}

        mock_client = mock.AsyncMock()
        mock_client.__aenter__ = mock.AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = mock.AsyncMock(return_value=False)
        mock_client.post.side_effect = [abort_resp, ok_resp]

        with (
            mock.patch("axon.utils.openai_client.httpx.AsyncClient", return_value=mock_client),
            mock.patch("axon.utils.openai_client.asyncio.sleep", new_callable=mock.AsyncMock),
        ):
            result = _run(
                poll_completions_openai(
                    "127.0.0.1:8000",
                    model="t",
                    prompt="p",
                    max_retries=1,
                )
            )
        assert result["choices"][0]["finish_reason"] == "stop"


# ---------------------------------------------------------------------------
# _call_api
# ---------------------------------------------------------------------------
class TestCallApi:
    def test_success(self):
        mock_client = _make_httpx_mock({"status": "ok"})
        with mock.patch("axon.utils.openai_client.httpx.AsyncClient", return_value=mock_client):
            result = _run(_call_api("127.0.0.1:8000", "/test", {"key": "val"}))
        assert result["success"] is True
        assert result["address"] == "127.0.0.1:8000"
        assert result["result"] == {"status": "ok"}

    def test_failure_after_retries(self):
        mock_client = mock.AsyncMock()
        mock_client.__aenter__ = mock.AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = mock.AsyncMock(return_value=False)
        mock_client.post.side_effect = ConnectionError("refused")

        with (
            mock.patch("axon.utils.openai_client.httpx.AsyncClient", return_value=mock_client),
            mock.patch("axon.utils.openai_client.asyncio.sleep", new_callable=mock.AsyncMock),
        ):
            result = _run(_call_api("127.0.0.1:8000", "/test", {}, max_retries=2))
        assert result["success"] is False
        assert "ConnectionError" in result["error"]


# ---------------------------------------------------------------------------
# fetch_responses_from_addresses
# ---------------------------------------------------------------------------
class TestFetchResponses:
    def test_parallel_calls(self):
        mock_client = _make_httpx_mock({"ok": True})
        addresses = ["127.0.0.1:8000", "127.0.0.1:8001"]
        with mock.patch("axon.utils.openai_client.httpx.AsyncClient", return_value=mock_client):
            results = _run(fetch_responses_from_addresses(addresses, "/test", {"key": "val"}))
        assert len(results) == 2
        for r in results:
            assert r["success"] is True
            assert r["result"] == {"ok": True}
