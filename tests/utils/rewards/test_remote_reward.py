"""Tests for axon.utils.rewards.remote_reward module."""

import os
from unittest import mock

import pytest
import requests

from axon.utils.rewards.remote_reward import remote_reward_fn


def _mock_response(json_data):
    """Create a mock response object."""
    resp = mock.Mock()
    resp.json.return_value = json_data
    resp.raise_for_status = mock.Mock()
    return resp


class TestRemoteRewardFn:
    def test_no_url_returns_zero(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            result = remote_reward_fn(task_info={}, action="test")
            assert result.reward == 0.0
            assert result.is_correct is False

    def test_task_info_url_overrides_env(self):
        with (
            mock.patch.dict(os.environ, {"AXON_REMOTE_RM_URL": "http://env:8000"}),
            mock.patch(
                "axon.utils.rewards.remote_reward.requests.post", return_value=_mock_response({"reward": 1.0})
            ) as mp,
        ):
            remote_reward_fn(task_info={"rm_url": "http://task:8000"}, action="x")
            assert mp.call_args[0][0] == "http://task:8000"

    def test_env_var_fallback(self):
        with (
            mock.patch.dict(os.environ, {"AXON_REMOTE_RM_URL": "http://env:8000"}),
            mock.patch(
                "axon.utils.rewards.remote_reward.requests.post", return_value=_mock_response({"reward": 1.0})
            ) as mp,
        ):
            remote_reward_fn(task_info={}, action="x")
            assert mp.call_args[0][0] == "http://env:8000"

    def test_exact_payload_construction(self):
        with mock.patch(
            "axon.utils.rewards.remote_reward.requests.post", return_value=_mock_response({"reward": 1.0})
        ) as mp:
            remote_reward_fn(
                task_info={"rm_url": "http://x", "question": "Q", "answer": "A"},
                action="R",
            )
            assert mp.call_args[1]["json"] == {"prompt": "Q", "response": "R", "label": "A"}

    def test_prompt_and_ground_truth_key_fallbacks(self):
        """Uses 'prompt' when 'question' absent, 'ground_truth' when 'answer' absent."""
        with mock.patch(
            "axon.utils.rewards.remote_reward.requests.post", return_value=_mock_response({"reward": 1.0})
        ) as mp:
            remote_reward_fn(
                task_info={"rm_url": "http://x", "prompt": "P", "ground_truth": "GT"},
                action="R",
            )
            payload = mp.call_args[1]["json"]
            assert payload["prompt"] == "P"
            assert payload["label"] == "GT"

    def test_dict_response_with_is_correct(self):
        resp = {"reward": 0.75, "is_correct": True, "detail": "good"}
        with mock.patch("axon.utils.rewards.remote_reward.requests.post", return_value=_mock_response(resp)):
            result = remote_reward_fn(task_info={"rm_url": "http://x"}, action="a")
            assert result.reward == 0.75
            assert result.is_correct is True
            assert result.metadata == resp

    def test_numeric_float_response(self):
        with mock.patch("axon.utils.rewards.remote_reward.requests.post", return_value=_mock_response(0.42)):
            result = remote_reward_fn(task_info={"rm_url": "http://x"}, action="a")
            assert result.reward == pytest.approx(0.42)
            assert result.is_correct is None
            assert result.metadata == {"raw": 0.42}

    def test_integer_response_coerced_to_float(self):
        with mock.patch("axon.utils.rewards.remote_reward.requests.post", return_value=_mock_response(1)):
            result = remote_reward_fn(task_info={"rm_url": "http://x"}, action="a")
            assert result.reward == 1.0
            assert isinstance(result.reward, float)

    def test_dict_response_missing_reward_key_defaults_to_zero(self):
        """If dict has no 'reward' key, float(data.get('reward', 0.0)) → 0.0."""
        with mock.patch(
            "axon.utils.rewards.remote_reward.requests.post", return_value=_mock_response({"detail": "ok"})
        ):
            result = remote_reward_fn(task_info={"rm_url": "http://x"}, action="a")
            assert result.reward == 0.0

    def test_empty_dict_response(self):
        with mock.patch("axon.utils.rewards.remote_reward.requests.post", return_value=_mock_response({})):
            result = remote_reward_fn(task_info={"rm_url": "http://x"}, action="a")
            assert result.reward == 0.0
            assert result.is_correct is None

    @pytest.mark.parametrize(
        "exc",
        [
            ConnectionError("refused"),
            requests.Timeout("timeout"),
            requests.HTTPError("500"),
            ValueError("bad json"),
        ],
    )
    def test_exceptions_return_zero(self, exc):
        with mock.patch("axon.utils.rewards.remote_reward.requests.post", side_effect=exc):
            result = remote_reward_fn(task_info={"rm_url": "http://x"}, action="a")
            assert result.reward == 0.0
            assert result.is_correct is False

    def test_timeout_kwarg_passed(self):
        """Verify the 30-second timeout is passed to requests.post."""
        with mock.patch(
            "axon.utils.rewards.remote_reward.requests.post", return_value=_mock_response({"reward": 1.0})
        ) as mp:
            remote_reward_fn(task_info={"rm_url": "http://x"}, action="a")
            assert mp.call_args[1]["timeout"] == 30
