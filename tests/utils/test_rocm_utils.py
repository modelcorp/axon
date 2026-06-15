"""Tests for axon.utils.rocm_utils module."""

from unittest import mock

from axon.utils.rocm_utils import get_rocm_env_vars, get_visible_devices_env_key, is_rocm


class TestIsRocm:
    def test_true_when_hip_present(self):
        with mock.patch("torch.version", hip="6.0"):
            assert is_rocm() is True

    def test_false_when_hip_none(self):
        with mock.patch("torch.version", hip=None):
            assert is_rocm() is False


class TestGetVisibleDevicesEnvKey:
    def test_cuda_key(self):
        with mock.patch("axon.utils.rocm_utils.is_rocm", return_value=False):
            assert get_visible_devices_env_key() == "CUDA_VISIBLE_DEVICES"

    def test_hip_key(self):
        with mock.patch("axon.utils.rocm_utils.is_rocm", return_value=True):
            assert get_visible_devices_env_key() == "HIP_VISIBLE_DEVICES"


class TestGetRocmEnvVars:
    def test_empty_on_cuda(self):
        with mock.patch("axon.utils.rocm_utils.is_rocm", return_value=False):
            assert get_rocm_env_vars() == {}

    def test_returns_expected_keys_on_rocm(self):
        with mock.patch("axon.utils.rocm_utils.is_rocm", return_value=True):
            env = get_rocm_env_vars()
            expected_keys = {
                "HIP_FORCE_DEV_KERNARG",
                "NCCL_MIN_NCHANNELS",
                "VLLM_FP8_PADDING",
                "SGLANG_USE_AITER",
            }
            assert expected_keys.issubset(env.keys())

    def test_all_values_are_strings(self):
        with mock.patch("axon.utils.rocm_utils.is_rocm", return_value=True):
            for v in get_rocm_env_vars().values():
                assert isinstance(v, str)
