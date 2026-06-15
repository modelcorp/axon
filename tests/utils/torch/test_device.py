"""Tests for axon.utils.torch.device module."""

from unittest import mock

import torch

from axon.utils.torch.device import (
    get_device_name,
    get_nccl_backend,
    get_torch_device,
    set_expandable_segments,
)


# ---------------------------------------------------------------------------
# get_device_name – priority ladder
# ---------------------------------------------------------------------------
class TestGetDeviceName:
    def test_cpu_fallback(self):
        with (
            mock.patch("axon.utils.torch.device.is_cuda_available", False),
            mock.patch("axon.utils.torch.device.is_npu_available", False),
        ):
            assert get_device_name() == "cpu"

    def test_cuda_priority_over_npu(self):
        """When both CUDA and NPU are present, CUDA wins."""
        with (
            mock.patch("axon.utils.torch.device.is_cuda_available", True),
            mock.patch("axon.utils.torch.device.is_npu_available", True),
        ):
            assert get_device_name() == "cuda"

    def test_npu_when_no_cuda(self):
        with (
            mock.patch("axon.utils.torch.device.is_cuda_available", False),
            mock.patch("axon.utils.torch.device.is_npu_available", True),
        ):
            assert get_device_name() == "npu"


# ---------------------------------------------------------------------------
# get_torch_device – module resolution
# ---------------------------------------------------------------------------
class TestGetTorchDevice:
    def test_cpu_returns_torch_cpu_module(self):
        with (
            mock.patch("axon.utils.torch.device.is_cuda_available", False),
            mock.patch("axon.utils.torch.device.is_npu_available", False),
        ):
            result = get_torch_device()
            assert result is torch.cpu

    def test_cuda_returns_torch_cuda_module(self):
        with mock.patch("axon.utils.torch.device.is_cuda_available", True):
            result = get_torch_device()
            assert result is torch.cuda

    def test_fallback_when_device_namespace_missing(self):
        """If getattr(torch, device_name) returns None, should fall back to torch.cuda."""
        with (
            mock.patch("axon.utils.torch.device.is_cuda_available", False),
            mock.patch("axon.utils.torch.device.is_npu_available", True),
            mock.patch("axon.utils.torch.device.get_device_name", return_value="nonexistent_device"),
        ):
            result = get_torch_device()
            # Falls back to torch.cuda per the source code
            assert result is torch.cuda


# ---------------------------------------------------------------------------
# get_nccl_backend
# ---------------------------------------------------------------------------
class TestGetNcclBackend:
    def test_returns_nccl_when_no_npu(self):
        with mock.patch("axon.utils.torch.device.is_npu_available", False):
            assert get_nccl_backend() == "nccl"

    def test_returns_hccl_when_npu(self):
        with mock.patch("axon.utils.torch.device.is_npu_available", True):
            assert get_nccl_backend() == "hccl"


# ---------------------------------------------------------------------------
# set_expandable_segments – no-op on CPU
# ---------------------------------------------------------------------------
class TestSetExpandableSegments:
    def test_no_op_on_cpu(self):
        """Should not raise even when CUDA is unavailable."""
        with mock.patch("axon.utils.torch.device.is_cuda_available", False):
            set_expandable_segments(True)
            set_expandable_segments(False)
