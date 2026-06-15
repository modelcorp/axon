"""Tests for axon.utils.torch.dtypes module."""

import pytest
import torch

from axon.utils.torch.dtypes import BFLOAT_LIST, FLOAT_LIST, HALF_LIST, PrecisionType


# ---------------------------------------------------------------------------
# to_dtype – exhaustive via parametrize
# ---------------------------------------------------------------------------
class TestToDtype:
    @pytest.mark.parametrize(
        "val,expected",
        [
            *[(v, torch.float16) for v in HALF_LIST],
            *[(v, torch.float32) for v in FLOAT_LIST],
            *[(v, torch.bfloat16) for v in BFLOAT_LIST],
        ],
    )
    def test_known_values(self, val, expected):
        assert PrecisionType.to_dtype(val) is expected

    @pytest.mark.parametrize("val", ["fp8", torch.float64, None, "mixed", 64, torch.int8])
    def test_rejects_unsupported(self, val):
        with pytest.raises(RuntimeError, match="unexpected precision"):
            PrecisionType.to_dtype(val)

    def test_int_32_vs_int_16_not_confused(self):
        assert PrecisionType.to_dtype(32) is torch.float32
        assert PrecisionType.to_dtype(16) is torch.float16


# ---------------------------------------------------------------------------
# to_str – roundtrip and rejection
# ---------------------------------------------------------------------------
class TestToStr:
    @pytest.mark.parametrize(
        "dtype,expected",
        [
            (torch.float16, "fp16"),
            (torch.float32, "fp32"),
            (torch.bfloat16, "bf16"),
        ],
    )
    def test_roundtrip(self, dtype, expected):
        assert PrecisionType.to_str(dtype) == expected
        assert PrecisionType.to_str(PrecisionType.to_dtype(expected)) == expected

    @pytest.mark.parametrize("dtype", [torch.float64, torch.int8, torch.int32])
    def test_rejects_unsupported(self, dtype):
        with pytest.raises(RuntimeError, match="unexpected precision"):
            PrecisionType.to_str(dtype)


# ---------------------------------------------------------------------------
# is_* – cross-type rejection (every member of other families must be False)
# ---------------------------------------------------------------------------
class TestIsCrossTypeRejection:
    @pytest.mark.parametrize("val", FLOAT_LIST + BFLOAT_LIST)
    def test_is_fp16_rejects(self, val):
        assert PrecisionType.is_fp16(val) is False

    @pytest.mark.parametrize("val", HALF_LIST + BFLOAT_LIST)
    def test_is_fp32_rejects(self, val):
        assert PrecisionType.is_fp32(val) is False

    @pytest.mark.parametrize("val", HALF_LIST + FLOAT_LIST)
    def test_is_bf16_rejects(self, val):
        assert PrecisionType.is_bf16(val) is False


# ---------------------------------------------------------------------------
# supported_type / supported_types – PrecisionType is not an Enum, so these crash
# ---------------------------------------------------------------------------
class TestSupportedTypeBroken:
    """Document that these methods are broken: PrecisionType is a plain class."""

    def test_supported_type_raises_type_error(self):
        with pytest.raises(TypeError):
            PrecisionType.supported_type("32")

    def test_supported_types_raises_type_error(self):
        with pytest.raises(TypeError):
            PrecisionType.supported_types()


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------
class TestEdgeCases:
    def test_int_0_rejected_by_all_families(self):
        assert PrecisionType.is_fp16(0) is False
        assert PrecisionType.is_fp32(0) is False
        assert PrecisionType.is_bf16(0) is False

    def test_bf16_string_not_confused_with_fp16(self):
        """'bf16' contains '16' but must NOT match fp16."""
        assert PrecisionType.is_fp16("bf16") is False
        assert PrecisionType.is_fp32("bf16") is False

    def test_to_dtype_returns_singleton_identity(self):
        """to_dtype should return the exact torch dtype singleton."""
        assert PrecisionType.to_dtype(torch.float16) is torch.float16
        assert PrecisionType.to_dtype(torch.bfloat16) is torch.bfloat16
        assert PrecisionType.to_dtype(torch.float32) is torch.float32
