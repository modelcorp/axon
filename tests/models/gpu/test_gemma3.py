"""GPU integration test: Gemma-3-4B-IT (gemma3 bridge).

Usage:
    pytest -m gpu tests/models/gpu/test_gemma3.py -v
"""

import pytest

from .conftest import run_mbridge_test

pytestmark = pytest.mark.gpu


@pytest.mark.xfail(
    reason="mbridge Gemma3 modules incompatible with current megatron-core (pg_collection kwarg)",
    strict=False,
)
class TestGemma3Mbridge:
    """Gemma-3-4B-IT (~8.6 GB, model_type=gemma3) on 1 GPU."""

    MODEL_ID = "google/gemma-3-4b-it"

    @pytest.fixture(scope="class")
    def result(self):
        return run_mbridge_test(self.MODEL_ID, port_offset=14)

    def test_bridge_created(self, result):
        assert result["passed"], f"Worker failed: {result.get('error')}\n{result.get('traceback', '')}"
        assert result["checks"]["bridge_created"]

    def test_bridge_class(self, result):
        assert "Gemma3" in result["checks"]["bridge_class"] or "gemma3" in result["checks"]["bridge_class"].lower()

    def test_weights_loaded(self, result):
        assert result["checks"]["weights_loaded"]

    def test_forward_pass_completed(self, result):
        assert result["checks"]["forward_pass_completed"]

    def test_output_no_nan(self, result):
        assert not result["checks"].get("output_has_nan", True), "NaN in output"

    def test_output_no_inf(self, result):
        assert not result["checks"].get("output_has_inf", True), "Inf in output"

    def test_model_type(self, result):
        assert result["checks"]["model_type"] == "gemma3"
