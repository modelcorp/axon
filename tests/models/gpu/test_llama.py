"""GPU integration test: Llama-3.1-8B (llama bridge).

Usage:
    pytest -m gpu tests/models/gpu/test_llama.py -v
"""

import pytest

from .conftest import run_mbridge_test

pytestmark = pytest.mark.gpu


class TestLlamaMbridge:
    """Llama-3.1-8B (~16 GB, model_type=llama) on 1 GPU."""

    MODEL_ID = "meta-llama/Llama-3.1-8B"

    @pytest.fixture(scope="class")
    def result(self):
        return run_mbridge_test(self.MODEL_ID, port_offset=12)

    def test_bridge_created(self, result):
        assert result["passed"], f"Worker failed: {result.get('error')}\n{result.get('traceback', '')}"
        assert result["checks"]["bridge_created"]

    def test_bridge_class(self, result):
        assert "LLaMA" in result["checks"]["bridge_class"] or "Llama" in result["checks"]["bridge_class"]

    def test_weights_loaded(self, result):
        assert result["checks"]["weights_loaded"]

    def test_forward_pass_completed(self, result):
        assert result["checks"]["forward_pass_completed"]

    def test_output_no_nan(self, result):
        assert not result["checks"].get("output_has_nan", True), "NaN in output"

    def test_output_no_inf(self, result):
        assert not result["checks"].get("output_has_inf", True), "Inf in output"

    def test_model_type(self, result):
        assert result["checks"]["model_type"] == "llama"
