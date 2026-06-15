"""GPU integration test: Qwen3-8B through mbridge pipeline.

Usage:
    pytest -m gpu tests/models/gpu/test_qwen3.py -v
"""

import pytest

from .conftest import run_mbridge_test

pytestmark = pytest.mark.gpu


class TestQwen3Mbridge:
    """Qwen3-8B (~16.4 GB) on 1 GPU, TP=1."""

    MODEL_ID = "Qwen/Qwen3-8B"

    @pytest.fixture(scope="class")
    def result(self):
        return run_mbridge_test(self.MODEL_ID, port_offset=1)

    def test_bridge_created(self, result):
        assert result["passed"], f"Worker failed: {result.get('error')}\n{result.get('traceback', '')}"
        assert result["checks"]["bridge_created"]

    def test_weights_loaded(self, result):
        assert result["checks"]["weights_loaded"]

    def test_forward_pass_completed(self, result):
        assert result["checks"]["forward_pass_completed"]

    def test_output_no_nan(self, result):
        assert not result["checks"].get("output_has_nan", True), "NaN in output"

    def test_output_no_inf(self, result):
        assert not result["checks"].get("output_has_inf", True), "Inf in output"

    def test_decoder_layer_count(self, result):
        n = result["checks"].get("decoder_layers_this_rank")
        assert n == result["checks"]["num_hidden_layers"]
