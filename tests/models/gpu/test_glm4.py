"""GPU integration test: GLM-4-9B through mbridge pipeline.

Usage:
    pytest -m gpu tests/models/gpu/test_glm4.py -v
"""

import pytest

from .conftest import run_mbridge_test

pytestmark = pytest.mark.gpu


class TestGLM4Mbridge:
    """GLM-4-9B-0414 (~18.8 GB) on 1 GPU, TP=1."""

    MODEL_ID = "THUDM/GLM-4-9B-0414"

    @pytest.fixture(scope="class")
    def result(self):
        return run_mbridge_test(self.MODEL_ID, port_offset=0)

    def test_bridge_created(self, result):
        assert result["passed"], f"Worker failed: {result.get('error')}\n{result.get('traceback', '')}"
        assert result["checks"]["bridge_created"]

    def test_bridge_class(self, result):
        assert "GLM4" in result["checks"]["bridge_class"] or "Glm4" in result["checks"]["bridge_class"]

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

    def test_model_type(self, result):
        assert result["checks"]["model_type"] == "glm4"
