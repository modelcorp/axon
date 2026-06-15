"""GPU integration test: Qwen3-Next-80B through mbridge pipeline with TP/PP.

Qwen3-Next has num_key_value_heads=2, so max TP=2.
Uses TP=2, PP=4 across 8 GPUs.

Usage:
    pytest -m gpu tests/models/gpu/test_qwen3_next.py -v
"""

import pytest

from .conftest import run_mbridge_test

pytestmark = pytest.mark.gpu


class TestQwen3NextTP2PP4:
    """Qwen3-Next-80B-A3B-Instruct (~152 GB) with TP=2, PP=4 on 8 GPUs."""

    MODEL_ID = "Qwen/Qwen3-Next-80B-A3B-Instruct"

    @pytest.fixture(scope="class")
    def result(self):
        return run_mbridge_test(self.MODEL_ID, port_offset=3, timeout=1200, tp=2, pp=4)

    def test_bridge_created(self, result):
        assert result["passed"], f"Worker failed: {result.get('error')}\n{result.get('traceback', '')}"
        assert result["checks"]["bridge_created"]

    def test_bridge_class(self, result):
        assert "Qwen3Next" in result["checks"]["bridge_class"]

    def test_weights_loaded(self, result):
        assert result["checks"]["weights_loaded"]

    def test_forward_pass_completed(self, result):
        assert result["checks"]["forward_pass_completed"]

    def test_output_no_nan(self, result):
        # Output only on last PP stage; rank 0 may be first stage
        has_output = "output_has_nan" in result["checks"]
        if has_output:
            assert not result["checks"]["output_has_nan"], "NaN in output"

    def test_output_no_inf(self, result):
        has_output = "output_has_inf" in result["checks"]
        if has_output:
            assert not result["checks"]["output_has_inf"], "Inf in output"

    def test_decoder_layers_split_across_pp(self, result):
        # PP=4, 48 layers: each rank should have 12 layers
        n_this_rank = result["checks"].get("decoder_layers_this_rank")
        n_total = result["checks"]["num_hidden_layers"]
        if n_this_rank is not None:
            assert n_this_rank == n_total // 4, f"PP=4: expected {n_total // 4} layers, got {n_this_rank}"

    def test_model_type(self, result):
        assert result["checks"]["model_type"] == "qwen3_next"

    def test_parallel_config(self, result):
        assert result["checks"]["tp_size"] == 2
        assert result["checks"]["pp_size"] == 4
