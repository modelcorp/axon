"""GPU integration tests: Gemma-4-31B-it (dense) and Gemma-4-26B-A4B-it (MoE).

Exercises the Gemma4 mbridge/mcore port over all Megatron "6D" parallelisms:
    TP (tensor) · PP (pipeline) · CP (context) · DP (data)
    EP (expert) · ETP (expert-tensor)

The published checkpoints are Gemma4ForConditionalGeneration, so text
weights live under ``model.language_model.*`` in their safetensors. The
bridge auto-detects the prefix at load time (see Gemma4Bridge).

Usage:
    pytest -m gpu tests/models/gpu/test_gemma4.py -v
"""

import pytest

from .conftest import run_mbridge_test

pytestmark = pytest.mark.gpu


# ---------------------------------------------------------------------------
# Dense 31B variant
# ---------------------------------------------------------------------------


class TestGemma4DenseSingleGPU:
    """Smoke test: DP=1, everything else trivial."""

    MODEL_ID = "google/gemma-4-31B-it"

    @pytest.fixture(scope="class")
    def result(self):
        return run_mbridge_test(self.MODEL_ID, port_offset=40, timeout=1200)

    def test_bridge_created(self, result):
        assert result["passed"], f"Worker failed: {result.get('error')}\n{result.get('traceback', '')}"
        assert result["checks"]["bridge_created"]

    def test_bridge_class(self, result):
        assert "Gemma4" in result["checks"]["bridge_class"]

    def test_weights_loaded(self, result):
        assert result["checks"]["weights_loaded"], "Weights failed to load (likely prefix mismatch)"

    def test_forward_pass_completed(self, result):
        assert result["checks"]["forward_pass_completed"]

    def test_output_no_nan(self, result):
        assert not result["checks"].get("output_has_nan", True), "NaN in output"

    def test_output_no_inf(self, result):
        assert not result["checks"].get("output_has_inf", True), "Inf in output"

    def test_model_type(self, result):
        assert result["checks"]["model_type"] == "gemma4"


@pytest.mark.parametrize(
    "tp,pp,cp",
    [
        pytest.param(2, 1, 1, id="TP2"),
        pytest.param(1, 2, 1, id="PP2"),
        pytest.param(1, 1, 2, id="CP2"),
        pytest.param(2, 2, 1, id="TP2PP2"),
        pytest.param(2, 1, 2, id="TP2CP2"),
        pytest.param(1, 2, 2, id="PP2CP2"),
        pytest.param(2, 2, 2, id="TP2PP2CP2"),
    ],
)
class TestGemma4DenseParallel:
    """Exercise attention-side parallelism for the dense model.

    TP shards MLP weights. Attention is the HF ``Gemma4TextAttention`` wrapped
    by ``HuggingfaceAttention`` — it handles SP gather/scatter but is NOT
    TP-sharded, so attn weights are replicated across TP ranks (memory cost,
    not a correctness issue). CP splits the sequence dim; see
    ``HuggingfaceAttention`` for the ring-style gather/scatter.
    """

    MODEL_ID = "google/gemma-4-31B-it"

    @pytest.fixture(scope="class")
    def result(self, tp, pp, cp):
        offset = 41 + tp * 4 + pp * 2 + cp  # unique per-combination port
        return run_mbridge_test(self.MODEL_ID, port_offset=offset, timeout=1500, tp=tp, pp=pp, cp=cp)

    def test_weights_loaded(self, result, tp, pp, cp):
        assert result["passed"], f"Worker failed at TP={tp},PP={pp},CP={cp}: {result.get('error')}"
        assert result["checks"]["weights_loaded"]

    def test_forward_ok(self, result):
        assert result["checks"]["forward_pass_completed"]
        assert not result["checks"].get("output_has_nan", True)
        assert not result["checks"].get("output_has_inf", True)


# ---------------------------------------------------------------------------
# MoE 26B-A4B variant (128 experts, top-k=8)
# ---------------------------------------------------------------------------


class TestGemma4MoESingleGPU:
    MODEL_ID = "google/gemma-4-26B-A4B-it"

    @pytest.fixture(scope="class")
    def result(self):
        return run_mbridge_test(self.MODEL_ID, port_offset=60, timeout=1200)

    def test_bridge_created(self, result):
        assert result["passed"], f"Worker failed: {result.get('error')}\n{result.get('traceback', '')}"
        assert result["checks"]["bridge_created"]

    def test_weights_loaded(self, result):
        assert result["checks"]["weights_loaded"]

    def test_forward_pass_completed(self, result):
        assert result["checks"]["forward_pass_completed"]

    def test_output_no_nan(self, result):
        assert not result["checks"].get("output_has_nan", True)

    def test_output_no_inf(self, result):
        assert not result["checks"].get("output_has_inf", True)


@pytest.mark.parametrize(
    "tp,pp,cp,ep,etp",
    [
        # ── single-axis ────────────────────────────────────────────────────
        pytest.param(8, 1, 1, 1, 1, id="TP8"),
        pytest.param(1, 1, 1, 2, 1, id="EP2"),
        pytest.param(1, 1, 1, 4, 1, id="EP4"),
        pytest.param(1, 1, 1, 8, 1, id="EP8"),
        pytest.param(1, 1, 8, 1, 1, id="CP8"),  # full CP (gather full seq, replicate attn)
        # ── ETP / mixed expert axes ────────────────────────────────────────
        pytest.param(2, 1, 1, 1, 2, id="TP2_ETP2"),
        pytest.param(2, 1, 1, 2, 2, id="TP2_EP2_ETP2"),  # the ETP-export bug case
        pytest.param(8, 1, 1, 4, 2, id="TP8_EP4_ETP2"),  # frozenlake recipe shape
        # ── Mixed attention + expert parallel ─────────────────────────────
        pytest.param(2, 1, 1, 2, 1, id="TP2_EP2"),
        pytest.param(4, 1, 1, 2, 1, id="TP4_EP2"),
        pytest.param(1, 2, 1, 2, 1, id="PP2_EP2"),
        # ── CP combinations ────────────────────────────────────────────────
        pytest.param(1, 1, 2, 2, 1, id="CP2_EP2"),
        pytest.param(2, 1, 2, 2, 1, id="TP2_CP2_EP2"),
        # ── PP + CP ───────────────────────────────────────────────────────
        pytest.param(2, 2, 2, 1, 1, id="TP2_PP2_CP2"),
        pytest.param(1, 2, 2, 2, 1, id="PP2_CP2_EP2"),
    ],
)
class TestGemma4MoEParallel:
    """Exercise MoE parallelism.

    The Gemma4 MoE block:
    - Router is non-TP (replicated) — weights small, loaded on all ranks.
    - Experts use Megatron's SequentialMLP with EP/ETP sharding. 128 experts
      means any ``ep_size | 128`` (1, 2, 4, 8, 16, 32, 64, 128) is valid.
    - Dense MLP uses standard Megatron TP.
    - Router receives the RAW residual (not pre-normed). ``Gemma4MoELayer``
      decouples ``router_input`` from the dispatch input so the router's
      internal ``with_scale=False`` RMSNorm matches HF.
    - Outer ``post_feedforward_layernorm`` is applied once in the transformer
      layer after the dense + MoE outputs are summed.
    """

    MODEL_ID = "google/gemma-4-26B-A4B-it"

    @pytest.fixture(scope="class")
    def result(self, tp, pp, cp, ep, etp):
        offset = 70 + tp * 8 + pp * 4 + cp * 2 + ep + etp  # unique enough
        return run_mbridge_test(
            self.MODEL_ID,
            port_offset=offset,
            timeout=1800,
            tp=tp,
            pp=pp,
            cp=cp,
            ep=ep,
            etp=etp,
        )

    def test_weights_loaded(self, result, tp, pp, cp, ep, etp):
        assert result["passed"], (
            f"Worker failed at TP={tp},PP={pp},CP={cp},EP={ep},ETP={etp}: "
            f"{result.get('error')}\n{result.get('traceback', '')[-1500:]}"
        )
        assert result["checks"]["weights_loaded"]

    def test_forward_ok(self, result):
        assert result["checks"]["forward_pass_completed"]
        assert not result["checks"].get("output_has_nan", True)
        assert not result["checks"].get("output_has_inf", True)


# ---------------------------------------------------------------------------
# VPP coverage — virtual pipeline parallel splits each PP stage into N chunks.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "tp,pp,vpp,ep,etp",
    [
        pytest.param(1, 2, 3, 4, 1, id="PP2_VPP3_EP4"),
        pytest.param(1, 2, 5, 4, 1, id="PP2_VPP5_EP4"),
        pytest.param(2, 2, 3, 2, 1, id="TP2_PP2_VPP3_EP2"),
    ],
)
class TestGemma4MoEVPP:
    """Virtual pipeline parallelism: each PP stage holds num_layers/(pp*vpp) layers
    per chunk. Catches the ``vp_stage`` propagation bug in Glm4 layer init."""

    MODEL_ID = "google/gemma-4-26B-A4B-it"

    @pytest.fixture(scope="class")
    def result(self, tp, pp, vpp, ep, etp):
        offset = 110 + tp * 8 + pp * 4 + vpp + ep + etp
        return run_mbridge_test(
            self.MODEL_ID,
            port_offset=offset,
            timeout=1500,
            tp=tp,
            pp=pp,
            ep=ep,
            etp=etp,
            vpp=vpp,
        )

    def test_loaded(self, result, tp, pp, vpp, ep, etp):
        assert result["passed"], (
            f"Worker failed at TP={tp},PP={pp},VPP={vpp},EP={ep},ETP={etp}: "
            f"{result.get('error')}\n{result.get('traceback', '')[-1500:]}"
        )
        assert result["checks"]["weights_loaded"]
        assert result["checks"]["num_model_chunks"] == vpp

    def test_forward_ok(self, result):
        assert result["checks"]["forward_pass_completed"]


# ---------------------------------------------------------------------------
# Trainer→sampler weight export path — runs `bridge.export_weights` end-to-end.
# Covers the ETP merge code that originally crashed during hybrid-engine sync.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "tp,ep,etp",
    [
        pytest.param(2, 2, 2, id="TP2_EP2_ETP2"),
        pytest.param(8, 4, 2, id="TP8_EP4_ETP2"),  # frozenlake recipe
        pytest.param(8, 8, 1, id="TP8_EP8"),
    ],
)
class TestGemma4MoEExport:
    """Verify ``bridge.export_weights`` completes without crashing under ETP>1.

    Runs the worker with ``GEMMA4_TEST_EXPORT=1`` so it iterates the bridge
    export generator and counts emitted tensors. The path being exercised is
    what the trainer→sampler hybrid-engine sync uses; an early version of the
    ETP gather called ``_weight_merge_across_tp`` (which asserts ``len ==
    tp_size``) and crashed at TP=8 ETP=2.
    """

    MODEL_ID = "google/gemma-4-26B-A4B-it"

    @pytest.fixture(scope="class")
    def result(self, tp, ep, etp):
        import json
        import os
        import subprocess
        import sys
        import tempfile

        from .conftest import _BASE_PORT, WORKER_SCRIPT

        offset = 200 + tp * 4 + ep * 2 + etp
        port = _BASE_PORT + offset
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            output_path = f.name
        env = os.environ.copy()
        env["GEMMA4_TEST_EXPORT"] = "1"
        nproc = tp  # ep/etp are orthogonal axes within the same world
        cmd = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            f"--nproc_per_node={nproc}",
            "--master_port",
            str(port),
            WORKER_SCRIPT,
            "--model-id",
            self.MODEL_ID,
            "--output",
            output_path,
            "--tp",
            str(tp),
            "--pp",
            "1",
            "--cp",
            "1",
            "--ep",
            str(ep),
            "--etp",
            str(etp),
            "--generate-tokens",
            "0",
        ]
        subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=1500)
        with open(output_path) as f:
            return json.load(f)

    def test_export_ok(self, result, tp, ep, etp):
        assert result["passed"], (
            f"export failed at TP={tp},EP={ep},ETP={etp}: {result.get('error')}\n{result.get('traceback', '')[-1500:]}"
        )
        c = result["checks"]
        assert c.get("export_weights_error") is None
        assert c.get("export_weights_count", 0) > 100, f"too few weights exported: {c.get('export_weights_count')}"


# ---------------------------------------------------------------------------
# Bit-exact round-trip: bridge.load_weights -> bridge.export_weights -> HF.
#
# Because load+export are pure reshape/gather/concat (no compute), every
# exported tensor MUST equal its HF source bit-exactly. This catches the class
# of bugs where sharded merges produce wrong values (e.g. gate/up interleaving
# under ETP>1 for gated MLPs), wrong shapes, or renamed/missing tensors —
# exactly the path that drives trainer→sampler hybrid-engine weight sync.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "tp,ep,etp",
    [
        pytest.param(1, 1, 1, id="baseline"),  # no sharding — isolates mapping logic
        pytest.param(8, 1, 1, id="TP8"),  # TP only, no MoE parallelism
        pytest.param(1, 8, 1, id="EP8"),  # EP only
        pytest.param(2, 2, 2, id="TP2_EP2_ETP2"),  # minimum ETP>1 case
        pytest.param(8, 4, 2, id="TP8_EP4_ETP2"),  # frozenlake recipe shape
        pytest.param(8, 8, 1, id="TP8_EP8"),  # all-EP, no ETP
    ],
)
class TestGemma4MoEExportCompare:
    """End-to-end weight sanity via bit-exact HF round-trip.

    Drives the exact same ``bridge.export_weights`` path the trainer→sampler
    hybrid-engine sync uses (see ``MegatronSyncSamplerMixin.sampler_mode``).
    A failure here means vLLM would receive corrupt weights — localized per
    tensor in ``result['checks']['export_compare']['worst_by_max_abs']``.
    """

    MODEL_ID = "google/gemma-4-26B-A4B-it"

    @pytest.fixture(scope="class")
    def result(self, tp, ep, etp):
        offset = 300 + tp * 8 + ep * 2 + etp
        return run_mbridge_test(
            self.MODEL_ID,
            port_offset=offset,
            timeout=1800,
            tp=tp,
            ep=ep,
            etp=etp,
            extra_env={"MBRIDGE_EXPORT_COMPARE": "1"},
        )

    def test_worker_ok(self, result, tp, ep, etp):
        assert result["passed"], (
            f"Worker failed at TP={tp},EP={ep},ETP={etp}: {result.get('error')}\n{result.get('traceback', '')[-1500:]}"
        )
        assert "export_compare" in result["checks"], (
            "export_compare block missing — did MBRIDGE_EXPORT_COMPARE propagate?"
        )

    def test_no_shape_mismatches(self, result):
        c = result["checks"]["export_compare"]
        assert c["n_shape_mismatch"] == 0, f"{c['n_shape_mismatch']} shape mismatches:\n" + "\n".join(
            f"  {x}" for x in c["shape_mismatches"]
        )

    def test_no_orphan_exports(self, result):
        """Every exported name must exist in the HF checkpoint."""
        c = result["checks"]["export_compare"]
        assert c["n_exported_not_in_hf"] == 0, (
            f"{c['n_exported_not_in_hf']} exported names have no HF counterpart "
            f"(name-mapping bug). Sample: {c['exported_not_in_hf_sample']}"
        )

    def test_all_hf_text_weights_exported(self, result):
        """Every text-model HF tensor must appear in the export stream."""
        c = result["checks"]["export_compare"]
        assert c["n_hf_text_not_exported"] == 0, (
            f"{c['n_hf_text_not_exported']} text-model HF weights never exported "
            f"(these would stay random-init in vLLM and drive garbage outputs). "
            f"Sample: {c['hf_text_not_exported_sample']}"
        )

    def test_values_bit_exact(self, result):
        """load -> export is a pure reshape round-trip; diff must be 0."""
        c = result["checks"]["export_compare"]
        assert c["max_max_abs"] == 0.0, (
            f"Exported values diverge from HF (max abs diff = {c['max_max_abs']:.3e}, "
            f"{c['n_value_diff']}/{c['n_exported_matched']} tensors differ). "
            f"Worst offenders:\n"
            + "\n".join(
                f"  {x['name']:80s}  max_abs={x['max_abs']:.3e}  shape={x['shape']}" for x in c["worst_by_max_abs"]
            )
        )
