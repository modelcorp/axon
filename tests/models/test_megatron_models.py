"""CPU-only tests for axon.models.megatron_models._resolve_vpp_rank.

Tests the pure-Python VPP rank resolution logic that unwraps
DDP -> Float16Module -> GPTModel to find the ``vp_stage`` attribute.

Usage:
    pytest tests/models/test_megatron_models.py -v
"""

from types import SimpleNamespace

import pytest

# _resolve_vpp_rank is a module-level function.  megatron_models.py imports
# heavy dependencies (megatron.core, axon.protocol, etc.) at the top level.
# If they are available, import directly; otherwise, mock them out so we can
# still test the pure-Python function.
try:
    from axon.models.megatron_models import _resolve_vpp_rank
except ImportError:
    import sys
    from unittest.mock import MagicMock

    # Provide stubs for heavy modules that may not be installed.
    _STUBS = [
        "megatron",
        "megatron.core",
        "megatron.core.parallel_state",
        "axon.protocol",
        "axon.trainer.algos.loss",
        "axon.utils.megatron.tensor_parallel",
        "axon.utils.print_utils",
        "axon.utils.rl.kl",
        "axon.utils.rl.sampler",
        "axon.utils.seqlen_balancing",
        "axon.utils.torch",
    ]
    for mod_name in _STUBS:
        if mod_name not in sys.modules:
            sys.modules[mod_name] = MagicMock()

    from axon.models.megatron_models import _resolve_vpp_rank


# ---------------------------------------------------------------------------
# _resolve_vpp_rank
# ---------------------------------------------------------------------------


class TestResolveVppRank:
    """Focused tests for VPP rank resolution: explicit override, module-depth
    unwrapping, deepest-first priority, fallback chain, and zero-is-valid."""

    # --- Explicit vpp_rank always wins ---

    @pytest.mark.parametrize(
        "vpp_rank, model_vp_stage",
        [
            (3, None),  # explicit with no model stage
            (0, 5),  # explicit zero overrides model stage (not treated as falsy)
            (2, 5),  # explicit non-zero overrides model stage
        ],
        ids=["explicit-3-no-model", "explicit-0-overrides-5", "explicit-2-overrides-5"],
    )
    def test_explicit_vpp_rank(self, vpp_rank, model_vp_stage):
        """When vpp_rank is explicitly provided (including 0), it is returned as-is."""
        if model_vp_stage is not None:
            model = SimpleNamespace(vp_stage=model_vp_stage)
        else:
            model = SimpleNamespace()
        assert _resolve_vpp_rank(model, vpp_rank=vpp_rank) == vpp_rank

    # --- Module-depth discovery (parametrized) ---

    @pytest.mark.parametrize(
        "depth, expected",
        [
            (0, 42),  # model.vp_stage
            (1, 7),  # model.module.vp_stage
            (2, 9),  # model.module.module.vp_stage
        ],
        ids=["bare-model", "single-wrap", "double-wrap"],
    )
    def test_vp_stage_at_depth(self, depth, expected):
        """vp_stage discovered at different nesting depths."""
        if depth == 0:
            model = SimpleNamespace(vp_stage=expected)
        elif depth == 1:
            inner = SimpleNamespace(vp_stage=expected)
            model = SimpleNamespace(module=inner)
        else:
            innermost = SimpleNamespace(vp_stage=expected)
            middle = SimpleNamespace(module=innermost, vp_stage=None)
            model = SimpleNamespace(module=middle)
        assert _resolve_vpp_rank(model, vpp_rank=None) == expected

    # --- Deepest-first priority ---

    def test_deepest_module_checked_first(self):
        """model.module.module is checked before model.module and model.

        The iteration order is: model.module.module, model.module, model.
        The first one with a non-None vp_stage wins.
        """
        innermost = SimpleNamespace(vp_stage=9)
        middle = SimpleNamespace(module=innermost, vp_stage=7)
        model = SimpleNamespace(module=middle, vp_stage=5)
        assert _resolve_vpp_rank(model, vpp_rank=None) == 9

    def test_middle_module_wins_when_deepest_is_none(self):
        """If model.module.module.vp_stage is None, model.module is used."""
        innermost = SimpleNamespace(vp_stage=None)
        middle = SimpleNamespace(module=innermost, vp_stage=7)
        model = SimpleNamespace(module=middle, vp_stage=5)
        assert _resolve_vpp_rank(model, vpp_rank=None) == 7

    def test_bare_model_wins_when_all_modules_none(self):
        """If nested modules have vp_stage=None, fall through to bare model."""
        innermost = SimpleNamespace(vp_stage=None)
        middle = SimpleNamespace(module=innermost, vp_stage=None)
        model = SimpleNamespace(module=middle, vp_stage=5)
        assert _resolve_vpp_rank(model, vpp_rank=None) == 5

    # --- Zero is a valid stage value ---

    def test_vp_stage_zero_on_model_is_found(self):
        """vp_stage=0 on the model should be returned, not treated as falsy."""
        model = SimpleNamespace(vp_stage=0)
        assert _resolve_vpp_rank(model, vpp_rank=None) == 0

    def test_vp_stage_zero_on_deepest_module(self):
        """vpp_rank=None + model.module.module.vp_stage=0 returns 0 (valid stage)."""
        innermost = SimpleNamespace(vp_stage=0)
        middle = SimpleNamespace(module=innermost, vp_stage=5)
        model = SimpleNamespace(module=middle, vp_stage=3)
        assert _resolve_vpp_rank(model, vpp_rank=None) == 0

    # --- Fallback to None ---

    def test_no_vpp_rank_found(self):
        """Model with no vp_stage anywhere returns None."""
        model = SimpleNamespace()
        assert _resolve_vpp_rank(model, vpp_rank=None) is None

    def test_no_vpp_rank_found_with_nested_modules(self):
        """Model with .module chain but no vp_stage anywhere returns None."""
        inner = SimpleNamespace()
        model = SimpleNamespace(module=inner)
        assert _resolve_vpp_rank(model, vpp_rank=None) is None
