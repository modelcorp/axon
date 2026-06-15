"""Tests for mcore model_forward_gen() orchestration logic (CPU-only).

Verifies the generated model_forward function from model_forward_gen():
- THD format: basic packing, padding handling, logits_processor, value_model
- BSHD format: basic compaction, left-padding, logits_processor
- Edge cases: invalid format, pre_process=False, post_process=False, multi-modal inputs
- Fused forward patching: _get_patching_model, patch/unpatch roundtrip

Usage:
    pytest tests/models/mcore/test_forward_model.py -v
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_env():
    """Mock megatron parallel state and unwrap_model for CPU testing.

    Sets up a single-GPU (tp=1, cp=1) environment so preprocess/postprocess
    helpers execute their non-distributed code paths.
    """
    mock_mpu = MagicMock()
    mock_mpu.get_tensor_model_parallel_world_size.return_value = 1
    mock_mpu.get_context_parallel_world_size.return_value = 1
    mock_mpu.get_context_parallel_rank.return_value = 0

    inner_model = SimpleNamespace(
        pre_process=True,
        post_process=True,
        config=SimpleNamespace(sequence_parallel=False, fp8=None),
    )

    with (
        patch("axon.models.mcore.forward.util.mpu", mock_mpu),
        patch("axon.models.mcore.forward.model_forward.unwrap_model", return_value=inner_model),
    ):
        yield inner_model, mock_mpu


def _make_model_callable(return_tensor):
    """Create a MagicMock model whose __call__ returns *return_tensor*."""
    model = MagicMock()
    model.return_value = return_tensor
    return model


# ---------------------------------------------------------------------------
# THD format tests
# ---------------------------------------------------------------------------


class TestModelForwardThd:
    """Tests for data_format='thd' path through model_forward_gen()."""

    def test_thd_format_basic(self, mock_env):
        """Basic THD: all-valid tokens, model output unpacked to original shape."""
        from axon.models.mcore.forward.model_forward import model_forward_gen

        inner_model, _ = mock_env
        batch, seq_len, hidden = 1, 8, 16

        input_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)  # [1, 8]
        attention_mask = torch.ones(batch, seq_len, dtype=torch.bool)  # all valid
        position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)  # [1, 8]

        # preprocess_packed_seqs will pack to [1, packed_len]; with tp=1, cp=1
        # packed_len == seq_len for all-valid mask
        packed_output = torch.randn(1, seq_len, hidden)

        model = _make_model_callable(packed_output)
        forward_fn = model_forward_gen(vision_model=False)
        result = forward_fn(
            model,
            input_ids,
            attention_mask,
            position_ids,
            multi_modal_inputs={},
            data_format="thd",
        )

        # Model must have been called once
        model.assert_called_once()
        call_kwargs = model.call_args
        # packed_seq_params should be passed
        assert "packed_seq_params" in call_kwargs.kwargs or (len(call_kwargs.args) > 0)

        # Output shape: [batch, seq_len, hidden]
        assert result.shape == (batch, seq_len, hidden)

    def test_thd_with_padding(self, mock_env):
        """THD with right-padding: padding positions should be zero in output."""
        from axon.models.mcore.forward.model_forward import model_forward_gen

        inner_model, _ = mock_env
        batch, seq_len, hidden = 1, 8, 4
        valid_len = 5

        input_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
        attention_mask = torch.zeros(batch, seq_len, dtype=torch.bool)
        attention_mask[0, :valid_len] = True
        position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)

        # With tp=1, cp=1, packed_len == valid_len (only valid tokens are packed)
        packed_output = torch.ones(1, valid_len, hidden) * 3.0

        model = _make_model_callable(packed_output)
        forward_fn = model_forward_gen(vision_model=False)
        result = forward_fn(
            model,
            input_ids,
            attention_mask,
            position_ids,
            multi_modal_inputs={},
            data_format="thd",
        )

        assert result.shape == (batch, seq_len, hidden)
        # Valid positions should have been filled with model output
        assert (result[0, :valid_len] != 0).any()
        # Padding positions should be zeros
        assert (result[0, valid_len:] == 0).all()

    def test_thd_with_logits_processor(self, mock_env):
        """THD with logits_processor: dict output, each value postprocessed."""
        from axon.models.mcore.forward.model_forward import model_forward_gen

        inner_model, _ = mock_env
        batch, seq_len, hidden = 1, 6, 8

        input_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
        attention_mask = torch.ones(batch, seq_len, dtype=torch.bool)
        position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)

        packed_output = torch.randn(1, seq_len, hidden)
        model = _make_model_callable(packed_output)

        # logits_processor receives model output + preprocessed args, returns dict
        def fake_logits_processor(output, **kwargs):
            return {
                "log_probs": output[..., 0],  # [1, packed_len]
                "entropy": output[..., 1],  # [1, packed_len]
            }

        # logits_processor_args: each value is preprocessed the same way as input_ids
        lp_args = {
            "labels": torch.randint(0, 100, (batch, seq_len), dtype=torch.long),
        }

        forward_fn = model_forward_gen(vision_model=False)
        result = forward_fn(
            model,
            input_ids,
            attention_mask,
            position_ids,
            multi_modal_inputs={},
            logits_processor=fake_logits_processor,
            logits_processor_args=lp_args,
            data_format="thd",
        )

        # Result should be a dict with postprocessed tensors
        assert isinstance(result, dict)
        assert "log_probs" in result
        assert "entropy" in result
        assert result["log_probs"].shape == (batch, seq_len)
        assert result["entropy"].shape == (batch, seq_len)

    def test_thd_value_model(self, mock_env):
        """THD with value_model=True: last dim squeezed from output."""
        from axon.models.mcore.forward.model_forward import model_forward_gen

        inner_model, _ = mock_env
        batch, seq_len = 1, 8
        # Value model output typically has hidden_dim=1
        packed_output = torch.randn(1, seq_len, 1)

        input_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
        attention_mask = torch.ones(batch, seq_len, dtype=torch.bool)
        position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)

        model = _make_model_callable(packed_output)
        forward_fn = model_forward_gen(vision_model=False)
        result = forward_fn(
            model,
            input_ids,
            attention_mask,
            position_ids,
            multi_modal_inputs={},
            data_format="thd",
            value_model=True,
        )

        # [batch, seq_len, 1] -> [batch, seq_len] via output[..., 0]
        assert result.shape == (batch, seq_len)


# ---------------------------------------------------------------------------
# BSHD format tests
# ---------------------------------------------------------------------------


class TestModelForwardBshd:
    """Tests for data_format='bshd' path through model_forward_gen()."""

    def test_bshd_format_basic(self, mock_env):
        """Basic BSHD: all-valid right-aligned tokens, no compaction needed."""
        from axon.models.mcore.forward.model_forward import model_forward_gen

        inner_model, _ = mock_env
        batch, seq_len, hidden = 1, 6, 8

        input_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
        attention_mask = torch.ones(batch, seq_len, dtype=torch.bool)
        position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)

        # Model receives compacted input; with all-valid mask, compacted_len == seq_len
        compacted_output = torch.randn(batch, seq_len, hidden)
        model = _make_model_callable(compacted_output)

        forward_fn = model_forward_gen(vision_model=False)
        result = forward_fn(
            model,
            input_ids,
            attention_mask,
            position_ids,
            multi_modal_inputs={},
            data_format="bshd",
        )

        model.assert_called_once()
        assert result.shape == (batch, seq_len, hidden)

    def test_bshd_with_left_padding(self, mock_env):
        """BSHD with left-padding: output preserves original padded layout."""
        from axon.models.mcore.forward.model_forward import model_forward_gen

        inner_model, _ = mock_env
        batch, seq_len, hidden = 1, 8, 4
        valid_len = 5
        pad_len = seq_len - valid_len

        input_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
        # Left-padded: first 3 positions are padding
        attention_mask = torch.zeros(batch, seq_len, dtype=torch.bool)
        attention_mask[0, pad_len:] = True
        position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)

        # After preprocess_bshd, the compacted tensor has shape [batch, valid_len, ...]
        compacted_output = torch.ones(batch, valid_len, hidden) * 5.0
        model = _make_model_callable(compacted_output)

        forward_fn = model_forward_gen(vision_model=False)
        result = forward_fn(
            model,
            input_ids,
            attention_mask,
            position_ids,
            multi_modal_inputs={},
            data_format="bshd",
        )

        assert result.shape == (batch, seq_len, hidden)
        # Left-pad positions should be zeros
        assert (result[0, :pad_len] == 0).all()
        # Valid positions should have been filled
        assert (result[0, pad_len:] != 0).any()

    def test_bshd_with_logits_processor(self, mock_env):
        """BSHD with logits_processor: dict output, each value postprocessed."""
        from axon.models.mcore.forward.model_forward import model_forward_gen

        inner_model, _ = mock_env
        batch, seq_len, hidden = 1, 6, 8

        input_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
        attention_mask = torch.ones(batch, seq_len, dtype=torch.bool)
        position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)

        compacted_output = torch.randn(batch, seq_len, hidden)
        model = _make_model_callable(compacted_output)

        def fake_logits_processor(output, **kwargs):
            return {
                "log_probs": output[..., 0],  # [batch, compacted_len]
                "entropy": output[..., 1],  # [batch, compacted_len]
            }

        lp_args = {
            "labels": torch.randint(0, 100, (batch, seq_len), dtype=torch.long),
        }

        forward_fn = model_forward_gen(vision_model=False)
        result = forward_fn(
            model,
            input_ids,
            attention_mask,
            position_ids,
            multi_modal_inputs={},
            logits_processor=fake_logits_processor,
            logits_processor_args=lp_args,
            data_format="bshd",
        )

        assert isinstance(result, dict)
        assert "log_probs" in result
        assert "entropy" in result
        assert result["log_probs"].shape == (batch, seq_len)
        assert result["entropy"].shape == (batch, seq_len)


# ---------------------------------------------------------------------------
# Edge-case tests
# ---------------------------------------------------------------------------


class TestModelForwardEdgeCases:
    """Edge cases: invalid format, pre/post_process flags, multi-modal inputs."""

    def test_invalid_data_format_raises(self, mock_env):
        """data_format='invalid' should raise AssertionError."""
        from axon.models.mcore.forward.model_forward import model_forward_gen

        inner_model, _ = mock_env
        batch, seq_len = 1, 4
        input_ids = torch.zeros(batch, seq_len, dtype=torch.long)
        attention_mask = torch.ones(batch, seq_len, dtype=torch.bool)
        position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)

        model = _make_model_callable(torch.zeros(1))
        forward_fn = model_forward_gen(vision_model=False)

        with pytest.raises(AssertionError, match="data_format must be"):
            forward_fn(
                model,
                input_ids,
                attention_mask,
                position_ids,
                multi_modal_inputs={},
                data_format="invalid",
            )

    def test_pre_process_false(self, mock_env):
        """When pre_process=False, input_ids are passed through unmodified to packing."""
        from axon.models.mcore.forward.model_forward import model_forward_gen

        inner_model, _ = mock_env
        inner_model.pre_process = False

        batch, seq_len, hidden = 1, 6, 4
        input_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
        attention_mask = torch.ones(batch, seq_len, dtype=torch.bool)
        position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)

        # When pre_process=False, preprocess_packed_seqs returns original input_ids
        # (no packing), so model gets the original tensor
        packed_output = torch.randn(1, seq_len, hidden)
        model = _make_model_callable(packed_output)

        forward_fn = model_forward_gen(vision_model=False)
        forward_fn(
            model,
            input_ids,
            attention_mask,
            position_ids,
            multi_modal_inputs={},
            data_format="thd",
        )

        model.assert_called_once()
        # With pre_process=False, the input_ids passed to model should be the
        # original (not packed) -- verify via the call args
        call_kwargs = model.call_args.kwargs
        passed_ids = call_kwargs["input_ids"]
        # pre_process=False means preprocess_packed_seqs returns input_ids as-is
        assert passed_ids.shape == input_ids.shape

    def test_post_process_false(self, mock_env):
        """When post_process=False, raw model output is returned without unpacking."""
        from axon.models.mcore.forward.model_forward import model_forward_gen

        inner_model, _ = mock_env
        inner_model.post_process = False

        batch, seq_len, hidden = 1, 6, 4
        input_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
        attention_mask = torch.ones(batch, seq_len, dtype=torch.bool)
        position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)

        raw_output = torch.randn(1, seq_len, hidden)
        model = _make_model_callable(raw_output)

        forward_fn = model_forward_gen(vision_model=False)
        result = forward_fn(
            model,
            input_ids,
            attention_mask,
            position_ids,
            multi_modal_inputs={},
            data_format="thd",
        )

        # postprocess_packed_seqs with post_process=False returns output unchanged
        assert result is raw_output

    def test_multi_modal_inputs(self, mock_env):
        """pixel_values and image_grid_thw should be forwarded to model kwargs."""
        from axon.models.mcore.forward.model_forward import model_forward_gen

        inner_model, _ = mock_env
        batch, seq_len, hidden = 1, 6, 4

        input_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
        attention_mask = torch.ones(batch, seq_len, dtype=torch.bool)
        position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)

        pixel_values = torch.randn(1, 3, 224, 224)
        image_grid_thw = torch.tensor([[1, 14, 14]])

        packed_output = torch.randn(1, seq_len, hidden)
        model = _make_model_callable(packed_output)

        forward_fn = model_forward_gen(vision_model=False)
        forward_fn(
            model,
            input_ids,
            attention_mask,
            position_ids,
            multi_modal_inputs={
                "pixel_values": pixel_values,
                "image_grid_thw": image_grid_thw,
            },
            data_format="thd",
        )

        call_kwargs = model.call_args.kwargs
        assert "pixel_values" in call_kwargs
        assert "image_grid_thw" in call_kwargs
        assert torch.equal(call_kwargs["pixel_values"], pixel_values)
        assert torch.equal(call_kwargs["image_grid_thw"], image_grid_thw)


# ---------------------------------------------------------------------------
# Fused forward patching tests
# ---------------------------------------------------------------------------


# A custom metaclass that lets us control isinstance() checks via a class-level
# set of "registered instances".
class _InstanceCheckMeta(type):
    """Metaclass whose isinstance() delegates to a class-level _instances set."""

    def __instancecheck__(cls, instance):
        return instance in getattr(cls, "_instances", set())


# A stand-in for GPTModel that we can control isinstance() checks on.
class _FakeGPTModel(metaclass=_InstanceCheckMeta):
    _instances: set = set()


class TestFusedForwardPatching:
    """Tests for _get_patching_model, patch_fused_forward, unpatch_fused_forward."""

    def test_get_patching_model_with_gpt_model(self):
        """A GPTModel instance is returned directly."""
        from unittest.mock import patch as mock_patch

        mock_gpt = MagicMock()
        _FakeGPTModel._instances = {mock_gpt}

        with (
            mock_patch(
                "axon.models.mcore.forward.model_forward_fused.unwrap_model",
                return_value=mock_gpt,
            ),
            mock_patch(
                "axon.models.mcore.forward.model_forward_fused.GPTModel",
                _FakeGPTModel,
            ),
        ):
            from axon.models.mcore.forward.model_forward_fused import _get_patching_model

            result = _get_patching_model(mock_gpt)
            assert result is mock_gpt

        _FakeGPTModel._instances = set()

    def test_get_patching_model_with_language_model(self):
        """Model with .language_model that is a GPTModel returns language_model."""
        from unittest.mock import patch as mock_patch

        mock_language_model = MagicMock()
        mock_outer = MagicMock()
        mock_outer.language_model = mock_language_model
        # Only language_model passes isinstance check
        _FakeGPTModel._instances = {mock_language_model}

        with (
            mock_patch(
                "axon.models.mcore.forward.model_forward_fused.unwrap_model",
                return_value=mock_outer,
            ),
            mock_patch(
                "axon.models.mcore.forward.model_forward_fused.GPTModel",
                _FakeGPTModel,
            ),
        ):
            from axon.models.mcore.forward.model_forward_fused import _get_patching_model

            result = _get_patching_model(mock_outer)
            assert result is mock_language_model

        _FakeGPTModel._instances = set()

    def test_get_patching_model_unsupported(self, capsys):
        """Non-GPTModel without .language_model returns None."""
        from unittest.mock import patch as mock_patch

        mock_model = MagicMock(spec=[])  # no language_model attribute
        _FakeGPTModel._instances = set()  # nothing passes isinstance

        with (
            mock_patch(
                "axon.models.mcore.forward.model_forward_fused.unwrap_model",
                return_value=mock_model,
            ),
            mock_patch(
                "axon.models.mcore.forward.model_forward_fused.GPTModel",
                _FakeGPTModel,
            ),
        ):
            from axon.models.mcore.forward.model_forward_fused import _get_patching_model

            result = _get_patching_model(mock_model)
            assert result is None
            captured = capsys.readouterr()
            assert "not a supported" in captured.out

    def test_patch_unpatch_roundtrip(self):
        """patch_fused_forward replaces forward; unpatch_fused_forward restores it."""
        from unittest.mock import patch as mock_patch

        original_forward = MagicMock(name="original_forward")
        mock_gpt = MagicMock()
        mock_gpt.forward = original_forward
        mock_gpt.__class__ = type("FakeGPT", (), {})
        _FakeGPTModel._instances = {mock_gpt}

        with (
            mock_patch(
                "axon.models.mcore.forward.model_forward_fused.unwrap_model",
                return_value=mock_gpt,
            ),
            mock_patch(
                "axon.models.mcore.forward.model_forward_fused.GPTModel",
                _FakeGPTModel,
            ),
            mock_patch(
                "axon.models.mcore.forward.model_forward_fused.mcore",
            ) as mock_mcore,
        ):
            # Satisfy version check
            mock_mcore.__version__ = "0.13.0"

            from axon.models.mcore.forward.model_forward_fused import (
                patch_fused_forward,
                unpatch_fused_forward,
            )

            # Patch: forward should be replaced and backup saved
            patch_fused_forward(mock_gpt)
            assert mock_gpt.forward_backup is original_forward
            assert mock_gpt.forward is not original_forward

            # Unpatch: forward should be restored
            unpatch_fused_forward(mock_gpt)
            assert mock_gpt.forward is original_forward

        _FakeGPTModel._instances = set()
