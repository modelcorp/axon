"""CPU-only tests for axon.models.fsdp_models key-selection, config, and create_model logic.

Tests the static methods ``forward_keys``, ``forward_backward_keys``, ``create_model``,
and ``_FWD_KWARGS`` for CausalLM, ValueModel, and RewardModel without requiring
a GPU or loading real model weights.

Usage:
    pytest tests/models/test_fsdp_models.py -v
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from axon.models.fsdp_models import CausalLM, RewardModel, ValueModel

# ---------------------------------------------------------------------------
# Helper: lightweight DataProto-like object
# ---------------------------------------------------------------------------


def _make_data(batch=None, non_tensor_batch=None):
    """Return a SimpleNamespace mimicking DataProto with .batch and .non_tensor_batch."""
    return SimpleNamespace(
        batch=batch or {},
        non_tensor_batch=non_tensor_batch or {},
    )


# ---------------------------------------------------------------------------
# forward_keys: parametrized across CausalLM / ValueModel / RewardModel
# (They share IDENTICAL logic)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model_cls", [CausalLM, ValueModel, RewardModel])
class TestForwardKeysParametrized:
    """forward_keys returns the same results for all three model classes."""

    def test_basic_keys(self, model_cls):
        data = _make_data(
            batch={"input_ids": 1, "attention_mask": 1, "position_ids": 1},
        )
        batch_keys, non_tensor_keys = model_cls.forward_keys(data)
        assert batch_keys == ["input_ids", "attention_mask", "position_ids"]
        assert non_tensor_keys == []

    def test_with_response_mask(self, model_cls):
        data = _make_data(
            batch={"input_ids": 1, "attention_mask": 1, "position_ids": 1, "response_mask": 1},
        )
        batch_keys, non_tensor_keys = model_cls.forward_keys(data)
        assert "response_mask" in batch_keys
        assert non_tensor_keys == []

    def test_with_multi_modal_inputs(self, model_cls):
        data = _make_data(
            batch={"input_ids": 1, "attention_mask": 1, "position_ids": 1},
            non_tensor_batch={"multi_modal_inputs": "something"},
        )
        batch_keys, non_tensor_keys = model_cls.forward_keys(data)
        assert batch_keys == ["input_ids", "attention_mask", "position_ids"]
        assert non_tensor_keys == ["multi_modal_inputs"]


# ---------------------------------------------------------------------------
# CausalLM.forward_backward_keys
# ---------------------------------------------------------------------------


class TestCausalLMForwardBackwardKeys:
    _REQUIRED = [
        "input_ids",
        "attention_mask",
        "position_ids",
        "response_mask",
        "old_log_probs",
        "advantages",
    ]

    def test_basic_keys_no_optionals(self):
        data = _make_data(batch={k: 1 for k in self._REQUIRED})
        batch_keys, non_tensor_keys = CausalLM.forward_backward_keys(data)
        assert batch_keys == self._REQUIRED
        assert non_tensor_keys is None

    def test_with_optional_batch_keys(self):
        extra = {"num_program_tokens": 1, "ref_log_prob": 1}
        data = _make_data(batch={**{k: 1 for k in self._REQUIRED}, **extra})
        batch_keys, non_tensor_keys = CausalLM.forward_backward_keys(data)
        assert "num_program_tokens" in batch_keys
        assert "ref_log_prob" in batch_keys
        assert "sampler_log_probs" not in batch_keys
        assert non_tensor_keys is None

    def test_optional_keys_appended_in_order(self):
        """Required keys come first, then optional keys in _optional_batch tuple order.

        The code iterates _optional_batch in declaration order, so the output
        should be: required + [present optionals in that same order].
        """
        # Provide optionals in a different insertion order than _optional_batch
        all_optional_in_order = [
            "num_program_tokens",
            "sampler_log_probs",
            "ref_log_prob",
            "valid_batch_size",
            "valid_token_count",
            "valid_program_count",
            "sampler_is_weights",
        ]
        # Provide them in reverse order in the dict
        data = _make_data(
            batch={**{k: 1 for k in self._REQUIRED}, **{k: 1 for k in reversed(all_optional_in_order)}},
        )
        batch_keys, _ = CausalLM.forward_backward_keys(data)
        # Required keys first
        assert batch_keys[: len(self._REQUIRED)] == self._REQUIRED
        # Then optional keys in _optional_batch declaration order
        assert batch_keys[len(self._REQUIRED) :] == all_optional_in_order

    def test_all_optional_plus_all_non_tensor(self):
        """Complete picture: all optional batch keys + all non_tensor keys present."""
        all_optional = [
            "num_program_tokens",
            "sampler_log_probs",
            "ref_log_prob",
            "valid_batch_size",
            "valid_token_count",
            "valid_program_count",
            "sampler_is_weights",
        ]
        data = _make_data(
            batch={**{k: 1 for k in self._REQUIRED}, **{k: 1 for k in all_optional}},
            non_tensor_batch={"multi_modal_inputs": "img", "num_program_steps": 3},
        )
        batch_keys, non_tensor_keys = CausalLM.forward_backward_keys(data)
        for opt in all_optional:
            assert opt in batch_keys
        assert non_tensor_keys == ["multi_modal_inputs", "num_program_steps"]

    def test_with_non_tensor_batch_keys(self):
        data = _make_data(
            batch={k: 1 for k in self._REQUIRED},
            non_tensor_batch={"multi_modal_inputs": "img", "num_program_steps": 3},
        )
        batch_keys, non_tensor_keys = CausalLM.forward_backward_keys(data)
        assert non_tensor_keys == ["multi_modal_inputs", "num_program_steps"]

    def test_returns_none_when_no_non_tensor_keys_match(self):
        data = _make_data(
            batch={k: 1 for k in self._REQUIRED},
            non_tensor_batch={"unrelated_key": "value"},
        )
        _, non_tensor_keys = CausalLM.forward_backward_keys(data)
        assert non_tensor_keys is None


# ---------------------------------------------------------------------------
# CausalLM._FWD_KWARGS
# ---------------------------------------------------------------------------


class TestCausalLMFwdKwargs:
    def test_expected_keys_present(self):
        expected = {
            "use_remove_padding",
            "use_fused_kernels",
            "ulysses_sp_size",
            "device_name",
            "param_dtype",
            "entropy_checkpointing",
            "entropy_from_logits_with_chunking",
            "use_torch_compile",
        }
        assert isinstance(CausalLM._FWD_KWARGS, tuple)
        assert set(CausalLM._FWD_KWARGS) == expected


# ---------------------------------------------------------------------------
# ValueModel.forward_backward_keys
# ---------------------------------------------------------------------------


class TestValueModelForwardBackwardKeys:
    _EXPECTED = ["input_ids", "response_mask", "attention_mask", "position_ids", "values", "returns"]

    def test_basic_keys(self):
        data = _make_data(batch={k: 1 for k in self._EXPECTED})
        batch_keys, non_tensor_keys = ValueModel.forward_backward_keys(data)
        assert batch_keys == self._EXPECTED
        assert non_tensor_keys is None

    def test_with_multi_modal_inputs(self):
        data = _make_data(
            batch={k: 1 for k in self._EXPECTED},
            non_tensor_batch={"multi_modal_inputs": "img"},
        )
        batch_keys, non_tensor_keys = ValueModel.forward_backward_keys(data)
        assert batch_keys == self._EXPECTED
        assert non_tensor_keys == ["multi_modal_inputs"]

    def test_returns_none_without_multi_modal(self):
        data = _make_data(
            batch={k: 1 for k in self._EXPECTED},
            non_tensor_batch={"other_key": "value"},
        )
        _, non_tensor_keys = ValueModel.forward_backward_keys(data)
        assert non_tensor_keys is None


# ---------------------------------------------------------------------------
# RewardModel.forward_backward_keys / forward_backward_fn raise NotImplementedError
# ---------------------------------------------------------------------------


class TestRewardModelNotImplemented:
    def test_forward_backward_keys_raises(self):
        data = _make_data()
        with pytest.raises(NotImplementedError, match="inference-only"):
            RewardModel.forward_backward_keys(data)

    def test_forward_backward_fn_raises(self):
        with pytest.raises(NotImplementedError, match="inference-only"):
            RewardModel.forward_backward_fn(None, None, None)


# ---------------------------------------------------------------------------
# ValueModel.create_model config mutations
# ---------------------------------------------------------------------------


class TestValueModelCreateModelConfig:
    def test_create_model_sets_config_fields(self):
        config = SimpleNamespace()
        with patch("axon.models.fsdp_models.load_valuehead_model") as mock_load:
            mock_load.return_value = MagicMock()
            ValueModel.create_model(
                config,
                model_path="dummy",
                torch_dtype="float32",
                trust_remote_code=False,
            )
        assert config.num_labels == 1
        assert config.classifier_dropout == 0.0
        assert config.hidden_dropout == 0.0
        assert config.summary_dropout_prob == 0.0

    def test_create_model_calls_load_valuehead_model(self):
        config = SimpleNamespace()
        with patch("axon.models.fsdp_models.load_valuehead_model") as mock_load:
            mock_load.return_value = MagicMock()
            result = ValueModel.create_model(
                config,
                model_path="/some/path",
                torch_dtype="bfloat16",
                trust_remote_code=True,
            )
        mock_load.assert_called_once_with("/some/path", "bfloat16", config, True)
        assert result is mock_load.return_value


# ---------------------------------------------------------------------------
# RewardModel.create_model config mutations
# ---------------------------------------------------------------------------


class TestRewardModelCreateModelConfig:
    def test_create_model_sets_num_labels_and_classifier_dropout(self):
        """RewardModel.create_model must set num_labels=1 and classifier_dropout=0.0."""
        config = SimpleNamespace(architectures=["SomeModel"])
        with patch.object(
            __import__("transformers", fromlist=["AutoModelForTokenClassification"]).AutoModelForTokenClassification,
            "from_pretrained",
            return_value=MagicMock(),
        ) as mock_fp:
            RewardModel.create_model(
                config,
                model_path="dummy",
                torch_dtype="float32",
                trust_remote_code=False,
            )
        assert config.num_labels == 1
        assert config.classifier_dropout == 0.0
        mock_fp.assert_called_once()

    def test_create_model_uses_bfloat16_regardless_of_dtype(self):
        """RewardModel always passes torch.bfloat16, ignoring kwargs['torch_dtype']."""
        import torch

        config = SimpleNamespace(architectures=["SomeModel"])
        with patch.object(
            __import__("transformers", fromlist=["AutoModelForTokenClassification"]).AutoModelForTokenClassification,
            "from_pretrained",
            return_value=MagicMock(),
        ) as mock_fp:
            RewardModel.create_model(
                config,
                model_path="dummy",
                torch_dtype="float32",  # NOT bfloat16
                trust_remote_code=False,
            )
        call_kwargs = mock_fp.call_args[1]
        assert call_kwargs["torch_dtype"] is torch.bfloat16


# ---------------------------------------------------------------------------
# CausalLM.create_model class selection logic
# ---------------------------------------------------------------------------


class TestCausalLMCreateModel:
    """Test the auto_class / module_class selection logic in CausalLM.create_model."""

    def test_auto_map_causal_lm(self):
        """auto_map with 'AutoModelForCausalLM' -> uses AutoModelForCausalLM."""
        from transformers import AutoModelForCausalLM as AMCLM

        config = SimpleNamespace(
            architectures=["TestModel"],
            auto_map={"AutoModelForCausalLM": "test_module.TestModel"},
        )
        with patch.object(AMCLM, "from_pretrained", return_value=MagicMock()) as mock_fp:
            CausalLM.create_model(config, model_path="dummy", torch_dtype="float32", trust_remote_code=True)
        mock_fp.assert_called_once()

    def test_auto_map_vision2seq(self):
        """auto_map with 'AutoModelForVision2Seq' -> uses AutoModelForVision2Seq.

        In transformers 5.x ``AutoModelForVision2Seq`` was removed and
        ``axon.models.fsdp_models`` aliases it to ``AutoModelForImageTextToText``;
        we import from there to stay compatible with both transformers majors.
        """
        from axon.models.fsdp_models import AutoModelForVision2Seq as AMV2S

        config = SimpleNamespace(
            architectures=["TestVisionModel"],
            auto_map={"AutoModelForVision2Seq": "test_module.TestVisionModel"},
        )
        with patch.object(AMV2S, "from_pretrained", return_value=MagicMock()) as mock_fp:
            CausalLM.create_model(config, model_path="dummy", torch_dtype="float32", trust_remote_code=True)
        mock_fp.assert_called_once()

    def test_auto_map_fallback_to_auto_model(self):
        """auto_map with unknown class name -> falls back to AutoModel."""
        from transformers import AutoModel as AM

        config = SimpleNamespace(
            architectures=["CustomModel"],
            auto_map={"AutoModelForSomethingElse": "test_module.CustomModel"},
        )
        with patch.object(AM, "from_pretrained", return_value=MagicMock()) as mock_fp:
            CausalLM.create_model(config, model_path="dummy", torch_dtype="float32", trust_remote_code=True)
        mock_fp.assert_called_once()

    def test_no_auto_map_model_mapping_causal_lm(self):
        """No auto_map, config type in AutoModelForCausalLM._model_mapping -> uses AutoModelForCausalLM."""
        from transformers import AutoModelForCausalLM as AMCLM

        from axon.models.fsdp_models import AutoModelForVision2Seq as AMV2S

        # Create a config type that is NOT in Vision2Seq mapping but IS in CausalLM mapping
        config = SimpleNamespace(architectures=["TestModel"])
        # No auto_map attribute at all -> hasattr check fails
        # Patch the _model_mapping.keys() to include our config type
        with (
            patch.object(AMV2S._model_mapping, "keys", return_value=[]),
            patch.object(AMCLM._model_mapping, "keys", return_value=[type(config)]),
            patch.object(AMCLM, "from_pretrained", return_value=MagicMock()) as mock_fp,
        ):
            CausalLM.create_model(config, model_path="dummy", torch_dtype="float32", trust_remote_code=False)
        mock_fp.assert_called_once()

    def test_no_auto_map_nothing_matches_falls_back_to_auto_model(self):
        """No auto_map, config type not in any mapping -> falls back to AutoModel."""
        from transformers import AutoModel as AM
        from transformers import AutoModelForCausalLM as AMCLM
        from transformers import AutoModelForImageTextToText as AMITT

        from axon.models.fsdp_models import AutoModelForVision2Seq as AMV2S

        config = SimpleNamespace(architectures=["ObscureModel"])

        with (
            patch.object(AMV2S._model_mapping, "keys", return_value=[]),
            patch.object(AMCLM._model_mapping, "keys", return_value=[]),
            patch.object(AMITT._model_mapping, "keys", return_value=[]),
            patch.object(AM, "from_pretrained", return_value=MagicMock()) as mock_fp,
        ):
            CausalLM.create_model(config, model_path="dummy", torch_dtype="float32", trust_remote_code=False)
        mock_fp.assert_called_once()
