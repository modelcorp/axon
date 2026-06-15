"""
Tests for policy loss functions and compute_loss_fn.
"""

import torch

from axon.protocol import DataProto
from axon.trainer.algos.loss.registry import (
    LossFn,
    compute_loss_fn,
    get_loss_entry,
    get_loss_fn,
)


class TestLossRegistration:
    """Domain-specific registration tests (all enum values wired up correctly)."""

    def test_all_loss_fns_registered(self):
        import axon.trainer.algos.loss.loss  # noqa: F401

        for fn_enum in LossFn:
            entry = get_loss_entry(fn_enum)
            assert entry is not None
            assert callable(entry.fn)

    def test_ppo_resolves_to_correct_fn(self):
        from axon.trainer.algos.loss.loss import ppo_loss_fn

        assert get_loss_fn(LossFn.PPO) is ppo_loss_fn


class TestComputeLossFn:
    def _make_data(self, **extra_tensors):
        batch_size, seq_len = 2, 4
        tensors = {
            "old_log_probs": torch.randn(batch_size, seq_len),
            "log_probs": torch.randn(batch_size, seq_len),
            "advantages": torch.randn(batch_size, seq_len),
            "response_mask": torch.ones(batch_size, seq_len),
        }
        tensors.update(extra_tensors)
        return DataProto.from_dict(tensors=tensors)

    def test_compute_loss_fn_ppo(self):
        data = self._make_data()
        loss, metrics = compute_loss_fn(data, loss_fn="ppo")
        assert loss.shape == ()
        assert "pg_clipfrac" in metrics

    def test_compute_loss_fn_with_args(self):
        data = self._make_data()
        loss, metrics = compute_loss_fn(data, loss_fn="ppo", loss_fn_args={"clip_ratio": 0.1})
        assert loss.shape == ()

    def test_compute_loss_fn_none_args(self):
        data = self._make_data()
        loss, metrics = compute_loss_fn(data, loss_fn="gpg", loss_fn_args=None)
        assert loss.shape == ()

    def _make_value_data(self):
        batch_size, seq_len = 2, 4
        tensors = {
            "vpreds": torch.randn(batch_size, seq_len),
            "values": torch.randn(batch_size, seq_len),
            "returns": torch.randn(batch_size, seq_len),
            "response_mask": torch.ones(batch_size, seq_len),
        }
        return DataProto.from_dict(tensors=tensors)

    def test_compute_loss_fn_all_registered(self):
        """Every enum value should be callable via compute_loss_fn."""
        actor_data = self._make_data()
        value_data = self._make_value_data()
        for fn_enum in LossFn:
            data = value_data if fn_enum == LossFn.VALUE else actor_data
            loss, metrics = compute_loss_fn(data, loss_fn=fn_enum.value)
            assert loss.shape == (), f"{fn_enum.value} did not return scalar loss"
            assert isinstance(metrics, dict)
