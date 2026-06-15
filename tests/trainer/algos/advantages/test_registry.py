"""
Tests for advantage estimator registration and functions.
"""

import numpy as np
import torch
from omegaconf import OmegaConf

from axon.protocol import DataProto
from axon.trainer.algos.advantages.registry import (
    ADV_REGISTRY,
    AdvantageFn,
    get_advantage_fn,
)


class TestAdvantageRegistration:
    """Domain-specific registration tests (all enum values wired up correctly)."""

    def test_all_advantage_fns_registered(self):
        import axon.trainer.algos.advantages.advantage  # noqa: F401

        for fn_enum in AdvantageFn:
            assert fn_enum.value in ADV_REGISTRY, f"{fn_enum.value} not registered"

    def test_grpo_resolves_to_correct_fn(self):
        from axon.trainer.algos.advantages.advantage import grpo_advantage_fn

        assert get_advantage_fn(AdvantageFn.GRPO) is grpo_advantage_fn

    def test_grpo_callable_with_data_config(self):
        from axon.trainer.algos.advantages.advantage import grpo_advantage_fn

        rewards = torch.tensor(
            [
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0],
            ]
        )
        mask = torch.ones_like(rewards)
        index = np.array(["a", "a"])

        data = DataProto.from_dict(
            tensors={"token_level_rewards": rewards, "response_mask": mask},
            non_tensors={"uid": index},
        )
        config = OmegaConf.create({})

        advantages, returns = grpo_advantage_fn(data, config)
        assert advantages.shape == rewards.shape
        assert returns.shape == rewards.shape
