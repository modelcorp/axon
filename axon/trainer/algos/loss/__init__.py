# Copyright 2025 Model AI Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Policy loss module for reinforcement learning algorithms.
"""

# Registry
# Import loss functions to register them
from axon.trainer.algos.loss import loss as _loss  # noqa: F401
from axon.trainer.algos.loss.registry import (
    LossFn,
    LossRegistryEntry,
    compute_loss_fn,
    get_loss_entry,
    get_loss_fn,
    register_loss,
)

# Utils
from axon.trainer.algos.loss.utils import (
    agg_loss,
    clip_by_value,
    entropy_from_logits,
    masked_mean,
)

__all__ = [
    # Registry
    "LossFn",
    "LossRegistryEntry",
    "register_loss",
    "get_loss_entry",
    "get_loss_fn",
    "compute_loss_fn",
    # Utils
    "agg_loss",
    "clip_by_value",
    "entropy_from_logits",
    "masked_mean",
]
