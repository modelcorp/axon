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
Metrics utilities for Axon training.
"""

from axon.utils.metrics.data_metrics import (
    compute_data_metrics,
    compute_response_mask,
    reduce_data_metrics,
)
from axon.utils.metrics.reduce_metrics import reduce_metrics
from axon.utils.metrics.repetition import (
    compute_repetition_metrics,
    has_repetition,
)
from axon.utils.metrics.timing import (
    compute_timing_metrics,
    marked_timer,
    reduce_timing_metrics,
)
from axon.utils.metrics.trainer_sampler_mismatch_metrics import (
    aggregate_trainer_sampler_metrics,
    compute_trainer_sampler_mismatch_metrics,
)

__all__ = [
    # Data metrics
    "compute_data_metrics",
    "compute_response_mask",
    "reduce_data_metrics",
    # Repetition detection
    "compute_repetition_metrics",
    "has_repetition",
    # Timing metrics
    "compute_timing_metrics",
    "reduce_timing_metrics",
    "marked_timer",
    # Reduce metrics
    "reduce_metrics",
    # Trainer-sampler mismatch metrics
    "compute_trainer_sampler_mismatch_metrics",
    "aggregate_trainer_sampler_metrics",
]
