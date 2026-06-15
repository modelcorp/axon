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
import math
import random
from collections.abc import Sized

import numpy as np

from axon.protocol import DataProto

from axon.data.data_sampler.abstract_sampler import AbstractSampler

class ExpWeightedCurriculumSampler(AbstractSampler):
    def __init__(self, data_source: Sized, min_weight: float = 0.2, max_weight: float = 1.0):
        self.data_source = data_source
        self.num_samples = len(data_source)
        self.weights = [1.0] * self.num_samples  # start with uniform weights
        # Config to determine range
        self.centroid = 0.0
        self.min_weight = min_weight
        self.max_weight = max_weight

    def __iter__(self):
        # Normalize weights
        total = sum(self.weights)
        if total == 0:
            probs = [1.0 / self.num_samples] * self.num_samples
        else:
            probs = [w / total for w in self.weights]
        # Sample indices with replacement
        return iter(random.choices(range(self.num_samples), weights=probs, k=self.num_samples))

    def __len__(self):
        return self.num_samples

    def update(self, batch: DataProto) -> None:
        # Assumes batch contains fields: batch['indices'] and batch['response']['k']
        indices = np.asarray(batch.non_tensor_batch["index"], dtype=np.int64)
        pass_rate = batch.non_tensor_batch["pass_rate"].tolist()  # List[float] or tensor-like

        for i, k in zip(indices, pass_rate, strict=False):
            if k > 0.8:
                self.weights[i] = self.min_weight
            elif k < 0.2:
                self.weights[i] = self.max_weight
            else:
                self.weights[i] = math.exp(-k + self.centroid)  # Update weight

    def state_dict(self) -> dict:
        """Return everything needed to restore sampler state."""
        return {
            "weights": self.weights,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore sampler state from a dict created by state_dict()."""
        self.weights = state["weights"]


class ThresholdMaskingSampler(AbstractSampler):
    """Sampler that permanently masks samples once avg pass_rate exceeds a threshold.
    
    Samples uniformly from all unmasked samples. Once a sample's average pass_rate
    (across all invocations) >= threshold, it is permanently excluded.
    """
    
    def __init__(self, data_source: Sized, threshold: float = 0.9):
        self.data_source = data_source
        self.num_samples = len(data_source)
        self.threshold = threshold
        # Use numpy arrays for efficiency
        self.pass_sum = np.zeros(self.num_samples, dtype=np.float32)
        self.invoke_count = np.zeros(self.num_samples, dtype=np.int32)
        self.masked = np.zeros(self.num_samples, dtype=bool)

    def __iter__(self):
        active_indices = np.where(~self.masked)[0]
        if len(active_indices) == 0:
            active_indices = np.arange(self.num_samples, dtype=np.int64)
        sampled = np.random.choice(active_indices, size=self.num_samples, replace=True)
        return iter(sampled.tolist())

    def __len__(self):
        return self.num_samples

    def update(self, batch: DataProto) -> None:
        indices = np.asarray(batch.non_tensor_batch["index"], dtype=np.int64)
        pass_rate = np.asarray(batch.non_tensor_batch["pass_rate"], dtype=np.float32)

        # Use np.add.at to properly handle duplicate indices
        np.add.at(self.pass_sum, indices, pass_rate)
        np.add.at(self.invoke_count, indices, 1)
        
        # Compute avg and check threshold for updated indices
        avg = self.pass_sum[indices] / self.invoke_count[indices]
        newly_masked = avg >= self.threshold
        if newly_masked.any():
            self.masked[indices[newly_masked]] = True

    def state_dict(self) -> dict:
        return {
            "masked": self.masked.tolist(),
            "pass_sum": self.pass_sum.tolist(),
            "invoke_count": self.invoke_count.tolist(),
        }

    def load_state_dict(self, state: dict) -> None:
        self.masked = np.array(state["masked"], dtype=bool)
        self.pass_sum = np.array(state["pass_sum"], dtype=np.float32)
        self.invoke_count = np.array(state["invoke_count"], dtype=np.int32)
