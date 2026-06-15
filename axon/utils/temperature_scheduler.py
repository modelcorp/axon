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
import logging

import numpy as np

logger = logging.getLogger(__name__)


class TemperatureScheduler:
    """Temperature scheduler for controlling sampling randomness during training."""

    def __init__(self, config: dict):
        """Initialize temperature scheduler.

        Args:
            config: Temperature schedule configuration with keys:
                - enable: Whether to enable temperature scheduling
                - scheduler: Type of scheduler ('linear', 'exponential', 'cosine')
                - start_temperature: Initial temperature value
                - end_temperature: Final temperature value
                - num_steps: Number of training steps over which to apply the schedule
        """
        self.enabled = config.get("enable", False)
        self.scheduler_type = config.get("scheduler", "linear")
        self.start_temperature = config.get("start_temperature", 1.0)
        self.end_temperature = config.get("end_temperature", 1.0)
        self.num_steps = config.get("num_steps", 1000)
        self.current_step = 0

        self._validate_config()

        if self.enabled:
            logger.info(
                f"Temperature scheduler enabled: {self.scheduler_type} from {self.start_temperature} to {self.end_temperature} over {self.num_steps} steps"
            )

    def _validate_config(self):
        """Validate scheduler configuration parameters."""
        if self.start_temperature <= 0:
            raise ValueError("Start temperature must be positive.")
        if self.end_temperature <= 0:
            raise ValueError("End temperature must be positive.")
        if self.scheduler_type not in ["linear", "exponential", "cosine"]:
            raise ValueError(f"Invalid scheduler type: {self.scheduler_type}")
        if self.num_steps <= 0:
            raise ValueError("Number of steps must be positive.")

    def get_temperature(self, step: int = None) -> float:
        """Get the temperature for the current step.

        Args:
            step: Optional step number. If None, uses internal counter.

        Returns:
            Temperature value for the current step.
        """
        if not self.enabled:
            return self.start_temperature

        current_step = step if step is not None else self.current_step
        progress = min(current_step / self.num_steps, 1.0)

        return self._compute_temperature(progress)

    def _compute_temperature(self, progress: float) -> float:
        """Compute temperature based on progress and scheduler type.

        Args:
            progress: Training progress as a float between 0 and 1.

        Returns:
            Computed temperature value.
        """
        if self.scheduler_type == "linear":
            return self._linear_schedule(progress)
        elif self.scheduler_type == "exponential":
            return self._exponential_schedule(progress)
        elif self.scheduler_type == "cosine":
            return self._cosine_schedule(progress)
        else:
            logger.warning(f"Unknown scheduler type: {self.scheduler_type}, using linear")
            return self._linear_schedule(progress)

    def _linear_schedule(self, progress: float) -> float:
        """Linear interpolation between start and end temperature."""
        return self.start_temperature + (self.end_temperature - self.start_temperature) * progress

    def _exponential_schedule(self, progress: float) -> float:
        """Exponential interpolation between start and end temperature."""
        ratio = self.end_temperature / self.start_temperature
        return self.start_temperature * (ratio**progress)

    def _cosine_schedule(self, progress: float) -> float:
        """Cosine annealing schedule."""
        return (
            self.end_temperature + (self.start_temperature - self.end_temperature) * (1 + np.cos(np.pi * progress)) / 2
        )

    def step(self):
        """Increment the internal step counter."""
        if self.enabled:
            self.current_step += 1

    def reset(self):
        """Reset the internal step counter."""
        self.current_step = 0

    def set_step(self, step: int):
        """Set the internal step counter to a specific value.

        Args:
            step: Step number to set.
        """
        if step < 0:
            raise ValueError("Step must be non-negative.")
        self.current_step = step
