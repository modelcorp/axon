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
This module defines data structures and base classes for reward calculations
to evaluate model responses for various problem types, including math and coding.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, runtime_checkable


@dataclass
class RewardConfig:
    apply_format_reward: bool = False

    # Config for math-bsed rewards
    math_reward_weight: float = 1.0
    use_math_orm: bool = False

    # Config for code-based rewards
    code_reward_weight: float = 1.0

    # Config for cot-based rewards
    cot_reward_weight: float = 0.0

    # General reward constants
    correct_reward: float = 1.0
    incorrect_reward: float = 0.0
    format_error_reward: float = 0.0
    unk_error_reward: float = 0.0

    # Bonus reward for calling tools.
    toolcall_bonus: float = 0.5

    # Toggle for using Together Code Interpreter
    use_together_code_interpreter: bool = False


class RewardType(Enum):
    """
    Enum class representing the different types of rewards that can be assigned.

    Attributes:
        MATH (str): Represents a math-related problem type.
        CODE (str): Represents a coding-related problem type.
        UNK (str): Represents an unknown or unclassified problem type.
    """

    MATH = "MATH"
    CODE = "CODE"
    WEB = "WEB"
    UNK = "UNK"


@dataclass(slots=True, kw_only=True)
class RewardInput:
    """Data structure for input required to calculate rewards.

    Attributes:
        task_info (Dict): The task dictionary containing question, answer, and other metadata
        action (str): The agent's response/solution that needs evaluation
    """

    task_info: dict
    action: str


@dataclass(slots=True, kw_only=True)
class RewardOutput:
    """Data structure for the output of reward calculations.

    Attributes:
        reward (float): The computed reward value based on the evaluation of the model's response.
        metadata (dict): Additional information about the reward calculation.
        is_correct (bool): A boolean flag indicating whether the model's response is deemed correct.
    """

    reward: float
    metadata: dict = field(default_factory=dict)
    is_correct: bool | None = None


@runtime_checkable
class RewardFunction(Protocol):
    """Protocol for reward functions"""

    def __call__(self, task_info: dict, action: str) -> RewardOutput:
        """
        Calculate the reward for an agent's action.

        Args:
            task_info: The task dictionary containing question, answer, and other metadata
            action: The agent's response/solution

        Returns:
            RewardOutput: The calculated reward value, either as a float or RewardOutput object
        """
        ...


# Simple example of a reward function
def zero_reward_fn(task_info: dict, action: str) -> RewardOutput:  # noqa: ARG001
    """
    A simple reward function that always returns zero.
    Useful as a placeholder when no specific reward logic is needed.

    Args:
        task_info: The task dictionary (unused)
        action: The agent's response (unused)

    Returns:
        float: Always returns 0.0
    """
    return RewardOutput(reward=0.0, metadata={})
