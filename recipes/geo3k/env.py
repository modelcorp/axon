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
Geo3k Environment and Reward Function

This module contains the Geo3kEnvironment class, which wraps the SingleTurnEnvironment
for geometry problems, along with the reward function for evaluating mathematical answers
based on their correctness.
"""

import re

from mathruler.grader import extract_boxed_content, grade_answer

from axon.core import SingleTurnEnvironment, register_env
from axon.utils.rewards.base import RewardConfig, RewardOutput

ORM_USER_TEMPLATE = """
Problem: {problem}
Answer 1: {answer_1}
Answer 2: {answer_2}
"""


def format_reward(predict_str: str) -> float:
    pattern = re.compile(r"<think>.*</think>.*\\boxed\{.*\}.*", re.DOTALL)
    match_result = re.fullmatch(pattern, predict_str)
    return 1.0 if match_result else 0.0


def acc_reward(predict_str: str, ground_truth: str, use_boxed: bool = False) -> float:
    if use_boxed:
        answer = extract_boxed_content(predict_str)
    else:
        answer = predict_str
    return 1.0 if grade_answer(answer, ground_truth) else 0.0


def compute_score(predict_str: str, ground_truth: str, use_boxed: bool = True, format_score: float = 0.0) -> float:
    return (1.0 - format_score) * acc_reward(predict_str, ground_truth, use_boxed) + format_score * format_reward(
        predict_str
    )


class RewardGeo3kFn:
    """
    Reward function for evaluating mathematical answers.

    This class implements the RewardFunction protocol to process the input and determine
    the reward based on the correctness of the provided answer compared to the ground truth.
    """

    def __init__(self, config: RewardConfig):
        self.config = config

    def __call__(self, task_info: dict, action: str) -> RewardOutput:
        """
        Calculate the reward for a math task based on the agent's action.

        Args:
            task_info: Dictionary containing problem, data_source, problem_type, and ground_truth
            action: The agent's response/solution

        Returns:
            RewardOutput: The calculated reward with correctness information
        """
        model_response = action

        # Handle None or empty response
        if model_response is None or model_response == "":
            print("DEBUG: Empty or None response")
            return RewardOutput(reward=self.config.format_error_reward, is_correct=False)

        model_solution = model_response

        model_answer = extract_boxed_content(model_solution)
        if model_answer is None:
            return RewardOutput(reward=self.config.format_error_reward, is_correct=False)

        # Process the ground truth(s)
        ground_truths = task_info.get("ground_truth", None)
        if ground_truths is None:
            ground_truths = task_info.get("answer", None)
            if ground_truths is None:
                return RewardOutput(reward=self.config.unk_error_reward, is_correct=False)

        # Convert single answer to list for uniform processing
        if isinstance(ground_truths, str | float | int):
            ground_truths = [ground_truths]

        # Process each ground truth
        processed_ground_truths = ground_truths

        # Check against all possible correct answers
        for ground_truth in processed_ground_truths:
            is_correct = acc_reward(model_answer, ground_truth)
            if is_correct:
                # Apply tool call bonus if applicable and answer is correct
                reward = self.config.correct_reward
                if task_info.get("has_toolcall", False):
                    reward += self.config.toolcall_bonus
                return RewardOutput(reward=reward, is_correct=True)

        return RewardOutput(reward=self.config.incorrect_reward, is_correct=False)


def geo3k_reward_fn(task_info: dict, action: str) -> RewardOutput:
    reward_config = RewardConfig()
    reward_fn = RewardGeo3kFn(reward_config)
    return reward_fn(task_info, action)


@register_env("geo3k")
class Geo3kEnvironment(SingleTurnEnvironment):
    @staticmethod
    def from_dict(env_args: dict) -> "SingleTurnEnvironment":
        reward_fn = env_args.pop("reward_fn", geo3k_reward_fn)
        task = env_args
        return SingleTurnEnvironment(task=task, reward_fn=reward_fn)
