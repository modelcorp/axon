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
"""Math reward function used by recipe-level verifiable reward tasks."""

import logging

from axon.globals import OAI_RM_MODEL, THOUGHT_DELIMITER_END
from axon.utils import call_gemini_llm, call_oai_rm_llm
from axon.utils.rewards.base import RewardConfig, RewardOutput, RewardType
from axon.utils.rewards.math_utils import (
    extract_answer,
    grade_answer_math_verify,
    grade_answer_mathd,
    grade_answer_sympy,
)
from axon.utils.system_prompts import ORM_PROMPT

ORM_USER_TEMPLATE = """
Problem: {problem}
Answer 1: {answer_1}
Answer 2: {answer_2}
"""

logger = logging.getLogger(__name__)


class RewardMathFn:
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
        # Extract information from task_info
        problem = task_info.get("problem", "")
        model_response = action

        # Handle None or empty response
        if model_response is None or model_response == "":
            logger.debug("Empty math response received; returning format error reward")
            return RewardOutput(reward=self.config.format_error_reward, is_correct=False)

        # Extract solution.
        if THOUGHT_DELIMITER_END in model_response:
            model_solution = model_response.split(THOUGHT_DELIMITER_END)[1]
        else:
            if self.config.apply_format_reward:
                return RewardOutput(reward=self.config.format_error_reward, is_correct=False)
            model_solution = model_response

        model_answer = extract_answer(model_solution)
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
        processed_ground_truths = []
        for truth in ground_truths:
            truth = str(truth)
            if "\\boxed" in truth:
                processed_truth = extract_answer(truth)
                if processed_truth is not None:
                    processed_ground_truths.append(processed_truth)
            else:
                processed_ground_truths.append(truth)

        if not processed_ground_truths:
            return RewardOutput(reward=self.config.unk_error_reward, is_correct=False)

        # Check against all possible correct answers.
        # Order: fast exact-match (mathd) → legacy sympy heuristic → math_verify
        # (latex2sympy2 + sympy equivalence, catches \sqrt[n], multi-var trig,
        # algebraic rearrangements that the legacy path's heuristics refuse).
        for ground_truth in processed_ground_truths:
            is_correct = (
                grade_answer_mathd(model_answer, ground_truth)
                or grade_answer_sympy(model_answer, ground_truth)
                or grade_answer_math_verify(model_answer, ground_truth)
            )
            if is_correct:
                # Apply tool call bonus if applicable and answer is correct
                reward = self.config.correct_reward
                if task_info.get("has_toolcall", False):
                    reward += self.config.toolcall_bonus
                return RewardOutput(reward=reward, is_correct=True)

        # If latex heuristics fail and ORM is enabled, use LLM as ORM to evaluate correctness
        if self.config.use_math_orm:
            for ground_truth in processed_ground_truths:
                try:
                    orm_response = call_gemini_llm(
                        system_prompt=ORM_PROMPT,
                        prompt=ORM_USER_TEMPLATE.format(problem=problem, answer_1=model_answer, answer_2=ground_truth),
                        temperature=0.0,
                    )

                    if "[[YES]]" in orm_response:
                        return RewardOutput(reward=self.config.correct_reward, is_correct=True)
                except Exception:
                    logger.debug("Gemini ORM scoring failed; falling back to OAI reward model", exc_info=True)
                    orm_response = call_oai_rm_llm(
                        system_prompt=ORM_PROMPT,
                        prompt=ORM_USER_TEMPLATE.format(problem=problem, answer_1=model_answer, answer_2=ground_truth),
                        temperature=0.0,
                        model_id=OAI_RM_MODEL,
                    )

                    if "[[YES]]" in orm_response:
                        return RewardOutput(reward=self.config.correct_reward, is_correct=True)
                    continue

        return RewardOutput(reward=self.config.incorrect_reward, is_correct=False)


def math_reward_fn(task_info: dict, action: str) -> RewardOutput:
    """
    A reward function for math tasks that implements the RewardFunction protocol.

    Args:
        task: The task dictionary containing data_source, ground_truth and other metadata
        action: The agent's response/solution

    Returns:
        float: The calculated reward value based on math evaluation
    """
    reward_config = RewardConfig()
    reward_fn = RewardMathFn(reward_config)
    return reward_fn(task_info, action)


if __name__ == "__main__":
    reward = RewardMathFn(RewardConfig())
    task_info = {
        "data_source": "",
        "problem": (
            "Let $P(x)=x^{4}+2 x^{3}-13 x^{2}-14 x+24$ be a polynomial with roots $r_{1}, r_{2}, r_{3}, r_{4}$. Let $Q$ be the quartic polynomial with roots $r_{1}^{2}, r_{2}^{2}, r_{3}^{2}, r_{4}^{2}$, such that the coefficient of the $x^{4}$ term of $Q$ is 1. Simplify the quotient $Q\\left(x^{2}\\right) / P(x)$, leaving your answer in terms of $x$. (You may assume that $x$ is not equal to any of $\\left.r_{1}, r_{2}, r_{3}, r_{4}\\right)$."
        ),
        "problem_type": RewardType.MATH,
        "ground_truth": ["10", "$x^{4}-2 x^{3}-13 x^{2}+14 x+24$"],
        "has_toolcall": True,
    }
    action = "<think>...</think>\nThe answer is \\boxed{24 + 14*x + (-13)*x^2 - 2*x^3 + x^4}."

    output = reward(task_info, action)
    print(output)
