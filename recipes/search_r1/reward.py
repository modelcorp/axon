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
#
# Reward shaping adapted from Search-R1 (github.com/PeterGriffinJin/Search-R1), Apache-2.0.
import logging
import re
import string
import unicodedata
from dataclasses import dataclass

from axon.utils.rewards.base import RewardInput, RewardOutput

logger = logging.getLogger(__name__)


@dataclass
class SearchR1RewardConfig:
    """
    Attributes:
        correct_reward: Reward for correct answer (default: 1.0)
        incorrect_reward: Reward for incorrect answer (default: 0.0)
        unk_error_reward: Reward for errors (default: 0.0)
        format_score: Bonus for having <answer> tags (default: 0.1)
        structure_format_score: Bonus for valid tag structure (default: 0.2)
        final_format_score: Bonus for proper final format (default: 0.1)
        retrieval_score: Bonus for retrieving answer (default: 0.1)
    """

    correct_reward: float = 1.0
    incorrect_reward: float = 0.0
    unk_error_reward: float = 0.0
    structure_format_score: float = 0.2
    final_format_score: float = 0.1
    retrieval_score: float = 0.1


class SearchR1RewardFn:
    def __init__(self, config: SearchR1RewardConfig):
        self.config = config

    def normalize_answer(self, s: str) -> str:
        """Normalize answer for comparison."""
        s = unicodedata.normalize("NFC", s)

        def remove_articles(text):
            return re.sub(r"\b(a|an|the)\b", " ", text)

        def white_space_fix(text):
            return " ".join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return "".join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))

    def em_check(self, prediction: str, golden_answers: list[str]) -> bool:
        """Exact match check."""
        if isinstance(golden_answers, str):
            golden_answers = [golden_answers]

        normalized_pred = self.normalize_answer(prediction)
        for golden in golden_answers:
            if self.normalize_answer(golden) == normalized_pred:
                return True
        return False

    def extract_solution(self, solution_str: str) -> str | None:
        """Extract the equation from the solution string."""

        answer_pattern = r"<answer>(.*?)</answer>"
        matches = list(re.finditer(answer_pattern, solution_str, re.DOTALL))

        # If there are 0 matches, return None
        if len(matches) < 1:
            return None

        # If there are matches, return the last one
        return matches[-1].group(1).strip()

    def extract_information_blocks(self, text: str) -> list[str]:
        """Extract all <information>...</information> blocks."""
        pattern = r"<information>(.*?)</information>"
        matches = re.findall(pattern, text, re.DOTALL)
        return [match.strip() for match in matches]

    def is_retrieval_correct(self, text: str, golden_answers: list[str]) -> bool:
        """Check if any golden answer appears in retrieved information blocks."""
        info_blocks = self.extract_information_blocks(text)
        for block in info_blocks:
            normalized_block = self.normalize_answer(block)
            for golden in golden_answers:
                if self.normalize_answer(golden) in normalized_block:
                    return True
        return False

    def is_valid_sequence(self, text: str) -> tuple[bool, str]:
        """Check if tags are properly balanced and in valid sequence."""
        # Check for balanced tags first (fast check)
        tags_to_check = ["think", "search", "information", "answer"]
        for tag in tags_to_check:
            opening_count = len(re.findall(f"<{tag}>", text))
            closing_count = len(re.findall(f"</{tag}>", text))
            if opening_count != closing_count:
                return False, f"Mismatch in {tag} tags: {opening_count} opening vs {closing_count} closing"

        # Now check for proper sequence pattern and no extraneous content
        # Split the content by any tags we recognize
        split_pattern = r"(</?(?:think|search|information|answer)>)"
        parts = re.split(split_pattern, text)

        # Keep track of the current position in the expected sequence
        # Valid transitions:
        # start -> think -> search -> information -> think -> ... -> answer -> end
        # OR: start -> think -> answer -> end (direct answer without search)
        state = "start"

        for part in parts:
            # Skip empty parts
            if not part.strip():
                continue

            # Check if this is a tag
            if re.match(r"</?(?:think|search|information|answer)>", part):
                # This is a tag, check if it's valid in the current state
                if part == "<think>" and state in ["start", "information"]:
                    state = "in_think"
                elif part == "</think>" and state == "in_think":
                    state = "after_think"
                elif part == "<search>" and state == "after_think":
                    state = "in_search"
                elif part == "</search>" and state == "in_search":
                    state = "after_search"
                elif part == "<information>" and state == "after_search":
                    state = "in_information"
                elif part == "</information>" and state == "in_information":
                    state = "information"
                elif part == "<answer>" and state == "after_think":
                    state = "in_answer"
                elif part == "</answer>" and state == "in_answer":
                    state = "end"
                else:
                    return False, f"Unexpected tag {part} in state {state}"
            else:
                # This is content, check if it's valid in the current state
                if state in ["in_think", "in_search", "in_information", "in_answer"]:
                    # Content is allowed inside tags
                    pass
                elif state in ["start", "after_think", "after_search", "information"]:
                    # Only whitespace is allowed between tags
                    if part.strip():
                        return False, f"Unexpected content '{part.strip()[:50]}...' between tags (state: {state})"
                else:
                    return False, f"Unexpected content in state {state}"

        # Check final state
        if state != "end":
            return False, f"Incomplete sequence, ended in state {state}"

        return True, "Valid sequence format"

    def compute_score_em(
        self,
        solution_str: str,
        ground_truth: list[str],
        structure_format_score: float = 0.0,
        final_format_score: float = 0.0,
        retrieval_score: float = 0.0,
        score: float = 1.0,
    ) -> float:
        """The scoring function for exact match (EM).

        Args:
            solution_str: the solution text
            ground_truth: the ground truth
            method: the method to extract the solution, choices are 'strict' and 'flexible'
            score: the score for the correct answer
        """
        # Check structure validity
        is_valid_format, _ = self.is_valid_sequence(solution_str)

        # Check retrieval correctness
        retrieval_correct = False
        if is_valid_format:
            retrieval_correct = self.is_retrieval_correct(solution_str, ground_truth)

        # Extract answer
        answer = self.extract_solution(solution_str)

        if answer is None:
            if is_valid_format:
                if retrieval_correct:
                    return structure_format_score + retrieval_score  # 0.3
                else:
                    return structure_format_score  # 0.2
            else:
                return 0
        else:
            if self.em_check(answer, ground_truth):
                if is_valid_format:
                    return score  # 1.0
                else:
                    return score - structure_format_score  # 0.8
            elif is_valid_format:
                if retrieval_correct:
                    return structure_format_score + retrieval_score  # 0.3
                else:
                    return structure_format_score  # 0.2
            else:
                return final_format_score  # 0.1

    def __call__(self, input: RewardInput) -> RewardOutput:
        """Main entry point for reward calculation."""
        action = input.action
        task_info = input.task_info

        # Extract ground truth
        if "answer" in task_info:
            ground_truth = task_info["answer"]
        else:
            logger.error(f"No ground truth found in task_info: {task_info}")
            return RewardOutput(
                reward=self.config.unk_error_reward, is_correct=False, metadata={"error": "No ground truth provided"}
            )

        score = self.compute_score_em(
            solution_str=action,
            ground_truth=ground_truth,
            structure_format_score=self.config.structure_format_score,
            final_format_score=self.config.final_format_score,
            retrieval_score=self.config.retrieval_score,
            score=self.config.correct_reward,
        )

        # Determine if answer is correct (score == 1.0 means fully correct)
        is_correct = score >= self.config.correct_reward

        metadata = {"score": score, "ground_truth": ground_truth}

        return RewardOutput(reward=score, is_correct=is_correct, metadata=metadata)
