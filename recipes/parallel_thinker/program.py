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
"""Parallel Thinker program: Solver -> Rewriter -> Selector.

Pipeline for each problem:
  1. N **Solvers** generate initial answers concurrently.
  2. N **Rewriters** see all solver outputs and each produce an improved answer.
  3. 1 **Selector** picks the best rewritten answer.

Uses per-role baselines for advantage computation:
  - Solver advantage   = (reward - LOO mean of other 4 solvers) × team_weight
  - Rewriter advantage = (reward - LOO mean of other 4 rewriters) × team_weight
  - Selector advantage = (reward - mean rewriter reward) × team_weight  [vs random pick]

The computed advantages are passed as ``step_rewards`` in ``ProgramResult``.
Use with ``advantage=identity`` so the training pipeline treats them as
pre-computed advantages.
"""

from __future__ import annotations

import asyncio
import logging
import re

from axon.programs.base_program import BaseProgram, ProgramResult, register_program
from axon.utils.rewards import math_reward_fn

logger = logging.getLogger(__name__)


def make_rewriter_prompt(problem: str, previous_solutions: list[str]) -> str:
    n = len(previous_solutions)
    solution_sections = "\n".join(f"#### Solution {i + 1}\n{sol}\n\n---" for i, sol in enumerate(previous_solutions))
    return (
        f"### Task: Solution Rewriting Based on Previous Solutions ###\n"
        f"You are being reactivated to revise your mathematical proof. "
        f"You are provided with two documents:\n"
        f"1.  The problem you need to solve.\n"
        f'2.  Your {n} different "Previous Solutions".\n\n'
        f"Your sole task is to generate a new, correct version of your solution "
        f"based on your previous discoveries in the provided {n} solutions.\n\n"
        f"Refer to the following {n} solutions and solve the problem.\n"
        f"---\n\n"
        f"### Problem\n\n{problem}\n\n---\n\n"
        f"### Candidates Solution\n{solution_sections}\n"
    )


def make_selector_prompt(problem: str, candidate_solutions: list[str]) -> str:
    n = len(candidate_solutions)
    solution_sections = "\n".join(f"#### Solution {i + 1}\n{sol}\n\n---" for i, sol in enumerate(candidate_solutions))
    return (
        f"You will be given a challenging math problem followed by {n} solutions.\n"
        f"Your task is to systematically analyze these solutions to identify "
        f"the most mathematically sound approach. \n\n"
        f"You are provided with two documents:\n"
        f"1.  The problem you need to solve.\n"
        f'2.  Your {n} "Candidate Solutions".\n\n'
        f"Evaluation Process:\n"
        f"1. Initial Screening\n"
        f"- Group solutions by their final answers\n"
        f"- Identify and explain mathematical contradictions between different answers\n"
        f"- Eliminate solutions with clear mathematical errors\n\n"
        f"2. Detailed Analysis\n"
        f"For remaining solutions, evaluate:\n"
        f"- Mathematical precision and accuracy\n"
        f"- Logical progression of steps\n"
        f"- Completeness of mathematical reasoning\n"
        f"- Handling of edge cases or special conditions\n"
        f"- For solutions containing and addressing errors, evaluate the error "
        f"identification and correction methodology.\n\n"
        f"3. Solution Comparison\n"
        f"Compare viable solutions based on:\n"
        f"- Efficiency of approach\n"
        f"- Clarity of mathematical reasoning\n"
        f"- Sophistication of method\n"
        f"- Robustness of solution (works for all cases)\n\n"
        f"Your response should include:\n"
        f"1. Brief analysis of conflicting answers\n"
        f"2. Detailed evaluation of mathematically sound solutions\n"
        f"3. Justification for eliminating incorrect solutions\n"
        f"4. Clear explanation for selecting the best approach\n\n"
        f"End your evaluation with exactly:\n"
        f"Judgment: IDX\n"
        f"where IDX is the index 1-{n} of the best solution\n\n"
        f"### Problem\n\n{problem}\n\n---\n\n"
        f"### Candidate Solutions\n{solution_sections}\n"
    )


def _extract_answer_content(response: str) -> str | None:
    """Extract the non-thinking part of a response (after </think>)."""
    if not response:
        return None
    final = response.replace("<|user|>", "")
    if "</think>" in final:
        parts = final.split("</think>")
        if len(parts) == 2 and parts[1].strip():
            return parts[1].strip()
    return None


def _extract_selected_index(response: str, num_candidates: int) -> int | None:
    """Return 0-based index of the selected solution, or None on failure."""
    matches = re.findall(r"Judgment:\s*(\d+)", response)
    if not matches:
        return None
    try:
        idx = int(matches[0]) - 1
        if 0 <= idx < num_candidates:
            return idx
    except (ValueError, IndexError):
        pass
    return None


def _leave_one_out_mean(rewards: dict[int, float], exclude_key: int) -> float:
    """Mean of all values except the one at exclude_key."""
    others = [r for k, r in rewards.items() if k != exclude_key]
    return sum(others) / len(others) if others else 0.0


@register_program("parallel_thinker")
class ParallelThinkerProgram(BaseProgram):
    """Orchestrates a Solver -> Rewriter -> Selector pipeline.

    Computes stage-conditioned advantages and passes them as ``step_rewards``.
    Pair with ``advantage=identity`` to skip further normalization.

    Args:
        num_parallel: How many solvers / rewriters to run per stage.
        correct_reward_weight: Team-level multiplier when the final answer is correct.
        incorrect_reward_weight: Team-level multiplier when the final answer is wrong.
        env_args: Per-sample data (question, answer) injected by the driver.
    """

    def __init__(
        self,
        num_parallel: int = 5,
        correct_reward_weight: float = 1.2,
        incorrect_reward_weight: float = 0.8,
        env_args: dict | None = None,
        # BaseProgram kwargs
        group_id: str = "",
        sample_params: dict | None = None,
        endpoint_url: str = "",
        retry_limit: int = 1,
        program_timeout: int = 10800,
    ):
        super().__init__(
            group_id=group_id,
            sample_params=sample_params,
            endpoint_url=endpoint_url,
            retry_limit=retry_limit,
            program_timeout=program_timeout,
        )
        self.num_parallel = num_parallel
        self.correct_reward_weight = correct_reward_weight
        self.incorrect_reward_weight = incorrect_reward_weight
        self.env_args = dict(env_args or {})

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _generate_one(self, user_content: str) -> tuple[str, str | None, int]:
        """Send a single-turn user message.

        Returns (raw_response, extracted_content, step_idx).
        """
        messages = [{"role": "user", "content": user_content}]
        raw_response, _stop, step_idx = await self.generate(messages=messages, sample_params=self.sample_params)
        return raw_response, _extract_answer_content(raw_response), step_idx

    def _compute_reward(self, raw_response: str) -> float:
        """Compute math reward on the full raw response."""
        reward_out = math_reward_fn(task_info=self.env_args, action=raw_response)
        return reward_out.reward

    # ------------------------------------------------------------------
    # Pipeline stages
    # ------------------------------------------------------------------

    async def _run_solvers(self, problem: str) -> list[tuple[str, str | None, int]]:
        """Stage 1: N independent solver responses (parallel)."""
        return list(await asyncio.gather(*(self._generate_one(problem) for _ in range(self.num_parallel))))

    async def _run_rewriters(self, problem: str, previous_solutions: list[str]) -> list[tuple[str, str | None, int]]:
        """Stage 2: N rewriter responses seeing all solver outputs (parallel)."""
        prompt = make_rewriter_prompt(problem, previous_solutions)
        return list(await asyncio.gather(*(self._generate_one(prompt) for _ in range(self.num_parallel))))

    async def _run_selector(self, problem: str, candidates: list[str]) -> tuple[str, str | None, int | None, int]:
        """Stage 3: one selector response."""
        prompt = make_selector_prompt(problem, candidates)
        raw, extracted, step_idx = await self._generate_one(prompt)
        if extracted is None:
            return raw, None, None, step_idx
        idx = _extract_selected_index(extracted, len(candidates))
        return raw, extracted, idx, step_idx

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def run(self) -> ProgramResult:
        """Execute the full Solver -> Rewriter -> Selector pipeline.

        Computes per-role advantages:
          solver_adv   = solver_reward   - LOO_mean(other solvers)     [peer comparison]
          rewriter_adv = rewriter_reward - LOO_mean(other rewriters)   [peer comparison]
          selector_adv = selector_reward - mean(rewriter rewards)      [vs random picking]

        Then multiplies all by team_weight (correct or incorrect).
        """
        problem = self.env_args.get("question", "")

        # --- Stage 1: Solvers ---
        solver_all = await self._run_solvers(problem)
        solver_rewards: dict[int, float] = {}
        for raw, _extracted, step_idx in solver_all:
            solver_rewards[step_idx] = self._compute_reward(raw)

        solver_valid = [(extracted, si) for _raw, extracted, si in solver_all if extracted is not None]
        if not solver_valid:
            logger.warning("All solvers failed to produce output")
            # All failed → advantage = reward - LOO mean, all scaled by incorrect weight
            weight = self.incorrect_reward_weight
            step_rewards = {
                si: (r - _leave_one_out_mean(solver_rewards, si)) * weight for si, r in solver_rewards.items()
            }
            return ProgramResult(reward=0.0, done=True, step_rewards=step_rewards)

        # --- Stage 2: Rewriters ---
        solver_texts = [c for c, _ in solver_valid]
        rewriter_all = await self._run_rewriters(problem, solver_texts)

        rewriter_rewards: dict[int, float] = {}
        for raw, _extracted, step_idx in rewriter_all:
            rewriter_rewards[step_idx] = self._compute_reward(raw)

        rewriter_valid = [(extracted, si) for _raw, extracted, si in rewriter_all if extracted is not None]
        if not rewriter_valid:
            logger.warning("All rewriters failed")
            weight = self.incorrect_reward_weight
            step_rewards: dict[int, float] = {}
            for si, r in solver_rewards.items():
                step_rewards[si] = (r - _leave_one_out_mean(solver_rewards, si)) * weight
            for si, r in rewriter_rewards.items():
                step_rewards[si] = (r - _leave_one_out_mean(rewriter_rewards, si)) * weight
            return ProgramResult(reward=0.0, done=True, step_rewards=step_rewards)

        # --- Stage 3: Selector ---
        rewriter_texts = [c for c, _ in rewriter_valid]
        selector_raw, selector_extracted, selected_idx, selector_step_idx = await self._run_selector(
            problem, rewriter_texts
        )

        # Selector reward = reward of the rewriter it picked
        if selector_extracted is None or selected_idx is None:
            selector_reward = 0.0
        else:
            _, picked_step_idx = rewriter_valid[selected_idx]
            selector_reward = rewriter_rewards.get(picked_step_idx, 0.0)

        # --- Team weight ---
        final_correct = selector_reward == 1.0
        weight = self.correct_reward_weight if final_correct else self.incorrect_reward_weight

        # --- Per-role advantages × team_weight ---
        rewriter_mean = sum(rewriter_rewards.values()) / len(rewriter_rewards)

        step_rewards: dict[int, float] = {}
        # Solvers: LOO baseline (compare against peer solvers)
        for si, r in solver_rewards.items():
            step_rewards[si] = (r - _leave_one_out_mean(solver_rewards, si)) * weight
        # Rewriters: LOO baseline (compare against peer rewriters)
        for si, r in rewriter_rewards.items():
            step_rewards[si] = (r - _leave_one_out_mean(rewriter_rewards, si)) * weight
        # Selector: baseline = mean(rewriter rewards) = expected reward of random picking
        step_rewards[selector_step_idx] = (selector_reward - rewriter_mean) * weight

        return ProgramResult(
            reward=selector_reward * weight,
            done=True,
            step_rewards=step_rewards,
        )
