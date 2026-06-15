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
from axon.core import MultiTurnEnvironment, register_env
from axon.utils.rewards import code_reward_fn


@register_env("competition_coding")
class CompetitionCodingEnv(MultiTurnEnvironment):
    """
    Environment for competitive coding tasks that inherits from MultiTurnEnvironment.
    """

    def __init__(self, task: dict | None = None, max_turns: int = 1, reward_bonus_coeff: float = 0.0, **kwargs):
        """
        Initialize the competitive coding environment.

        Args:
            task: Dictionary containing the task information
            max_turns: Maximum number of turns before terminating the interaction
        """
        super().__init__(task=task, max_turns=max_turns, **kwargs)
        self.reward_fn = code_reward_fn
        self.prev_reward: float | None = None
        self.reward_bonus_coeff = reward_bonus_coeff

    def reset(self, task=None, seed=None):
        """Reset the environment and return initial observations."""
        import random

        if seed is not None:
            random.seed(seed)

        # Use the provided task if available, otherwise use the default task
        if task is not None:
            self.task = task

        assert self.task is not None, "Task must be set before reset"

        self.done = False
        self.current_turn = 0
        self.history = []
        self.prev_reward = None

        # Return the first question
        return {"question": self.task["question"]}, {}

    def step(self, action):
        """
        Take a step in the environment based on the action.

        Args:
            action: Response string from the LLM

        Returns:
            next_observation, reward, terminated, truncated, info
        """
        # Store the action in history
        self.history.append(action)

        # Calculate reward for the current turn using the abstract method
        assert self.task is not None, "Task is not set"
        raw_reward, next_obs = self.get_reward_and_next_obs(self.task, action)

        # Reward shaping
        if self.prev_reward is None:
            reward = raw_reward
        else:
            bonus = self.reward_bonus_coeff * (raw_reward - self.prev_reward)
            reward = raw_reward + bonus

        self.prev_reward = raw_reward

        # Increment turn counter
        self.current_turn += 1

        # Check if we've reached the maximum number of turns
        # if self.current_turn >= self.max_turns or reward == 1:
        if self.current_turn >= self.max_turns:
            self.done = True
            return {}, reward, self.done, self.task

        return next_obs, reward, self.done, self.task

    def get_reward_and_next_obs(self, task: dict, action: str) -> tuple[float, dict]:
        """
        Compute the reward for a competitive coding task.

        Args:
            task: The task dictionary containing relevant information
            action: The response string from the LLM

        Returns:
            Tuple of (reward: float, metadata: Dict)
        """
        assert self.reward_fn is not None, "Reward function is not set"
        reward_response = self.reward_fn(task_info=task, action=action)
        return reward_response.reward, reward_response.metadata

    @staticmethod
    def from_dict(env_args: dict) -> "CompetitionCodingEnv":
        max_turns = env_args.pop("max_turns", 1)
        reward_bonus_coeff = env_args.pop("reward_bonus_coeff", 0.0)
        return CompetitionCodingEnv(
            task=env_args,
            max_turns=max_turns,
            reward_bonus_coeff=reward_bonus_coeff,
        )
