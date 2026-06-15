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
import warnings
from abc import ABC, abstractmethod
from typing import Any

from axon.utils.registry import ClassRegistry
from axon.utils.rewards.base import RewardFunction, zero_reward_fn

# Environment registry
ENV_CLASS_MAPPING = ClassRegistry("env")
register_env = ENV_CLASS_MAPPING.register


class BaseEnv(ABC):
    """Helper class for environments used by :class:`~axon.programs.react_program.ReactProgram`.

    An environment owns the world state, the ``step`` transition, and the
    reward. It pairs with a :class:`~axon.core.agent.BaseAgent` inside a
    ReactProgram-style rollout. The rollout abstraction itself is
    :class:`~axon.programs.base_program.BaseProgram`; custom programs may bypass
    ``BaseEnv`` entirely.

    Subclasses register themselves through ``@register_env("name")`` so recipes can
    refer to them by string name in yaml. Two ready-made shapes are provided:

    * :class:`SingleTurnEnvironment` — one observation, one action, one reward.
      Used by math-style tasks.
    * :class:`MultiTurnEnvironment` — multiple `(observation, action)` rounds before
      terminal reward. Used by FrozenLake, code-with-feedback, SWE, search-r1, etc.

    Subclasses must implement :meth:`reset`, :meth:`step`, and :meth:`from_dict`.
    """

    @property
    def idx(self) -> Any:
        """The index or identifier of the environment, often used within a batch.

        Returns:
            The assigned index or identifier, or None if not set.
        """
        # Return the stored _idx value if it exists, otherwise return None.
        return getattr(self, "_idx", None)

    @idx.setter
    def idx(self, value: Any):
        """Set the environment index or identifier.

        This allows assigning an index or identifier (e.g., its position in a batch)
        to the environment instance after it has been created.

        Example:
            env = MyEnvSubclass()  # Assuming MyEnvSubclass inherits from BaseEnv
            env.idx = 5            # Set the index externally

        Args:
            value: The index or identifier to set for this environment.
        """
        self._idx = value

    @abstractmethod
    def reset(self) -> tuple[dict, dict]:
        """Standard Gym reset method. Resets the environment to an initial state.

        Returns:
            A tuple typically containing the initial observation and auxiliary info.
        """
        pass

    @abstractmethod
    def step(self, action: Any) -> tuple[Any, float, bool, dict]:
        """Standard Gym step method. Executes one time step within the environment.

        Args:
            action: An action provided by the agent.

        Returns:
            A tuple containing (observation, reward, done, info).
        """
        pass

    def close(self):
        """Standard Gym close method. Performs any necessary cleanup."""
        return

    @staticmethod
    @abstractmethod
    def from_dict(info: dict) -> "BaseEnv":
        """Creates an environment instance from a dictionary.

        This method should be implemented by concrete subclasses to handle
        environment-specific initialization from serialized data.

        Args:
            info: A dictionary containing the necessary information to initialize the environment.

        Returns:
            An instance of the specific BaseEnv subclass.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        # BaseEnv is abstract, subclasses must implement this factory method.
        raise NotImplementedError("Subclasses must implement the 'from_dict' static method.")

    @staticmethod
    def is_multithread_safe() -> bool:
        return True


@register_env("multi_turn")
class MultiTurnEnvironment(BaseEnv, ABC):
    """
    An environment for multi-turn interactions with LLMs.
    The environment provides a series of questions/prompts and evaluates responses using a custom reward function.
    The interaction terminates after reaching the maximum number of turns.
    """

    def __init__(self, task: dict | None = None, max_turns: int = 3, **kwargs):
        """
        Initialize the multi-turn environment.

        Args:
            task: Dictionary containing the task information, including at least a "questions" field
                  with a list of questions for each turn
            max_turns: Maximum number of turns before terminating the interaction
        """
        super().__init__()
        self.task = task
        self.max_turns = max_turns
        self.current_turn = 0
        self.done = False
        self.history: list[Any] = []

    def reset(self):
        self.done = False
        self.current_turn = 0
        self.history = []

        return self.task, {}

    def step(self, action):
        """
        Take a step in the environment based on the action.

        Args:
            action: Response string from the LLM

        Returns:
            next_observation, reward, terminated, info
        """
        # Store the action in history
        self.history.append(action)

        # Calculate reward for the current turn using the abstract method
        assert self.task is not None, "Task is not set"
        reward, next_obs = self.get_reward_and_next_obs(self.task, action)

        # Increment turn counter
        self.current_turn += 1

        # Check if we've reached the maximum number of turns
        if self.current_turn >= self.max_turns:
            self.done = True
            return {}, reward, self.done, self.task

        return next_obs, reward, self.done, self.task

    def get_reward_and_next_obs(self, task: dict, action: Any) -> tuple[float, dict]:
        """
        Compute the reward and next observation based on the task and action.

        Subclasses that use the default ``step`` implementation must override this.
        Subclasses that override ``step`` directly may leave it as-is.

        Args:
            task: The task dictionary containing relevant information
            action: The action taken by the agent

        Returns:
            Tuple of (reward: float, next_observation: Dict)
        """
        raise NotImplementedError(
            f"{type(self).__name__} uses the default step() but does not override get_reward_and_next_obs()."
        )

    @staticmethod
    def from_dict(env_args: dict) -> "MultiTurnEnvironment":
        raise NotImplementedError(
            "MultiTurnEnvironment is abstract and cannot be instantiated directly. Use a concrete subclass."
        )


@register_env("single_turn")
class SingleTurnEnvironment(MultiTurnEnvironment):
    """
    A simple environment for single-turn interactions with LLMs.
    This is a special case of MultiTurnEnvironment where max_turns=1.
    The environment provides a question/prompt and evaluates the response using a custom reward function.
    """

    def __init__(self, task: dict | None = None, reward_fn: RewardFunction | str | None = None, **kwargs):
        """
        Initialize the single turn environment.

        Args:
            task: Dictionary containing the task information, including at least a "question" field
            reward_fn: A callable, a string name from REWARD_FN_REGISTRY, or None.
        """
        super().__init__(task=task, max_turns=1, **kwargs)
        if isinstance(reward_fn, str):
            from axon.utils.rewards import REWARD_FN_REGISTRY

            if reward_fn not in REWARD_FN_REGISTRY:
                raise KeyError(f"Unknown reward_fn '{reward_fn}'. Available: {list(REWARD_FN_REGISTRY)}")
            reward_fn = REWARD_FN_REGISTRY[reward_fn]
        if reward_fn is None:
            warnings.warn("No reward function provided, using zero reward", stacklevel=2)
        self.reward_fn = reward_fn or zero_reward_fn

    def get_reward_and_next_obs(self, task: dict, action: Any) -> tuple[float, dict]:
        """
        Compute the reward based on the task and action.

        Args:
            task: The task dictionary containing relevant information
            action: The action taken by the agent

        Returns:
            Tuple of (reward: float, next_observation: Dict)
        """
        reward_output = self.reward_fn(task_info=task, action=action)

        return reward_output.reward, {}

    @staticmethod
    def from_dict(env_args: dict) -> "SingleTurnEnvironment":
        env_args = dict(env_args)  # shallow copy to avoid mutating caller's dict
        reward_fn = env_args.pop("reward_fn", None)
        if "task" in env_args:
            task = env_args["task"]
        else:
            task = env_args
        return SingleTurnEnvironment(task=task, reward_fn=reward_fn)
