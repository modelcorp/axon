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
from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from axon.core import Action, BaseAgent, BaseEnv
from axon.core.agent import AGENT_CLASS_MAPPING
from axon.core.env import ENV_CLASS_MAPPING
from axon.programs.base_program import BaseProgram, ProgramResult, register_program
from axon.utils.print_utils import colorful_print


def _broadcast(val, n: int) -> list[dict]:
    """Normalize None|dict|list[dict]|Mapping to a list of *n* dicts."""
    if val is None:
        return [{}] * n
    if isinstance(val, Mapping):
        return [dict(val) for _ in range(n)]
    out = list(val)
    assert len(out) == n, f"Expected list of length {n}, got {len(out)}"
    return out


@register_program("react")
class ReactProgram(BaseProgram):
    """ReAct-style multi-turn rollout loop. The default program shipped with Axon.

    Each turn: ask the agent for an action, step the environment with that
    action, feed the resulting observation back into the agent. Repeat until
    the env reports ``done`` or ``max_steps`` is reached. Used by math, code,
    FrozenLake, search-r1, SWE, and tool-using recipes.

    Selected via ``program.name: react`` in a recipe yaml. Workflows that
    don't fit a single linear program flow (parallel solvers, multi-agent,
    search trees) subclass :class:`~axon.programs.base_program.BaseProgram`
    directly; see ``recipes/parallel_thinker/`` for an example.

    Each rollout instantiates one env and one agent. Per-rollout selection
    happens in :meth:`init_env_and_agent`, which reads the data row's
    ``env_name`` / ``agent_name`` against :data:`ENV_CLASS_MAPPING` /
    :data:`AGENT_CLASS_MAPPING`. The constructor's role is to *register* the
    classes the program may dispatch to and supply their default init kwargs.

    Args:
        agent_name: Import path of the agent class
            (``"recipes/<name>/agent.py:MyAgent"`` or ``"pkg://module:Cls"``).
            A list registers multiple agent classes; the data row picks which
            one each rollout uses.
        env_name: Same shape as ``agent_name`` for the environment.
        agent_args: Default kwargs forwarded to the agent's ``__init__``.
            When ``agent_name`` is a list, pass a parallel list of dicts to
            give each registered class its own defaults.
        env_args: Default kwargs forwarded to the env's ``from_dict``. Same
            list-shape convention as ``agent_args``.
        accumulate_thinking: Keep ``<think>`` content in conversation history
            across turns. ``True`` for cumulative-context recipes.
        accumulate_history: Keep prior assistant + observation messages in the
            conversation. Disable for step-mode (each turn sees only the
            current observation).
    """

    def __init__(
        self,
        agent_name: str | list[str],
        env_name: str | list[str],
        agent_args: dict | list[dict] | None = None,
        env_args: dict | list[dict] | None = None,
        accumulate_thinking: bool = True,
        accumulate_history: bool = True,
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
        self.accumulate_thinking = accumulate_thinking
        self.accumulate_history = accumulate_history

        # Config env/agent names trigger imports that register classes,
        # and store per-env/agent default args for init_env_and_agent.
        env_names = [env_name] if isinstance(env_name, str) else list(env_name)
        agent_names = [agent_name] if isinstance(agent_name, str) else list(agent_name)
        n = len(env_names)
        assert len(agent_names) == n, f"env_name/agent_name length mismatch: {n} vs {len(agent_names)}"
        env_args_list = _broadcast(env_args, n)
        agent_args_list = _broadcast(agent_args, n)

        # Per-class default args from config (e.g. {FrozenLakeEnv: {max_turns: 8}}).
        self._env_defaults: dict[type, dict] = {}
        self._agent_defaults: dict[type, dict] = {}
        for en, an, ea, aa in zip(env_names, agent_names, env_args_list, agent_args_list, strict=False):
            self._env_defaults[ENV_CLASS_MAPPING[en]] = dict(ea)
            self._agent_defaults[AGENT_CLASS_MAPPING[an]] = dict(aa)

        # Backward compat attrs
        self.env_class = ENV_CLASS_MAPPING[env_names[0]]
        self.agent_class = AGENT_CLASS_MAPPING[agent_names[0]]
        self.env_name = env_names[0] if len(env_names) == 1 else env_names
        self.agent_name = agent_names[0] if len(agent_names) == 1 else agent_names
        self.env_args = env_args if isinstance(env_args, dict) else {}

    def init_env_and_agent(self) -> tuple[BaseEnv, BaseAgent]:
        """Resolve env and agent independently from their registries.

        The data row is the source of truth:
          - ``env_name`` → look up in ENV_CLASS_MAPPING.
            Not found → SingleTurnEnvironment(reward_fn=env_name).
            Not specified → SingleTurnEnvironment.
          - ``agent_name`` → look up in AGENT_CLASS_MAPPING.
            Not specified → try env_name as agent name.
            Still not found → DefaultAgent.
        """
        from axon.core.agent import DefaultAgent
        from axon.core.env import SingleTurnEnvironment

        sample_args = dict(self.env_args)
        sample_env_name = sample_args.pop("env_name", None)
        sample_agent_name = sample_args.pop("agent_name", None)

        # -- Resolve env from registry, fallback to SingleTurn --
        env_cls = None
        if sample_env_name:
            try:
                env_cls = ENV_CLASS_MAPPING[sample_env_name]
            except (ValueError, KeyError):
                env_cls = SingleTurnEnvironment
                sample_args.setdefault("reward_fn", sample_env_name)
        if env_cls is None:
            env_cls = SingleTurnEnvironment

        # -- Resolve agent: explicit > env_name > default --
        agent_cls = None
        for name in [sample_agent_name, sample_env_name]:
            if name is None:
                continue
            try:
                agent_cls = AGENT_CLASS_MAPPING[name]
                break
            except (ValueError, KeyError):
                continue
        if agent_cls is None:
            agent_cls = DefaultAgent

        sample_args.pop("env_name", None)
        sample_args.pop("agent_name", None)

        # Merge per-class config defaults with row data (row overrides).
        env_defaults = self._env_defaults.get(env_cls, {})
        final_env_args = {**env_defaults, **sample_args}
        self.env = env_cls.from_dict(final_env_args)

        agent_defaults = self._agent_defaults.get(agent_cls, {})
        self.agent = agent_cls(**agent_defaults)
        return self.env, self.agent

    def format_messages(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        """Return conversation history for model interaction."""
        # Remove thinking from prior assistant messages.
        messages = deepcopy(messages)
        if not self.accumulate_thinking:
            for msg in messages:
                if msg["role"] == "assistant":
                    _, sep, after = msg["content"].partition("</think>")
                    if sep:
                        msg["content"] = after

        if not self.accumulate_history:
            if len(messages) <= 1:
                return messages
            return [messages[0], messages[-1]]
        return messages

    def add_observation_to_messages(self, observation: Any):
        if isinstance(observation, list) and all(isinstance(item, dict) for item in observation):
            self.messages.extend(observation)
        else:
            self.messages.append({"role": "user", "content": observation})

    def add_action_to_messages(self, action: Any):
        self.messages.append({"role": "assistant", "content": action})

    async def run(self):
        """Running a React agent loop using the new API server approach."""
        # Initialize environment and agent
        self.init_env_and_agent()

        # Reset agent
        self.agent.reset()
        # Reset environment.
        observation, info = self.env.reset()
        # Update agent internal state from environment.
        processed_observation = self.agent.process_observation(
            observation=observation,  # Raw observation from environment
            reward=0.0,
            done=False,
            info=info,
        )

        # Construct initial messages. System is called later b/c it may depend on observation.
        system_prompt = self.agent.system_prompt
        self.messages = [{"role": "system", "content": system_prompt}] if system_prompt else []
        self.add_observation_to_messages(processed_observation)

        # statistics
        steps = 0
        reward = 0
        done = False
        while not done:
            # Use engine API to generate responses
            model_response, stop_program, _ = await self.generate(
                messages=self.format_messages(self.messages),
                sample_params=self.sample_params,
            )
            self.add_action_to_messages(model_response)

            action: Action = self.agent.process_action(model_response)
            next_observation, reward, done, info = self.env.step(action.action)
            if stop_program:
                done = True

            steps += 1
            # Update agent internal state via environment interaction.
            # info is passed through from env.step() — env owns max_turns
            processed_observation = self.agent.process_observation(
                observation=next_observation,
                reward=reward,
                done=done,
                info=info,
            )
            self.add_observation_to_messages(processed_observation)

        # For special environment such as SWE Agent env (where reward must be computed after the program is complete)
        if hasattr(self.env, "post_compute_reward"):
            reward = self.env.post_compute_reward()

        if steps == 0:
            colorful_print(
                "Warning: Program completed before able to perform 1 complete action. This might cause unexpected behavior. Consider increasing program timeout limit.\n",
                "red",
            )

        self.env.close()

        # Return final reward and done; BaseProgram.__call__ will end the session
        return ProgramResult(reward=reward, done=done)
