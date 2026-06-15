# Add a program

A **program** is Axon's rollout abstraction — it defines what *one rollout is*. Adding a new task means either reusing the shipped program or writing your own. Two paths:

- **The ReAct path** — for the common `agent → action → environment → observation → repeat` shape (math, code, FrozenLake, search, SWE, tools). You write an **environment** and (optionally) an **agent**, and reuse the shipped `ReactProgram`. Most tasks take this path.
- **The custom-program path** — for any other shape (parallel solvers, multi-agent, search trees, test-time search). You subclass `BaseProgram` and write the control flow as plain async Python.

This guide walks both. For the concepts behind them — the program / agent / environment contracts, the tool surface, reward graders, and external integrations — see [Programs, agents, and tools](../core-concepts/programs-agents-and-tools.md); this page is the hands-on counterpart.

## Data

A training example is one row of a `.parquet` or `.jsonl` file, and **there is no fixed schema**. `RLDataset` (`axon/data/dataset.py`) loads the rows as plain dicts; each row's dict is handed to your environment's `from_dict(...)` (ReAct path) or is available to your program. The environment decides which fields it needs — a question, a ground-truth answer, image paths, a retrieval key, a per-sample `max_turns`, anything.

So `data.py` is just "write the dicts your task understands":

```python
# recipes/myrecipe/data.py
import pandas as pd

rows = [{"question": "What is 21 + 21?", "answer": "42"}, ...]   # whatever your env / reward reads
pd.DataFrame(rows).to_parquet("data/train.parquet")
```

A row commonly carries a prompt plus ground-truth fields, but the convention is yours. On the ReAct path a row may also name its own `env_name` / `agent_name` / `reward_fn` to select per-row (useful for mixed-task datasets).

---

## Path A — the ReAct path (environment + agent)

The shipped `ReactProgram` runs the loop for you: reset the environment, ask the agent for an action, `step` the environment, feed the observation back to the agent, repeat until the env reports `done`. You supply two pieces — an environment, and optionally an agent.

### The environment

An environment owns world state, the `step` transition, and the reward. Subclass one of two ready-made shapes; both register with `@register_env("name")` and implement a `from_dict` factory that builds the env from a data row.

**Single-turn** (one observation, one action, one reward — math-style). You usually don't need a subclass at all: reuse `SingleTurnEnvironment` and give it a reward function — a callable `(task_info, action) -> RewardOutput`, or a name from `REWARD_FN_REGISTRY` (`"math"`, `"code"`, `"f1"`, `"gpqa"`, `"ifbench"`, …):

```python
# recipes/myrecipe/env.py — only needed for a custom reward
from axon.core.env import register_env, SingleTurnEnvironment  # noqa: F401
from axon.utils.rewards.base import RewardOutput

def my_reward_fn(task_info: dict, action: str) -> RewardOutput:
    correct = action.strip().endswith(str(task_info["answer"]))
    return RewardOutput(reward=1.0 if correct else 0.0)
```

Point the recipe at `my_reward_fn` (or a registry name) and `SingleTurnEnvironment` does the rest — no subclass required.

**Multi-turn** (several rounds before a terminal reward — FrozenLake, SWE). Subclass `MultiTurnEnvironment`, implement `from_dict`, and supply the transition via `get_reward_and_next_obs` (the default `step` calls it and terminates at `max_turns`; override `step` directly if you need custom termination):

```python
# recipes/myrecipe/env.py
from typing import Any
from axon.core.env import register_env, MultiTurnEnvironment

@register_env("guessing")
class GuessingEnv(MultiTurnEnvironment):
    @staticmethod
    def from_dict(env_args: dict) -> "GuessingEnv":
        return GuessingEnv(task=env_args, max_turns=env_args.get("max_turns", 8))

    def get_reward_and_next_obs(self, task: dict, action: Any) -> tuple[float, dict]:
        guess = int(str(action).strip())
        if guess == task["secret"]:
            self.done = True                       # end early on success
            return 1.0, {}
        return 0.0, {"observation": "higher" if guess < task["secret"] else "lower"}
```

`step` returns `(next_observation, reward, done, info)`; `reset` returns `(observation, info)`.

### The agent

The agent turns environment observations into chat messages and the LLM's reply into an action. Subclass `BaseAgent` and implement two methods (plus an optional `system_prompt` and `reset`):

```python
# recipes/myrecipe/agent.py
from axon.core.agent import register_agent, BaseAgent, Action

@register_agent("guessing")
class GuessingAgent(BaseAgent):
    @property
    def system_prompt(self) -> str:
        return "Guess the secret number. Reply with just the number."

    def process_observation(self, observation, reward, done, info, **kwargs):
        # Turn the env observation into a chat message (a string, or a dict / list of messages).
        return observation["observation"] if isinstance(observation, dict) else str(observation)

    def process_action(self, response: str) -> Action:
        # Parse the model's text into an Action; `.action` is what env.step receives,
        # `.thought` captures any reasoning trace.
        return Action(thought="", action=response)
```

If the task needs no transformation, skip the agent entirely — the shipped `DefaultAgent` (`agent_name: default`) treats the LLM response as the action verbatim and forwards observations with minimal handling (it reads the `question` field from a dict observation, otherwise stringifies it).

### Wire it in the recipe

`ReactProgram` resolves the env and agent per rollout from `env_name` / `agent_name` — a `file.py:Class` path (imported and registered at runtime), a `pkg://module:Class` path, or an already-registered name:

```yaml
program:
  name: react
  env_name: recipes/myrecipe/env.py:GuessingEnv
  agent_name: recipes/myrecipe/agent.py:GuessingAgent
  env_args: {max_turns: 8}          # default kwargs forwarded to env.from_dict
```

For a single-turn task with a custom grader, point `env_name` at `SingleTurnEnvironment` and pass `env_args: {reward_fn: recipes/myrecipe/env.py:my_reward_fn}` (or a registry name). On the CLI the same fields are `+program.env_name=…`, `+program.agent_name=…`, `+program.env_args.max_turns=8`.

---

## Path B — the custom program

When the rollout isn't `agent → action → env → repeat`, subclass `BaseProgram` and write the flow yourself. You implement one async method, `run()`; inside it you call `self.generate(messages)` — which returns `(response_text, stop, step_idx)` — as many times and in whatever structure you want, then return a `ProgramResult`:

```python
# recipes/myrecipe/program.py
from axon.programs.base_program import register_program, BaseProgram, ProgramResult

@register_program("guessing_game")
class GuessingGame(BaseProgram):
    async def run(self) -> ProgramResult:
        secret, messages = 42, [{"role": "user",
            "content": "Guess my number in [1, 100]. I'll reply 'higher' or 'lower'."}]
        for _ in range(8):
            reply, stop, _ = await self.generate(messages=messages)
            guess = int(reply.strip())
            if guess == secret:
                return ProgramResult(reward=1.0, done=True)
            messages += [{"role": "assistant", "content": reply},
                         {"role": "user", "content": "higher" if guess < secret else "lower"}]
            if stop:
                break
        return ProgramResult(reward=0.0, done=True)
```

The engine handles everything else: every `generate` call — across all turns and any branches you fan out — is recorded into the rollout's prefix tree, tokenized once, and read off as trainer-ready sequences with response masks and sampler logprobs. You never touch tokens. Parallel solvers, multi-agent debates, and search trees are the same pattern with richer control flow; `recipes/parallel_thinker/` is the worked example (N solvers in parallel → N rewriters in parallel → a selector). The data row's fields are available to the program, so `run` can branch on per-sample inputs.

Select it with `program.name: guessing_game`, pointing the recipe at the file so the class registers.

---

## The recipe and the run

The recipe ties the program to a model, an algorithm, and a GPU layout. The algorithm is a `loss` + `advantage` pair (see [Add an algorithm](add-an-algorithm.md)); a verifiable-reward task uses `loss: ppo` + `advantage: grpo`:

```bash
axon train -- \
  model_path=Qwen/Qwen2.5-1.5B-Instruct strategy=fsdp hybrid_engine=true \
  loss=ppo advantage=grpo decoding.n=8 \
  train_files=recipes/myrecipe/data/train.parquet \
  program.name=react \
  +program.env_name=recipes/myrecipe/env.py:GuessingEnv \
  +program.agent_name=recipes/myrecipe/agent.py:GuessingAgent
```

The shipped recipes are the templates — copy the closest one (each `recipes/<name>/` has a README with its run command and algorithm). While it runs, watch `batch/reward/mean` (should climb) and `batch/sampler_probs_diff_mean` (the sampler-trainer gap — should stay small; see [Sampler-trainer agreement](../core-concepts/sampler-trainer-agreement.md)).

## Where to go next

- [Programs, agents, and tools](../core-concepts/programs-agents-and-tools.md) — the full conceptual surface: tool-call parsers, the reward-function library, MCP, and external-environment hubs.
- [Add an algorithm](add-an-algorithm.md) — a new loss or advantage estimator.
- [Add a model family](add-a-model.md) — bring up a new model.
