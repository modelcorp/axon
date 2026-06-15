# Programs, agents, and tools

How Axon expresses agentic rollouts — the program abstraction, agent and environment helpers, the tool surface, reward graders, and external integrations.

## Program is the rollout abstraction

A **program** defines what one rollout is. It owns the workflow — when to call the LLM, what to do with each response, when to terminate. The engine underneath supplies the plumbing the trainer depends on: tokenization, partial-rollout suspend/resume, and the per-token training signals (response masks, sampler logprobs, MoE routing decisions) read off the rollout's prefix tree.

`BaseProgram` (`axon/programs/base_program.py`) is the extension point. Subclass it, implement `run`, register with `@register_program("name")`, and the recipe yaml refers to it as `program.name: <name>`.

### The common pattern: ReactProgram + Agent + Environment

The shipped **`ReactProgram`** (`axon/programs/react_program.py`) drives a ReAct-style multi-turn loop and covers math, code, FrozenLake, search, SWE, and tool-using recipes. It pairs with two helper abstractions:

- **`BaseAgent`** (`axon/core/agent.py`) — prompt construction and response parsing.
- **`BaseEnv`** (`axon/core/env.py`) — world state, the `step` transition, and reward.

Both helpers register the same way (`@register_agent`, `@register_env`) and are wired together in the recipe yaml via `program.agent_name` and `program.env_name`.

### Custom programs

When the rollout shape isn't `agent → action → env → obs → repeat`, write a custom program. `recipes/parallel_thinker/` is the worked example: a `ParallelThinkerProgram` that runs N solver calls in parallel, then N rewriters, then a selector — no `BaseAgent`, no `BaseEnv`. Search trees, multi-agent dialogues, and MCTS-style test-time-search fit the same pattern. Subclass `BaseProgram`, override `run`, and reuse the partial-rollout, tokenization, and signal-emission machinery.

For a step-by-step walkthrough, see [Add a program](../guides/add-a-program.md).

## Tools

### Tool-call parsers

Each major LLM family formats tool calls differently. Axon ships a parser per format:

| Parser | Format |
|---|---|
| `JsonToolCallParser` | OpenAI-style JSON tool calls |
| `XMLToolCallParser` | XML-tagged tool calls |
| `QwenToolCallParser` | Qwen's chat-template tool format |
| `Gemma4ToolCallParser` | Gemma4's native tool format |
| `OpenAIHarmonyToolCallParser` | OpenAI Harmony format (GPT-OSS) |
| `GlmToolCallParser` | GLM-family format |
| `R1ToolCallParser` | DeepSeek-R1 format |

All inherit from `ToolCallParser` (`axon/tools/parsers/base_parser.py`) and are registered through `@register_parser(...)`. To support a new model's tool format, write one parser file and decorate it.

### Tool definitions

```python
from axon.tools.tools import Tool
from axon.tools.types import ToolOutput

class WeatherTool(Tool):
    def __init__(self):
        super().__init__(name="weather", description="Get current weather")

    @property
    def json(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {"type": "object", "properties": {...}},
            },
        }

    def forward(self, location: str) -> ToolOutput:
        return ToolOutput(name=self.name, output=fetch_weather(location))
```

For async / network-bound tools, `MCPTool` is the analogue with `async_forward`. Any [Model Context Protocol](https://modelcontextprotocol.io) server can be wrapped as a tool through `MCPTool` and surfaces in the same registry.

### Tool executors

`LocalToolExecutor` and `HTTPToolExecutor` (`axon/tools/executors.py`) handle the dispatch, and both support batched calls. `LocalToolExecutor` accepts Tool instances, classes, or import-path strings and bridges async / sync (it implements the synchronous entry points). `HTTPToolExecutor` is async-only and holds no Tool instances — it POSTs to a tool server and schemas are managed externally.

## Reward function library

| Reward | What it grades |
|---|---|
| `math_reward_fn` | Math problems (numeric and symbolic equality via sympy / math-verify) |
| `code_reward_fn` | Code execution against unit tests (HumanEval+, LiveCodeBench, TACO) |
| `f1_reward_fn` | Token-level F1 (NQ-style QA) |
| `gpqa_reward_fn` | GPQA-style multiple choice |
| `ifbench_reward_fn` | IFBench instruction-following |
| `remote_reward_fn` | Hits a configurable HTTP endpoint — for custom graders |

## External integrations

- **Tinker SDK** — Axon ships a Tinker-SDK-compatible training and sampling client, so code targeting Tinker runs against Axon-managed GPUs with minimal change.
- **MCP** — Any Model Context Protocol server is callable as an `MCPTool`, registered alongside native tools.
- **Environment hubs** — Adapters for the [Verifiers Environments Hub](https://github.com/PrimeIntellect-ai/verifiers) (`recipes/verifiers/`) and NeMo Gym (`recipes/nemo_gym/`). Environments registered in either hub load directly from a recipe.
- **OpenAI-compatible HTTP** — With `engine_endpoint.enable=true`, the engine exposes an Axon-native API and an OpenAI-compatible chat-completions endpoint. External agents (LangChain, OpenAI Agents SDK, Anthropic SDK, custom clients) drive rollouts by pointing their `base_url` at the engine while still feeding the RL loop.
- **Code execution** — Managed E2B cloud sandboxes, a local Python executor, and LiveCodeBench runners share the agent's tool surface.
