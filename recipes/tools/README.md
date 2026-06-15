# Tools

General-purpose tool-using agent: it picks from a tool registry, calls a tool, observes the result, and uses it in further reasoning. Ships with a calculator and a Python interpreter; `mcp_connection_manager.py` reaches any MCP server.

## Run

```bash
cd recipes/tools/
python ../math/data.py    # the tools recipe trains on math data
./train_fsdp.sh
```

With E2B sandboxes:

```bash
export E2B_API_KEY=<key>
TOOLS="calculator=math_tools/calculator.py:CalculatorTool e2b_python=code_tools/e2b_tool.py:E2BPythonInterpreter" ./train_fsdp.sh
```

## Tool surface

| Tool | Source | Purpose |
|---|---|---|
| `calculator` | `math_tools/calculator.py` | Basic arithmetic: add, subtract, multiply, divide |
| `python` | `code_tools/python_interpreter.py` | Local Python interpreter |
| `e2b_python` | `code_tools/e2b_tool.py` | E2B cloud-sandboxed execution (`E2BPythonInterpreter`) |
| `python` (LCB backend) | `code_tools/lcb_tool.py` | LiveCodeBench-sandboxed runner (`LCBPythonInterpreter`, registers as `python`) |
| `tavily_search` / `tavily_extract` / `firecrawl` / `google_search` | `web_tools/` | Web search & page extraction |
| any MCP tool | `mcp_connection_manager.py` | Anything your MCP servers expose |

Register a tool by adding `name=path.py:ClassName` to the `TOOLS` env var; add a new one by subclassing `Tool` (`axon/tools/tools.py`). The full surface is documented in [Programs, agents, and tools](../../docs/core-concepts/programs-agents-and-tools.md).

## Algorithm

Default model `Qwen/Qwen3-4B`; `loss: gspo` (sequence-level, tight clip `0.003` / `0.005`), `advantage: rloo`, FSDP, one node. `LOSS_MODE=ppo ./train_fsdp.sh` switches to PPO with the wider `0.2` / `0.28` clip.

## Tool-call parser

Pick a parser from `axon/tools/parsers/` that matches your model (Qwen, OpenAI Harmony, GLM, R1, JSON, XML) — a mismatched parser silently produces malformed calls.
