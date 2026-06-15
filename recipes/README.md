# Recipes

Worked examples covering math reasoning, code, agentic environments, search,
software engineering, tools, games, and environment hubs. Treat the catalog by
maturity: documented recipes are the best starting points; additional
directories are useful scaffolds or integrations whose external services and run
status should be checked locally.

Each recipe folder below has its own README with setup, run, and algorithm details.

## Layout

A reasoning recipe ships four pieces:

| File | Purpose |
|---|---|
| `data.py` | Generates the parquet train / val datasets |
| `env.py` | Subclass of `BaseEnv`  |
| `agent.py` | Subclass of `BaseAgent` |
| `train_*.sh` or `train_*.yaml` | Wrapper for `axon train` with sensible defaults |

Some recipes also ship recipe-local prompts (`prompts.py`) or extra glue (`reward.py`, `prompts.py`, `mcp_connection_manager.py`, …).

## Catalog

### Reasoning and reward-graded tasks

| Recipe | Domain | Notes |
|---|---|---|
| [`frozenlake/`](frozenlake/) | Grid-world RL — the smoke-test recipe | Launch yaml plus per-family layout wrappers; check run notes before treating a wrapper as support |
| [`math/`](math/) | Math reasoning, single-turn | Dataset/model variants; maturity depends on the specific launcher |
| [`code/`](code/) | Competitive programming with sandboxed execution | DeepCoder defaults |
| [`swe/`](swe/) | Repo-level software engineering (SWE-Bench) | Needs r2egym + per-task containers |
| [`search_r1/`](search_r1/) | Search-augmented reasoning | Standalone retrieval server |
| [`tools/`](tools/) | Tool-using agents, MCP integration | E2B / local / web tools |
| [`2048/`](2048/) | 2048 puzzle, multi-turn | Small-model demo |
| [`sudoku/`](sudoku/) | Sudoku solving | Small-model demo |
| [`geo3k/`](geo3k/) | Geometry QA, multimodal | Check model/backend notes before treating as publicly supported |
| [`miniwob/`](miniwob/) | Web interaction (simplified) | BrowserGym subset |
| [`webarena/`](webarena/) | Full web-browsing agent | Needs WebArena gateway, set `WEBARENA_URL` |
| [`formal_math/`](formal_math/) | Formal theorem proving | Lean-style, kimina wrapper |
| [`router/`](router/) | Routing / expert-selection task | Multi-turn |

### Workflow patterns

| Recipe | What's different |
|---|---|
| [`parallel_thinker/`](parallel_thinker/) | Custom `ParallelThinkerProgram` — N solvers + rewriter + selector. Showcase of `BaseProgram` extension beyond the shipped `ReactProgram`. |
| [`multi_env/`](multi_env/) | Single training step mixing multiple environments via per-row `env_name` / `agent_name` routing. |
| [`nemo_gym/`](nemo_gym/) | NeMo-Gym environment hub integration. |
| [`verifiers/`](verifiers/) | Verifiers Environments Hub integration. |

### Utilities

| Recipe | What's there |
|---|---|
| [`conversion/`](conversion/) | HF → quantised converters (INT4, MXFP8). |
| [`eval/`](eval/) | Eval-only data preparation. |
