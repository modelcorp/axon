# SWE — Software Engineering

Repo-level multi-turn agent for SWE-Bench-style tasks. The agent navigates a real codebase — reads files, applies patches, runs tests — and iterates until the failing test passes. Reward is binary: did the patched code pass the previously-failing test?

## Run

SWE training is a multi-step setup:

```bash
cd recipes/swe/
python data.py            # download the SWE datasets (R2E-Gym-Subset train / SWE-Bench-Verified val), write task lists to parquet
python cache_images_k8.py # pre-pull per-task Docker images onto your K8s nodes
./train_deepswe_fsdp.sh
```

Image caching is the long pole — pulling a few thousand task images onto every node can take hours. Skip it only if the images are already cached or you're running a small subset.

## Sandbox: R2E-Gym

The environment wraps the [R2E-Gym](https://github.com/R2E-Gym/R2E-Gym) API, which manages per-task containers and exposes a tool interface — `file_editor`, `execute_bash`, `search`, `finish`. `agent.py`'s tool surface is a thin wrapper around these.

## Files

| File | Purpose |
|---|---|
| `agent.py` | `SWEAgent` — repo context + tool surface, parses tool calls |
| `env.py` | `SWEEnv` — wraps the r2egym repo-interaction API |
| `data.py` | Pulls the SWE datasets (R2E-Gym-Subset for training, SWE-Bench-Verified for validation) and writes task lists to parquet (containers are created at env reset) |
| `cache_images_k8.py` | Pre-pulls per-task Docker images onto all Kubernetes nodes (via a DaemonSet) |
| `prompts.py` | Recipe-local prompts |

## Algorithm

Default model `Qwen/Qwen3-32B`; PPO loss + RLOO (`loop`) advantage, asymmetric clip (high `0.28`), `token_reduce: mean-norm`, `batch_reduce: step-mean`, FSDP, 8 nodes. `partial_rollout` isn't set by default — enable it for long debugging chains.

## Customize

- **Tools** — add or remove tools in `agent.py` and the matching handlers in `env.py`.
- **Reward** — defaults to binary test-pass; add partial credit for compile-success or newly-passing tests.
- **Container backend** — r2egym supports backends other than Docker.
- **Task subset** — filter in `data.py` to run one repo at a time.

SWE rollouts pile file contents into the conversation — use a 32K+ context model, and consider context-parallel (CP) on the trainer if you hit limits.
