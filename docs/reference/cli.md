# CLI

The `axon` command is installed as a console script by `pip install -e .` (entry point: `axon.cli:main`). It exposes four subcommands.

```bash
axon --help
```

## `axon train`

Launch an agent-PPO training run.

```bash
axon train [OPTIONS] [-- HYDRA_OVERRIDES...]
```

| Flag | Type | Default | Maps to |
|---|---|---|---|
| `--config`, `-c` | path | — | User YAML overrides (flattened to `++key=value` Hydra overrides). |
| `--model`, `-m` | str | — | `model_path` |
| `--train-data` | str | — | `train_files` |
| `--val-data` | str | — | `val_files` |
| `--gpus` | int | — | `num_gpus_per_node` |
| `--nodes` | int | — | `num_nodes` |
| `--experiment-name` | str | — | `experiment_name` |
| `--output-dir` | str | — | `output_dir` |
| `--resume` | str | — | `resume_from_checkpoint` |
| `--dry-run` | flag | false | Resolve and print the final config, then exit. Validation runs first. |
| `--foreground`, `--fg` | flag | false | Run in-process (blocking) instead of detaching. |
| `HYDRA_OVERRIDES` | varargs | — | Anything after `--` is passed straight to Hydra. |

### Override precedence

Last writer wins:

```
base config (config.yaml + defaults block)
  → YAML file (--config), as ++ overrides
  → CLI flags (--model, --gpus, ...), as bare = overrides
  → raw Hydra overrides (after --)
```

So `axon train --config recipe.yaml --gpus 4 -- actor.fsdp.fsdp_size=2`:

1. Loads `config.yaml` defaults.
2. Applies `recipe.yaml` overrides.
3. Sets `num_gpus_per_node=4` from the flag.
4. Sets `actor.fsdp.fsdp_size=2` from the raw Hydra override.

### Behavior

1. **Compose** the final `DictConfig` via the override chain above.
2. **Validate** the config (`validate_axon_config`). On error: fast-fail without launching workers.
3. **Dry run** (`--dry-run`): print the resolved config and exit.
4. **Foreground mode** (`--foreground` / `--fg`): create the run dir under `~/.axon/runs/<run_id>/`, persist `config.yaml`, tee stdout+stderr into `train.log`, and call `run_ppo_agent(cfg)` in-process.
5. **Async mode** (default): create the run dir, then spawn a detached child process (`start_new_session=True`) that re-loads the config from `<run_dir>/config.yaml` and calls `run_ppo_agent`. The parent updates `meta.json` and prints the run ID.

### Raw Hydra overrides

Anything after `--` is passed through. Examples:

```bash
axon train -- strategy=megatron actor.megatron.tensor_model_parallel_size=4
axon train -- 'sampler.engine_kwargs.vllm={swap_space: 8}'
axon train -- '++sampler.profiler.enable=true'
```

This gets you the full Hydra surface area: `+` (force-add), `++` (replace-or-add), group selectors, list literals, etc.

## `axon status`

```bash
axon status              # list all known runs
axon status <run_id>     # detailed status of one run (prefix-matches if < 8 chars)
```

States: `starting` (very brief), `running`, `completed`, `crashed`, `cancelled`. The `running` → `crashed` flip happens lazily, when `axon status` notices the recorded PID is no longer alive.

## `axon logs`

```bash
axon logs <run_id>              # last 50 lines
axon logs <run_id> -n 200       # last 200 lines
axon logs <run_id> -f           # follow (poll-based tail)
```

`-n / --lines` defaults to 50. `-f / --follow` follows the log by polling.

## `axon cancel`

```bash
axon cancel <run_id>
```

Sends `SIGTERM` to the recorded PID and flips status to `cancelled`. Behavior:

1. Resolve run.
2. Refuse if status is not `running`.
3. Refuse if `pid <= 0` (run still in `starting` window).
4. Send `SIGTERM`.
5. Update `meta.json` with `status="cancelled"`.

There is no second-pass `SIGKILL`. Ray actors spawned by `run_ppo_agent` rely on their own signal handlers and on the `start_new_session=True` session-leader behavior to tear down cleanly.

## Run state location

Every run gets a directory `~/.axon/runs/<run_id>/`:

| File | Content |
|---|---|
| `meta.json` | Run metadata: `run_id`, `pid`, `status`, `started` (ISO8601 UTC), `log_path`, `base`, `experiment`. |
| `config.yaml` | Fully-resolved Hydra `DictConfig`, dumped pre-launch. |
| `train.log` | Combined stdout + stderr from the worker. |

Run IDs are 8-char hex. Any unique prefix is accepted by the run-targeting subcommands (`status`, `logs`, `cancel`).

To use a different runs directory, set `AXON_RUNS_DIR` (read by `axon/cli/_runs.py`).
