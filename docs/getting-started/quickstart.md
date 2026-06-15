# Quickstart

From a fresh install to a running RL job in five minutes. The example uses **FrozenLake**, a small grid-world environment that exercises the full agent → environment → trainer loop without external services.

## Before you start

Make sure you have:

- Followed [Installation](installation.md) and the post-install `huggingface-cli login` / `wandb login` steps.
- 8× H100 (or B200) on one node — the quickstart recipe is laid out for 8 GPUs (actor TP×PP, sampler TP). For fewer GPUs, switch to a 7B/14B recipe.
- ~60 GB of free disk for the Qwen3-30B-A3B weights (≈30B params in bf16).

## 1. Generate the dataset

```bash
cd recipes/frozenlake/
python data.py
```

This builds the train/test datasets from a procedural grid-world generator and writes parquet files into `data/frozenlake/` (under the repo root, where the recipe's `train_files` points). The script is fully offline — it generates numeric env configs and makes no network calls.

## 2. Pre-cache the model weights

```bash
huggingface-cli download Qwen/Qwen3-30B-A3B
```

Pre-caching is optional but avoids streaming weights at the start of training.

## 3. Launch training

```bash
./train_frozenlake_qwen_30b_a3b.sh
```

While it runs you see, in order:

1. **Preflight pass** — config parsing, model materialization, and a fail-fast `validate_config.py` check before any GPU memory is touched.
2. **Trainer / sampler bring-up** — Ray actors come online, FSDP shards or Megatron parallel groups are initialized, and the vLLM sampler ingests the initial weights.
3. **Step 1** — programs run on the controller (sync mode), each calling the engine for multi-turn rollouts. Training metrics start streaming to W&B.

## 4. Watch the run

The script streams training logs to your console and metrics to W&B (loss / reward / IS-correction plots); checkpoints land in `output_dir` (HF format by default; `sharded` or `both` via `checkpoint_format`).

For CLI run management — list, inspect, tail, cancel — launch with `axon train -c <recipe>.yaml` instead of the script; that path registers the run so these commands can find it:

```bash
axon status              # list active and recent runs
axon status <run_id>     # detailed status of one run
axon logs <run_id> -f    # tail logs
axon cancel <run_id>     # stop the run cleanly
```

## What just happened

Each `SyncPPO` step samples rollouts from the program runner, computes advantages (RLOO in this recipe), recomputes the proximal logprob on the trainer side, runs the PPO loss for the configured PPO epochs, and syncs the updated weights back to the sampler. In hybrid mode the sampler and trainer share GPUs and weight sync is an in-place mode switch.

For the full mental model, see the [Architecture overview](../core-concepts/architecture.md).

## Try a different recipe

Once FrozenLake runs, the same flow applies to every recipe — each folder has a README with its setup, run command, and algorithm details:

| Recipe | Domain |
|---|---|
| [`recipes/math/`](https://github.com/modelcorp/axon/tree/main/recipes/math) | Math reasoning, single-turn rollouts |
| [`recipes/code/`](https://github.com/modelcorp/axon/tree/main/recipes/code) | Competitive programming with code execution |
| [`recipes/swe/`](https://github.com/modelcorp/axon/tree/main/recipes/swe) | Repo-level software engineering (SWE-Bench) |
| [`recipes/search_r1/`](https://github.com/modelcorp/axon/tree/main/recipes/search_r1) | Search-augmented reasoning |
| [`recipes/tools/`](https://github.com/modelcorp/axon/tree/main/recipes/tools) | Tool-using agents, MCP integration |

## Common first-run issues

??? failure "OOM during model materialization"
    Shard the model across more GPUs — raise `actor.megatron.tensor_model_parallel_size` (or `pipeline_model_parallel_size`) — turn on `param_offload` / `optimizer_offload`, lower `sampler.gpu_memory_utilization`, or use a smaller model. The shell scripts assume the documented GPU count; a single-node setup should drop to a 7B or 14B model.

??? failure "vLLM hangs at first generate call"
    Almost always a pinned-vLLM-fork mismatch. Verify `pip show vllm` matches `modelcorp/v0.19.0` (or your pinned version). The user-mode install (`bash install/install.sh`) handles this automatically; dev mode requires the explicit editable-install step.

??? failure "Trainer-sampler logprob mismatch is huge"
    The first step's mismatch is expected to be small but non-zero. If `batch/sampler_probs_diff_mean` is >> 0.005, check that you are using the Axon vLLM fork (not upstream) and that the model's monkey-patches have loaded. See the [sampler-trainer agreement page](../core-concepts/sampler-trainer-agreement.md).
