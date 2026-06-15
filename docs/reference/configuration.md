# Configuration

Axon uses [Hydra](https://hydra.cc) for configuration. The root config lives at `axon/config/config.yaml`, with `trainer/trainer.yaml` and `sampler/sampler.yaml` imported via `defaults:`.

This page is the authoritative reference for every configuration knob. Use the right-hand TOC to navigate.

## How configuration is composed

The Hydra `defaults:` block at the top of `config.yaml`:

```yaml
defaults:
  - trainer/trainer@actor
  - trainer/trainer@ref
  - trainer/trainer@critic
  - trainer/trainer@reward_model
  - sampler/sampler@sampler
  - _self_
```

The `@<key>` syntax tells Hydra "load `trainer/trainer.yaml`, but graft its contents under the path `actor:` in the resulting config" (and so on for each role). `_self_` last means anything in `config.yaml` overrides values pulled in by the defaults.

So `actor`, `ref`, `critic`, and `reward_model` all share the same trainer template — and any field documented under [Trainer template](#trainer-template) applies to all four. Fields are then specialised under the matching key in `config.yaml` (e.g. `critic.optimizer_args.lr` overrides the template's default for the critic role).

### Override precedence

Last writer wins:

```
base config (config.yaml + defaults block)
  → YAML file passed via --config, applied as ++ overrides
  → CLI flags (--model, --gpus, ...) applied as bare = overrides
  → raw Hydra overrides passed after `--`
```

So a `-- actor.fsdp.fsdp_size=4` on the command line beats whatever the user yaml set, which beats the trainer template default.

To preview the resolved config without launching workers: `axon train --config <recipe.yaml> --dry-run`.

## Validation

`axon/config/validate_config.py` runs at startup (called from `axon train` before workers spawn). It dispatches to nine validators and fails fast with a clear error before any GPU memory is touched. Things it catches:

- Invalid `advantage` / `kl_reward` / `loss_args.{token_reduce, batch_reduce, sampler_is, sampler_rs}` enum values.
- `n_gpus % (TP × PP × CP) != 0` for Megatron actors.
- `real_train_batch_size` not divisible by the minimal possible batch size — derived from `strategy`, `n_gpus`, and (for Megatron) `TP×PP×CP` and `micro_batch_size_per_gpu`; checked only when `use_dynamic_bsz` is `false`.
- `train_batch_size < mini_batch_size`, or `mini_batch_size % micro_batch_size != 0`.
- Both `micro_batch_size` and `micro_batch_size_per_gpu` set, or neither set (where both legal forms are accepted).
- Sequence parallelism (`ulysses_sequence_parallel_size > 1`) without `use_remove_padding`.
- vLLM with LoRA rank > 512.
- `moe_replay` with `enable_prefix_caching` (incompatible — prefix-cached tokens skip the forward pass and therefore can't capture routing).

---

## Top-level config

### Model identity

| Field | Default | Type | Purpose |
|---|---|---|---|
| `model_path` | `~/models/deepseek-llm-7b-chat` | path | Path or HF ID for model weights. Inherited by actor, ref, and sampler unless overridden per-role. |
| `trust_remote_code` | `false` | bool | Trust remote code in HF model / tokenizer configs. |
| `external_lib` | `null` | str? | Python module to import before model init (e.g. for custom modeling code). |

### Strategy & distributed

| Field | Default | Type | Purpose |
|---|---|---|---|
| `strategy` | `fsdp` | enum | Trainer backend: `fsdp`, `fsdp2`, or `megatron`. |
| `hybrid_engine` | `true` | bool | `true` = actor and sampler share GPUs (mode-switch); `false` = separate GPU pools (NCCL P2P). |
| `num_nodes` | `1` | int | Number of nodes. |
| `num_gpus_per_node` | `8` | int | GPUs per node. |
| `sampler_trainer_gpu_ratio` | `1` | int | Ratio of sampler GPUs to trainer GPUs (disaggregated mode only). |
| `enable_ray_collective` | `false` | bool | Use Ray Collective for distributed communication. |

### Output & logging

| Field | Default | Type | Purpose |
|---|---|---|---|
| `output_dir` | `outputs` | path | Output base. Per-run path is `{output_dir}/{project_name}/{experiment_name}/`. |
| `project_name` | `axon` | str | W&B project key / output organisation. |
| `experiment_name` | `agent_ppo` | str | Run name. |
| `logger` | `[console, wandb]` | list[str] | Logger backends. Supports `console`, `wandb`, `mlflow`, `tensorboard`, `clearml`, `swanlab`, `trackio`, `file`. |

### Checkpointing

| Field | Default | Type | Purpose |
|---|---|---|---|
| `save_steps` | `-1` | int | Steps between checkpoints. `-1` disables saving. |
| `resume_from_checkpoint` | `true` | bool / str | `false` = fresh; `true` / `auto` = auto-resume from latest if it exists; explicit path also accepted. |
| `max_checkpoints_to_keep` | `null` | int? | Max checkpoints retained; `null` keeps all. Oldest deleted on excess. |
| `checkpoint_format` | `hf` | enum | `sharded` (training-only), `hf` (inference-ready), or `both`. |
| `async_save` | `false` | bool | Background-thread checkpoint save (Megatron only). |

### Training loop

| Field | Default | Type | Purpose |
|---|---|---|---|
| `driver_mode` | `null` | enum? | `sync` or `async`; `null` infers from `hybrid_engine`. |
| `mode` | `train` | enum | `train` or `eval` (validation only, no optimizer updates). |
| `ppo_epochs` | `1` | int | PPO epochs per training step. |
| `mini_batch_size` | `64` | int | PPO mini-batch size. Must divide evenly into `train_batch_size`. |
| `mini_batch_shuffle` | `false` | bool | Shuffle mini-batches within PPO epochs. |
| `mini_batch_seed` | `42` | int | Seed for mini-batch shuffle. |
| `total_training_steps` | `null` | int? | Total RL training steps. |
| `total_epochs` | `null` | int? | Total RL training epochs (alternative to steps). |
| `critic_warmup` | `0` | int | Steps to train only the critic before actor updates. |

### Data

| Field | Default | Type | Purpose |
|---|---|---|---|
| `train_files` | `null` | str / list[str] | Training file paths (`.parquet` or `.jsonl`). Validator warns if unset. |
| `data_mix` | `null` | list[float] | Per-file sampling weights, parallel to `train_files`. Forces `data_sampler=random`. |
| `val_files` | `null` | str / list[str] | Validation file paths. |
| `train_batch_size` | `64` | int | Training batch size. |
| `max_seq_length` | `8192` | int | Total token budget per row (single dimension for all per-position tensors). |
| `max_prompt_length` | `1024` | int | Initial-prompt truncation threshold. Validator enforces `≤ max_seq_length`. |
| `prompt_truncation` | `null` | enum? | `null` = terminate program; `left` / `right` = truncate. |
| `enable_one_off_pipeline` | `false` | bool | Enables one-off-pipeline mode (overlapped sample / train under disaggregated). |
| `moe_replay` | `false` | bool | Capture sampler routing decisions and replay them on the trainer. Megatron only. Incompatible with `sampler.enable_prefix_caching`. |
| `partial_rollout.enable` | `false` | bool | Enable partial rollout — long-tail trajectory recovery across weight updates. |
| `partial_rollout.n_iters` | `2` | int | Response is split into `n_iters` chunks; weight updates can land between them. |
| `data_sampler` | `random` | enum / path | `sequential`, `random`, `exp_weighted_curriculum`, `threshold_masking_curriculum`, or `pkg://...:Class` / `file://...:Class`. |
| `data_sampler_args.seed` | `42` | int | RNG seed. |
| `data_sampler_args.min_weight` | `0.2` | float | `exp_weighted_curriculum`: floor weight for high-pass-rate (easy) samples. |
| `data_sampler_args.max_weight` | `1.0` | float | `exp_weighted_curriculum`: ceiling weight. |
| `data_sampler_args.threshold` | `0.9` | float | `threshold_masking_curriculum`: solved-sample exclusion threshold. |

### Decoding

Used for training rollouts. Validation uses its own `validation.decoding.*` block.

| Field | Default | Type | Purpose |
|---|---|---|---|
| `decoding.temperature` | `1.0` | float | Sampling temperature. |
| `decoding.top_k` | `-1` | int | Top-k filter; `-1` disables. |
| `decoding.top_p` | `1.0` | float | Nucleus sampling threshold. |
| `decoding.repetition_penalty` | `1.0` | float | HF-style repetition penalty. |
| `decoding.n` | `1` | int | Rollouts per prompt. |
| `decoding.logprobs` | `1` | int | Number of top log-probabilities the sampler returns per generated token (vLLM / OpenAI `logprobs` param). |
| `decoding.temperature_schedule.enable` | `false` | bool | Enable temperature scheduling over training. |
| `decoding.temperature_schedule.scheduler` | `linear` | enum | `linear`, `exponential`, or `cosine`. |
| `decoding.temperature_schedule.start_temperature` | `1.0` | float | Start temperature. |
| `decoding.temperature_schedule.end_temperature` | `2.0` | float | End temperature. |
| `decoding.temperature_schedule.num_steps` | `1000` | int | Steps over which to apply the schedule. |

### Validation

| Field | Default | Type | Purpose |
|---|---|---|---|
| `validation.steps` | `-1` | int | Steps between validation passes; `-1` disables. |
| `validation.before_train` | `true` | bool | Run a validation pass before training starts. |
| `validation.decoding.temperature` | `0` | float | Validation temperature. `0` = greedy. |
| `validation.decoding.top_k` | `-1` | int | Validation top-k. |
| `validation.decoding.top_p` | `1.0` | float | Validation top-p. |
| `validation.decoding.n` | `1` | int | Validation rollouts per prompt. |
| `validation.decoding.repetition_penalty` | `1.0` | float | Validation repetition penalty. |

### Agent / program

| Field | Default | Type | Purpose |
|---|---|---|---|
| `program.name` | `react` | str | Program class name. Built-ins: `react`, `proxy`. Recipes can register custom names. |
| `program.env_name` | recipe-set | str | Environment class, e.g. `recipes/<name>/env.py:ClassName`. |
| `program.env_args` | `{}` | dict | Kwargs forwarded to the environment. |
| `program.agent_name` | recipe-set | str | Agent class, e.g. `recipes/<name>/agent.py:ClassName`. |
| `program.agent_args` | `{}` | dict | Kwargs forwarded to the agent. |
| `launch_on_head` | `true` | bool | Launch program on the Ray head node. |
| `max_steps` | `1000000000` | int | Safety cap on `engine.generate` calls per program. |
| `max_tokens_per_step` | `null` | int? | Per-step token cap. |
| `program_timeout` | `10800` | int (s) | Program generation timeout (3 h default). |
| `overlong_filter` | `false` | bool | Filter programs that exceed length / timeout / max steps. |
| `drop_zero_advantage_samples` | `true` | bool | Drop samples whose advantage is zero (no learning signal). |
| `stepwise_advantage_mode` | `broadcast` | str | Step-wise advantage mode. |
| `use_sampler_logprobs` | `false` | bool | Use sampler-emitted logprobs as `old_log_probs` (skip the trainer-side recompute). |
| `terminate_on_error` | `false` | bool | Terminate training after a program exceeds the retry limit. |
| `save_programs_flag` | `true` | bool | Persist chat-completion messages during training. |
| `filter_program_errors` | `false` | bool | Strict program-error filtering. |
| `retry_limit` | `1` | int | Retries per failing program. |
| `accumulate_history` | recipe | bool | Recipe-specific: keep cumulative conversation history across turns. |
| `accumulate_thinking` | recipe | bool | Recipe-specific: keep the agent's `<think>` content across turns. |
| `engine_args` | `{}` | dict | Extra kwargs to the agent execution engine. |
| `max_concurrency` | `0` | int | Max concurrent programs (`0` = unlimited). |
| `sampler_pausing_strategy` | `continue` | enum | `drain`, `hold`, `continue`, or `reset`. How the sampler handles in-flight requests during weight sync. |
| `sampler_channel` | `nccl` | enum | Async data channel: `nccl` or `ray`. |
| `offload_p2p_buffer` | `false` | bool | Offload P2P weight-transfer buffers to CPU between transfers (disaggregated only). |
| `use_dummy_batch` | `false` | bool | Use dummy batch for testing/debugging. |
| `memory_stress_test` | `false` | bool | Run two worst-case steps to validate config doesn't OOM, then exit. |

### Engine HTTP endpoint

When enabled, exposes both an Axon-native API and an OpenAI-compatible chat-completions endpoint, letting external agents drive rollouts by pointing their `base_url` at the engine.

| Field | Default | Type | Purpose |
|---|---|---|---|
| `engine_endpoint.enable` | `false` | bool | Enable HTTP endpoint. |
| `engine_endpoint.host` | `0.0.0.0` | str | Host to bind. |
| `engine_endpoint.port` | `9001` | int | Port to bind. |
| `engine_endpoint.force_port` | `True` | bool | Forcefully take the port if it's busy. |

### Global profiler

`global_profiler.tool` selects between `nsys`, `torch`, `torch_memory`, or `null` (off). Per-role profilers under `actor.profiler.*`, `sampler.profiler.*`, etc. inherit the global tool unless overridden.

| Field | Default | Type | Purpose |
|---|---|---|---|
| `global_profiler.tool` | `null` | enum? | `nsys`, `torch`, `torch_memory`, or `null`. |
| `global_profiler.steps` | `null` | list[int]? | Specific steps to profile. |
| `global_profiler.profile_continuous_steps` | `false` | bool | Continuous (vs step-by-step) profiling. |
| `global_profiler.save_path` | `outputs/profile` | path | Profiler output directory. |

NSYS-specific options live under `global_profiler.global_tool_config.nsys.*` (controller / worker nsight options, capture range, kill mode); `torch_memory` options under `global_profiler.global_tool_config.torch_memory.*`.

---

## Algorithm selection

Recipes pick an RL algorithm by composing a **loss** (the per-token objective) with an **advantage estimator** (the per-token target). Both are decorated registries — `@register_loss(LossFn.X)` in `axon/trainer/algos/loss/loss.py` and `@register_advantage(AdvantageFn.X)` in `axon/trainer/algos/advantages/advantage.py`. A method named in the literature is a loss + advantage composed in yaml — sometimes pulling in a dedicated loss (GSPO, CISPO) or advantage (GRPO), sometimes a pure composition (DAPO is the PPO loss with group-relative advantages and asymmetric clipping).

### Advantage

| Field | Default | Type | Purpose |
|---|---|---|---|
| `advantage` | `loop` | enum | One of `gae`, `grpo`, `rloo`, `loop` (alias of `rloo`), `reinforce_plus_plus`, `reinforce_plus_plus_baseline`, `remax`, `opo`, `grpo_passk`, `gpg`, `chunked_gae`, `identity`, `kimi_k1_5`. |
| `advantage_args.gamma` | `0.99` | float | Discount factor (GAE, REINFORCE++, chunked-GAE). |
| `advantage_args.lam` | `0.95` | float | GAE lambda — bias/variance trade-off (GAE, chunked-GAE). |
| `advantage_args.norm_adv_by_std` | `true` | bool | GRPO / GRPO-PassK / Kimi-K1.5: normalize group advantages by std. |
| `advantage_args.epsilon` | `1e-6` | float | Numerical stability constant. |
| `advantage_args.f_norm` | `1.0` | float | GPG: normalization factor. |
| `advantage_args.chunk_size` | `128` | int | Chunked GAE: chunk size for parallel prefix scan. |
| `advantage_args.length_coef` | `0.0` | float | Kimi-K1.5: length-shaping coefficient (1.0 = paper, 0 = disabled). |
| `advantage_args.length_coef_warmup_steps` | `0` | int | Kimi-K1.5: hold coefficient at 0 for `step < warmup`. |
| `advantage_args.length_coef_ramp_steps` | `0` | int | Kimi-K1.5: linear ramp after warmup (0 = hard switch). |
| `advantage_args.correct_threshold` | `1.0` | float | Kimi-K1.5: score ≥ counts as correct. |
| `advantage_args.clip_advantages` | `false` | bool | Clip advantages into `[-1, 1]`. |

### KL-in-reward

KL penalty applied to the token-level reward *before* advantage computation. Distinct from `loss_args.kl_coef`, which adds KL as an auxiliary loss term.

!!! note "Not implemented in the current driver"
    `kl_reward` is on the configuration surface but the consumer in `axon/driver/sync_ppo.py` raises `NotImplementedError`. Use `loss_args.kl_coef` for a KL term in the policy loss instead. Recipes that set `kl_reward_args.kl_coef` without setting `kl_reward` are silent no-ops.

| Field | Default | Type | Purpose |
|---|---|---|---|
| `kl_reward` | `null` | enum? | `null`, `kl`, `abs`, `mse`, `low_var_kl`, or `full`. |
| `kl_reward_args.type` | `fixed` | enum | `fixed` (constant β) or `adaptive` (tracks `target_kl`). |
| `kl_reward_args.kl_coef` | `0.001` | float | Initial / fixed KL coefficient. |
| `kl_reward_args.horizon` | `10000` | int | Adaptive: horizon for coefficient updates. |
| `kl_reward_args.target_kl` | `0.1` | float | Adaptive: target KL divergence. |

### Loss

| Field | Default | Type | Purpose |
|---|---|---|---|
| `loss` | `ppo` | enum | One of `ppo`, `gspo`, `gpg` (aliases REINFORCE), `clip_cov`, `kl_cov`, `geo_mean`, `cispo`, `reinforce`, `value`. |
| `loss_args.token_reduce` | `sum` | enum | See [Token / batch reduction](#token-batch-reduction). |
| `loss_args.batch_reduce` | `token-mean` | enum | See [Token / batch reduction](#token-batch-reduction). |
| `loss_args.clip_ratio` | `0.2` | float | Symmetric PPO clip range. Used when `clip_ratio_low`/`high` not set. |
| `loss_args.clip_ratio_low` | `0.2` | float | Asymmetric lower bound (falls back to `clip_ratio` when unset). |
| `loss_args.clip_ratio_high` | `0.2` | float | Asymmetric upper bound. The DAPO-style `0.2 / 0.28` split allows the high-reward tail to push harder. |
| `loss_args.clip_ratio_c` | `3.0` | float | Dual-clip threshold — caps the pessimistic objective for negative advantages. Must be > 1.0. |
| `loss_args.token_level_mask` | `false` | bool | CISPO: enable Eq 7 unified-formulation token mask. |
| `loss_args.clip_cov_ratio` | `0.0002` | float | CLIP_COV: fraction of tokens selected for covariance-based clipping. |
| `loss_args.clip_cov_lb` | `1.0` | float | CLIP_COV: covariance lower bound for selection. |
| `loss_args.clip_cov_ub` | `5.0` | float | CLIP_COV: covariance upper bound for selection. |
| `loss_args.kl_cov_ratio` | `0.0002` | float | KL_COV: fraction of high-covariance tokens for KL injection. |
| `loss_args.ppo_kl_coef` | `0.1` | float | KL_COV: KL penalty coefficient on selected tokens. |
| `loss_args.cliprange_value` | `0.5` | float | Value-loss clipping range (critic). |
| `loss_args.entropy_coef` | `0` | float | Entropy bonus coefficient. `0` disables. |
| `loss_args.kl_coef` | `0.0` | float | KL term coefficient added to the policy loss. Setting > 0 also requires a reference policy in the resource pool. |
| `loss_args.kl_type` | `low_var_kl` | enum | KL estimator. See [KL estimators](#kl-estimators). |
| `loss_args.sampler_is` | `null` | enum? | Importance-sampling correction mode. `null`, `token`, or `sequence`. |
| `loss_args.sampler_is_threshold` | `2.0` | float | Upper truncation for IS weights — clamped at this bound only (`min(weight, threshold)`); no lower clamp. |
| `loss_args.sampler_is_batch_normalize` | `false` | bool | Normalize IS weights to mean = 1.0 across batch after clipping. |
| `loss_args.sampler_rs` | `null` | enum? | Rejection-sampling mode. `null`, `token`, `sequence`, or `geometric`. |
| `loss_args.sampler_rs_threshold` | `null` | float? | Upper RS threshold. Lower defaults to `1 / upper`. |
| `loss_args.sampler_rs_threshold_lower` | `null` | float? | Explicit lower RS threshold. |
| `loss_args.sampler_token_veto_threshold` | `null` | float? | If any token's IS weight falls below this, the entire sequence is masked from the loss. |

### Token / batch reduction

The loss is reduced in two stages:

```
# Stage 1: token_reduce (B, T) -> (B,)
row_loss = aggregate(loss_mat * loss_mask, dim=-1)

# Stage 2: batch_reduce (B,) -> scalar
loss = aggregate(row_loss)
```

**`token_reduce`**:

- `sum` — raw sum across tokens. Standard PPO. `row_loss[i] = Σ_t loss[i,t]`.
- `mean` — divide by valid-token count per row. `row_loss[i] = Σ_t loss[i,t] / N_valid[i]`.
- `mean-norm` — divide by `T` (the fixed sequence-length dimension). Context-length independent. Recommended when batches mix sequences of very different lengths.
- `mean-program` — divide by per-program token count (multi-step pooling).

**`batch_reduce`**:

- `token-mean` — sum row losses, divide by total valid tokens. Standard PPO.
- `step-mean` — sum row losses, divide by valid step count. GRPO default — every step counts equally regardless of length.
- `program-mean` — sum row losses, divide by valid program count. Multi-step agent RL where each program (multi-turn rollout) gets equal weight.

Common compositions seen in shipped recipes:

| Style | `token_reduce` | `batch_reduce` |
|---|---|---|
| Standard PPO | `sum` | `token-mean` |
| GRPO (length-independent) | `mean` | `step-mean` |
| Length-normalized stable | `mean-norm` | `step-mean` |
| Multi-step agent | `mean` | `program-mean` |

### KL estimators

`loss_args.kl_type` selects the estimator for the `loss_args.kl_coef` auxiliary-loss term (and for `kl_reward`, when implemented). All are computed against the **reference** policy; write `lr = log π_θ − log π_ref` for the per-token log-ratio.

| `kl_type` | Formula |
|---|---|
| `kl` / `k1` | `lr` |
| `abs` | `\|lr\|` |
| `mse` / `k2` | `½ · lr²` |
| `low_var_kl` / `k3` | `exp(−lr) + lr − 1` — Schulman's low-variance, always-non-negative estimator ([derivation](http://joschu.net/blog/kl-approx.html)). **Recommended.** |

### Sampler-IS / sampler-RS

Both knobs implement the token-level correction described in [sampler-trainer agreement](../core-concepts/sampler-trainer-agreement.md#off-policy-correction-importance-sampling-rejection-and-veto). They are independent — you can use either, both, or neither.

- **IS (importance sampling)** reweights tokens by their `π_old / π_sampler` ratio — the trainer-recomputed (rollout-time) log-prob over the sampler's, computed once and detached, not the current policy being optimized. Set `sampler_is = token` for token-level weighting, `sequence` for whole-sequence weighting via summed log-probabilities. The threshold truncates the weight at its upper bound only (weights above `T` are capped to `T`).
- **RS (rejection sampling)** masks tokens / sequences whose ratio falls outside the threshold band entirely. `sampler_rs = token | sequence | geometric`. `geometric` aggregates via the geometric mean (log-weight average).
- **Sequence veto** (`sampler_token_veto_threshold`) is the most aggressive option — if any one token's IS weight is below this, the *entire sequence* is zeroed out of the loss.

---

## Trainer template

Path: `axon/config/trainer/trainer.yaml`. Imported four times via Hydra defaults — every field listed here exists under `actor.<field>`, `ref.<field>`, `critic.<field>`, and `reward_model.<field>`.

### Model identity

| Field | Default | Type | Purpose |
|---|---|---|---|
| `model_path` | `null` | path | HF model weights path (per-role override). |
| `trust_remote_code` | `false` | bool | Trust remote HF code. |
| `external_lib` | `null` | str? | External Python module to import. |
| `override_hf_config` | `{}` | dict | HF AutoConfig overrides. |

### Optimizer

| Field | Default | Type | Purpose |
|---|---|---|---|
| `optimizer` | `AdamW` | str | Optimizer class name. |
| `optimizer_args.optimizer_impl` | `torch.optim` | str | Module path to import optimizer from. |
| `optimizer_args.lr` | `1.0e-06` | float | Base learning rate. |
| `optimizer_args.weight_decay` | `0.1` | float | L2 regularisation. |
| `optimizer_args.betas` | `[0.9, 0.95]` | list[float] | Adam betas (β₁, β₂). |
| `optimizer_args.override_optimizer_args` | `{}` | dict | Extra kwargs merged into optimizer init. |

### LR scheduler

| Field | Default | Type | Purpose |
|---|---|---|---|
| `lr_scheduler` | `constant` | enum | FSDP: `constant`, `cosine`. Megatron: `constant`, `linear`, `cosine`, `inverse_square_root`. |
| `lr_scheduler_args.total_training_steps` | `-1` | int | Set automatically by trainer at init. |
| `lr_scheduler_args.lr_warmup_steps` | `-1` | int | Linear warmup steps. < 0 uses ratio. |
| `lr_scheduler_args.lr_warmup_steps_ratio` | `0.0` | float | Warmup as fraction of total. |
| `lr_scheduler_args.min_lr_ratio` | `0.0` | float | Cosine annealing floor as ratio of base LR. |
| `lr_scheduler_args.min_lr` | `0.0` | float | Absolute minimum LR (takes precedence over ratio when > 0). |
| `lr_scheduler_args.num_cycles` | `0.5` | float | Cosine half-cycles (0.5 = monotonic). |
| `lr_scheduler_args.lr_warmup_init` | `0.0` | float | Megatron: initial LR ratio at warmup start. |
| `lr_scheduler_args.lr_decay_steps` | `null` | int? | Megatron: decay-curve length (`null` = total). |
| `lr_scheduler_args.weight_decay_incr_style` | `constant` | enum | Megatron WSD: `constant`, `linear`, `exponential`. |
| `lr_scheduler_args.lr_wsd_decay_style` | `exponential` | enum | Megatron WSD curve: `linear` or `exponential`. |
| `lr_scheduler_args.lr_wsd_decay_steps` | `null` | int? | Megatron WSD steps. |

### Memory offload & precision

| Field | Default | Type | Purpose |
|---|---|---|---|
| `param_offload` | `false` | bool | Offload model params to CPU between forward/backward. |
| `optimizer_offload` | `false` | bool | Offload optimizer states to CPU. |
| `forward_only` | `false` | bool | Skip optimizer/scheduler construction (used by `ref` and `reward_model`). |
| `dtype` | `bfloat16` | enum | Model dtype: `fp32`, `fp16`, `bfloat16`. |
| `grad_clip` | `1.0` | float | Max gradient norm for clipping. |
| `strategy` | `${strategy}` | enum | Inherited from top-level. |

### Batching

| Field | Default | Type | Purpose |
|---|---|---|---|
| `micro_batch_size` | `null` | int? | Global micro-batch size. |
| `micro_batch_size_per_gpu` | `null` | int? | Per-GPU micro-batch size. **When `use_dynamic_bsz` is `false`, exactly one of `micro_batch_size` / `micro_batch_size_per_gpu` must be set** (per-GPU preferred). Under dynamic batching (the default) neither is required — batching is driven by `max_token_len_per_gpu`. |
| `use_dynamic_bsz` | `true` | bool | Dynamic batch sizing by token count. Overrides sample-count batching. |
| `max_token_len_per_gpu` | `16384` | int | Max tokens per GPU per micro-batch when dynamic. |
| `forward_micro_batch_size` | `null` | int? | Forward-only mbs (inference / eval / ref). |
| `forward_micro_batch_size_per_gpu` | `null` | int? | Forward-only mbs per GPU. |
| `forward_use_dynamic_bsz` | `true` | bool | Dynamic batching for forward-only passes. |
| `forward_max_token_len_per_gpu` | `16384` | int | Max tokens per GPU for forward-only. |
| `use_fused_kernels` | `true` | bool | Fused forward kernels (return logprobs + entropy directly). |
| `offload_p2p_buffer` | `false` | bool | Offload P2P weight-transfer buffers to CPU. |

### FSDP

Applies when `strategy` is `fsdp` or `fsdp2`.

| Field | Default | Type | Purpose |
|---|---|---|---|
| `fsdp.wrap_policy.min_num_params` | `0` | int | Min parameter count for a module to be FSDP-wrapped (0 = wrap all). |
| `fsdp.offload_policy` | `false` | bool | FSDP2 CPU offload policy. |
| `fsdp.reshard_after_forward` | `false` | bool | FSDP2: reshard params after forward (memory optimization). |
| `fsdp.fsdp_size` | `-1` | int | GPUs per shard group (-1 = all). > 1 enables hybrid DDP across shard groups. |
| `fsdp.forward_prefetch` | `true` | bool | Prefetch next layer's params during forward (FSDP1). |
| `fsdp.model_dtype` | `fp32` | enum | Default param dtype for mixed precision. |
| `fsdp.use_orig_params` | `false` | bool | Use unflattened params. Required for LoRA on FSDP1. |
| `fsdp.ulysses_sequence_parallel_size` | `1` | int | Ulysses sequence-parallel group size. |
| `fsdp.freeze_vision_tower` | `false` | bool | Freeze vision encoder in multimodal models. |
| `fsdp.use_torch_compile` | `true` | bool | torch.compile for entropy computation. |
| `fsdp.entropy_from_logits_with_chunking` | `false` | bool | Chunked entropy (memory optimization). |
| `fsdp.entropy_checkpointing` | `false` | bool | Gradient checkpointing for entropy. |
| `fsdp.grad_norm_threshold` | `1.0e+05` | float | Warn when gradient norm exceeds. |
| `fsdp.enable_gradient_checkpointing` | `true` | bool | Activation checkpointing. |
| `fsdp.enable_activation_offload` | `false` | bool | CPU-offload saved activations between forward and backward. |
| `fsdp.use_remove_padding` | `true` | bool | Skip computation on padding tokens. Required when `ulysses_sequence_parallel_size > 1`. |
| `fsdp.use_liger` | `false` | bool | Liger-kernel monkey-patch for HF models. |
| `fsdp.use_fused_kernels` | `true` | bool | Fused forward kernels (FSDP-side). |
| `fsdp.fused_kernel_options.impl_backend` | `triton` | enum | `triton` (avoids vocab materialisation) or `torch`. |

### LoRA

Set `lora.rank > 0` to enable. FSDP needs `fsdp.use_orig_params=true` for LoRA.

| Field | Default | Type | Purpose |
|---|---|---|---|
| `lora.rank` | `0` | int | LoRA rank dimension. `0` = disabled. |
| `lora.alpha` | `16` | int | LoRA scaling factor. Effective scale = `alpha / rank`. |
| `lora.target_modules` | `all-linear` | str / list | FSDP: `all-linear`. Megatron: explicit list, e.g. `[linear_qkv, linear_proj]`. |
| `lora.exclude_modules` | `null` | list? | Module names to exclude. |
| `lora.adapter_path` | `null` | path? | Pre-trained adapter checkpoint to start from. |
| `lora.type` | `lora` | enum | Megatron variants: `lora`, `vlm_lora`, `canonical_lora`, `dora`. |
| `lora.dropout` | `0.0` | float | LoRA dropout. |
| `lora.dropout_position` | `pre` | enum | `pre` or `post`. |
| `lora.lora_A_init_method` | `xavier` | enum | A-matrix init. |
| `lora.lora_B_init_method` | `zero` | enum | B-matrix init (zero is standard). |
| `lora.a2a_experimental` | `false` | bool | Experimental all-to-all comm for EP-LoRA. |
| `lora.dtype` | `null` | enum? | Override LoRA param dtype. `null` = inherit. |
| `lora.freeze_vision_model` | `true` | bool | `vlm_lora`: freeze the vision tower. |
| `lora.freeze_vision_projection` | `true` | bool | `vlm_lora`: freeze the projection. |
| `lora.freeze_language_model` | `true` | bool | `vlm_lora`: freeze the language model. |

### Megatron

Applies when `strategy` is `megatron`.

| Field | Default | Type | Purpose |
|---|---|---|---|
| `megatron.freeze_moe_router` | `false` | bool | Freeze MoE router weights. |
| `megatron.grad_offload` | `false` | bool | CPU-offload gradients during backward. |
| `megatron.tensor_model_parallel_size` | `1` | int | Tensor parallelism. |
| `megatron.expert_model_parallel_size` | `1` | int | Expert parallelism (MoE). |
| `megatron.expert_tensor_parallel_size` | `null` | int? | TP within EP groups. |
| `megatron.pipeline_model_parallel_size` | `1` | int | Pipeline parallelism. |
| `megatron.virtual_pipeline_model_parallel_size` | `null` | int? | Interleaved PP. |
| `megatron.context_parallel_size` | `1` | int | Context parallelism (ring attention with local-loss). |
| `megatron.sequence_parallel` | `true` | bool | Sequence parallelism within TP groups. Requires `tensor_model_parallel_size > 1`. |
| `megatron.use_distributed_optimizer` | `true` | bool | ZeRO-style optimizer sharding. |
| `megatron.use_dist_checkpointing` | `false` | bool | Load from a distributed checkpoint. |
| `megatron.dist_checkpointing_path` | `null` | path? | Distributed-checkpoint path. |
| `megatron.dist_checkpointing_prefix` | `''` | str | Key prefix when loading. |
| `megatron.seed` | `42` | int | Megatron init seed. |
| `megatron.override_ddp_config.overlap_grad_reduce` | `true` | bool | Overlap grad all-reduce with backward. |
| `megatron.override_transformer_config.gradient_accumulation_fusion` | `true` | bool | Fuse grad accumulation with backward GEMM (APEX). |
| `megatron.override_transformer_config.hidden_dropout` | `0.0` | float | Hidden-layer dropout. |
| `megatron.override_transformer_config.attention_dropout` | `0.0` | float | Attention dropout. |
| `megatron.override_transformer_config.recompute_granularity` | `null` | enum? | `null`, `selective`, `full`. |
| `megatron.override_transformer_config.recompute_modules` | `[core_attn]` | list[str] | Which modules to recompute. |
| `megatron.override_transformer_config.recompute_method` | `null` | enum? | `null`, `uniform`, `block`. |
| `megatron.override_transformer_config.recompute_num_layers` | `null` | int? | Layer count for recompute method. |
| `megatron.override_transformer_config.attention_backend` | `flash` | enum | `flash`, `fused`, `unfused`. |
| `megatron.use_mbridge` | `true` | bool | MBridge for HF↔Megatron weight conversion. |
| `megatron.vanilla_mbridge` | `true` | bool | Vanilla (slower, safer) mbridge. |
| `megatron.use_remove_padding` | `true` | bool | Skip padding tokens. |
| `megatron.load_weight` | `true` | bool | Load pretrained weights. `false` = random init (useful for ablations / dummy tests). |

### Profiler (per-role)

| Field | Default | Type | Purpose |
|---|---|---|---|
| `profiler.tool` | `${oc.select:global_profiler.tool,null}` | enum? | Inherit global, or override. |
| `profiler.enable` | `false` | bool | Profiling active for this role. |
| `profiler.all_ranks` | `false` | bool | Profile all ranks vs specific list. |
| `profiler.ranks` | `[]` | list[int] | Specific ranks when not `all_ranks`. |
| `profiler.save_path` | `${oc.select:global_profiler.save_path,null}` | path? | Inherit global. |
| `profiler.tool_config.nsys.discrete` | inherits | bool | NSYS discrete mode. |
| `profiler.tool_config.torch.step_start` | `0` | int | First step to profile. |
| `profiler.tool_config.torch.step_end` | `null` | int? | Last step (inclusive). |

### Per-role overrides

A handful of fields are interpolated automatically across roles. The most useful:

- `ref.*` mostly inherits from `actor.*` via `${oc.select:actor.X,default}` interpolations — so setting `actor.megatron.tensor_model_parallel_size=4` is enough for ref. `ref.forward_only` is hard-set to `true`.
- `critic.optimizer_args.lr` defaults to `1.0e-05` (10× actor's `1.0e-06`).
- `critic.max_token_len_per_gpu` defaults to `32768` (2× actor's).
- `critic.enable` is `null` by default — inferred from `advantage` (only `gae` needs a critic).
- `reward_model.enable` is `false` by default — set to `true` to add an RM to the resource pool.
- `reward_model.model_path` defaults to a separate path (`~/models/FsfairX-LLaMA3-RM-v0.1`).

---

## Sampler template

Path: `axon/config/sampler/sampler.yaml`. Imported once under the `sampler:` key.

### Engine & model

| Field | Default | Type | Purpose |
|---|---|---|---|
| `name` | `vllm` | enum | Inference engine: `vllm` (primary) or `sglang` (secondary). |
| `model_path` | `null` | path | Model weights path. Inherited from top-level when not set. |
| `trust_remote_code` | `false` | bool | Trust remote HF code. |
| `lora_rank` | `0` | int | LoRA rank for inference. `0` = no LoRA. |
| `external_lib` | `null` | str? | External library to import. |
| `dtype` | `bfloat16` | enum | Model weight dtype. |
| `gpu_memory_utilization` | `0.85` | float | KV cache GPU memory fraction (0.0–1.0). |
| `enforce_eager` | `false` | bool | Disable CUDA graphs (debug mode). |
| `cudagraph_capture_sizes` | `null` | list[int]? | Explicit batch sizes for graph capture. |
| `offload_sampler` | `true` | bool | Hybrid mode: offload sampler weights/KV during training. |

### Parallelism

| Field | Default | Type | Purpose |
|---|---|---|---|
| `tensor_model_parallel_size` | `1` | int | Inference TP. |
| `data_parallel_size` | `1` | int | Inference DP. |
| `expert_parallel_size` | `1` | int | Inference EP (MoE). |
| `pipeline_model_parallel_size` | `1` | int | Inference PP. |

### Batching & sequence limits

| Field | Default | Type | Purpose |
|---|---|---|---|
| `max_num_batched_tokens` | `32768` | int | Max tokens per batch (also exported as `VLLM_MAX_NUM_BATCHED_TOKENS`). |
| `max_model_len` | `null` | int? | Max context length. The template default is `null`; the root config sets it to `${max_seq_length}`. |
| `max_num_seqs` | `1024` | int | Max concurrent sequences. |

### Engine optimizations

| Field | Default | Type | Purpose |
|---|---|---|---|
| `enable_chunked_prefill` | `true` | bool | Split long prefills for TTFT reduction. |
| `enable_prefix_caching` | `false` | bool | Cache common prompt prefixes. **Incompatible with `moe_replay`** — enforced by validator. |
| `load_format` | `dummy` | enum | `dummy` (random init, weights sent by trainer first) or `auto` (real weights). The `dummy` default is correct for the standard hybrid / disaggregated flow where the trainer streams initial weights to the sampler. |
| `quantization` | `null` | enum? | `null`, `fp8`, `fp8_fast`, `int4`, `mxfp8`. |
| `int4_group_size` | `128` | int | INT4 per-group quantisation size. |
| `int4_symmetric` | `true` | bool | INT4 symmetric vs asymmetric. |
| `disable_log_stats` | `true` | bool | Suppress engine logging stats. |
| `engine_kwargs.vllm` | `{}` | dict | Extra vLLM init kwargs (passed straight through). |
| `engine_kwargs.sglang` | `{}` | dict | Extra SGLang init kwargs. |
| `skip_tokenizer_init` | `false` | bool | Skip tokenizer init inside the engine. |

### Multimodal

| Field | Default | Type | Purpose |
|---|---|---|---|
| `limit_images` | `null` | int? | Cap on images per prompt (vLLM `limit_mm_per_prompt`). |

### Speculative decoding

| Field | Default | Type | Purpose |
|---|---|---|---|
| `speculative_config` | `{}` | dict | Speculative decoding config; empty disables. |

### MoE communication & P2P

| Field | Default | Type | Purpose |
|---|---|---|---|
| `all2all_backend` | `deepep` | enum | All-to-all backend for EP. |
| `enable_eplb` | `false` | bool | Expert-parallel load balancing. |
| `offload_p2p_buffer` | `false` | bool | Offload P2P buffers to CPU between transfers. |
| `layered_summon` | `false` | bool | Layered LoRA parameter summoning (FSDP only). |

### Profiler

Same shape as the trainer profiler block (`profiler.tool`, `profiler.enable`, `profiler.all_ranks`, `profiler.ranks`, `profiler.save_path`, `profiler.tool_config`).

---

## Worked examples

Override patterns from shipped recipes:

```yaml
# PPO + RLOO with DAPO-style asymmetric clipping (recipes/frozenlake/train_frozenlake_qwen_30b_a3b.yaml)
loss: ppo
advantage: rloo
loss_args:
  clip_ratio_low: 0.2
  clip_ratio_high: 0.28
  token_reduce: mean-norm
  batch_reduce: step-mean
  kl_coef: 0.0
  kl_type: low_var_kl
  entropy_coef: 0
moe_replay: true
```

```yaml
# Switch to disaggregated mode with Megatron + 6D parallelism
strategy: megatron
hybrid_engine: false
sampler_trainer_gpu_ratio: 1
actor:
  megatron:
    tensor_model_parallel_size: 4
    expert_model_parallel_size: 4
    pipeline_model_parallel_size: 2
    context_parallel_size: 1
    sequence_parallel: true
sampler:
  tensor_model_parallel_size: 8
```

```yaml
# Swap loss + advantage at the command line (no yaml change needed)
# axon train --config <recipe> -- loss=cispo advantage=grpo loss_args.sampler_is=token
```

For runnable recipes, see the per-recipe READMEs under [`recipes/`](https://github.com/modelcorp/axon/tree/main/recipes).
