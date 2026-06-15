# Add an algorithm

This guide describes the integration path for a new RL loss or advantage estimator. Both registries follow the same pattern; the difference is which file you put your function in.

## The minimum diff

To add a new loss called `MY_OPT`:

```python
# axon/trainer/algos/loss/registry.py
class LossFn(str, Enum):
    PPO = "ppo"
    GSPO = "gspo"
    ...
    MY_OPT = "my_opt"   # add this line
```

```python
# axon/trainer/algos/loss/loss.py
@register_loss(LossFn.MY_OPT)
def my_opt_loss_fn(data: DataProto, config: DictConfig) -> tuple[torch.Tensor, dict[str, Any]]:
    log_probs = data.batch["log_probs"]               # current-policy logp (this update)
    old_log_probs = data.batch["old_log_probs"]       # old-policy logp (trainer-recomputed by default; = sampler_log_probs if use_sampler_logprobs)
    advantages = data.batch["advantages"]
    response_mask = data.batch["response_mask"]

    # ... your loss math here ...
    loss = ...
    metrics = {"my_opt/loss": loss.detach()}
    return loss, metrics
```

```yaml
# In your recipe yaml
loss: my_opt
```

The driver discovers the function through the registry — no second registration step.

## The function contract

Every loss has the same signature:

```python
fn(data: DataProto, config: DictConfig) -> tuple[torch.Tensor, dict[str, Any]]
```

- `data` is a `DataProto` carrying the rollout batch enriched by the driver: `log_probs` (current policy), `old_log_probs` (the old-policy snapshot — by default recomputed trainer-side at the start of the PPO update; when `use_sampler_logprobs: true` it is instead set to the sampler's rollout logp, i.e. equal to `sampler_log_probs`), `advantages`, `response_mask`, and whatever else the driver added (KL terms, reference logprobs, MoE routing maps, etc.).
- `config` is the recipe's `loss_args` block directly (not the full training config); read per-loss knobs as `config.clip_ratio`, `config.clip_ratio_c`, `config.token_reduce`, etc. (Note: `kl_coef` and `entropy_coef` are *not* read inside the loss — the trainer's model wrapper adds those terms to whatever loss you return, so don't re-apply them here.)
- The returned `loss` is a scalar tensor that the trainer will `.backward()`.
- The returned metrics dict gets logged each step.

The advantage-estimator contract is the same shape, in `axon/trainer/algos/advantages/advantage.py`.

## What's already provided

You don't need to re-implement these — read the helpers in `axon/trainer/algos/loss/utils.py` and reuse:

- **Token / batch reduction** — the two-stage reduce (`token_reduce` × `batch_reduce`). The global default is `batch_reduce: token-mean` (sum tokens, divide by total token count); `step-mean` (length-independent) is the common choice in GRPO-style recipes. Driven by `loss_args.token_reduce` / `loss_args.batch_reduce`.
- **KL terms** — `kl`, `abs`, `mse`, `low_var_kl` via `loss_args.kl_type` and `loss_args.kl_coef` (estimators live in `axon/utils/rl/kl.py`).
- **Sampler correction** — IS / rejection / veto weights are computed upstream (`axon/utils/rl/sampler.py`) and arrive as `data.batch["sampler_is_weights"]`; multiply them into your per-token loss the way the PPO loss does. The knobs (`sampler_is`, `sampler_is_threshold`, `sampler_token_veto_threshold`) are set in `loss_args`.
- **Entropy bonus** — `loss_args.entropy_coef`.


## Adding a new advantage estimator

Same shape, different file:

```python
# axon/trainer/algos/advantages/registry.py
class AdvantageFn(str, Enum):
    GAE = "gae"
    GRPO = "grpo"
    ...
    MY_ADV = "my_adv"
```

```python
# axon/trainer/algos/advantages/advantage.py
@register_advantage(AdvantageFn.MY_ADV)
def my_adv_fn(data: DataProto, config: DictConfig) -> tuple[torch.Tensor, torch.Tensor]:
    token_level_rewards = data.batch["token_level_rewards"]
    response_mask = data.batch["response_mask"]
    # ... your advantage shaping ...
    advantages = ...
    return advantages, advantages   # (advantages, returns) — identical for group-relative estimators
```

```yaml
advantage: my_adv
```

The advantage estimator runs in the driver's "enrich" phase and returns an `(advantages, returns)` tensor pair — identical for group-relative estimators (GRPO, RLOO, …), while GAE, chunked-GAE, REINFORCE++, and ReMax return distinct `returns` — which the driver feeds to the loss.

## Validating a new algorithm

Before you compare against published results:

1. **Smoke test on FrozenLake or GSM8K** — does it train to non-trivial reward in 50 steps?
2. **Disable IS correction first** (`sampler_is: null`) — get the math right before the variance-reduction layer kicks in.
3. **Compare to a known-good baseline** — same data, same model, same seed. PPO + GRPO is the natural reference for both new losses and new advantages.
4. **Watch the metrics** — entropy, advantage-mean, advantage-std, log-ratio-mean, log-ratio-std. A loss that's "training" but driving entropy to zero in 10 steps is broken regardless of the reward curve.

## Tips

??? tip "Numerical stability"
    The standard tricks: clamp log-ratios into a reasonable range before exponentiating; add a small `epsilon` to denominators in normalization; for sequence-level reductions, mask before reducing rather than after.

??? tip "Match the response mask"
    Most failures in custom losses come from indexing into the wrong axis. The response mask is what tells you which tokens count for the loss — apply it once, consistently, and don't mix it with the attention mask.

??? tip "Test with `IDENTITY` advantage first"
    `AdvantageFn.IDENTITY` uses the env's `token_level_rewards` directly as advantages — no normalization, no baseline. Useful for testing a new loss against a hand-computed signal without the noise from a learned estimator.
