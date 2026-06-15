# Losses

Every loss in this list is registered via `@register_loss(LossFn.X)` in `axon/trainer/algos/loss/loss.py` and selected by `loss: <name>` in a recipe yaml. All losses share the same signature:

```python
def my_loss_fn(data: DataProto, config: DictConfig) -> tuple[torch.Tensor, dict[str, Any]]
```

Notation in this page:

- `r = exp(log_prob - old_log_prob)` is the proximal importance ratio.
- `A` is the per-token advantage.
- `M` is the response mask (1 for tokens that count, 0 otherwise).
- `agg_loss(...)` performs two-stage reduction; see [Configuration → Token / batch reduction](../configuration.md#token-batch-reduction).
- IS / RS / token-veto correction is configured per-recipe through `loss_args.sampler_is`, `loss_args.sampler_rs`, `loss_args.sampler_token_veto_threshold`.

---

## PPO

```yaml
loss: ppo
```

Standard PPO with **dual-clip** for the negative-advantage tail.

```
ratio              = exp(log_prob - old_log_prob)
clipped_ratio      = clip(ratio, 1 - clip_ratio_low, 1 + clip_ratio_high)
pg_unclipped       = -A * ratio
pg_clipped         = -A * clipped_ratio
pg_loss_pessim     = max(pg_unclipped, pg_clipped)         # standard PPO max
# Dual clip (only kicks in when A < 0):
pg_loss_dual       = min(pg_loss_pessim, -A * clip_ratio_c)
pg_losses          = where(A < 0, pg_loss_dual, pg_loss_pessim)
loss               = agg_loss(pg_losses * sampler_is_weights, M)
```

**Knobs:** `clip_ratio_low`, `clip_ratio_high`, `clip_ratio_c` (default 3.0; must be > 1.0). Optional KL term via `kl_coef + kl_type`.

**Notes:**

- Asymmetric clipping (the DAPO-style `0.2 / 0.28`) lets the high-reward tail push harder while bounding pessimism on negative advantages.
- The dual-clip cap fires only when `A < 0` and the pessimistic objective itself would exceed `-A * clip_ratio_c`.

## GSPO

```yaml
loss: gspo
```

Sequence-level objective using a stop-gradient ratio so the policy update is decoupled from the IS-weight gradient.

**Knobs:** `clip_ratio_low` / `clip_ratio_high` (no dual-clip — `clip_ratio_c` is PPO-only). The stop-gradient is hard-coded inside the loss.

## CISPO

```yaml
loss: cispo
```

Clipped-IS-weight stop-grad objective. The unclipped ratio's gradient flows through `log π_θ`; the clipped weight is stopped and used as a coefficient.

```
loss = -stop_grad(clipped_ratio) * A * log π_θ
```

**Knobs:** clip-ratio set; `token_level_mask` (default `false`) — when `true`, also drops tokens whose unclipped ratio exceeds the clip bound in the advantage's own direction (`A > 0` with `ratio > 1 + high`, or `A < 0` with `ratio < 1 − low`).

**When to use:** long-horizon, sparse-reward agentic settings, where preserving gradient signal from rare high-importance-weight tokens matters most. (CISPO was introduced in MiniMax-M1 to improve RL efficiency by clipping the IS weight instead of the token update, so high-weight tokens aren't discarded by the trust-region clip.) The `tools` and `search-r1` recipes default to GSPO but take `loss=cispo` like any other.

## CLIP_COV

```yaml
loss: clip_cov
```

Selects a small fraction (`clip_cov_ratio`) of tokens whose advantage-logprob covariance falls in the open interval `(clip_cov_lb, clip_cov_ub)` and zeroes their gradient; every other token gets the standard pessimistic PPO clip. The intuition is to drop the few tokens where advantage and log-prob co-move most strongly — the ones that drive entropy collapse.

**Knobs:** `clip_cov_ratio` (fraction of tokens to consider, default `0.0002`), `clip_cov_lb`, `clip_cov_ub`.

## KL_COV

```yaml
loss: kl_cov
```

KL injection on high-covariance tokens, in the same spirit as CLIP_COV — but the selection differs. KL_COV deterministically takes the top `kl_cov_ratio` fraction of tokens by advantage-logprob covariance (`torch.topk`) and adds a per-token KL penalty, whereas CLIP_COV draws a random subset from a bounded covariance band.

```
loss_token = -A * ratio + ppo_kl_coef * |log_ratio|     # for selected tokens
```

**Knobs:** `kl_cov_ratio` (default `0.0002`), `ppo_kl_coef` (default `0.1`).

## GEO_MEAN

```yaml
loss: geo_mean
```

GMPO — sequence-level geometric-mean policy objective.

**Note:** `GEO_MEAN` bypasses the `agg_loss` reduction stage and computes its own sequence-level reduction. Setting `loss_args.token_reduce` / `loss_args.batch_reduce` has no effect with this loss.

## REINFORCE

```yaml
loss: reinforce
```

Plain policy-gradient objective with optional IS reweighting.

```
pg_losses = -A * log_prob          # or -A * log_prob * sampler_is_weights when IS is on
loss      = agg_loss(pg_losses, M)
```

Also returns a `ppo_kl` diagnostic — the mean negative log-ratio between the current and old-policy (`old_log_probs`) logprobs. This equals sampler-trainer drift only when `use_sampler_logprobs=true`; otherwise `old_log_probs` is a fresh trainer forward, so it measures current-vs-old drift across PPO epochs.

## GPG

```yaml
loss: gpg
```

GPG (Group Policy Gradient) loss, adapted from <https://github.com/AMAP-ML/GPG>. The objective is mathematically identical to REINFORCE; **the GPG composition differentiates itself in the advantage**, not the loss. `loss: gpg` is implemented as a thin alias to `reinforce_loss_fn` to make this explicit.

To run "GPG", pair it with the GPG advantage:

```yaml
loss: gpg
advantage: gpg
```

## VALUE

```yaml
loss: value     # only used internally when a critic is present
```

Clipped value-function loss for the critic. Selected automatically when `advantage: gae` (which requires a critic). Knob: `cliprange_value` (default `0.5`).

---

## Cross-cutting knobs (all policy losses)

These apply to every policy loss above (not VALUE):

| Knob | Default | What it does |
|---|---|---|
| `entropy_coef` | `0` | Adds `-entropy_coef * H(π)` to the loss (entropy bonus). Set `0` to disable. |
| `kl_coef` | `0.0` | Adds `kl_coef * KL(π ‖ π_ref)` to the loss (auxiliary KL against the reference policy). Requires a reference policy when `> 0`. |
| `kl_type` | `low_var_kl` | KL estimator. See [Configuration → KL estimators](../configuration.md#kl-estimators). |
| `sampler_is` | `null` | Token / sequence-level IS reweighting (applied by every policy loss except CISPO, which ignores it). |
| `sampler_is_threshold` | `2.0` | Upper-bounds IS weights at `T` (`min(weight, T)`); no lower clamp. |
| `sampler_rs` | `null` | Token / sequence / geometric rejection-sampling — masks out-of-range tokens entirely. |
| `sampler_token_veto_threshold` | `null` | If any one token's IS weight falls below this, the entire sequence is masked from the loss. |
| `token_reduce` / `batch_reduce` | `sum` / `token-mean` | Two-stage reduction. See [Configuration](../configuration.md#token-batch-reduction). |
