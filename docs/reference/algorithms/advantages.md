# Advantages

Every advantage estimator listed here is registered via `@register_advantage(AdvantageFn.X)` in `axon/trainer/algos/advantages/advantage.py` and selected by `advantage: <name>` in a recipe yaml.

Notation:

- `R_t` — token-level reward
- `r_seq` — sequence-level (terminal) reward
- `g(...)` — group of sequences sharing the same prompt (`uid`). Group-based estimators read `non_tensor_batch["uid"]`.
- `A` — per-token advantage tensor populated into `data.batch["advantages"]`.

---

## GAE

```yaml
advantage: gae
```

Generalized Advantage Estimation. Requires a critic — set up automatically when `advantage: gae`.

```
δ_t  = R_t + γ * V(s_{t+1}) - V(s_t)
A_t  = δ_t + γ*λ * A_{t+1}
```

**Knobs:** `gamma` (default `0.99`), `lam` (default `0.95`).

## GRPO

```yaml
advantage: grpo
```

Group-relative Policy Optimization. Within each group of rollouts that share the same prompt:

```
A = (r_seq - mean(g)) / (std(g) + epsilon)            # if norm_adv_by_std
A = r_seq - mean(g)                                   # otherwise
```

The same group-normalized value is broadcast to every response token.

**Knobs:** `norm_adv_by_std` (default `true`), `epsilon` (default `1e-6`).

## RLOO / LOOP

```yaml
advantage: rloo    # equivalent
advantage: loop    # equivalent (alias; preferred name in shipped recipes)
```

Leave-one-out group baseline.

```
A_i = (n * r_i - sum(g)) / (n - 1)
```

where `n = len(g)`. Both `rloo` and `loop` are registered to the same function. `loop` is the more common name in shipped recipes.

## REINFORCE++ {#reinforce-plus-plus}

```yaml
advantage: reinforce_plus_plus
```

Discounted REINFORCE with global whitening:

```
G_t      = sum_{k>=t} γ^{k-t} * R_k
A_t      = (G_t - mean) / std        # mean and std computed across the entire batch
```

**Knobs:** `gamma` — the code fallback is `1.0`, but the shipped `advantage_args.gamma` is `0.99` (shared with GAE), so REINFORCE++ runs discounted by default; set `advantage_args.gamma=1.0` for undiscounted returns.

## REINFORCE++ baseline {#reinforce-plus-plus-baseline}

```yaml
advantage: reinforce_plus_plus_baseline
```

Two-step variant: first subtract a per-group baseline, then apply global whitening.

## REMAX

```yaml
advantage: remax
```

ReMax — uses a greedy-decoding rollout's return as the baseline.

```
A = r_seq - r_greedy_baseline
```

REMAX reads the baseline from `data.batch["reward_baselines"]`; the greedy-rollout pass that fills it is not wired into the default sampling path, so a recipe using REMAX must supply it.

## OPO

```yaml
advantage: opo
```

Length-weighted group baseline. Subtracts a baseline that weights each member of the group by its sequence length.

## GRPO-PassK {#grpo-passk}

```yaml
advantage: grpo_passk
```

Pass@k-shaped advantage. Only the top response in the group gets a non-zero advantage:

```
A_top  = (r_max - r_second_max) / (std(group) + epsilon)   # if norm_adv_by_std (default)
A_top  = r_max - r_second_max                               # otherwise
A_else = 0
```

**Knobs:** `norm_adv_by_std` (default `true`), `epsilon` (default `1e-6`).

## GPG

```yaml
advantage: gpg
```

Group Policy Gradient advantage shaping:

```
α       = batch_size / count_nonzero(scores)        # rescaling for sparse rewards
A_token = α * (score - group_mean) / f_norm
```

Pair with `loss: gpg` (which is itself an alias of REINFORCE) to recover the GPG composition.

**Knobs:** `f_norm` (default `1.0`).

## Chunked GAE {#chunked-gae}

```yaml
advantage: chunked_gae
```

Same mathematics as GAE, but computes the advantage via parallel prefix scan over chunks of size `chunk_size`. Sequential complexity drops from `O(T)` to `O(T / chunk_size)`. Same numerical result as GAE up to floating-point precision.

**Knobs:** `gamma`, `lam`, `chunk_size` (default `128`).

## Kimi-K1.5 {#kimi-k1-5}

```yaml
advantage: kimi_k1_5
```

Length-shaped advantage from the Kimi-K1.5 recipe ([arXiv:2501.12599](https://arxiv.org/abs/2501.12599)). First shapes rewards by the rollout's min-max-normalized length within its group, then applies GRPO-style group normalization.

**Knobs:**

- `length_coef` (default `0.0`) — strength of length shaping. `1.0` = paper formula; `0` disables.
- `length_coef_warmup_steps` (default `0`) — hold `length_coef` at 0 until this many steps.
- `length_coef_ramp_steps` (default `0`) — linear ramp after warmup. `0` = hard switch.
- `correct_threshold` (default `1.0`) — score ≥ counts as correct.
- `norm_adv_by_std`, `epsilon` — passed through to the GRPO step.

## Identity

```yaml
advantage: identity
```

Pass-through. Use when `data.batch["token_level_rewards"]` is already advantage-shaped and no further transformation is needed.
