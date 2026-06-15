# Algorithms

Two registries make up Axon's algorithm surface:

- `@register_loss(LossFn.X)` in `axon/trainer/algos/loss/loss.py` — the per-token objective the gradient flows through.
- `@register_advantage(AdvantageFn.X)` in `axon/trainer/algos/advantages/advantage.py` — the per-token target the loss is computed against.

A recipe selects one of each in yaml. A named method from the literature is a loss + an advantage + a clip / IS / reduction policy — sometimes a dedicated loss (GSPO, CISPO) or advantage (GRPO), sometimes a pure recipe composition (DAPO is the PPO loss with group-relative advantages and asymmetric clipping).

For per-knob detail (clip ratios, KL types, sampler-IS / RS, token / batch reduction), see [Configuration → Loss](../configuration.md#loss).

## Losses

| Name | When to use |
|---|---|
| [PPO](losses.md#ppo) | Default. Dual-clip on the proximal ratio. Pairs with any advantage. |
| [GSPO](losses.md#gspo) | Sequence-level objective with stop-gradient ratio. |
| [CISPO](losses.md#cispo) | Clipped-IS-weight stop-grad. Stable on long-horizon sparse-reward. |
| [CLIP_COV](losses.md#clip_cov) | Selective clipping based on advantage-logprob covariance. |
| [KL_COV](losses.md#kl_cov) | KL injection on the top-covariance tokens (top-k, vs CLIP_COV's banded random subset). |
| [GEO_MEAN](losses.md#geo_mean) | Sequence-level geometric-mean policy objective (GMPO). |
| [REINFORCE](losses.md#reinforce) | Plain policy gradient. |
| [GPG](losses.md#gpg) | Alias of REINFORCE — the GPG paper differs in advantage shaping; pair with `advantage: gpg`. |
| [VALUE](losses.md#value) | Critic loss; not a policy loss. Used implicitly when a critic is present. |

## Advantages

| Name | When to use |
|---|---|
| [GAE](advantages.md#gae) | Classic actor-critic. Needs a critic. |
| [GRPO](advantages.md#grpo) | Group-relative; per-prompt mean-and-std normalization. |
| [RLOO / LOOP](advantages.md#rloo-loop) | Group-relative leave-one-out. `loop` is an alias of `rloo`; both registered names work. |
| [REINFORCE++](advantages.md#reinforce-plus-plus) | Discounted REINFORCE with global whitening. |
| [REINFORCE++ baseline](advantages.md#reinforce-plus-plus-baseline) | Two-step: group baseline subtraction, then global whitening. |
| [REMAX](advantages.md#remax) | Greedy-decoding baseline (subtract a greedy rollout's return). |
| [OPO](advantages.md#opo) | Length-weighted group baseline. |
| [GRPO-PassK](advantages.md#grpo-passk) | Pass@k-shaped GRPO. |
| [GPG](advantages.md#gpg) | `α · (score − group_mean) / f_norm` with `α = batch_size / count_nonzero(scores)`. |
| [Chunked GAE](advantages.md#chunked-gae) | GAE via parallel prefix scan over chunks; same math, faster on long sequences. |
| [Kimi-K1.5](advantages.md#kimi-k1-5) | Length-shaped advantage from the Kimi-K1.5 recipe. |
| [Identity](advantages.md#identity) | No advantage transformation. Use when token-level rewards are already advantage-shaped. |

## Adding a new algorithm

```python
# axon/trainer/algos/loss/registry.py — add the enum value first
class LossFn(str, Enum):
    ...
    MY_LOSS = "my_loss"
```

```python
# axon/trainer/algos/loss/loss.py — implement and register
@register_loss(LossFn.MY_LOSS)
def my_loss_fn(data: DataProto, config: DictConfig) -> tuple[torch.Tensor, dict[str, Any]]:
    log_prob = data.batch["log_probs"]
    advantages = data.batch["advantages"]
    response_mask = data.batch["response_mask"]
    sampler_is_weights = data.batch.get("sampler_is_weights", None)
    # ... your loss math ...
    return loss, metrics
```

Then in your recipe yaml: `loss: my_loss`. No second registration step. Same shape for advantages (in `algos/advantages/`).

## Common compositions

| Recipe nickname | `loss` | `advantage` | Notable knobs |
|---|---|---|---|
| GRPO | `ppo` | `grpo` | symmetric clip, group-normalized advantages, `kl_coef = 0` |
| DAPO | `ppo` | `grpo` | asymmetric clip (clip-higher, e.g. `0.2 / 0.28`), token-level loss, no entropy bonus |
| RLOO / LOOP | `ppo` | `loop` (or `rloo`) | leave-one-out group baseline |
| GSPO | `gspo` | `grpo` | sequence-level stop-grad ratio |
| CISPO | `cispo` | `grpo` | token-level clipped IS weight, gradient kept through all tokens |
| GPG | `gpg` (= REINFORCE) | `gpg` | the GPG-specific `α` and `f_norm` shaping |
| Vanilla PPO | `ppo` | `gae` | needs a critic in the resource pool |
| REINFORCE | `reinforce` | `identity` (or `reinforce_plus_plus`) | simplest baseline |
