# Sampler-trainer agreement

Inference engines (vLLM, SGLang) and training frameworks (Megatron, FSDP) compute logprobs through different kernel and dtype combinations. Without active mitigation the gap widens over training, the PPO clip ratio drifts, and the run goes unstable.

Axon closes the gap three ways — matching kernels and dtypes in the inference engine, replaying MoE routing on the trainer, and correcting whatever residual remains per token (importance sampling, rejection, or veto) — and it measures what's left every step. Exact where it can be, measured where it can't.

## Kernel and dtype matching

Axon ships against a pinned vLLM fork. Per-family patches pick the kernel and accumulation-dtype combination closest to the trainer-side HF Transformers / Megatron-Core path. The fork also exposes an opt-in batch-invariant ops mode that swaps non-deterministic vLLM ops for deterministic equivalents — off by default.

## MoE routing replay

On the Megatron path, the inference engine emits the per-token expert routing decision and the trainer forces the same routing on the same tokens. Without this, MoE training is materially less stable — sampler and trainer pick different experts and the resulting gradient noise compounds. Toggleable so the same run can be measured with and without.

## Off-policy correction: importance sampling, rejection, and veto

When the trainer-recomputed logprob and the sampler's logprob disagree, Axon can correct for the gap instead of training through it: reweight a token's loss by its importance ratio, reject tokens or sequences whose ratio falls outside an accept band, or veto a whole sequence when a single token's ratio is catastrophic. All are configured per recipe via `loss_args`:

| Knob | What it controls |
|---|---|
| `sampler_is` | `null` / `token` / `sequence` — importance-weight correction mode |
| `sampler_is_threshold` | Upper truncation threshold for the IS weight |
| `sampler_rs` | `null` / `token` / `sequence` / `geometric` — rejection-sampling mode |
| `sampler_rs_threshold` (+ `…_lower`) | Accept band for rejection sampling (lower defaults to `1/upper`) |
| `sampler_token_veto_threshold` | Min token-level IS weight below which the whole sequence is vetoed |

## MTP-aware training

Some MoE families ship a second, multi-token-prediction (MTP) head whose log-probs drift from the main head if it isn't trained alongside it. Axon wires joint MTP + main-token training on the Megatron path for the families that carry that head, including Qwen3-Next.

## Diagnostic metrics

Every step Axon logs sampler-trainer agreement statistics: `batch/sampler_probs_diff_mean` in probability space (e.g. `0.01` is a 1% mean probability gap), the matching `batch/sampler_logprobs_diff_*` in log-prob space, `batch/sampler_actor_probs_pearson_corr`, and the IS-weight / veto stats (`sampler_corr/sampler_is_veto_fraction`).
