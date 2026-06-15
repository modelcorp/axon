# Parallelism and performance

Axon supports six parallelism axes on the Megatron-Core backend, plus a set of throughput and memory knobs that compose with them. The FSDP / FSDP2 backend uses sharded data parallelism (with HSDP across shard groups) and Ulysses sequence parallelism. The same recipe scales from a single H100 up to multi-node trillion-parameter runs.

## The six axes

| Axis | What it splits | Where it lives |
|---|---|---|
| **TP** — Tensor parallel | Each layer's weight matrices across GPUs | Megatron parallel groups |
| **PP** — Pipeline parallel | Layers across stages | Megatron pipeline schedules |
| **EP** — Expert parallel | MoE experts across GPUs | Megatron EP groups |
| **ETP** — Expert tensor parallel | Within-expert tensor split | Megatron expert-TP groups |
| **CP** — Context parallel (ring attention) | Long sequences across GPUs | `axon/utils/megatron/cp_utils.py` |
| **DP** — Data parallel | Distinct minibatches across replicas | The remaining dimension |

These six are the Megatron-Core axes. The FSDP / FSDP2 backend shards parameters as data parallel (FULL_SHARD, or HSDP across shard groups) and adds Ulysses sequence parallelism (`axon/utils/ulysses.py`) for long-context runs where CP is overkill or doesn't fit the model's attention pattern.

CP uses ring-attention with local-loss computation — each GPU computes attention on its slice of the sequence and the loss on its slice, which avoids redundant all-gathers. Long-context RL trainings (32K+ tokens) rely on this.

EP and ETP compose with [MoE routing replay](sampler-trainer-agreement.md#moe-routing-replay) so MoE training scales without sampler-trainer drift.

## RoutingTable — making mismatched layouts work

Actor and sampler use different parallelism layouts. A 235B-MoE recipe, for instance, runs the trainer at TP=4 (with expert and pipeline parallelism carrying the rest) and the sampler at TP=8–16 (to fit the model and KV cache for low-latency decoding). Forcing both sides to share a layout leaves throughput on the table.

`RoutingTable` (`axon/utils/p2p/`) handles the head/group arithmetic so weight transfer stays correct when actor TP, EP, or PP differ from sampler TP / EP / PP. For each parameter, the routing table computes which `(actor_rank, sampler_rank)` pair owns which slice and emits the NCCL P2P send/recv plan. Per-model helpers handle family-specific quirks — Gemma4 KV-head replication, MoE expert sharding for Qwen3.5 / DeepSeek-V3 / GLM-5.1.

Weight transfer is direct GPU-to-GPU over NCCL, with no detour through host memory or disk.

## Memory knobs

Three composable mechanisms for fitting frontier models into available HBM:

### Activation offload

`axon/utils/fsdp/activation_offload.py` moves saved activations to CPU between the forward and backward passes. The trade-off is bandwidth — copying activations across PCIe — for memory headroom.

### Optimizer-state offload

Adam state lives at 2-3× the parameter count (parameters + first moment + second moment). Offloading it to CPU is the difference between fitting and not fitting on a given GPU count for the largest models. Configurable via the trainer config.

### Gradient checkpointing

The classic technique — recompute selected activations during backward instead of storing them. Axon's gradient-checkpointing is per-layer-granular and composes with both offload mechanisms.

## Throughput knobs

### Dynamic token-batching

With `use_dynamic_bsz`, micro-batches are packed by token count (`max_token_len_per_gpu`) rather than a fixed sample count, so a batch of mixed-length sequences keeps every GPU near its token budget instead of padding out to the longest sequence.

### Custom fused kernels

Where the off-the-shelf kernels leave performance on the table, Axon ships its own:

| Kernel | File |
|---|---|
| Fused-experts forward (reuses vLLM's inference kernel) | `axon/utils/kernel/fused_experts.py` |
| Fused-MoE backward (the training pass vLLM's kernel lacks) | `axon/utils/kernel/fused_moe_backward.py` |
| Fused linear-cross-entropy / entropy | `axon/utils/kernel/linear_cross_entropy.py`, `kernels.py` |

For the largest MoE models, the fused-MoE backward is the difference between viable and unviable step times.

## I/O knobs

### Async distributed checkpointing

`axon/utils/megatron/async_zarr_dist_checkpointing.py` writes checkpoints in Megatron's distributed-checkpoint format from a background thread, so I/O does not block training.

### PEFT / LoRA rank-aware checkpointing

`axon/utils/megatron/peft_utils.py` handles rank-aware PEFT / LoRA checkpoints — only the trainable adapter weights are saved, with the rank metadata preserved for resume.

## Data-side knobs

### Sequence-length balancing

`axon/utils/seqlen_balancing.py` balances DP ranks under the long-tail sequence-length distribution typical of RL trajectories. Without it, one slow rank gates the entire DP group.

### Async tokenizer pool

`axon/utils/tokenizer_pool.py` runs tokenization in a process pool to keep the engine event loop free of CPU-bound work under high QPS.

## A note on tuning

There is no single right setting; the right combination depends on the model, the GPU count, and the recipe. Rules of thumb:

1. **Start with the recipe's defaults.** Every shipped recipe in `recipes/` has been tuned for a specific GPU layout — match the layout if you can.
2. **Measure before you cut.** `validate_config.py` will tell you if a setting is illegal; only your throughput numbers will tell you if it's *good*.
3. **Memory first, throughput second.** Get a configuration that runs end-to-end at all, then turn on offload and dynamic batching to claw back throughput.
