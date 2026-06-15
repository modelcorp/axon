# Add a model family

This guide describes the integration path for a new model family in Axon.

## What "adding a model" actually entails

Five concerns, in roughly the order you'll hit them:

1. **Model code on the trainer side** — Megatron bridge or FSDP-side wrapper.
2. **Weight conversion** — HF ↔ Megatron for the bridge path (FSDP loads HF directly, no conversion).
3. **Inference-side support** — vLLM model file (often upstream already has it; sometimes needs a precision-parity patch).
4. **Sampler-trainer agreement** — kernel and dtype matching against the trainer-side path, for any unusual ops (softcap, GELU-tanh, etc.).
5. **A recipe yaml** — wire it all together.

Most of the time, items 1, 2, and 5 are the work. 3 is upstream's, 4 is a one-time audit.

## Where each piece lives

| Concern | Megatron path | FSDP path |
|---|---|---|
| Model code | `axon/models/mbridge/<family>.py` | `axon/models/transformers/<family>.py` |
| Weight conversion | Inside the same bridge file (`load_weights`, `export_weights`) | none — FSDP loads directly from HF |
| Precision-parity patches (if needed) | a module in `axon/monkey_patches/megatron/` | a module in `axon/monkey_patches/transformers/` |
| vLLM patches (if needed) | Axon's fork or upstream to vLLM | (same) |
| Recipe yaml | `recipes/<task>/train_<task>_<family>.{sh,yaml}` | (same) |


## The Megatron bridge

A bridge file does three things:

1. **Constructs the Megatron model** — picks the right Megatron `TransformerConfig` and instantiates the model with the right TP / PP / EP layout.
2. **Loads HF weights into Megatron format** — `load_weights(models, weights_path)`. This is the part that takes the most care: parameter-name mapping (the `_weight_name_mapping_*` helpers), per-shard slicing, KV-head replication for GQA, expert-tensor reshaping for MoE.
3. **Exports Megatron state back to HF format** — `export_weights(models)`, the inverse, used when you want to ship a checkpoint as a standard HF model.

`axon/models/mbridge/export_weights_patch.py` provides shared utilities for the conversion path — reuse those before adding new ones.

## The FSDP wrapper

For FSDP, the model loads directly from HF. Per-family files in `axon/models/transformers/` carry compatibility patches — custom forward hooks, fused kernels, op fixes — for families HF doesn't handle cleanly; the shard / offload / gradient-checkpointing policy itself is configured by the FSDP trainer (`axon/utils/fsdp/`, `axon/trainer/fsdp_trainer.py`).

If your family has unusual ops that need special handling under FSDP (Gemma4's three RMSNorms, GPT-OSS's attention sinks), add a patch module under `axon/monkey_patches/transformers/` (e.g. `qwen3_5_gdn.py`).

## The recipe yaml

A recipe is a flat set of overrides merged onto `config.yaml` at launch — model, parallelism, sampler, algorithm. Minimum diff over an existing recipe:

```yaml
# recipes/<task>/train_<task>_<family>.yaml
model_path: <hf-model-id-or-local-path>     # the tokenizer loads from the same path

actor:
  megatron:                          # FSDP path: use `fsdp:` instead (fsdp_size, ulysses_sequence_parallel_size)
    tensor_model_parallel_size: <pick for your GPU layout>
    pipeline_model_parallel_size: 1  # raise for very large models
    expert_model_parallel_size: <if MoE>

sampler:
  tensor_model_parallel_size: <independent of the trainer's TP — sized to hold the model for inference; smaller for dense, often larger for a big MoE>
  enforce_eager: false
  max_model_len: <generation budget>

# loss / advantage / data — copy from a similar recipe
```

The shell wrapper script picks up `NUM_NODES`, `NUM_GPUS_PER_NODE`, and resource-pool layout from environment variables.

## Sampler-trainer agreement audit

After the model trains in isolation (loss decreases, eval reward improves), the next thing to check is sampler-trainer agreement — see [Sampler-trainer agreement](../core-concepts/sampler-trainer-agreement.md) for the framework.

Process:

1. Run a few steps and look at the `batch/sampler_probs_diff_*` stats (`_mean`, `_max`) in the logs.
2. If `batch/sampler_probs_diff_mean` stays above ~0.005, the kernel and dtype matching needs work for this family:
   - RMSNorm dtype matching the trainer?
   - Vocab projection dtype?
   - Activation function (GELU exact vs tanh)?
   - Any model-specific op (softcap, attention sink, GQA replication)?
3. If the residual can't be driven below ~0.01, lean on token-level IS correction in `loss_args.sampler_is` and live with the residual.
4. For MoE, turn on routing replay — it removes the largest MoE-specific source of drift.

## Where to ask questions

- The other bridge files encode every per-family workaround — read the closest one before writing new code.
- `axon/monkey_patches/<system>/` modules carry the precision-parity work, with comments explaining what each patch does and why.
- Open an issue if you hit a class of problem that none of the existing families have hit.
