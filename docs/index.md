---
hide:
  - navigation
  - toc
  - title
---

<div class="hero" markdown>

<p align="center">
  <img src="assets/axon-logo-light.svg#only-light" alt="Axon logo" width="520">
  <img src="assets/axon-logo-dark.svg#only-dark" alt="Axon logo" width="520">
</p>

## Reinforcement learning for agentic programs

Axon is a framework for post-training language agents with reinforcement learning. Bring your agent as plain Python, your algorithm, your cluster, and your model — Axon trains whatever you assemble, from a single H100 to multi-node, trillion-parameter clusters, and keeps it faithful to the run you rolled out.

[Get started in 5 minutes](getting-started/quickstart.md){ .md-button .md-button--primary }
[Read the architecture](core-concepts/architecture.md){ .md-button }

</div>

---

## What's in the box

<div class="grid cards" markdown>

- :material-puzzle: **`BaseProgram` is the rollout abstraction**

    A program defines what one rollout is. The shipped `ReactProgram` covers ReAct-style multi-turn loops (math, code, FrozenLake, SWE, search, tools); custom programs subclass `BaseProgram` directly for parallel solvers, multi-agent, and search trees. Tool-call parsers for every major model format; reward graders for math / code / F1 / GPQA / IFBench.

- :material-flash: **Sampler-trainer agreement**

    Inference and training compute logprobs through different kernels and dtypes; without mitigation, the gap drifts the PPO clip ratio over training. Axon matches kernels and dtypes in a pinned vLLM fork, applies per-recipe token-level IS correction with sequence veto, and runs MoE routing replay on the Megatron path.

- :material-swap-horizontal: **Two algorithm registries**

    A loss + an advantage estimator compose at config time through `@register_loss` and `@register_advantage`, so named methods like GRPO and DAPO are yaml compositions rather than new code paths.

- :material-rocket-launch: **One class hierarchy, composed by mixins**

    FSDP and Megatron-Core backends, hybrid and disaggregated topologies, and sync and async drivers compose at runtime via mixins (async runs disaggregated) — one set of classes, so a new loss, a checkpointing fix, or a weight-sync transport lands once.

- :material-graph: **Layout-mismatched P2P weight transfer**

    Trainer and sampler use different parallelism layouts — a large MoE model can train at TP=4 (leaning on expert and pipeline parallelism) and sample at TP=8–16. `RoutingTable` lets each side pick independently and computes the per-parameter NCCL P2P send/recv plan — including KV-head replication and MoE expert sharding.

- :material-connection: **Integrations**

    Tinker-SDK-compatible client, MCP tool registry, OpenAI-compatible chat endpoint, adapters for the Verifiers and NeMo Gym environment hubs. Multi-backend metrics tracking (W&B, MLflow, TensorBoard, ClearML, SwanLab, TrackIO).

</div>

---

## Supported Models

Axon supports dense, MoE, and multimodal model families across parameter scales, on both FSDP and Megatron-Core.

| Supported model families |
|---|
| Gemma4 family · Qwen3.5 / Qwen3 / Qwen3-Next · Qwen2.5-VL (image inputs) · DeepSeek-R1/V3 · GLM-5.1 · GPT-OSS · Kimi K2 · Llama 3 |

vLLM is the primary inference engine; Axon ships a pinned vLLM fork that adds precision-parity and routing-replay support.

---

## Quick install

```bash
conda create -n axon python=3.10 -y
conda activate axon
bash install/install.sh
```

See [Installation](getting-started/installation.md) for the full guide, including the dev-mode vLLM-fork install.

---

## Five minutes to a running RL job

```bash
cd recipes/frozenlake/
python data.py
huggingface-cli download Qwen/Qwen3-30B-A3B
./train_frozenlake_qwen_30b_a3b.sh
```

Walk through this step by step in the [Quickstart](getting-started/quickstart.md).

---

## Where to go next

- New to Axon? Start with the [Architecture overview](core-concepts/architecture.md).
- Building a custom program? Read [Add a program](guides/add-a-program.md).
- Looking for the API surface? Jump to the [API reference](reference/api/index.md).

---

## Acknowledgements

Axon began as a fork of [verl](https://github.com/volcengine/verl) and was then substantially rebuilt. It also builds on vLLM, Megatron-Core, mbridge, slime, miles, Flash-RL, and TileLang, among others. See the [Acknowledgements](https://github.com/modelcorp/axon#acknowledgements) in the README; file headers carry per-file attribution.

The authors are among the original creators of rLLM ([rllm-org/rllm](https://github.com/rllm-org/rllm)); we're grateful to everyone who has contributed to it.
