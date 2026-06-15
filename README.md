<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/axon-logo-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="docs/assets/axon-logo-light.svg">
    <img src="docs/assets/axon-logo-light.svg" alt="Axon logo" width="640">
  </picture>
</p>

<h2 align="center">Reinforcement Learning for Agentic Programs</h2>

<p align="center">
  <a href="https://axon-rl.readthedocs.io"><img alt="Documentation" src="https://img.shields.io/badge/Documentation-black?style=for-the-badge&logo=googledocs&logoColor=white"></a>
  <a href="https://modelcorp.ai"><img alt="Website" src="https://img.shields.io/badge/Website-000000?style=for-the-badge&logo=semanticweb&logoColor=white"></a>
  <a href="https://x.com/Agentica_"><img alt="Twitter/X" src="https://img.shields.io/badge/ModelAI-white?style=for-the-badge&logo=x&logoColor=000&color=000&labelColor=white"></a>
  <a href="https://huggingface.co/agentica-org"><img alt="Hugging Face" src="https://img.shields.io/badge/ModelAI-fcd022?style=for-the-badge&logo=huggingface&logoColor=000"></a>
  <a href="https://www.python.org/downloads/"><img alt="Python 3.10+" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white"></a>
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/License-Apache%202.0-000000?style=for-the-badge"></a>
</p>

**Axon** is a framework for post-training language agents with reinforcement learning. Bring your agent as plain Python, your algorithm, your cluster, and your model — Axon trains whatever you assemble, from a single H100 to large multi-node clusters for trillion-parameter models, and keeps it faithful to the run you rolled out.

In one line: write the rollout you want to run, and Axon records the exact LLM calls needed to train it.

## News

- **2026-06** — Axon is open source. Read the announcement: [Train the Agent You Actually Run](https://modelcorp.ai/blog/axon).

## Highlights

- **Programs, not configs.** Write the agent as plain async Python (`BaseProgram`); the engine turns every `generate` call into trained tokens read off one prefix tree — no re-tokenization between rollout and gradient.
- **One codebase, the whole matrix.** FSDP and Megatron-Core backends, colocated and disaggregated topologies, and sync and async drivers compose through mixins — so a new loss, kernel, or weight-sync path lands once and works everywhere.
- **Frontier models, composable algorithms.** Bridges for the Gemma4, GLM, GPT-OSS, DeepSeek-V3, and Qwen3 families (dense and MoE), and a registry of losses and advantages — GRPO, GSPO, DAPO, CISPO, and more — composed in yaml, not bolted on as new code paths.
- **Weight sync on the GPU fabric.** For disaggregated runs, a `RoutingTable` computes the per-parameter NCCL P2P plan across mismatched trainer/sampler layouts (TP / EP / PP), so a 671B model re-shards and syncs in seconds — no detour through host memory.
- **The run you train is the run you rolled out.** Axon matches the sampler's kernels and dtypes, replays MoE routing on the trainer, and corrects the residual per token — logging the gap every step. Exact where it can be, measured where it can't.

## Features

### Supported Models

Axon supports dense, MoE, and multimodal model families across parameter scales, on both FSDP and Megatron-Core.

| Supported model families |
|---|
| Gemma4 family · Qwen3.5 / Qwen3 / Qwen3-Next · Qwen2.5-VL (image inputs) · DeepSeek-R1/V3 · GLM-5.1 · GPT-OSS · Kimi K2 · Llama 3 |

vLLM is the primary inference engine; Axon ships a pinned vLLM fork that adds precision-parity and routing-replay support — see [install/vllm/README.md](install/vllm/README.md). New model families plug in through a model-bridge file plus a recipe yaml; see [docs/core-concepts/architecture.md](docs/core-concepts/architecture.md) for the integration path.


### Codebase Features

#### Interface — what you compose

| Feature | Description |
|---|---|
| **Agentic Program Abstraction** | `BaseProgram` *is* the rollout — plain async Python, not config. `ReactProgram` covers single-turn, multi-turn, and tool use; subclass it for parallel solvers, multi-agent, or search trees. |
| **Algorithm Registries** | Compose a loss and an advantage in yaml via `@register_loss` / `@register_advantage` — named methods like GRPO, GSPO, and DAPO are compositions, not separate code paths. |
| **OpenAI / MCP Surface** | OpenAI-compatible chat endpoint and MCP tool integration, so external agents (LangChain, OpenAI / Anthropic SDKs, custom clients) can drive rollouts. |
| **Tinker SDK Compatibility** | A Tinker-compatible training-client surface, so code written against the Tinker SDK runs on Axon-managed GPUs with minimal change. |
| **Code Execution Tools** | Run agent-generated code during rollouts through local, E2B cloud-sandbox, or LiveCodeBench backends (shipped with the `tools` / `code` recipes). |

#### Modeling — faithful training across model families

| Feature | Description |
|---|---|
| **Sampler-trainer agreement** | Keeps inference and training logprobs aligned so the PPO ratio doesn't drift — matched kernels and dtypes in the vLLM fork, per-token IS / rejection / veto correction, and the gap logged every step. |
| **MoE Routing Replay** | On the Megatron path, the trainer replays the exact expert routing the sampler used — removing the largest MoE-specific source of sampler-trainer drift. |
| **Multimodal Support** | Image-input RL, verified end-to-end on Qwen2.5-VL; Gemma4-VL / Qwen3-VL / GLM-4V / Kimi-VL are on the integration path. |
| **Multi-Token Prediction (MTP)** | Joint MTP + main-token loss wired into Megatron for RL. |
| **Curriculum & Adaptive Sampling** | Pluggable data samplers — exponential-weighted and threshold-masking curricula, or your own by import path. |

#### Infra — scale and performance

| Feature | Description |
|---|---|
| **Parallelism** | Megatron 6D parallelism (TP / PP / EP / ETP / CP / DP) and FSDP / FSDP2 sharding with Ulysses sequence parallelism — one recipe scales from a single GPU to trillion-parameter runs. |
| **P2P Weight Transfer** | For disaggregated runs, a `RoutingTable` syncs weights GPU-to-GPU over NCCL across mismatched trainer/sampler layouts — no detour through host memory. |
| **Async PPO with overlap** | The `AsyncPPO` driver overlaps rollout with training through a bounded queue, syncing weights after each optimizer step. |
| **Partial Rollout** | Straggling long rollouts are suspended, weight-synced, and resumed across updates instead of dropped (after Kimi k1.5). |
| **Custom Fused Kernels** | A Triton fused-MoE backward — the training pass vLLM's inference kernel lacks — plus a fused linear-cross-entropy / entropy kernel. |
| **Memory Optimizations** | Composable activation offload, optimizer-state offload, and gradient checkpointing for fitting large models. |
| **Low Precision Training** | bf16, fp8 (blockwise; mxfp8 on Blackwell), and int4 inference, with mixed-precision recipes. |
| **Async Distributed Checkpointing** | Non-blocking checkpoint saves in Megatron's distributed-checkpoint format, on a background thread. |

### Agent and environment surface

- **Shipped recipes** (`recipes/`): math, code, FrozenLake, search-R1, SWE, tools, MiniWoB, WebArena, formal math, geo3k (multimodal), parallel-thinker, multi-env, NeMo Gym, Verifiers. See the [recipe catalog](recipes/README.md).
- **Reward graders**: built-in graders for math, code (unit-test execution), F1 QA, GPQA, and IFBench (`axon/utils/rewards/`).
- **Tool surface**: tool-call parsers (Gemma4, Qwen, OpenAI Harmony, GLM, R1, JSON, XML), `LocalToolExecutor` / `HTTPToolExecutor`, and MCP integration via `MCPTool`.

---

## Installation

### Prerequisites
- Python >= 3.10
- CUDA 12.8 
- H100 (sm_90) or B200 (sm_100) GPUs
- conda or micromamba

### Quick Install (new environment)

```bash
# 1. Create and activate a new conda env
conda create -n axon python=3.10 -y
conda activate axon

# 2. Clone the repo (if not already cloned)
git clone https://github.com/modelcorp/axon.git
cd axon

# 3. Run full install (CUDA toolkit, Python deps, PyTorch, vllm, extensions)
bash install/install.sh
```

This runs five stages in order:
1. **CUDA toolkit** — installs CUDA 12.8, cuDNN, NCCL via conda
2. **PyTorch** — installs PyTorch 2.10 (cu128), torchvision, torchaudio, triton
3. **Python deps** — `pip install -e .` (transformers and the base Python requirements)
4. **Compiled extensions** — flash-attn, FlashInfer (sm_90+), transformer_engine, apex, megatron-core, vllm, plus CUDA-compiled flash-linear-attention and causal-conv1d
5. **Agent extras** — browsergym, swebench, and friends for the agentic recipes

FlashInfer is installed automatically on H100+ (sm_90+) and is required for models with GDN attention (Qwen3.5, Qwen3-Next).

### Dev Install (with vLLM fork)

For development that involves modifying vLLM:

```bash
# 1. Clone the pinned vllm fork
git clone https://github.com/modelcorp/vllm.git /path/to/vllm
cd /path/to/vllm && git checkout modelcorp/v0.19.0

# 2. Install axon + the fork (editable) in one step — the installer does the editable vllm install
cd /path/to/axon
VLLM_DEV_PATH=/path/to/vllm AXON_VLLM_DEV=1 bash install/install.sh
```

(Skip step 1 and omit `VLLM_DEV_PATH` to let the installer clone the fork into `./vllm` itself.)

See [install/vllm/README.md](install/vllm/README.md) for details on patch generation and vllm upgrades.

### Post-install Setup

```bash
huggingface-cli login
wandb login
axon --help
```

---

## Quick Start

End-to-end examples live in `recipes/`. The default FrozenLake smoke test uses Qwen3-30B-A3B on 8 H100/B200 GPUs; for smaller machines, start from one of the smaller recipe launchers.

### FrozenLake

```bash
cd recipes/frozenlake/

# Generate dataset
python data.py

# Download model locally (saves time)
huggingface-cli download Qwen/Qwen3-30B-A3B

# Run training. The recipe yaml encodes model + parallelism (8 H100s / 1 node).
./train_frozenlake_qwen_30b_a3b.sh
```

To use the managed CLI path instead of the shell wrapper:

```bash
axon train -c recipes/frozenlake/train_frozenlake_qwen_30b_a3b.yaml --foreground

# Or detach the run and manage it later:
axon train -c recipes/frozenlake/train_frozenlake_qwen_30b_a3b.yaml
axon status
axon logs <run_id> -f
axon cancel <run_id>
```

While it runs, expect config checks, trainer/sampler bring-up, and then reward/loss/sampler-trainer agreement metrics in the console and W&B. Checkpoints land in the recipe `output_dir`.

For custom tasks, start with [docs/guides/add-a-program.md](docs/guides/add-a-program.md).

---

## Development

### Setup

```bash
# Install dev dependencies
pip install -e .[dev]

# Set up pre-commit hooks
pre-commit install
```

### Code Style

```bash
# Lint code
ruff check .

# Auto-fix issues
ruff check --fix .

# Format code
ruff format .
```

### Testing

```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/recipes/math/test_agent.py

# With coverage
pytest --cov=axon tests/
```

### Type Checking

```bash
mypy axon/
```

---


## Acknowledgements

Axon began as a fork of [verl](https://github.com/volcengine/verl) and was then substantially rebuilt. Beyond verl, Axon is thankful for the vibrant LLM system ecosystem: [vLLM](https://github.com/vllm-project/vllm) for inference, [Megatron-Core](https://github.com/NVIDIA/Megatron-LM) and [mbridge](https://github.com/ISEEKYAN/mbridge) for model parallelism, and [slime](https://github.com/THUDM/slime), [miles](https://github.com/radixark/miles), [Flash-RL](https://github.com/LLM360/Flash-RL), and [TileLang](https://github.com/tile-ai/tilelang) for quantized rollout and custom kernels.

The authors are among the original creators of rLLM ([rllm-org/rllm](https://github.com/rllm-org/rllm)); we're grateful to everyone who has contributed to it.

Per-file attribution lives in file headers and [LICENSE](LICENSE).

---


## Citation

If you find Axon useful in your research or work, please cite:

```bibtex
@misc{axon2026,
  title  = {Axon: Reinforcement Learning for Agentic Programs},
  author = {Cai, Colin and Luo, Michael and Zhang, Tianjun and {Axon Contributors}},
  year   = {2026},
  howpublished = {\url{https://github.com/modelcorp/axon}}
}
```

---

## License

See [LICENSE](LICENSE).
