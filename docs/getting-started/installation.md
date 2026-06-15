# Installation

## Prerequisites

- Python ≥ 3.10
- CUDA 12.8 (the installer pins the 12.8 toolkit)
- H100 (sm_90) or B200 (sm_100) GPUs — count is recipe-specific (the FrozenLake quickstart needs 8 on one node; 7B/14B recipes run on fewer)
- `conda` or `micromamba`

Axon is tested on Linux. The installer detects your GPU arch and builds the compiled extensions for it; H100 (sm_90) and B200 (sm_100) are the primary targets, with an Ampere (sm_80, e.g. A100) build path available. FlashInfer ships wheels for sm_90+ and is required for models with GDN attention (Qwen3.5, Qwen3-Next).

## Quick install — new environment

```bash
conda create -n axon python=3.10 -y
conda activate axon

cd /path/to/axon
bash install/install.sh
```

`install/install.sh` runs five stages in order:

1. **CUDA toolkit** — CUDA 12.8, cuDNN, NCCL via conda.
2. **PyTorch** — PyTorch 2.10 (cu128), torchvision, torchaudio, triton.
3. **Python deps** — `pip install -e .` (transformers and the rest of the base Python requirements).
4. **Compiled extensions** — flash-attn, FlashInfer (sm_90+), transformer_engine, Apex, Megatron-Core, vLLM, plus CUDA-compiled flash-linear-attention and causal-conv1d.
5. **Agent extras** — browsergym, swebench, and friends for the agentic recipes.

Megatron-Core is installed with `--no-deps` to avoid a NumPy<2 conflict.

## Dev install — editable vLLM fork

Axon ships against a pinned vLLM fork that adds the precision-parity patches and routing-replay capture used by the sampler-trainer agreement. For development that involves modifying vLLM:

```bash
# 1. Clone the pinned vLLM fork
git clone https://github.com/modelcorp/vllm.git /path/to/vllm
cd /path/to/vllm && git checkout modelcorp/v0.19.0

# 2. Install axon and the fork (editable) in one step — the installer does the editable vLLM install
cd /path/to/axon
VLLM_DEV_PATH=/path/to/vllm AXON_VLLM_DEV=1 bash install/install.sh
```

(Skip step 1 and omit `VLLM_DEV_PATH` to let the installer clone the fork into `./vllm` itself.)

See [`install/vllm/README.md`](https://github.com/modelcorp/axon/blob/main/install/vllm/README.md) for the full walkthrough on patch generation and vLLM upgrades.

## Post-install setup

```bash
huggingface-cli login
wandb login
```

These credentials let recipes pull weights from the Hugging Face Hub and stream metrics to Weights & Biases.

## Verify the install

```bash
axon --help
```

The command prints the four CLI commands: `train`, `status`, `logs`, `cancel`. If `axon` is not on your `PATH`, activate the conda env and rerun the editable install.

For a deeper smoke test, run the [FrozenLake quickstart](quickstart.md) — it exercises the trainer, sampler, and engine end-to-end on a 30B-class model.

## Optional dependencies

| Group | Install command | What you get |
|---|---|---|
| Docs | `pip install -e ".[docs]"` (or `pip install -r install/dependencies/requirements-docs.txt`) | mkdocs, mkdocs-material, mkdocstrings — for building this site locally |
| Dev | `pip install -r install/dependencies/requirements-dev.txt` | ruff, pytest, pre-commit |
| Megatron-only | already covered by `install.sh` | Megatron-Core + Apex + transformer_engine |
| SGLang (alt sampler) | `pip install -r install/dependencies/requirements-sglang.txt` | SGLang as an alternative inference path |

## Troubleshooting

??? failure "FlashInfer wheel not found"
    FlashInfer ships precompiled wheels only for sm_90+ GPUs. On older GPUs, models that require GDN attention (Qwen3.5, Qwen3-Next) will not load. The CUDA-graph code path in vLLM also assumes FlashInfer is present on H100.

??? failure "`mkdocs` not on path after install"
    `pip install -e ".[docs]"` installs into the current env. If you ran the install with the wrong conda env active, re-activate and re-install.

??? failure "Megatron-Core import error"
    `install.sh` installs Megatron-Core with `--no-deps`. If you upgraded NumPy after the install, you may have shadowed the pinned version. Reinstall with `pip install --no-deps megatron-core==<pinned>`.
