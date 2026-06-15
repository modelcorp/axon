# Tests

## Running tests

```bash
pytest                          # CPU-only tests (default)
pytest -m gpu                   # GPU-marked tests only; skipped without CUDA
pytest -m all                   # CPU + GPU-marked tests; GPU tests still skip without CUDA
```

## Filtering

```bash
pytest -k glm4                  # match test names
pytest tests/models/            # specific directory
pytest -m gpu -k Qwen3Next      # combine markers and filters
```

## GPU tests

GPU tests (`@pytest.mark.gpu`) are **excluded by default**. They:
- Currently require 8 visible H100+ GPUs through the shared mbridge fixture.
  Some individual cases use a smaller TP/PP layout after that availability
  check passes.
- Download real model weights from HuggingFace (~80GB+ first run, cached after)
- Run full mbridge pipeline: bridge creation, megatron init, weight loading, forward pass
- Launch worker processes via `torchrun` for distributed tests (TP/PP)

| Model | Size | Config | File |
|---|---|---|---|
| GLM-4-9B | 18.8 GB | TP=1, 1 GPU | `gpu/test_glm4.py` |
| Qwen3-8B | 16.4 GB | TP=1, 1 GPU | `gpu/test_qwen3.py` |
| GPT-OSS-20B | 41.8 GB | TP=1, 1 GPU | `gpu/test_gpt_oss.py` |
| Qwen3-Next-80B | ~152 GB | TP=2 PP=4, 8 GPUs | `gpu/test_qwen3_next.py` |
