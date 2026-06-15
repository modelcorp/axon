# Math

Single-turn math reasoning — the simplest recipe shape. The agent produces a step-by-step solution and a boxed answer; the environment grades it with `math_reward_fn` (exact-match and symbolic equality via math-verify / sympy, with an optional LLM grader).

## Run

```bash
cd recipes/math/
python data.py            # build parquet (writes to repo-root data/math/)
huggingface-cli download deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
./train_deepscaler_fsdp.sh
```
