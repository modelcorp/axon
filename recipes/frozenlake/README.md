# FrozenLake

The smoke-test recipe: a multi-turn agent navigating a procedurally-generated frozen-lake grid. Small enough for a single node, and it exercises the full agent → environment → trainer loop. Run this first.

## Run

```bash
cd recipes/frozenlake/
python data.py                                    # build train/val parquet
huggingface-cli download Qwen/Qwen3-30B-A3B
./train_frozenlake_qwen_30b_a3b.sh
```
