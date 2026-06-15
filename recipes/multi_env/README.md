# Multi-Environment

Mixture training: each parquet row carries `env_name` / `agent_name`, routed at rollout time. Defaults mix FrozenLake + Math.

```bash
python recipes/frozenlake/data.py     # any envs you want to mix
python recipes/math/data.py
cd recipes/multi_env/
./train_multi_env_qwen_30b_a3b.sh
```

Per-row routing happens in `ReactProgram.init_env_and_agent` (looks the row's `env_name` up in `ENV_CLASS_MAPPING`). Edit the script's `env_name` / `agent_name` / `train_files` to change the mixture.
