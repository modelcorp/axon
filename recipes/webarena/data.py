import argparse
import os
import random

import gymnasium as gym
import pandas as pd

import axon

if __name__ == "__main__":
    import importlib
    import os

    import browsergym.async_webarena

    importlib.reload(browsergym.async_webarena)

    AXON_DIR = os.path.dirname(os.path.dirname(os.path.abspath(axon.__file__)))
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default=os.path.join(AXON_DIR, "data/webarena"))
    parser.add_argument(
        "--train_ratio", type=float, default=0.8, help="Ratio of data to use for training (default: 80%)"
    )
    args = parser.parse_args()

    local_dir = args.local_dir
    os.makedirs(os.path.expanduser(local_dir), exist_ok=True)

    train_ratio = max(0.0, min(1.0, args.train_ratio))

    # Get all MiniWoB environment IDs from gym
    env_ids = [env_id for env_id in gym.envs.registry.keys() if env_id.startswith("browsergym_async/webarena")]

    def process_fn(env_id):
        return {"env_name": "browsergym", "agent_name": "webarena", "env_id": env_id}

    # Split train/test
    train_size = int(train_ratio * len(env_ids))  # 80% for training
    random.seed(42)
    random.shuffle(env_ids)
    train_envs = env_ids[:train_size]
    test_envs = env_ids[train_size:]

    # Process train data
    train_data = [process_fn(env_id) for env_id in train_envs]

    # Process test data
    test_data = [process_fn(env_id) for env_id in test_envs]

    print("Train data size:", len(train_data))
    print("Test data size:", len(test_data))

    # Convert to DataFrame and save as Parquet
    print("Saving train data to", os.path.join(local_dir, "train.parquet"))
    train_df = pd.DataFrame(train_data)
    train_df.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_df = pd.DataFrame(test_data)
    test_df.to_parquet(os.path.join(local_dir, "test.parquet"))
