# Copyright 2025 Model AI Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import os
import random

import browsergym.miniwob
import gymnasium as gym
import pandas as pd

import axon


def process_fn(env_id):
    return {"env_name": "browsergym", "agent_name": "miniwob", "env_id": env_id}


if __name__ == "__main__":
    import importlib
    import os

    import browsergym.miniwob

    importlib.reload(browsergym.miniwob)
    # Get the directory for Axon repo (axon.__file__)
    AXON_DIR = os.path.dirname(os.path.dirname(os.path.abspath(axon.__file__)))
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default=os.path.join(AXON_DIR, "data", "miniwob"))
    parser.add_argument(
        "--train_ratio", type=float, default=0.80, help="Ratio of data to use for training (default: 76.8%)"
    )
    args = parser.parse_args()

    local_dir = args.local_dir
    os.makedirs(os.path.expanduser(local_dir), exist_ok=True)

    train_ratio = max(0.0, min(1.0, args.train_ratio))

    # Get all MiniWoB environment IDs from gym
    print(gym.envs.registry.keys())
    env_ids = [env_id for env_id in gym.envs.registry.keys() if env_id.startswith("browsergym/miniwob")]
    random.seed(42)
    random.shuffle(env_ids)

    # Split train/test
    train_size = int(train_ratio * len(env_ids))
    train_envs = env_ids[:train_size]
    test_envs = env_ids[train_size:]

    # Process train data
    train_data = [process_fn(env_id) for idx, env_id in enumerate(train_envs)]

    # Process test data
    test_data = [process_fn(env_id) for idx, env_id in enumerate(test_envs)]

    print("Train data size:", len(train_data))
    print("Test data size:", len(test_data))

    # Convert to DataFrame and save as Parquet
    train_df = pd.DataFrame(train_data)
    os.makedirs(os.path.join(local_dir, "train"), exist_ok=True)
    train_df.to_parquet(os.path.join(local_dir, "train", "miniwob.parquet"))
    test_df = pd.DataFrame(test_data)
    os.makedirs(os.path.join(local_dir, "test"), exist_ok=True)
    test_df.to_parquet(os.path.join(local_dir, "test", "miniwob.parquet"))
