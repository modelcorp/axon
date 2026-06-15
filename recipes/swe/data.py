import argparse
import json
import os

import pandas as pd
from datasets import load_dataset

import axon

# Get the directory for Axon repo (axon.__file__)
AXON_DIR = os.path.dirname(os.path.dirname(os.path.abspath(axon.__file__)))

SWE_DATASETS = [
    "R2E-Gym/R2E-Gym-Subset",
    "R2E-Gym/R2E-Gym-Lite",
    "R2E-Gym/R2E-Gym-V1",
    "R2E-Gym/SWE-Bench-Lite",
    "R2E-Gym/SWE-Bench-Verified",
    "r2e-edits/SweSmith-RL-Dataset",
]


def main():
    parser = argparse.ArgumentParser(description="Generate programs using specified environment and policy.")
    parser.add_argument("--local_dir", default=os.path.join(AXON_DIR, "data/swe"))

    args = parser.parse_args()

    local_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_dir, exist_ok=True)

    def process_fn(row):
        row_dict = dict(row)
        row_dict["env_name"] = "swe"
        return json.dumps(row_dict)

    revision = os.environ.get("HF_HUB_REVISION")

    for dataset_name in SWE_DATASETS:
        print(f"Processing dataset: {dataset_name}")
        try:
            # Load the dataset dictionary (which contains splits like 'train' or 'test')
            dataset_splits = load_dataset(dataset_name, revision=revision)  # nosec B615
        except Exception as e:
            print(f"Failed to load dataset {dataset_name}: {e}")
            continue

        output_name_base = dataset_name.split("/")[-1].replace("-", "_")  # Use underscore for consistency

        # Determine which split exists ('train' or 'test')
        if "train" in dataset_splits:
            split_name = "train"
            split_data = dataset_splits["train"]
        elif "test" in dataset_splits:
            split_name = "test"
            split_data = dataset_splits["test"]
        else:
            print(f"Skipping {dataset_name} as it contains neither 'train' nor 'test' split.")
            continue

        print(f"Using '{split_name}' split for {dataset_name}")

        # Process the data from the identified split
        processed_data = [process_fn(row) for row in split_data]

        # Create DataFrame and save to a single parquet file
        df = pd.DataFrame(processed_data)
        output_filepath = os.path.join(local_dir, f"{output_name_base}.parquet")
        df.to_parquet(output_filepath)
        print(f"Saved {len(df)} records from '{split_name}' split to {output_filepath}")


if __name__ == "__main__":
    main()
