#!/usr/bin/env python3
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
#
# NQ data processing adapted from Search-R1 / FlashRAG (github.com/PeterGriffinJin/Search-R1), Apache-2.0.
"""
NQ Dataset Processing for Search-R1 Training.

Processes the Natural Questions (NQ) dataset from FlashRAG into the format
expected by Search-R1 training.

Based on Search-R1/scripts/data_process/nq_search.py

Usage:
    python search_r1_dataset.py
"""

import argparse
import os

import datasets
import axon

# Get the directory for Axon repo (axon.__file__)
AXON_DIR = os.path.dirname(os.path.dirname(os.path.abspath(axon.__file__)))


def process_nq_data(output_dir: str = "./data/nq_search", template_type: str = "base"):
    """
    Process NQ dataset for Search-R1 style training.

    Matches Search-R1/scripts/data_process/nq_search.py

    Args:
        output_dir: Directory to save processed parquet files
        template_type: Template type ('base' is the only supported type)
    """
    print("Loading NQ dataset from FlashRAG...")
    dataset = datasets.load_dataset("RUC-NLPIR/FlashRAG_datasets", "nq")

    def process_example(example, idx, split):
        """
        Process a single example into format for Axon.
        Note: Instruction is in agent's system prompt, not in user message.
        """
        # Ensure question ends with '?'
        question = example["question"].strip()
        if question and not question.endswith("?"):
            question += "?"

        # Create data entry
        # prompt, extra_info (task metadata)
        return {
            "env_name": "search_r1",
            "question": question,
            "answer": example["golden_answers"],
        }

    # Process train and test splits
    print("Processing training data...")
    train_dataset = dataset["train"].map(lambda ex, idx: process_example(ex, idx, "train"), with_indices=True)

    print("Processing test data...")
    test_dataset = dataset["test"].map(lambda ex, idx: process_example(ex, idx, "test"), with_indices=True)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Save to parquet
    train_path = os.path.join(output_dir, "train.parquet")
    test_path = os.path.join(output_dir, "test.parquet")

    print(f"Saving training data to {train_path}...")
    train_dataset.to_parquet(train_path)

    print(f"Saving test data to {test_path}...")
    test_dataset.to_parquet(test_path)

    print("\nDataset processing complete!")
    print(f"  Train examples: {len(train_dataset)}")
    print(f"  Test examples: {len(test_dataset)}")
    print(f"  Output directory: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process NQ dataset for Search-R1 training")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(AXON_DIR, "data/axon-search-r1"),
        help="Directory to save processed data (default: ./data/axon-search-r1)",
    )
    parser.add_argument("--template_type", type=str, default="base", help="Template type to use (default: base)")

    args = parser.parse_args()

    process_nq_data(output_dir=args.output_dir, template_type=args.template_type)
