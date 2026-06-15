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

import enum
import json
import logging
import os
import urllib.request
from typing import Any
from urllib.parse import urlparse

import axon
from axon.utils.system_prompts import (
    LCB_FORMATTING_MESSAGE_WITH_STARTER_CODE,
    LCB_FORMATTING_WITHOUT_STARTER_CODE,
    LCB_SYSTEM_MESSAGE_GENERIC,
)

logger = logging.getLogger(__name__)

AGENTICA_GCP_BUCKET = "https://storage.googleapis.com/agentica-dataset"

# Datasets sourced from Hugging Face instead of the GCS bucket.
# Maps the lowercased enum value -> (hf_repo_id, split, optional column rename map).
# Rows are written to disk as a JSON list with the same fields used elsewhere
# (problem, answer, ...). The rename map is applied before serialization so the
# downstream `process_fn` in recipes can stay uniform.
HF_DATASET_SOURCES: dict[str, tuple] = {
    "aime26": ("MathArena/aime_2026", "train"),
    "acereason_math": ("nvidia/AceReason-Math", "train"),
    "nemotron_rl_math": (
        "nvidia/Nemotron-RL-math-OpenMathReasoning",
        "train",
        {"question": "problem", "expected_answer": "answer"},
    ),
}


class TrainDataset:
    class Math(enum.Enum):
        # The standard American beginner competitions.
        AIME = "AIME"
        AMC = "AMC"
        # Omni math dataset
        OMNI_MATH = "OMNI_MATH"
        # Unique Olympiad problems from NUMINA
        NUMINA_OLYMPIAD = "OLYMPIAD"
        # Dan Hendrycks math
        MATH = "MATH"
        GSM8K = "GSM8K"
        STILL = "STILL"
        DEEPSCALER = "DEEPSCALER"
        POLARIS = "POLARIS"
        ACEREASON_MATH = "ACEREASON_MATH"
        NEMOTRON_RL_MATH = "NEMOTRON_RL_MATH"

    class Code(enum.Enum):
        TACO = "TACO"
        APPS = "APPS"
        CODEFORCES = "CODEFORCES"
        CODE_CONTESTS = "CODE_CONTESTS"
        LIVECODEBENCH = "LIVECODEBENCH"
        LEETCODE = "LEETCODE"
        PRIMEINTELLECT = "PRIMEINTELLECT"
        KODCODE = "KODCODE"


class TestDataset:
    class Math(enum.Enum):
        AIME24 = "AIME24"
        AIME25 = "AIME25"
        AIME26 = "AIME26"
        AMC23 = "AMC23"
        MATH = "MATH"
        GSM8K = "GSM8K"
        MINERVA = "MINERVA"
        OLYMPIAD_BENCH = "OLYMPIAD_BENCH"

    class Code(enum.Enum):
        TACO = "TACO"
        CODEFORCES = "CODEFORCES"
        CODE_CONTESTS = "CODE_CONTESTS"
        LIVECODEBENCH = "LIVECODEBENCH"
        LEETCODE = "LEETCODE"
        HUMANEVALPLUS = "HUMANEVALPLUS"


def load_dataset(
    dataset_enum: TrainDataset.Math | TrainDataset.Code | TestDataset.Math | TestDataset.Code,
) -> list[dict[str, Any]]:
    """Load a dataset from a JSON file based on the dataset enum.

    This function takes a dataset enum value and loads the corresponding JSON file
    from the appropriate directory structure. The directory structure follows the pattern:
    {category_dir}/{data_dir}/{dataset_name}.json
    where:
    - category_dir is either 'math' or 'code'
    - data_dir is either 'train' or 'test'
    - dataset_name is the lowercase value of the enum

    Args:
        dataset_enum: An enum value from either TrainDataset or TestDataset classes,
                     specifying which dataset to load.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing the dataset items.
            Each dictionary represents one item in the dataset with its associated fields.

    Raises:
        ValueError: If the dataset file cannot be found or contains invalid JSON.

    Examples:
        >>> # Load training AIME dataset
        >>> aime_data = load_dataset(TrainDataset.Math.AIME)
        >>> # Load test APPS dataset
        >>> apps_data = load_dataset(TestDataset.Code.APPS)
    """
    dataset_name = dataset_enum.value.lower()
    category_dir = dataset_enum.__class__.__name__.lower()

    # Determine if dataset is for training or testing
    if dataset_enum.__class__ in [TrainDataset.Math, TrainDataset.Code]:
        data_dir = "train"
    else:
        data_dir = "test"

    # Construct file path
    axon_dir = os.path.dirname(os.path.dirname(axon.__file__))
    current_dir = os.path.join(axon_dir, "datasets")

    file_path = os.path.join(current_dir, category_dir, data_dir, f"{dataset_name}.json")

    if not os.path.exists(file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        if dataset_name in HF_DATASET_SOURCES:
            spec = HF_DATASET_SOURCES[dataset_name]
            repo_id, split = spec[0], spec[1]
            rename_map: dict[str, str] = spec[2] if len(spec) >= 3 else {}
            try:
                from datasets import load_dataset as hf_load_dataset

                logger.info("Downloading dataset from Hugging Face %s (split=%s) to %s", repo_id, split, file_path)
                hf_ds = hf_load_dataset(repo_id, split=split)
                if rename_map:
                    # Drop columns not referenced in the rename map and rename the rest.
                    keep = set(rename_map.keys())
                    drop_cols = [c for c in hf_ds.column_names if c not in keep]
                    hf_ds = hf_ds.remove_columns(drop_cols).rename_columns(rename_map)
                rows = [dict(r) for r in hf_ds]
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(rows, f, ensure_ascii=False)
                logger.info("Successfully downloaded %s dataset (%d rows)", dataset_name, len(rows))
            except Exception as e:
                raise ValueError(f"Failed to download dataset from Hugging Face {repo_id}: {str(e)}") from e
        else:
            # Download from GCS bucket gs://agentica-datasets/[category_dir]/[data_dir]/[dataset_name].json
            gcs_url = f"{AGENTICA_GCP_BUCKET}/{category_dir}/{data_dir}/{dataset_name}.json"
            try:
                parsed = urlparse(gcs_url)
                if parsed.scheme not in {"https"}:
                    raise ValueError(f"Unsupported URL scheme for dataset download: {parsed.scheme}")
                logger.info("Downloading dataset from %s to %s", gcs_url, file_path)
                # URL scheme validated above.
                urllib.request.urlretrieve(gcs_url, file_path)  # nosec B310
                logger.info("Successfully downloaded %s dataset", dataset_name)
            except Exception as e:
                raise ValueError(f"Failed to download dataset from {gcs_url}: {str(e)}") from e

    try:
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)
        return data
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON format in {file_path}") from None
    except Exception as e:
        raise ValueError(f"Error loading dataset: {str(e)}") from e


def fetch_live_code_bench_system_prompt(prompt: str, starter_code: str | None = None):
    """Fetch system prompt for LiveCodeBench format."""
    # https://github.com/LiveCodeBench/LiveCodeBench/blob/main/lcb_runner/prompts/code_generation.py
    prompt = LCB_SYSTEM_MESSAGE_GENERIC + "\n\n" + prompt
    if starter_code:
        prompt += f"### Format: {LCB_FORMATTING_MESSAGE_WITH_STARTER_CODE}\n"
        prompt += f"```python\n{starter_code}\n```\n\n"
    else:
        prompt += f"### Format: {LCB_FORMATTING_WITHOUT_STARTER_CODE}\n"
        prompt += "```python\n# YOUR CODE HERE\n```\n\n"
    prompt += "### Answer: (use the provided format with backticks)\n\n"
    return prompt
