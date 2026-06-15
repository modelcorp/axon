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
import logging
import os
import re
import shutil

logger = logging.getLogger(__name__)


def is_valid_checkpoint(ckpt_path):
    """
    Returns True if the checkpoint directory has all required files/folders.

    Supports both FSDP and Megatron checkpoint structures:
    - FSDP sharded/both: actor/model_world_size_*, optim_world_size_*, extra_state_world_size_* files
    - FSDP hf: actor/optim_world_size_*, extra_state_world_size_* files + huggingface/ with model weights
    - Megatron: actor/*.distcp files or common.pt (distributed checkpoint format)

    Both require:
    - data.pt at the checkpoint root
    - actor/ folder with huggingface/ subfolder containing config + tokenizer
    """
    # 1) Check for required top-level files
    data_file = os.path.join(ckpt_path, "data.pt")
    if not os.path.exists(data_file):
        logger.warning("Checkpoint %s is missing required file: data.pt", ckpt_path)
        return False

    # 2) Check for 'actor' folder
    actor_dir = os.path.join(ckpt_path, "actor")
    if not os.path.isdir(actor_dir):
        logger.warning("Checkpoint %s is missing the 'actor' folder.", ckpt_path)
        return False

    # 3) Check for huggingface folder (always created, contains config + tokenizer)
    hf_dir = os.path.join(actor_dir, "huggingface")
    if not os.path.isdir(hf_dir):
        logger.warning("Checkpoint %s is missing 'huggingface' folder in 'actor'.", ckpt_path)
        return False

    # 4) Check for valid checkpoint format (FSDP or Megatron)
    actor_files = os.listdir(actor_dir)

    # Check for Megatron distributed checkpoint files (*.distcp or common.pt)
    has_megatron_distcp = any(f.endswith(".distcp") for f in actor_files)
    has_megatron_common = "common.pt" in actor_files
    if has_megatron_distcp or has_megatron_common:
        # Megatron checkpoint - distributed checkpoint files exist
        return True

    # Check for FSDP sharded checkpoint files
    # Optimizer and extra_state shards are always required
    # Model shards are optional (may be saved in HF format instead for 'hf' save_mode)
    required_prefixes = ["optim_world_size_", "extra_state_world_size_"]
    for prefix in required_prefixes:
        matched_files = [f for f in actor_files if f.startswith(prefix) and f.endswith(".pt")]
        if not matched_files:
            logger.warning(
                "Checkpoint %s is missing required FSDP shard files with prefix '%s' in 'actor'.", ckpt_path, prefix
            )
            return False

    # Model shards OR HF model weights must exist
    has_model_shards = any(f.startswith("model_world_size_") and f.endswith(".pt") for f in actor_files)
    hf_files = os.listdir(hf_dir) if os.path.isdir(hf_dir) else []
    has_hf_model = any(f.endswith(".safetensors") or f == "pytorch_model.bin" for f in hf_files)

    if not has_model_shards and not has_hf_model:
        logger.warning("Checkpoint %s is missing model weights (no model shards or HF model found).", ckpt_path)
        return False

    return True


def get_checkpoint_directories(path: str) -> list[tuple[int, str]]:
    """
    Find all checkpoint directories in the given path.

    Args:
        path: Base directory containing checkpoints.

    Returns:
        List of (step_number, full_path) tuples for each checkpoint directory,
        sorted by step number ascending.
    """
    if path is None or not os.path.exists(path):
        return []

    try:
        entries = os.listdir(path)
    except Exception as e:
        logger.warning("Failed to list directory %s: %s", path, e)
        return []

    # Compile patterns for different checkpoint directory formats
    patterns = [
        re.compile(r"^(\d+)$"),  # [ID]
        re.compile(r"^step_(\d+)$"),  # step_[ID]
        re.compile(r"^global_step_(\d+)$"),  # global_step_[ID]
    ]

    candidates = []
    for entry in entries:
        entry_path = os.path.join(path, entry)
        if os.path.isdir(entry_path):
            for pattern in patterns:
                m = pattern.match(entry)
                if m:
                    candidates.append((int(m.group(1)), entry_path))
                    break  # Found a match, no need to check other patterns

    # Sort by step number ascending
    return sorted(candidates, key=lambda x: x[0])


def delete_oldest_checkpoints(path: str, max_to_keep: int) -> list[str]:
    """
    Delete oldest checkpoints, keeping at most max_to_keep checkpoints.

    Args:
        path: Base directory containing checkpoints (e.g., "{output_dir}/{project}/{experiment}/checkpoints/")
        max_to_keep: Maximum number of checkpoints to keep. If None or <= 0, no deletion occurs.

    Returns:
        List of paths that were deleted.

    Example:
        >>> delete_oldest_checkpoints("/path/to/checkpoints", max_to_keep=3)
        # If checkpoints exist: step_1, step_2, step_3, step_4, step_5
        # Deletes: step_1, step_2
        # Keeps: step_3, step_4, step_5
    """
    if max_to_keep is None or max_to_keep <= 0:
        return []

    checkpoints = get_checkpoint_directories(path)
    if len(checkpoints) <= max_to_keep:
        return []

    # Calculate how many to delete
    num_to_delete = len(checkpoints) - max_to_keep
    checkpoints_to_delete = checkpoints[:num_to_delete]

    deleted_paths = []
    for step, ckpt_path in checkpoints_to_delete:
        try:
            logger.info("Deleting old checkpoint: %s (step %s)", ckpt_path, step)
            shutil.rmtree(ckpt_path)
            deleted_paths.append(ckpt_path)
        except Exception as e:
            logger.warning("Failed to delete checkpoint %s: %s", ckpt_path, e)

    if deleted_paths:
        logger.info("Deleted %d old checkpoint(s), keeping %d most recent.", len(deleted_paths), max_to_keep)

    return deleted_paths


def find_latest_ckpt_path(path):
    """
    Return the most recent valid checkpoint directory.

    Args:
        path (str): Base directory containing checkpoints.

    Returns:
        str or None: Full path to the latest valid checkpoint directory, or
        None if no valid checkpoint is found.
    """
    if path is None:
        return None

    candidates = get_checkpoint_directories(path)
    if not candidates:
        logger.info("No checkpoint directories found in %s", path)
        return None

    # Return the first valid checkpoint (sorted by step descending)
    for step, candidate in reversed(candidates):
        if is_valid_checkpoint(candidate):
            logger.info("Found valid checkpoint: %s (step: %s)", candidate, step)
            return candidate
        logger.warning("Checkpoint %s is malformed, skipping.", candidate)

    logger.warning("No valid checkpoint found in %s", path)
    return None
