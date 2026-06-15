# Copyright 2025 Model AI Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Program execution component for training.

Executes agent programs (BaseProgram, ReactProgram, etc.) and transforms
their results into tokenized training batches (DataProto).

Components:
    - :class:`ProgramProcessor` — transforms finished programs into DataProto
    - :class:`ProgramRunner` — creates and executes agent programs
    - :func:`create_program_components` — shared factory for all component objects
"""

from __future__ import annotations

import asyncio
import json
import logging
import multiprocessing as mp
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, NamedTuple

import numpy as np
import torch

from axon.engine.engine import Engine
from axon.engine.state.program_state import MultiModalData, ProgramMetrics, ProgramState, Step
from axon.globals import DEBUG_MOE_REPLAY
from axon.programs import PROGRAM_CLASS_MAPPING
from axon.protocol import DataProto
from axon.tinker import SamplingClient
from axon.utils.print_utils import colorful_print
from axon.utils.process import cleanup_subprocesses
from axon.utils.torch.ops import pad_sequence_to_length

logger = logging.getLogger(__name__)

_PROGRAM_CREATION_WORKERS = 64  # Thread pool size for parallel program instantiation

# ===========================================================================
# ProgramProcessor — transforms finished programs into tokenized batches
# ===========================================================================


@dataclass
class ProgramTransformConfig:
    """Configuration for program-to-training-batch transformation.

    Decoupled from Hydra config so that ProgramProcessor can be instantiated
    independently of the full training config system.
    """

    max_seq_length: int
    max_prompt_length: int
    moe_replay: bool = False
    save_programs_flag: bool = False
    filter_program_errors: bool = True
    train_batch_size: int = 1
    n_samples: int = 1

    @classmethod
    def from_config(cls, config) -> ProgramTransformConfig:
        """Create from a Hydra/OmegaConf training config object.

        Handles both attribute access (config.max_seq_length) and dict
        access (config["max_seq_length"]) patterns from Hydra configs.
        """
        # n_samples may live under config.decoding.n or be absent
        n_samples = 1
        decoding = getattr(config, "decoding", None)
        if decoding is not None:
            n_samples = getattr(decoding, "n", 1) if not isinstance(decoding, dict) else decoding.get("n", 1)

        return cls(
            max_seq_length=config.max_seq_length,
            max_prompt_length=config.max_prompt_length,
            moe_replay=getattr(config, "moe_replay", False),
            save_programs_flag=getattr(config, "save_programs_flag", False),
            filter_program_errors=getattr(config, "filter_program_errors", True),
            train_batch_size=getattr(config, "train_batch_size", 1),
            n_samples=n_samples,
        )


class ProgramProcessor:
    """Transforms finished agent programs into tokenized DataProto batches.

    Stateless except for tokenizer/processor references. All per-step state
    (experiment_dir, global_steps, uid_to_index) is passed as method arguments.

    Usage::

        config = ProgramTransformConfig(max_seq_length=4096, max_prompt_length=2048)
        pp = ProgramProcessor(config, tokenizer, processor)
        batch, metrics = pp.transform_programs(finished_programs)
    """

    def __init__(self, config: ProgramTransformConfig, tokenizer, processor=None):
        self.config = config
        self.tokenizer = tokenizer
        self.processor = processor

    def transform_programs(
        self,
        programs: list[ProgramState],
        experiment_dir: str | None = None,
        global_steps: int = 0,
        uid_to_index: dict[str, int] | None = None,
    ) -> tuple[DataProto, dict]:
        """Transform a list of finished programs into a tokenized training batch.

        Each program contains one or more Steps. Steps are flattened into rows
        of the output batch, with metadata linking each row back to its parent
        program (via program_uids, program_group_ids).

        Args:
            programs: Finished ProgramState objects with training_steps populated.
            experiment_dir: Directory for saving program data (if save_programs_flag set).
            global_steps: Current training step (for save file naming).
            uid_to_index: Optional UID->index mapping for curriculum sampling.

        Returns:
            Tuple of (DataProto training batch, aggregated metrics dict).
        """
        # Per-program results
        program_scores: list[float] = []
        metrics_list: list[ProgramMetrics] = []
        step_numbers: list[int] = []

        # Non-flattened per-program results (list of lists, one inner list per program)
        transformed_programs_list: list[dict[str, Any]] = []
        program_tokens_list: list[list[torch.Tensor]] = []
        program_logprobs_list: list[list[torch.Tensor]] = []
        program_masks_list: list[list[torch.Tensor]] = []
        program_moe_routermap_list: list[list[torch.Tensor]] = []

        # Flattened per-step metadata (one entry per step across all programs)
        steps_id: list[str] = []
        steps_program_uid: list[str] = []
        steps_program_group_id: list[str] = []
        steps_program_step_id: list[str] = []
        steps_total_step_num: list[int] = []
        steps_is_last_step: list[bool] = []
        steps_has_step_rewards: list[bool] = []
        steps_multi_modal_data: list[MultiModalData] = []
        steps_multi_modal_inputs: list[dict] = []

        # Process each program into tokenized step data
        for program in programs:
            transformed = self.transform_single_program(program)
            assert transformed["step_numbers"] > 0, "Every program used for training must have at least 1 step."
            program_scores.append(transformed["reward"])
            metrics_list.append(transformed["metrics"])
            step_numbers.append(transformed["step_numbers"])
            steps_has_step_rewards.extend([transformed["has_step_rewards"]] * transformed["step_numbers"])

            transformed_programs_list.append(transformed)
            program_tokens_list.append(transformed["tokens"])
            program_logprobs_list.append(transformed["logprobs"])
            program_masks_list.append(transformed["masks"])
            program_moe_routermap_list.append(transformed["moe_routermap"])

            steps_id.extend(transformed["steps_id"])
            steps_program_uid.extend(transformed["steps_program_uid"])
            steps_program_group_id.extend(transformed["steps_program_group_id"])
            steps_program_step_id.extend(transformed["steps_program_step_id"])
            steps_total_step_num.extend(transformed["steps_total_step_num"])
            steps_is_last_step.extend(transformed["steps_is_last_step"])
            steps_multi_modal_data.extend(transformed["multi_modal_data"])

        # Aggregate per-program metrics (timing, token counts, etc.)
        metrics = _aggregate_program_metrics(metrics_list=metrics_list)
        if self.config.save_programs_flag and experiment_dir:
            self.save_programs(transformed_programs_list, experiment_dir, global_steps)

        # Pad and stack all step tokens/masks/logprobs into 2D tensors.
        # All per-position tensors use max_seq_length as the single dimension.
        # No separate prompt padding — the initial prompt is embedded in the
        # sequence data (first tokens of step 0).
        response_batch = _pad_list_list_1d_tensor_to_2d_tensor_of_length(
            program_tokens_list, self.config.max_seq_length, -1, left_pad=False
        )
        response_mask = _pad_list_list_1d_tensor_to_2d_tensor_of_length(
            program_masks_list, self.config.max_seq_length, 0, left_pad=False
        )
        logprobs_batch = _pad_list_list_1d_tensor_to_2d_tensor_of_length(
            program_logprobs_list, self.config.max_seq_length, 0, left_pad=False
        )

        # MoE routermap: 3D per token (seq_len, num_moe_layers, num_experts)
        moe_shape = _infer_moe_shape(program_moe_routermap_list)
        moe_routermap_batch = _pad_moe_routermap_batch(
            program_moe_routermap_list,
            self.config.max_seq_length,
            -1,
            left_pad=False,
            moe_shape=moe_shape,
        )

        # input_ids is the full sequence — no prompt placeholder prefix.
        input_ids_batch = response_batch
        attention_mask = torch.where(input_ids_batch != -1, 1, 0)

        # Replace -1 padding with actual pad_token_id (handles edge case where
        # pad_token_id == eos_token_id, e.g. Deepseek-Qwen tokenizers)
        response_batch = torch.where(response_batch == -1, self.tokenizer.pad_token_id, response_batch)
        input_ids_batch = torch.where(input_ids_batch == -1, self.tokenizer.pad_token_id, input_ids_batch)

        # Multimodal: compute modal_inputs from processor
        if self.processor:
            for i in range(input_ids_batch.shape[0]):
                multi_modal_data = steps_multi_modal_data[i]
                if multi_modal_data:
                    input_ids = input_ids_batch[i]
                    images = multi_modal_data.image
                    current_text = self.tokenizer.decode(input_ids.squeeze(0), skip_special_tokens=True)
                    processor_kwargs = multi_modal_data.processor_kwargs
                    multi_modal_inputs = self.processor(
                        text=[current_text], images=images, return_tensors="pt", **processor_kwargs
                    )
                    multi_modal_inputs.pop("input_ids", None)
                    multi_modal_inputs.pop("attention_mask", None)
                    # mm_token_type_ids has shape (1, full_seq_len) which varies per sample;
                    # position_ids are already computed externally via _compute_position_ids
                    multi_modal_inputs.pop("mm_token_type_ids", None)
                    multi_modal_inputs = dict(multi_modal_inputs.convert_to_tensors("pt"))
                    steps_multi_modal_inputs.append(multi_modal_inputs)
                else:
                    steps_multi_modal_inputs.append({})

        # Position IDs (handles both text-only and multimodal mRoPE)
        position_ids = _compute_position_ids(input_ids_batch, attention_mask, steps_multi_modal_inputs, self.processor)

        # Place rewards at last response token position for each step.
        # prompt_length=0 since there is no separate prompt region in the tensor.
        score_batch = _compute_score_placement(
            program_rewards=program_scores,
            step_numbers=step_numbers,
            attention_mask=attention_mask,
        )

        # Assemble tensor batch — responses equals input_ids (no separate prompt tensor).
        tensor_batch = {
            "input_ids": input_ids_batch,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "responses": response_batch,
            "token_level_scores": score_batch,
            "response_mask": response_mask,
            "sampler_log_probs": logprobs_batch,
        }
        if self.config.moe_replay:
            tensor_batch["moe_routermap"] = moe_routermap_batch

        # Assemble non-tensor batch (metadata per flattened step)
        non_tensor_batch = {
            "uid": np.array(steps_program_group_id, dtype=object),
            "step_ids": np.array(steps_id, dtype=object),
            "program_uids": np.array(steps_program_uid, dtype=object),
            "program_group_ids": np.array(steps_program_group_id, dtype=object),
            "program_step_ids": np.array(steps_program_step_id, dtype=object),
            "num_program_steps": np.array(steps_total_step_num, dtype=object),
            "is_last_step": np.array(steps_is_last_step, dtype=object),
            "has_step_rewards": np.array(steps_has_step_rewards, dtype=object),
            "is_padding": np.array([False for _ in range(len(steps_is_last_step))], dtype=object),
        }
        if self.processor and any(mmi for mmi in steps_multi_modal_inputs):
            non_tensor_batch["multi_modal_inputs"] = np.array(steps_multi_modal_inputs, dtype=object)
        # Restore data index from uid->index mapping (for curriculum sampling)
        if uid_to_index:
            indices = [uid_to_index.get(uid, -1) for uid in steps_program_group_id]
            non_tensor_batch["index"] = np.array(indices, dtype=np.int64)

        result = DataProto.from_dict(tensors=tensor_batch, non_tensors=non_tensor_batch)

        # Compute per-program metadata (num_program_tokens, num_program_steps tensor).
        # All steps of each program are present in the batch at this point.
        # response_mask is already in the tensor batch; prepare_program_metadata reads it.
        prepare_program_metadata(result)

        # Visualize a few sample steps for debugging
        last_step_indices = [i for i, is_last in enumerate(non_tensor_batch["is_last_step"]) if is_last]
        non_last_step_indices = [i for i, is_last in enumerate(non_tensor_batch["is_last_step"]) if not is_last]
        n_last = min(2, len(last_step_indices))
        n_non_last = min(2, len(non_last_step_indices))
        sampled_last = np.random.choice(last_step_indices, size=n_last, replace=False) if last_step_indices else []
        sampled_non_last = (
            np.random.choice(non_last_step_indices, size=n_non_last, replace=False) if non_last_step_indices else []
        )
        for idx in list(sampled_last) + list(sampled_non_last):
            self.visualize_program(result, sample_idx=idx, max_samples=1)

        return result, metrics

    def transform_single_program(self, program: ProgramState) -> dict[str, Any]:
        """Transform one finished program into tokenized step data.

        Each program has N training steps. This method tokenizes each step and
        collects per-step metadata (UIDs, ordering, multi-modal data).

        Args:
            program: A finished ProgramState with training_steps populated.

        Returns:
            Dict with keys: tokens, logprobs, masks, moe_routermap,
            multi_modal_data, reward, metrics, step_numbers, completions,
            steps_id, steps_program_uid, steps_program_group_id, steps_program_step_id,
            steps_total_step_num, steps_is_last_step, metadata.
        """
        steps: list[Step] = program.training_steps

        group_id = program.group_id
        program_uid = program.uid
        training_reward: float = program.reward
        metrics = program.metrics

        # Per-step rewards: if the program set step_rewards, build a list
        # matching the training steps (looked up by order_idx, falling back
        # to the session-level reward).
        if program.step_rewards:
            per_step_rewards = [program.step_rewards.get(s.order_idx, training_reward) for s in steps]
        else:
            per_step_rewards = None

        results = {
            "program_group_id": group_id,
            "program_uid": program_uid,
            "reward": per_step_rewards if per_step_rewards is not None else training_reward,
            "has_step_rewards": per_step_rewards is not None,
            "metrics": metrics,
            "step_numbers": len(steps),
            "metadata": program.metadata,
            "completions": [],
            "tokens": [],
            "logprobs": [],
            "masks": [],
            "moe_routermap": [],
            "multi_modal_data": [],
            "steps_id": [],
            "steps_program_uid": [],
            "steps_program_group_id": [],
            "steps_program_step_id": [],
            "steps_total_step_num": [],
            "steps_is_last_step": [],
        }
        max_order_idx = max([s.order_idx for s in steps]) if steps else 0
        for step in steps:
            transformed_step = _transform_agent_step(step, self.config.moe_replay)
            transformed_step = _truncate_trailing_masked_tokens(transformed_step)
            results["completions"].append(transformed_step["completions"])
            results["tokens"].append(transformed_step["tokens"])
            results["logprobs"].append(transformed_step["logprobs"])
            results["masks"].append(transformed_step["masks"])
            results["moe_routermap"].append(transformed_step["moe_routermap"])
            results["multi_modal_data"].append(transformed_step["multi_modal_data"])

            results["steps_id"].append(step.uid)
            results["steps_program_uid"].append(program_uid)
            results["steps_program_group_id"].append(group_id)
            results["steps_program_step_id"].append(f"{group_id}_step{step.order_idx}")
            results["steps_total_step_num"].append(len(steps))
            results["steps_is_last_step"].append(step.order_idx == max_order_idx)

        return results

    def create_dummy_batch(
        self,
        original_batch: DataProto,
        batch_size: int | None = None,
        n_samples: int | None = None,
        global_steps: int = 0,
    ) -> DataProto:
        """Create a dummy batch with maximum context length for warmup/testing.

        Produces a DataProto with the same structure as transform_programs()
        output, but filled with dummy tokens. Useful for memory pre-allocation
        or testing the training loop without running actual programs.

        Args:
            original_batch: The original batch from the data loader (UIDs reused).
            batch_size: Override batch size (defaults to config.train_batch_size).
            n_samples: Override n_samples (defaults to config.n_samples).
            global_steps: Current training step for meta_info.

        Returns:
            DataProto with dummy data matching transform_programs output structure.
        """
        if batch_size is None:
            batch_size = self.config.train_batch_size
        if n_samples is None:
            n_samples = self.config.n_samples
        total_size = batch_size * n_samples

        max_seq_length = self.config.max_seq_length
        dummy_token_id = 42

        dummy_input_ids = torch.full((total_size, max_seq_length), dummy_token_id, dtype=torch.long)
        dummy_attention_mask = torch.ones_like(dummy_input_ids)
        dummy_position_ids = torch.arange(max_seq_length).unsqueeze(0).expand(total_size, -1)
        dummy_scores = torch.zeros((total_size, max_seq_length), dtype=torch.float32)
        dummy_scores[:, -1] = 1.0
        dummy_response_mask = torch.ones((total_size, max_seq_length), dtype=torch.long)
        dummy_logprobs = torch.zeros((total_size, max_seq_length), dtype=torch.float32)

        tensor_batch = {
            "input_ids": dummy_input_ids,
            "attention_mask": dummy_attention_mask,
            "position_ids": dummy_position_ids,
            "responses": dummy_input_ids,
            "token_level_scores": dummy_scores,
            "response_mask": dummy_response_mask,
            "sampler_log_probs": dummy_logprobs,
        }

        num_groups = batch_size
        step_ids = []
        program_uids = []
        program_group_ids = []
        program_step_ids = []
        step_nums = []
        is_last_step = []
        has_step_rewards = []

        for group_idx in range(num_groups):
            group_id = (
                original_batch.non_tensor_batch["uid"][group_idx]
                if original_batch.non_tensor_batch is not None and "uid" in original_batch.non_tensor_batch
                else str(uuid.uuid4())
            )
            for sample_idx in range(n_samples):
                program_uid = str(uuid.uuid4())
                step_uid = str(uuid.uuid4())
                step_ids.append(step_uid)
                program_uids.append(program_uid)
                program_group_ids.append(group_id)
                program_step_ids.append(f"{group_id}_step0")
                step_nums.append(1)
                is_last_step.append(True)
                has_step_rewards.append(False)

        non_tensor_batch = {
            "uid": np.array(program_group_ids, dtype=object),
            "step_ids": np.array(step_ids, dtype=object),
            "program_uids": np.array(program_uids, dtype=object),
            "program_group_ids": np.array(program_group_ids, dtype=object),
            "program_step_ids": np.array(program_step_ids, dtype=object),
            "num_program_steps": np.array(step_nums, dtype=object),
            "is_last_step": np.array(is_last_step, dtype=object),
            "has_step_rewards": np.array(has_step_rewards, dtype=object),
            "is_padding": np.array([False] * total_size, dtype=object),
        }

        if original_batch.non_tensor_batch is not None:
            for key, value in original_batch.non_tensor_batch.items():
                if key not in non_tensor_batch and key != "uid":
                    if isinstance(value, np.ndarray) and len(value) == num_groups:
                        expanded_value = np.repeat(value, n_samples)
                        non_tensor_batch[key] = expanded_value

        meta_info = {}
        if hasattr(original_batch, "meta_info") and original_batch.meta_info is not None:
            meta_info.update(original_batch.meta_info)
        meta_info.update(
            {
                "sample_params": {"dummy_batch": True},
                "global_steps": global_steps,
            }
        )

        return DataProto.from_dict(tensors=tensor_batch, non_tensors=non_tensor_batch, meta_info=meta_info)

    def visualize_program(self, tensor_batch, sample_idx=0, max_samples=1, mask_key="response_mask"):
        """Visualize a program step by detokenizing and highlighting masked/rewarded tokens.

        Tokens with mask=0 are shown in red (not trained on), tokens with
        non-zero rewards are highlighted in green, and normal trained tokens
        are shown in blue.

        Args:
            tensor_batch: DataProto containing program data.
            sample_idx: Starting index of samples to visualize.
            max_samples: Maximum number of samples to visualize.
            mask_key: Key for the mask tensor in the batch.
        """
        input_ids = tensor_batch.batch["input_ids"]
        response_mask = tensor_batch.batch[mask_key]
        token_level_scores = tensor_batch.batch["token_level_scores"]

        batch_size = input_ids.shape[0]
        end_idx = min(sample_idx + max_samples, batch_size)

        def debug_visible(s: str) -> str:
            return s.replace("\n", "\\n\n")

        for i in range(sample_idx, end_idx):
            colorful_print(f"\n===== Sample {i} =====", fg="cyan", bold=True)

            seq_tokens = input_ids[i]
            seq_mask = response_mask[i]

            last_non_pad_idx = len(seq_tokens) - 1
            while last_non_pad_idx >= 0 and seq_tokens[last_non_pad_idx] == self.tokenizer.pad_token_id:
                last_non_pad_idx -= 1

            if last_non_pad_idx < len(seq_tokens) - 1:
                valid_response_tokens = seq_tokens[: last_non_pad_idx + 2]
                valid_response_mask = seq_mask[: last_non_pad_idx + 2]
            else:
                valid_response_tokens = seq_tokens
                valid_response_mask = seq_mask

            colorful_print("Step with masking:", fg="yellow", bold=True)

            for j, (token, mask) in enumerate(zip(valid_response_tokens, valid_response_mask, strict=False)):
                token_text = self.tokenizer.decode(token, skip_special_tokens=False, clean_up_tokenization_spaces=False)
                has_reward = token_level_scores[i, j] != 0

                if mask == 0:
                    colorful_print(debug_visible(token_text), fg="red", end="")
                elif has_reward:
                    colorful_print(debug_visible(token_text), bg="green", end="")
                    reward_info = ""
                    if has_reward:
                        reward_info += f" R:{token_level_scores[i, j].item():.2f}"
                    colorful_print(reward_info, fg="magenta", end="")
                else:
                    colorful_print(debug_visible(token_text), fg="blue", end="")

            colorful_print("")
            total_reward = token_level_scores[i].sum().item()
            colorful_print("Rewards:", fg="green", bold=True)
            print(f" Program Reward={total_reward:.2f}")

    def save_programs(
        self,
        programs: list[dict[str, Any]],
        experiment_dir: str,
        global_steps: int,
        validation: bool = False,
    ):
        """Save transformed program data to a JSONL file.

        Args:
            programs: List of transformed program dicts (from transform_single_program).
            experiment_dir: Base experiment directory.
            global_steps: Current training step (used for file naming).
            validation: If True, append '_validation' to filename.
        """
        save_dir = os.path.join(experiment_dir, "generations")
        os.makedirs(save_dir, exist_ok=True)
        file_name = f"{global_steps}.jsonl"
        if validation:
            file_name = f"{global_steps}_validation.jsonl"
        with open(os.path.join(save_dir, file_name), "w") as f:
            for program in programs:
                entry = {
                    "reward": program["reward"],
                    "completions": program["completions"],
                    "group_id": program["program_group_id"],
                    "uid": program["program_uid"],
                    "metadata": program["metadata"],
                }
                f.write(json.dumps(entry) + "\n")


# ===========================================================================
# ProgramRunner — creates and executes agent programs
# ===========================================================================


@dataclass
class ProgramConfig:
    """Configuration for program creation and execution.

    Attributes:
        program_name: Name of the program class to instantiate (e.g. "react").
        program_kwargs: Extra kwargs passed to the program constructor.
        program_timeout: Default timeout for programs (seconds).
        filter_program_errors: If True, filter out programs that errored.
        endpoint_enable: Whether to use HTTP endpoint mode for programs.
        endpoint_host: Host for the program endpoint.
        endpoint_port: Port for the program endpoint.
        max_concurrency: Max number of programs to run concurrently.
        max_subprocess_concurrency: Max concurrent subprocesses.
        terminate_on_error: Whether to terminate training on program error.
        partial_rollout_enable: Whether partial rollout is enabled.
    """

    program_name: str = "react"
    program_kwargs: dict = field(default_factory=dict)
    program_timeout: float | None = None
    filter_program_errors: bool = True
    endpoint_enable: bool = False
    endpoint_host: str = "localhost"
    endpoint_port: int = 8000
    max_concurrency: int = 64
    max_subprocess_concurrency: int = 64
    terminate_on_error: bool = False
    partial_rollout_enable: bool = False

    @classmethod
    def from_config(cls, config) -> ProgramConfig:
        """Create ProgramConfig from a Hydra/OmegaConf config object.

        Args:
            config: The full training config. Expected structure:
                config.program.name, config.program.*, config.program_timeout,
                config.filter_program_errors, config.endpoint.*,
                config.max_concurrency, config.partial_rollout.enable
        """
        program_config = config.get("program", {})
        program_name = program_config.get("name", "react")
        program_kwargs = {k: v for k, v in program_config.items() if k != "name"}

        endpoint_config = config.get("engine_endpoint", {})

        return cls(
            program_name=program_name,
            program_kwargs=program_kwargs,
            program_timeout=getattr(config, "program_timeout", None),
            filter_program_errors=getattr(config, "filter_program_errors", True),
            endpoint_enable=endpoint_config.get("enable", False)
            if isinstance(endpoint_config, dict)
            else getattr(endpoint_config, "enable", False),
            endpoint_host=endpoint_config.get("host", "localhost")
            if isinstance(endpoint_config, dict)
            else getattr(endpoint_config, "host", "localhost"),
            endpoint_port=endpoint_config.get("port", 8000)
            if isinstance(endpoint_config, dict)
            else getattr(endpoint_config, "port", 8000),
            max_concurrency=getattr(config, "max_concurrency", 64),
            max_subprocess_concurrency=getattr(config, "max_subprocess_concurrency", 64),
            terminate_on_error=getattr(config, "terminate_on_error", False),
            partial_rollout_enable=getattr(config.get("partial_rollout", {}), "enable", False)
            if hasattr(config, "get")
            else False,
        )


class ProgramRunner:
    """Creates and executes agent programs.

    Handles program instantiation from batches, async execution (both
    in-process and subprocess modes), and collection of finished programs.

    Args:
        config: ProgramConfig with program and endpoint settings.
        engine: Engine instance for running programs.
        sampling_client: Optional SamplingClient for hybrid engine wake/sleep.
        hybrid_engine: Whether using hybrid engine mode (requires sampling_client).
    """

    def __init__(
        self,
        config: ProgramConfig,
        engine,
        sampling_client=None,
        hybrid_engine: bool = False,
    ):
        self.config = config
        self.engine = engine
        self.sampling_client = sampling_client
        self.hybrid_engine = hybrid_engine

        # Concurrency controls (0 means unlimited, matching old trainer behavior)
        concurrency_limit = config.max_concurrency if config.max_concurrency > 0 else 10**9
        self._run_sem = asyncio.Semaphore(concurrency_limit)
        self._process_sem = mp.Semaphore(config.max_subprocess_concurrency)
        self._tasks_loop = None

        # State
        self.queued_programs: list = []
        self.subprocess_handles: list[mp.Process] = []

    def set_tasks_loop(self, loop: asyncio.AbstractEventLoop):
        """Set the event loop used for scheduling async tasks."""
        self._tasks_loop = loop

    def shutdown(self):
        """Stop the background event loop and clean up resources."""
        if self._tasks_loop is not None and self._tasks_loop.is_running():
            self._tasks_loop.call_soon_threadsafe(self._tasks_loop.stop)
            self._tasks_loop = None

    def create_programs(
        self,
        batch: DataProto,
        global_steps: int,
    ) -> tuple[list, dict[str, int]]:
        """Create agent programs from a batch of environment args.

        Programs are created in parallel using a thread pool and queued
        for later execution via run_and_collect().

        Args:
            batch: DataProto with non_tensor_batch containing "env_args"
                and "uid" arrays, and optionally "index" for curriculum.
            global_steps: Current training step.

        Returns:
            Tuple of (list of programs, uid_to_index mapping for curriculum).
        """
        self.engine.set_global_steps(global_steps)

        program_cls = PROGRAM_CLASS_MAPPING[self.config.program_name]

        env_args = batch.non_tensor_batch["env_args"].tolist()
        uids = batch.non_tensor_batch["uid"].tolist()

        # Build uid -> index mapping for curriculum sampling
        data_indices = batch.non_tensor_batch.get("index", None)
        uid_to_index = {}
        if data_indices is not None:
            for i, uid in enumerate(uids):
                uid_to_index[uid] = int(data_indices[i])

        sample_params = batch.meta_info.get("sample_params", {})

        def _create_program(i):
            instance_kwargs = deepcopy(self.config.program_kwargs)
            if isinstance(env_args[i], str):
                env_args[i] = json.loads(env_args[i])
            config_env_args = instance_kwargs.get("env_args", {})
            if isinstance(config_env_args, dict):
                instance_kwargs["env_args"] = {**config_env_args, **env_args[i]}

            if "program_timeout" not in instance_kwargs and self.config.program_timeout is not None:
                instance_kwargs["program_timeout"] = self.config.program_timeout

            program = program_cls(**instance_kwargs)

            # List env_args: per-env defaults already in program._env_map,
            # set per-sample data directly.
            if not isinstance(config_env_args, dict):
                program.env_args = env_args[i]

            if not self.config.endpoint_enable:
                program.set_engine(self.engine)
            else:
                endpoint_url = f"http://{self.config.endpoint_host}:{self.config.endpoint_port}"
                program.set_endpoint_url(endpoint_url)
            program.set_group_id(uids[i])
            if sample_params:
                program.set_sample_params(sample_params)
            return i, program

        # Create programs in parallel while preserving order
        programs = [None] * len(env_args)
        with ThreadPoolExecutor(max_workers=_PROGRAM_CREATION_WORKERS) as executor:
            agent_futures = [executor.submit(_create_program, i) for i in range(len(env_args))]
            for future in as_completed(agent_futures):
                idx, program = future.result()
                programs[idx] = program

        self.queued_programs = programs
        return programs, uid_to_index

    async def run_and_collect(self, val: bool = False) -> tuple[list[ProgramState], dict]:
        """Run queued programs and collect finished programs.

        This method:
        1. Optionally wakes up the sampler engine (hybrid mode)
        2. Resumes paused partial rollout programs from previous iteration
        3. Launches queued programs (in-process or subprocesses)
        4. Waits until all programs are finished or paused
        5. Collects finished programs from the engine
        6. Optionally puts the sampler engine to sleep

        Args:
            val: Whether this is a validation run.

        Returns:
            Tuple of (list of finished ProgramState objects, engine metrics dict).
        """
        # Wake up sampler engine for hybrid mode
        if self.hybrid_engine and self.sampling_client is not None:
            await self.sampling_client.wake_up()

        if not val:
            # Resume programs paused in previous iteration (partial rollout)
            await self.engine.run_in_engine_loop_async(self.engine.resume_partial_rollout_programs())

        # Launch queued programs
        if self.queued_programs:
            # Initialize all sessions before launching
            await asyncio.gather(
                *(
                    p.init_session(
                        group_id=p.group_id,
                        sample_params=getattr(p, "sample_params", None),
                    )
                    for p in self.queued_programs
                ),
            )
            if self.config.endpoint_enable:
                await self._run_programs_in_subprocesses(self.queued_programs)
            else:
                await self._run_programs_in_process(self.queued_programs)

            self.queued_programs = []

        # Wait until all programs are finished or paused on partial sampler futures
        last_print_time = time.time()
        while not await self.engine.run_in_engine_loop_async(self.engine.all_programs_safe_to_collect()):
            await asyncio.sleep(10)
            current_time = time.time()
            if current_time - last_print_time >= 30:
                session_info = await self.engine.run_in_engine_loop_async(self.engine.get_session_diagnostics())
                colorful_print(f"Waiting for programs... {session_info}", "yellow")
                last_print_time = current_time

        # Collect results
        engine_metrics = await self.engine.run_in_engine_loop_async(self.engine.get_engine_metrics())
        finished_programs = await self.engine.run_in_engine_loop_async(self.engine.get_finished_programs())
        engine_metrics["engine/returned_programs"] = len(finished_programs)

        # Clean up zombie processes when partial rollout is disabled
        if self.config.endpoint_enable and not self.config.partial_rollout_enable:
            cleanup_subprocesses(self.subprocess_handles)
        elif self.config.endpoint_enable and self.config.partial_rollout_enable:
            # Prune finished processes to prevent unbounded list growth
            self.subprocess_handles = [p for p in self.subprocess_handles if p.is_alive()]

        # Put sampler engine to sleep for hybrid mode
        if self.hybrid_engine and self.sampling_client is not None:
            await self.sampling_client.sleep()

        return finished_programs, engine_metrics

    async def _run_programs_in_process(self, programs):
        """Run programs in the current process using asyncio tasks.

        Programs are submitted as background tasks and not awaited here.
        The caller waits for completion via engine.all_programs_safe_to_collect().

        Args:
            programs: List of program instances to run.
        """
        terminate_on_error = self.config.terminate_on_error

        async def run_program_safely(program):
            try:
                async with self._run_sem:
                    await program.run_program()
            except Exception as e:
                import traceback

                colorful_print(f"Program execution failed: {e}", "red")
                traceback.print_exc()
                if terminate_on_error:
                    raise e

        for program in programs:
            self._tasks_loop.call_soon_threadsafe(asyncio.create_task, run_program_safely(program))

    async def _run_programs_in_subprocesses(self, programs):
        """Launch programs in separate subprocesses.

        Used when endpoint mode is enabled, allowing programs to run
        independently and communicate with the engine via HTTP API.

        This method launches subprocesses and returns immediately. The
        caller waits for completion via engine.all_programs_safe_to_collect().

        Args:
            programs: List of program instances to run.
        """
        terminate_on_error = self.config.terminate_on_error
        process_sem = self._process_sem

        def _run_program_in_subprocess(program, sem):
            try:
                with sem:
                    asyncio.run(program.run_program())
            except Exception as e:
                import traceback

                colorful_print(f"Program execution failed in subprocess: {e}", "red")
                traceback.print_exc()
                if terminate_on_error:
                    raise e

        colorful_print(f"Launching {len(programs)} programs in subprocesses...", "cyan")

        for program in programs:
            process = mp.Process(target=_run_program_in_subprocess, args=(program, process_sem))
            process.start()
            self.subprocess_handles.append(process)

        colorful_print(f"Launched {len(programs)} subprocesses", "green")


# ===========================================================================
# Program metadata
# ===========================================================================


def prepare_program_metadata(batch: DataProto) -> None:
    """Compute per-program metadata fields needed by the loss function.

    Sets the following fields on ``batch.batch`` (tensor batch):

    * ``num_program_tokens`` — total valid tokens across all steps of each
      program. Used by ``token_reduce="mean-program"`` in ``agg_loss``.
    * ``num_program_steps`` — number of steps in the program (converted from
      ``non_tensor_batch`` to tensor). Used by ``batch_reduce="program-mean"``.

    Must be called AFTER all steps of each program are present in the batch
    (e.g. after stepwise advantage broadcast re-merges last + other steps).
    """
    program_uids = batch.non_tensor_batch["program_uids"]
    mask = batch.batch["response_mask"]
    device = mask.device

    # Total valid tokens per program (sum across all steps sharing a program_uid)
    uid_to_token_count: dict[str, torch.Tensor] = {}
    for i, uid in enumerate(program_uids):
        row_tokens = mask[i].sum()
        if uid in uid_to_token_count:
            uid_to_token_count[uid] = uid_to_token_count[uid] + row_tokens
        else:
            uid_to_token_count[uid] = row_tokens

    batch.batch["num_program_tokens"] = torch.tensor(
        [uid_to_token_count[uid] for uid in program_uids],
        device=device,
    )

    # Convert num_program_steps from non_tensor_batch (numpy) to tensor batch
    if "num_program_steps" in batch.non_tensor_batch:
        step_nums = batch.non_tensor_batch["num_program_steps"]
        batch.batch["num_program_steps"] = torch.tensor(
            [int(s) for s in step_nums],
            device=device,
            dtype=torch.float32,
        )


# ===========================================================================
# Component factory
# ===========================================================================


class ProgramComponents(NamedTuple):
    """Components returned by :func:`create_program_components`.

    Attributes:
        sampling_client: HTTP client for communicating with sampler servers.
        engine: Manages program lifecycle and sessions.
        program_processor: Transforms finished programs into tokenized DataProto.
        program_runner: Creates and executes agent programs.
        tasks_loop: asyncio event loop for in-process program execution,
            or None if endpoint mode is enabled.
    """

    sampling_client: SamplingClient
    engine: Engine
    program_processor: ProgramProcessor
    program_runner: ProgramRunner
    tasks_loop: asyncio.AbstractEventLoop | None


def create_program_components(
    config,
    tokenizer,
    processor,
    server_addresses: list[str],
    sampler_servers: list,
    hybrid_engine: bool = False,
    thread_name_prefix: str = "Pipeline",
) -> ProgramComponents:
    """Create the standard components for program execution.

    This factory is used by both the controller-resident trainer (runs on
    the controller process) and the AsyncSamplerMixin (runs on a sampler
    worker). The only difference is that the sampler worker passes a
    ``sampler_engine`` for direct engine access.

    Args:
        config: Full training config (Hydra/OmegaConf).
        tokenizer: HuggingFace tokenizer instance.
        processor: HuggingFace processor (for multimodal), or None.
        server_addresses: List of sampler server HTTP addresses.
        sampler_servers: List of sampler server handles.
        hybrid_engine: Whether actor and sampler share GPUs. Passed to
            ProgramRunner for mode-specific behavior.
        thread_name_prefix: Prefix for the background event loop thread name.

    Returns:
        ProgramComponents named tuple with all initialized components.
    """

    sampling_client = SamplingClient(
        config=config,
        tokenizer=tokenizer,
        processor=processor,
        addresses=server_addresses,
        sampler_servers=sampler_servers,
    )

    engine_kwargs = config.get("engine_args", {})
    engine = Engine(
        sampling_client=sampling_client,
        config=config,
        tokenizer=tokenizer,
        processor=processor,
        model_path=config.model_path,
        max_steps=config.max_steps,
        max_seq_length=config.max_seq_length,
        max_prompt_length=config.max_prompt_length,
        max_tokens_per_step=getattr(config, "max_tokens_per_step", None),
        prompt_truncation=getattr(config, "prompt_truncation", None),
        program_timeout=config.program_timeout,
        overlong_filter=config.overlong_filter,
        **engine_kwargs,
    )

    transform_config = ProgramTransformConfig.from_config(config)
    program_processor = ProgramProcessor(transform_config, tokenizer, processor)

    program_config = ProgramConfig.from_config(config)
    program_runner = ProgramRunner(
        config=program_config,
        engine=engine,
        sampling_client=sampling_client,
        hybrid_engine=hybrid_engine,
    )

    # Create a background event loop for in-process program execution.
    # When endpoint mode is enabled, programs run via an external HTTP
    # service and no local event loop is needed.
    tasks_loop = None
    if not program_config.endpoint_enable:
        tasks_loop = asyncio.new_event_loop()
        loop_thread = threading.Thread(
            target=tasks_loop.run_forever,
            name=f"{thread_name_prefix}-ProgramsLoop",
            daemon=True,
        )
        loop_thread.start()
        program_runner.set_tasks_loop(tasks_loop)

    return ProgramComponents(
        sampling_client=sampling_client,
        engine=engine,
        program_processor=program_processor,
        program_runner=program_runner,
        tasks_loop=tasks_loop,
    )


# ===========================================================================
# Private utility functions (no class state needed)
# ===========================================================================


def _transform_agent_step(step: Step, moe_replay: bool = False) -> dict:
    """Transform one Step into tokenized tensors.

    A Step represents a single generation turn within a multi-step program.
    This function extracts the raw token IDs, log-probabilities, training
    masks, MoE router maps, and multimodal data into tensor form.

    Args:
        step: A Step object from program_state.
        moe_replay: Whether MoE router replay is enabled.

    Returns:
        Dict with tokens, logprobs, masks, moe_routermap, multi_modal_data, completions.
    """
    result = {
        "completions": step.text,
        "tokens": torch.tensor(step.tokens, dtype=torch.long),
        "logprobs": torch.tensor(step.logprobs, dtype=torch.float32),
        "masks": torch.tensor(step.masks, dtype=torch.long),
        "moe_routermap": step.moe_routermap.to(dtype=torch.int16)
        if isinstance(step.moe_routermap, torch.Tensor) and step.moe_routermap.numel() > 0
        else torch.empty(0, 0, 0, dtype=torch.int16),
        "multi_modal_data": step.multi_modal_data,
    }
    if moe_replay and DEBUG_MOE_REPLAY:
        assert len(step.moe_routermap) == len(step.tokens), (
            f"MOE routermap must be the same length as tokens: {len(step.moe_routermap)} vs {len(step.tokens)}"
        )

        expected_len = len(step.tokens)
        has_moe = (isinstance(step.moe_routermap, torch.Tensor) and step.moe_routermap.numel() > 0) or (
            isinstance(step.moe_routermap, list) and len(step.moe_routermap) > 0
        )
        actual_len = len(step.moe_routermap) if has_moe else 0
        if actual_len != expected_len:
            if DEBUG_MOE_REPLAY:
                print(
                    f"[DEBUG_MOE_REPLAY, _transform_agent_step] Step {step.uid}: routermap length mismatch: expected {expected_len}, got {actual_len}"
                )

        if has_moe and len(step.moe_routermap) > 1:
            neg1_positions = []
            for i, entry in enumerate(step.moe_routermap[:-1]):
                if isinstance(entry, list | np.ndarray):
                    if all(v == -1 for row in entry for v in (row if isinstance(row, list | np.ndarray) else [row])):
                        neg1_positions.append(i)
                elif hasattr(entry, "tolist"):
                    entry_list = entry.tolist()
                    if all(v == -1 for row in entry_list for v in (row if isinstance(row, list) else [row])):
                        neg1_positions.append(i)

            if neg1_positions and DEBUG_MOE_REPLAY:
                print(
                    f"[DEBUG_MOE_REPLAY, _transform_agent_step] Step {step.uid}: Found {len(neg1_positions)} positions with all -1 routermap (excluding last)"
                )
                print(f"[DEBUG_MOE_REPLAY, _transform_agent_step]   token_len={len(step.tokens)}")
                print(f"[DEBUG_MOE_REPLAY, _transform_agent_step]   First 10 bad positions: {neg1_positions[:10]}")
                if len(neg1_positions) > 10:
                    print(f"[DEBUG_MOE_REPLAY, _transform_agent_step]   Last 10 bad positions: {neg1_positions[-10:]}")

    return result


def _truncate_trailing_masked_tokens(step_data: dict) -> dict:
    """Truncate trailing tokens where mask=0.

    Ensures response_mask always ends with 1 for each step, removing wasteful
    trailing tokens that won't contribute to training loss.
    """
    masks = step_data["masks"]

    if len(masks) == 0:
        return step_data

    nonzero_indices = torch.nonzero(masks, as_tuple=True)[0]

    if len(nonzero_indices) == 0:
        return step_data

    last_mask1_idx = nonzero_indices[-1].item()
    truncate_len = last_mask1_idx + 1

    if truncate_len < len(masks):
        step_data["tokens"] = step_data["tokens"][:truncate_len]
        step_data["masks"] = step_data["masks"][:truncate_len]
        step_data["logprobs"] = step_data["logprobs"][:truncate_len]

        if step_data["moe_routermap"].numel() > 0:
            step_data["moe_routermap"] = step_data["moe_routermap"][:truncate_len]

    return step_data


def _aggregate_program_metrics(metrics_list: list[ProgramMetrics]) -> dict:
    """Aggregate a list of per-program ProgramMetrics into summary statistics.

    Computes mean/min/max for each numeric field across all programs.
    """
    metrics = {}
    if not metrics_list:
        return metrics

    field_names = metrics_list[0].__dataclass_fields__.keys()
    per_field_values = {name: [] for name in field_names}

    for program_metrics in metrics_list:
        for name in field_names:
            per_field_values[name].append(getattr(program_metrics, name))

    for k, v_list in per_field_values.items():
        v_list = [v for v in v_list if v is not None and v >= 0]
        if not v_list:
            continue
        v_arr = np.array(v_list)
        metrics.update(
            {
                f"program/{k}_mean": v_arr.mean(),
                f"program/{k}_min": v_arr.min(),
                f"program/{k}_max": v_arr.max(),
            }
        )

    # Compute derived speculative decoding rates from totals
    total_draft = sum(v for v in per_field_values.get("spec_draft_tokens", []) if v and v > 0)
    total_accepted = sum(v for v in per_field_values.get("spec_accepted_tokens", []) if v and v > 0)
    total_verify = sum(v for v in per_field_values.get("spec_verify_count", []) if v and v > 0)
    if total_draft > 0:
        metrics["program/spec_accept_rate"] = total_accepted / total_draft
    if total_verify > 0:
        metrics["program/spec_accept_length"] = (total_accepted + total_verify) / total_verify

    return metrics


def _compute_position_ids(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    multi_modal_inputs: list[dict],
    processor=None,
) -> torch.Tensor:
    """Compute position_ids for both text-only and multimodal models.

    For text-only models, returns cumulative attention mask positions.
    For Qwen2/3-VL multimodal models, computes mRoPE position IDs
    that account for image/video grid dimensions.

    Returns:
        text-only: (B, L)
        Qwen2/3-VL mRoPE: (B, 4, L)
    """
    if processor is None:
        return (torch.cumsum(attention_mask, dim=1) - 1) * attention_mask

    position_ids_list = []

    # Only Qwen2/3-VL needs special mRoPE position IDs.
    # Other processors (e.g. Gemma4Processor) use standard RoPE.
    if (
        not hasattr(processor, "image_processor")
        or "Qwen2VLImageProcessor" not in processor.image_processor.__class__.__name__
    ):
        return (torch.cumsum(attention_mask, dim=1) - 1) * attention_mask

    if "Qwen2VLImageProcessor" in processor.image_processor.__class__.__name__:
        if "Qwen3VLProcessor" in processor.__class__.__name__:
            from axon.models.transformers.qwen3_vl import get_rope_index
        else:
            from axon.models.transformers.qwen2_vl import get_rope_index

        for i in range(input_ids.shape[0]):
            model_inputs = multi_modal_inputs[i] if i < len(multi_modal_inputs) else {}
            vision_position_ids = get_rope_index(
                processor,
                input_ids=input_ids[i],
                image_grid_thw=model_inputs.get("image_grid_thw"),
                video_grid_thw=model_inputs.get("video_grid_thw"),
                second_per_grid_ts=model_inputs.get("second_per_grid_ts"),
                attention_mask=attention_mask[i],
            )
            valid_mask = attention_mask[i].bool()
            text_position_ids = torch.ones((1, len(input_ids[i])), dtype=torch.long)
            text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item())
            position_ids_list.append(torch.cat((text_position_ids, vision_position_ids), dim=0))

    return torch.stack(position_ids_list, dim=0)


def _compute_score_placement(
    program_rewards: list[float] | list[list[float]],
    step_numbers: list[int],
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Place rewards at the last response token position for each step.

    For single-reward programs, the reward is duplicated across all steps.
    For per-step reward lists, each step gets its own reward value.

    Args:
        program_rewards: One reward per program, or one reward per step.
        step_numbers: Number of steps per program.
        attention_mask: Padded attention mask for the full sequence.

    Returns:
        Score tensor with rewards placed at last valid token positions.
    """
    assert len(program_rewards) == len(step_numbers), (
        f"Number of rewards must match number of programs: {len(program_rewards)}, {len(step_numbers)}"
    )
    expanded_rewards = []
    for i, reward in enumerate(program_rewards):
        if isinstance(reward, list):
            assert len(reward) == step_numbers[i], (
                f"Mismatch in step rewards for program {i}: {len(reward)} vs {step_numbers[i]} steps"
            )
            expanded_rewards.extend(reward)
        else:
            expanded_rewards.extend([reward] * step_numbers[i])

    response_attention_mask = attention_mask[:, :]
    valid_response_length = response_attention_mask.sum(dim=-1)
    score_batch = torch.zeros_like(response_attention_mask, dtype=torch.float32)
    for i, step_score in enumerate(expanded_rewards):
        last_valid_idx = valid_response_length[i] - 1
        if last_valid_idx >= 0 and last_valid_idx < score_batch.shape[1]:
            score_batch[i, last_valid_idx] = step_score
    return score_batch


def _pad_list_list_1d_tensor_to_2d_tensor_of_length(
    data: list[list[torch.Tensor]], target_length: int, pad_value: int = -1, left_pad: bool = False
) -> torch.Tensor:
    """Flatten a list-of-lists of 1D tensors into a padded 2D tensor.

    Input is nested: outer list = programs, inner list = steps per program.
    Steps are flattened into rows, then padded/truncated to target_length.

    Args:
        data: Nested list of 1D tensors [programs][steps].
        target_length: Target sequence length to pad/truncate to.
        pad_value: Value to use for padding.
        left_pad: If True, pad on the left side.

    Returns:
        2D tensor of shape (total_steps, target_length).
    """

    flat_data = [t for sublist in data for t in sublist]
    if left_pad:
        batch = torch.nn.utils.rnn.pad_sequence(
            [torch.flip(i, dims=[0]) for i in flat_data],
            batch_first=True,
            padding_value=pad_value,
        ).flip(dims=[1])
    else:
        batch = torch.nn.utils.rnn.pad_sequence(
            flat_data,
            batch_first=True,
            padding_value=pad_value,
        )
    batch = pad_sequence_to_length(batch, target_length, pad_value, left_pad=left_pad)
    return batch


def _pad_moe_routermap_batch(
    data: list[list[torch.Tensor]],
    target_length: int,
    pad_value: int = -1,
    left_pad: bool = False,
    moe_shape: tuple | None = None,
) -> torch.Tensor:
    """Pad MoE routermap tensors to target length.

    MoE routermaps are 3D per token: (seq_len, num_moe_layers, num_experts).
    This function pads them into a uniform 4D batch tensor.

    Returns:
        4D tensor of shape (batch_size, target_length, num_moe_layers, num_experts).
    """
    flat_data = [t for sublist in data for t in sublist]

    if len(flat_data) == 0:
        if moe_shape is None:
            moe_shape = (0, 0)
        return torch.empty(0, target_length, moe_shape[0], moe_shape[1], dtype=torch.int16)

    if moe_shape is None:
        for t in flat_data:
            if t.numel() > 0 and t.dim() == 3:
                moe_shape = (t.shape[1], t.shape[2])
                break

    if moe_shape is None:
        moe_shape = (0, 0)

    num_moe_layers, num_experts = moe_shape

    processed_data = []
    for t in flat_data:
        if t.numel() == 0:
            processed_data.append(torch.empty(0, num_moe_layers, num_experts, dtype=t.dtype, device=t.device))
        elif t.dim() == 2:
            processed_data.append(t.unsqueeze(-1).expand(-1, -1, num_experts))
        else:
            processed_data.append(t)

    batch_size = len(processed_data)
    result = torch.full(
        (batch_size, target_length, num_moe_layers, num_experts),
        pad_value,
        dtype=processed_data[0].dtype if processed_data else torch.int16,
    )

    for i, t in enumerate(processed_data):
        seq_len = t.shape[0]
        if seq_len > 0:
            if seq_len > target_length:
                if left_pad:
                    t = t[-target_length:]
                else:
                    t = t[:target_length]
                seq_len = target_length

            if left_pad:
                result[i, -seq_len:, :, :] = t
            else:
                result[i, :seq_len, :, :] = t

    return result


def _infer_moe_shape(data: list[list[torch.Tensor]]) -> tuple[int, int]:
    """Infer MoE shape (num_moe_layers, num_experts) from routermap tensors."""
    for sublist in data:
        for t in sublist:
            if t.numel() > 0 and t.dim() == 3:
                return (t.shape[1], t.shape[2])
    return (0, 0)
