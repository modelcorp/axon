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
import asyncio
import logging
import math
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import torch

from axon.engine.state.prefix_tree import PrefixTree
from axon.globals import DEBUG_MOE_REPLAY

logger = logging.getLogger(__name__)


def _has_moe_routermap(moe_routermap) -> bool:
    """Check if moe_routermap is valid (not None, not empty list, and non-empty tensor)."""
    if isinstance(moe_routermap, torch.Tensor):
        return moe_routermap.numel() > 0
    elif isinstance(moe_routermap, list):
        return len(moe_routermap) > 0
    return False


class StopPartialProgram(Exception):
    """
    Exception to indicate that partially completed program should be returned.
    """

    pass


class StopProgram(Exception):
    """
    Exception to indicate that Program has terminated.
    """

    pass


@dataclass
class MultiModalData:
    image: list[Any] | None = None
    video: list[Any] | None = None
    processor_kwargs: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class ModelStopReason:
    LENGTH = "length"
    STOP = "stop"


@dataclass
class ModelOutput:
    # Raw sampled text. At the sampler boundary this is derived from
    # token_strs so special tokens are preserved without a decode round-trip.
    response: str
    token_ids: list[int]
    logprobs: list[float]
    stop_reason: str
    # Kept as a Python list at the engine boundary; ProgramProcessor tensorizes it.
    moe_routermap: list[list[list[float]]]  # seq_len, n_experts, n_tokens
    # Per-token strings from vLLM (same length as token_ids).
    # Used to build exact response text without lossy decode() round-trip.
    token_strs: list[str] = field(default_factory=list)
    # Speculative decoding metrics (EAGLE / MTP / n-gram). None when disabled.
    spec_decode_metrics: object | None = None

    @classmethod
    def from_token_strs(
        cls,
        *,
        token_ids: list[int],
        token_strs: list[str],
        logprobs: list[float],
        stop_reason: str,
        moe_routermap,
        spec_decode_metrics: object | None = None,
    ) -> "ModelOutput":
        """Build raw response text from vLLM's exact per-token strings."""
        return cls(
            response="".join(token_strs),
            token_ids=token_ids,
            logprobs=logprobs,
            stop_reason=stop_reason,
            moe_routermap=moe_routermap,
            token_strs=token_strs,
            spec_decode_metrics=spec_decode_metrics,
        )


class TerminationReason(Enum):
    PROMPT_TRUNCATION = "PROMPT_TRUNCATION"
    MAX_STEPS = "MAX_STEPS"
    PROGRAM_TIMEOUT = "PROGRAM_TIMEOUT"
    TRUNCATION = "TRUNCATION"
    ENV_DONE = "ENV_DONE"
    ENV_TIMEOUT = "ENV_TIMEOUT"
    INTERNAL_ERROR = "INTERNAL_ERROR"
    BAD_PROGRAM = "BAD_PROGRAM"


@dataclass
class ProgramMetrics:
    steps: int = 0
    reward_time: float | None = None
    env_time: float = 0.0
    llm_time: float = 0.0
    total_time: float = 0.0
    # Speculative decoding aggregated metrics across all generation steps
    spec_draft_tokens: int = 0
    spec_accepted_tokens: int = 0
    spec_verify_count: int = 0


@dataclass
class Step:
    uid: str
    session_id: str
    chat_completions: list[dict[str, Any]] = field(default_factory=list)
    reward: float = 0  # Step reward currently is not used
    order_idx: int = -1  # denotes the order of which request was received
    # Inherited from ProgramState but also can override
    sample_params: dict[str, Any] = field(default_factory=dict)

    # Step Response variables. Also track response logprobs and moe_routermap.
    # training/prefix-replay token stream exactly
    text: str = ""
    tokens: list = field(default_factory=list)
    token_len: int = 0
    masks: list = field(default_factory=list)
    logprobs: list = field(default_factory=list)
    moe_routermap: list = field(default_factory=list)

    # Chat completions variables.
    multi_modal_data: MultiModalData | None = (
        None  # Inputs to serving engine during inference. Also used to construct modal_inputs for update.
    )

    # Partial Sampler Parameters to track intermediate state.
    partial_rollout_max_tokens: int = 0
    partial_tokens: list = field(default_factory=list)
    partial_text: str = ""
    partial_logprobs: list = field(default_factory=list)
    partial_moe_routermap: list = field(default_factory=list)
    partial_token_strs: list = field(default_factory=list)  # vLLM per-token strings (pending)
    response_token_strs: list = field(default_factory=list)  # vLLM per-token strings (committed)

    def check_empty_partial(self):
        """Check if partial state is empty"""
        return len(self.partial_tokens) == 0 and len(self.partial_text) == 0 and len(self.partial_logprobs) == 0

    def mask_out_response(self):
        """
        Mask out the entire response
        """
        assert self.check_empty_partial(), "Partial response should be empty when masking out existing responses"
        self.masks = [0] * len(self.tokens)
        self.logprobs = [0] * len(self.tokens)

    def set_response(self, text, tokens, masks, logprobs, moe_routermap):
        """Sets a step's response variables. This should be called after all partial state is committed."""
        assert len(tokens) == len(masks) == len(logprobs), (
            f"All inputs require same length: {len(tokens)}, {len(masks)}, {len(logprobs)}"
        )
        self.text += text
        self.tokens.extend(tokens)
        self.token_len += len(tokens)
        self.masks.extend(masks)
        self.logprobs.extend(logprobs)

        # Check for debugging if moe_routermap has -1 entries (only for tensors)
        if isinstance(moe_routermap, torch.Tensor) and moe_routermap.numel() > 0 and DEBUG_MOE_REPLAY:
            # Check each position (except last) for all -1 values
            for i in range(len(moe_routermap) - 1):
                if torch.all(torch.abs(moe_routermap[i] - (-1)) < 1e-5).item():
                    if DEBUG_MOE_REPLAY:
                        print(
                            f"[DEBUG_MOE_REPLAY, set_response] BUG: Found position {i} with all -1 routermap (excluding last)"
                        )

        # Handle moe_routermap - must be set AFTER extending tokens
        # Check if moe_routermap is valid (not None, not empty list, or non-empty tensor)
        has_moe_routermap = False
        if isinstance(moe_routermap, torch.Tensor):
            has_moe_routermap = moe_routermap.numel() > 0
        elif isinstance(moe_routermap, list):
            has_moe_routermap = len(moe_routermap) > 0

        if has_moe_routermap:
            expected_len = len(self.tokens)
            if len(moe_routermap) == expected_len:
                # Full routermap provided (normal case)
                self.moe_routermap = moe_routermap
            elif len(moe_routermap) < expected_len:
                # Partial routermap (prefix cache hit case) - need to keep cached portion
                # Save old moe_routermap before updating
                if isinstance(self.moe_routermap, torch.Tensor) and self.moe_routermap.numel() > 0:
                    old_moe_routermap = self.moe_routermap
                else:
                    old_moe_routermap = (
                        torch.empty(
                            (0,) + moe_routermap.shape[1:], dtype=moe_routermap.dtype, device=moe_routermap.device
                        )
                        if isinstance(moe_routermap, torch.Tensor)
                        else []
                    )

                cached_len = expected_len - len(moe_routermap)
                if DEBUG_MOE_REPLAY:
                    print(
                        f"[DEBUG_MOE_REPLAY, set_response] Step {self.uid}: partial routermap - expected {expected_len}, got {len(moe_routermap)}, need {cached_len} cached entries"
                    )
                    print(
                        f"[DEBUG_MOE_REPLAY, set_response]   token_len={len(self.tokens)}, old_routermap_len={len(old_moe_routermap)}"
                    )
                if len(old_moe_routermap) >= cached_len:
                    # Use torch.cat for tensors
                    if isinstance(moe_routermap, torch.Tensor):
                        self.moe_routermap = torch.cat([old_moe_routermap[:cached_len], moe_routermap], dim=0)
                    else:
                        self.moe_routermap = old_moe_routermap[:cached_len] + moe_routermap
                    if DEBUG_MOE_REPLAY:
                        print(
                            f"[DEBUG_MOE_REPLAY, set_response]   Using cached entries: {cached_len} cached + {len(moe_routermap)} new = {len(self.moe_routermap)}"
                        )
                else:
                    # Not enough cached entries, pad with -1
                    padding_len = cached_len - len(old_moe_routermap)
                    if DEBUG_MOE_REPLAY:
                        print(
                            f"[DEBUG_MOE_REPLAY, set_response]   WARNING: Not enough cached entries! Padding {padding_len} entries with -1"
                        )
                    # moe_routermap is a tensor with shape [seq_len, num_layers, num_experts]
                    padding_entry = torch.full(
                        (padding_len,) + moe_routermap.shape[1:],
                        -1,
                        dtype=moe_routermap.dtype,
                        device=moe_routermap.device,
                    )
                    self.moe_routermap = torch.cat([old_moe_routermap, padding_entry, moe_routermap], dim=0)
            else:
                # moe_routermap is longer than expected - truncate
                if DEBUG_MOE_REPLAY:
                    print(
                        f"[DEBUG_MOE_REPLAY, set_response] Step {self.uid}: routermap longer than expected - {len(moe_routermap)} > {expected_len}, truncating"
                    )
                self.moe_routermap = moe_routermap[:expected_len]
        else:
            self.moe_routermap = moe_routermap

    def reset_partial_state(self):
        self.partial_tokens = []
        self.partial_logprobs = []
        self.partial_moe_routermap = []
        self.partial_token_strs = []
        self.partial_text = ""

    def reset(self):
        self.token_len = 0
        self.tokens = []
        self.masks = []
        self.text = ""
        self.logprobs = []
        self.moe_routermap = []
        self.partial_rollout_max_tokens = 0
        self.reset_partial_state()
        self.multi_modal_data = None

    def is_empty(self):
        # Not containing unmasked content means empty
        return not any(m > 0 for m in self.masks)


@dataclass
class ProgramState:
    # Group ID for the program.
    group_id: str = ""
    # Unique ID for the program. By default, it is the session ID.
    uid: str = ""
    # ID for user session
    session_id: str = ""
    # Metrics collection
    metrics: ProgramMetrics = field(default_factory=ProgramMetrics)

    # Program state
    num_steps: int = 0
    global_steps: int = -1
    done: bool = False
    reward: float = 0
    termination_reason: TerminationReason | str | None = None

    # Override sampling parameters.
    sample_params: dict[str, Any] = field(default_factory=dict)

    # Variables tracked by engine, do not modify!
    session_lock: asyncio.Lock = field(default=None)
    last_llm_call_time: float = field(default_factory=time.monotonic)
    enable_partial_rollout: bool = False
    partial_rollout_futures: list[asyncio.Future] = field(default_factory=list)
    # Internal data strucutres to manage program state. Do not modify!
    steps: list[Step] = field(default_factory=list)
    prefix_tree: PrefixTree = field(default_factory=PrefixTree)  # A Trie of Tokens.

    # For final output for Programs.
    training_steps: list[Step] = field(default_factory=list)

    # Per-step rewards keyed by step order_idx. When non-empty, these override
    # the session-level ``reward`` for individual steps during score placement.
    step_rewards: dict[int, float] = field(default_factory=dict)

    # Any extra metadata, saved with each program
    metadata: dict[Any, Any] = field(default_factory=dict)

    def reset(self):
        """Reset all fields to default values except group_id, uid, agent, env, sample_params, and batch."""
        self.metrics = ProgramMetrics()
        self.num_steps = 0
        self.global_steps = -1
        self.termination_reason = None
        self.done = False
        self.reward = 0
        self.steps = []
        self.prefix_tree = PrefixTree()
        self.session_lock = None
        self.last_llm_call_time = time.monotonic()
        self.partial_rollout_futures = []
        self.step_rewards = {}
        self.metadata = {}

    # --- Engine time bookkeeping helpers ---
    def record_env_time(self):
        curr_time = time.monotonic()
        delta_time = curr_time - self.last_llm_call_time
        self.metrics.env_time += delta_time
        self.metrics.total_time += delta_time

    def record_llm_call_finish_time(self):
        self.last_llm_call_time = time.monotonic()

    def get_steps_from_prefix_tree(self) -> list[Step]:
        """
        Extract all possible paths from root to leaf nodes in the prefix tree.
        Each path represents a complete step with all tokens going into the response.

        All tokens and text are placed in the response fields.
        Prompt fields remain empty.

        Returns:
            list[Step]: List of Step objects, one for each path from root to leaf
        """
        steps = []

        # Use an explicit stack for iterative DFS
        # Each stack entry is a tuple: (node, path_token_ids, path_token_strs, path_logprobs, path_masks, path_moe_routermaps, max_step_idx, multi_modal_data)
        stack = [(self.prefix_tree.root, [], [], [], [], [], -1, None)]

        while stack:
            (
                node,
                path_token_ids,
                path_token_strs,
                path_logprobs,
                path_masks,
                path_moe_routermaps,
                max_step_idx,
                multi_modal_data,
            ) = stack.pop()

            # If leaf node, create a Step
            if node.is_leaf():
                # Skip empty paths (just root)
                if not path_token_ids:
                    continue

                # Create a new Step
                step = Step(
                    uid=self.uid,
                    session_id=self.session_id,
                )

                # Set the order_idx to the maximum step_idx seen along this path
                step.order_idx = max_step_idx
                # Put all tokens and text into step
                step.tokens = path_token_ids
                step.text = "".join(path_token_strs)
                step.token_len = len(path_token_ids)
                step.masks = path_masks
                step.logprobs = path_logprobs
                step.multi_modal_data = multi_modal_data
                # Handle moe_routermaps - check for None entries (indicates bug in prefix tree updates)
                if path_moe_routermaps:
                    none_count = sum(1 for e in path_moe_routermaps if e is None)
                    if none_count > 0:
                        # Replace None entries with -1 padding to preserve alignment with tokens
                        ref = next((e for e in path_moe_routermaps if e is not None), None)
                        if ref is not None and isinstance(ref, torch.Tensor):
                            filled = [e if e is not None else torch.full_like(ref, -1) for e in path_moe_routermaps]
                            step.moe_routermap = torch.stack(filled)
                        else:
                            step.moe_routermap = [e for e in path_moe_routermaps if e is not None]
                    else:
                        # Stack list of tensors [each shape: num_layers, num_experts] into [seq_len, num_layers, num_experts]
                        if path_moe_routermaps and isinstance(path_moe_routermaps[0], torch.Tensor):
                            step.moe_routermap = torch.stack(path_moe_routermaps)
                        else:
                            step.moe_routermap = path_moe_routermaps

                steps.append(step)
                continue

            # Iteratively explore children (push them onto stack)
            children = list(node.children.values())
            first_child = children[0]
            mask = max(first_child.masks)
            # Update max_step_idx with the current child's step_idx
            new_max_step_idx = max(max_step_idx, first_child.step_idx)
            # Always add moe_routermap entry to maintain alignment with token positions
            # Use None as placeholder if moe_routermap is empty (will be handled by padding logic later)
            first_child_moe = first_child.moe_routermap if _has_moe_routermap(first_child.moe_routermap) else None
            stack.append(
                (
                    first_child,
                    path_token_ids + [first_child.token_id],
                    path_token_strs + [first_child.token_str],
                    path_logprobs + [first_child.logprob],
                    path_masks + [mask],
                    path_moe_routermaps + [first_child_moe],
                    new_max_step_idx,
                    first_child.multi_modal_data if first_child.multi_modal_data else multi_modal_data,
                )
            )
            if len(children) > 1:
                for child in children[1:]:
                    # Pick one mask from the set (use max for consistency)
                    mask = max(child.masks)
                    # Update max_step_idx with the current child's step_idx
                    new_max_step_idx = max(max_step_idx, child.step_idx)
                    # Always add moe_routermap entry to maintain alignment with token positions
                    child_moe = child.moe_routermap if _has_moe_routermap(child.moe_routermap) else None
                    stack.append(
                        (
                            child,
                            path_token_ids + [child.token_id],
                            path_token_strs + [child.token_str],
                            path_logprobs + [child.logprob],
                            [0] * len(path_masks) + [mask],
                            path_moe_routermaps + [child_moe],
                            new_max_step_idx,
                            child.multi_modal_data if child.multi_modal_data else multi_modal_data,
                        )
                    )

        return steps

    def get_training_steps(self):
        """Return all non-empty steps for training.

        When ``step_rewards`` is set the raw steps are used directly —
        each generate() call is independent so prefix-tree reconstruction
        is unnecessary and would lose the 1:1 step_idx mapping.
        """
        if self.training_steps:
            return self.training_steps
        if self.step_rewards and self.steps:
            return [s for s in self.steps if not s.is_empty()]
        steps = self.get_steps_from_prefix_tree()
        training_steps = [s for s in steps if not s.is_empty()]
        return training_steps

    def is_trainable(self, strict=False):
        """Return if program can be used for training"""
        if not self.done:
            # incomplete program is not used for training
            return False

        if (
            self.reward == float("inf")
            or self.reward == float("-inf")
            or math.isnan(self.reward)
            or self.reward >= 1e9
            or self.reward <= -1e9
        ):
            # reward should be finite and reasonable
            return False

        if strict:
            if self.termination_reason not in [TerminationReason.ENV_DONE]:
                # program should be completed naturally
                return False

        return len(self.get_training_steps()) > 0


class ProgramManager:
    def __init__(self):
        # Maps group_id to a dictionary of program uid to ProgramState.
        self.programs: dict[str, dict[str, ProgramState]] = {}
        self._lock = threading.Lock()

    def add_programs(self, programs: list[ProgramState]):
        with self._lock:
            for program in programs:
                self._add_program(program)

    def _add_program(self, program: ProgramState):
        # Note: This method assumes the lock is already held by the caller
        group_id = program.group_id
        if group_id not in self.programs:
            self.programs[group_id] = {}
        if program.uid in self.programs[group_id]:
            logger.warning(
                "Duplicate uid '%s' in group '%s'; existing program will be overwritten", program.uid, group_id
            )
        self.programs[group_id][program.uid] = program

    def pop_programs(self, completed=False):
        """Generator that yields programs from the buffer.

        Args:
            completed: If True, yields only completed programs from groups where all programs are completed.
                      If False, yields incomplete programs.
        """
        result = []
        with self._lock:
            groups_to_remove = []

            for group_id, group_programs in self.programs.items():
                target_programs = []
                uids_to_remove = []

                if completed:
                    # For completed=True, only pop if ALL programs in the group are completed
                    all_completed = all(program.done for program in group_programs.values())
                    if all_completed:
                        for uid, program in group_programs.items():
                            target_programs.append(program)
                            uids_to_remove.append(uid)
                else:
                    # For completed=False, pop all incomplete programs
                    for uid, program in group_programs.items():
                        if not program.done:
                            target_programs.append(program)
                            uids_to_remove.append(uid)

                # Remove target programs from the group
                for uid in uids_to_remove:
                    del group_programs[uid]

                # If group is empty, mark it for removal
                if not group_programs:
                    groups_to_remove.append(group_id)

                result.extend(target_programs)

            # Remove empty groups
            for group_id in groups_to_remove:
                del self.programs[group_id]

        return result

    def get_num_programs(self):
        with self._lock:
            return sum(len(group_programs) for group_programs in self.programs.values())

    def get_num_completed_programs(self):
        with self._lock:
            num_completed = 0
            for group_programs in self.programs.values():
                for program in group_programs.values():
                    if program.done:
                        num_completed += 1
            return num_completed

    def get_num_incomplete_programs(self):
        with self._lock:
            num_incompleted = 0
            for group_programs in self.programs.values():
                for program in group_programs.values():
                    if not program.done:
                        num_incompleted += 1
            return num_incompleted

    def get_completed_programs_statistics(self):
        with self._lock:
            termination_reason_stats = {}
            for termination_reason in TerminationReason:
                termination_reason_stats[f"terminate_states/{termination_reason.value}"] = 0
            termination_reason_stats["terminate_states/empty"] = 0
            for group_programs in self.programs.values():
                for program in group_programs.values():
                    if program.done:
                        reason = program.termination_reason
                        reason_str = reason.value if isinstance(reason, TerminationReason) else str(reason)
                        key = f"terminate_states/{reason_str}"
                        if key not in termination_reason_stats:
                            termination_reason_stats[key] = 0
                        termination_reason_stats[key] += 1
                        if not program.training_steps:
                            # empty program
                            termination_reason_stats["terminate_states/empty"] += 1
            return termination_reason_stats
