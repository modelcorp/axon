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
import os
import threading
import time
import uuid
from copy import deepcopy
from typing import Any

import torch
import uvicorn
from fastapi import FastAPI

from axon.engine.chat_template.parser import MM_PART_TYPES, ChatTemplateParser, TokenizeContext, TokenizeOutput
from axon.engine.server.engine_router import build_engine_router
from axon.engine.server.oai_router import build_openai_router
from axon.engine.state.program_state import (
    ModelOutput,
    ModelStopReason,
    MultiModalData,
    ProgramManager,
    ProgramState,
    Step,
    StopPartialProgram,
    StopProgram,
    TerminationReason,
)
from axon.globals import DEBUG_MOE_REPLAY
from axon.utils.networking_utils import ensure_port_available
from axon.utils.print_utils import colorful_print
from axon.utils.tokenizer_pool import TokenizerPool

logger = logging.getLogger(__name__)


def _combine_mm_data(recovered_mm: list[tuple], new_mm_data: MultiModalData | None) -> MultiModalData | None:
    """Combine tree-recovered MM data with suffix processor output."""
    if not recovered_mm and not new_mm_data:
        return None
    mm = MultiModalData()
    old_imgs = []
    old_vids = []
    for mod, data in recovered_mm:
        if mod == "image":
            old_imgs.append(data)
        elif mod == "video":
            old_vids.append(data)
    new_imgs = new_mm_data.image if new_mm_data and new_mm_data.image else []
    new_vids = new_mm_data.video if new_mm_data and new_mm_data.video else []
    mm.image = old_imgs + new_imgs if (old_imgs or new_imgs) else None
    mm.video = old_vids + new_vids if (old_vids or new_vids) else None
    if new_mm_data and new_mm_data.processor_kwargs:
        mm.processor_kwargs = new_mm_data.processor_kwargs
    return mm


def _detect_mm_expansions(token_ids: list[int], mm_regions: list[tuple], tokenizer) -> list[tuple] | None:
    """Find MM expansion regions in a token sequence for tree annotation.

    Detects consecutive runs of pad tokens (image_pad / video_pad) and maps
    each run to the content hash from the corresponding ``mm_region``.

    Returns:
        List of ``(token_position, count, content_hash)`` or ``None``.
    """
    if not mm_regions:
        return None

    # Get pad token IDs for each modality
    pad_ids = set()
    for pid_name in ("image_pad", "video_pad"):
        token_name = f"<|{pid_name}|>"
        tid = tokenizer.convert_tokens_to_ids(token_name)
        if tid is not None and tid != tokenizer.unk_token_id:
            pad_ids.add(tid)

    if not pad_ids:
        return None

    expansions = []
    region_iter = iter(mm_regions)
    i = 0
    while i < len(token_ids):
        if token_ids[i] in pad_ids:
            # Count consecutive pad tokens
            count = 1
            while i + count < len(token_ids) and token_ids[i + count] in pad_ids:
                count += 1
            # Match with next MM region's content hash
            region = next(region_iter, None)
            chash = region[3] if region else ""
            expansions.append((i, count, chash))
            i += count
        else:
            i += 1

    return expansions or None


def _as_token_id_set(token_id_or_ids) -> set[int]:
    if token_id_or_ids is None:
        return set()
    if isinstance(token_id_or_ids, int):
        return {token_id_or_ids}
    if isinstance(token_id_or_ids, list | tuple | set):
        return {int(token_id) for token_id in token_id_or_ids if token_id is not None}
    return set()


def _first_or_none(values):
    if values is None:
        return None
    if isinstance(values, list | tuple):
        if not values:
            return None
        return values[0]
    return values


class Engine:
    """Coordinates multi-turn agent rollouts against the sampler.

    The engine sits between the driver (which holds programs) and the sampler
    (which generates tokens). For each program, it manages:

    * **Per-session state** (:class:`~axon.engine.state.program_state.ProgramState`)
      — conversation history, response masks, sampler logprobs, partial-rollout
      checkpoints.
    * **Tokenisation** via a process-pool tokenizer to keep the event loop free
      under high QPS.
    * **Partial-rollout suspend / resume** so a weight update mid-generation
      doesn't drop the in-flight rollout — the suspended state is restored
      against the new weights.
    * **Optional FastAPI HTTP surface** when ``engine_endpoint.enable: true``,
      exposing both an Axon-native API and an OpenAI-compatible
      chat-completions endpoint.

    The engine is instantiated by the driver and shared by all programs in a
    training step.
    """

    def __init__(
        self,
        tokenizer=None,
        processor=None,
        sampling_client=None,
        chat_parser=None,
        program_timeout=10800,
        gamma=0.2,
        max_steps=5,
        max_seq_length=8192,
        max_prompt_length=1024,
        max_tokens_per_step=None,
        prompt_truncation=None,  # "left", "right", or None (terminate on truncation)
        config=None,
        overlong_filter=False,  # Filter for overlong programs (i.e. TRUNCATION, MAX_STEPS, TIMEOUT)
        **kwargs,
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.processor = processor
        self.sampling_client = sampling_client
        self.overlong_filter = overlong_filter
        self.prompt_truncation = prompt_truncation  # "left", "right", or None

        self.enable_api_server: bool = self.config.engine_endpoint.enable
        self.api_server_host: str = self.config.engine_endpoint.host
        self.api_server_port: int = self.config.engine_endpoint.port
        self.api_server_force_port: bool = self.config.engine_endpoint.force_port

        # For interaction
        self.partial_rollout_config = self.config.partial_rollout
        self.enable_partial_rollout = self.partial_rollout_config.enable
        self.partial_rollout_n_iters = self.partial_rollout_config.n_iters
        self.moe_replay = self.config.moe_replay
        self.gamma = gamma

        self.max_steps = max_steps
        self.max_seq_length = max_seq_length
        self.max_seq_length_per_iter = max_seq_length // self.partial_rollout_n_iters
        self.max_prompt_length = max_prompt_length
        self.max_tokens_per_step = max_tokens_per_step
        self.program_timeout = program_timeout

        # Create persistent event loop for this engine
        # All async operations will run in this loop to maintain state across iterations
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(target=self._loop.run_forever, name="Engine-EventLoop", daemon=True)
        self._loop_thread.start()

        # Create lock in the engine's event loop context
        # Use run_coroutine_threadsafe since the loop is now running in another thread
        future = asyncio.run_coroutine_threadsafe(self._create_lock(), self._loop)
        self._session_state_lock = future.result()

        # Tokenizer pool configuration
        # Initialize tokenizer pool for async tokenization (reduces blocking on CPU-bound tokenization)
        assert self.tokenizer is not None, f"Tokenizer cannot be none but received: {self.tokenizer}"
        self._tokenizer_pool = TokenizerPool(
            tokenizer=self.tokenizer,
            num_workers=32,
        )
        # Start the pool in the engine's event loop
        future = asyncio.run_coroutine_threadsafe(self._tokenizer_pool.start(), self._loop)
        future.result()

        if chat_parser is None:
            self.chat_parser = ChatTemplateParser.get_parser(
                tokenizer_pool=self._tokenizer_pool,
                processor=self.processor,
                disable_thinking=kwargs.get("disable_thinking", False),
            )
        else:
            self.chat_parser = chat_parser

        self.validation_mode = False
        self.program_manager = ProgramManager()
        self.val_program_manager = ProgramManager()

        self.session_state_map: dict[str, ProgramState] = {}  # Maps session_id to ProgramState
        self.e_sid_to_i_sid: dict[
            str, str
        ] = {}  # Map of external session id to internal session id in case user wants to use their managed id

        # Optional factory for creating programs from environment arguments (used by add_batch_to_engine)
        self.program_factory = None
        self.global_steps = -1

        # Initialize API server
        if self.enable_api_server:
            self._init_api_server()

    def set_global_steps(self, global_steps: int):
        self.global_steps = global_steps

    async def _create_lock(self):
        """Helper to create an asyncio.Lock in the engine's event loop context."""
        return asyncio.Lock()

    def run_in_engine_loop(self, coro):
        """
        Run a coroutine in the engine's persistent event loop (synchronous blocking call).
        This ensures all async state (locks, futures, tasks) persists across calls.

        Note: Since the loop is running in a background thread, we use run_coroutine_threadsafe.
        """
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()

    async def run_in_engine_loop_async(self, coro):
        """
        Async version: Run a coroutine in the engine's persistent event loop from an async context.

        This allows calling engine methods from a different async event loop while ensuring
        the coroutine executes in the engine's dedicated loop where all state persists.

        Args:
            coro: A coroutine to execute in the engine's event loop

        Returns:
            The result of the coroutine execution

        Example:
            result = await engine.run_in_engine_loop_async(engine.generate(messages, session_id))
        """

        # We're in a different event loop, so we need to submit to the engine's loop
        # Use run_coroutine_threadsafe to safely execute in the engine's loop
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)

        # Wait for the result in a way that doesn't block the current event loop
        # We wrap the future in an asyncio-friendly awaitable
        return await asyncio.wrap_future(future)

    def eval(self):
        self.validation_mode = True

    def train(self):
        self.validation_mode = False

    ##########################
    ### Api server helpers ###
    ##########################
    def _init_api_server(self):
        """Initialize FastAPI server with endpoints for program communication."""
        ensure_port_available(self.api_server_port, force=self.api_server_force_port)

        self.app = FastAPI(title="Engine API")
        self.app.include_router(build_engine_router(self))
        self.app.include_router(build_openai_router(self))

        # Start API server in a separate thread
        self._api_server_thread = threading.Thread(target=self._run_api_server, name="Engine-APIServer", daemon=True)
        self._api_server_thread.start()

        colorful_print(f"⭐ Agent Execution API Server started on port {self.api_server_port} ⭐", "cyan")

    def _run_api_server(self):
        """Run the FastAPI server in a separate thread."""
        # Create a new event loop for this thread since uvicorn needs one
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        config = uvicorn.Config(
            self.app,
            # Bind host is configurable for internal deployments.
            host=os.environ.get("AXON_API_SERVER_HOST", "0.0.0.0"),  # nosec B104
            port=self.api_server_port,
            log_level="warning",
            loop=loop,
            # Performance optimizations
            backlog=2048,  # Increase connection backlog
            limit_concurrency=2048,  # Allow more concurrent requests
        )
        server = uvicorn.Server(config)
        loop.run_until_complete(server.serve())

    ##########################################
    ### Helper methods for post processing ###
    ##########################################

    async def postprocess_program_state(self, state: ProgramState):
        """Postprocess the program state."""
        program_id = state.uid
        masked_out = False

        state.training_steps = state.get_training_steps()
        self.pad_moe_routermap(state)

        # Use batch decoding for efficiency
        if state.training_steps:
            tokens_list = [s.tokens for s in state.training_steps]
            decoded_texts = await self._tokenizer_pool.batch_decode(tokens_list, skip_special_tokens=False)
            for s, decoded_text in zip(state.training_steps, decoded_texts, strict=False):
                s.text = decoded_text

        # here is because we pad the moe_routermaps before
        # we don't check for prompt truncation, as it is meant for partial sampler, last env message will be padded next turn
        if self.moe_replay and state.termination_reason not in [TerminationReason.PROMPT_TRUNCATION]:
            for s in state.training_steps:
                # Check if moe_routermap is valid (not empty list or non-empty tensor)
                has_moe = (isinstance(s.moe_routermap, torch.Tensor) and s.moe_routermap.numel() > 0) or (
                    isinstance(s.moe_routermap, list) and len(s.moe_routermap) > 0
                )
                if has_moe:  # Only check if moe_routermap is not empty
                    assert len(s.moe_routermap) == len(s.tokens), "MOE routermap must be the same length as tokens"

        if self.overlong_filter:
            if state.termination_reason in [
                TerminationReason.TRUNCATION,
                TerminationReason.MAX_STEPS,
                TerminationReason.PROGRAM_TIMEOUT,
            ]:
                # Mask out the entire response for overlong programs if the reward is 0.
                for s in state.training_steps:
                    s.masks = [0] * len(s.masks)
                masked_out = True

        # Log the program completion.
        if state.termination_reason:
            if state.reward and state.reward > 0:
                color = "green"
            else:
                color = "yellow"
            colorful_print(
                f"Program {program_id} completed due to: {state.termination_reason.value}. Reward is {state.reward}. \n",
                color,
            )
            if masked_out:
                colorful_print(f"Program {program_id} is masked out due to overlong filter.", "red")

    def pad_moe_routermap(self, state: ProgramState):
        for step in state.training_steps:
            # Check if moe_routermap is valid (not empty list or non-empty tensor)
            has_moe = (isinstance(step.moe_routermap, torch.Tensor) and step.moe_routermap.numel() > 0) or (
                isinstance(step.moe_routermap, list) and len(step.moe_routermap) > 0
            )
            if self.moe_replay and has_moe:
                expected_len = len(step.tokens)
                current_len = len(step.moe_routermap)
                if current_len < expected_len:
                    padding_needed = expected_len - current_len
                    if DEBUG_MOE_REPLAY:
                        print(
                            f"[DEBUG_MOE_REPLAY, pad_moe_routermap] Step {step.uid}: padding from {current_len} to {expected_len} (+{padding_needed})"
                        )
                        print(f"[DEBUG_MOE_REPLAY, pad_moe_routermap] token_len={len(step.tokens)}")

                    # moe_routermap is a tensor with shape [seq_len, num_layers, num_experts]
                    # Create padding with shape [padding_needed, num_layers, num_experts]
                    moe_routermap_padding = torch.full(
                        (padding_needed,) + step.moe_routermap.shape[1:],
                        -1,
                        dtype=step.moe_routermap.dtype,
                        device=step.moe_routermap.device,
                    )
                    step.moe_routermap = torch.cat([step.moe_routermap, moe_routermap_padding], dim=0)
                elif current_len > expected_len:
                    if DEBUG_MOE_REPLAY:
                        print(
                            f"[DEBUG_MOE_REPLAY, pad_moe_routermap] Step {step.uid}: WARNING - routermap longer than expected: {current_len} > {expected_len}"
                        )
                        print(f"[DEBUG_MOE_REPLAY, pad_moe_routermap] token_len={len(step.tokens)}")

    ######################################################################################################################
    ### External facing method to Frontend, User or API Server (Can be called from any thread and asyncio loop safely)###
    ######################################################################################################################
    async def register_external_session_id(self, external_sid: str, internal_sid: str):
        self.e_sid_to_i_sid[external_sid] = internal_sid

    async def get_internal_session_id(self, e_sid: str):
        return self.e_sid_to_i_sid.get(e_sid, None)

    async def generate_external_session_id(
        self, messages: list[dict[str, str]], e_sid: str, **kwargs
    ) -> tuple[str, bool]:
        assert e_sid in self.e_sid_to_i_sid, f"External session id missing, {e_sid}"
        session_id = self.e_sid_to_i_sid[e_sid]
        if session_id not in self.session_state_map:
            timeout = 30
            await asyncio.sleep(timeout)
            if session_id not in self.session_state_map:
                print(f"[WARN] session_id {session_id} not found after {timeout}s for generation")
                return "", True
        result = await self.generate(messages=messages, session_id=session_id)
        return result[0], result[1]  # drop step_idx for external session callers

    async def end_session_external_session_id(self, reward: float, e_sid: str, **kwargs):
        assert e_sid in self.e_sid_to_i_sid, f"External session id missing, {e_sid}"
        session_id = self.e_sid_to_i_sid[e_sid]
        if session_id not in self.session_state_map:
            timeout = 30
            await asyncio.sleep(timeout)
            if session_id not in self.session_state_map:
                print(f"[WARN] session_id {session_id} not found after {timeout}s for end session")
                return
        await self.end_session(reward=reward, session_id=session_id)
        self.e_sid_to_i_sid.pop(e_sid)

    async def add_to_program_metadata(self, session_id, metadata_key, metadata_val):
        if session_id not in self.session_state_map:
            return
        state: ProgramState = self.session_state_map[session_id]
        async with state.session_lock:
            state.metadata[metadata_key] = metadata_val

    async def append_to_program_metadata(self, session_id, metadata_key, metadata_val):
        if session_id not in self.session_state_map:
            return
        state: ProgramState = self.session_state_map[session_id]
        async with state.session_lock:
            if metadata_key not in state.metadata:
                state.metadata[metadata_key] = []
            state.metadata[metadata_key].append(metadata_val)

    async def check_program_status(self, session_id: str):
        """
        Return if the program is done or not.
        """
        if session_id not in self.session_state_map:
            # Assuming non existing means returned
            return True
        state: ProgramState = self.session_state_map[session_id]
        async with state.session_lock:
            return state.done

    async def init_session(self, group_id: str | None = None, sample_params: dict | None = None):
        """Initializes a new session for the engine.

        Args:
            group_id: Optional group ID for batching programs together
            sample_params: Optional sampling parameters for generation

        Returns:
            session_id: str
        """

        async def _add_lock_for_state(program_state: ProgramState):
            lock = program_state.session_lock
            if lock is None:
                lock = asyncio.Lock()
                program_state.session_lock = lock

        session_id = str(uuid.uuid4())
        uid = session_id
        state = ProgramState(session_id=session_id, uid=uid)

        # Set group_id for program management
        state.group_id = group_id if group_id is not None else str(uuid.uuid4())

        # Set sample_params and global_steps from batch meta_info
        state.sample_params = sample_params if sample_params is not None else {}
        state.global_steps = self.global_steps

        async with self._session_state_lock:
            self.session_state_map[session_id] = state

        await _add_lock_for_state(state)
        state.record_llm_call_finish_time()

        # Add program to program manager immediately to track all programs (including incomplete ones)
        state.enable_partial_rollout = self.enable_partial_rollout
        if self.validation_mode and self.enable_partial_rollout:
            state.enable_partial_rollout = False
        program_manager = self.val_program_manager if self.validation_mode else self.program_manager
        program_manager.add_programs([state])
        # colorful_print(f"Program {session_id} initialized", "cyan")
        return session_id

    async def generate(
        self,
        messages: list[dict[str, Any]],
        session_id: str,
        sample_params: dict | None = None,
        images: list[Any] | None = None,
        videos: list[Any] | None = None,
        processor_kwargs: dict | None = None,
        tools_json: list[dict[str, Any]] | None = None,  # list of definitions for available tools in dict format
        parser_kwargs: dict | None = None,
    ) -> tuple[str, bool]:
        """
        Generate based on messages. Automatically match with existing prefix conversations.
        If no match found, assume all messages are the starting prompt regardless of role.
        The newly generated content is automatically attached to prefix tree.
        Only content generated via this method would be considered trainable.

        NOTE: For tools (please check the chat template parser support tools):
            1. For definition, pass via the tools parameter.
            2. For assistant made tool calls, either provide the exact assistant message in "content" or in "tool_calls" in an accepted format.
            3. For tool responses, use the "tool" role.
        Returns:
            response: str
            stop_program: bool
        """
        async with self._session_state_lock:
            assert session_id in self.session_state_map, f"session_id is not found: {session_id}"
            state: ProgramState = self.session_state_map[session_id]
        state.record_env_time()

        # 1) Check if the program is in termination state.
        try:
            self.check_state_before_generation(state)
        except StopProgram:
            return "", True, -1

        # 2) Create new Step and match with existing prefix tree.
        async with state.session_lock:
            step = Step(
                uid=str(uuid.uuid4()),
                session_id=session_id,
                chat_completions=deepcopy(messages),
            )

            parser_kwargs = parser_kwargs or {}
            if tools_json:
                parser_kwargs["tools"] = tools_json
            input_key: str = self.chat_parser.parse(messages, add_generation_prompt=True, **parser_kwargs)

            # Detect whether this request carries actual multimodal content.
            # Some models (e.g. Qwen3.5) are architecturally VL models even for
            # text-only tasks, so self.processor alone is not a reliable signal.
            has_mm_input = bool(images or videos)
            if not has_mm_input and messages:
                for msg in messages:
                    content = msg.get("content")
                    if isinstance(content, list):
                        for part in content:
                            if isinstance(part, dict) and part.get("type") in MM_PART_TYPES:
                                has_mm_input = True
                                break
                    if has_mm_input:
                        break

            tokenize_ctx = None
            if self.processor and has_mm_input:
                # Multimodal context construction — only when there is real
                # multimodal payload.  VL-architecture models used for text-only
                # tasks (e.g. Qwen3.5 on frozenlake) must still go through the
                # string-based prefix matching path below.
                tokenize_ctx = TokenizeContext(
                    messages=messages,
                    images=images if images else [],
                    videos=videos if videos else [],
                    processor_kwargs=processor_kwargs if processor_kwargs else {},
                )

            # Detect MM placeholder regions in input_key ([] for text-only).
            mm_regions = self.chat_parser.get_mm_regions(input_key, messages) if has_mm_input else []

            # Walk the prefix tree to recover ground-truth token ids for the
            # already-generated portion, then tokenize only the remainder.
            # For MM inputs, the walk handles processor-expanded pad token
            # regions via mm_regions so that placeholder text in input_key
            # correctly maps to the N expanded tokens in the tree.
            prefix_ids, prefix_strs, prefix_text_pos, recovered_mm = state.prefix_tree.longest_text_prefix(
                input_key, mm_regions or None
            )

            if prefix_ids:
                remaining = input_key[prefix_text_pos:]
                if remaining:
                    # Check if suffix contains new MM content that needs processor
                    new_mm_in_suffix = [r for r in mm_regions if r[0] >= prefix_text_pos]
                    if new_mm_in_suffix and tokenize_ctx is not None:
                        # Suffix has new images/videos — run processor on suffix
                        new_images = [r[4] for r in new_mm_in_suffix if r[2] == "image"]
                        new_videos = [r[4] for r in new_mm_in_suffix if r[2] == "video"]
                        suffix_ctx = TokenizeContext(
                            messages=None,  # not needed — we pass raw text + images
                            images=new_images,
                            videos=new_videos,
                            processor_kwargs=tokenize_ctx.processor_kwargs if tokenize_ctx else {},
                        )
                        suffix_output = await self.chat_parser.tokenize(remaining, ctx=suffix_ctx)
                        suffix_ids, suffix_strs = suffix_output.token_ids, suffix_output.token_strs
                        new_mm_data = suffix_output.multi_modal_data
                    else:
                        suffix_output = await self.chat_parser.tokenize(remaining)
                        suffix_ids, suffix_strs = suffix_output.token_ids, suffix_output.token_strs
                        new_mm_data = None
                else:
                    suffix_ids, suffix_strs, new_mm_data = [], [], None

                input_tokens = prefix_ids + suffix_ids
                prompt_token_strs = prefix_strs + suffix_strs
                mm = _combine_mm_data(recovered_mm, new_mm_data)
                mm_copy = deepcopy(mm) if mm else None
            else:
                # No prefix match (first turn or empty tree) — full tokenization
                tokenize_output: TokenizeOutput = await self.chat_parser.tokenize(input_key, ctx=tokenize_ctx)
                input_tokens = tokenize_output.token_ids
                mm: MultiModalData | None = tokenize_output.multi_modal_data
                mm_copy = deepcopy(mm) if mm else None
                prompt_token_strs = tokenize_output.token_strs
            step.set_response(
                text=input_key,
                tokens=input_tokens,
                masks=[0] * len(input_tokens),
                logprobs=[0.0] * len(input_tokens),
                moe_routermap=[],
            )
            if mm:
                step.multi_modal_data = mm  # used for generation
            step.sample_params = deepcopy(state.sample_params)
            if sample_params:
                step.sample_params = deepcopy(sample_params)
        # 3) Generate text with model.
        llm_response, stop_program, stop_partial_program = await self.generate_llm_response_with_checks(step)

        # For partial sampler, we need to generate text with model until the program is terminated.
        # This may persist across RL iterations if the program is not terminated.
        while stop_partial_program:
            if stop_program:
                break
            loop = asyncio.get_running_loop()
            fut = loop.create_future()
            async with state.session_lock:
                state.partial_rollout_futures.append(fut)
            await fut
            llm_response, stop_program, stop_partial_program = await self.generate_llm_response_with_checks(step)

        async with state.session_lock:
            # Update the prefix trie with new key
            all_tokens = step.tokens
            step.order_idx = len(state.steps)
            state.steps.append(step)
            state.num_steps += 1

            moe_routermaps = step.moe_routermap

            # Check if moe_routermaps is valid
            has_moe = (isinstance(moe_routermaps, torch.Tensor) and moe_routermaps.numel() > 0) or (
                isinstance(moe_routermaps, list) and len(moe_routermaps) > 0
            )
            if self.moe_replay and has_moe:
                expected_len = len(all_tokens)
                current_len = len(moe_routermaps)
                if current_len != expected_len:
                    if DEBUG_MOE_REPLAY:
                        print(
                            f"[DEBUG_MOE_REPLAY, generate] Step {step.uid}: routermap length {current_len} != expected {expected_len}"
                        )
                    if current_len < expected_len:
                        # moe_routermaps is a tensor with shape [seq_len, num_layers, num_experts]
                        padding_needed = expected_len - current_len
                        padding_entry = torch.full(
                            (padding_needed,) + moe_routermaps.shape[1:],
                            -1,
                            dtype=moe_routermaps.dtype,
                            device=moe_routermaps.device,
                        )
                        moe_routermaps = torch.cat([moe_routermaps, padding_entry], dim=0)
                    else:
                        moe_routermaps = moe_routermaps[:expected_len]
                    step.moe_routermap = moe_routermaps

            # Prompt strs computed during tokenization; response strs from vLLM.
            # Truncation checks may shorten step.tokens — align token_strs to match.
            token_strs = (prompt_token_strs + list(step.response_token_strs))[: len(all_tokens)]
            # Detect MM expansion regions in the token sequence for tree annotation.
            mm_expansions = _detect_mm_expansions(all_tokens, mm_regions, self.tokenizer) if mm_regions else None
            state.prefix_tree.insert(
                token_ids=all_tokens,
                token_strs=token_strs,
                masks=step.masks,
                logprobs=step.logprobs,
                moe_routermaps=moe_routermaps,
                step_idx=step.order_idx,
                multi_modal_data=mm_copy,
                mm_expansions=mm_expansions,
            )
            state.record_llm_call_finish_time()
            state.metrics.steps += 1
        return llm_response, stop_program, step.order_idx

    async def end_session(self, session_id: str, reward: float, step_rewards: dict[int, float] | None = None):
        """End a session and mark the program as complete for trainer retrieval."""
        async with self._session_state_lock:
            if session_id not in self.session_state_map:
                colorful_print(f"Warning: session_id {session_id} not found in session_state_map", "yellow")
                return
            state = self.session_state_map[session_id]
            if state.done:
                # don't allow update to a closed program
                return
            state.record_env_time()
            state.done = True
            state.reward = reward
            if step_rewards:
                state.step_rewards = step_rewards
            if reward >= 1e9 or reward <= -1e9:
                state.termination_reason = TerminationReason.BAD_PROGRAM
            elif not state.termination_reason:
                state.termination_reason = TerminationReason.ENV_DONE

        # Process the completed program outside the lock
        await self.postprocess_program_state(state)

        colorful_print(f"Program {session_id} ended", "cyan")

    async def get_finished_programs(self) -> list[ProgramState]:
        """Retrieve all finished programs for trainer processing and remove them from manager."""
        program_manager = self.val_program_manager if self.validation_mode else self.program_manager
        finished_programs = list(program_manager.pop_programs(completed=True))

        # Remove finished sessions from session_state_map
        # This is the primary cleanup location - sessions remain in the map until retrieved by trainer
        async with self._session_state_lock:
            for program in finished_programs:
                session_id = program.session_id
                if session_id in self.session_state_map:
                    self.session_state_map.pop(session_id, None)
        return finished_programs

    async def get_engine_metrics(self) -> dict:
        """Get current engine metrics for monitoring."""
        program_manager = self.val_program_manager if self.validation_mode else self.program_manager
        return {
            "engine/pending_programs": program_manager.get_num_incomplete_programs(),
            "engine/completed_programs": program_manager.get_num_completed_programs(),
            **program_manager.get_completed_programs_statistics(),
        }

    async def resume_partial_rollout_programs(self):
        """
        Resume any programs that are paused on partial sampler futures.

        This method:
        1. Finds all incomplete programs in the program manager
        2. Unblocks their partial sampler futures so they can continue generation
        3. Re-adds them to the program manager to continue tracking

        This should be called at the start of each RL iteration to continue
        programs that were paused in the previous iteration.
        """
        if not self.enable_partial_rollout:
            return 0

        resumed_count = 0
        # Use session_state_map as source of truth for active sessions
        async with self._session_state_lock:
            for _, program_state in list(self.session_state_map.items()):
                async with program_state.session_lock:
                    # Skip completed programs
                    if program_state.done:
                        continue

                    # Unblock all pending partial sampler futures
                    while program_state.partial_rollout_futures:
                        fut = program_state.partial_rollout_futures.pop(0)
                        if not fut.done():
                            fut.set_result(None)
                        resumed_count += 1

        if resumed_count > 0:
            colorful_print(f"Resumed {resumed_count} partial sampler programs.", "cyan")

        return resumed_count

    async def all_programs_safe_to_collect(self) -> bool:
        """
        Check if all active programs are in a safe state to proceed with RL iteration.

        This is the critical synchronization point for partial sampler support:
        - The trainer MUST wait until this returns True before proceeding with RL updates
        - This ensures all programs have either completed or are paused waiting for the next iteration

        A program is "safe to collect" if it is either:
        1. Finished (done=True) - ready to be collected via get_finished_programs()
        2. Paused on partial sampler future - waiting for next RL iteration to continue

        Returns:
            True if all programs are either finished or paused, False if any are still actively running
        """
        # Check all active sessions with lock to avoid race conditions
        async with self._session_state_lock:
            session_items = list(self.session_state_map.items())

            if len(session_items) == 0:
                return False

            for _, state in session_items:
                async with state.session_lock:
                    # If done, it's safe
                    if state.done:
                        continue
                    # If has pending partial sampler futures, it's paused and safe
                    if state.partial_rollout_futures:
                        continue
                    # Otherwise, program is still actively running - not safe yet
                    return False

            return True

    async def get_session_diagnostics(self) -> str:
        """Return a summary of session states for debugging."""
        async with self._session_state_lock:
            total = len(self.session_state_map)
            done = sum(1 for s in self.session_state_map.values() if s.done)
            partial = sum(1 for s in self.session_state_map.values() if not s.done and s.partial_rollout_futures)
            running = total - done - partial
        return f"sessions={total} done={done} partial={partial} running={running}"

    async def shutdown(self):
        """
        Shutdown the engine and release all resources.

        This method should be called when the engine is no longer needed to:
        1. Shutdown the tokenizer pool worker processes
        2. Stop the event loop
        3. Clean up any other resources
        """
        # Shutdown tokenizer pool (check prevents double shutdown)
        if self._tokenizer_pool is not None:
            await self._tokenizer_pool.shutdown()
            self._tokenizer_pool = None
            colorful_print("TokenizerPool shut down", "cyan")

    ###########################
    ### Model query helpers ###
    ###########################
    async def generate_llm_response_with_checks(self, step: Step) -> tuple[str, bool, bool]:
        """
        Perform necessary checks before and after query the model based on step.
        Automatically accumulates the model response as an assistant message to Step and StepToken.

        Return:
            cumulative_response: str
            stop_program: bool
            stop_partial_program: bool
        """
        state = self.session_state_map[step.session_id]
        cumulative_response = ""
        stop_program = False
        stop_partial_program = False
        try:
            # Check if this state can do generation
            self.check_state_before_generation(state)
            # Check if this results in prompt truncation
            #  Note this should never happen since everything is moved to response.
            self.check_prompt_truncation_termination(step)
            # Check if this results in response truncation due to new messages
            self.check_truncation_termination(step, last_assistant_response=False)

            # Query model
            cumulative_response = await self.generate_llm_response(step)

            # Check if now there is truncation after new response
            self.check_truncation_termination(step, last_assistant_response=True)
        except StopProgram:
            stop_program = True
        except StopPartialProgram:
            step.partial_rollout_max_tokens += self.max_seq_length_per_iter
            colorful_print(f"Program {state.uid} preempted due to: PARTIAL_ROLLOUT.", "magenta")
            stop_partial_program = True

        return cumulative_response, stop_program, stop_partial_program

    async def generate_llm_response(self, step: Step) -> str:
        """
        Query the model based on step. Update all state metrics and update the StepToken associated.
        Automatically accumulates the model response as an assistant message to Step and StepToken.
        """

        # Fetch the state from the session state map.
        state = self.session_state_map[step.session_id]

        # Max token needs to account for existing generated content
        if not state.enable_partial_rollout:
            max_tokens = self.max_seq_length
            if self.max_tokens_per_step is not None:
                max_tokens = min(max_tokens, len(step.tokens) + self.max_tokens_per_step)
        else:
            max_tokens = step.partial_rollout_max_tokens + self.max_seq_length_per_iter
        max_tokens = max_tokens - len(step.tokens) - len(step.partial_tokens)

        # Edge case: If the env message is so long that it covers multiple partial sampler iterations, we should preempt the program.
        if max_tokens <= 0 and state.enable_partial_rollout:
            raise StopPartialProgram(f"Program {state.uid} preempted due to partial sampler limit.")
        else:
            assert max_tokens > 0, f"Max tokens must be non-negative. {max_tokens} < 0"

        step.sample_params["max_tokens"] = max_tokens
        start_time = time.time()
        model_output = await self._get_model_response(
            prompt=step.text + step.partial_text,
            prompt_ids=step.tokens + step.partial_tokens,
            application_id=state.uid,
            multi_modal_data=step.multi_modal_data,
            **step.sample_params,
        )

        step.partial_text += model_output.response
        step.partial_tokens.extend(model_output.token_ids)
        step.partial_logprobs.extend(model_output.logprobs)
        step.partial_token_strs.extend(model_output.token_strs)

        # Handle moe_routermap prefix cache case - fill in missing entries from prefix tree
        expected_moe_len = len(step.tokens) + len(step.partial_logprobs)
        if self.moe_replay and len(model_output.moe_routermap) < expected_moe_len:
            cached_len = expected_moe_len - len(model_output.moe_routermap)
            all_tokens = step.tokens + step.partial_tokens
            cached_tokens = all_tokens[:cached_len]

            # Look up moe_routermap for cached tokens from prefix tree

            cached_moe_routermap = []
            node = state.prefix_tree.root
            for token_id in cached_tokens:
                child = node.get_child(token_id)
                # Check if moe_routermap is valid
                has_moe = child is not None and (
                    (isinstance(child.moe_routermap, torch.Tensor) and child.moe_routermap.numel() > 0)
                    or (isinstance(child.moe_routermap, list) and len(child.moe_routermap) > 0)
                )
                if has_moe:
                    cached_moe_routermap.append(child.moe_routermap)
                    node = child
                else:
                    # No moe_routermap in prefix tree, will be handled by padding later
                    break

            if len(cached_moe_routermap) == cached_len:
                # Successfully recovered all cached moe_routermap from prefix tree
                # cached_moe_routermap is a list of 2D tensors, need to stack and concatenate with model_output.moe_routermap (3D tensor)
                if cached_moe_routermap:
                    cached_tensor = torch.stack(cached_moe_routermap)  # [cached_len, num_layers, num_experts]
                    step.partial_moe_routermap = torch.cat([cached_tensor, model_output.moe_routermap], dim=0)
                else:
                    step.partial_moe_routermap = model_output.moe_routermap
            else:
                # Couldn't recover all entries - this will trigger padding with -1 in set_response
                step.partial_moe_routermap = model_output.moe_routermap
        else:
            step.partial_moe_routermap = model_output.moe_routermap

        if len(step.partial_moe_routermap) != expected_moe_len and DEBUG_MOE_REPLAY:
            print(
                f"[DEBUG_MOE_REPLAY, generate_llm_response]: Partial moe_routermap length: {len(step.partial_moe_routermap)} != expected {expected_moe_len} (tokens={len(step.tokens)} + partial={len(step.partial_logprobs)})"
            )
        delta_time = time.time() - start_time

        # Update metrics.
        state.metrics.llm_time += delta_time
        state.metrics.total_time += delta_time
        if model_output.spec_decode_metrics is not None:
            m = model_output.spec_decode_metrics
            state.metrics.spec_draft_tokens += m.num_draft_tokens
            state.metrics.spec_accepted_tokens += m.num_accepted_tokens
            state.metrics.spec_verify_count += m.num_completions

        stop_reason = model_output.stop_reason
        if (
            state.enable_partial_rollout
            and stop_reason == ModelStopReason.LENGTH
            and step.token_len + len(step.partial_tokens) < self.max_seq_length
        ):  # Stopped due to length limit.
            raise StopPartialProgram(f"Program {state.uid} preempted due to partial sampler limit.")

        # Commit the changes to stored data
        # Keep the raw sampled text for training/prefix replay, but store
        # semantic assistant text in chat history and return it to programs.
        chat_content = self.chat_parser.assistant_message_content(step.partial_text)
        step.set_response(
            text=step.partial_text,
            tokens=step.partial_tokens,
            masks=[1] * len(step.partial_tokens),
            logprobs=step.partial_logprobs,
            moe_routermap=step.partial_moe_routermap,
        )
        step.chat_completions.append({"role": "assistant", "content": chat_content})
        # Save response token_strs before reset clears them
        step.response_token_strs.extend(step.partial_token_strs)
        # Reset partial state
        step.reset_partial_state()
        return chat_content

    ###############################################
    ### Check for termination condition helpers ###
    ###############################################
    def check_prompt_truncation_termination(self, step: Step):
        state = self.session_state_map[step.session_id]
        program_id = state.uid

        # At step 0, the initial observation (system + observation) is stored in tokens
        # Check if it exceeds max_prompt_length (which limits initial observation length).
        if state.num_steps == 0:
            initial_prompt_len = step.token_len
            if initial_prompt_len > self.max_prompt_length:
                # If prompt_truncation is set, truncate instead of terminating
                if self.prompt_truncation in ("left", "right"):
                    tokens_to_remove = initial_prompt_len - self.max_prompt_length
                    if self.prompt_truncation == "left":
                        # Truncate from the left (keep the most recent/rightmost tokens)
                        step.tokens = step.tokens[tokens_to_remove:]
                        step.masks = step.masks[tokens_to_remove:]
                        step.logprobs = step.logprobs[tokens_to_remove:]
                    else:  # "right"
                        # Truncate from the right (keep the earliest/leftmost tokens)
                        step.tokens = step.tokens[: self.max_prompt_length]
                        step.masks = step.masks[: self.max_prompt_length]
                        step.logprobs = step.logprobs[: self.max_prompt_length]
                    # Update token length after truncation
                    step.token_len = len(step.tokens)
                    colorful_print(
                        f"Program {program_id}'s prompt truncated from {initial_prompt_len}->{step.token_len} tokens. Direction is {self.prompt_truncation}.",
                        "yellow",
                    )
                else:
                    # No truncation mode set, terminate the program
                    state.termination_reason = TerminationReason.PROMPT_TRUNCATION
                    state.reward = 0
                    state.done = True
                    colorful_print(
                        f"Program {program_id} terminated due to prompt truncation ({initial_prompt_len} > {self.max_prompt_length}). Reward is 0.",
                        "yellow",
                    )
                    raise StopProgram(
                        f"Program {program_id} terminated due to prompt truncation ({initial_prompt_len} > {self.max_prompt_length})."
                    )

    def check_truncation_termination(self, step: Step, last_assistant_response: bool = False):
        state = self.session_state_map[step.session_id]
        program_id = state.uid
        # Check for truncation exit condition.
        if step.token_len >= self.max_seq_length:
            truncation_length = self.max_seq_length - step.token_len
            if truncation_length < 0:
                step.tokens = step.tokens[:truncation_length]
                step.masks = step.masks[:truncation_length]
                step.logprobs = step.logprobs[:truncation_length]
            # Check if moe_routermap is valid

            has_moe = (isinstance(step.moe_routermap, torch.Tensor) and step.moe_routermap.numel() > 0) or (
                isinstance(step.moe_routermap, list) and len(step.moe_routermap) > 0
            )
            if self.moe_replay and has_moe:
                step.moe_routermap = step.moe_routermap[: len(step.tokens)]  # truncate the moe_routermap
            # Edge case: If assistant message is truncated, set reward to 0.
            if last_assistant_response:
                state.reward = 0.0
            step.token_len = self.max_seq_length
            state.termination_reason = TerminationReason.TRUNCATION
            raise StopProgram(f"Program {program_id} terminated due to truncation.")

    def check_state_before_generation(self, state: ProgramState):
        # Check if program is in "done" state.
        program_id = state.uid
        if state.done:
            if not state.termination_reason:
                state.termination_reason = TerminationReason.ENV_DONE
            raise StopProgram(f"Program {program_id} terminated previously already.")

        # Check if program has exceeded max steps.
        if state.num_steps >= self.max_steps:
            state.termination_reason = TerminationReason.MAX_STEPS
            raise StopProgram(f"Program {program_id} terminated due to max steps.")

        # NOTE: disabled timeout right now since handled by program initiator. Check run_program method.
        # # Check for program timeout exit condition.
        # if metrics.total_time >= self.program_timeout:
        #     state.termination_reason = TerminationReason.PROGRAM_TIMEOUT
        #     raise StopProgram(f"Program {program_id} terminated due to program timeout.")

    #########################################################################################
    ### Model generation engine interation helpers. Running in dedicated generation loop  ###
    #########################################################################################
    async def _get_model_response(
        self,
        prompt: str,
        prompt_ids: list[int],
        application_id: str,
        multi_modal_data: MultiModalData | None = None,
        **override_sampling_params,
    ) -> ModelOutput:
        """
        Compute model response asynchronously via the SamplingClient.

        Args:
            prompt: The input prompt text (unused, kept for interface compatibility)
            prompt_ids: The tokenized prompt as a list of token IDs
            application_id: Unique identifier for the application
            multi_modal_data: Optional multimodal data for the request
            **override_sampling_params: Additional arguments to pass to the model (e.g. top_p, temperature, etc.)

        Returns:
            ModelOutput: The model's response text, logprobs, and stop reason
        """
        output = await self.sampling_client.sample(
            [prompt_ids],
            application_id=application_id,
            multi_modal_data_list=[multi_modal_data] if multi_modal_data else [None],
            **override_sampling_params,
        )

        token_ids = output["token_ids"][0]
        logprobs = output["logprobs"][0]
        token_strs = output.get("token_strs", [[]])[0]
        finish_reason = _first_or_none(output.get("finish_reasons"))
        sampler_stop_reason = _first_or_none(output.get("stop_reasons"))
        # Dense models don't emit routed_experts → output won't have moe_routermap.
        # Treat missing/empty the same as moe_replay disabled.
        has_routermap = self.moe_replay and "moe_routermap" in output and len(output["moe_routermap"]) > 0
        if has_routermap:
            # Because moe_routermap will be off by 1, last token don't need to go through forward
            # output["moe_routermap"][0] is a tensor with shape [seq_len, num_layers, num_experts]
            raw_moe_routermap = output["moe_routermap"][0]
            prompt_len = len(prompt_ids)
            response_len = len(token_ids)
            expected_n_minus_1 = prompt_len + response_len - 1
            vllm_returned_len = len(raw_moe_routermap)

            assert vllm_returned_len == expected_n_minus_1, (
                f"MOE routermap length mismatch: vllm returned {vllm_returned_len} entries, "
                f"expected prompt_len({prompt_len}) + response_len({response_len}) - 1 = {expected_n_minus_1}. "
                f"This usually means prefix caching is enabled in vLLM (check --no-enable-prefix-caching)."
            )

            # Create padding tensor with shape [1, num_layers, num_experts]
            moe_routermap_padding = torch.full_like(raw_moe_routermap[0:1], -1)
            # Concatenate along sequence dimension: [seq_len+1, num_layers, num_experts]
            moe_routermap = torch.cat([raw_moe_routermap, moe_routermap_padding], dim=0)
        else:
            moe_routermap = []

        stop_reason = self._model_stop_reason(
            token_ids=token_ids,
            finish_reason=finish_reason,
            sampler_stop_reason=sampler_stop_reason,
            sampling_params=override_sampling_params,
        )

        assert len(token_ids) == len(logprobs), "Token IDs and logprobs must have the same length."
        # vLLM guarantees len(token_strs) == len(token_ids) when logprobs >= 1
        # and echo is disabled (our default).
        assert len(token_strs) == len(token_ids), (
            f"vLLM token_strs length mismatch: {len(token_strs)} != {len(token_ids)}. "
            f"Ensure logprobs=1 is set and echo is not enabled."
        )

        return ModelOutput.from_token_strs(
            token_ids=token_ids,
            token_strs=token_strs,
            logprobs=logprobs,
            stop_reason=stop_reason,
            moe_routermap=moe_routermap,
        )

    def _model_stop_reason(
        self,
        *,
        token_ids: list[int],
        finish_reason: str | None,
        sampler_stop_reason: Any,
        sampling_params: dict[str, Any],
    ) -> str:
        """Map sampler completion metadata to the engine's stop/length split."""
        if isinstance(finish_reason, str):
            return ModelStopReason.LENGTH if finish_reason.lower() == "length" else ModelStopReason.STOP

        if sampler_stop_reason is not None:
            return ModelStopReason.STOP

        stop_token_ids = _as_token_id_set(getattr(self.tokenizer, "eos_token_id", None))
        stop_token_ids.update(_as_token_id_set(sampling_params.get("stop_token_ids")))
        if token_ids and token_ids[-1] in stop_token_ids:
            return ModelStopReason.STOP
        return ModelStopReason.LENGTH
