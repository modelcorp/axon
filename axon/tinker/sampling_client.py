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
import base64
import logging
from copy import deepcopy

import numpy as np
import torch

from axon.engine.state.program_state import MultiModalData
from axon.utils.openai_client import (
    _qwen2_5_vl_dedup_image_tokens,
    fetch_responses_from_addresses,
    poll_completions_openai,
)

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("asyncio").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)


def _decode_routed_experts(payload: dict) -> torch.Tensor:
    """Decode routed_experts from vllm's API response.

    Args:
        payload: dict with "data" (base64), "shape" (list[int]), "dtype" (str)

    Returns:
        torch.Tensor of shape [seq_len, num_layers, topk], dtype int32
    """
    raw = base64.b64decode(payload["data"])
    shape = payload["shape"]
    dtype = np.dtype(payload["dtype"])
    arr = np.frombuffer(raw, dtype=dtype).reshape(shape)
    return torch.from_numpy(arr.copy()).to(torch.int32)


class SamplingClient:
    """
    Client for generating text completions from sampler servers.

    The SamplingClient manages a pool of server addresses and distributes requests using
    a least-used strategy with data locality considerations. It maintains usage
    counts for each server and provides methods for generating text completions
    and controlling generation state across all servers.

    Attributes:
        addresses: List of server addresses in "host:port" format
        tensor_parallel_size: Number of tensor parallel processes per server
        config: Configuration object containing model and generation parameters
        tokenizer: Tokenizer instance for the model
        pad_token_id: Padding token ID from the tokenizer
        eos_token_id: End-of-sequence token ID from the tokenizer
        model_name: Name of the model being served
        pausing_strategy: Strategy for pausing generation ("drain", "hold", "continue", "reset")
    """

    def __init__(
        self,
        config,
        tokenizer,
        processor,
        addresses: list[str],
        sampler_servers=None,
        max_concurrent_requests: int = 128,
    ):
        """
        Initialize the SamplingClient with configuration and server addresses.

        Args:
            config: Configuration object containing model and generation settings
            tokenizer: Tokenizer instance for the model
            addresses: List of server addresses in "host:port" format
            sampler_servers: List of Server (Ray actor) handles for managing model lifecycle
            max_concurrent_requests: Maximum concurrent requests per vLLM engine (default 128).
        """
        # List of "ip:port" strings
        self.addresses = addresses
        self.sampler_servers = sampler_servers if sampler_servers is not None else []
        self.tensor_parallel_size = config.sampler.get("tensor_model_parallel_size", 1)
        self._lock = asyncio.Lock()
        self._usage: dict[str, int] = {}
        self._application_id_to_address: dict[str, str] = {}
        # Semaphore per server address to limit concurrent requests to each vLLM engine
        # Note: high concurrent requests / engine can cause vLLM to have weird performance issues (i.e. repeat tokens).
        # After extensive testing, 128 is most likely the best default value.
        self._max_concurrent_requests = max_concurrent_requests
        self._address_semaphores: dict[str, asyncio.Semaphore] = {}
        # Initialize usage counts and semaphores for any new addresses
        for addr in self.addresses:
            if addr not in self._usage:
                self._usage[addr] = 0
            self._address_semaphores[addr] = asyncio.Semaphore(max_concurrent_requests)
        self.counter = 0
        self.config = config
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.processor = processor
        self.model_name = config.model_path

        if self.config.moe_replay:
            assert not self.config.sampler.enable_prefix_caching, "MOE replay is not supported with prefix caching"

        self.pausing_strategy = self.config.sampler_pausing_strategy
        assert self.pausing_strategy in ["drain", "hold", "continue", "reset"], f"{self.pausing_strategy} not supported"

    async def get_address(self, application_id: str) -> str:
        """
        Select a server address for the given application ID using load balancing.

        This method implements a least-used server selection strategy with data locality
        considerations. If an application ID has been seen before, it tries to route to
        the same server unless there's significant load imbalance.

        Args:
            application_id: Unique identifier for the application/session making the request

        Returns:
            str: Server address in "host:port" format that should handle the request
        """
        async with self._lock:
            min_address, min_usage = min(self._usage.items(), key=lambda x: x[1])
            if application_id not in self._application_id_to_address:
                self._application_id_to_address[application_id] = min_address
                self._usage[min_address] += 1
            else:
                # Data locality
                cur_address = self._application_id_to_address[application_id]
                cur_usage = self._usage[cur_address]
                # Load balance if there is skew
                if (min_usage == 0 or cur_usage - min_usage >= 4) and cur_usage > 0:
                    self._application_id_to_address[application_id] = min_address
                    self._usage[min_address] += 1
                else:
                    self._usage[cur_address] += 1
        return self._application_id_to_address[application_id]

    async def release_address(self, addr: str, application_id: str) -> None:
        """
        Release a server address by decrementing its usage count.

        This method should be called when a request is completed to properly
        maintain the load balancing state.

        Args:
            addr: Server address that was used for the request
            application_id: Application ID that made the request (for future use)
        """
        async with self._lock:
            self._usage[addr] = max(0, self._usage.get(addr, 0) - 1)

    def get_tokenizer(self):
        """Returns the tokenizer instance used by this client."""
        return self.tokenizer

    async def sample(
        self,
        prompt_token_ids_list: list[list[int]],
        application_id: str,
        multi_modal_data_list: list[MultiModalData | None],
        **override_sampling_params,
    ):
        """
        Generate text completions for a batch of prompts.

        This method takes a list of tokenized prompts and generates completions using
        the configured sampling parameters. It handles load balancing across servers,
        processes the responses, and returns structured results including token IDs,
        log probabilities, and optionally MoE router maps.

        Args:
            prompt_token_ids_list: List of tokenized prompts, where each prompt is a list of token IDs
            application_id: Unique identifier for the application making the request
            multi_modal_data_list: List of multimodal data (one per prompt, None if text-only)
            **override_sampling_params: Additional sampling parameters to override defaults.
                                      Special parameter 'validate' can be set to True for validation mode.

        Returns:
            Dict containing:
                - "token_ids": List of generated token sequences (flattened from [B, N, L] to [B*N, L])
                - "logprobs": List of log probability sequences corresponding to token_ids
                - "moe_routermap": MoE router activation maps (only if moe_replay is enabled)
        """
        override_sampling_params = deepcopy(override_sampling_params)
        kwargs = dict(
            n=self.config.decoding.n,
            max_tokens=self.config.sampler.max_model_len,
            temperature=self.config.decoding.temperature,
            top_p=self.config.decoding.top_p,
            top_k=self.config.decoding.top_k,
            repetition_penalty=self.config.decoding.repetition_penalty,
            logprobs=self.config.decoding.get("logprobs", 1),
            # Keep token-level completion details so trainer-side agreement can
            # compare the sampler surface without a decode round-trip.
            spaces_between_special_tokens=False,
            include_stop_str_in_output=True,
            skip_special_tokens=False,
            return_token_ids=True,
        )

        is_validate = override_sampling_params.pop("validate", False)
        if is_validate:
            kwargs.update(
                {
                    # Keep validation top-p/top-k aligned with training for now;
                    # changing them has increased sampler/trainer logprob drift
                    # in observed vLLM runs.
                    # "top_p": self.config.validation.decoding.top_p,
                    # "top_k": self.config.validation.decoding.top_k,
                    "temperature": self.config.validation.decoding.temperature,
                    "n": self.config.validation.decoding.n,
                    "repetition_penalty": self.config.validation.decoding.get("repetition_penalty", 1.0),
                }
            )
        kwargs.update(override_sampling_params)
        # Router routes one application_id to one address, so n=1 is required.
        if kwargs.get("n", 1) != 1:
            kwargs["n"] = 1

        # Fetch the address for the application_id (Agentix routing policy).
        address = await self.get_address(application_id)

        tasks = []
        assert len(prompt_token_ids_list) == len(multi_modal_data_list), (
            f"Number of prompts should equal to number of mm data (empty list for those not needing it) but received {len(prompt_token_ids_list)}, {len(multi_modal_data_list)}"
        )
        for prompt_token_ids, multi_modal_data in zip(prompt_token_ids_list, multi_modal_data_list, strict=False):
            tasks.append(
                self.submit_completions(
                    address=address,
                    model=self.model_name,
                    prompt=prompt_token_ids,
                    multi_modal_data=multi_modal_data,
                    **kwargs,
                )
            )

        # Potential blocking: asyncio.gather can block if any task takes too long
        logger.debug("Sending total requests: %s", len(tasks))
        try:
            completions_list = await asyncio.gather(*tasks)
        finally:
            await self.release_address(address, application_id)

        batch_size = len(prompt_token_ids_list)
        batch_response_ids: list[list[int]] = [[] for _ in range(batch_size)]
        batch_logprobs: list[list[float]] = [[] for _ in range(batch_size)]
        batch_token_strs: list[list[str]] = [[] for _ in range(batch_size)]
        batch_finish_reasons: list[list[str | None]] = [[] for _ in range(batch_size)]
        batch_stop_reasons: list[list[str | int | None]] = [[] for _ in range(batch_size)]
        batch_moe_routermaps: list[list[list[list[float]]]] = [[] for _ in range(batch_size)]

        for batch_index, completions in enumerate(completions_list):
            comps = []
            logprobs = []
            token_strs = []
            finish_reasons = []
            stop_reasons = []
            moe_routermap = []
            for choice in completions.get("choices", []):
                token_ids = choice.get("token_ids", [])
                log_prob_floats = choice.get("logprobs", {}).get("token_logprobs", [])
                tok_strs = choice.get("logprobs", {}).get("tokens", [])
                assert len(token_ids) == len(log_prob_floats) == len(tok_strs), (
                    "Token IDs and log probabilities and token strs must have the same length: "
                    + str(len(token_ids))
                    + " "
                    + str(len(log_prob_floats))
                    + " "
                    + str(len(tok_strs))
                )
                comps.append(token_ids)
                logprobs.append(log_prob_floats)
                token_strs.append(tok_strs)
                finish_reasons.append(choice.get("finish_reason"))
                stop_reasons.append(choice.get("stop_reason"))
                # vLLM fork puts routed_experts on each choice (base64 + shape dict).
                if self.config.moe_replay:
                    re = choice.get("routed_experts")
                    if re is not None:
                        moe_routermap.append(_decode_routed_experts(re))
            batch_response_ids[batch_index] = comps
            batch_logprobs[batch_index] = logprobs
            batch_token_strs[batch_index] = token_strs
            batch_finish_reasons[batch_index] = finish_reasons
            batch_stop_reasons[batch_index] = stop_reasons
            batch_moe_routermaps[batch_index] = moe_routermap

        # Flatten a list of token IDs and their logprobs. Before: [B, N, L] -> After: [B * N, L]
        batch_response_ids = [r for r_ids in batch_response_ids if r_ids is not None for r in r_ids]
        batch_logprobs = [ll for l_probs in batch_logprobs if l_probs is not None for ll in l_probs]
        batch_token_strs = [s for s_lists in batch_token_strs if s_lists is not None for s in s_lists]
        batch_finish_reasons = [r for reasons in batch_finish_reasons if reasons is not None for r in reasons]
        batch_stop_reasons = [r for reasons in batch_stop_reasons if reasons is not None for r in reasons]

        batch_dict = {
            "token_ids": batch_response_ids,
            "logprobs": batch_logprobs,
            "token_strs": batch_token_strs,
            "finish_reasons": batch_finish_reasons,
            "stop_reasons": batch_stop_reasons,
        }
        if self.config.moe_replay:
            batch_moe_routermaps = [r for r_maps in batch_moe_routermaps if r_maps is not None for r in r_maps]
            # Only emit moe_routermap if vLLM actually returned routing data.
            # Dense models don't emit routed_experts → nothing to replay.
            if batch_moe_routermaps:
                batch_dict["moe_routermap"] = batch_moe_routermaps

        return batch_dict

    async def submit_completions(
        self, address, model, prompt, multi_modal_data: MultiModalData | None = None, **kwargs
    ):
        """
        Submit a single completion request to a specific server address.

        This is a wrapper around poll_completions_openai that handles the
        network communication for a single completion request.
        Uses a semaphore to limit concurrent requests to vLLM.

        Args:
            address: Server address to send the request to
            model: Model name to use for completion
            prompt: Tokenized prompt (list of token IDs)
            **kwargs: Additional parameters for the completion request

        Returns:
            Dict containing the completion response from the server
        """
        # Current Tinker path handles image payloads through the vLLM renderer.
        # Additional modalities and SGLang support should share this request
        # assembly path once their renderer interfaces are stable.
        image_data = multi_modal_data.image if multi_modal_data else None
        prompt = _qwen2_5_vl_dedup_image_tokens(prompt, self.processor)

        # Use per-address semaphore to limit concurrent requests to each vLLM engine
        semaphore = self._address_semaphores.get(address)
        if semaphore is None:
            # Lazily create semaphore for unknown addresses
            semaphore = asyncio.Semaphore(self._max_concurrent_requests)
            self._address_semaphores[address] = semaphore
        async with semaphore:
            return await poll_completions_openai(
                address=address, model=model, prompt=prompt, image_data=image_data, **kwargs
            )

    async def wake_up(self):
        """Wake up the sampler servers (load model weights and build kv cache)."""
        await asyncio.gather(*[replica.wake_up() for replica in self.sampler_servers])

    async def sleep(self):
        """Put the sampler servers to sleep (offload model weights and discard kv cache)."""
        await asyncio.gather(*[replica.sleep() for replica in self.sampler_servers])

    async def pause_all(self):
        """
        Pause text generation on all servers in the router pool.

        This method sends a pause request to all configured server addresses
        using the configured pausing strategy. The pause behavior depends on
        the strategy:
        - "drain": Complete current requests then pause
        - "hold": Pause immediately, holding current state
        - "continue": Continue generation (no-op for pause)
        - "reset": Reset generation state and pause

        Note: This functionality requires vLLM servers with dp=1 and only works
        with the completion API.

        Returns:
            List of response dictionaries from each server, containing success
            status and any error information

        Raises:
            Exception: If any server fails to pause successfully
        """
        # NOTE: for vllm, must be dp=1 and only working on completion api.
        # NOTE: for reset mode, needs to resend requests with abort status from router. Currently already doing 3 retries automatically.
        results = await fetch_responses_from_addresses(
            addresses=self.addresses, endpoint="/pause_generation", payload={"strategy": self.pausing_strategy}
        )
        # Check results
        for result in results:
            if result["success"]:
                print(f"{result['address']}: Paused vLLM successfully")
            else:
                print(f"{result['address']}: Paused vLLMfailed with {result}")
                raise Exception(result["error"])

        return results

    async def continue_all(self):
        """
        Resume text generation on all servers in the router pool.

        This method sends a continue request to all configured server addresses
        to resume generation that was previously paused. All servers should
        respond successfully for the operation to be considered complete.

        Returns:
            List of response dictionaries from each server, containing success
            status and any error information

        Raises:
            Exception: If any server fails to resume generation successfully
        """
        results = await fetch_responses_from_addresses(
            addresses=self.addresses, endpoint="/continue_generation", payload={}
        )
        # Check results
        for result in results:
            if result["success"]:
                print(f"{result['address']}: Continue vLLM successfully")
            else:
                print(f"{result['address']}: Continue vLLM failed with {result}")
                raise Exception(result["error"])

        return results
