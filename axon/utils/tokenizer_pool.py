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
"""
TokenizerPool: A subprocess-based tokenizer pool for high-throughput tokenization.

This module provides async tokenization using a process pool to avoid blocking
the event loop during CPU-intensive tokenization operations. This is especially
important when handling thousands of concurrent requests.
"""

from __future__ import annotations

import asyncio
import logging
import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

logger = logging.getLogger(__name__)


# Global tokenizer instance for worker processes
_worker_tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast | None = None


def _init_worker(tokenizer_path: str, tokenizer_kwargs: dict | None = None):
    """
    Initialize the tokenizer in a worker process.

    This function is called once when a worker process starts. It loads the
    tokenizer from the given path and stores it in a global variable for
    subsequent tokenization calls.

    Args:
        tokenizer_path: Path to the tokenizer (HuggingFace model path or local path)
        tokenizer_kwargs: Optional kwargs to pass to AutoTokenizer.from_pretrained
    """
    global _worker_tokenizer
    from transformers import AutoTokenizer

    kwargs = tokenizer_kwargs or {}
    _worker_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, **kwargs)  # nosec B615
    logger.debug(f"Worker {os.getpid()} initialized tokenizer from {tokenizer_path}")


def _encode_in_worker(
    text: str,
    add_special_tokens: bool = True,
) -> list[int]:
    """
    Encode text to token IDs in a worker process.

    Args:
        text: The text to encode
        add_special_tokens: Whether to add special tokens (default True)

    Returns:
        List of token IDs
    """
    global _worker_tokenizer
    if _worker_tokenizer is None:
        raise RuntimeError("Worker tokenizer not initialized")
    return _worker_tokenizer.encode(text, add_special_tokens=add_special_tokens)


def _encode_with_strs_in_worker(
    text: str,
    add_special_tokens: bool = True,
) -> tuple[list[int], list[str]]:
    """
    Encode text to token IDs and per-token strings in a single call.

    Uses the fast tokenizer's Rust backend which computes character offsets
    during tokenization itself — O(N), no extra decode pass.
    ``"".join(token_strs) == text`` is guaranteed by the offset spans.

    Falls back to per-token ``decode([tid])`` for slow tokenizers.

    Returns:
        (token_ids, token_strs) — same length lists.
    """
    global _worker_tokenizer
    if _worker_tokenizer is None:
        raise RuntimeError("Worker tokenizer not initialized")
    # Fast path: Rust tokenizer gives offsets as part of encoding — O(N), bijective.
    backend = getattr(_worker_tokenizer, "_tokenizer", None)
    if backend is not None:
        encoding = backend.encode(text, add_special_tokens=add_special_tokens)
        # Build per-token strings from character offsets.
        # When byte-level BPE splits a multi-byte character into multiple tokens,
        # all sub-tokens share the same (start, end) offset. We assign the text
        # slice to the FIRST token in each span and empty strings to the rest,
        # so "".join(token_strs) == text holds exactly.
        token_strs = []
        prev_end = 0
        for s, e in encoding.offsets:
            if s >= prev_end:
                # New or non-overlapping span → take the text slice
                token_strs.append(text[s:e])
                prev_end = e
            else:
                # Overlapping span (sub-token of same char) → empty
                token_strs.append("")
        return encoding.ids, token_strs
    # Slow tokenizer fallback: per-token decode (not guaranteed bijective).
    token_ids = _worker_tokenizer.encode(text, add_special_tokens=add_special_tokens)
    token_strs = [_worker_tokenizer.decode([tid], skip_special_tokens=False) or "" for tid in token_ids]
    return token_ids, token_strs


def _decode_in_worker(
    token_ids: list[int],
    skip_special_tokens: bool = True,
    clean_up_tokenization_spaces: bool = True,
) -> str:
    """
    Decode token IDs to text in a worker process.

    Args:
        token_ids: List of token IDs to decode
        skip_special_tokens: Whether to skip special tokens (default True)
        clean_up_tokenization_spaces: Whether to clean up tokenization spaces (default True)

    Returns:
        Decoded text string
    """
    global _worker_tokenizer
    if _worker_tokenizer is None:
        raise RuntimeError("Worker tokenizer not initialized")
    return _worker_tokenizer.decode(
        token_ids,
        skip_special_tokens=skip_special_tokens,
        clean_up_tokenization_spaces=clean_up_tokenization_spaces,
    )


def _batch_encode_in_worker(
    texts: list[str],
    add_special_tokens: bool = True,
) -> list[list[int]]:
    """
    Batch encode multiple texts to token IDs in a worker process.

    Args:
        texts: List of texts to encode
        add_special_tokens: Whether to add special tokens (default True)

    Returns:
        List of token ID lists
    """
    global _worker_tokenizer
    if _worker_tokenizer is None:
        raise RuntimeError("Worker tokenizer not initialized")
    # Use batch encoding for efficiency
    result = _worker_tokenizer(texts, add_special_tokens=add_special_tokens)
    return result["input_ids"]


def _batch_decode_in_worker(
    token_ids_list: list[list[int]],
    skip_special_tokens: bool = True,
    clean_up_tokenization_spaces: bool = True,
) -> list[str]:
    """
    Batch decode multiple token ID sequences to text in a worker process.

    Args:
        token_ids_list: List of token ID lists to decode
        skip_special_tokens: Whether to skip special tokens (default True)
        clean_up_tokenization_spaces: Whether to clean up tokenization spaces (default True)

    Returns:
        List of decoded text strings
    """
    global _worker_tokenizer
    if _worker_tokenizer is None:
        raise RuntimeError("Worker tokenizer not initialized")
    return _worker_tokenizer.batch_decode(
        token_ids_list,
        skip_special_tokens=skip_special_tokens,
        clean_up_tokenization_spaces=clean_up_tokenization_spaces,
    )


class TokenizerPool:
    """
    A subprocess-based tokenizer pool for high-throughput async tokenization.

    This class manages a pool of worker processes, each with its own tokenizer
    instance. It provides async methods for encoding and decoding that don't
    block the event loop, making it suitable for handling thousands of
    concurrent tokenization requests.

    Example:
        ```python
        pool = TokenizerPool(
            tokenizer=tokenizer,
            tokenizer_path="meta-llama/Llama-2-7b-hf",
            num_workers=4
        )
        await pool.start()

        # Single encode/decode
        tokens = await pool.encode("Hello, world!")
        text = await pool.decode(tokens)

        # Batch operations for better throughput
        token_lists = await pool.batch_encode(["Hello", "World"])
        texts = await pool.batch_decode(token_lists)

        await pool.shutdown()
        ```

    Attributes:
        tokenizer: The main tokenizer instance (for accessing attributes like eos_token_id)
        num_workers: Number of worker processes in the pool
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
        tokenizer_path: str | None = None,
        num_workers: int | None = None,
        tokenizer_kwargs: dict | None = None,
    ):
        """
        Initialize the TokenizerPool.

        Args:
            tokenizer: A pre-loaded tokenizer instance. Used for accessing tokenizer
                      attributes (eos_token_id, pad_token_id, etc.) without IPC overhead.
            tokenizer_path: Path to load the tokenizer in worker processes. If None,
                           will try to use tokenizer.name_or_path.
            num_workers: Number of worker processes. Defaults to min(8, cpu_count).
            tokenizer_kwargs: Optional kwargs passed to AutoTokenizer.from_pretrained
                             in worker processes (e.g., trust_remote_code=True).
        """
        self.tokenizer = tokenizer
        self._tokenizer_path = tokenizer_path or getattr(tokenizer, "name_or_path", None)
        if self._tokenizer_path is None:
            raise ValueError("Could not determine tokenizer path. Please provide tokenizer_path explicitly.")

        self._tokenizer_kwargs = tokenizer_kwargs or {}
        # Add common kwargs that might be needed
        if hasattr(tokenizer, "trust_remote_code") and tokenizer.trust_remote_code:
            self._tokenizer_kwargs.setdefault("trust_remote_code", True)

        # Default to min(8, cpu_count) workers - tokenization is CPU-bound
        # but too many workers can cause memory issues
        self._num_workers = num_workers or min(8, os.cpu_count() or 4)
        self._executor: ProcessPoolExecutor | None = None
        self._started = False
        self._loop: asyncio.AbstractEventLoop | None = None

    @property
    def num_workers(self) -> int:
        """Return the number of worker processes."""
        return self._num_workers

    # Forward common tokenizer attributes for convenience
    @property
    def eos_token_id(self) -> int:
        """Return the EOS token ID."""
        return self.tokenizer.eos_token_id

    @property
    def pad_token_id(self) -> int | None:
        """Return the PAD token ID."""
        return self.tokenizer.pad_token_id

    @property
    def bos_token_id(self) -> int | None:
        """Return the BOS token ID."""
        return self.tokenizer.bos_token_id

    @property
    def vocab_size(self) -> int:
        """Return the vocabulary size."""
        return self.tokenizer.vocab_size

    async def start(self):
        """
        Start the worker processes.

        This method initializes the process pool and loads the tokenizer in each
        worker. It should be called before any encode/decode operations.
        """
        if self._started:
            logger.warning("TokenizerPool already started")
            return

        self._loop = asyncio.get_running_loop()

        # Create the process pool with initializer
        self._executor = ProcessPoolExecutor(
            max_workers=self._num_workers,
            initializer=_init_worker,
            initargs=(self._tokenizer_path, self._tokenizer_kwargs),
        )

        # Warm up the pool by submitting a simple task to each worker
        # This ensures all workers have loaded the tokenizer before we return
        warmup_tasks = []
        for _ in range(self._num_workers):
            task = self._loop.run_in_executor(self._executor, _encode_in_worker, "warmup", False)
            warmup_tasks.append(task)

        await asyncio.gather(*warmup_tasks)
        self._started = True
        logger.info(f"TokenizerPool started with {self._num_workers} workers")

    async def shutdown(self, wait: bool = True):
        """
        Shutdown the worker processes.

        Args:
            wait: If True, wait for all pending tasks to complete before shutting down.
        """
        if self._executor is not None:
            self._executor.shutdown(wait=wait)
            self._executor = None
        self._started = False
        logger.info("TokenizerPool shut down")

    def _ensure_started(self):
        """Ensure the pool is started before operations."""
        if not self._started or self._executor is None:
            raise RuntimeError("TokenizerPool not started. Call 'await pool.start()' first.")

    async def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
    ) -> list[int]:
        """
        Encode text to token IDs asynchronously.

        Args:
            text: The text to encode
            add_special_tokens: Whether to add special tokens (default True)

        Returns:
            List of token IDs
        """
        self._ensure_started()
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            partial(_encode_in_worker, text, add_special_tokens),
        )

    async def encode_with_strs(
        self,
        text: str,
        add_special_tokens: bool = True,
    ) -> tuple[list[int], list[str]]:
        """
        Encode text to (token_ids, token_strs) asynchronously.

        Uses offset mapping to slice the original text, so the per-token
        strings are exact and concatenate back to the input.
        """
        self._ensure_started()
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            partial(_encode_with_strs_in_worker, text, add_special_tokens),
        )

    async def decode(
        self,
        token_ids: list[int],
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = True,
    ) -> str:
        """
        Decode token IDs to text asynchronously.

        Args:
            token_ids: List of token IDs to decode
            skip_special_tokens: Whether to skip special tokens (default True)
            clean_up_tokenization_spaces: Whether to clean up tokenization spaces (default True)

        Returns:
            Decoded text string
        """
        self._ensure_started()
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            partial(
                _decode_in_worker,
                token_ids,
                skip_special_tokens,
                clean_up_tokenization_spaces,
            ),
        )

    async def batch_encode(
        self,
        texts: list[str],
        add_special_tokens: bool = True,
    ) -> list[list[int]]:
        """
        Batch encode multiple texts to token IDs asynchronously.

        This method distributes the batch across workers for parallel processing.
        For small batches (< num_workers), it may process in a single worker for efficiency.

        Args:
            texts: List of texts to encode
            add_special_tokens: Whether to add special tokens (default True)

        Returns:
            List of token ID lists
        """
        self._ensure_started()
        if not texts:
            return []

        loop = asyncio.get_running_loop()

        # For small batches, process in a single worker
        if len(texts) <= self._num_workers:
            return await loop.run_in_executor(
                self._executor,
                partial(_batch_encode_in_worker, texts, add_special_tokens),
            )

        # For larger batches, distribute across workers
        chunk_size = (len(texts) + self._num_workers - 1) // self._num_workers
        chunks = [texts[i : i + chunk_size] for i in range(0, len(texts), chunk_size)]

        tasks = [
            loop.run_in_executor(
                self._executor,
                partial(_batch_encode_in_worker, chunk, add_special_tokens),
            )
            for chunk in chunks
        ]

        results = await asyncio.gather(*tasks)
        # Flatten results
        return [token_ids for chunk_result in results for token_ids in chunk_result]

    async def batch_decode(
        self,
        token_ids_list: list[list[int]],
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = True,
    ) -> list[str]:
        """
        Batch decode multiple token ID sequences to text asynchronously.

        This method distributes the batch across workers for parallel processing.
        For small batches (< num_workers), it may process in a single worker for efficiency.

        Args:
            token_ids_list: List of token ID lists to decode
            skip_special_tokens: Whether to skip special tokens (default True)
            clean_up_tokenization_spaces: Whether to clean up tokenization spaces (default True)

        Returns:
            List of decoded text strings
        """
        self._ensure_started()
        if not token_ids_list:
            return []

        loop = asyncio.get_running_loop()

        # For small batches, process in a single worker
        if len(token_ids_list) <= self._num_workers:
            return await loop.run_in_executor(
                self._executor,
                partial(
                    _batch_decode_in_worker,
                    token_ids_list,
                    skip_special_tokens,
                    clean_up_tokenization_spaces,
                ),
            )

        # For larger batches, distribute across workers
        chunk_size = (len(token_ids_list) + self._num_workers - 1) // self._num_workers
        chunks = [token_ids_list[i : i + chunk_size] for i in range(0, len(token_ids_list), chunk_size)]

        tasks = [
            loop.run_in_executor(
                self._executor,
                partial(
                    _batch_decode_in_worker,
                    chunk,
                    skip_special_tokens,
                    clean_up_tokenization_spaces,
                ),
            )
            for chunk in chunks
        ]

        results = await asyncio.gather(*tasks)
        # Flatten results
        return [text for chunk_result in results for text in chunk_result]

    # Synchronous methods that fall back to the main tokenizer
    # Useful for non-async code paths or when pool isn't needed
    def encode_sync(
        self,
        text: str,
        add_special_tokens: bool = True,
    ) -> list[int]:
        """
        Synchronous encode using the main tokenizer (no subprocess).

        Use this for single synchronous encode operations where async overhead
        isn't worth it.

        Args:
            text: The text to encode
            add_special_tokens: Whether to add special tokens (default True)

        Returns:
            List of token IDs
        """
        return self.tokenizer.encode(text, add_special_tokens=add_special_tokens)

    def decode_sync(
        self,
        token_ids: list[int],
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = True,
    ) -> str:
        """
        Synchronous decode using the main tokenizer (no subprocess).

        Use this for single synchronous decode operations where async overhead
        isn't worth it.

        Args:
            token_ids: List of token IDs to decode
            skip_special_tokens: Whether to skip special tokens (default True)
            clean_up_tokenization_spaces: Whether to clean up tokenization spaces (default True)

        Returns:
            Decoded text string
        """
        return self.tokenizer.decode(
            token_ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
        )
