# Copyright 2025 Model AI Corp.
# Copyright 2025 z.ai
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
# Ported from THUDM/slime (https://github.com/THUDM/slime), Apache License 2.0.
"""
Compression-based repetition detection for generated responses.

Detects degenerate repetitive outputs by measuring how compressible the
generated text is. Highly repetitive text compresses to a much smaller size,
yielding a high compression ratio.

Ported from the slime framework's metric_utils module.
"""

from __future__ import annotations

import zlib
from typing import Any, Literal

from axon.protocol import DataProto


def compression_ratio(
    data: str | bytes,
    *,
    encoding: str = "utf-8",
    algorithm: Literal["zlib", "gzip", "bz2", "lzma"] = "zlib",
    level: int = 9,
) -> tuple[float, float]:
    """Compute the compression ratio and savings percentage for the given data.

    Args:
        data: The text or bytes to compress.
        encoding: Encoding to use when converting str to bytes.
        algorithm: Compression algorithm to use.
        level: Compression level (higher = more compression).

    Returns:
        Tuple of (compression_ratio, savings_percentage).
        A higher ratio means more repetitive content.
    """
    if isinstance(data, str):
        raw = data.encode(encoding)
    else:
        raw = data

    original = len(raw)
    if original == 0:
        return float("inf"), 0.0

    if algorithm == "zlib":
        compressed = zlib.compress(raw, level)
    elif algorithm == "gzip":
        import gzip

        compressed = gzip.compress(raw, compresslevel=level)
    elif algorithm == "bz2":
        import bz2

        compressed = bz2.compress(raw, compresslevel=level)
    elif algorithm == "lzma":
        import lzma

        compressed = lzma.compress(raw, preset=level)
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    comp_len = len(compressed)
    if comp_len == 0:
        return float("inf"), 100.0

    ratio = original / comp_len
    savings_pct = 100.0 * (1.0 - comp_len / original)
    return ratio, savings_pct


def has_repetition(text: str, *, tail_chars: int = 10000, threshold: float = 10.0) -> bool:
    """Check whether the text exhibits degenerate repetition.

    Uses compression ratio on the tail of the text as a heuristic:
    highly repetitive text compresses extremely well (ratio >> 1).

    Args:
        text: The generated text to check.
        tail_chars: Number of trailing characters to evaluate.
        threshold: Compression ratio above which text is considered repetitive.

    Returns:
        True if the text is repetitive.
    """
    if len(text) > tail_chars and compression_ratio(text[-tail_chars:])[0] > threshold:
        return True
    return False


def compute_repetition_metrics(
    batch: DataProto,
    tokenizer,
) -> dict[str, Any]:
    """Compute repetition fraction from a training batch.

    Decodes the full program token IDs (including env observation tokens)
    and checks each for degenerate repetition using compression-ratio analysis.

    Uses batch decoding for speed.

    Args:
        batch: DataProto containing ``responses``, ``attention_mask``, and
            ``response_mask`` tensors.
        tokenizer: HuggingFace tokenizer for decoding token IDs.

    Returns:
        Dict with ``"response/repetition_frac"`` key.
    """
    input_ids = batch.batch["input_ids"]  # (batch_size, max_seq_length)
    response_mask = batch.batch["response_mask"]

    # Use response_mask to identify response tokens. response_mask is 1 for
    # response tokens and 0 for prompt/padding tokens.
    # Extract response tokens per row: tokens where response_mask is 1,
    # so batch_decode never sees prompt or pad tokens.
    token_id_lists = [input_ids[i, response_mask[i].bool()].tolist() for i in range(input_ids.size(0))]

    # Batch decode all responses at once (accepts list[list[int]]).
    texts = tokenizer.batch_decode(token_id_lists, skip_special_tokens=False)

    repetition_count = sum(1 for text in texts if has_repetition(text))

    return {"response/repetition_frac": repetition_count / len(texts) if texts else 0.0}
