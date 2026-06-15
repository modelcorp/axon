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
"""Context Parallelism utilities for efficient Megatron loss computation.

When context_parallel_size > 1, Megatron splits each sequence across CP ranks
using a zigzag (ring attention) pattern.  After the forward pass each rank only
has logits/log_probs for its local chunk.

The default path all-gathers the output so every CP rank has the full sequence,
then computes loss redundantly on all ranks.  With CP=K this wastes (K-1)/K of
the loss compute and requires an all-gather of the output tensor.

These utilities enable computing loss *only* on the local chunk by building a
boolean mask that identifies which response tokens are owned by this CP rank.
Combined with ``skip_cp_gather=True`` in ``postprocess_packed_seqs``, this
eliminates both the communication and the redundant compute.

Ported from the slime project's ``cp_utils.py`` and adapted for axon's padded
2-D batch layout ``(B, S)``.
"""

from __future__ import annotations

import math

import torch
from megatron.core import parallel_state as mpu

# ---------------------------------------------------------------------------
# Shared helpers — single source of truth for CP padding & chunking.
#
# Both ``preprocess_packed_seqs`` (axon/models/mcore/util.py) and
# ``get_cp_local_response_mask`` (below) must agree on how sequences are
# padded and split into zigzag chunks.  These helpers centralise that logic
# so the two call-sites cannot drift.
# ---------------------------------------------------------------------------


def compute_cp_padded_lens(
    seqlens: list[int],
    tp_size: int,
    cp_size: int,
    use_fp8_padding: bool = False,
) -> list[int]:
    """Compute per-sequence padded lengths matching ``preprocess_packed_seqs``.

    This is the **single source of truth** for the alignment formula.
    ``preprocess_packed_seqs`` should call this instead of inline math to
    guarantee consistency with the loss-mask computation.

    Args:
        seqlens: original valid lengths per sequence.
        tp_size: tensor-parallel world size.
        cp_size: context-parallel world size (must be > 1).
        use_fp8_padding: apply the FP8-aware alignment used by
            Transformer Engine (``lcm(16, align)`` plus 128× last-seq pad).

    Returns:
        List of padded lengths, one per sequence.
    """
    align_size = tp_size * cp_size * 2
    if use_fp8_padding:
        original_align_size = align_size
        align_size = math.lcm(16, align_size)

    padded_lens: list[int] = []
    for total_len in seqlens:
        pad = (align_size - total_len % align_size) % align_size
        padded_lens.append(total_len + pad)

    if use_fp8_padding:
        align_size_last = original_align_size * 128
        cum_padded = sum(padded_lens)
        pad_last = (align_size_last - cum_padded % align_size_last) % align_size_last
        padded_lens[-1] += pad_last

    return padded_lens


def compute_cp_chunk_boundaries(
    padded_len: int,
    cp_size: int,
    cp_rank: int,
) -> tuple[int, list[tuple[int, int]]]:
    """Return ``(half_chunk_size, [(cs0, ce0), (cs1, ce1)])`` for a zigzag split.

    Each sequence of ``padded_len`` is split into ``2 * cp_size`` equal chunks.
    CP rank *r* receives chunk *r* (from the start) and chunk
    ``2*cp_size - r - 1`` (from the end).

    Args:
        padded_len: padded sequence length (from :func:`compute_cp_padded_lens`).
        cp_size: context-parallel world size.
        cp_rank: this rank's index within the CP group.

    Returns:
        ``(half, chunks)`` where ``half = padded_len // (2 * cp_size)`` and
        ``chunks`` is a list of two ``(start, end)`` half-open ranges.
    """
    half = padded_len // (2 * cp_size)
    chunks = [
        (cp_rank * half, (cp_rank + 1) * half),
        ((2 * cp_size - cp_rank - 1) * half, (2 * cp_size - cp_rank) * half),
    ]
    return half, chunks


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_cp_local_response_mask(
    attention_mask: torch.Tensor,
    response_length: int,
    use_fp8_padding: bool = False,
) -> torch.Tensor | None:
    """Build a boolean mask indicating which response tokens are local to this CP rank.

    Uses the same zigzag chunking as Megatron's ``preprocess_packed_seqs``:
    each sequence of padded length L is split into ``2 * cp_size`` chunks of
    size ``L / (2 * cp_size)``.  CP rank *r* gets chunk *r* (from the start)
    and chunk ``2*cp_size - r - 1`` (from the end).

    Args:
        attention_mask: ``[B, S]`` original attention mask (1/True for valid tokens).
        response_length: sequence length (``data["input_ids"].size(1)``).
        use_fp8_padding: match the FP8-aware alignment used in
            ``preprocess_packed_seqs`` (``lcm(16, align_size)``).

    Returns:
        ``[B, response_length]`` bool tensor.  ``True`` where this CP rank owns
        the token.  Returns ``None`` when ``cp_size == 1`` (no masking needed).
    """
    cp_size = mpu.get_context_parallel_world_size()
    if cp_size <= 1:
        return None

    cp_rank = mpu.get_context_parallel_rank()
    tp_size = mpu.get_tensor_model_parallel_world_size()

    batch_size = attention_mask.shape[0]
    device = attention_mask.device

    # Move to CPU for the per-sample loop (avoids GPU sync per iteration).
    seqlens_cpu: list[int] = attention_mask.sum(dim=-1, dtype=torch.int32).tolist()

    # Use shared helper — single source of truth for padding logic.
    padded_lens = compute_cp_padded_lens(seqlens_cpu, tp_size, cp_size, use_fp8_padding)

    mask = torch.zeros(batch_size, response_length, dtype=torch.bool, device=device)

    for i in range(batch_size):
        total_len = seqlens_cpu[i]
        _, chunks = compute_cp_chunk_boundaries(padded_lens[i], cp_size, cp_rank)

        # The logit at valid-stream position k predicts the token at k+1.
        # ``log_prob[:, j]`` (j-th response slot) corresponds to the logit at
        # valid-stream position ``total_len - response_length - 1 + j``.
        logit_start = total_len - response_length - 1

        for cs, ce in chunks:
            # Intersect chunk [cs, ce) with valid logit range
            # [max(0, logit_start), total_len - 1).
            s = max(cs, max(0, logit_start))
            e = min(ce, total_len - 1)  # last logit position is total_len - 2 (predicts last token)
            if s >= e:
                continue
            # Convert valid-stream logit positions to response-tensor indices.
            j_start = max(0, s - logit_start)
            j_end = min(response_length, e - logit_start)
            if j_start < j_end:
                mask[i, j_start:j_end] = True

    return mask
