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
"""Utility functions for policy loss computation.

Two-axis loss aggregation framework
====================================

Loss aggregation reduces a ``(B, T)`` per-token loss matrix to a scalar
in two orthogonal stages:

1. **token_reduce** ``(B, T) → (B,)`` — normalize tokens within each row.
2. **batch_reduce** ``(B,) → scalar`` — normalize rows across the batch.

Both axes always apply. No special cases.

Supported token_reduce modes:

=========== =============== ======================================================
Mode        ÷ what          Use case
=========== =============== ======================================================
``sum``     1 (nothing)     Raw token sum per step. REINFORCE log-prob sum.
                            Also the token_reduce for the standard ``token-mean``
                            reduction (``sum`` + ``token-mean`` batch_reduce).
``mean``    valid tokens    Per-step mean. Each step equal regardless of length.
            in this row     GRPO paper convention.
``mean-norm`` T (context    Fixed-scale. Consistent gradient magnitude regardless
            length)         of response length. Empirically useful.
``mean-program`` total      Per-program token pool. All tokens across all steps
            program tokens  of a program share one denominator.
=========== =============== ======================================================

Supported batch_reduce modes:

============== =============== =====================================================
Mode           ÷ what          Use case
============== =============== =====================================================
``token-mean`` total valid     Every token equal weight. Standard LM/PPO.
               tokens
``step-mean``  valid step      Every step equal weight. GRPO default.
               count
``program-mean`` valid program Every program equal weight. Multi-step RL.
               count (÷nps)    Rows are divided by ``num_program_steps`` first.
============== =============== =====================================================

Recommended configs::

    # Standard PPO / LM (every token equal):
    token_reduce="sum", batch_reduce="token-mean"

    # GRPO (each step equal, length-independent):
    token_reduce="mean", batch_reduce="step-mean"

    # Multi-step agent RL (each program equal):
    token_reduce="mean", batch_reduce="program-mean"

    # REINFORCE-style (log-prob sum per step, avg over steps):
    token_reduce="sum", batch_reduce="step-mean"

    # Fixed-scale normalization (empirically good):
    token_reduce="mean-norm", batch_reduce="step-mean"
"""

from __future__ import annotations

import torch

from axon.trainer.algos.constants import VALID_BATCH_REDUCE, VALID_TOKEN_REDUCE

__all__ = ["agg_loss", "clip_by_value", "entropy_from_logits", "masked_mean"]


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean of *values* where *mask* is nonzero.  Supports gradient flow."""
    mask = mask.to(values.dtype)
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


def clip_by_value(tensor: torch.Tensor, min_val: torch.Tensor | float, max_val: torch.Tensor | float) -> torch.Tensor:
    """Element-wise clamp of *tensor* to ``[min_val, max_val]``."""
    if isinstance(min_val, torch.Tensor) or isinstance(max_val, torch.Tensor):
        return torch.maximum(
            torch.minimum(tensor, max_val if isinstance(max_val, torch.Tensor) else torch.tensor(max_val)),
            min_val if isinstance(min_val, torch.Tensor) else torch.tensor(min_val),
        )
    return torch.clamp(tensor, min=min_val, max=max_val)


def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Categorical entropy from logits.  Shape ``(..., V) → (...)``."""
    logits = logits - logits.max(dim=-1, keepdim=True).values
    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log_softmax(logits, dim=-1)
    return -(probs * log_probs).sum(dim=-1)


# ---------------------------------------------------------------------------
# Loss aggregation
# ---------------------------------------------------------------------------

_VALID_TOKEN_REDUCE = VALID_TOKEN_REDUCE
_VALID_BATCH_REDUCE = VALID_BATCH_REDUCE


def agg_loss(
    loss_mat: torch.Tensor,
    loss_mask: torch.Tensor,
    *,
    token_reduce: str = "sum",
    batch_reduce: str = "token-mean",
    num_program_tokens: torch.Tensor | None = None,
    num_program_steps: torch.Tensor | None = None,
    valid_token_count: torch.Tensor | float | None = None,
    valid_batch_size: torch.Tensor | float | None = None,
    valid_program_count: torch.Tensor | float | None = None,
    per_row_token_count: torch.Tensor | None = None,
) -> torch.Tensor:
    """Aggregate a ``(B, T)`` per-token loss matrix into a scalar.

    Two-stage reduction — both stages always apply:

    1. **token_reduce** ``(B, T) → (B,)``: see module docstring for modes.
    2. **batch_reduce** ``(B,) → ()``: see module docstring for modes.

    Args:
        loss_mat: Per-token loss, shape ``(B, T)``.
        loss_mask: Mask, shape ``(B, T)``.  Zeros mark padding tokens.
        token_reduce: Token reduction mode (``sum``, ``mean``, ``mean-norm``,
            ``mean-program``).
        batch_reduce: Batch reduction mode (``token-mean``, ``step-mean``,
            ``program-mean``).
        num_program_tokens: Total token count across all steps of the
            program, per row ``(B,)``.  Required for ``mean-program``.
        num_program_steps: Step count of the program, per row ``(B,)``.
            Required for ``program-mean`` batch reduction.
        valid_token_count: Mini-batch-wide count of valid tokens.
        valid_batch_size: Mini-batch-wide count of valid rows (steps).
        valid_program_count: Mini-batch-wide count of valid programs.
        per_row_token_count: Optional ``(B,)`` override for the per-row
            denominator used by ``token_reduce="mean"``.  When CP-local
            loss masking is active, ``loss_mask`` only covers this rank's
            chunk, so ``loss_mask.sum(dim=-1)`` is ~1/K of the true count.
            Pass the **original** (pre-CP-mask) per-row counts here to
            keep the per-row mean correct.

    Returns:
        Scalar loss tensor.
    """
    if token_reduce not in _VALID_TOKEN_REDUCE:
        raise ValueError(f"Invalid token_reduce: {token_reduce!r}. Choose from: {sorted(_VALID_TOKEN_REDUCE)}")
    if batch_reduce not in _VALID_BATCH_REDUCE:
        raise ValueError(f"Invalid batch_reduce: {batch_reduce!r}. Choose from: {sorted(_VALID_BATCH_REDUCE)}")

    # ---- token reduce: (B, T) → (B,) ------------------------------------
    row_losses = _reduce_tokens(loss_mat, loss_mask, token_reduce, num_program_tokens, per_row_token_count)

    # ---- per-program step normalization ----------------------------------
    # For program-* batch modes, divide each row by its program's step count
    # so multi-step programs contribute once (not S times).
    if batch_reduce.startswith("program-"):
        assert num_program_steps is not None, f"num_program_steps required for batch_reduce='{batch_reduce}'"
        row_losses = row_losses / num_program_steps.clamp(min=1)

    # ---- batch reduce: (B,) → () ----------------------------------------
    return _reduce_batch(
        row_losses,
        loss_mask,
        batch_reduce,
        valid_token_count=valid_token_count,
        valid_batch_size=valid_batch_size,
        valid_program_count=valid_program_count,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _reduce_batch(
    row_losses: torch.Tensor,
    loss_mask: torch.Tensor,
    mode: str,
    *,
    valid_token_count: torch.Tensor | float | None = None,
    valid_batch_size: torch.Tensor | float | None = None,
    valid_program_count: torch.Tensor | float | None = None,
) -> torch.Tensor:
    """Reduce per-row losses to a scalar: ``(B,) → ()``."""
    raw = row_losses.sum()

    if mode == "token-mean":
        denom = valid_token_count if valid_token_count is not None else loss_mask.sum().clamp_min(1.0)
        return raw / _to_scalar(denom, raw)

    if mode == "step-mean":
        if valid_batch_size is not None:
            return raw / _to_scalar(valid_batch_size, raw)
        local_count = (loss_mask.sum(dim=-1) > 0).float().sum()
        return raw / local_count.clamp(min=1)

    if mode == "program-mean":
        assert valid_program_count is not None, "valid_program_count required for batch_reduce='program-mean'"
        return raw / _to_scalar(valid_program_count, raw)

    raise ValueError(f"Unknown batch_reduce mode: {mode!r}")


def _to_scalar(val, ref: torch.Tensor) -> torch.Tensor:
    """Cast *val* to a scalar on the same device/dtype as *ref*, clamped ≥ 1."""
    t = torch.as_tensor(val, dtype=ref.dtype, device=ref.device)
    if t.dim() > 0:
        t = t[0]
    return t.clamp(min=1)


def _reduce_tokens(
    loss_mat: torch.Tensor,
    loss_mask: torch.Tensor,
    mode: str,
    num_program_tokens: torch.Tensor | None = None,
    per_row_token_count: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reduce per-token losses to per-row losses: ``(B, T) → (B,)``."""
    token_sum = (loss_mat * loss_mask).sum(dim=-1)
    if mode == "sum":
        return token_sum
    if mode == "mean":
        denom = per_row_token_count if per_row_token_count is not None else loss_mask.sum(dim=-1)
        return token_sum / denom.clamp(min=1)
    if mode == "mean-norm":
        return token_sum / loss_mask.shape[-1]
    if mode == "mean-program":
        assert num_program_tokens is not None, "num_program_tokens required for token_reduce='mean-program'"
        return token_sum / num_program_tokens.clamp(min=1)
    raise ValueError(f"Unknown token_reduce mode: {mode!r}")
