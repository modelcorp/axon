# Copyright 2025 Model AI Corp
# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
Utilities for computing log-probabilities and entropy from a single HF model
forward pass, with support for remove-padding and Ulysses sequence parallelism.
"""

import torch

import axon.utils.torch.ops as axon_F
from axon.utils.torch.attention import index_first_axis, pad_input, rearrange, unpad_input
from axon.utils.torch.ops import logprobs_from_logits
from axon.utils.ulysses import gather_outputs_and_unpad, ulysses_pad, ulysses_pad_and_slice_inputs

# Cache for compiled/configured entropy functions keyed by (use_chunking, use_compile)
_ENTROPY_FN_CACHE: dict[tuple[bool, bool], callable] = {}

# Module-level storage for cu_seqlens so recurrent layers (GDN) can access
# sequence boundaries when use_remove_padding=True (packed sequences).
# Set by fsdp_model_forward(), must persist through backward for gradient
# checkpointing recomputation. Uses a plain global (not thread-local) because
# PyTorch's autograd engine may run recomputation on different threads.
_current_cu_seqlens: torch.Tensor | None = None


def get_rmpad_cu_seqlens() -> torch.Tensor | None:
    """Get cu_seqlens for the current forward pass (set by fsdp_model_forward)."""
    return _current_cu_seqlens


def _get_entropy_fn(use_chunking: bool, use_compile: bool) -> callable:
    """Return a (possibly compiled) entropy function, cached across calls."""
    key = (use_chunking, use_compile)
    if key not in _ENTROPY_FN_CACHE:
        fn = axon_F.entropy_from_logits_with_chunking if use_chunking else axon_F.entropy_from_logits
        _ENTROPY_FN_CACHE[key] = torch.compile(fn, dynamic=True) if use_compile else fn
    return _ENTROPY_FN_CACHE[key]


def _has_internal_ulysses_sp_slicing(module: torch.nn.Module) -> bool:
    """Return True if any submodule's class is marked as having internal
    Ulysses SP input slicing (set by ``patch_vlm_for_ulysses_input_slicing``).

    Such models slice ``inputs_embeds`` / ``position_ids`` inside their own
    forward, so the external ``prepare_rmpad_inputs`` must NOT slice again —
    that would double-slice and break shapes.  For multimodal-architecture
    models that do NOT have the internal wrapper (e.g. text-only training of
    Gemma4-E2B/E4B), the external slice path is correct and required.

    Result is cached on the (unwrapped) module to avoid re-walking submodules.
    """
    inner = getattr(module, "module", module)
    cached = getattr(inner, "_axon_has_internal_ulysses_sp_cached", None)
    if cached is not None:
        return cached
    has_it = any(getattr(type(m), "_axon_ulysses_internal_slice", False) for m in inner.modules())
    try:
        inner._axon_has_internal_ulysses_sp_cached = has_it
    except (AttributeError, TypeError):
        # Some FSDP wrappers may forbid setattr; fall through to recompute
        # next time — correctness is preserved, only the cache is lost.
        pass
    return has_it


def _maybe_checkpoint(fn, tensor, use_checkpointing):
    if use_checkpointing:
        # use_reentrant=False avoids corrupting autograd state when this runs
        # inside an already-checkpointed region (e.g. activation offloading).
        return torch.utils.checkpoint.checkpoint(fn, tensor, use_reentrant=False)
    return fn(tensor)


def unpad_inputs(input_ids, attention_mask, position_ids):
    """Unpad inputs and handle position_ids reshaping.

    Returns ``(input_ids_rmpad, position_ids_rmpad, indices, cu_seqlens)``.
    ``input_ids_rmpad`` has shape ``(1, total_nnz)``.
    """
    input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)
    input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

    if position_ids.dim() == 3:
        position_ids_rmpad = (
            index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices).transpose(0, 1).unsqueeze(1)
        )
    else:
        position_ids_rmpad = index_first_axis(
            rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
        ).transpose(0, 1)

    return input_ids_rmpad, position_ids_rmpad, indices, cu_seqlens


def prepare_rmpad_inputs(
    input_ids, attention_mask, position_ids, multi_modal_inputs, ulysses_sp_size, use_ulysses_sp, module
):
    """Unpad inputs, handle all-zero masks, and apply Ulysses SP padding/slicing.

    Returns (fwd_ids, fwd_pos, rolled_labels, indices, is_mask_all_zero, pad_size, multi_modal_inputs).
    """
    input_ids_rmpad, position_ids_rmpad, indices, cu_seqlens = unpad_inputs(input_ids, attention_mask, position_ids)

    is_mask_all_zero = attention_mask.sum() == 0
    if is_mask_all_zero:
        input_ids_rmpad = torch.zeros((1, ulysses_sp_size), device=input_ids.device, dtype=input_ids.dtype)
        pos_shape = (position_ids.shape[0], 1, ulysses_sp_size) if position_ids.dim() == 3 else (1, ulysses_sp_size)
        position_ids_rmpad = torch.zeros(pos_shape, device=position_ids.device, dtype=position_ids.dtype)

    if "image_bound" in multi_modal_inputs:
        from axon.utils.vision_utils import process_multi_modal_inputs_for_minicpmo

        multi_modal_inputs = process_multi_modal_inputs_for_minicpmo(
            input_ids, attention_mask, position_ids, cu_seqlens, multi_modal_inputs
        )

    rolled_labels = torch.roll(input_ids_rmpad, shifts=-1, dims=1)
    pad_size = 0
    if use_ulysses_sp:
        # Use ulysses_pad (no external slice) only for models that slice inputs
        # internally in their forward (qwen2_vl / qwen3_vl / glm4v).  Models
        # whose class is multimodal-shaped but currently called with text-only
        # inputs (e.g. Gemma4-E2B/E4B for text-only RL training) take the
        # standard slice path.
        has_internal_slice = _has_internal_ulysses_sp_slicing(module)
        pad_fn = ulysses_pad if has_internal_slice else ulysses_pad_and_slice_inputs
        input_ids_rmpad, position_ids_rmpad, pad_size = pad_fn(
            input_ids_rmpad,
            position_ids_rmpad=position_ids_rmpad,
            sp_size=ulysses_sp_size,
        )
        rolled_labels, _, _ = ulysses_pad_and_slice_inputs(
            rolled_labels,
            position_ids_rmpad=None,
            sp_size=ulysses_sp_size,
        )

    return (
        input_ids_rmpad,
        position_ids_rmpad,
        rolled_labels.squeeze(0),
        indices,
        is_mask_all_zero,
        pad_size,
        multi_modal_inputs,
        cu_seqlens,
    )


def rmpad_postprocess(t, *, use_ulysses_sp, pad_size, is_mask_all_zero, indices, batch_size, seqlen):
    """Gather from Ulysses SP, handle all-zero masks, pad back to batch dims, and align per-token logprobs.

    After ``pad_input``, ``full[b, t]`` holds the model output at position *t* which
    predicts **token t+1** (autoregressive shift).  We re-align so the returned
    tensor has ``result[b, t] = logprob of token t`` (given context 0..t-1),
    with ``result[b, 0] = 0`` (no prediction for the first token).
    """
    if t is None:
        return None
    if use_ulysses_sp:
        t = gather_outputs_and_unpad(t, gather_dim=0, unpad_dim=0, padding_size=pad_size)
    if is_mask_all_zero:
        t = t[:0]
    if t.dim() == 1:
        t = t.unsqueeze(-1)
    full = pad_input(t, indices=indices, batch=batch_size, seqlen=seqlen).squeeze(-1)
    # Shift: full[b, t] predicts token t+1 → result[b, t] = logprob of token t
    return torch.nn.functional.pad(full[:, :-1], (1, 0), value=0.0)


def fsdp_model_forward(
    module: torch.nn.Module,
    micro_batch: dict,
    temperature: float,
    calculate_entropy: bool = False,
    *,
    use_remove_padding: bool,
    use_fused_kernels: bool,
    ulysses_sp_size: int,
    device_name: str,
    param_dtype: torch.dtype,
    entropy_checkpointing: bool,
    entropy_from_logits_with_chunking: bool = False,
    use_torch_compile: bool = True,
) -> tuple[torch.Tensor | None, torch.Tensor]:
    """Forward one micro-batch through an HF model, returning ``(entropy, log_probs)``.

    Both outputs have shape ``(batch_size, seq_length)`` where ``seq_length``
    equals ``input_ids.size(-1)``.  Position 0 is always 0 (no prediction for
    the first token); position *t* holds log P(token_t | context_0..t-1).
    *entropy* is ``None`` when *calculate_entropy* is ``False``.
    """
    compute_entropy_from_logits = _get_entropy_fn(entropy_from_logits_with_chunking, use_torch_compile)
    use_ulysses_sp = ulysses_sp_size > 1

    multi_modal_inputs = {}
    if "multi_modal_inputs" in micro_batch:
        from axon.utils.hf_model import extract_multi_modal_inputs

        multi_modal_inputs = extract_multi_modal_inputs(micro_batch["multi_modal_inputs"])

    with torch.autocast(device_type=device_name, dtype=param_dtype):
        input_ids = micro_batch["input_ids"]
        batch_size, seqlen = input_ids.shape
        attention_mask = micro_batch["attention_mask"]
        position_ids = micro_batch["position_ids"]
        entropy = None

        if position_ids.dim() == 3:  # qwen2vl mrope edgecase
            position_ids = position_ids.transpose(0, 1)

        # Prepare inputs
        cu_seqlens = None
        if use_remove_padding:
            (fwd_ids, fwd_pos, rolled_labels, indices, is_mask_all_zero, pad_size, multi_modal_inputs, cu_seqlens) = (
                prepare_rmpad_inputs(
                    input_ids,
                    attention_mask,
                    position_ids,
                    multi_modal_inputs,
                    ulysses_sp_size,
                    use_ulysses_sp,
                    module,
                )
            )
            fwd_mask = None
        else:
            indices, pad_size, is_mask_all_zero, rolled_labels = None, 0, False, None
            fwd_ids, fwd_mask, fwd_pos = input_ids, attention_mask, position_ids

        # Set cu_seqlens in module-level global so GDN layers can access
        # sequence boundaries for packed sequences. Must persist through
        # backward (for gradient checkpointing recomputation).
        global _current_cu_seqlens
        _current_cu_seqlens = cu_seqlens

        extra_args = {"temperature": temperature, "return_dict": True} if use_fused_kernels else {}
        # Under Ulysses SP, the model's internal ``torch.roll(input_ids, -1)``
        # wraps each rank's slice to its OWN first token rather than the next
        # rank's first token — wrong labels at the K-1 SP boundary positions.
        # ``prepare_rmpad_inputs`` rolls BEFORE slicing (correct), so pass that
        # rolled tensor through to ``forward_with_*_backend`` to override the
        # internal roll. Always pass when fused kernels + remove-padding so the
        # SP=1 and SP>1 paths share one code path.
        if use_fused_kernels and use_remove_padding and rolled_labels is not None:
            extra_args["pre_rolled_labels"] = rolled_labels.unsqueeze(0)
        output = module(
            input_ids=fwd_ids,
            attention_mask=fwd_mask,
            position_ids=fwd_pos,
            **multi_modal_inputs,
            use_cache=False,
            **extra_args,
        )

        _F_pad = torch.nn.functional.pad

        # Extract log_probs & entropy.
        # Model outputs at position t predict token t+1 (autoregressive).
        # We re-align so result[b, t] = log P(token_t | context_0..t-1),
        # with result[b, 0] = 0 (no prediction for the first token).
        if use_fused_kernels:
            log_probs = output.log_probs
            entropy = output.entropy if calculate_entropy else None
            if use_remove_padding:
                log_probs = log_probs.squeeze(0)
                entropy = entropy.squeeze(0) if entropy is not None else None
            else:
                # Fused kernel: output[b, t] predicts token t+1 → shift
                log_probs = _F_pad(log_probs[:, :-1], (1, 0), value=0.0)
                entropy = _F_pad(entropy[:, :-1], (1, 0), value=0.0) if entropy is not None else None

        elif use_remove_padding:
            logits = output.logits.squeeze(0)
            logits.div_(temperature)
            log_probs = logprobs_from_logits(logits, rolled_labels, inplace_backward=not calculate_entropy)
            if calculate_entropy:
                entropy = _maybe_checkpoint(compute_entropy_from_logits, logits, entropy_checkpointing)

        else:  # padded, non-fused
            logits = output.logits
            logits.div_(temperature)
            # logits[:, t] predicts token t+1 → use positions :-1 with labels shifted by 1
            logits = logits[:, :-1, :]
            labels = micro_batch["input_ids"][:, 1:]
            log_probs = _F_pad(logprobs_from_logits(logits, labels), (1, 0), value=0.0)
            if calculate_entropy:
                ent = _maybe_checkpoint(compute_entropy_from_logits, logits, entropy_checkpointing)
                entropy = _F_pad(ent, (1, 0), value=0.0)

        # rmpad post-processing: gather SP, handle zeros, pad back, align per-token
        if use_remove_padding:
            pp_kwargs = dict(
                use_ulysses_sp=use_ulysses_sp,
                pad_size=pad_size,
                is_mask_all_zero=is_mask_all_zero,
                indices=indices,
                batch_size=batch_size,
                seqlen=seqlen,
            )
            log_probs = rmpad_postprocess(log_probs, **pp_kwargs)
            entropy = rmpad_postprocess(entropy, **pp_kwargs)

        return entropy, log_probs
