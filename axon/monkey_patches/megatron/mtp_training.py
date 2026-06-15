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
# Comprehensive patch for megatron/core/transformer/multi_token_prediction.py
from collections.abc import Callable

import torch
from megatron.core.packed_seq_params import PackedSeqParams
from torch import Tensor

DEBUG = False


def debug_print(*args, **kwargs):
    if DEBUG:
        print("[MTP_PATCH]", *args, **kwargs)


def get_sp_world_size_from_tensors(local_tokens: int, position_ids: Tensor) -> int:
    """
    Infer sequence parallel world size from the ratio of position_ids size to local token count.

    Note: This function tries to infer SP world size, but it can be inaccurate if:
    - position_ids is in padded format while local_tokens is packed (sum of actual lengths)
    - The ratio doesn't cleanly divide into SP world size

    For more reliable SP detection, use megatron's parallel_state directly when available.
    """
    # Try to get SP world size from megatron parallel state first
    try:
        from megatron.core import parallel_state as mpu

        tp_world_size = mpu.get_tensor_model_parallel_world_size()
        # If sequence_parallel is enabled, SP world size equals TP world size
        # This is more reliable than inference from tensor shapes
        if tp_world_size > 1:
            return tp_world_size
    except Exception:
        pass

    # Fallback to inference from tensor shapes
    if position_ids.dim() == 2:
        full_seq_len = position_ids.shape[1]
    elif position_ids.dim() == 3:
        full_seq_len = position_ids.shape[2]
    else:
        return 1

    batch_size = position_ids.shape[0]
    full_tokens_approx = full_seq_len * batch_size

    if local_tokens > 0 and full_tokens_approx > local_tokens:
        ratio = full_tokens_approx / local_tokens
        sp_size = round(ratio)
        if sp_size > 1:
            return sp_size
    return 1


def reconstruct_full_cu_seqlens(
    local_cu_seqlens: Tensor, sp_world_size: int, hidden_states_tokens: int = None
) -> Tensor:
    """
    Reconstruct full cumulative sequence lengths from local cu_seqlens.

    IMPORTANT: In many cases, packed_seq_params.cu_seqlens_q already contains FULL sequence lengths
    (not SP-split), because the packing happens before SP scattering. In this case, we should NOT
    multiply by sp_world_size.

    We detect this by comparing the total tokens in cu_seqlens with the expected local token count
    (from hidden_states after SP split).

    Args:
        local_cu_seqlens: Cumulative sequence lengths tensor
        sp_world_size: Sequence parallel world size
        hidden_states_tokens: Number of tokens in hidden_states (after SP split). If provided,
                              used to detect if cu_seqlens is already full.
    """
    if sp_world_size <= 1:
        return local_cu_seqlens

    cu_seqlens_total = int(local_cu_seqlens[-1].item())

    # If hidden_states_tokens is provided, check if cu_seqlens is already full
    if hidden_states_tokens is not None:
        # If cu_seqlens total roughly matches hidden_states_tokens * sp_world_size,
        # then cu_seqlens is already full (not SP-split)
        expected_full_tokens = hidden_states_tokens * sp_world_size
        if abs(cu_seqlens_total - expected_full_tokens) < sp_world_size:
            # cu_seqlens is already full
            debug_print(
                f"[reconstruct_full_cu_seqlens] cu_seqlens already full: "
                f"cu_seqlens_total={cu_seqlens_total}, expected_full={expected_full_tokens}"
            )
            return local_cu_seqlens
        elif cu_seqlens_total == hidden_states_tokens:
            # cu_seqlens is local (SP-split), need to multiply
            debug_print(f"[reconstruct_full_cu_seqlens] cu_seqlens is local, multiplying by {sp_world_size}")
            pass  # Fall through to multiplication logic
        else:
            debug_print(
                f"[reconstruct_full_cu_seqlens] WARNING: unclear if cu_seqlens is full or local. "
                f"cu_seqlens_total={cu_seqlens_total}, hidden_states_tokens={hidden_states_tokens}, "
                f"expected_full={expected_full_tokens}"
            )

    device = local_cu_seqlens.device
    dtype = local_cu_seqlens.dtype

    local_seqlens = local_cu_seqlens[1:] - local_cu_seqlens[:-1]
    full_seqlens = local_seqlens * sp_world_size

    full_cu_seqlens = torch.zeros(len(local_cu_seqlens), device=device, dtype=dtype)
    full_cu_seqlens[1:] = torch.cumsum(full_seqlens, dim=0)

    return full_cu_seqlens


def pack_position_ids_to_1d(position_ids: Tensor, cu_seqlens: Tensor) -> Tensor:
    batch_size = len(cu_seqlens) - 1
    total_tokens = int(cu_seqlens[-1].item())
    device = position_ids.device
    dtype = position_ids.dtype

    if position_ids.dim() == 3:
        num_sections = position_ids.shape[1]
        output = torch.zeros(1, num_sections, total_tokens, device=device, dtype=dtype)
        for i in range(batch_size):
            start_idx = int(cu_seqlens[i].item())
            end_idx = int(cu_seqlens[i + 1].item())
            actual_len = end_idx - start_idx
            copy_len = min(actual_len, position_ids.shape[2])
            output[0, :, start_idx : start_idx + copy_len] = position_ids[i, :, :copy_len]
            if copy_len < actual_len:
                for j in range(copy_len, actual_len):
                    output[0, :, start_idx + j] = output[0, :, start_idx + copy_len - 1] + (j - copy_len + 1)
        return output
    elif position_ids.dim() == 2:
        output = torch.zeros(1, total_tokens, device=device, dtype=dtype)
        for i in range(batch_size):
            start_idx = int(cu_seqlens[i].item())
            end_idx = int(cu_seqlens[i + 1].item())
            actual_len = end_idx - start_idx
            copy_len = min(actual_len, position_ids.shape[1])
            output[0, start_idx : start_idx + copy_len] = position_ids[i, :copy_len]
            if copy_len < actual_len:
                for j in range(copy_len, actual_len):
                    output[0, start_idx + j] = output[0, start_idx + copy_len - 1] + (j - copy_len + 1)
        return output
    else:
        return position_ids


def roll_packed_position_ids_1d(position_ids: Tensor, cu_seqlens: Tensor, shifts: int = -1) -> Tensor:
    batch_size = len(cu_seqlens) - 1
    output = position_ids.clone()

    if position_ids.dim() == 3:
        for i in range(batch_size):
            start_idx = int(cu_seqlens[i].item())
            end_idx = int(cu_seqlens[i + 1].item())
            if shifts == -1 and end_idx > start_idx + 1:
                output[0, :, start_idx : end_idx - 1] = position_ids[0, :, start_idx + 1 : end_idx]
                output[0, :, end_idx - 1] = position_ids[0, :, end_idx - 1] + 1
    elif position_ids.dim() == 2:
        for i in range(batch_size):
            start_idx = int(cu_seqlens[i].item())
            end_idx = int(cu_seqlens[i + 1].item())
            if shifts == -1 and end_idx > start_idx + 1:
                output[0, start_idx : end_idx - 1] = position_ids[0, start_idx + 1 : end_idx]
                output[0, end_idx - 1] = position_ids[0, end_idx - 1] + 1
    return output


def convert_embedding_output_to_thd(decoder_input: Tensor, cu_seqlens: Tensor, target_dtype: torch.dtype) -> Tensor:
    """
    Convert embedding output to THD (token, head, dim) format for packed sequences.

    For packed sequences:
    - If decoder_input is already [total_tokens, 1, hidden], just return as-is (with dtype conversion)
    - If decoder_input is [seq_len, batch, hidden], convert to [total_tokens, 1, hidden]
    """
    batch_size = len(cu_seqlens) - 1
    total_tokens = int(cu_seqlens[-1].item())
    device = decoder_input.device

    # Check if already in THD format for packed sequences
    # THD format: [total_tokens, 1, hidden] where dim 1 is 1 for packed sequences
    if decoder_input.dim() == 3 and decoder_input.shape[1] == 1:
        # Already in THD format - just ensure dtype and return
        if decoder_input.shape[0] == total_tokens:
            return decoder_input.to(target_dtype)
        else:
            # Shape mismatch - need to handle
            debug_print(
                f"[convert_embedding_output_to_thd] THD format but size mismatch: "
                f"decoder_input.shape[0]={decoder_input.shape[0]}, total_tokens={total_tokens}"
            )
            # Take the first total_tokens if decoder_input is larger, or pad if smaller
            if decoder_input.shape[0] >= total_tokens:
                return decoder_input[:total_tokens].to(target_dtype)
            else:
                # Pad with zeros
                hidden_size = decoder_input.shape[2]
                output = torch.zeros(total_tokens, 1, hidden_size, device=device, dtype=target_dtype)
                output[: decoder_input.shape[0]] = decoder_input.to(target_dtype)
                return output

    # Original logic for non-packed sequences: [seq_len, batch, hidden] -> [total_tokens, 1, hidden]
    seq_len, batch, hidden_size = decoder_input.shape

    output = torch.zeros(total_tokens, hidden_size, device=device, dtype=target_dtype)
    for i in range(batch_size):
        start_idx = int(cu_seqlens[i].item())
        end_idx = int(cu_seqlens[i + 1].item())
        actual_len = end_idx - start_idx
        copy_len = min(actual_len, seq_len)
        # Handle case where batch dim doesn't match batch_size (for packed sequences where batch=1)
        batch_idx = min(i, batch - 1)
        output[start_idx : start_idx + copy_len, :] = decoder_input[:copy_len, batch_idx, :].to(target_dtype)
    return output.unsqueeze(1)


def is_packed_mode(packed_seq_params: PackedSeqParams) -> bool:
    return packed_seq_params is not None and packed_seq_params.cu_seqlens_q is not None


def create_mtp_packed_seq_params(
    original_packed_seq_params: PackedSeqParams,
    full_cu_seqlens: Tensor,
    position_ids_rolled: Tensor,
) -> PackedSeqParams:
    mtp_params = PackedSeqParams(
        cu_seqlens_q=full_cu_seqlens,
        cu_seqlens_kv=full_cu_seqlens,
        qkv_format=original_packed_seq_params.qkv_format,
    )

    seqlens = full_cu_seqlens[1:] - full_cu_seqlens[:-1]
    max_seqlen = int(seqlens.max().item())
    mtp_params.max_seqlen_q = max_seqlen
    mtp_params.max_seqlen_kv = max_seqlen
    mtp_params.position_ids = position_ids_rolled
    mtp_params.skip_sequence_parallel_gather = True

    return mtp_params


def get_model_dtype_from_module(module) -> torch.dtype:
    """Get the model's parameter dtype by checking parameters."""
    try:
        for p in module.parameters():
            if p.dtype in (torch.bfloat16, torch.float16, torch.float32):
                return p.dtype
    except Exception:
        pass
    return torch.bfloat16


def patched_mtp_layer_forward(
    self,
    input_ids: Tensor,
    position_ids: Tensor,
    hidden_states: Tensor,
    attention_mask: Tensor,
    context: Tensor = None,
    context_mask: Tensor = None,
    rotary_pos_emb: Tensor = None,
    rotary_pos_cos: Tensor = None,
    rotary_pos_sin: Tensor = None,
    attention_bias: Tensor = None,
    inference_params=None,
    packed_seq_params: PackedSeqParams = None,
    sequence_len_offset: Tensor = None,
    embedding: Callable = None,
) -> tuple:
    from megatron.core.tensor_parallel.mappings import (
        gather_from_sequence_parallel_region,
        scatter_to_sequence_parallel_region,
    )
    from megatron.core.utils import make_viewless_tensor

    assert context is None, "multi token prediction + cross attention is not yet supported."

    use_packed_mode = is_packed_mode(packed_seq_params)

    if use_packed_mode:
        # Get model dtype from this layer's parameters
        model_dtype = get_model_dtype_from_module(self)

        local_cu_seqlens = packed_seq_params.cu_seqlens_q
        local_total_tokens = int(local_cu_seqlens[-1].item())

        sp_world_size = get_sp_world_size_from_tensors(local_total_tokens, position_ids)
        use_sp = self.sequence_parallel or sp_world_size > 1

        # Get local hidden_states token count BEFORE gathering
        # This is used to detect if cu_seqlens is already full or local
        hidden_states_local_tokens = hidden_states.shape[0]

        debug_print(
            f"[MTP Layer] local_cu_seqlens total={local_total_tokens}, "
            f"hidden_states_local_tokens={hidden_states_local_tokens}, "
            f"sp_world_size={sp_world_size}, use_sp={use_sp}"
        )

        full_cu_seqlens = reconstruct_full_cu_seqlens(
            local_cu_seqlens, sp_world_size, hidden_states_tokens=hidden_states_local_tokens
        )

        # Gather hidden_states and cast to model dtype
        if use_sp:
            hidden_states = gather_from_sequence_parallel_region(hidden_states)
        hidden_states = hidden_states.to(model_dtype)

        debug_print(
            f"[MTP Layer] After gather: hidden_states.shape={hidden_states.shape}, "
            f"full_cu_seqlens[-1]={int(full_cu_seqlens[-1].item())}"
        )

        # Pack and roll position_ids
        position_ids_packed = pack_position_ids_to_1d(position_ids, full_cu_seqlens)
        position_ids_rolled = roll_packed_position_ids_1d(position_ids_packed, full_cu_seqlens, shifts=-1)

        # Create MTP packed_seq_params
        mtp_packed_seq_params = create_mtp_packed_seq_params(packed_seq_params, full_cu_seqlens, position_ids_rolled)

        # Get embeddings
        input_ids_for_emb = torch.roll(input_ids, shifts=-1, dims=-1)
        position_ids_for_emb = torch.roll(position_ids, shifts=-1, dims=-1)

        debug_print(
            f"[MTP Layer] Before embedding: input_ids_for_emb.shape={input_ids_for_emb.shape}, "
            f"position_ids_for_emb.shape={position_ids_for_emb.shape}"
        )

        decoder_input = embedding(input_ids=input_ids_for_emb, position_ids=position_ids_for_emb)

        debug_print(f"[MTP Layer] After embedding: decoder_input.shape={decoder_input.shape}")

        # Gather decoder_input only if it was scattered by embedding
        # For packed sequences, embedding might not scatter if input_ids is already in packed format
        if use_sp:
            # Check if decoder_input needs gathering (if it's SP-split, dim 0 would be local_tokens)
            # Compare with hidden_states dim 0 to see if they match
            if decoder_input.shape[0] != hidden_states.shape[0]:
                debug_print(f"[MTP Layer] Gathering decoder_input: before shape={decoder_input.shape}")
                decoder_input = gather_from_sequence_parallel_region(decoder_input)
                debug_print(f"[MTP Layer] After gather: decoder_input.shape={decoder_input.shape}")
            else:
                debug_print("[MTP Layer] Skipping gather - decoder_input already matches hidden_states dim 0")

        # Convert to THD format with model dtype
        debug_print(
            f"[MTP Layer] Before convert_embedding_output_to_thd: decoder_input.shape={decoder_input.shape}, "
            f"full_cu_seqlens={full_cu_seqlens}"
        )
        decoder_input = convert_embedding_output_to_thd(decoder_input, full_cu_seqlens, target_dtype=model_dtype)
        debug_print(f"[MTP Layer] After convert: decoder_input.shape={decoder_input.shape}")

        # Debug logging for shape mismatch
        if hidden_states.shape[0] != decoder_input.shape[0]:
            debug_print("[MTP Layer] Shape mismatch!")
            debug_print(f"  hidden_states.shape: {hidden_states.shape}")
            debug_print(f"  decoder_input.shape: {decoder_input.shape}")
            debug_print(f"  input_ids.shape: {input_ids.shape}")
            debug_print(f"  position_ids.shape: {position_ids.shape}")
            debug_print(f"  local_cu_seqlens: {local_cu_seqlens}")
            debug_print(f"  full_cu_seqlens: {full_cu_seqlens}")
            debug_print(f"  local_total_tokens: {local_total_tokens}")
            debug_print(f"  sp_world_size: {sp_world_size}")
            debug_print(f"  use_sp: {use_sp}")

        assert hidden_states.shape[0] == decoder_input.shape[0], (
            f"MTP Layer shape mismatch: hidden_states.shape[0]={hidden_states.shape[0]} != "
            f"decoder_input.shape[0]={decoder_input.shape[0]}. "
            f"hidden_states.shape={hidden_states.shape}, decoder_input.shape={decoder_input.shape}"
        )

        hidden_states = make_viewless_tensor(inp=hidden_states, requires_grad=True, keep_graph=True)

        # Call transformer layer
        hidden_states_out = self._proj_and_transformer_layer(
            hidden_states=hidden_states,
            decoder_input=decoder_input,
            attention_mask=attention_mask,
            context=context,
            context_mask=context_mask,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            attention_bias=attention_bias,
            inference_params=inference_params,
            packed_seq_params=mtp_packed_seq_params,
            sequence_len_offset=sequence_len_offset,
        )

        # Force output to model dtype
        hidden_states_out = hidden_states_out.to(model_dtype)

        # Scatter output
        if use_sp:
            hidden_states_out = scatter_to_sequence_parallel_region(hidden_states_out)
            hidden_states_out = hidden_states_out.to(model_dtype)

        return hidden_states_out, input_ids, position_ids

    else:
        # Original non-packed behavior
        input_ids, position_ids, decoder_input, hidden_states = self._get_embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            embedding=embedding,
            hidden_states=hidden_states,
            packed_seq_params=packed_seq_params,
        )

        if self.config.recompute_granularity == "full" and self.training:
            hidden_states = self._checkpointed_forward(
                self._proj_and_transformer_layer,
                hidden_states=hidden_states,
                decoder_input=decoder_input,
                attention_mask=attention_mask,
                context=context,
                context_mask=context_mask,
                rotary_pos_emb=rotary_pos_emb,
                rotary_pos_cos=rotary_pos_cos,
                rotary_pos_sin=rotary_pos_sin,
                attention_bias=attention_bias,
                inference_params=inference_params,
                packed_seq_params=packed_seq_params,
                sequence_len_offset=sequence_len_offset,
            )
        else:
            hidden_states = self._proj_and_transformer_layer(
                hidden_states=hidden_states,
                decoder_input=decoder_input,
                attention_mask=attention_mask,
                context=context,
                context_mask=context_mask,
                rotary_pos_emb=rotary_pos_emb,
                rotary_pos_cos=rotary_pos_cos,
                rotary_pos_sin=rotary_pos_sin,
                attention_bias=attention_bias,
                inference_params=inference_params,
                packed_seq_params=packed_seq_params,
                sequence_len_offset=sequence_len_offset,
            )

        return hidden_states, input_ids, position_ids


def patched_mtp_block_forward(
    self,
    input_ids: Tensor,
    position_ids: Tensor,
    hidden_states: Tensor,
    attention_mask: Tensor,
    context: Tensor = None,
    context_mask: Tensor = None,
    rotary_pos_emb: Tensor = None,
    rotary_pos_cos: Tensor = None,
    rotary_pos_sin: Tensor = None,
    attention_bias: Tensor = None,
    inference_params=None,
    packed_seq_params: PackedSeqParams = None,
    sequence_len_offset: Tensor = None,
    extra_block_kwargs: dict = None,
    embedding=None,
) -> Tensor:
    from megatron.core.transformer.multi_token_prediction import get_mtp_layer_offset

    # Clear MoE routing context before MTP forward to prevent MTP layers from
    # incorrectly using decoder layer routing maps. MTP layer's layer_number can
    # conflict with decoder layer numbers (MTP uses layer_number=1+offset which
    # matches decoder layers). By clearing the context, MTP MoE layers will
    # compute fresh routing instead of replaying decoder routing decisions.
    try:
        from axon.monkey_patches.megatron.moe_replay import set_moe_routing_context

        set_moe_routing_context(None)
    except ImportError:
        pass  # MoE replay not available

    # Get model dtype from first layer's parameters
    model_dtype = torch.bfloat16  # default
    if len(self.layers) > 0:
        model_dtype = get_model_dtype_from_module(self.layers[0])

    debug_print(f"MTP Block: input_dtype={hidden_states.dtype}, model_dtype={model_dtype}")

    # CRITICAL FIX: Cast hidden_states to model dtype BEFORE chunking
    # This ensures ALL chunks have the correct dtype from the start
    if hidden_states.dtype != model_dtype:
        hidden_states = hidden_states.to(model_dtype)

    offset = get_mtp_layer_offset(self.config)
    hidden_states_list = list(torch.chunk(hidden_states, 1 + offset, dim=0))
    current_hidden_states = hidden_states_list[offset]

    for layer_number in range(len(self.layers)):
        (current_hidden_states, input_ids, position_ids) = self.layers[layer_number](
            input_ids=input_ids,
            position_ids=position_ids,
            hidden_states=current_hidden_states,
            attention_mask=attention_mask,
            inference_params=inference_params,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            packed_seq_params=packed_seq_params,
            sequence_len_offset=sequence_len_offset,
            embedding=embedding,
            **(extra_block_kwargs or {}),
        )
        # Ensure layer output has correct dtype
        if current_hidden_states.dtype != model_dtype:
            current_hidden_states = current_hidden_states.to(model_dtype)
        hidden_states_list.append(current_hidden_states)

    result = torch.cat(hidden_states_list, dim=0)

    # Final dtype check
    if result.dtype != model_dtype:
        result = result.to(model_dtype)

    debug_print(f"MTP Block: output_dtype={result.dtype}")
    return result


def patch_mtp_for_packed_sequences():
    from megatron.core.transformer.multi_token_prediction import (
        MultiTokenPredictionBlock,
        MultiTokenPredictionLayer,
    )

    if not hasattr(MultiTokenPredictionBlock, "_original_forward"):
        MultiTokenPredictionBlock._original_forward = MultiTokenPredictionBlock.forward
    MultiTokenPredictionBlock.forward = patched_mtp_block_forward

    if not hasattr(MultiTokenPredictionLayer, "_original_forward"):
        MultiTokenPredictionLayer._original_forward = MultiTokenPredictionLayer.forward
    MultiTokenPredictionLayer.forward = patched_mtp_layer_forward

    # Apply Context Parallelism patches
    patch_roll_tensor()
    patch_mtp_get_embedding()

    print("MTP packed sequence patch applied (cast ALL chunks at start)")


def unpatch_mtp():
    from megatron.core.transformer import multi_token_prediction as mtp_module
    from megatron.core.transformer.multi_token_prediction import (
        MultiTokenPredictionBlock,
        MultiTokenPredictionLayer,
    )

    if hasattr(MultiTokenPredictionBlock, "_original_forward"):
        MultiTokenPredictionBlock.forward = MultiTokenPredictionBlock._original_forward
    if hasattr(MultiTokenPredictionLayer, "_original_forward"):
        MultiTokenPredictionLayer.forward = MultiTokenPredictionLayer._original_forward

    # Restore roll_tensor
    if hasattr(mtp_module, "_original_roll_tensor"):
        mtp_module.roll_tensor = mtp_module._original_roll_tensor

    # Restore _get_embeddings
    if hasattr(MultiTokenPredictionLayer, "_original_get_embedding"):
        MultiTokenPredictionLayer._get_embeddings = MultiTokenPredictionLayer._original_get_embedding

    print("MTP patch removed")


# =============================================================================
# Context Parallelism Support for MTP
# =============================================================================


def roll_tensor(tensor, shifts=-1, dims=-1, cp_group=None, packed_seq_params=None):
    """Roll the tensor input along the sequence dimension with Context Parallelism (CP) support.

    This function extends the original roll_tensor to support Context Parallelism, which allows
    MTP to work with CP > 1. When CP is enabled, the sequence dimension is split across CP ranks,
    and tensor rolling requires communication between adjacent CP ranks to properly handle the
    boundary conditions.

    For CP=1 (default behavior): Uses standard torch.roll with zero padding
    For CP>1: Splits tensor into chunks, performs rolling within each chunk, then exchanges
    boundary elements between adjacent CP ranks to maintain sequence continuity.

    For packed sequences: Respects sequence boundaries when rolling to avoid mixing tokens
    from different sequences.

    Args:
        tensor (Tensor): The input tensor to roll.
        shifts (int): The shift of the tensor (typically -1 for MTP).
        dims (int): The dimension to roll (typically -1 for sequence dimension).
        cp_group (ProcessGroup): The context parallelism process group. If None or size=1,
                               falls back to standard rolling behavior.
        packed_seq_params (PackedSeqParams): Parameters for packed sequence processing.
                                            If provided, respects sequence boundaries.
    Returns:
        tuple: (rolled_tensor, sum_of_rolled_tensor)
    """
    # Standard rolling behavior when CP is not enabled (cp_group is None or size=1)
    if cp_group is None or cp_group.size() == 1:
        rolled_tensor = torch.roll(tensor, shifts=shifts, dims=dims)

        # Handle packed sequences: zero out positions at sequence boundaries
        if packed_seq_params is not None and packed_seq_params.cu_seqlens_q is not None:
            # cu_seqlens_q contains cumulative sequence lengths [0, len1, len1+len2, ...]
            cu_seqlens = packed_seq_params.cu_seqlens_q
            # For each sequence boundary, zero out the position that would cross the boundary
            for i in range(1, len(cu_seqlens)):
                seq_end = cu_seqlens[i]
                if shifts < 0:
                    # Rolling left (shifts=-1): zero out the position just before the boundary
                    # This prevents the first token of the next sequence from appearing at the end
                    if seq_end - 1 < rolled_tensor.size(dims) and seq_end > 0:
                        rolled_tensor.select(dims, seq_end - 1).fill_(0)
                else:
                    # Rolling right: zero out the position at the boundary
                    if seq_end < rolled_tensor.size(dims):
                        rolled_tensor.select(dims, seq_end).fill_(0)
        else:
            # For non-packed sequences, just zero out the boundary position
            rolled_tensor.select(dims, shifts).fill_(0)

        return rolled_tensor, rolled_tensor.sum()

    # CP-enabled rolling: Split tensor into chunks and handle boundary communication
    # This matches the batch splitting logic in get_batch_on_this_cp_rank() function

    # Note: When using packed sequences with CP, we need to be careful about sequence boundaries
    # The current implementation handles CP boundaries but may need additional logic for
    # packed sequence boundaries within CP chunks
    if packed_seq_params is not None:
        import warnings

        warnings.warn(
            "Using packed sequences with Context Parallelism (CP > 1) in MTP. "
            "Ensure sequence boundaries are properly handled within CP chunks.",
            stacklevel=2,
        )

    tensor_list = tensor.chunk(2, dim=dims)
    rolled_tensor_list = []
    for i in range(len(tensor_list)):
        rolled_tensor_list.append(torch.roll(tensor_list[i], shifts=shifts, dims=dims))

    # Prepare tensors for communication between CP ranks
    # Each CP rank needs to send boundary elements to adjacent ranks
    tensor_send_list = []
    tensor_recv_list = []
    for i in range(len(rolled_tensor_list)):
        tensor_send_list.append(rolled_tensor_list[i].select(dims, shifts).contiguous())
        empty_tensor = torch.empty(
            tensor_send_list[i].shape,
            dtype=tensor_send_list[i].dtype,
            device=tensor_send_list[i].device,
        )
        tensor_recv_list.append(empty_tensor)

    # Get the global rank of next and prev process in the cp group
    global_ranks = torch.distributed.get_process_group_ranks(group=cp_group)
    local_rank = torch.distributed.get_rank(group=cp_group)
    next_rank = global_ranks[(local_rank + 1) % len(global_ranks)]
    prev_rank = global_ranks[(local_rank - 1) % len(global_ranks)]

    # Start send and recv ops
    ops = []
    if local_rank != 0:
        req_send_first_part = torch.distributed.isend(tensor=tensor_send_list[0], dst=prev_rank)
        ops.append(req_send_first_part)
        req_recv_second_part = torch.distributed.irecv(tensor=tensor_recv_list[1], src=prev_rank)
        ops.append(req_recv_second_part)
    else:
        # Inserted elements are set to be 0.0.
        tensor_recv_list[1] = 0
    if local_rank != len(global_ranks) - 1:
        req_recv_first_part = torch.distributed.irecv(tensor=tensor_recv_list[0], src=next_rank)
        ops.append(req_recv_first_part)
        req_send_second_part = torch.distributed.isend(tensor=tensor_send_list[1], dst=next_rank)
        ops.append(req_send_second_part)
    else:
        # For the last CP rank, the removed elements of second part go into the first part
        tensor_recv_list[0] = tensor_send_list[1]

    # Wait for all communication operations to complete
    for op in ops:
        op.wait()

    # Splicing: Replace boundary elements with received elements from adjacent ranks
    # This ensures proper sequence continuity across CP boundaries
    index = [slice(None)] * rolled_tensor_list[0].dim()
    index[dims] = shifts
    for i in range(len(rolled_tensor_list)):
        rolled_tensor_list[i][tuple(index)] = tensor_recv_list[i]

    # Concatenate the processed chunks back into a single tensor
    rolled_tensor = torch.cat(rolled_tensor_list, dim=dims)

    return rolled_tensor, rolled_tensor.sum()


def _get_embeddings(
    self,
    input_ids: torch.Tensor,
    position_ids: torch.Tensor,
    embedding,  # Callable
    hidden_states: torch.Tensor,
    packed_seq_params=None,  # Optional[PackedSeqParams]
):
    """
    Preprocesses input data for the Multi-Token Prediction (MTP) layers.

    This function computes the decoder input and sends updated input_ids and position_ids to
    the next layer.

    Args:
        input_ids (torch.Tensor): The input token IDs.
        position_ids (torch.Tensor): The position IDs corresponding to the input tokens.
        embedding (Callable): The embedding module from gpt model to compute the decoder input.
        hidden_states (torch.Tensor): hidden states tensor of shape [s, b, h] where s is the
            sequence length, b is the batch size, and h is the hidden size.
        packed_seq_params (PackedSeqParams): Parameters for packed sequence processing.
    """
    # Calc logits for the current Multi-Token Prediction (MTP) layers.
    assert packed_seq_params is not None, "packed_seq_params is required for MTP layers"
    input_ids, _ = roll_tensor(
        input_ids, shifts=-1, dims=-1, cp_group=self.cp_group, packed_seq_params=packed_seq_params
    )
    position_ids, _ = roll_tensor(
        position_ids, shifts=-1, dims=-1, cp_group=self.cp_group, packed_seq_params=packed_seq_params
    )
    # embedding
    decoder_input = embedding(input_ids=input_ids, position_ids=position_ids)

    # Import make_viewless_tensor
    try:
        from megatron.core.tensor_parallel.utils import make_viewless_tensor
    except ImportError:
        # Fallback if make_viewless_tensor is not available
        def make_viewless_tensor(inp, requires_grad=True, keep_graph=True):
            return inp

    hidden_states = make_viewless_tensor(inp=hidden_states, requires_grad=True, keep_graph=True)

    return input_ids, position_ids, decoder_input, hidden_states


def patch_roll_tensor():
    """Patch the roll_tensor function to support Context Parallelism."""
    try:
        from megatron.core.transformer import multi_token_prediction as mtp_module

        # Store original function
        if not hasattr(mtp_module, "_original_roll_tensor"):
            mtp_module._original_roll_tensor = mtp_module.roll_tensor

        # Apply patch
        mtp_module.roll_tensor = roll_tensor
        print("[MTP_PATCH] roll_tensor patched for Context Parallelism support", flush=True)
    except ImportError:
        print(
            "[MTP_PATCH] Warning: Could not import megatron.core.transformer.multi_token_prediction, skipping roll_tensor patch",
            flush=True,
        )
    except AttributeError:
        print("[MTP_PATCH] Warning: roll_tensor not found in multi_token_prediction module", flush=True)


def patch_mtp_get_embedding():
    """Patch the MTP layer's get_embedding method to support packed sequences and CP."""
    try:
        from megatron.core.transformer.multi_token_prediction import MultiTokenPredictionLayer

        # Store original method
        if not hasattr(MultiTokenPredictionLayer, "_original_get_embedding"):
            MultiTokenPredictionLayer._original_get_embedding = MultiTokenPredictionLayer._get_embeddings

        # Apply patch
        MultiTokenPredictionLayer._get_embeddings = _get_embeddings
        print(
            "[MTP_PATCH] MultiTokenPredictionLayer._get_embeddings patched for packed sequences and CP support",
            flush=True,
        )
    except ImportError:
        print(
            "[MTP_PATCH] Warning: Could not import megatron.core.transformer.multi_token_prediction, skipping MTP layer patch",
            flush=True,
        )
    except Exception as e:
        print(f"[MTP_PATCH] Warning: Could not patch MTP layer _get_embeddings: {e}", flush=True)
