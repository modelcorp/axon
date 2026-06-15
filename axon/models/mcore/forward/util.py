# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

import math

import torch
from megatron.core import parallel_state as mpu
from megatron.core.packed_seq_params import PackedSeqParams

from axon.utils.hf_model import CausalLMOutputForPPO


def validate_moe_routermap(
    layer_map: torch.Tensor,
    attention_mask: torch.Tensor,
    debug: bool = False,
    max_prompt_length: int = None,
) -> None:
    """
    Validate moe_routermap data consistency with attention_mask.

    Expected data layout (sequence with no separate prompt padding):
    - Sequence: [prompt][response][right_pad]
    - Attention mask: 0 for padding, 1 for valid tokens
    - moe_routermap: -1 for padding tokens, valid values for non-padding tokens
    - EXCEPTION: The LAST non-padding token should have -1 (vLLM returns n-1 routermaps)

    Rules:
    1. For non-padding tokens (except the last non-padding token): moe_routermap should NOT be all -1
    2. For padding tokens: moe_routermap should be all -1
    3. For the last non-padding token: moe_routermap should be all -1 (vLLM returns n-1 routermaps)

    Args:
        layer_map: tensor of shape [batch_size, seq_len, n_experts] - routermap for ONE layer
        attention_mask: tensor of shape [batch_size, seq_len]
        debug: if True, print debug information
        max_prompt_length: optional, prompt token count for debug info (derived from response_mask if needed)
    """
    batch_size, seq_len = attention_mask.shape
    assert layer_map.shape[0] == batch_size, f"batch size mismatch: {layer_map.shape[0]} vs {batch_size}"
    assert layer_map.shape[1] == seq_len, f"seq_len mismatch: {layer_map.shape[1]} vs {seq_len}"

    # Precompute which tokens have all -1 routermap for entire batch
    all_neg1_mask = (layer_map == -1).all(dim=-1)  # [batch_size, seq_len]
    mask = attention_mask.bool()

    for i in range(batch_size):
        non_pad_indices = torch.where(mask[i])[0]
        if len(non_pad_indices) == 0:
            continue

        last_non_pad_idx = non_pad_indices[-1].item()

        if debug:
            print(
                f"[MoE Routermap Validation] Batch {i}: "
                f"seq_len={seq_len}, total_non_pad={len(non_pad_indices)}, "
                f"total_pad={seq_len - len(non_pad_indices)}, "
                f"first_non_pad_idx={non_pad_indices[0].item()}, last_non_pad_idx={last_non_pad_idx}"
            )
            if max_prompt_length:
                print(f"  max_prompt_length={max_prompt_length}, response part starts at {max_prompt_length}")

        # Check 1: Non-padding tokens (except last) should NOT have all -1
        if len(non_pad_indices) > 1:
            non_last_mask = mask[i].clone()
            non_last_mask[last_non_pad_idx] = False
            problematic_non_pad = torch.where(non_last_mask & all_neg1_mask[i])[0].tolist()

            if problematic_non_pad:
                if debug:
                    _print_check1_debug(
                        i, problematic_non_pad, layer_map, attention_mask, last_non_pad_idx, seq_len, max_prompt_length
                    )
                raise AssertionError(
                    f"MoE Routermap Validation Failed: batch {i}, {len(problematic_non_pad)} non-padding tokens "
                    f"(excluding last) have all -1 routermap. First bad: {problematic_non_pad[0]}"
                )

        # Check 2: Padding tokens should have all -1
        problematic_pad = torch.where(~mask[i] & ~all_neg1_mask[i])[0].tolist()
        if problematic_pad:
            if debug:
                print(
                    f"[MoE Routermap Validation FAIL] batch {i}: "
                    f"{len(problematic_pad)} padding tokens don't have all -1 routermap"
                )
                print(f"  - First bad padding token: {problematic_pad[0]}")
                print(f"  - Values: {layer_map[i, problematic_pad[0]].tolist()}")
            raise AssertionError(
                f"MoE Routermap Validation Failed: batch {i}, {len(problematic_pad)} padding tokens "
                f"don't have all -1 routermap. First bad: {problematic_pad[0]}"
            )

        # Check 3: Last non-padding token should have all -1
        if not all_neg1_mask[i, last_non_pad_idx].item():
            if debug:
                print(
                    f"[MoE Routermap Validation FAIL] batch {i}, last non-pad token {last_non_pad_idx}: "
                    f"should be all -1 but got {layer_map[i, last_non_pad_idx].tolist()}"
                )
            raise AssertionError(
                f"MoE Routermap Validation Failed: batch {i}, last non-padding token {last_non_pad_idx} "
                f"should have all -1 routermap but got {layer_map[i, last_non_pad_idx].tolist()}"
            )

    if debug:
        print(f"[MoE Routermap Validation] PASSED for batch_size={batch_size}, seq_len={seq_len}")


def _print_check1_debug(
    batch_idx, problematic_non_pad, layer_map, attention_mask, last_non_pad_idx, seq_len, max_prompt_length
):
    """Helper to print debug info for Check 1 failures."""
    num_bad = len(problematic_non_pad)
    first_bad, last_bad = problematic_non_pad[0], problematic_non_pad[-1]

    print(
        f"[MoE Routermap Validation FAIL] batch {batch_idx}: "
        f"{num_bad} non-padding tokens (not last) have all -1 routermap"
    )
    print(f"  - First bad token: {first_bad}, Last bad token: {last_bad}")
    print(f"  - First 20 bad indices: {problematic_non_pad[:20]}")

    if num_bad > 1:
        gaps = [problematic_non_pad[j + 1] - problematic_non_pad[j] for j in range(min(num_bad - 1, 20))]
        print(f"  - Gaps between first 20 bad tokens: {gaps}")

    if max_prompt_length:
        bad_in_prompt = sum(1 for idx in problematic_non_pad if idx < max_prompt_length)
        print(f"  - Bad tokens in prompt region (idx < {max_prompt_length}): {bad_in_prompt}")
        print(f"  - Bad tokens in response region (idx >= {max_prompt_length}): {num_bad - bad_in_prompt}")

    ctx_start, ctx_end = max(0, first_bad - 5), min(seq_len, first_bad + 6)
    print(f"  - Context around first bad token ({ctx_start}:{ctx_end}):")
    for ctx_idx in range(ctx_start, ctx_end):
        is_neg1 = (layer_map[batch_idx, ctx_idx] == -1).all().item()
        is_non_pad = attention_mask[batch_idx, ctx_idx].item()
        marker = " <-- BAD" if (is_neg1 and is_non_pad and ctx_idx != last_non_pad_idx) else ""
        print(f"      pos {ctx_idx}: attn_mask={is_non_pad}, all_neg1={is_neg1}{marker}")


def preprocess_packed_seqs_moe_layer_map(
    layer_map: torch.Tensor,
    attention_mask: torch.Tensor,
    pre_process: bool = True,
    validate: bool = False,
    debug: bool = False,
    use_fp8_padding: bool = False,
) -> tuple[torch.Tensor, PackedSeqParams]:
    """
    Preprocess packed sequences for layer_map
    CP splits sequence into CP*2 chunks, and each GPU gets 2 chunks (GPU0 gets first and last chunks, GPU1 gets second and second last chunks, and so on), this is for load balancing with causal masking.
    See https://github.com/NVIDIA/TransformerEngine/issues/1368

    Args:
        layer_map: tensor of shape [batch_size, seq_len, n_experts]
        attention_mask: tensor of shape [batch_size, seq_len]
        pre_process: whether to preprocess (remove padding and pack)
        validate: if True, validate the moe_routermap before processing
        debug: if True, print debug information
    """
    batch_size = layer_map.shape[0]

    # Optional validation
    if validate:
        validate_moe_routermap(layer_map, attention_mask, debug=debug)

    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    tp_size = mpu.get_tensor_model_parallel_world_size()
    cp_size = mpu.get_context_parallel_world_size()
    cp_rank = mpu.get_context_parallel_rank()
    align_size = tp_size * cp_size * 2 if cp_size > 1 else tp_size
    if use_fp8_padding:
        original_align_size = align_size
        align_size = math.lcm(16, align_size)

    pad_size = (align_size - seqlens_in_batch % align_size) % align_size
    seqlens_in_batch_padded = seqlens_in_batch + pad_size
    cu_seqlens = torch.zeros(batch_size + 1, dtype=torch.int32, device=layer_map.device)
    cu_seqlens[1:] = torch.cumsum(seqlens_in_batch, dim=0)
    cu_seqlens_padded = torch.zeros(batch_size + 1, dtype=torch.int32, device=layer_map.device)
    cu_seqlens_padded[1:] = torch.cumsum(seqlens_in_batch_padded, dim=0)

    if use_fp8_padding:
        align_size_last = original_align_size * 128
        pad_size_last = (align_size_last - cu_seqlens_padded[-1] % align_size_last) % align_size_last
        cu_seqlens_padded[-1] += pad_size_last
        seqlens_in_batch_padded[-1] += pad_size_last

    max_seqlen_in_batch = seqlens_in_batch_padded.max().item()

    shape = list(layer_map.shape[1:])
    shape[0] = seqlens_in_batch_padded.sum().item() // cp_size

    if pre_process:
        layer_map_rmpad = -1 * torch.ones(shape, dtype=layer_map.dtype, device=layer_map.device)

        # Convert to CPU lists to avoid per-iteration GPU-to-CPU sync
        seqlens_list = seqlens_in_batch.tolist()
        cu_seqlens_padded_list = cu_seqlens_padded.tolist()
        seqlens_padded_list = seqlens_in_batch_padded.tolist()

        for i in range(batch_size):
            if cp_size <= 1:
                seqlen = seqlens_list[i]
                layer_map_rmpad[cu_seqlens_padded_list[i] : cu_seqlens_padded_list[i] + seqlen] = layer_map[
                    i, attention_mask[i]
                ]
                continue

            seqlen = seqlens_padded_list[i] // cp_size
            half_seqlen = seqlen // 2
            start_idx = cu_seqlens_padded_list[i] // cp_size

            # split to 2 chunks
            d_layer_map = layer_map[i, attention_mask[i]]
            layer_map_rmpad[start_idx : start_idx + half_seqlen] = d_layer_map[
                half_seqlen * cp_rank : half_seqlen * (cp_rank + 1)
            ]

            remain_start = seqlens_padded_list[i] - half_seqlen * (cp_rank + 1)
            remain_end = seqlens_padded_list[i] - half_seqlen * cp_rank
            remain_end = min(remain_end, d_layer_map.shape[0])
            remain_len = remain_end - remain_start
            if remain_len > 0:
                layer_map_rmpad[start_idx + half_seqlen : start_idx + half_seqlen + remain_len] = d_layer_map[
                    remain_start:remain_end
                ]

    packed_seq_params = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu_seqlens_padded,
        max_seqlen_q=max_seqlen_in_batch,
        cu_seqlens_kv=cu_seqlens_padded,
        max_seqlen_kv=max_seqlen_in_batch,
        cu_seqlens_q_padded=cu_seqlens_padded,
        cu_seqlens_kv_padded=cu_seqlens_padded,
    )

    if pre_process:
        return layer_map_rmpad.unsqueeze(0), packed_seq_params
    else:
        return layer_map, packed_seq_params


def preprocess_packed_seqs(
    input_ids: torch.Tensor, attention_mask: torch.Tensor, pre_process: bool = True, use_fp8_padding=False
) -> tuple[torch.Tensor, PackedSeqParams]:
    """
    Preprocess packed sequences
    CP splits sequence into CP*2 chunks, and each GPU gets 2 chunks (GPU0 gets first and last chunks, GPU1
    gets second and second last chunks, and so on), this is for load balancing with causal masking.
    See https://github.com/NVIDIA/TransformerEngine/issues/1368
    """
    from axon.utils.megatron.cp_utils import compute_cp_chunk_boundaries, compute_cp_padded_lens

    batch_size = input_ids.shape[0]

    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    tp_size = mpu.get_tensor_model_parallel_world_size()
    cp_size = mpu.get_context_parallel_world_size()
    cp_rank = mpu.get_context_parallel_rank()

    # --- Padding ---
    seqlens_in_batch_cpu: list[int] = seqlens_in_batch.tolist()
    if cp_size > 1:
        # Use shared helper (single source of truth for CP padding).
        seqlens_in_batch_padded_cpu = compute_cp_padded_lens(
            seqlens_in_batch_cpu,
            tp_size,
            cp_size,
            use_fp8_padding,
        )
    else:
        # No CP: align to tp_size (plus FP8 adjustments when enabled).
        align_size = tp_size
        if use_fp8_padding:
            original_align_size = align_size
            align_size = math.lcm(16, align_size)
        seqlens_in_batch_padded_cpu = [sl + (align_size - sl % align_size) % align_size for sl in seqlens_in_batch_cpu]
        if use_fp8_padding:
            align_size_last = original_align_size * 128
            cum_padded = sum(seqlens_in_batch_padded_cpu)
            pad_last = (align_size_last - cum_padded % align_size_last) % align_size_last
            seqlens_in_batch_padded_cpu[-1] += pad_last

    # Build cumulative-sum tensors on GPU (needed by PackedSeqParams).
    seqlens_in_batch_padded = torch.tensor(seqlens_in_batch_padded_cpu, dtype=torch.int32, device=input_ids.device)
    cu_seqlens_padded = torch.zeros(batch_size + 1, dtype=torch.int32, device=input_ids.device)
    cu_seqlens_padded[1:] = torch.cumsum(seqlens_in_batch_padded, dim=0)
    cu_seqlens_padded_cpu: list[int] = cu_seqlens_padded.tolist()

    max_seqlen_in_batch = max(seqlens_in_batch_padded_cpu)

    shape = list(input_ids.shape[1:])
    shape[0] = sum(seqlens_in_batch_padded_cpu) // cp_size
    if pre_process:
        input_ids_rmpad = torch.zeros(shape, dtype=input_ids.dtype, device=input_ids.device)
        for i in range(batch_size):
            if cp_size <= 1:
                seqlen = seqlens_in_batch_cpu[i]
                start_idx = cu_seqlens_padded_cpu[i]
                input_ids_rmpad[start_idx : start_idx + seqlen] = input_ids[i, attention_mask[i]]
                continue

            # Use shared helper for chunk boundaries.
            half_seqlen, chunks = compute_cp_chunk_boundaries(
                seqlens_in_batch_padded_cpu[i],
                cp_size,
                cp_rank,
            )
            start_idx = cu_seqlens_padded_cpu[i] // cp_size
            d = input_ids[i, attention_mask[i]]

            # Chunk 1 (from start of sequence)
            c1s, c1e = chunks[0]
            input_ids_rmpad[start_idx : start_idx + half_seqlen] = d[c1s:c1e]

            # Chunk 2 (from end of sequence — may extend past valid tokens into padding)
            c2s, c2e = chunks[1]
            c2e_clamped = min(c2e, d.shape[0])
            remain_len = c2e_clamped - c2s
            if remain_len > 0:
                input_ids_rmpad[start_idx + half_seqlen : start_idx + half_seqlen + remain_len] = d[c2s:c2e_clamped]

    packed_seq_params = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu_seqlens_padded,
        max_seqlen_q=max_seqlen_in_batch,
        cu_seqlens_kv=cu_seqlens_padded,
        max_seqlen_kv=max_seqlen_in_batch,
        cu_seqlens_q_padded=cu_seqlens_padded,
        cu_seqlens_kv_padded=cu_seqlens_padded,
    )
    if pre_process:
        return input_ids_rmpad.unsqueeze(0), packed_seq_params
    else:
        return input_ids, packed_seq_params


def postprocess_packed_seqs(
    output: torch.Tensor,
    packed_seq_params: PackedSeqParams,
    attention_mask: torch.Tensor,
    batch_size: int,
    seq_len: int,
    post_process: bool = True,
    skip_cp_gather: bool = False,
) -> torch.Tensor:
    """
    Postprocess packed sequences.

    Args:
        skip_cp_gather: When True and cp_size > 1, skip the all-gather across
            the CP group.  Only the local rank's zigzag chunk is placed into
            the output tensor; non-local positions remain zero.  This avoids
            ``(cp_size - 1) / cp_size`` redundant communication and enables
            efficient local-only loss computation.
    """
    if not post_process:
        return output

    # -------------------------------------------------------------------------
    # Move the lengths and offsets needed for subsequent Python-level indexing to the CPU in advance,
    # to avoid a large number of .item() calls in the loop
    # -------------------------------------------------------------------------
    cu_padded_cpu: list[int] = packed_seq_params.cu_seqlens_q_padded.tolist()
    seq_lens_cpu: list[int] = attention_mask.sum(dim=1, dtype=torch.int32).cpu().tolist()

    shape = [batch_size, seq_len] + list(output.shape[2:])  # 1,packed, dim -> batch_size, seq_len, dim
    output_new = torch.zeros(shape, dtype=output.dtype, device=output.device)

    cp_size = mpu.get_context_parallel_world_size()

    if cp_size > 1 and not skip_cp_gather:
        # all gather output across context parallel group
        # output shape: [1, packed_len, hidden_dim]
        # need to gather across cp group and concatenate in sequence dimension
        output_list = [torch.empty_like(output, dtype=output.dtype) for _ in range(cp_size)]
        torch.distributed.all_gather(output_list, output.detach(), group=mpu.get_context_parallel_group())
        output_list[mpu.get_context_parallel_rank()] = output
    else:
        output_list = [output]

    from axon.utils.megatron.cp_utils import compute_cp_chunk_boundaries

    for i in range(batch_size):
        if cp_size <= 1:
            s = seq_lens_cpu[i]
            start_idx = cu_padded_cpu[i]
            output_new[i, attention_mask[i]] = output[0][start_idx : start_idx + s]
            continue

        padded_len_i = cu_padded_cpu[i + 1] - cu_padded_cpu[i]
        s_len_padded_chunk = padded_len_i // cp_size
        half_seqlen = s_len_padded_chunk // 2
        s_len = seq_lens_cpu[i]
        s_len_padded = padded_len_i
        tmp = torch.zeros(s_len_padded, *output.shape[2:], device=output.device, dtype=output.dtype)

        if skip_cp_gather:
            # Only place local rank's chunk; non-local positions stay zero.
            j = mpu.get_context_parallel_rank()
            _, chunks = compute_cp_chunk_boundaries(padded_len_i, cp_size, j)
            o = output[0]
            packed_start_idx = cu_padded_cpu[i] // cp_size
            o0 = o[packed_start_idx : packed_start_idx + half_seqlen]
            o1 = o[packed_start_idx + half_seqlen : packed_start_idx + s_len_padded_chunk]
            tmp[chunks[0][0] : chunks[0][1]] = o0
            tmp[chunks[1][0] : chunks[1][1]] = o1
        else:
            for j in range(cp_size):
                _, chunks = compute_cp_chunk_boundaries(padded_len_i, cp_size, j)
                o = output_list[j][0]
                packed_start_idx = cu_padded_cpu[i] // cp_size
                o0 = o[packed_start_idx : packed_start_idx + half_seqlen]
                o1 = o[packed_start_idx + half_seqlen : packed_start_idx + s_len_padded_chunk]
                tmp[chunks[0][0] : chunks[0][1]] = o0
                tmp[chunks[1][0] : chunks[1][1]] = o1

        output_new[i, attention_mask[i]] = tmp[:s_len]

    return output_new


def preprocess_bshd(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    sequence_parallel: bool = False,
    pre_process: bool = True,
):
    """
    Remove left padding from input_ids, attention_mask and position_ids
    return new_input_ids, new_attention_mask, new_position_ids
    """
    assert attention_mask.ndim == 2
    assert position_ids.ndim == 2
    cp_size = mpu.get_context_parallel_world_size()
    assert cp_size == 1, "Context parallel size without seq_pack is not supported"
    batch_size = input_ids.shape[0]
    shape = list(input_ids.shape)  # batch_size, seq_len,...
    seq_lens = attention_mask.sum(dim=1)
    seq_len = seq_lens.max().item()
    if sequence_parallel:
        sp_world_size = mpu.get_tensor_model_parallel_world_size()
        pad_size = (sp_world_size - seq_len % sp_world_size) % sp_world_size
        seq_len = seq_len + pad_size
    shape[1] = seq_len
    if pre_process:
        new_input_ids = torch.zeros(dtype=input_ids.dtype, device=input_ids.device, size=shape)
    new_attention_mask = torch.zeros(
        dtype=attention_mask.dtype, device=attention_mask.device, size=(batch_size, seq_len)
    )
    new_position_ids = torch.zeros(dtype=position_ids.dtype, device=position_ids.device, size=(batch_size, seq_len))
    for i in range(batch_size):
        if pre_process:
            new_input_ids[i, : seq_lens[i]] = input_ids[i, attention_mask[i]]
        new_attention_mask[i, : seq_lens[i]] = attention_mask[i, attention_mask[i]]
        new_position_ids[i, : seq_lens[i]] = position_ids[i, attention_mask[i]]
    if pre_process:
        return new_input_ids, new_attention_mask, new_position_ids
    else:
        return input_ids, new_attention_mask, new_position_ids


def postprocess_bshd(
    result,
    attention_mask: torch.Tensor,
    original_attention_mask: torch.Tensor,
    origin_seqlen: int,
    post_process: bool = True,
):
    """
    Recover left padding from result
    return result
    """
    if not post_process:
        return result
    shape = list(result.shape)
    batch_size = shape[0]
    shape[1] = origin_seqlen
    new_result = torch.zeros(dtype=result.dtype, device=result.device, size=shape)
    for i in range(batch_size):
        new_result[i, original_attention_mask[i]] = result[i, attention_mask[i]]
    return new_result


def postprocess_packed_seqs_for_dict_output(
    labels_mask: torch.Tensor,
    output: CausalLMOutputForPPO,
    packed_seq_params: PackedSeqParams,
    attention_mask: torch.Tensor,
    batch_size: int,
    seq_len: int,
    post_process: bool = True,
    skip_cp_gather: bool = False,
) -> dict[str, torch.Tensor]:
    """_summary_
    For fused kernels, the output is a dictionary with keys like 'log_probs', 'entropy', etc.
    This function post-processes each tensor in the output dictionary.
    Args:
        output (CausalLMOutputForPPO): _description_
        packed_seq_params (PackedSeqParams): _description_
        attention_mask (torch.Tensor): _description_
        batch_size (int): _description_
        seq_len (int): _description_
        post_process (bool, optional): _description_. Defaults to True.
    Returns:
        CausalLMOutputForPPO: _description_
    """
    ret = {}
    output.entropy = output.entropy.view(1, -1)
    output.log_probs = output.log_probs.view(1, -1)
    output.log_probs = output.log_probs.masked_fill(~labels_mask, 0.0)
    ret["entropy"] = postprocess_packed_seqs(
        output.entropy,
        packed_seq_params,
        attention_mask,
        batch_size,
        seq_len,
        post_process=post_process,
        skip_cp_gather=skip_cp_gather,
    )
    ret["log_probs"] = postprocess_packed_seqs(
        output.log_probs,
        packed_seq_params,
        attention_mask,
        batch_size,
        seq_len,
        post_process=post_process,
        skip_cp_gather=skip_cp_gather,
    )
    return ret


### No padding versions for model engine
### inputs are nested tensors


def preprocess_thd_no_padding(
    input_ids: torch.Tensor, pre_process: bool = True, need_roll: bool = False
) -> tuple[torch.Tensor, PackedSeqParams]:
    """
    Preprocess packed sequences
    CP splits sequence into CP*2 chunks, and each GPU gets 2 chunks (GPU0 gets first and last chunks, GPU1
    gets second and second last chunks, and so on), this is for load balancing with causal masking.
    See https://github.com/NVIDIA/TransformerEngine/issues/1368
    """
    batch_size = input_ids.shape[0]

    tp_size = mpu.get_tensor_model_parallel_world_size()
    cp_size = mpu.get_context_parallel_world_size()
    cp_rank = mpu.get_context_parallel_rank()
    align_size = tp_size * cp_size * 2 if cp_size > 1 else tp_size
    seqlens_in_batch = input_ids.offsets().diff()

    pad_size = (align_size - seqlens_in_batch % align_size) % align_size
    seqlens_in_batch_padded = seqlens_in_batch + pad_size

    cu_seqlens = torch.zeros(batch_size + 1, dtype=torch.int32, device=input_ids.device)
    cu_seqlens[1:] = torch.cumsum(seqlens_in_batch, dim=0)
    cu_seqlens_padded = torch.zeros(batch_size + 1, dtype=torch.int32, device=input_ids.device)
    cu_seqlens_padded[1:] = torch.cumsum(seqlens_in_batch_padded, dim=0)

    # ----------------------------------------------------------------------------
    # Move the index information needed in the subsequent loop to the CPU at once,
    # to avoid frequent .item() calls in the loop that cause D2H synchronization
    # ----------------------------------------------------------------------------
    seqlens_in_batch_cpu: list[int] = seqlens_in_batch.tolist()  # original valid lengths
    seqlens_in_batch_padded_cpu: list[int] = seqlens_in_batch_padded.tolist()  # lengths after padding
    cu_seqlens_padded_cpu: list[int] = cu_seqlens_padded.tolist()  # start positions (after padding)

    # Pure Python int calculation to avoid further synchronization
    max_seqlen_in_batch = max(seqlens_in_batch_padded_cpu)

    shape = list(input_ids.shape[1:])
    shape[0] = sum(seqlens_in_batch_padded_cpu) // cp_size
    if pre_process:
        input_ids_rmpad = torch.zeros(shape, dtype=input_ids.dtype, device=input_ids.device)
        if need_roll:
            saved_roll_dict = {}
        for i in range(batch_size):
            # Use Python int, so no GPU→CPU sync in the loop
            if cp_size <= 1:
                seqlen = seqlens_in_batch_cpu[i]
                start_idx = cu_seqlens_padded_cpu[i]
                input_ids_rmpad[start_idx : start_idx + seqlen] = input_ids[i]
                continue

            seqlen_padded_i = seqlens_in_batch_padded_cpu[i]
            seqlen = seqlen_padded_i // cp_size
            half_seqlen = seqlen // 2
            start_idx = cu_seqlens_padded_cpu[i] // cp_size
            # split to 2 chunks
            d = input_ids[i]
            input_ids_rmpad[start_idx : start_idx + half_seqlen] = d[
                half_seqlen * cp_rank : half_seqlen * (cp_rank + 1)
            ]

            remain_start = seqlen_padded_i - half_seqlen * (cp_rank + 1)
            remain_end = seqlen_padded_i - half_seqlen * cp_rank
            remain_end = min(remain_end, d.shape[0])
            remain_len = remain_end - remain_start
            if remain_len > 0:
                input_ids_rmpad[start_idx + half_seqlen : start_idx + half_seqlen + remain_len] = d[
                    remain_start:remain_end
                ]

            if need_roll:
                # Handle roll for cp_size > 1 case
                saved_roll_dict[start_idx + half_seqlen - 1] = d[(cp_rank + 1) * half_seqlen]
                if remain_len > 0:
                    if remain_end == d.shape[0]:
                        saved_roll_dict[start_idx + half_seqlen + remain_len - 1] = d[0]
                    else:
                        saved_roll_dict[start_idx + half_seqlen + remain_len - 1] = d[remain_end]

        if need_roll:
            input_ids_rmpad = torch.roll(input_ids_rmpad, shifts=-1, dims=0)
            if len(saved_roll_dict) > 0:
                for k, v in saved_roll_dict.items():
                    input_ids_rmpad[k] = v

    packed_seq_params = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu_seqlens_padded,
        max_seqlen_q=max_seqlen_in_batch,
        cu_seqlens_kv=cu_seqlens_padded,
        max_seqlen_kv=max_seqlen_in_batch,
        cu_seqlens_q_padded=cu_seqlens_padded,
        cu_seqlens_kv_padded=cu_seqlens_padded,
    )
    if pre_process:
        return input_ids_rmpad.unsqueeze(0), packed_seq_params
    else:
        return input_ids, packed_seq_params


def postprocess_thd_no_padding(
    output: torch.Tensor,
    packed_seq_params: PackedSeqParams,
    input_ids: torch.Tensor,
    batch_size: int,
    post_process: bool = True,
) -> torch.Tensor:
    """
    Postprocess packed sequences
    """
    if not post_process:
        return output

    # -------------------------------------------------------------------------
    # Move the lengths and offsets needed for subsequent Python-level indexing to the CPU in advance,
    # to avoid a large number of .item() calls in the loop
    # -------------------------------------------------------------------------
    cu_padded_cpu: list[int] = packed_seq_params.cu_seqlens_q_padded.tolist()
    # The reason why we use input_ids.offsets() instead of packed_seq_params.cu_seqlens_q.diff()
    # is that the latter one is the padded length, while the former one is the original length.
    cu_seqlens = input_ids.offsets()
    seq_lens_cpu: list[int] = cu_seqlens.diff().tolist()

    output_new = []

    cp_size = mpu.get_context_parallel_world_size()
    # all gather output across context parallel group
    if cp_size > 1:
        # output shape: [1, packed_len, hidden_dim]
        # need to gather across cp group and concatenate in sequence dimension
        output_list = [torch.empty_like(output) for _ in range(cp_size)]
        torch.distributed.all_gather(output_list, output.detach(), group=mpu.get_context_parallel_group())
        output_list[mpu.get_context_parallel_rank()] = output
    else:
        output_list = [output]

    for i in range(batch_size):
        if cp_size <= 1:
            s = seq_lens_cpu[i]
            start_idx = cu_padded_cpu[i]
            output_new.append(output[0][start_idx : start_idx + s])
            continue
        s_len_padded_chunk = (cu_padded_cpu[i + 1] - cu_padded_cpu[i]) // cp_size
        half_seqlen = s_len_padded_chunk // 2
        s_len = seq_lens_cpu[i]
        s_len_padded = s_len_padded_chunk * cp_size
        tmp = torch.empty(s_len_padded, *output.shape[2:], device=output.device)
        for j in range(cp_size):
            o = output_list[j][0]
            # split to 2 chunks
            packed_start_idx = cu_padded_cpu[i] // cp_size
            o0, o1 = (
                o[packed_start_idx : packed_start_idx + half_seqlen],
                o[packed_start_idx + half_seqlen : packed_start_idx + s_len_padded_chunk],
            )
            tmp[j * half_seqlen : (j + 1) * half_seqlen] = o0
            tmp[s_len_padded - (j + 1) * half_seqlen : s_len_padded - j * half_seqlen] = o1
        output_new.append(tmp[:s_len])

    output_new_tensor = torch.nested.as_nested_tensor(output_new, layout=torch.jagged)

    return output_new_tensor


def preprocess_bshd_no_padding(input_ids: torch.Tensor, pre_process: bool = True, need_roll: bool = False):
    """
    Preprocess bshd sequences
    return "input_ids, attention_mask, position_ids"
    """
    cp_size = mpu.get_context_parallel_world_size()
    # BSHD no-padding preprocessing currently runs without context parallelism.
    assert cp_size == 1, "Context parallel size without bshd is not supported yet"

    batch_size = input_ids.shape[0]
    seqlens_in_batch = input_ids.offsets().diff()
    max_seqlen = seqlens_in_batch.max().item()
    if mpu.get_tensor_model_parallel_world_size() > 1:
        sp_world_size = mpu.get_tensor_model_parallel_world_size()
        pad_size = (sp_world_size - max_seqlen % sp_world_size) % sp_world_size
        max_seqlen = max_seqlen + pad_size

    attention_mask = torch.zeros(batch_size, max_seqlen, dtype=torch.bool, device=input_ids.device)
    input_ids_bshd = torch.zeros(batch_size, max_seqlen, dtype=input_ids.dtype, device=input_ids.device)
    for i in range(batch_size):
        attention_mask[i, : seqlens_in_batch[i]] = True
        input_ids_bshd[i, : seqlens_in_batch[i]] = input_ids[i]
    position_ids = torch.arange(max_seqlen, dtype=torch.long, device=input_ids.device)
    position_ids = position_ids.unsqueeze(0).expand_as(input_ids_bshd)
    if need_roll:
        input_ids_bshd = torch.roll(input_ids_bshd, shifts=-1, dims=1)

    return input_ids_bshd, attention_mask, position_ids


def postprocess_bshd_no_padding(
    output: torch.Tensor,
    attention_mask: torch.Tensor,
    post_process: bool = True,
) -> torch.Tensor:
    """
    Postprocess bshd sequences
    """
    if not post_process:
        return output

    batch_size = output.shape[0]
    output_new = []

    for i in range(batch_size):
        mask = attention_mask[i].bool()
        output_new.append(output[i][mask])

    output_new_tensor = torch.nested.as_nested_tensor(output_new, layout=torch.jagged)

    return output_new_tensor
