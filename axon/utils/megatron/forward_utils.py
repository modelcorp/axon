# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
Megatron forward/forward-backward with pipeline parallelism.
"""

import logging
import os
from functools import partial

import torch
import torch.distributed
from megatron.core import parallel_state as mpu
from megatron.core.pipeline_parallel import get_forward_backward_func

from axon.globals import DEBUG_MOE_REPLAY
from axon.monkey_patches.megatron.moe_replay import (
    MoERoutingContext,
    clear_moe_layer_routing_queues,
    set_moe_routing_context,
)
from axon.utils.megatron.pipeline_parallel import make_batch_generator
from axon.utils.megatron.utils import get_moe_layer_distribution, is_moe_layer, unwrap_model
from axon.utils.seqlen_balancing import rearrange_micro_batches
from axon.utils.torch import get_device_id, get_torch_device
from axon.utils.torch.ops import broadcast_dict_tensor

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("AXON_LOGGING_LEVEL", "WARN"))


def _debug_moe(msg: str) -> None:
    logger.debug("[DEBUG_MOE_REPLAY, forward_step] %s", msg)


def setup_moe_routing(model, batch, vpp_rank, attention_mask, *, moe_routermap=None, moe_vpp_counts=None):
    """Build per-layer MoE routing maps and set the global MoERoutingContext.

    The routermap ``[bs, seq_len, n_local_layers, topk]`` is passed explicitly
    via *moe_routermap* (kept on CPU, moved to GPU one layer at a time).
    Falls back to ``batch["local_moe_routermap"]`` for backwards compatibility.
    """
    unwrapped = unwrap_model(model)
    if not (hasattr(unwrapped, "decoder") and hasattr(unwrapped.decoder, "layers")):
        set_moe_routing_context(None)
        return

    # Resolve routermap source: explicit kwarg or batch key (legacy).
    if moe_routermap is not None:
        moe_routermaps = moe_routermap
        local_vpp_counts = moe_vpp_counts[0] if moe_vpp_counts is not None else None
    elif "local_moe_routermap" in batch:
        moe_routermaps = batch["local_moe_routermap"]
        local_vpp_counts = batch["local_vpp_counts"][0] if "local_vpp_counts" in batch else None
    else:
        set_moe_routing_context(None)
        return
    n_layers = moe_routermaps.shape[2]  # layers on dim 2
    if moe_routermaps is None or n_layers == 0:
        set_moe_routing_context(None)
        return

    from axon.models.mcore.forward.util import preprocess_packed_seqs_moe_layer_map, validate_moe_routermap

    decoder_layers = unwrapped.decoder.layers
    assert moe_routermaps.shape[1] == attention_mask.shape[1], (
        f"MoE routermap seq_len mismatch: {moe_routermaps.shape[1]} vs {attention_mask.shape[1]}"
    )
    actual_moe_count = sum(1 for layer in decoder_layers if is_moe_layer(layer))

    # Slice routermaps for the current VPP chunk (layer dim = 2)
    if local_vpp_counts is not None:
        if vpp_rank is None:
            start, end = 0, local_vpp_counts.sum().item()
        elif vpp_rank < len(local_vpp_counts):
            start = local_vpp_counts[:vpp_rank].sum().item()
            expected = local_vpp_counts[vpp_rank].item()
            end = start + (actual_moe_count if expected != actual_moe_count else expected)
        else:
            start = local_vpp_counts.sum().item()
            end = start + actual_moe_count
        total = n_layers
        moe_routermaps = moe_routermaps[:, :, min(start, total) : min(end, total), :]
        n_layers = moe_routermaps.shape[2]

    if n_layers > 0 and DEBUG_MOE_REPLAY:
        # No separate prompt region; max_prompt_length is not meaningful for debug validation
        max_prompt_length = None
        try:
            _layer0 = moe_routermaps[:, :, 0, :].to(attention_mask.device)
            validate_moe_routermap(_layer0, attention_mask, debug=True, max_prompt_length=max_prompt_length)
        except AssertionError as e:
            for msg in [
                f"MoE Routermap validation failed: {e}",
                f"Routermap shape: {moe_routermaps[:, :, 0, :].shape}",
                f"Attention mask shape: {attention_mask.shape}",
                f"max_prompt_length: {max_prompt_length}",
                f"Attention mask sum per sample: {attention_mask.sum(dim=1).tolist()}",
            ]:
                _debug_moe(msg)
            raise

    # Detect FP8 mode to align routermap padding with token padding
    fp8 = getattr(unwrapped.config, "fp8", None) if hasattr(unwrapped, "config") else None
    use_fp8_padding = fp8 in ("e4m3", "hybrid")

    layer_routing_maps = {}
    num_decoder_layers = len(decoder_layers)
    local_moe_index = 0
    tp_size, tp_rank = mpu.get_tensor_model_parallel_world_size(), mpu.get_tensor_model_parallel_rank()
    cp_size, cp_rank = mpu.get_context_parallel_world_size(), mpu.get_context_parallel_rank()

    for idx, layer in enumerate(decoder_layers):
        if not is_moe_layer(layer):
            continue
        # Standard Megatron MoE: layer.mlp is the MoELayer
        # Gemma4 MoE: layer.moe_block.moe_layer is the MoELayer
        if hasattr(layer, "moe_block") and layer.moe_block is not None:
            _moe_module = layer.moe_block.moe_layer
        else:
            _moe_module = layer.mlp
        layer_num = _moe_module.layer_number
        if layer_num > num_decoder_layers:
            if DEBUG_MOE_REPLAY:
                _debug_moe(f"Skipping MTP MoE layer {layer_num} (>= num_decoder_layers={num_decoder_layers})")
            continue
        if local_moe_index < n_layers:
            # Move one layer slice to GPU — [micro_bs, seq_len, topk] is small.
            layer_map_gpu = moe_routermaps[:, :, local_moe_index, :].to(attention_mask.device)
            metadata, _ = preprocess_packed_seqs_moe_layer_map(
                layer_map_gpu,
                attention_mask,
                validate=DEBUG_MOE_REPLAY,
                debug=DEBUG_MOE_REPLAY,
                use_fp8_padding=use_fp8_padding,
            )
            sl = metadata.shape[1]
            if DEBUG_MOE_REPLAY:
                _debug_moe(
                    f"Layer {idx}, MoE index {local_moe_index}: before TP slice: "
                    f"metadata shape {metadata.shape}, tp={tp_rank}/{tp_size}, cp={cp_rank}/{cp_size}"
                )
            # Current replay metadata slicing follows the column-wise MoE layout.
            # Extend this branch if a row-wise routed layout is added.
            metadata = metadata[0, tp_rank * sl // tp_size : (tp_rank + 1) * sl // tp_size].long().contiguous()
            if DEBUG_MOE_REPLAY:
                num_neg1 = (metadata == -1).all(dim=-1).sum().item()
                _debug_moe(
                    f"Layer {idx}, MoE index {local_moe_index}: after TP/CP slice: "
                    f"metadata shape {metadata.shape}, -1 rows: {num_neg1}/{metadata.shape[0]}, "
                    f"layer_num={layer_num}"
                )
            layer_routing_maps[layer_num] = metadata.clone()
            # Mark layer so patched_router_and_preprocess knows this is a first-forward.
            _moe_module._moe_routing_fresh = True
        local_moe_index += 1

    set_moe_routing_context(MoERoutingContext(layer_routing_maps=layer_routing_maps) if layer_routing_maps else None)


def _prepare_mini_batch(data, actor_module):
    """Broadcast data across PP ranks and prepare MoE/multimodal metadata."""
    # Separate moe_routermap before bulk GPU transfer to reduce peak GPU memory.
    # The routermap is [bs, seq_len, n_moe_layers, topk] (int16).
    # IMPORTANT: use exclude() to create a new TensorDict instead of del —
    # mutating and then calling .contiguous() corrupts the internal storage layout.
    moe_routermap = None
    if "moe_routermap" in data.batch:
        moe_routermap = data.batch["moe_routermap"]
        data.batch = data.batch.exclude("moe_routermap")

    data.to(get_device_id())
    data.batch = data.batch.contiguous()
    broadcast_dict_tensor(
        data.batch,
        src=mpu.get_pipeline_model_parallel_last_rank(),
        group=mpu.get_pipeline_model_parallel_group(),
    )
    data.to("cpu")
    data.batch["attention_mask"] = data.batch["attention_mask"].to(bool)

    # Broadcast moe_routermap across PP ranks, then slice local layers and store on CPU.
    # Only the broadcast requires GPU (NCCL). Everything else runs on CPU to avoid
    # holding the large routermap on GPU alongside model weights.
    # Layout: [bs, seq_len, n_layers, topk] — setup_moe_routing indexes dim 2.
    if moe_routermap is not None:
        pp_size = mpu.get_pipeline_model_parallel_world_size()
        if pp_size > 1:
            # NCCL requires GPU tensors and does not support int16.
            device = get_device_id()
            moe_routermap = moe_routermap.to(device=device, dtype=torch.int32)
            torch.distributed.broadcast(
                moe_routermap,
                src=mpu.get_pipeline_model_parallel_last_rank(),
                group=mpu.get_pipeline_model_parallel_group(),
                async_op=False,
            )
            moe_routermap = moe_routermap.cpu().to(dtype=torch.int16)
        # PP=1: broadcast is a no-op, skip GPU round-trip entirely.
        # moe_routermap is already int16 on CPU from program_component.

        pp_rank = mpu.get_pipeline_model_parallel_rank()
        batch_size = data.batch["attention_mask"].shape[0]
        moe_layer_distribution, local_vpp_counts, total_decoder_moe_layers = get_moe_layer_distribution(
            actor_module, len(actor_module)
        )
        if moe_routermap.shape[2] > total_decoder_moe_layers:
            moe_routermap = moe_routermap[:, :, :total_decoder_moe_layers, :]
        local_moe_indices = moe_layer_distribution[pp_rank]
        local_moe_routermap = (
            moe_routermap[:, :, local_moe_indices, :] if local_moe_indices else moe_routermap[:, :, :0, :]
        )
        data.batch["local_moe_routermap"] = local_moe_routermap.contiguous()
        del moe_routermap, local_moe_routermap
        data.batch["local_vpp_counts"] = local_vpp_counts.repeat(batch_size, 1)

    has_mm = "multi_modal_inputs" in data.non_tensor_batch
    if has_mm:
        mm_inputs = data.non_tensor_batch["multi_modal_inputs"]
        data.batch["multi_modal_inputs"] = mm_inputs
        data.batch["multi_modal_inputs_idx"] = torch.arange(len(mm_inputs), dtype=torch.int64)

    # qwen2vl mrope [bs, 3, seq_len] -> [bs, seq_len]; mcore recomputes pos ids
    if data.batch["position_ids"].dim() == 3:
        data.batch["position_ids"] = data.batch["position_ids"][:, 0]

    return has_mm


def megatron_forward_backward(
    module,
    data,
    forward_step_fn,
    forward_only=False,
    use_dynamic_bsz=False,
    micro_batch_size=None,
    max_token_len=None,
    tf_config=None,
    **forward_step_kwargs,
):
    """Run a forward (or forward+backward) pass over micro-batches using
    Megatron pipeline parallelism.

    This is a generic scaffolding function.  Model-specific behavior lives in
    ``forward_step_fn`` and the ``loss_func`` it returns.

    Args:
        module: List of Megatron model chunks (possibly wrapped in DDP).
        data: DataProto containing the mini-batch.
        forward_step_fn: ``(batch_iter, model, **kwargs) -> (output, loss_func)``.
            Typically ``CausalLM.forward_step``, ``ValueModel.forward_step``, etc.
        forward_only: If True, skip backward pass.
        use_dynamic_bsz: Use dynamic micro-batch sizing based on token length.
        micro_batch_size: Fixed micro-batch size (required when not dynamic).
        max_token_len: Max token length per micro-batch (required when dynamic).
        tf_config: Megatron TransformerConfig (needed for VPP micro-batch grouping).
        **forward_step_kwargs: Extra kwargs passed to ``forward_step_fn`` via
            ``functools.partial`` (e.g. ``hf_config``, ``config``, ``temperature``).
    """
    # Flush per-layer recompute FIFOs from any previous call (e.g. forward-only old_log_probs).
    clear_moe_layer_routing_queues(module)

    has_mm = _prepare_mini_batch(data, module)
    indices = None

    # Split into micro-batches
    if use_dynamic_bsz:
        assert max_token_len is not None, "max_token_len must be set when use_dynamic_bsz is True"
        vpp_size = mpu.get_virtual_pipeline_model_parallel_world_size()
        if vpp_size is not None and vpp_size > 1:
            group_size = tf_config.microbatch_group_size_per_vp_stage
            micro_batches, indices = rearrange_micro_batches(
                batch=data.batch, num_batches_divided_by=group_size, max_token_len=max_token_len
            )
            assert len(micro_batches) % group_size == 0, (
                f"micro_batches must be divisible by microbatch_group_size_per_vp_stage {group_size}"
            )
        else:
            micro_batches, indices = rearrange_micro_batches(batch=data.batch, max_token_len=max_token_len)
        total_seqlen = max_token_len
    else:
        assert micro_batch_size is not None, "micro_batch_size is required when not using dynamic batch size"
        micro_batches = data.batch.split(micro_batch_size)
        total_seqlen = micro_batch_size * micro_batches[0]["input_ids"].shape[1]

    n_micro_batch = len(micro_batches)

    # Bind all kwargs to forward_step_fn via partial
    bound_forward_step = partial(
        forward_step_fn,
        forward_only=forward_only,
        n_micro_batches=n_micro_batch,
        tf_config=tf_config,
        **forward_step_kwargs,
    )

    batch_generator = make_batch_generator(micro_batches, vpp_size=len(module))
    losses_reduced = get_forward_backward_func()(
        forward_step_func=bound_forward_step,
        data_iterator=batch_generator,
        model=module,
        num_microbatches=n_micro_batch,
        seq_length=total_seqlen,
        micro_batch_size=1,
        forward_only=forward_only,
    )

    # Clear MoE routing context to prevent stale data leaking to subsequent forward passes
    # (e.g., from actor forward to ref model forward on the same thread)
    set_moe_routing_context(None)

    # Flush any unconsumed FIFO entries. With recompute_granularity=full, the
    # transformer-level recompute may not consume all MoE FIFO entries (e.g., the
    # last 1-2 micro-batches' entries can survive if the schedule doesn't trigger
    # recompute for them). Without this, ~75 MB/step leaks on GPU.
    clear_moe_layer_routing_queues(module)

    get_torch_device().empty_cache()

    if has_mm:
        data.batch.pop("multi_modal_inputs")
        data.batch.pop("multi_modal_inputs_idx")
        if "multi_modal_inputs" in data.non_tensor_batch:
            data.non_tensor_batch.pop("multi_modal_inputs")

    result = {"output": losses_reduced}
    if use_dynamic_bsz:
        result["indices"] = indices
    return result
