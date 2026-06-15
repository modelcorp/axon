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
from abc import ABC, abstractmethod

import torch
import torch._dynamo
import torch.distributed as dist
import torch.nn as nn
from megatron.core import mpu, tensor_parallel
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.module import MegatronModule


class _ScaleGradient(torch.autograd.Function):
    """
    Scale gradients in backward pass only.
    Forward: identity
    Backward: multiply gradient by scale factor

    This fixes ACTIVATION gradients but NOT weight gradients.
    """

    @staticmethod
    def forward(ctx, input_: torch.Tensor, scale: float):
        ctx.scale = scale
        return input_

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output * ctx.scale, None


def scale_gradient(input_: torch.Tensor, scale: float) -> torch.Tensor:
    """Scale gradients by a factor during backward pass (identity in forward)."""
    return _ScaleGradient.apply(input_, scale)


def scale_weight_gradients(module: nn.Module, scale: float):
    """
    Register backward hooks to scale weight gradients.
    Call this after model initialization for all attention submodules.

    Usage:
        for layer in model.layers:
            if hasattr(layer, 'attention'):
                scale_weight_gradients(layer.attention, 1.0 / tp_world_size)
    """

    def make_hook(s):
        def hook(grad):
            return grad * s

        return hook

    for param in module.parameters():
        if param.requires_grad:
            param.register_hook(make_hook(scale))


class HuggingfaceAttention(MegatronModule, ABC):
    """Attention layer abstract class.

    This layer only contains common modules required for the "self attn" and
    "cross attn" specializations.
    """

    def __init__(
        self,
        config,
        hf_config,
        layer_number: int,
        layer_type: str = None,
        cp_comm_type: str = "p2p",
        pg_collection=None,
    ):
        super().__init__(config=config)
        self.config = config
        # Note that megatron layer_number starts at 1
        self.layer_number = layer_number
        self.hf_layer_idx = layer_number - 1
        self.hf_config = hf_config
        # hardcode to fa2 at the moment.
        self.hf_config._attn_implementation = "flash_attention_2"

        # Track TP world size for gradient scaling
        self._tp_world_size = None
        self._weight_grad_hooks_registered = False

    def _maybe_register_weight_grad_hooks(self):
        """
        Register hooks to scale weight gradients by 1/TP.
        Called lazily on first forward to ensure all submodules exist.
        """
        if self._weight_grad_hooks_registered:
            return

        tp_world_size = mpu.get_tensor_model_parallel_world_size()
        if tp_world_size > 1 and self.config.sequence_parallel:
            scale = 1.0 / tp_world_size

            def make_hook(s):
                def hook(grad):
                    if grad is not None:
                        return grad * s
                    return grad

                return hook

            # Register hooks for all parameters in this module
            for name, param in self.named_parameters():
                if param.requires_grad:
                    param.register_hook(make_hook(scale))

            self._weight_grad_hooks_registered = True

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        key_value_states: torch.Tensor | None = None,
        inference_context: BaseInferenceContext | None = None,
        rotary_pos_emb: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None = None,
        rotary_pos_cos: torch.Tensor | None = None,
        rotary_pos_sin: torch.Tensor | None = None,
        rotary_pos_cos_sin: torch.Tensor | None = None,
        attention_bias: torch.Tensor | None = None,
        packed_seq_params: PackedSeqParams | None = None,
        sequence_len_offset: int | None = None,
        *,
        inference_params: BaseInferenceContext | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert packed_seq_params is not None
        cu_seqlens = packed_seq_params.cu_seqlens_q

        # Register weight gradient hooks on first forward (after submodules are created)
        self._maybe_register_weight_grad_hooks()

        # Only do sequence parallel gather/scatter if TP > 1 (SP requires TP > 1)
        tp_world_size = mpu.get_tensor_model_parallel_world_size()
        do_sequence_parallel = self.config.sequence_parallel and tp_world_size > 1
        skip_sp_gather = (
            hasattr(packed_seq_params, "skip_sequence_parallel_gather")
            and packed_seq_params.skip_sequence_parallel_gather
        )
        if skip_sp_gather:
            do_sequence_parallel = False

        if do_sequence_parallel:
            hidden_states = tensor_parallel.gather_from_sequence_parallel_region(
                hidden_states, group=mpu.get_tensor_model_parallel_group()
            )
            # FIX #1: Scale ACTIVATION gradients by 1/TP.
            # gather_backward uses reduce_scatter which sums gradients.
            # Since all TP ranks compute identical gradients, the sum is TP× too large.
            hidden_states = scale_gradient(hidden_states, 1.0 / tp_world_size)

        if mpu.get_context_parallel_world_size() > 1:
            cp_size = mpu.get_context_parallel_world_size()
            hidden_states_list = dist.nn.all_gather(
                hidden_states,
                group=mpu.get_context_parallel_group(),
            )

            # Build per-sequence views at training time so the HF-format attention
            # mask follows the active context-parallel split.
            whole_hidden_states_list = []

            local_cu_seqlens = cu_seqlens // cp_size
            for i in range(len(cu_seqlens) - 1):
                seqlen = cu_seqlens[i + 1] - cu_seqlens[i]
                chunk_size = seqlen // 2 // cp_size
                whole_hidden_states_list.extend(
                    [
                        hidden_states_list[cp_rank][local_cu_seqlens[i] : local_cu_seqlens[i] + chunk_size]
                        for cp_rank in range(cp_size)
                    ]
                    + [
                        hidden_states_list[cp_rank][local_cu_seqlens[i] + chunk_size : local_cu_seqlens[i + 1]]
                        for cp_rank in range(cp_size)
                    ][::-1],
                )
            hidden_states = torch.cat(whole_hidden_states_list, dim=0)

        position_ids = []
        for i in range(len(cu_seqlens) - 1):
            seqlen = cu_seqlens[i + 1] - cu_seqlens[i]
            position_ids.append(torch.arange(seqlen, device=hidden_states.device))
        position_ids = torch.cat(position_ids, dim=0).unsqueeze(0)
        hidden_states = hidden_states.permute(1, 0, 2)  # [bsz, seq_len, hidden_dim]

        output = self.hf_forward(hidden_states, position_ids, packed_seq_params)
        bias = None

        # Permute back to megatron format before sequence parallel scatter
        output = output.permute(1, 0, 2)  # [seq_len, bsz, hidden_dim]

        if mpu.get_context_parallel_world_size() > 1:
            cp_rank = mpu.get_context_parallel_rank()
            output_list = []
            for i in range(len(cu_seqlens) - 1):
                seqlen = cu_seqlens[i + 1] - cu_seqlens[i]
                chunk_size = seqlen // 2 // cp_size
                seq = output[cu_seqlens[i] : cu_seqlens[i + 1]]
                chunks = torch.chunk(seq, 2 * cp_size, dim=0)
                output_list.append(chunks[cp_rank])
                output_list.append(chunks[2 * cp_size - 1 - cp_rank])
            output = torch.cat(output_list, dim=0)

        if do_sequence_parallel:
            output = tensor_parallel.scatter_to_sequence_parallel_region(
                output, group=mpu.get_tensor_model_parallel_group()
            )

        return output, bias

    @abstractmethod
    def hf_forward(self, hidden_states, position_ids, packed_seq_params):
        """Huggingface forward function"""

    def _build_attention_mask(
        self,
        batch_size: int,
        seq_len: int,
        dtype: torch.dtype,
        device: torch.device,
        sliding_window: int | None = None,
    ) -> torch.Tensor:
        """
        Create additive causal or sliding-window causal mask in HF format [B, 1, S, S].
        Values are 0 for allowed positions and -inf for disallowed positions in the attention logits space.
        """
        neg_inf = torch.finfo(dtype).min
        arange = torch.arange(seq_len, device=device)

        # Disallow attending to future tokens (upper triangle)
        future_mask = arange[None, :] > arange[:, None]

        if sliding_window is not None and sliding_window > 0:
            # Disallow attending beyond the sliding window to the past
            past_too_far_mask = (arange[:, None] - arange[None, :]) > sliding_window
            invalid = future_mask | past_too_far_mask
        else:
            invalid = future_mask

        attn_mask_2d = torch.zeros((seq_len, seq_len), dtype=dtype, device=device)
        attn_mask_2d = attn_mask_2d.masked_fill(invalid, neg_inf)
        attn_mask = attn_mask_2d.view(1, 1, seq_len, seq_len).expand(batch_size, 1, seq_len, seq_len)
        return attn_mask
