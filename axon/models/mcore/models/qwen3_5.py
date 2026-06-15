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
Qwen3.5 hybrid attention wrapper for Megatron-Core.

Handles both GDN (linear_attention) and full attention layers using
HuggingFace modules wrapped in the Megatron SP/CP/TP framework.

Key difference from Qwen3-Next: Qwen3.5 uses separate in_proj_{qkv,z,b,a}
instead of merged in_proj_{qkvz,ba}, and has attn_output_gate on full
attention layers.
"""

import torch
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5Attention,
    Qwen3_5GatedDeltaNet,
    Qwen3_5RMSNorm,
    Qwen3_5TextRotaryEmbedding,
)

from axon.models.mcore.models.huggingface_attention import HuggingfaceAttention


class Attention(HuggingfaceAttention):
    """Wraps Qwen3.5 HF attention modules (GDN or full) in Megatron-Core."""

    def __init__(
        self,
        config,
        hf_config,
        layer_number: int,
        layer_type: str = None,
        cp_comm_type: str = "p2p",
        model_comm_pgs=None,
        pg_collection=None,
    ):
        super().__init__(
            config,
            hf_config,
            layer_number,
            cp_comm_type,
            model_comm_pgs,
            pg_collection,
        )

        # Qwen3.5 is a VL model — text config is nested
        text_config = hf_config.text_config if hasattr(hf_config, "text_config") else hf_config
        self.text_config = text_config
        self.text_config._attn_implementation = "eager"

        if layer_type is not None:
            self.layer_type = layer_type
        else:
            self.layer_type = text_config.layer_types[self.hf_layer_idx]

        if self.layer_type == "linear_attention":
            self.linear_attn = Qwen3_5GatedDeltaNet(text_config, self.hf_layer_idx)
        elif self.layer_type == "full_attention":
            self.rotary_emb = Qwen3_5TextRotaryEmbedding(config=text_config)
            self.self_attn = Qwen3_5Attention(text_config, self.hf_layer_idx)

        self.input_layernorm = Qwen3_5RMSNorm(text_config.hidden_size, eps=text_config.rms_norm_eps)

    def hf_forward(self, hidden_states, position_ids, packed_seq_params):
        hidden_states = self.input_layernorm(hidden_states)

        cu_seqlens = packed_seq_params.cu_seqlens_q
        num_seqs = len(cu_seqlens) - 1

        # For packed sequences with eager attention, process each sequence
        # separately to avoid building a massive [total_len, total_len] attention matrix.
        # For GDN layers, pass cu_seqlens so conv1d handles sequence boundaries correctly.
        if num_seqs > 1 or cu_seqlens[-1] > 8192:
            outputs = []
            for i in range(num_seqs):
                start_idx = cu_seqlens[i].item() if isinstance(cu_seqlens[i], torch.Tensor) else cu_seqlens[i]
                end_idx = cu_seqlens[i + 1].item() if isinstance(cu_seqlens[i + 1], torch.Tensor) else cu_seqlens[i + 1]
                seq_len = end_idx - start_idx

                seq_hidden = hidden_states[:, start_idx:end_idx, :]
                seq_position_ids = position_ids[:, start_idx:end_idx]

                mask = self._build_attention_mask(
                    batch_size=1,
                    seq_len=seq_len,
                    dtype=seq_hidden.dtype,
                    device=seq_hidden.device,
                )

                if self.layer_type == "linear_attention":
                    seq_output = self.linear_attn(
                        hidden_states=seq_hidden,
                        attention_mask=mask,
                    )
                elif self.layer_type == "full_attention":
                    seq_position_embeddings = self.rotary_emb(seq_hidden, seq_position_ids)
                    seq_output, _ = self.self_attn(
                        hidden_states=seq_hidden,
                        attention_mask=mask,
                        position_ids=seq_position_ids,
                        position_embeddings=seq_position_embeddings,
                    )
                outputs.append(seq_output)

            hidden_states = torch.cat(outputs, dim=1)
        else:
            bsz, seq_len, _ = hidden_states.shape
            mask = self._build_attention_mask(
                batch_size=bsz,
                seq_len=seq_len,
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )

            if self.layer_type == "linear_attention":
                hidden_states = self.linear_attn(
                    hidden_states=hidden_states,
                    attention_mask=mask,
                )
            elif self.layer_type == "full_attention":
                position_embeddings = self.rotary_emb(hidden_states, position_ids)
                hidden_states, _ = self.self_attn(
                    hidden_states=hidden_states,
                    attention_mask=mask,
                    position_ids=position_ids,
                    position_embeddings=position_embeddings,
                )

        return hidden_states
