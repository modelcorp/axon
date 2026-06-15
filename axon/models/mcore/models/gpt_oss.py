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
import torch
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssAttention, GptOssRMSNorm, GptOssRotaryEmbedding

from axon.models.mcore.models.huggingface_attention import HuggingfaceAttention


class GPTOSSAttention(HuggingfaceAttention):
    def __init__(
        self,
        config,
        hf_config,
        layer_number: int,
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
        print(f"GPTOSSAttention init {layer_number}")
        if GptOssAttention is None:
            raise ImportError("Please install transformers with GPT-OSS support to use GptOssAttention.")

        # GPT-OSS uses eager attention (with sinks and sliding-window handling)
        # Ensure we do not force FlashAttention2 here.
        self.hf_config._attn_implementation = "eager"

        # Cache layer type for mask selection parity with HF
        if hasattr(self.hf_config, "layer_types") and self.hf_config.layer_types is not None:
            self.layer_type = self.hf_config.layer_types[self.hf_layer_idx]
        else:
            self.layer_type = "full_attention"

        self.self_attn = GptOssAttention(self.hf_config, self.hf_layer_idx)
        self.rotary_emb = GptOssRotaryEmbedding(config=self.hf_config)
        self.input_layernorm = GptOssRMSNorm(self.hf_config.hidden_size, eps=self.hf_config.rms_norm_eps)

    def hf_forward(self, hidden_states, position_ids, packed_seq_params):
        hidden_states = self.input_layernorm(hidden_states)

        # Get cu_seqlens for packed sequence handling
        cu_seqlens = packed_seq_params.cu_seqlens_q
        num_seqs = len(cu_seqlens) - 1

        # For packed sequences with eager attention, we must process each sequence
        # separately to avoid building a massive [total_len, total_len] attention matrix
        if num_seqs > 1 or cu_seqlens[-1] > 8192:
            # Process each sequence in the pack separately
            outputs = []
            for i in range(num_seqs):
                start_idx = cu_seqlens[i].item() if isinstance(cu_seqlens[i], torch.Tensor) else cu_seqlens[i]
                end_idx = cu_seqlens[i + 1].item() if isinstance(cu_seqlens[i + 1], torch.Tensor) else cu_seqlens[i + 1]
                seq_len = end_idx - start_idx

                # Extract this sequence
                seq_hidden = hidden_states[:, start_idx:end_idx, :]  # [1, seq_len, hidden]
                seq_position_ids = position_ids[:, start_idx:end_idx]  # [1, seq_len]

                # Build rotary embeddings for this sequence
                seq_position_embeddings = self.rotary_emb(seq_hidden, seq_position_ids)

                # Build attention mask for this sequence only
                sliding_window = (
                    self.hf_config.sliding_window
                    if getattr(self, "layer_type", "full_attention") == "sliding_attention"
                    else None
                )
                mask = self._build_attention_mask(
                    batch_size=1,
                    seq_len=seq_len,
                    dtype=seq_hidden.dtype,
                    device=seq_hidden.device,
                    sliding_window=sliding_window,
                )

                # Run attention for this sequence
                seq_output, _ = self.self_attn(
                    hidden_states=seq_hidden,
                    attention_mask=mask,
                    position_embeddings=seq_position_embeddings,
                    past_key_values=None,
                    cache_position=None,
                )
                outputs.append(seq_output)

            # Concatenate all sequence outputs
            hidden_states = torch.cat(outputs, dim=1)
        else:
            # Single short sequence - can use regular path
            bsz, seq_len, _ = hidden_states.shape

            # Build rotary embeddings
            position_embeddings = self.rotary_emb(hidden_states, position_ids)

            # Build attention mask
            sliding_window = (
                self.hf_config.sliding_window
                if getattr(self, "layer_type", "full_attention") == "sliding_attention"
                else None
            )
            mask = self._build_attention_mask(
                batch_size=bsz,
                seq_len=seq_len,
                dtype=hidden_states.dtype,
                device=hidden_states.device,
                sliding_window=sliding_window,
            )

            # HuggingFace GptOssAttention uses eager attention with sinks and optional sliding window
            hidden_states, _ = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=mask,
                position_embeddings=position_embeddings,
                past_key_values=None,
                cache_position=None,
            )

        return hidden_states
