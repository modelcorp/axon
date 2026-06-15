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
Monkey patch for Qwen3.5 GDN (Gated Delta Net) layers to support packed
sequences (use_remove_padding=True).

Bug:
  When use_remove_padding=True, multiple sequences are packed into a single
  tensor [1, total_tokens, hidden].  The GDN layer's recurrent computation
  (conv1d + chunk_gated_delta_rule) processes this as ONE long sequence,
  causing recurrent state from sequence A to bleed into sequence B.  This
  corrupts the output and causes ~0.07 prob diff vs vLLM (which processes
  each request independently).

Fix:
  Read cu_seqlens from the module-level global (set by fsdp_model_forward)
  and pass it to both causal_conv1d_fn (via seq_idx) and
  chunk_gated_delta_rule so they respect sequence boundaries.
"""

import logging
import types

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def _cu_seqlens_to_seq_idx(cu_seqlens: torch.Tensor, total_len: int) -> torch.Tensor:
    """Convert cu_seqlens [0, 5, 12, 20] → seq_idx [0,0,0,0,0, 1,1,…, 2,2,…].

    Used by causal_conv1d_fn to prevent convolution across sequence boundaries.
    """
    seq_idx = torch.zeros(total_len, dtype=torch.int32, device=cu_seqlens.device)
    for i in range(1, cu_seqlens.shape[0] - 1):
        seq_idx[cu_seqlens[i] :] = i
    return seq_idx


def _gdn_forward_with_cu_seqlens(
    self,
    hidden_states: torch.Tensor,
    cache_params=None,
    cache_position=None,
    attention_mask: torch.Tensor | None = None,
):
    """Replacement forward for Qwen3_5GatedDeltaNet that respects sequence
    boundaries in packed tensors.

    Reads ``cu_seqlens`` from the module-level global set by
    :func:`axon.utils.fsdp.forward_utils.fsdp_model_forward` and passes it
    to ``causal_conv1d_fn`` (via ``seq_idx``) and ``chunk_gated_delta_rule``
    so that recurrent state does not bleed across sequences.
    """
    from causal_conv1d import causal_conv1d_fn

    from axon.utils.fsdp.forward_utils import get_rmpad_cu_seqlens

    # Zero-out padding positions (same as upstream).
    if attention_mask is not None and not torch.all(attention_mask == 1):
        hidden_states = (hidden_states * attention_mask[:, :, None]).to(hidden_states.dtype)

    batch_size, seq_len, _ = hidden_states.shape
    cu_seqlens = get_rmpad_cu_seqlens()

    # ---- Input projections (unchanged from upstream) ----
    mixed_qkv = self.in_proj_qkv(hidden_states)
    z = self.in_proj_z(hidden_states).reshape(batch_size, seq_len, -1, self.head_v_dim)
    b = self.in_proj_b(hidden_states)
    a = self.in_proj_a(hidden_states)

    # ---- Conv1d with sequence boundaries ----
    # Always build seq_idx to keep the tensor count deterministic for
    # gradient-checkpointing recomputation.
    if cu_seqlens is not None and batch_size == 1:
        seq_idx = _cu_seqlens_to_seq_idx(cu_seqlens, seq_len).unsqueeze(0)
    else:
        seq_idx = torch.zeros(batch_size, seq_len, dtype=torch.int32, device=hidden_states.device)

    mixed_qkv = causal_conv1d_fn(
        x=mixed_qkv.transpose(1, 2),
        weight=self.conv1d.weight.squeeze(1),
        bias=self.conv1d.bias,
        activation=self.activation,
        seq_idx=seq_idx,
    ).transpose(1, 2)

    # ---- Split Q, K, V and reshape ----
    query, key, value = torch.split(mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
    query = query.reshape(batch_size, seq_len, -1, self.head_k_dim).contiguous()
    key = key.reshape(batch_size, seq_len, -1, self.head_k_dim).contiguous()
    value = value.reshape(batch_size, seq_len, -1, self.head_v_dim).contiguous()

    # ---- Gating ----
    beta = b.sigmoid()
    g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias.float())

    # ---- GQA head expansion ----
    if self.num_v_heads // self.num_k_heads > 1:
        query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
        key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

    # ---- Delta rule with cu_seqlens ----
    core_attn_out, _ = self.chunk_gated_delta_rule(
        query,
        key,
        value,
        g=g,
        beta=beta,
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=cu_seqlens,
    )

    # ---- RMSNormGated + output projection ----
    core_attn_out = self.norm(
        core_attn_out.reshape(-1, self.head_v_dim),
        z.reshape(-1, self.head_v_dim),
    ).reshape(batch_size, seq_len, -1)
    return self.out_proj(core_attn_out)


def patch_qwen3_5_gdn_attention(model: torch.nn.Module) -> bool:
    """Patch Qwen3.5 GDN layers to handle packed sequences correctly.

    Replaces the forward of every ``Qwen3_5GatedDeltaNet`` layer with
    :func:`_gdn_forward_with_cu_seqlens` which reads ``cu_seqlens`` from
    the thread-local context and passes it to ``causal_conv1d_fn`` and
    ``chunk_gated_delta_rule``.

    Returns ``True`` if any layers were patched.
    """
    patched = 0
    for _name, module in model.named_modules():
        if "GatedDeltaNet" not in type(module).__name__:
            continue
        if not hasattr(module, "in_proj_qkv"):
            continue
        module.forward = types.MethodType(_gdn_forward_with_cu_seqlens, module)
        patched += 1

    if patched > 0:
        logger.info(f"Patched {patched} Qwen3.5 GDN layer(s) with cu_seqlens-aware forward")
    return patched > 0
