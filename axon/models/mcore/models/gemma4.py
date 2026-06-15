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
Gemma4 custom modules for Megatron-Core integration.

Gemma4 (google/gemma-4-26B-A4B-it, google/gemma-4-31B-it) combines:
- Mixed attention: sliding window (head_dim=256) and full attention
  (global_head_dim=512, k_eq_v=True) in 5:1 ratio
- Dense MLP + optional parallel MoE (128 experts, top-k=8) — outputs summed
- Four layernorms per decoder layer + three extra when MoE enabled
- Per-head QK normalization, layer-type-specific RoPE
- Embedding scaling by sqrt(hidden_size)

Components:
- Gemma4Attention: HF attention wrapper for variable head_dim + layer-type RoPE
- Gemma4Router: Gemma4's custom routing logic for Megatron's MoE infrastructure
- Gemma4MoEBlock: MoE path using Megatron's MoELayer (EP/ETP-capable)
- Gemma4TransformerLayer: Extends Glm4TransformerLayer with MoE support

Attention and router scale/norm are non-TP HF modules; expert weights use
Megatron's GroupedMLP for EP/ETP sharding; dense MLP uses Megatron TP.
"""

import torch
import torch.distributed as dist
import torch.nn as nn
from flash_attn import flash_attn_varlen_func as _fa2_varlen_func
from mbridge.models.glm4_vl.transformer_layer import Glm4TransformerLayer
from megatron.core import mpu, parallel_state, tensor_parallel
from megatron.core.extensions.transformer_engine import (
    TEColumnParallelGroupedLinear,
    TERowParallelGroupedLinear,
)
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.moe.experts import TEGroupedMLP
from megatron.core.transformer.moe.moe_layer import MoELayer, MoESubmodules
from megatron.core.transformer.moe.router import Router
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.utils import make_viewless_tensor
from transformers.models.gemma4.modeling_gemma4 import (
    Gemma4RMSNorm,
    Gemma4TextRotaryEmbedding,
    apply_rotary_pos_emb,
)

from axon.models.mcore.models.huggingface_attention import scale_gradient

try:
    from flash_attn_interface import flash_attn_varlen_func as _fa3_varlen_func

    _HAS_FA3 = True
except ImportError:
    _fa3_varlen_func = None
    _HAS_FA3 = False

# Prefer FA3 (Hopper) when available; fall back to FA2.
flash_attn_varlen_func = _fa3_varlen_func if _HAS_FA3 else _fa2_varlen_func

# Max symmetric head_dim supported by the flash_attn kernel we dispatch to.
FLASH_ATTN_MAX_HEADDIM = 256


@torch.compile(dynamic=True)
def _gemma4_fp32_rms_norm(hidden_states: torch.Tensor, weight: torch.Tensor | None, eps: float) -> torch.Tensor:
    """Fused fp32 RMSNorm (norm in fp32, weight multiply in fp32, round once).

    Single compiled kernel reused by ``Gemma4Fp32RMSNorm`` and by the
    post-creation wrapper that rewraps TE-owned norms in mbridge/gemma4.py.
    Cuts per-layer overhead from ~7 unfused kernel launches to 1.
    """
    orig_dtype = hidden_states.dtype
    x = hidden_states.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    if weight is not None:
        x = x * weight.float()
    return x.to(orig_dtype)


class Gemma4Fp32RMSNorm(MegatronModule):
    """Pure-PyTorch RMSNorm matching HF's Gemma4RMSNorm precision.

    Used to replace TELayerNormColumnParallelLinear's fused norm for
    ``pre_mlp_layernorm``: the fused TE kernel does the weight multiply in
    bf16, while HF/vLLM both do it in fp32. This custom module matches
    HF/vLLM by upcasting to fp32 for the whole norm + weight multiply,
    rounding once to input dtype at the end.

    Deliberately avoids TE's module stack (no workspace/FP8 state) to
    prevent clashes with vLLM's KV cache allocator in hybrid-engine mode
    (unfused TENorm caused illegal memory access in vLLM KV cache).
    """

    def __init__(self, config, hidden_size: int, eps: float = 1e-6, **kwargs):
        super().__init__(config=config)
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return _gemma4_fp32_rms_norm(hidden_states, self.weight, self.eps)


class Gemma4SelfAttention(MegatronModule):
    """TP-sharded Gemma4 self-attention, HF-equivalent op-by-op.

    q_proj / o_proj are ColumnParallel / RowParallel. K/V are either
    ColumnParallel (when num_kv_heads % tp_size == 0) or plain replicated
    Linear + per-rank slice (when num_kv_heads < tp_size).

    Forward: q/k/v proj → per-head q_norm/k_norm/v_norm → RoPE (layer-type
    specific) → GQA-repeat → FlashAttention (head_dim ≤ 256) or SDPA
    (head_dim = 512 for full-attention layers).
    """

    def __init__(
        self,
        config,
        hf_config,
        layer_number: int,
        cp_comm_type: str = "p2p",
        model_comm_pgs=None,
        pg_collection=None,
    ):
        super().__init__(config=config)
        self.config = config
        self.hf_config = hf_config
        self.layer_number = layer_number
        self.hf_layer_idx = layer_number - 1

        # Layer-type-dependent geometry (sliding vs full attention)
        layer_types = getattr(hf_config, "layer_types", None)
        self.layer_type = layer_types[self.hf_layer_idx] if layer_types else "full_attention"
        self.is_sliding = self.layer_type == "sliding_attention"
        self.sliding_window = hf_config.sliding_window if self.is_sliding else None

        # Per-layer head_dim: global_head_dim for full_attention, head_dim for sliding
        global_hd = getattr(hf_config, "global_head_dim", None)
        self.head_dim = global_hd if (not self.is_sliding and global_hd) else hf_config.head_dim

        # Per-layer num_kv_heads: num_global_key_value_heads for full (with k_eq_v),
        # else num_key_value_heads
        self.use_alternative_attention = getattr(hf_config, "attention_k_eq_v", False) and not self.is_sliding
        self.num_attention_heads = hf_config.num_attention_heads
        if self.use_alternative_attention:
            self.num_kv_heads = hf_config.num_global_key_value_heads
        else:
            self.num_kv_heads = hf_config.num_key_value_heads
        assert self.num_attention_heads % self.num_kv_heads == 0, (
            f"num_heads ({self.num_attention_heads}) must be divisible by num_kv_heads ({self.num_kv_heads})"
        )
        self.num_kv_groups = self.num_attention_heads // self.num_kv_heads

        # TP info
        self.tp_size = parallel_state.get_tensor_model_parallel_world_size()
        self.tp_rank = parallel_state.get_tensor_model_parallel_rank()
        assert self.num_attention_heads % self.tp_size == 0, (
            f"num_attention_heads ({self.num_attention_heads}) must be divisible by tp_size ({self.tp_size})"
        )
        self.num_heads_per_rank = self.num_attention_heads // self.tp_size

        # Decide on KV sharding strategy. We want to use ColumnParallelLinear for
        # K/V when we can cleanly shard along num_kv_heads; fall back to a
        # replicated plain Linear when num_kv_heads < tp_size (and slice per rank).
        self._kv_is_sharded = self.num_kv_heads >= self.tp_size and self.num_kv_heads % self.tp_size == 0

        if self._kv_is_sharded:
            # Each rank owns num_kv_heads / tp_size contiguous KV heads.
            kv_per_rank = self.num_kv_heads // self.tp_size
            kv_start = self.tp_rank * kv_per_rank
            kv_end = kv_start + kv_per_rank
        else:
            # num_kv_heads < tp_size: each KV head is shared across
            # (tp_size / num_kv_heads) ranks; K/V weights are replicated and
            # each rank slices the one KV head it uses post-projection.
            assert self.tp_size % self.num_kv_heads == 0, (
                f"when num_kv_heads < tp_size, we require tp_size divisible "
                f"by num_kv_heads: got {self.num_kv_heads} kv heads, tp_size={self.tp_size}"
            )
            ranks_per_kv = self.tp_size // self.num_kv_heads
            kv_idx = self.tp_rank // ranks_per_kv
            kv_start = kv_idx
            kv_end = kv_idx + 1
        self.kv_head_start = kv_start
        self.kv_head_end = kv_end
        # KV heads this rank actually feeds into attention (after the K/V path).
        self.num_kv_heads_local = kv_end - kv_start

        attention_bias = bool(getattr(hf_config, "attention_bias", False))

        # q_proj: ColumnParallelLinear sharding the output dim
        # (num_heads * head_dim) -> (num_heads/tp * head_dim) per rank
        self.q_proj = ColumnParallelLinear(
            hf_config.hidden_size,
            self.num_attention_heads * self.head_dim,
            config=config,
            init_method=config.init_method,
            bias=attention_bias,
            gather_output=False,
            skip_bias_add=False,
            is_expert=False,
        )

        # k_proj / v_proj:
        #   - If sharded (num_kv_heads divisible by tp_size): ColumnParallelLinear,
        #     output dim = num_kv_heads * head_dim, split to num_kv_heads/tp * head_dim
        #     per rank. Megatron handles SP gather and grad math natively.
        #   - Otherwise: plain nn.Linear (replicated on every rank) and slice per-rank
        #     post-projection.
        if self._kv_is_sharded:
            self.k_proj = ColumnParallelLinear(
                hf_config.hidden_size,
                self.num_kv_heads * self.head_dim,
                config=config,
                init_method=config.init_method,
                bias=attention_bias,
                gather_output=False,
                skip_bias_add=False,
                is_expert=False,
            )
            if not self.use_alternative_attention:
                self.v_proj = ColumnParallelLinear(
                    hf_config.hidden_size,
                    self.num_kv_heads * self.head_dim,
                    config=config,
                    init_method=config.init_method,
                    bias=attention_bias,
                    gather_output=False,
                    skip_bias_add=False,
                    is_expert=False,
                )
            else:
                # k_eq_v: no v_proj; value states alias key states (matches HF).
                self.v_proj = None
        else:
            self.k_proj = nn.Linear(
                hf_config.hidden_size,
                self.num_kv_heads * self.head_dim,
                bias=attention_bias,
            )
            if not self.use_alternative_attention:
                self.v_proj = nn.Linear(
                    hf_config.hidden_size,
                    self.num_kv_heads * self.head_dim,
                    bias=attention_bias,
                )
            else:
                self.v_proj = None

        # Per-head RMSNorm on q/k (with learned scale) and v (no scale).
        eps = hf_config.rms_norm_eps
        self.q_norm = Gemma4RMSNorm(self.head_dim, eps=eps, with_scale=True)
        self.k_norm = Gemma4RMSNorm(self.head_dim, eps=eps, with_scale=True)
        self.v_norm = Gemma4RMSNorm(self.head_dim, eps=eps, with_scale=False)

        # o_proj: RowParallelLinear sharding the input dim
        # (num_heads * head_dim) -> (num_heads/tp * head_dim) per rank (input)
        self.o_proj = RowParallelLinear(
            self.num_attention_heads * self.head_dim,
            hf_config.hidden_size,
            config=config,
            init_method=config.output_layer_init_method,
            bias=attention_bias,
            input_is_parallel=True,
            skip_bias_add=False,
            is_expert=False,
        )

        # Per-layer-type rotary. HF's class caches inv_freq buffers for each
        # layer_type in config.layer_types; forward dispatches on layer_type.
        self.rotary_emb = Gemma4TextRotaryEmbedding(hf_config)

        # Gradient-scaling bookkeeping (mirrors HuggingfaceAttention).
        self._weight_grad_hooks_registered = False

    def _maybe_register_weight_grad_hooks(self):
        """Scale SP-replicated weight grads by 1/TP.

        Under SP, the gather's backward reduce_scatter sums identical per-rank
        grads across TP → TP× too large. Applies to q/k/v_norm (always
        replicated) and k/v_proj when kept as plain Linear.
        """
        if self._weight_grad_hooks_registered:
            return
        tp_world_size = mpu.get_tensor_model_parallel_world_size()
        if tp_world_size > 1 and self.config.sequence_parallel:
            scale = 1.0 / tp_world_size
            replicated = [self.q_norm, self.k_norm, self.v_norm]
            if not self._kv_is_sharded:
                replicated.append(self.k_proj)
                if self.v_proj is not None:
                    replicated.append(self.v_proj)

            def make_hook(s):
                def hook(grad):
                    return grad * s if grad is not None else grad

                return hook

            for mod in replicated:
                for param in mod.parameters():
                    if param.requires_grad:
                        param.register_hook(make_hook(scale))
        self._weight_grad_hooks_registered = True

    def _cp_gather_and_reorder(self, x: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
        """Undo CP zigzag chunking: [s/cp, b, *] → [s, b, *] via all-gather.

        Megatron CP shards each sequence into ``2*cp_size`` chunks, giving each
        rank chunks ``(i, 2*cp_size-1-i)`` for load-balanced causal attention.
        """
        cp_size = parallel_state.get_context_parallel_world_size()
        if cp_size <= 1:
            return x
        cp_group = parallel_state.get_context_parallel_group()
        # Use autograd-aware all_gather so backward becomes reduce_scatter.
        gathered = dist.nn.all_gather(x.contiguous(), group=cp_group)
        # gathered: list of cp_size tensors, each [s/cp, b, *]

        whole = []
        local_cu = cu_seqlens // cp_size
        for i in range(len(cu_seqlens) - 1):
            seqlen = cu_seqlens[i + 1] - cu_seqlens[i]
            chunk_size = seqlen // 2 // cp_size
            # First half — chunk index `cp_rank` from each CP rank
            whole.extend(gathered[cp_rank][local_cu[i] : local_cu[i] + chunk_size] for cp_rank in range(cp_size))
            # Second half — chunk index `2*cp_size - 1 - cp_rank`, reversed
            whole.extend(
                gathered[cp_rank][local_cu[i] + chunk_size : local_cu[i + 1]] for cp_rank in reversed(range(cp_size))
            )
        return torch.cat(whole, dim=0)

    def _cp_scatter_zigzag(self, x: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
        """Inverse of ``_cp_gather_and_reorder``: extract this CP rank's chunks.

        Args:
            x: ``[s, b, *]`` — full sequence (megatron layout).
            cu_seqlens: cumulative sequence lengths in the *full* sequence.

        Returns:
            ``[s/cp, b, *]`` — this rank's two zigzag chunks per sequence.
        """
        cp_size = parallel_state.get_context_parallel_world_size()
        if cp_size <= 1:
            return x
        cp_rank = parallel_state.get_context_parallel_rank()
        out = []
        for i in range(len(cu_seqlens) - 1):
            seq = x[cu_seqlens[i] : cu_seqlens[i + 1]]
            chunks = torch.chunk(seq, 2 * cp_size, dim=0)
            out.append(chunks[cp_rank])
            out.append(chunks[2 * cp_size - 1 - cp_rank])
        return torch.cat(out, dim=0)

    def _flash_window_size(self):
        """Convert HF ``sliding_window=W`` to flash_attn ``(left, right)``.

        HF: W keys including current ([i-W+1, i]) → flash left=W-1, right=0
        (causal clamps right to i). None → (-1, -1) for no window.
        """
        if self.sliding_window is not None and self.sliding_window > 0:
            return (self.sliding_window - 1, 0)
        return (-1, -1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        key_value_states: torch.Tensor | None = None,
        inference_context=None,
        rotary_pos_emb=None,
        rotary_pos_cos=None,
        rotary_pos_sin=None,
        attention_bias=None,
        packed_seq_params=None,
        sequence_len_offset=None,
        *,
        inference_params=None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward in Megatron layout.

        Input ``hidden_states`` is ``[s/tp, b, h]`` when SP is enabled, else
        ``[s, b, h]``. Output follows the same convention.
        """
        assert packed_seq_params is not None
        self._maybe_register_weight_grad_hooks()

        cu_seqlens = packed_seq_params.cu_seqlens_q
        num_seqs = len(cu_seqlens) - 1

        tp_world_size = mpu.get_tensor_model_parallel_world_size()
        do_sp = self.config.sequence_parallel and tp_world_size > 1
        skip_sp_gather = getattr(packed_seq_params, "skip_sequence_parallel_gather", False)
        if skip_sp_gather:
            do_sp = False

        cp_size = parallel_state.get_context_parallel_world_size()

        # Replicated K/V needs the full sequence for its plain nn.Linear.
        # Sharded K/V (ColumnParallelLinear) handles SP+CP gather internally.
        need_full_hidden = not self._kv_is_sharded
        if do_sp and need_full_hidden:
            hidden_states_full = tensor_parallel.gather_from_sequence_parallel_region(
                hidden_states, group=mpu.get_tensor_model_parallel_group()
            )
            # gather backward reduce_scatters identical TP copies → scale by 1/tp.
            hidden_states_full = scale_gradient(hidden_states_full, 1.0 / tp_world_size)
        else:
            hidden_states_full = hidden_states
        if cp_size > 1 and need_full_hidden:
            hidden_states_full = self._cp_gather_and_reorder(hidden_states_full, cu_seqlens)

        # Build per-token position_ids from cu_seqlens (matches HF attention).
        position_ids_list = []
        for i in range(num_seqs):
            seqlen = cu_seqlens[i + 1] - cu_seqlens[i]
            position_ids_list.append(torch.arange(seqlen, device=hidden_states.device))
        position_ids = torch.cat(position_ids_list, dim=0).unsqueeze(0)

        q_out, _ = self.q_proj(hidden_states)
        if self._kv_is_sharded:
            k_out, _ = self.k_proj(hidden_states)
            if self.v_proj is not None:
                v_out, _ = self.v_proj(hidden_states)
            else:
                v_out = k_out  # k_eq_v — same per-rank slice as K
        else:
            k_out = self.k_proj(hidden_states_full)
            v_out = self.v_proj(hidden_states_full) if self.v_proj is not None else k_out

        # CP-gather q (and sharded k/v). Replicated k/v was gathered above.
        if cp_size > 1:
            q_out = self._cp_gather_and_reorder(q_out, cu_seqlens)
            if self._kv_is_sharded:
                k_out = self._cp_gather_and_reorder(k_out, cu_seqlens)
                if self.v_proj is not None:
                    v_out = self._cp_gather_and_reorder(v_out, cu_seqlens)
                else:
                    v_out = k_out

        # Permute [s,b,h] → [b,s,h] and reshape to per-head views.
        num_kv_view = self.num_kv_heads_local if self._kv_is_sharded else self.num_kv_heads
        q_out_bsh = q_out.permute(1, 0, 2)
        k_out_bsh = k_out.permute(1, 0, 2)
        v_out_bsh = v_out.permute(1, 0, 2)
        bsz, seq_len, _ = q_out_bsh.shape
        q = q_out_bsh.view(bsz, seq_len, self.num_heads_per_rank, self.head_dim)
        k = k_out_bsh.view(bsz, seq_len, num_kv_view, self.head_dim)
        v = v_out_bsh.view(bsz, seq_len, num_kv_view, self.head_dim)

        # Per-head q/k_norm (with scale), v_norm (no scale) — HF order.
        q = self.q_norm(q)
        k = self.k_norm(k)
        v = self.v_norm(v)

        # For replicated K/V, slice this rank's heads post-norm.
        if not self._kv_is_sharded:
            k = k[:, :, self.kv_head_start : self.kv_head_end, :].contiguous()
            v = v[:, :, self.kv_head_start : self.kv_head_end, :].contiguous()

        cos, sin = self.rotary_emb(q_out_bsh, position_ids, layer_type=self.layer_type)
        q = apply_rotary_pos_emb(q, cos, sin, unsqueeze_dim=2)
        k = apply_rotary_pos_emb(k, cos, sin, unsqueeze_dim=2)

        # Attention kernel dispatch:
        #   Sliding layers (head_dim ≤ 256): flash_attn_varlen_func — native
        #     cu_seqlens, GQA, sliding window, no mask materialization.
        #   Full layers (head_dim = 512, exceeds FA2 support): PyTorch SDPA
        #     with is_causal + enable_gqa (per-sequence loop for packing).
        assert bsz == 1, (
            f"Packed sequences expected bsz=1, got bsz={bsz}. "
            f"Megatron packs all sequences into a single batch item with cu_seqlens."
        )

        if self.head_dim <= FLASH_ATTN_MAX_HEADDIM:
            # Flash attention path (sliding layers)
            q_fa, k_fa, v_fa = q.squeeze(0), k.squeeze(0), v.squeeze(0)
            if isinstance(cu_seqlens, torch.Tensor):
                cu_fa = cu_seqlens.to(dtype=torch.int32, device=q.device)
            else:
                cu_fa = torch.tensor(cu_seqlens, dtype=torch.int32, device=q.device)
            max_seqlen = int((cu_fa[1:] - cu_fa[:-1]).max())
            attn_output = flash_attn_varlen_func(
                q_fa,
                k_fa,
                v_fa,
                cu_fa,
                cu_fa,
                max_seqlen,
                max_seqlen,
                softmax_scale=1.0,
                causal=True,
                window_size=self._flash_window_size(),
            ).unsqueeze(0)
        else:
            # SDPA path (full-attention layers, head_dim = 512)
            q_sdpa = q.transpose(1, 2)
            k_sdpa = k.transpose(1, 2)
            v_sdpa = v.transpose(1, 2)

            if num_seqs > 1:
                # Per-sequence loop for packed sequences.
                outputs = []
                for i in range(num_seqs):
                    si = cu_seqlens[i].item() if isinstance(cu_seqlens[i], torch.Tensor) else cu_seqlens[i]
                    ei = cu_seqlens[i + 1].item() if isinstance(cu_seqlens[i + 1], torch.Tensor) else cu_seqlens[i + 1]
                    outputs.append(
                        torch.nn.functional.scaled_dot_product_attention(
                            q_sdpa[:, :, si:ei, :],
                            k_sdpa[:, :, si:ei, :],
                            v_sdpa[:, :, si:ei, :],
                            scale=1.0,
                            is_causal=True,
                            enable_gqa=True,
                        )
                    )
                attn_output = torch.cat(outputs, dim=2)  # [1, h, s, d]
            else:
                attn_output = torch.nn.functional.scaled_dot_product_attention(
                    q_sdpa,
                    k_sdpa,
                    v_sdpa,
                    scale=1.0,
                    is_causal=True,
                    enable_gqa=True,
                )
            # [1, num_heads, s, d] → [1, s, num_heads, d]
            attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.view(bsz, seq_len, self.num_heads_per_rank * self.head_dim)

        attn_output = attn_output.permute(1, 0, 2)  # [s, b, heads/tp * head_dim]

        # If CP was on, the attention output spans the whole sequence on every
        # rank. Scatter back to this rank's zigzag chunk so o_proj's
        # SP-aware reduce_scatter sees [s/cp, b, ...] as expected.
        if cp_size > 1:
            attn_output = self._cp_scatter_zigzag(attn_output, cu_seqlens)

        # o_proj: RowParallelLinear with SP outputs [s/(tp*cp), b, h] (reduce_scatter);
        # without SP it all-reduces to [s/cp, b, h].
        output, bias = self.o_proj(attn_output)
        return output, bias


class Gemma4Router(Router):
    """Gemma4's custom MoE router compatible with Megatron's MoE infrastructure.

    Gemma4 routing:
      1. RMSNorm(input) * scale * (hidden_size^-0.5)
      2. Linear projection → expert scores
      3. Softmax → top-k selection
      4. Renormalize top-k weights, apply per_expert_scale

    Non-TP parameters: norm (no weight), scale, per_expert_scale, proj.weight.
    These are loaded directly from HF weights, not through Megatron TP sharding.
    """

    def __init__(self, config, hf_config, pg_collection=None):
        # Initialize the Router ABC but skip its default weight creation
        # by calling MegatronModule.__init__ directly
        nn.Module.__init__(self)
        self.config = config
        self.num_experts = config.num_moe_experts
        self.topk = config.moe_router_topk
        self.layer_number = None

        # Gemma4-specific router components (non-TP, loaded from HF weights)
        hidden_size = hf_config.hidden_size
        self.scalar_root_size = hidden_size**-0.5
        self.norm = Gemma4RMSNorm(hidden_size, eps=hf_config.rms_norm_eps, with_scale=False)
        self.proj = nn.Linear(hidden_size, self.num_experts, bias=False)
        self.scale = nn.Parameter(torch.ones(hidden_size))
        self.per_expert_scale = nn.Parameter(torch.ones(self.num_experts))

        # Mark replicated params for SP gradient all-reduce (same as
        # Megatron's TopKRouter line 70-73). Without this, finalize_model_grads
        # skips these and gradients are 1/TP too small.
        if config.sequence_parallel:
            self.proj.weight.sequence_parallel = True
            self.scale.sequence_parallel = True

    def routing(self, logits: torch.Tensor, existing_topk_ids: torch.Tensor | None = None):
        """Top-k routing with fp32-stable selection, input-dtype output.

        Matches vLLM's ``GateLinear(out_dtype=torch.float32)`` path. bf16
        routing with 128 experts × top-k=8 can swap close scores → drift.

        Args:
            logits: Router logits ``[num_tokens, num_experts]``.
            existing_topk_ids: Pre-computed top-k expert indices from vLLM
                (MoE replay). When provided, expert *selection* is forced to
                match vLLM while *weights* are recomputed from the Megatron
                logits for correct gradients.
        """
        orig_dtype = logits.dtype
        logits_f = logits.float()
        router_probs = torch.softmax(logits_f, dim=-1)

        if existing_topk_ids is not None:
            # MoE replay: use vLLM's expert selection, recompute weights.
            # Routing maps may contain -1 sentinels for padding tokens —
            # clamp to valid range for gather, then fall back to fresh
            # routing for any token that had -1.
            has_neg1 = (existing_topk_ids == -1).any(dim=1)
            safe_ids = existing_topk_ids.clamp(min=0)
            top_k_indices = safe_ids
            top_k_weights = router_probs.gather(1, top_k_indices.long())

            if has_neg1.any():
                # For padding tokens only: recompute routing from scratch
                orig_weights, orig_indices = torch.topk(router_probs, k=self.topk, dim=-1)
                top_k_indices = torch.where(has_neg1.unsqueeze(1), orig_indices, safe_ids)
                top_k_weights = torch.where(has_neg1.unsqueeze(1), orig_weights, top_k_weights)
        else:
            top_k_weights, top_k_indices = torch.topk(router_probs, k=self.topk, dim=-1)

        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        top_k_weights = top_k_weights * self.per_expert_scale[top_k_indices].float()

        # Respect moe_router_dtype config to match the standard TopKRouter
        # behavior (which produces probs in router_dtype via router_gating_linear).
        output_dtype = orig_dtype
        if getattr(self.config, "moe_router_dtype", None) == "fp32":
            output_dtype = torch.float32
        elif getattr(self.config, "moe_router_dtype", None) == "fp64":
            output_dtype = torch.float64
        top_k_weights = top_k_weights.to(output_dtype)

        probs = torch.zeros(router_probs.shape, dtype=output_dtype, device=router_probs.device)
        probs.scatter_(1, top_k_indices, top_k_weights)
        routing_map = torch.zeros(router_probs.shape, dtype=torch.bool, device=router_probs.device)
        routing_map.scatter_(1, top_k_indices, True)
        return probs, routing_map

    def forward(self, input: torch.Tensor, existing_topk_ids: torch.Tensor | None = None):
        """Forward: norm → scale → project → route.

        Args:
            input: Hidden states ``[num_tokens, hidden_size]`` or ``[s, b, h]``.
            existing_topk_ids: Pre-computed top-k expert indices for MoE replay.
        """
        if input.ndim == 3:
            input = input.reshape(-1, input.shape[-1])
        hidden_states = self.norm(input)
        hidden_states = hidden_states * self.scale * self.scalar_root_size
        logits = self.proj(hidden_states)
        return self.routing(logits, existing_topk_ids=existing_topk_ids)

    def set_layer_number(self, layer_number: int):
        self.layer_number = layer_number


class Gemma4MoELayer(MoELayer):
    """MoELayer that accepts a decoupled ``router_input``.

    Gemma4 routes from the RAW residual (router applies its own RMSNorm) but
    dispatches experts over the ``pre_feedforward_layernorm_2``-normed tensor.
    Upstream MoELayer feeds the same tensor to both → double-normed router
    input → wrong top-k.

    This subclass intercepts the forward call to temporarily swap the router's
    input to the raw residual, then delegates to the parent ``MoELayer.forward``
    (which may be monkey-patched for MoE replay).  This ensures full
    compatibility with the replay FIFO, recompute, and alltoall dispatcher.
    """

    def forward(self, hidden_states, router_input=None):  # type: ignore[override]
        if router_input is None:
            # No decoupling needed — standard MoELayer path.
            return super().forward(hidden_states)

        # Store the decoupled router input so it persists across activation
        # recompute.  The wrapper must stay active through both the initial
        # forward AND any recompute triggered during backward — a try/finally
        # pattern would restore the original too early (after the first forward
        # but before recompute).
        self._gemma4_router_input = router_input
        original_router_forward = self.router.forward

        if not getattr(self, "_gemma4_router_wrapped", False):

            def _decoupled_router_forward(hidden_states_ignored, existing_topk_ids=None):
                ri = self._gemma4_router_input
                return original_router_forward(ri, existing_topk_ids=existing_topk_ids)

            self.router.forward = _decoupled_router_forward
            self._gemma4_router_wrapped = True
            self._gemma4_original_router_forward = original_router_forward

        output, mlp_bias = super().forward(hidden_states)

        return output, mlp_bias


class Gemma4MoEBlock(MegatronModule):
    """MoE block running in parallel with the dense MLP.

    Returns ``post_ff_layernorm_1(dense_mlp) + post_ff_layernorm_2(moe)``;
    the outer ``post_mlp_layernorm`` is applied by Gemma4TransformerLayer.
    MoE path: router sees RAW residual, experts see pre_ff_layernorm_2(residual).
    """

    def __init__(self, mcore_config, hf_config, layer_number=1, pg_collection=None):
        super().__init__(config=mcore_config)
        hidden_size = hf_config.hidden_size
        eps = hf_config.rms_norm_eps
        self.post_feedforward_layernorm_1 = Gemma4RMSNorm(hidden_size, eps=eps)
        self.pre_feedforward_layernorm_2 = Gemma4RMSNorm(hidden_size, eps=eps)
        self.post_feedforward_layernorm_2 = Gemma4RMSNorm(hidden_size, eps=eps)

        # Mark MoE norm weights for SP gradient all-reduce. These HF norms
        # receive SP-split input ([s/tp, b, h]) so each TP rank sees 1/TP of
        # the gradient. Without this, finalize_model_grads skips them and
        # norm gradients are 1/TP too small.
        if mcore_config.sequence_parallel:
            for norm in (
                self.post_feedforward_layernorm_1,
                self.pre_feedforward_layernorm_2,
                self.post_feedforward_layernorm_2,
            ):
                if hasattr(norm, "weight") and norm.weight is not None:
                    norm.weight.sequence_parallel = True

        from megatron.core.transformer.mlp import MLPSubmodules

        # Experts via TEGroupedMLP (all local experts dispatched in a single
        # batched grouped GEMM through TE's GroupedLinear) — much faster than
        # SequentialMLP's per-expert Python loop, which dominated MoE wall time.
        mlp_submodules = MLPSubmodules(
            linear_fc1=TEColumnParallelGroupedLinear,
            linear_fc2=TERowParallelGroupedLinear,
        )
        submodules = MoESubmodules(experts=ModuleSpec(module=TEGroupedMLP, submodules=mlp_submodules))
        self.moe_layer = Gemma4MoELayer(
            config=mcore_config,
            submodules=submodules,
            layer_number=layer_number,
            pg_collection=pg_collection,
        )

        self.moe_layer.router = Gemma4Router(mcore_config, hf_config, pg_collection=pg_collection)
        self.moe_layer.router.set_layer_number(layer_number)

    def forward(self, dense_mlp_output, residual):
        """Returns post_ff_ln_1(dense) + post_ff_ln_2(moe) in megatron [s,b,h] layout.

        Outer ``post_mlp_layernorm`` is applied by Gemma4TransformerLayer.
        """
        mlp_hf = dense_mlp_output.permute(1, 0, 2)
        mlp_normed = self.post_feedforward_layernorm_1(mlp_hf)

        # Router gets RAW residual; experts get pre_ff_layernorm_2(residual).
        residual_hf = residual.permute(1, 0, 2)
        expert_input_mcore = self.pre_feedforward_layernorm_2(residual_hf).permute(1, 0, 2)
        moe_output, _bias = self.moe_layer(expert_input_mcore, router_input=residual)

        moe_normed = self.post_feedforward_layernorm_2(moe_output.permute(1, 0, 2))
        return (mlp_normed + moe_normed).permute(1, 0, 2)


class Gemma4TransformerLayer(Glm4TransformerLayer):
    """Gemma4 transformer layer with parallel MoE + ``layer_scalar`` gate.

    ``_forward_mlp`` sums the dense MLP and (when present) the MoE
    output. ``layer_scalar`` is a per-layer scalar buffer (~0.04–0.99
    range) applied after the residual add, matching HF
    ``Gemma4TextDecoderLayer.forward``.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Persistent=True by default — mbridge loads the trained value.
        self.register_buffer("layer_scalar", torch.ones(1))

        # Build moe_block in __init__ so its expert params are registered
        # with Megatron's MoE-aware sharding from the start.
        hf_config = getattr(self.config, "_gemma4_hf_config", None)
        enable_moe = getattr(self.config, "_gemma4_enable_moe", False)
        if enable_moe and hf_config is not None:
            # Capture device/dtype from an already-built layer param
            # before constructing — Gemma4RMSNorm uses torch.ones (CPU/fp32)
            # regardless of the active device context.
            ref = next(
                (p for p in self.parameters(recurse=True) if not p.is_meta and p.numel() > 0),
                None,
            )
            moe_block = Gemma4MoEBlock(
                mcore_config=self.config,
                hf_config=hf_config,
                layer_number=self.layer_number,
            )
            if ref is not None:
                moe_block = moe_block.to(device=ref.device, dtype=ref.dtype)
            self.moe_block = moe_block

    def forward(self, *args, **kwargs):
        import inspect

        attn_params = inspect.signature(Glm4TransformerLayer._forward_attention).parameters
        attn_kwargs = {k: v for k, v in kwargs.items() if k in attn_params}

        pre_mlp_output, residual, context = Glm4TransformerLayer._forward_attention(self, *args, **attn_kwargs)
        output = self._forward_mlp(pre_mlp_output, residual)
        output = output * self.layer_scalar.to(output.dtype)

        return output, context

    def _forward_mlp(self, pre_mlp_layernorm_output, residual):
        """Dense MLP (+ optional parallel MoE) → outer post_mlp_layernorm."""
        if self.recompute_mlp:
            mlp_output_with_bias = tensor_parallel.checkpoint(self.mlp, False, pre_mlp_layernorm_output)
        else:
            mlp_output_with_bias = self.mlp(pre_mlp_layernorm_output)
        if self.recompute_pre_mlp_layernorm:
            self.pre_mlp_norm_checkpoint.discard_output_and_register_recompute(mlp_output_with_bias[0])
        mlp_output, bias_output = mlp_output_with_bias
        assert bias_output is None

        if hasattr(self, "moe_block") and self.moe_block is not None:
            mlp_output = self.moe_block(mlp_output, residual)
        mlp_output = self.post_mlp_layernorm(mlp_output)

        mlp_output_with_bias = (mlp_output, bias_output)

        with self.bias_dropout_add_exec_handler():
            hidden_states = self.mlp_bda(self.training, self.config.bias_dropout_fusion)(
                mlp_output_with_bias, residual, self.hidden_dropout
            )

        output = make_viewless_tensor(inp=hidden_states, requires_grad=hidden_states.requires_grad, keep_graph=True)
        return output
