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

import logging

import torch
from transformers import PretrainedConfig

from axon.utils.torch import get_torch_device

logger = logging.getLogger(__name__)

VALID_CONFIG_TYPE = {
    "llama",
    "qwen2",
    "qwen2_moe",
    "qwen2_vl",
    "qwen2_5_vl",
    "qwen3",
    "qwen3_moe",
    "qwen3_vl",
    "qwen3_vl_moe",
    "deepseek_v3",
    "minicpmv",
    "minicpmo",
    "mistral",
    "gemma3_text",
    "gemma4",
    "gemma4_text",
    "seed_oss",
    "apertus",
    "glm4v",
}


def _dtype_multiplier(dtype):
    """Return a FLOPS multiplier relative to BF16 for the given dtype."""
    if dtype in (torch.float8_e4m3fn, torch.float8_e5m2, torch.float8_e4m3fnuz, torch.float8_e5m2fnuz):
        return 2.0
    if dtype in (torch.bfloat16, torch.float16):
        return 1.0
    if dtype == torch.float32:
        # PyTorch uses TF32 by default on Ampere+, which is ~0.5x BF16
        return 0.5
    return 1.0


def get_device_flops(unit="T", dtype=torch.bfloat16):
    """Get the theoretical FLOPS (Floating Point Operations Per Second) capacity of the current device.

    Args:
        unit (str): The unit to return the FLOPS in. Supported values are:
            "B" - Billion (1e9)
            "K" - Thousand (1e3)
            "M" - Million (1e6)
            "G" - Giga (1e9)
            "T" - Tera (1e12, default)
            "P" - Peta (1e15)
        dtype (torch.dtype): The training dtype. Peak FLOPS are scaled relative to the BF16 baseline.
            Supported: torch.bfloat16, torch.float16, torch.float32, torch.float8_* variants.

    Returns:
        float: The theoretical FLOPS capacity of the current device in the specified unit.
        Returns float('inf') for unknown GPU types.
    """

    def unit_convert(number, level):
        units = ["B", "K", "M", "G", "T", "P"]
        if number <= 0:
            return number
        ptr = 0
        while ptr < len(units) and units[ptr] != level:
            number /= 1000
            ptr += 1
        return number

    device = get_torch_device()
    if device == torch.cpu:
        device_name = "CPU"
    else:
        device_name = get_torch_device().get_device_name()
    flops = float("inf")  # INF flops for unkown gpu type

    if "CPU" in device_name:
        # use a general CPU flops placeholder to make the function CPU compatible
        flops = 448e9
    elif "GB200" in device_name:
        flops = 1.25e15  # dense bf16 (not 2:4 sparsity)
    elif "B200" in device_name:
        flops = 1.125e15  # dense bf16 (not 2:4 sparsity)
    elif "MI300X" in device_name:
        flops = 1336e12
    elif "H100" in device_name or "H800" in device_name or "H200" in device_name:
        flops = 989e12
    elif "A100" in device_name or "A800" in device_name:
        flops = 312e12
    elif "L40S" in device_name:
        flops = 181e12  # dense bf16 (not 2:4 sparsity)
    elif "L40" in device_name:
        flops = 181.05e12
    elif "A40" in device_name:
        flops = 149.7e12
    elif "L20" in device_name:
        flops = 119.5e12
    elif "H20" in device_name:
        flops = 148e12
    elif "910B" in device_name:
        flops = 354e12
    elif "Ascend910" in device_name:
        flops = 354e12
    elif "RTX 3070 Ti" in device_name:
        flops = 21.75e12
    flops *= _dtype_multiplier(dtype)
    flops_unit = unit_convert(flops, unit)
    return flops_unit


class FlopsCounter:
    """
    Used to count mfu during training loop

    Example:
        flops_counter = FlopsCounter(config)
        flops_achieved, flops_promised = flops_counter.estimate_flops(tokens_list, delta_time)

    """

    def __init__(self, config: PretrainedConfig, dtype: torch.dtype = torch.bfloat16):
        if config.model_type not in VALID_CONFIG_TYPE:
            logger.warning(
                "Only support config type of %s, but got %s. MFU will always be zero.",
                VALID_CONFIG_TYPE,
                config.model_type,
            )

        self.estimate_func = {
            "qwen2": self._estimate_qwen2_flops,
            "llama": self._estimate_qwen2_flops,
            "qwen2_moe": self._estimate_qwen2_moe_flops,
            "qwen2_vl": self._estimate_qwen2_flops,
            "qwen2_5_vl": self._estimate_qwen2_flops,
            "qwen3": self._estimate_qwen2_flops,
            "qwen3_moe": self._estimate_qwen2_moe_flops,
            "qwen3_vl": self._estimate_qwen2_flops,
            "qwen3_vl_moe": self._estimate_qwen2_moe_flops,
            "deepseek_v3": self._estimate_deepseek_v3_flops,
            "minicpmv": self._estimate_qwen2_flops,
            "minicpmo": self._estimate_qwen2_flops,
            "mistral": self._estimate_qwen2_flops,
            "gemma3_text": self._estimate_gemma3_flops,
            "gemma4_text": self._estimate_gemma4_flops,
            "seed_oss": self._estimate_qwen2_flops,
            "apertus": self._estimate_apertus_flops,
            "glm4v": self._estimate_qwen2_flops,
        }
        self.config = getattr(config, "text_config", config)
        self.dtype = dtype

    def _estimate_unknown_flops(self, tokens_sum, batch_seqlens, delta_time):
        return 0

    def _estimate_qwen2_flops(self, tokens_sum, batch_seqlens, delta_time):
        hidden_size = self.config.hidden_size
        vocab_size = self.config.vocab_size
        num_hidden_layers = self.config.num_hidden_layers
        num_key_value_heads = self.config.num_key_value_heads
        num_attention_heads = self.config.num_attention_heads
        intermediate_size = self.config.intermediate_size

        head_dim = getattr(self.config, "head_dim", self.config.hidden_size // self.config.num_attention_heads)
        q_size = num_attention_heads * head_dim
        k_size = num_key_value_heads * head_dim
        v_size = num_key_value_heads * head_dim

        # non-attn per layer parm
        # Qwen2/LLama use SwiGelu, gate, having up and down linear layer in mlp
        mlp_N = hidden_size * intermediate_size * 3
        attn_linear_N = hidden_size * (q_size + k_size + v_size + num_attention_heads * head_dim)
        emd_and_lm_head_N = vocab_size * hidden_size * 2
        # non-attn all_layer parm
        dense_N = (mlp_N + attn_linear_N) * num_hidden_layers + emd_and_lm_head_N
        # non-attn all_layer & all_token fwd & bwd flops
        dense_N_flops = 6 * dense_N * tokens_sum

        # attn all_layer & all_token fwd & bwd flops
        seqlen_square_sum = 0
        for seqlen in batch_seqlens:
            seqlen_square_sum += seqlen * seqlen
        attn_qkv_flops = 12 * seqlen_square_sum * head_dim * num_attention_heads * num_hidden_layers

        # all_layer & all_token fwd & bwd flops
        flops_all_token = dense_N_flops + attn_qkv_flops
        flops_achieved = flops_all_token * (1.0 / delta_time) / 1e12
        return flops_achieved

    def _estimate_deepseek_v3_flops(self, tokens_sum, batch_seqlens, delta_time):
        hidden_size = self.config.hidden_size
        vocab_size = self.config.vocab_size
        moe_intermediate_size = self.config.moe_intermediate_size
        num_hidden_layers = self.config.num_hidden_layers
        first_k_dense_replace = self.config.first_k_dense_replace
        num_query_heads = self.config.num_attention_heads
        moe_num_expert = self.config.n_routed_experts

        moe_topk = self.config.num_experts_per_tok
        share_expert_num = self.config.n_shared_experts

        # non-attn per layer parm
        moe_gata_N = hidden_size * moe_num_expert
        # moe has fc1_1, fc1_2 and fc2 using SwiGLU in ExpertMlp layer & shared experts
        moe_expertmlp_N = hidden_size * moe_intermediate_size * (moe_topk + share_expert_num) * 3
        # MLA attn
        attn_linear_N = 0
        q_head_dim = self.config.qk_nope_head_dim + self.config.qk_rope_head_dim
        if self.config.q_lora_rank is None:
            attn_linear_N += hidden_size * num_query_heads * q_head_dim
        else:
            attn_linear_N += hidden_size * self.config.q_lora_rank
            attn_linear_N += num_query_heads * q_head_dim * self.config.q_lora_rank

        attn_linear_N += hidden_size * (self.config.kv_lora_rank + self.config.qk_rope_head_dim)
        attn_linear_N += (
            num_query_heads
            * (q_head_dim - self.config.qk_rope_head_dim + self.config.v_head_dim)
            * self.config.kv_lora_rank
        )
        attn_linear_N += num_query_heads * self.config.v_head_dim * hidden_size
        emd_and_lm_head_N = vocab_size * hidden_size * 2
        # non-attn all_layer parm
        moe_N = (
            (moe_gata_N + moe_expertmlp_N + attn_linear_N) * (num_hidden_layers - first_k_dense_replace)
            + (hidden_size * self.config.intermediate_size * 3 + attn_linear_N) * first_k_dense_replace
            + emd_and_lm_head_N
        )
        # non-attn all_layer & all_token fwd & bwd flops
        dense_N_flops = 6 * moe_N * tokens_sum

        # attn all_layer & all_token fwd & bwd flops
        seqlen_square_sum = 0
        for seqlen in batch_seqlens:
            seqlen_square_sum += seqlen * seqlen * num_hidden_layers

        # QK^T uses q_head_dim, attn@V uses v_head_dim; with fwd+bwd (3x): 6 * (q + v)
        v_head_dim = self.config.v_head_dim
        attn_qkv_flops = 6 * seqlen_square_sum * (q_head_dim + v_head_dim) * num_query_heads
        # all_layer & all_token fwd & bwk flops
        flops_all_token = dense_N_flops + attn_qkv_flops
        flops_achieved = flops_all_token * (1.0 / delta_time) / 1e12

        return flops_achieved

    def _estimate_qwen2_moe_flops(self, tokens_sum, batch_seqlens, delta_time):
        hidden_size = self.config.hidden_size
        vocab_size = self.config.vocab_size
        num_hidden_layers = self.config.num_hidden_layers
        num_key_value_heads = self.config.num_key_value_heads
        num_attention_heads = self.config.num_attention_heads
        moe_intermediate_size = self.config.moe_intermediate_size
        moe_topk = self.config.num_experts_per_tok
        num_experts = self.config.num_experts

        head_dim = getattr(self.config, "head_dim", self.config.hidden_size // self.config.num_attention_heads)
        q_size = num_attention_heads * head_dim
        k_size = num_key_value_heads * head_dim
        v_size = num_key_value_heads * head_dim

        # non-attn per layer parm
        # gate + routed experts + shared expert
        shared_expert_intermediate_size = getattr(self.config, "shared_expert_intermediate_size", 0)
        moe_mlp_N = (
            hidden_size * moe_topk * moe_intermediate_size * 3
            + hidden_size * num_experts
            + hidden_size * shared_expert_intermediate_size * 3
        )
        attn_linear_N = hidden_size * (q_size + k_size + v_size + num_attention_heads * head_dim)
        emd_and_lm_head_N = vocab_size * hidden_size * 2
        # non-attn all_layer parm
        dense_N = (moe_mlp_N + attn_linear_N) * num_hidden_layers + emd_and_lm_head_N
        # non-attn all_layer & all_token fwd & bwd flops
        dense_N_flops = 6 * dense_N * tokens_sum

        # attn all_layer & all_token fwd & bwd flops
        seqlen_square_sum = 0
        for seqlen in batch_seqlens:
            seqlen_square_sum += seqlen * seqlen
        attn_qkv_flops = 12 * seqlen_square_sum * head_dim * num_attention_heads * num_hidden_layers

        # all_layer & all_token fwd & bwd flops
        flops_all_token = dense_N_flops + attn_qkv_flops
        flops_achieved = flops_all_token * (1.0 / delta_time) / 1e12
        return flops_achieved

    def _estimate_gemma3_flops(self, tokens_sum, batch_seqlens, delta_time):
        hidden_size = self.config.hidden_size
        vocab_size = self.config.vocab_size
        num_hidden_layers = self.config.num_hidden_layers
        num_key_value_heads = self.config.num_key_value_heads
        num_attention_heads = self.config.num_attention_heads
        intermediate_size = self.config.intermediate_size

        head_dim = getattr(self.config, "head_dim", self.config.hidden_size // self.config.num_attention_heads)
        q_size = num_attention_heads * head_dim
        k_size = num_key_value_heads * head_dim
        v_size = num_key_value_heads * head_dim

        # non-attn per layer parm
        # Gemma3 uses GeGLU (gelu_pytorch_tanh), having 3 matrices in MLP (inherited from Gemma2MLP)
        mlp_N = hidden_size * intermediate_size * 3
        attn_linear_N = hidden_size * (q_size + k_size + v_size + num_attention_heads * head_dim)
        emd_and_lm_head_N = vocab_size * hidden_size * 2
        # non-attn all_layer parm
        dense_N = (mlp_N + attn_linear_N) * num_hidden_layers + emd_and_lm_head_N
        # non-attn all_layer & all_token fwd & bwd flops
        dense_N_flops = 6 * dense_N * tokens_sum

        # attn all_layer & all_token fwd & bwd flops
        # Gemma3 alternates between full and sliding window attention based on layer_types
        seqlen_square_sum = 0

        layer_types = getattr(self.config, "layer_types", None)
        sliding_window = getattr(self.config, "sliding_window", 1024)  # default 1024
        # default pattern: every 6th layer is full
        sliding_window_pattern = getattr(self.config, "sliding_window_pattern", 6)

        # If layer_types is not provided, generate it based on sliding_window_pattern
        if layer_types is None and sliding_window is not None and sliding_window_pattern is not None:
            layer_types = [
                "sliding_attention" if bool((i + 1) % sliding_window_pattern) else "full_attention"
                for i in range(num_hidden_layers)
            ]

        if layer_types:
            # Calculate attention flops per layer based on attention type
            for layer_idx in range(num_hidden_layers):
                is_sliding = False
                if layer_types and layer_idx < len(layer_types):
                    is_sliding = layer_types[layer_idx] == "sliding_attention"

                for seqlen in batch_seqlens:
                    if is_sliding and sliding_window:
                        # Sliding window limits each token to attend to at most window_size tokens
                        effective_seqlen = min(seqlen, sliding_window)
                        seqlen_square_sum += seqlen * effective_seqlen
                    else:
                        # Full attention
                        seqlen_square_sum += seqlen * seqlen
        else:
            # If no layer_types config, assume all layers use full attention
            for seqlen in batch_seqlens:
                seqlen_square_sum += seqlen * seqlen
            seqlen_square_sum *= num_hidden_layers

        attn_qkv_flops = 12 * seqlen_square_sum * head_dim * num_attention_heads

        # all_layer & all_token fwd & bwd flops
        flops_all_token = dense_N_flops + attn_qkv_flops
        flops_achieved = flops_all_token * (1.0 / delta_time) / 1e12
        return flops_achieved

    def _estimate_gemma4_flops(self, tokens_sum, batch_seqlens, delta_time):
        """FLOPs estimator for Gemma4 (google/gemma-4-26B-A4B-it, google/gemma-4-31B-it).

        Per-layer geometry varies by layer_type:
          - sliding_attention: head_dim, num_key_value_heads
          - full_attention: global_head_dim, num_global_key_value_heads,
            and k_eq_v (no v_proj) when ``attention_k_eq_v`` is set
        Extras: parallel dense MLP + MoE (when ``enable_moe_block``), per-layer
        input projections (when ``hidden_size_per_layer_input > 0``), and
        double-wide dense MLP on KV-shared layers (when ``use_double_wide_mlp``).
        """
        hidden_size = self.config.hidden_size
        vocab_size = self.config.vocab_size
        num_hidden_layers = self.config.num_hidden_layers
        num_key_value_heads = self.config.num_key_value_heads
        num_attention_heads = self.config.num_attention_heads
        intermediate_size = self.config.intermediate_size

        head_dim = getattr(self.config, "head_dim", hidden_size // num_attention_heads)
        global_head_dim = getattr(self.config, "global_head_dim", None) or head_dim
        num_global_kv_heads = getattr(self.config, "num_global_key_value_heads", None) or num_key_value_heads
        attention_k_eq_v = bool(getattr(self.config, "attention_k_eq_v", False))

        # KV sharing: last `num_kv_shared_layers` layers reuse previous layer's K/V
        # (no k_proj / v_proj params on those layers).
        num_kv_shared_layers = int(getattr(self.config, "num_kv_shared_layers", 0) or 0)
        first_kv_shared_layer_idx = num_hidden_layers - num_kv_shared_layers
        use_double_wide_mlp = bool(getattr(self.config, "use_double_wide_mlp", False))

        # MoE (optional: e.g., 26B-A4B has 128 experts, top-k=8; 31B is dense)
        enable_moe_block = bool(getattr(self.config, "enable_moe_block", False))
        num_experts = int(getattr(self.config, "num_experts", 0) or 0)
        top_k_experts = int(getattr(self.config, "top_k_experts", 0) or 0)
        moe_intermediate_size = int(getattr(self.config, "moe_intermediate_size", 0) or 0)
        moe_active = enable_moe_block and num_experts > 0 and top_k_experts > 0 and moe_intermediate_size > 0

        # Per-layer input (disabled on 26B / 31B — kept for future configs)
        hidden_size_per_layer_input = int(getattr(self.config, "hidden_size_per_layer_input", 0) or 0)
        vocab_size_per_layer_input = int(getattr(self.config, "vocab_size_per_layer_input", 0) or 0)

        # layer_types: 5:1 sliding:full by default, last layer forced to full_attention
        layer_types = getattr(self.config, "layer_types", None)
        sliding_window = getattr(self.config, "sliding_window", None)
        if layer_types is None:
            sliding_window_pattern = 6
            layer_types = [
                "sliding_attention" if bool((i + 1) % sliding_window_pattern) else "full_attention"
                for i in range(num_hidden_layers)
            ]
        if layer_types and layer_types[-1] != "full_attention":
            layer_types = list(layer_types)
            layer_types[-1] = "full_attention"

        # --- Accumulate per-token dense params across all layers ---
        dense_N = 0
        for layer_idx in range(num_hidden_layers):
            is_sliding = layer_types[layer_idx] == "sliding_attention"
            is_kv_shared = num_kv_shared_layers > 0 and layer_idx >= first_kv_shared_layer_idx

            if is_sliding:
                layer_head_dim = head_dim
                layer_num_kv_heads = num_key_value_heads
                layer_k_eq_v = False  # k_eq_v only applies to full layers
            else:
                layer_head_dim = global_head_dim
                layer_num_kv_heads = num_global_kv_heads
                layer_k_eq_v = attention_k_eq_v

            q_size = num_attention_heads * layer_head_dim
            # q_proj + o_proj (always present)
            attn_linear_N = hidden_size * q_size * 2
            if not is_kv_shared:
                # k_proj
                attn_linear_N += hidden_size * (layer_num_kv_heads * layer_head_dim)
                # v_proj — omitted when k_eq_v on full-attention layers
                if not layer_k_eq_v:
                    attn_linear_N += hidden_size * (layer_num_kv_heads * layer_head_dim)
            dense_N += attn_linear_N

            # Dense MLP: GeGLU (gate + up + down). Double-width on KV-shared
            # layers when enabled.
            layer_intermediate_size = intermediate_size * (2 if (use_double_wide_mlp and is_kv_shared) else 1)
            dense_N += hidden_size * layer_intermediate_size * 3

            # Parallel MoE block: router (dense) + top-k activated experts.
            # Each expert is SwiGLU-like (gate_up is packed 2× intermediate).
            if moe_active:
                # Router projection: hidden -> num_experts (run for every token)
                dense_N += hidden_size * num_experts
                # Activated-expert cost per token: top_k × (gate + up + down)
                dense_N += top_k_experts * hidden_size * moe_intermediate_size * 3

            # Per-layer input path: gate + projection (norms ignored)
            if hidden_size_per_layer_input:
                dense_N += hidden_size * hidden_size_per_layer_input * 2

        # Token embedding + LM head (tied weights, but both contribute to fwd/bwd
        # FLOPs in the lm_head matmul; keeping the existing "2×" convention used
        # by other estimators for apples-to-apples comparability).
        dense_N += vocab_size * hidden_size * 2

        # Global per-layer-input projections (shared across decoder layers)
        if hidden_size_per_layer_input:
            # per_layer_model_projection: hidden -> num_hidden_layers * per_layer_hidden
            dense_N += hidden_size * num_hidden_layers * hidden_size_per_layer_input
            # embed_tokens_per_layer (counted as parameter per existing convention)
            dense_N += vocab_size_per_layer_input * num_hidden_layers * hidden_size_per_layer_input

        # Fwd + bwd linear FLOPs: 6 × params × tokens
        dense_N_flops = 6 * dense_N * tokens_sum

        # Attention-kernel FLOPs (QK^T + attn@V): 12 × s × s_eff × head_dim × num_heads per layer.
        # Sliding layers cap the effective kv-length at `sliding_window`.
        attn_qkv_flops = 0
        for layer_idx in range(num_hidden_layers):
            is_sliding = layer_types[layer_idx] == "sliding_attention"
            layer_head_dim = head_dim if is_sliding else global_head_dim
            for seqlen in batch_seqlens:
                if is_sliding and sliding_window:
                    effective_seqlen = min(seqlen, sliding_window)
                    attn_qkv_flops += 12 * seqlen * effective_seqlen * layer_head_dim * num_attention_heads
                else:
                    attn_qkv_flops += 12 * seqlen * seqlen * layer_head_dim * num_attention_heads

        flops_all_token = dense_N_flops + attn_qkv_flops
        flops_achieved = flops_all_token * (1.0 / delta_time) / 1e12
        return flops_achieved

    def _estimate_apertus_flops(self, tokens_sum, batch_seqlens, delta_time):
        hidden_size = self.config.hidden_size
        vocab_size = self.config.vocab_size
        num_hidden_layers = self.config.num_hidden_layers
        num_key_value_heads = self.config.num_key_value_heads
        num_attention_heads = self.config.num_attention_heads
        intermediate_size = self.config.intermediate_size

        head_dim = getattr(self.config, "head_dim", self.config.hidden_size // self.config.num_attention_heads)
        q_size = num_attention_heads * head_dim
        k_size = num_key_value_heads * head_dim
        v_size = num_key_value_heads * head_dim

        # Apertus MLP with XIELU activation uses only 2 linear layers (up_proj, down_proj)
        # No gate_proj for XIELU, unlike SwiGLU which has 3 layers
        mlp_N = hidden_size * intermediate_size * 2
        attn_linear_N = hidden_size * (q_size + k_size + v_size + num_attention_heads * head_dim)

        # ApertusConfig has qk_norm defaulting to True.
        # This adds params for q_norm (on H) and k_norm (on num_kv_heads * head_dim)
        qk_norm_params_per_layer = hidden_size + num_key_value_heads * head_dim  # q_norm + k_norm

        emd_and_lm_head_N = vocab_size * hidden_size * 2
        # non-attn all_layer params
        dense_N = (mlp_N + attn_linear_N + qk_norm_params_per_layer) * num_hidden_layers + emd_and_lm_head_N
        # non-attn all_layer & all_token fwd & bwd flops
        dense_N_flops = 6 * dense_N * tokens_sum

        # attn all_layer & all_token fwd & bwd flops
        seqlen_square_sum = 0
        for seqlen in batch_seqlens:
            seqlen_square_sum += seqlen * seqlen
        attn_qkv_flops = 12 * seqlen_square_sum * head_dim * num_attention_heads * num_hidden_layers

        # all_layer & all_token fwd & bwd flops
        flops_all_token = dense_N_flops + attn_qkv_flops
        flops_achieved = flops_all_token * (1.0 / delta_time) / 1e12
        return flops_achieved

    def estimate_flops(self, batch_seqlens, delta_time):
        """
        Estimate the FLOPS based on the number of valid tokens in the current batch and the time taken.

        Args:
            batch_seqlens (List[int]): A list where each element represents the number of valid tokens in the
                current batch.
            delta_time (float): The time taken to process the batch, in seconds.

        Returns:
            estimated_flops (float): The estimated FLOPS based on the input tokens and time.
            promised_flops (float): The expected FLOPS of the current device.
        """
        tokens_sum = sum(batch_seqlens)
        func = self.estimate_func.get(self.config.model_type, self._estimate_unknown_flops)
        estimated_flops = func(tokens_sum, batch_seqlens, delta_time)
        promised_flops = get_device_flops(dtype=self.dtype)
        return estimated_flops, promised_flops
