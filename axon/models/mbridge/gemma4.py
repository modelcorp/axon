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
#
# load / export weight machinery adapted from mbridge Bridge, BSD-3-Clause (github.com/ISEEKYAN/mbridge).
"""
MBridge bridge for Gemma4 text models (model_type="gemma4" / "gemma4_text").

Supports both dense and MoE variants:
- google/gemma-4-31B-it: Dense, 60 layers, no MoE
- google/gemma-4-26B-A4B-it: MoE, 30 layers, 128 experts, top-k=8
"""

from collections.abc import Generator
from functools import partial

import torch
import torch.nn.functional as F
from mbridge.core import LLMBridge, register_model
from mbridge.core.util import (
    broadcast_from_megatron_pp,
    broadcast_str_from_megatron_pp,
    unwrap_model,
)

from axon.models.mcore.models.gemma4 import Gemma4TransformerLayer


@register_model(["gemma4", "gemma4_text"])
class Gemma4Bridge(LLMBridge):
    """
    Bridge for Gemma4 text models with EP/ETP support for MoE experts.

    Uses HuggingFace attention wrappers (non-TP) for variable head_dim,
    Megatron TEGroupedMLP for MoE experts (EP/ETP-sharded), and Megatron
    TP-parallel dense MLP.
    """

    # HF checkpoint prefix — Gemma4ForConditionalGeneration uses
    # "model.language_model.*", Gemma4ForCausalLM uses "model.*". Detected
    # at load time via _resolve_hf_lm_prefix; the {lm} placeholder is
    # substituted into the mapping tables below.
    _DEFAULT_HF_LM_PREFIX = "model.language_model"

    _DIRECT_MAPPING = {
        "embedding.word_embeddings.weight": "{lm}.embed_tokens.weight",
        "decoder.final_layernorm.weight": "{lm}.norm.weight",
        # With tie_word_embeddings=True, HF doesn't save lm_head.weight —
        # it reuses embed_tokens.weight. Map to the embedding weight so PP
        # stages that have output_layer but not embedding can load it.
        "output_layer.weight": "{lm}.embed_tokens.weight",
    }

    _ATTENTION_MAPPING = {
        "self_attention.q_proj.weight": ["{lm}.layers.{layer_number}.self_attn.q_proj.weight"],
        "self_attention.k_proj.weight": ["{lm}.layers.{layer_number}.self_attn.k_proj.weight"],
        "self_attention.v_proj.weight": ["{lm}.layers.{layer_number}.self_attn.v_proj.weight"],
        "self_attention.o_proj.weight": ["{lm}.layers.{layer_number}.self_attn.o_proj.weight"],
        "self_attention.q_norm.weight": ["{lm}.layers.{layer_number}.self_attn.q_norm.weight"],
        "self_attention.k_norm.weight": ["{lm}.layers.{layer_number}.self_attn.k_norm.weight"],
    }

    _MLP_MAPPING = {
        "mlp.linear_fc1.weight": [
            "{lm}.layers.{layer_number}.mlp.gate_proj.weight",
            "{lm}.layers.{layer_number}.mlp.up_proj.weight",
        ],
        # After unfusing pre_mlp_layernorm from linear_fc1 (for fp32 norm),
        # the weight moves to a standalone `pre_mlp_layernorm.weight` param.
        # Keep the legacy fused name for backward-compat with old checkpoints.
        "pre_mlp_layernorm.weight": ["{lm}.layers.{layer_number}.pre_feedforward_layernorm.weight"],
        "mlp.linear_fc1.layer_norm_weight": ["{lm}.layers.{layer_number}.pre_feedforward_layernorm.weight"],
        "mlp.linear_fc2.weight": ["{lm}.layers.{layer_number}.mlp.down_proj.weight"],
    }

    # MoE router params — small, replicated across TP.
    _MOE_ROUTER_MAPPING = {
        "moe_block.moe_layer.router.scale": ["{lm}.layers.{layer_number}.router.scale"],
        "moe_block.moe_layer.router.per_expert_scale": ["{lm}.layers.{layer_number}.router.per_expert_scale"],
        "moe_block.moe_layer.router.proj.weight": ["{lm}.layers.{layer_number}.router.proj.weight"],
    }

    # MoE expert weights — HF stores stacked per-layer as 3D tensors
    # [num_experts, ...]; mcore TEGroupedMLP stores per-expert as ``linear_fc1.weight{i}``
    # / ``linear_fc2.weight{i}``. The keyword match below uses substring-``in``
    # so both ``linear_fc1.weight`` and ``linear_fc1.weight5`` route correctly.
    _MOE_EXPERT_MAPPING = {
        "linear_fc1.weight": ["{lm}.layers.{layer_number}.experts.gate_up_proj"],
        "linear_fc2.weight": ["{lm}.layers.{layer_number}.experts.down_proj"],
    }

    # MoE extra layernorms (post_ff_1, pre_ff_2, post_ff_2) — replicated.
    _MOE_NORM_MAPPING = {
        "moe_block.post_feedforward_layernorm_1.weight": [
            "{lm}.layers.{layer_number}.post_feedforward_layernorm_1.weight"
        ],
        "moe_block.pre_feedforward_layernorm_2.weight": [
            "{lm}.layers.{layer_number}.pre_feedforward_layernorm_2.weight"
        ],
        "moe_block.post_feedforward_layernorm_2.weight": [
            "{lm}.layers.{layer_number}.post_feedforward_layernorm_2.weight"
        ],
    }

    # Weights that are replicated (not TP-sharded) across every TP rank.
    # q_proj/o_proj (TP-sharded) and k_proj/v_proj (conditional on
    # num_kv_heads vs tp_size) go through the default TP-split path.
    _NON_TP_WEIGHT_PATTERNS = [
        "self_attention.q_norm.",
        "self_attention.k_norm.",
        "self_attention.rotary_emb.",
        "pre_mlp_layernorm.",
        "moe_block.moe_layer.router.",
        "moe_block.post_feedforward_layernorm",
        "moe_block.pre_feedforward_layernorm",
        ".layer_scalar",
    ]

    def _get_hf_text_config(self):
        return getattr(self.hf_config, "text_config", self.hf_config)

    def _resolve_hf_lm_prefix(self, available_hf_weights: set) -> str:
        """Detect the HF checkpoint prefix used for text-model weights.

        Gemma4ForConditionalGeneration saves text weights under
        'model.language_model.*'. Gemma4ForCausalLM (pure-text) saves them
        under 'model.*'. We probe for a known key (embed_tokens.weight) to
        pick the right prefix for this checkpoint.
        """
        # Cache on first call
        if getattr(self, "_hf_lm_prefix_cache", None) is not None:
            return self._hf_lm_prefix_cache

        candidates = ("model.language_model", "model")
        for cand in candidates:
            if f"{cand}.embed_tokens.weight" in available_hf_weights:
                self._hf_lm_prefix_cache = cand
                return cand

        # Fall back to the default; if this is wrong, all mappings will miss
        # and the model will end up randomly initialized — the caller should
        # surface a clear error (see load_weights below).
        self._hf_lm_prefix_cache = self._DEFAULT_HF_LM_PREFIX
        return self._hf_lm_prefix_cache

    def _hf_prefix(self) -> str:
        """Get the currently-resolved HF prefix, or the default if unset."""
        return getattr(self, "_hf_lm_prefix_cache", None) or self._DEFAULT_HF_LM_PREFIX

    @staticmethod
    def _fmt(tmpl: str, lm_prefix: str, **kwargs) -> str:
        return tmpl.format(lm=lm_prefix, **kwargs)

    def _build_config(self):
        text_config = getattr(self.hf_config, "text_config", None)
        config_key = "text_config" if text_config is not None else None
        hf_config = self._get_hf_text_config()

        extra_kwargs = {}
        if getattr(hf_config, "enable_moe_block", False):
            extra_kwargs.update(
                num_moe_experts=hf_config.num_experts,
                moe_ffn_hidden_size=hf_config.moe_intermediate_size,
                moe_router_topk=hf_config.top_k_experts,
                # Grouped GEMM via TEGroupedMLP: local experts run as a single
                # batched matmul instead of a Python per-expert loop.
                moe_grouped_gemm=True,
                moe_router_load_balancing_type="none",
                moe_router_score_function="softmax",
                moe_router_pre_softmax=False,
                moe_token_dispatcher_type="alltoall",
                moe_router_dtype="fp32",
            )

        config = self._build_base_config(
            text_config_key=config_key,
            add_qkv_bias=False,
            add_bias_linear=False,
            qk_layernorm=False,
            rotary_interleaved=False,
            persist_layer_norm=True,
            bias_activation_fusion=True,
            bias_dropout_fusion=True,
            normalization="RMSNorm",
            gated_linear_unit=True,
            layernorm_zero_centered_gamma=False,
            hetereogenous_dist_checkpoint=True,
            **extra_kwargs,
        )
        # Gemma4's dense MLP and MoE experts use `hidden_activation="gelu_pytorch_tanh"`
        # (i.e. `F.gelu(..., approximate="tanh")`). mbridge defaults to `F.silu`, which
        # produces numerically different intermediate values. For the 31B dense model
        # this manifests as near-saturated, spread-out logits (many tokens tied near
        # the softcap) and degenerate greedy output. Override here.
        hidden_act = getattr(hf_config, "hidden_activation", None) or getattr(hf_config, "hidden_act", None)
        if hidden_act == "gelu_pytorch_tanh":
            config.activation_func = partial(F.gelu, approximate="tanh")
            # Megatron's fused bias+activation path only supports plain `gelu`
            # and `swiglu` — disable fusion so our tanh-approximated gelu runs.
            config.bias_activation_fusion = False
        elif hidden_act == "gelu":
            config.activation_func = F.gelu
        elif hidden_act == "silu":
            config.activation_func = F.silu
        # Fall through for any other value.

        # Stashed for Gemma4TransformerLayer.__init__ to read; building
        # moe_block in __init__ (vs a post-creation callback) is what gets
        # expert params into the EDP sharding group, so optimizer master
        # fp32 round-trips correctly through dist_checkpointing on resume.
        config._gemma4_hf_config = hf_config
        config._gemma4_enable_moe = getattr(hf_config, "enable_moe_block", False)
        return config

    def _get_gptmodel_args(self) -> dict:
        hf_config = self._get_hf_text_config()
        rope_params = getattr(hf_config, "rope_parameters", {})
        sliding_rope = rope_params.get("sliding_attention", {})
        rope_theta = sliding_rope.get("rope_theta", getattr(hf_config, "rope_theta", 10000.0))
        return dict(
            vocab_size=hf_config.vocab_size,
            max_sequence_length=hf_config.max_position_embeddings,
            position_embedding_type="rope",
            rotary_base=rope_theta,
        )

    def _get_transformer_layer_spec(self, vp_stage: int | None = None):
        from megatron.core.transformer.spec_utils import ModuleSpec
        from megatron.core.transformer.transformer_block import TransformerBlockSubmodules

        from axon.models.mcore.models.gemma4 import Gemma4SelfAttention

        # Gemma4 runs dense MLP ∥ MoE block (summed). The MoE block is
        # built in Gemma4TransformerLayer.__init__; hide num_moe_experts
        # here so the base spec builder leaves the MLP slot dense.
        original_num_experts = getattr(self.config, "num_moe_experts", None)
        try:
            self.config.num_moe_experts = None
            block_spec = super()._get_transformer_layer_spec(vp_stage=vp_stage)
        finally:
            self.config.num_moe_experts = original_num_experts
        assert isinstance(block_spec, TransformerBlockSubmodules)
        hf_config = self._get_hf_text_config()

        # Unfuse pre_mlp_layernorm from TELayerNormColumnParallelLinear so
        # the norm weight multiply runs in fp32 (matching HF/vLLM).
        from megatron.core.extensions.transformer_engine import TEColumnParallelLinear

        from axon.models.mcore.models.gemma4 import Gemma4Fp32RMSNorm

        fp32_norm_spec = ModuleSpec(module=Gemma4Fp32RMSNorm)

        for layer_spec in block_spec.layer_specs:
            layer_spec.module = Gemma4TransformerLayer
            layer_spec.submodules.post_self_attn_layernorm = block_spec.layer_norm
            layer_spec.submodules.post_mlp_layernorm = block_spec.layer_norm
            layer_spec.submodules.input_layernorm = block_spec.layer_norm
            layer_spec.submodules.self_attention = ModuleSpec(
                module=Gemma4SelfAttention,
                params={"hf_config": hf_config},
            )
            # Unfuse pre_mlp_layernorm: plain TEColumnParallelLinear (no fused
            # norm) + a standalone fp32 RMSNorm.
            layer_spec.submodules.pre_mlp_layernorm = fp32_norm_spec
            layer_spec.submodules.mlp.submodules.linear_fc1 = TEColumnParallelLinear

        return block_spec

    def get_model(self, post_model_creation_callbacks=None, wrap_with_ddp=True, **kwargs):
        if post_model_creation_callbacks is None:
            post_model_creation_callbacks = []

        hf_config = self._get_hf_text_config()
        embed_scale = hf_config.hidden_size**0.5
        softcap = getattr(hf_config, "final_logit_softcapping", None)

        def _post_creation(model, **cb_kwargs):
            model = unwrap_model(model)

            # HF's Gemma4TextScaledWordEmbedding scales output by sqrt(hidden_size).
            if hasattr(model, "embedding"):
                original_forward = model.embedding.forward

                def scaled_forward(*args, **fwd_kwargs):
                    return original_forward(*args, **fwd_kwargs) * embed_scale

                model.embedding.forward = scaled_forward

            # Wrap standard norms to do fp32 affine (HF's Gemma4RMSNorm
            # does fp32 norm + fp32 weight multiply; TENorm does bf16 affine).
            # Wrap rather than replace so param names stay unchanged.
            # pre_mlp_layernorm is fused into TELayerNormColumnParallelLinear
            # and runs in bf16. The wrapper below handles all other norms.
            if hasattr(model, "decoder"):
                _eps = hf_config.rms_norm_eps
                from axon.models.mcore.models.gemma4 import _gemma4_fp32_rms_norm

                def _wrap_norm_fp32(norm_module):
                    def _fp32_forward(hidden_states):
                        w = getattr(norm_module, "weight", None)
                        return _gemma4_fp32_rms_norm(hidden_states, w, _eps)

                    norm_module.forward = _fp32_forward

                from megatron.core.transformer.identity_op import IdentityOp

                for layer in model.decoder.layers:
                    for attr in (
                        "input_layernorm",
                        "post_self_attn_layernorm",
                        "pre_mlp_layernorm",
                        "post_mlp_layernorm",
                    ):
                        norm = getattr(layer, attr, None)
                        if norm is not None and not isinstance(norm, IdentityOp):
                            _wrap_norm_fp32(norm)
                if hasattr(model.decoder, "final_layernorm") and model.decoder.final_layernorm is not None:
                    _wrap_norm_fp32(model.decoder.final_layernorm)

            # Apply final_logit_softcapping via output_layer.forward wrap
            # (tanh is elementwise, safe on vocab-parallel shards). Also
            # surface `_final_logit_softcapping` on the model so the fused
            # forward path can route to a softcap-aware cross-entropy.
            if softcap is not None and hasattr(model, "output_layer") and model.output_layer is not None:
                original_output_forward = model.output_layer.forward
                cap = float(softcap)

                def capped_forward(*args, **fwd_kwargs):
                    out = original_output_forward(*args, **fwd_kwargs)
                    if isinstance(out, tuple):
                        logits, bias = out
                        return torch.tanh(logits / cap) * cap, bias
                    return torch.tanh(out / cap) * cap

                model.output_layer.forward = capped_forward
                model._final_logit_softcapping = cap

        post_model_creation_callbacks.append(_post_creation)
        return super().get_model(
            post_model_creation_callbacks=post_model_creation_callbacks,
            wrap_with_ddp=wrap_with_ddp,
            **kwargs,
        )

    # --- P2P hooks ---
    def p2p_mcore_name_to_vllm(self, mcore_name, hf_name, meta):
        name = hf_name.replace("model.language_model.", "model.", 1)
        if ".moe_block.moe_layer.experts.linear_fc" in mcore_name:
            idx = meta["expert_idx"]
            for src, dst in [
                (".experts.gate_up_proj", ".w13_weight"),
                (".experts.down_proj", ".w2_weight"),
            ]:
                if name.endswith(src):
                    return f"{name[: -len(src)]}.moe.experts.{idx}{dst}"
            raise ValueError(f"unexpected Gemma4 expert HF name: {hf_name}")
        return name.replace(".router.per_expert_scale", ".moe.per_expert_scale").replace(
            ".mlp.gate_proj.", ".mlp.gate_up_proj."
        )

    def p2p_extra_params(self, metadata_list, actor_parameters):
        """Alias v_proj → k_proj for k_eq_v layers (V == K by construction).

        Both dict entries point at the same tensor object, so memory
        isn't duplicated. The tensor is still sent over the wire twice,
        since vLLM's fused qkv_proj has physically distinct K/V slots.
        """
        k = {m["sampler_name"]: m for m in metadata_list if ".self_attn.k_proj." in m["sampler_name"]}
        v = {
            m["sampler_name"].replace(".v_proj.", ".k_proj.")
            for m in metadata_list
            if ".self_attn.v_proj." in m["sampler_name"]
        }

        extras = []
        for k_name, k_meta in k.items():
            if k_name in v:
                continue
            v_name = k_name.replace(".k_proj.", ".v_proj.")
            actor_parameters[v_name] = k_meta["tensor"]
            extras.append({**k_meta, "sampler_name": v_name})
        return metadata_list + extras

    # --- Weight name mapping ---

    def _is_non_tp_weight(self, name: str) -> bool:
        for pattern in self._NON_TP_WEIGHT_PATTERNS:
            if pattern in name:
                return True
        return False

    def _is_expert_weight(self, name: str) -> bool:
        # TEGroupedMLP: moe_block.moe_layer.experts.linear_fc{1,2}.weight{i} / .bias{i}
        if ".moe_block.moe_layer.experts.linear_fc" not in name:
            return False
        for kw in (".weight", ".bias"):
            if kw in name:
                tail = name.rsplit(kw, 1)[-1]
                if tail.isdigit():
                    return True
        return False

    def _weight_name_mapping_mcore_to_hf(self, mcore_weights_name: str) -> list[str]:
        assert "_extra_state" not in mcore_weights_name

        lm = self._hf_prefix()

        if mcore_weights_name in self._DIRECT_MAPPING:
            return [self._DIRECT_MAPPING[mcore_weights_name].format(lm=lm)]

        if "post_self_attn_layernorm" in mcore_weights_name:
            layer_number = mcore_weights_name.split(".")[2]
            return [f"{lm}.layers.{layer_number}.post_attention_layernorm.weight"]

        if "post_mlp_layernorm" in mcore_weights_name and "moe_block" not in mcore_weights_name:
            layer_number = mcore_weights_name.split(".")[2]
            return [f"{lm}.layers.{layer_number}.post_feedforward_layernorm.weight"]

        # Per-layer scalar buffer: "decoder.layers.{i}.layer_scalar"
        if mcore_weights_name.endswith(".layer_scalar"):
            layer_number = mcore_weights_name.split(".")[2]
            return [f"{lm}.layers.{layer_number}.layer_scalar"]

        if "input_layernorm" in mcore_weights_name and "self_attention" not in mcore_weights_name:
            layer_number = mcore_weights_name.split(".")[2]
            return [f"{lm}.layers.{layer_number}.input_layernorm.weight"]

        if "moe_block" in mcore_weights_name:
            return self._weight_name_mapping_moe(mcore_weights_name)

        if "self_attention" in mcore_weights_name:
            return self._weight_name_mapping_attention(mcore_weights_name)

        if "mlp" in mcore_weights_name:
            return self._weight_name_mapping_mlp(mcore_weights_name)

        raise NotImplementedError(f"Unsupported parameter name: {mcore_weights_name}")

    def _weight_name_mapping_attention(self, name: str) -> list[str]:
        layer_number = name.split(".")[2]
        lm = self._hf_prefix()
        for keyword, mapping_names in self._ATTENTION_MAPPING.items():
            if keyword in name:
                return [x.format(lm=lm, layer_number=layer_number) for x in mapping_names]
        raise NotImplementedError(f"Unsupported attention parameter: {name}")

    def _weight_name_mapping_mlp(self, name: str) -> list[str]:
        layer_number = name.split(".")[2]
        lm = self._hf_prefix()
        for keyword, mapping_names in self._MLP_MAPPING.items():
            if keyword in name:
                return [x.format(lm=lm, layer_number=layer_number) for x in mapping_names]
        raise NotImplementedError(f"Unsupported MLP parameter: {name}")

    def _weight_name_mapping_moe(self, name: str) -> list[str]:
        layer_number = name.split(".")[2]
        lm = self._hf_prefix()

        # Router weights
        for keyword, mapping_names in self._MOE_ROUTER_MAPPING.items():
            if keyword in name:
                return [x.format(lm=lm, layer_number=layer_number) for x in mapping_names]

        # Expert weights (TEGroupedMLP: experts.linear_fc{1,2}.weight{i})
        if self._is_expert_weight(name):
            for keyword, mapping_names in self._MOE_EXPERT_MAPPING.items():
                if keyword in name:
                    return [x.format(lm=lm, layer_number=layer_number) for x in mapping_names]

        # Extra layernorms
        for keyword, mapping_names in self._MOE_NORM_MAPPING.items():
            if keyword in name:
                return [x.format(lm=lm, layer_number=layer_number) for x in mapping_names]

        raise NotImplementedError(f"Unsupported MoE parameter: {name}")

    def _get_local_expert_id(self, name: str) -> int:
        """Extract local expert ID from TEGroupedMLP weight name.

        e.g., 'decoder.layers.0.moe_block.moe_layer.experts.linear_fc1.weight3' → 3
        """
        for kw in (".weight", ".bias"):
            if kw in name:
                tail = name.rsplit(kw, 1)[-1]
                if tail.isdigit():
                    return int(tail)
        raise ValueError(f"Cannot parse expert id from TEGroupedMLP name: {name}")

    # --- Weight loading ---

    def load_weights(self, models: list[torch.nn.Module], weights_path: str, memory_efficient: bool = False) -> None:
        self.safetensor_io = self._get_safetensor_io(weights_path)
        available_hf_weights = set(self.safetensor_io.load_hf_weight_names())

        # Resolve the HF text-model prefix ('model' vs 'model.language_model')
        # from the actual safetensor keys so mappings don't silently miss.
        resolved_prefix = self._resolve_hf_lm_prefix(available_hf_weights)
        if f"{resolved_prefix}.embed_tokens.weight" not in available_hf_weights:
            raise RuntimeError(
                f"Gemma4Bridge: could not find '{resolved_prefix}.embed_tokens.weight' "
                f"in safetensors at {weights_path}. Checked prefixes: "
                f"'model.language_model.*', 'model.*'. Available example keys: "
                f"{sorted(list(available_hf_weights))[:5]}"
            )

        hf_config = self._get_hf_text_config()
        num_experts = getattr(hf_config, "num_experts", 0)

        for model in models:
            local_to_global_map = self._weight_name_mapping_mcore_local_to_global(model)
            local_to_hf_map = {}
            _unmapped_keys = []
            _missing_hf_keys = []
            for k, v in local_to_global_map.items():
                if "_extra_state" in k:
                    continue
                try:
                    hf_names = self._weight_name_mapping_mcore_to_hf(v)
                except NotImplementedError:
                    _unmapped_keys.append(v)
                    continue
                if all(hf_name in available_hf_weights for hf_name in hf_names):
                    local_to_hf_map[k] = hf_names
                else:
                    _missing_hf_keys.append((v, hf_names))

            # Determine which weights to load from disk
            to_load_from_disk = set()
            for local_name, hf_names in local_to_hf_map.items():
                if self._is_non_tp_weight(local_name):
                    to_load_from_disk.update(hf_names)
                elif self._is_expert_weight(local_name):
                    # Expert weights: etp_rank==0 loads from disk
                    if self.mpu.etp_rank == 0:
                        to_load_from_disk.update(hf_names)
                else:
                    if self.mpu.tp_rank == 0:
                        to_load_from_disk.update(hf_names)
                    elif "lm_head.weight" in hf_names:
                        to_load_from_disk.update(hf_names)

            to_load_from_disk = list(to_load_from_disk)
            if not memory_efficient:
                hf_weights_map = self.safetensor_io.load_some_hf_weight(to_load_from_disk)

            # Cache the state_dict once — calling model.state_dict() inside the
            # loop was O(num_params × model_depth) because it recursively
            # traverses the entire module tree on every call.
            cached_state_dict = model.state_dict()

            for local_name, hf_names in local_to_hf_map.items():
                param = cached_state_dict[local_name]

                # Non-TP weights: load directly (no TP sharding)
                if self._is_non_tp_weight(local_name):
                    if set(to_load_from_disk) & set(hf_names):
                        if len(hf_names) > 1:
                            # Multi-source (e.g., gate_proj + up_proj → fc1)
                            parts = []
                            for hn in hf_names:
                                w = (
                                    hf_weights_map[hn]
                                    if not memory_efficient
                                    else self.safetensor_io.load_one_hf_weight(hn)
                                )
                                parts.append(w.to(param.device, dtype=param.dtype))
                            hf_weight = torch.cat(parts, dim=0)
                        else:
                            hf_weight = (
                                hf_weights_map[hf_names[0]]
                                if not memory_efficient
                                else self.safetensor_io.load_one_hf_weight(hf_names[0])
                            )
                        param.copy_(hf_weight.to(param.device, dtype=param.dtype).contiguous())
                    continue

                # Expert weights: EP/ETP sharding from HF 3D tensor
                if self._is_expert_weight(local_name):
                    global_name = local_to_global_map[local_name]
                    self._load_expert_weight(
                        local_name,
                        global_name,
                        hf_names,
                        param,
                        hf_weights_map if not memory_efficient else None,
                        memory_efficient,
                        num_experts,
                    )
                    continue

                # Regular TP weights
                if set(to_load_from_disk) & set(hf_names):
                    hf_weights = (
                        [hf_weights_map[x] for x in hf_names]
                        if not memory_efficient
                        else [self.safetensor_io.load_one_hf_weight(x) for x in hf_names]
                    )
                    mcore_weight = self._weight_to_mcore_format(local_name, hf_weights)
                else:
                    mcore_weight = None

                if hf_names[0] in {"lm_head.weight", "model.embed_tokens.weight"}:
                    if param.shape[0] == 1 and (mcore_weight is None or mcore_weight.shape[0] != 1):
                        continue

                param_to_load = torch.empty_like(param)
                if self.mpu.tp_rank == 0:
                    splits = list(self._weight_split_across_tp(local_name, mcore_weight, param, self.mpu.tp_size))
                    splits = [t.to(param.device, dtype=param.dtype).contiguous() for t in splits]
                else:
                    splits = None
                torch.distributed.scatter(
                    param_to_load,
                    splits,
                    src=torch.distributed.get_global_rank(self.mpu.tp_group, 0),
                    group=self.mpu.tp_group,
                )
                param.copy_(param_to_load)

            torch.cuda.empty_cache()

    def _load_expert_weight(
        self,
        local_name,
        global_name,
        hf_names,
        param,
        hf_weights_map,
        memory_efficient,
        num_experts,
    ):
        """Load a single expert's weight with EP/ETP sharding.

        HF format: 3D tensor [num_experts, out_dim, in_dim].
        TEGroupedMLP: per-expert linear_fc{1,2}.weight{i} / .bias{i}.

        The expert's global_id is computed from EP rank + local expert index.
        """
        local_expert_id = self._get_local_expert_id(global_name)
        ep_size = self.mpu.ep_size
        num_local_experts = num_experts // ep_size
        global_expert_id = self.mpu.ep_rank * num_local_experts + local_expert_id

        is_fc1 = "linear_fc1" in local_name

        if self.mpu.etp_rank == 0:
            if hf_weights_map is not None:
                hf_3d = hf_weights_map[hf_names[0]]
            else:
                hf_3d = self.safetensor_io.load_one_hf_weight(hf_names[0])

            # Extract this expert from the 3D tensor
            expert_weight = hf_3d[global_expert_id]  # [out_dim, in_dim]

            # fc1 gate_up_proj and fc2 down_proj share HF↔mcore layout.
            mcore_weight = expert_weight if is_fc1 else expert_weight.contiguous()
        else:
            mcore_weight = None

        # ETP: scatter across etp ranks
        param_to_load = torch.empty_like(param)
        if self.mpu.etp_rank == 0 and mcore_weight is not None:
            splits = list(self._weight_split_across_tp(local_name, mcore_weight, param, self.mpu.etp_size))
            splits = [t.to(param.device, dtype=param.dtype).contiguous() for t in splits]
        else:
            splits = None

        torch.distributed.scatter(
            param_to_load,
            splits,
            src=torch.distributed.get_global_rank(self.mpu.etp_group, 0),
            group=self.mpu.etp_group,
        )
        param.copy_(param_to_load)

    # --- Weight export ---

    @staticmethod
    def _iter_params_and_persistent_buffers(model):
        """Yield ``(name, tensor)`` for every param AND persistent buffer.

        ``named_parameters()`` alone misses ``register_buffer(persistent=True)``
        tensors like Gemma4's per-layer ``layer_scalar`` (0.04–0.99). Without
        these, vLLM keeps its dummy ``ones(1)`` init and activations blow up
        across layers. Iteration order (params → buffers in module order) must
        match between both call sites in ``export_weights``.
        """
        yield from model.named_parameters()
        for mod_name, module in model.named_modules():
            non_persistent = getattr(module, "_non_persistent_buffers_set", set())
            for buf_name, buf in module._buffers.items():
                if buf is None or buf_name in non_persistent:
                    continue
                yield (f"{mod_name}.{buf_name}" if mod_name else buf_name), buf

    def export_weights(self, models: list[torch.nn.Module]) -> Generator[tuple[str, torch.Tensor], None, None]:
        models = [unwrap_model(model) for model in models]
        hf_config = self._get_hf_text_config()
        num_experts = getattr(hf_config, "num_experts", 0)

        def get_model_chunk_generator():
            for model in models:
                yield from self._iter_params_and_persistent_buffers(model)

        weights_names = []
        for vpp_rank, model in enumerate(models):
            for name, _param in self._iter_params_and_persistent_buffers(model):
                weights_names.append((self.mpu.pp_rank, vpp_rank, name))

        weights_names_all_pp = [None] * self.mpu.pp_size
        torch.distributed.all_gather_object(
            object_list=weights_names_all_pp, obj=weights_names, group=self.mpu.pp_group
        )
        weights_names_all_pp = sum(weights_names_all_pp, [])
        model_chunk_generator = get_model_chunk_generator()
        local_to_global_maps = [
            self._weight_name_mapping_mcore_local_to_global(model, consider_ep=False) for model in models
        ]

        # Buffer for accumulating expert weights into 3D tensors
        expert_buffer = {}  # key: hf_name → {global_expert_id: tensor}
        emitted_hf_names: set[str] = set()

        for iter_pp_rank, iter_vpp_rank, iter_name in weights_names_all_pp:
            local_to_global_map = local_to_global_maps[iter_vpp_rank]
            if iter_pp_rank == self.mpu.pp_rank:
                try:
                    name, param = next(model_chunk_generator)
                except StopIteration:
                    name, param = None, None
                name = local_to_global_map[iter_name]
            else:
                name, param = None, None

            name = broadcast_str_from_megatron_pp(name)
            broad_pp_param = broadcast_from_megatron_pp(param)

            # Non-TP weights
            if self._is_non_tp_weight(name):
                try:
                    converted_names, converted_params = self._weight_to_hf_format(name, broad_pp_param)
                except NotImplementedError:
                    continue
                emitted_hf_names.update(converted_names)
                yield from zip(converted_names, [p.detach() for p in converted_params], strict=False)
                continue

            # Expert weights: gather across ETP/EP, accumulate into 3D tensor
            if self._is_expert_weight(name):
                result = self._export_expert_weight(name, broad_pp_param, num_experts, expert_buffer)
                if result is not None:
                    emitted_hf_names.add(result[0])
                    yield result
                continue

            # Regular TP weights
            if hasattr(broad_pp_param, "tensor_model_parallel") and broad_pp_param.tensor_model_parallel:
                if self.mpu.tp_size <= 1:
                    infer_params = [broad_pp_param]
                else:
                    infer_params = [torch.empty_like(broad_pp_param) for _ in range(self.mpu.tp_size)]
                    torch.distributed.all_gather(infer_params, broad_pp_param, group=self.mpu.tp_group)
                infer_params = self._weight_merge_across_tp(name, infer_params, broad_pp_param)
            else:
                infer_params = broad_pp_param

            converted_names, converted_params = self._weight_to_hf_format(name, infer_params)
            if len(converted_names) == 0:
                continue
            emitted_hf_names.update(converted_names)
            yield from zip(converted_names, [p.detach() for p in converted_params], strict=False)

        # Pass through HF keys the trainer doesn't own (vision_tower etc.) —
        # without this, save_hf_weight_merge crashes reading missing shards.
        yield from self._export_passthrough_hf_weights(emitted_hf_names)

    def _export_passthrough_hf_weights(
        self, emitted_hf_names: set[str]
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        """Stream HF tensors not produced by the trainer, grouped by source shard."""
        if not getattr(self, "safetensor_io", None):
            return
        try:
            all_hf = set(self.safetensor_io.load_hf_weight_names())
        except Exception:
            return
        missing = set(all_hf) - emitted_hf_names
        if not missing:
            return

        import os as _os
        from collections import defaultdict
        from glob import glob as _glob

        from safetensors import safe_open

        hf_dir = self.safetensor_io.hf_dir
        index = getattr(self.safetensor_io, "index", None)

        file_to_names: dict[str, list[str]] = defaultdict(list)
        if index:
            for name in missing:
                fn = index.get(name)
                if fn is not None:
                    file_to_names[fn].append(name)
        else:
            for shard_path in sorted(_glob(_os.path.join(hf_dir, "*.safetensors"))):
                with safe_open(shard_path, framework="pt", device="cpu") as f:
                    hits = missing.intersection(f.keys())
                if hits:
                    file_to_names[_os.path.basename(shard_path)] = sorted(hits)

        for filename in sorted(file_to_names):
            shard_path = _os.path.join(hf_dir, filename)
            with safe_open(shard_path, framework="pt", device="cpu") as f:
                for name in sorted(file_to_names[filename]):
                    yield name, f.get_tensor(name).detach()

    def _export_expert_weight(self, name, param, num_experts, expert_buffer):
        """Export a single expert's weight back to HF 3D format.

        Accumulates individual expert weights into a 3D tensor. Returns the
        complete tensor only when all experts have been collected.
        """
        local_expert_id = self._get_local_expert_id(name)
        ep_size = self.mpu.ep_size
        etp_size = self.mpu.etp_size
        num_local_experts = num_experts // ep_size
        global_expert_id = self.mpu.ep_rank * num_local_experts + local_expert_id

        # Gather across ETP. linear_fc1 uses GLU layout: each rank stores
        # [gate_i, up_i], so naive cat would interleave. Reassemble per-half
        # to recover HF's [gate_full, up_full] layout.
        if etp_size > 1:
            etp_params = [torch.empty_like(param) for _ in range(etp_size)]
            torch.distributed.all_gather(etp_params, param, group=self.mpu.etp_group)
            if "linear_fc1" in name:
                gates, ups = [], []
                for w in etp_params:
                    gate, up = w.chunk(2, dim=0)
                    gates.append(gate)
                    ups.append(up)
                merged = torch.cat([torch.cat(gates, dim=0), torch.cat(ups, dim=0)], dim=0).contiguous()
            else:
                # linear_fc2 is RowParallelLinear sharding dim 1 (moe_ffn).
                merged = torch.cat(etp_params, dim=1).contiguous()
        else:
            merged = param

        # Gather across EP
        if ep_size > 1:
            ep_params = [torch.empty_like(merged) for _ in range(ep_size)]
            torch.distributed.all_gather(ep_params, merged, group=self.mpu.ep_group)
            # Each EP rank contributes this local_expert_id's data
            all_experts = {
                ep_rank * num_local_experts + local_expert_id: ep_params[ep_rank] for ep_rank in range(ep_size)
            }
        else:
            all_experts = {global_expert_id: merged}

        # Determine HF name
        hf_names = self._weight_name_mapping_mcore_to_hf(name)
        hf_name = hf_names[0]
        is_fc1 = "linear_fc1" in name

        if hf_name not in expert_buffer:
            expert_buffer[hf_name] = {}

        expert_buffer[hf_name].update(all_experts)

        # Check if all experts collected
        if len(expert_buffer[hf_name]) == num_experts:
            sorted_experts = [expert_buffer[hf_name][i] for i in range(num_experts)]

            if is_fc1:
                # linear_fc1: [2*moe_ffn, hidden] per expert → stack → [E, 2*moe_ffn, hidden]
                hf_3d = torch.stack(sorted_experts)
            else:
                # linear_fc2: [hidden, moe_ffn] per expert → stack → [E, hidden, moe_ffn]
                hf_3d = torch.stack([w.contiguous() for w in sorted_experts])

            del expert_buffer[hf_name]
            return (hf_name, hf_3d.detach())

        return None

    def export_weights_without_gather(self, models):
        """Label Gemma4 expert tuples with EP info; upstream's gate matches
        only ``.mlp.experts.linear_fc`` and misses Gemma4's nested path.
        """
        for tup in super().export_weights_without_gather(models):
            name, tp_rank, tp_size, ep_rank, ep_size, tmp, pd, param = tup
            if ".moe_block.moe_layer.experts.linear_fc" in name and self.mpu.ep_size >= 1 and ep_size == 0:
                num_experts_per_rank = self.config.num_moe_experts // self.mpu.ep_size
                name_prefix, local_expert_id = name.rsplit(".weight", 1)
                global_expert_id = num_experts_per_rank * self.mpu.ep_rank + int(local_expert_id)
                yield (
                    f"{name_prefix}.weight{global_expert_id}",
                    self.mpu.etp_rank,
                    self.mpu.etp_size,
                    self.mpu.ep_rank,
                    self.mpu.ep_size,
                    tmp,
                    pd,
                    param,
                )
            else:
                yield tup

    def _weight_to_hf_format(self, mcore_weights_name, mcore_weights):
        """Stack per-expert tensors into HF's 3D ``[num_experts, ...]`` layout.

        ``_save_weights_fast`` calls this once per expert; HF stores them as
        one stacked tensor under the same name, so we buffer until all
        ``num_experts`` are collected then return the stacked 3D tensor.
        """
        if ".moe_block.moe_layer.experts.linear_fc" in mcore_weights_name:
            _, expert_id_str = mcore_weights_name.rsplit(".weight", 1)
            if expert_id_str.isdigit():
                expert_id = int(expert_id_str)
                hf_names = self._weight_name_mapping_mcore_to_hf(mcore_weights_name)
                if len(hf_names) == 1:
                    hf_name = hf_names[0]
                    if not hasattr(self, "_gemma4_expert_stack_buffer"):
                        self._gemma4_expert_stack_buffer = {}
                    buf = self._gemma4_expert_stack_buffer.setdefault(hf_name, {})
                    buf[expert_id] = mcore_weights.cpu()

                    num_experts = self.config.num_moe_experts
                    if len(buf) < num_experts:
                        return [], []

                    sorted_experts = [buf[i] for i in range(num_experts)]
                    is_fc1 = "linear_fc1" in mcore_weights_name
                    if is_fc1:
                        hf_3d = torch.stack(sorted_experts)
                    else:
                        hf_3d = torch.stack([w.contiguous() for w in sorted_experts])
                    del self._gemma4_expert_stack_buffer[hf_name]
                    return [hf_name], [hf_3d]

        return super()._weight_to_hf_format(mcore_weights_name, mcore_weights)

    def _weight_merge_across_tp(self, mcore_weights_name, mcore_weights, param):
        """ETP-aware merge for Gemma4 expert weights; upstream's gate matches
        only ``mlp.experts.linear_fc`` and asserts on TP size for ours.
        """
        if ".moe_block.moe_layer.experts.linear_fc" not in mcore_weights_name:
            return super()._weight_merge_across_tp(mcore_weights_name, mcore_weights, param)

        assert len(mcore_weights) == self.mpu.etp_size, (
            f"Gemma4 expert merge: expected etp_size={self.mpu.etp_size} weights, "
            f"got {len(mcore_weights)} for {mcore_weights_name}"
        )
        if self.mpu.etp_size == 1:
            return mcore_weights[0]

        if "linear_fc1.weight" in mcore_weights_name:
            # Column-parallel + GLU: split each rank's slice into gate/up
            # halves, concat halves separately, then rejoin.
            mcore_config = self._get_mcore_config_by_name(mcore_weights_name)
            if not mcore_config.gated_linear_unit:
                return torch.cat(mcore_weights, dim=0)
            gates, ups = [], []
            for w in mcore_weights:
                gate, up = w.chunk(2, dim=0)
                gates.append(gate)
                ups.append(up)
            return torch.cat([torch.cat(gates, dim=0), torch.cat(ups, dim=0)], dim=0)

        if "linear_fc2.weight" in mcore_weights_name:
            return torch.cat(mcore_weights, dim=1)

        assert hasattr(param, "tensor_model_parallel") and param.tensor_model_parallel
        return torch.cat(mcore_weights, dim=param.partition_dim)

    def save_weights(
        self,
        models: list,
        weights_path: str,
        memory_efficient: bool = False,
        distributed_filesystem: bool = False,
    ) -> None:
        """Wrap fast-path save to write tmp files for HF keys the trainer
        doesn't own (vision_tower etc.) before the final shard merge.
        """
        if distributed_filesystem and memory_efficient:
            io = self.safetensor_io
            original_merge = io.save_hf_weight_merge
            bridge = self

            def merge_with_passthrough(new_hf_dir, rank=0, world_size=1):
                bridge._write_passthrough_tmp_files(new_hf_dir, rank, world_size)
                # Synchronize so no rank starts merging before all passthrough
                # tmp files are visible on the distributed filesystem.
                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    torch.distributed.barrier()
                return original_merge(new_hf_dir, rank, world_size)

            io.save_hf_weight_merge = merge_with_passthrough
            try:
                return super().save_weights(models, weights_path, memory_efficient, distributed_filesystem)
            finally:
                io.save_hf_weight_merge = original_merge
        return super().save_weights(models, weights_path, memory_efficient, distributed_filesystem)

    def _write_passthrough_tmp_files(self, new_hf_dir: str, rank: int, world_size: int) -> None:
        """Stream HF keys missing from the tmp dir from the source safetensors,
        sharded across ranks, one open per source shard.
        """
        import os
        from collections import defaultdict

        from safetensors import safe_open

        if not getattr(self, "safetensor_io", None):
            return
        index = getattr(self.safetensor_io, "index", None)
        if not index:
            return

        all_hf = set(index.keys())
        try:
            existing_tmp = {f[: -len(".safetensors")] for f in os.listdir(new_hf_dir) if f.endswith(".safetensors")}
        except FileNotFoundError:
            existing_tmp = set()

        missing = sorted(all_hf - existing_tmp)
        if not missing:
            return

        my_share = missing[rank :: max(world_size, 1)]
        if not my_share:
            return

        file_to_names: dict[str, list[str]] = defaultdict(list)
        for name in my_share:
            shard_filename = index.get(name)
            if shard_filename is not None:
                file_to_names[shard_filename].append(name)

        hf_dir = self.safetensor_io.hf_dir
        for shard_filename, names in file_to_names.items():
            shard_path = os.path.join(hf_dir, shard_filename)
            with safe_open(shard_path, framework="pt", device="cpu") as f:
                for name in names:
                    tensor = f.get_tensor(name)
                    self.safetensor_io.save_tmp_weight(name, tensor, new_hf_dir)
                    del tensor
