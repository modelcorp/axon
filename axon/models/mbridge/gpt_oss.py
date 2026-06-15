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
# load_weights / export_weights adapted from mbridge Bridge/LLMBridge, BSD-3-Clause (github.com/ISEEKYAN/mbridge).
from collections.abc import Generator

import torch
from mbridge.core import register_model
from mbridge.core.util import (
    broadcast_from_megatron_pp,
    broadcast_str_from_megatron_pp,
    get_model,
    unwrap_model,
)
from mbridge.models.qwen2moe import Qwen2MoEBridge


@register_model("gpt_oss")
class GPTOSSBridge(Qwen2MoEBridge):
    # Weight names that exist in the model architecture but not in the checkpoint
    # These will be skipped during loading
    _SKIP_LOADING_WEIGHTS = {
        "self_attn.q_norm.weight",
        "self_attn.k_norm.weight",
    }

    # Weight patterns that are NOT tensor-parallel (HuggingFace attention weights)
    # These should be broadcast instead of split+scatter
    _NON_TP_WEIGHT_PATTERNS = [
        "self_attention.self_attn.",  # HuggingFace attention modules
        "self_attention.linear_attn.",  # HuggingFace linear attention modules
        "self_attention.rotary_emb.",  # Rotary embeddings
    ]

    _ATTENTION_MAPPING = Qwen2MoEBridge._ATTENTION_MAPPING | {
        f"self_attention.{weight_name}": ["model.layers.{layer_number}." + weight_name]
        for weight_name in [
            "input_layernorm.weight",
            # gated attn
            "self_attn.k_proj.weight",
            "self_attn.k_proj.bias",
            "self_attn.o_proj.weight",
            "self_attn.o_proj.bias",
            "self_attn.q_proj.weight",
            "self_attn.q_proj.bias",
            "self_attn.v_proj.weight",
            "self_attn.v_proj.bias",
            "self_attn.sinks",
            # qk norm (may not exist in all checkpoints, will be skipped if not present)
            "self_attn.q_norm.weight",
            "self_attn.k_norm.weight",
        ]
    }

    _MLP_MAPPING = {
        "mlp.router.weight": ["model.layers.{layer_number}.mlp.router.weight"],
        "mlp.router.bias": ["model.layers.{layer_number}.mlp.router.bias"],
        "pre_mlp_layernorm": ["model.layers.{layer_number}.post_attention_layernorm.weight"],
        "mlp.experts.linear_fc1.weight": [
            "model.layers.{layer_number}.mlp.experts.gate_up_proj",
        ],
        "mlp.experts.linear_fc2.weight": ["model.layers.{layer_number}.mlp.experts.down_proj"],
        "mlp.experts.linear_fc1.bias": [
            "model.layers.{layer_number}.mlp.experts.gate_up_proj_bias",
        ],
        "mlp.experts.linear_fc2.bias": [
            "model.layers.{layer_number}.mlp.experts.down_proj_bias",
        ],
    }

    def _build_config(self):
        from megatron.core.fusions.fused_bias_geglu import quick_gelu

        # Get expert intermediate size - GPT-OSS may use different config keys
        expert_intermediate_size = getattr(
            self.hf_config,
            "moe_intermediate_size",
            getattr(self.hf_config, "expert_intermediate_size", self.hf_config.intermediate_size),
        )

        config = self._build_base_config(
            use_cpu_initialization=False,
            # MoE specific
            moe_ffn_hidden_size=expert_intermediate_size,
            moe_router_bias_update_rate=0,  # Match HF: no router bias updates
            moe_router_topk=self.hf_config.num_experts_per_tok,
            num_moe_experts=self.hf_config.num_local_experts,
            moe_aux_loss_coeff=self.hf_config.router_aux_loss_coef,
            moe_router_load_balancing_type="none",  # default None for RL
            moe_grouped_gemm=True,
            moe_router_score_function="softmax",
            # Other optimizations
            persist_layer_norm=True,
            bias_activation_fusion=True,  # CRITICAL: Must be True for TEGroupedMLP quick_gelu!
            bias_dropout_fusion=False,
            # GPT-OSS specific
            moe_router_pre_softmax=False,
            qk_layernorm=False,
            add_bias_linear=True,
            activation_func=quick_gelu,
            activation_func_clamp_value=7.0,
            glu_linear_offset=1.0,
            gated_linear_unit=True,
            window_size=128,
            softmax_type="learnable",  # Learnable softmax for sinks
            window_attn_skip_freq=2,  # Every 2nd layer is full attention
        )
        config.position_embedding_type = "yarn"
        config.yarn_rotary_scaling_factor = 32.0
        config.yarn_original_max_position_embeddings = 4096
        config.yarn_beta_fast = 32.0
        config.yarn_beta_slow = 1.0
        config.yarn_correction_range_round_to_int = False
        config.yarn_mscale = 1.0
        config.yarn_mscale_all_dim = 1.0
        config.use_pre_softmax = False
        config.use_te_activation_func = False

        # Router behavior: match HF eager routing (no z-loss, no jitter, no token drop/capacity)
        config.moe_z_loss_coeff = None
        config.moe_input_jitter_eps = None
        config.moe_expert_capacity_factor = None
        config.moe_token_drop_policy = None
        config.moe_pad_expert_input_to_capacity = False
        config.moe_router_fusion = False
        config.moe_router_dtype = "fp32"

        # Configure layer types for sliding window attention pattern
        if hasattr(self.hf_config, "layer_types") and self.hf_config.layer_types is not None:
            config.layer_types = self.hf_config.layer_types
        else:
            config.layer_types = [
                "full_attention" if i % config.window_attn_skip_freq == 0 else "sliding_attention"
                for i in range(config.num_layers)
            ]

        return config

    def _weight_name_mapping_mlp(self, name: str) -> list[str]:
        layer_number = name.split(".")[2]
        convert_names = []

        # Build dynamic mapping for MoE experts
        mlp_mapping = self._MLP_MAPPING.copy()
        for i in range(self.config.num_moe_experts):
            new_mapping = {
                f"mlp.experts.local_experts.{i}.linear_fc1.weight": [
                    "model.layers.{layer_number}.mlp.experts.gate_up_proj",
                ],
                f"mlp.experts.local_experts.{i}.linear_fc2.weight": [
                    "model.layers.{layer_number}.mlp.experts.down_proj"
                ],
                f"mlp.experts.local_experts.{i}.linear_fc1.bias": [
                    "model.layers.{layer_number}.mlp.experts.gate_up_proj_bias",
                ],
                f"mlp.experts.local_experts.{i}.linear_fc2.bias": [
                    "model.layers.{layer_number}.mlp.experts.down_proj_bias",
                ],
            }
            mlp_mapping.update(new_mapping)

        # Check for matches in the mapping
        for keyword, mapping_names in mlp_mapping.items():
            if keyword in name:
                if "{expert_id}" in mapping_names[0]:
                    if "bias" in name:
                        expert_id = name.split("bias")[-1]
                    else:
                        expert_id = name.split("weight")[-1]
                    convert_names.extend(
                        [x.format(layer_number=layer_number, expert_id=expert_id) for x in mapping_names]
                    )
                else:
                    convert_names.extend([x.format(layer_number=layer_number) for x in mapping_names])
                break
        if len(convert_names) == 0:
            raise NotImplementedError(f"Unsupported parameter name: {name}")
        return convert_names

    def _weight_name_mapping_mcore_local_to_global(
        self, model: torch.nn.Module, consider_ep: bool = True
    ) -> dict[str, str]:
        """
        Map local weight names to global weight names, supporting VPP and EP.
        """
        # vpp
        local_layer_to_global_layer = {}
        model = unwrap_model(model)
        if hasattr(model, "decoder"):
            for idx, layer in enumerate(model.decoder.layers):
                local_layer_to_global_layer[idx] = layer.layer_number - 1
        all_param_names = [k for k in model.state_dict().keys() if "_extra_state" not in k]
        ret = {}
        for param_name in all_param_names:
            keyword = "decoder.layers."
            if keyword in param_name:
                layer_idx = int(param_name.split(keyword)[1].split(".")[0])
                global_layer_idx = local_layer_to_global_layer[layer_idx]
                ret[param_name] = param_name.replace(f"layers.{layer_idx}.", f"layers.{global_layer_idx}.")
            else:
                ret[param_name] = param_name

        # ep
        if self.mpu.ep_size > 1 and consider_ep:
            num_experts = self.config.num_moe_experts
            num_experts_per_rank = num_experts // self.mpu.ep_size
            local_expert_to_global_expert = {
                i: i + num_experts_per_rank * self.mpu.ep_rank for i in range(num_experts_per_rank)
            }
            for k in ret.keys():
                v = ret[k]
                if ".mlp.experts.linear_fc" in v:
                    # Handle both weight and bias parameters
                    if ".weight" in v:
                        name_prefix, local_expert_id = v.split(".weight")
                        suffix = ".weight"
                    elif ".bias" in v:
                        name_prefix, local_expert_id = v.split(".bias")
                        suffix = ".bias"
                    else:
                        raise ValueError(f"Unexpected expert parameter format: {v}")

                    global_expert_idx = local_expert_to_global_expert[int(local_expert_id)]
                    ret[k] = f"{name_prefix}{suffix}{global_expert_idx}"

        return ret

    def get_model(
        self,
        post_model_creation_callbacks=None,
        wrap_with_ddp=True,
    ):
        """
        Override get_model to use custom GPTOSSAttention module for attention layers.
        """
        from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
        from megatron.core.transformer.spec_utils import ModuleSpec

        from axon.models.mcore.models.gpt_oss import GPTOSSAttention

        config = self.config

        def model_provider(pre_process, post_process, vp_stage=None):
            from megatron.core.models.gpt.gpt_model import GPTModel

            extra_kwargs = {"vp_stage": vp_stage} if vp_stage is not None else {}
            transformer_block_spec = get_gpt_decoder_block_spec(config, use_transformer_engine=True, **extra_kwargs)

            # Replace self_attention with our custom GPTOSSAttention module for all layers
            for i in range(len(transformer_block_spec.layer_specs)):
                transformer_block_spec.layer_specs[i].submodules.self_attention = ModuleSpec(
                    module=GPTOSSAttention,
                    params={"hf_config": self.hf_config},
                )

            model = GPTModel(
                config=config,
                transformer_layer_spec=transformer_block_spec,
                vocab_size=self.hf_config.vocab_size,
                max_sequence_length=self.hf_config.max_position_embeddings,
                pre_process=pre_process,
                post_process=post_process,
                share_embeddings_and_output_weights=False,
                position_embedding_type="rope",
                rotary_base=self.hf_config.rope_theta,
            )

            return model

        models = get_model(
            model_provider,
            wrap_with_ddp=wrap_with_ddp,
        )

        if post_model_creation_callbacks:
            for callback in post_model_creation_callbacks:
                for model in models:
                    callback(model)

        return models

    def _extract_expert_id_from_name(self, name: str) -> tuple[int, str]:
        """
        Extract expert ID from either format:
        - GroupedMLP: mlp.experts.linear_fc1.weight0 -> (0, "weight")
        - SequentialMLP: mlp.experts.local_experts.0.linear_fc1.weight -> (0, "weight")
        """
        if "local_experts" in name:
            parts = name.split("local_experts.")
            if len(parts) > 1:
                expert_str = parts[1].split(".")[0]
                expert_id = int(expert_str)
                keyword = "bias" if "bias" in name else "weight"
                return expert_id, keyword
        else:
            if "bias" in name:
                keyword = "bias"
                expert_str = name.split("bias")[-1]
            else:
                keyword = "weight"
                expert_str = name.split("weight")[-1]

            if expert_str.isdigit():
                return int(expert_str), keyword

        raise ValueError(f"Cannot extract expert_id from name: {name}")

    def load_weights(
        self,
        models: list[torch.nn.Module],
        weights_path: str,
        memory_efficient: bool = False,
    ) -> None:
        """
        Load weights from a Hugging Face model into a Megatron-Core model.
        """

        self.safetensor_io = self._get_safetensor_io(weights_path)

        for i, model in enumerate(models):
            # map local weight names to global weight names
            local_to_global_map = self._weight_name_mapping_mcore_local_to_global(model)
            # map local weight names to huggingface weight names
            # Skip weights that should not be loaded (e.g., q_norm/k_norm not in checkpoint)
            local_to_hf_map = {}
            for k, v in local_to_global_map.items():
                if "_extra_state" in k:
                    continue
                # Check if this weight should be skipped
                should_skip = False
                for skip_pattern in self._SKIP_LOADING_WEIGHTS:
                    if skip_pattern in v:
                        should_skip = True
                        break
                if should_skip:
                    continue
                local_to_hf_map[k] = self._weight_name_mapping_mcore_to_hf(v)

            # Helper to check if a weight is non-tensor-parallel (HuggingFace weights)
            def is_non_tp_weight(name):
                for pattern in self._NON_TP_WEIGHT_PATTERNS:
                    if pattern in name:
                        return True
                return False

            # only tp_rank0/etp_rank0 load from disk, others load from tp_rank0/etp_rank0
            # Exception: non-TP weights are loaded by all ranks
            to_load_from_disk = []
            for local_name, hf_names in local_to_hf_map.items():
                if ".mlp.experts." in local_name and ("linear_fc1" in local_name or "linear_fc2" in local_name):
                    if self.mpu.etp_rank == 0:
                        to_load_from_disk.extend(hf_names)
                elif is_non_tp_weight(local_name):
                    # Non-TP weights: all ranks load from disk (no scattering needed)
                    to_load_from_disk.extend(hf_names)
                else:
                    if self.mpu.tp_rank == 0:
                        to_load_from_disk.extend(hf_names)
                    else:
                        # special case for lm_head.weight
                        if "lm_head.weight" in hf_names:
                            to_load_from_disk.extend(hf_names)

            # load huggingface weights
            if not memory_efficient:
                hf_weights_map = self.safetensor_io.load_some_hf_weight(to_load_from_disk)

            # import mcore weights
            for local_name, hf_names in local_to_hf_map.items():
                param = model.state_dict()[local_name]

                # For non-TP HuggingFace weights, load raw HF weights directly (no mcore conversion)
                if is_non_tp_weight(local_name):
                    if set(to_load_from_disk) & set(hf_names):
                        if not memory_efficient:
                            hf_weight = hf_weights_map[hf_names[0]]
                        else:
                            hf_weight = self.safetensor_io.load_one_hf_weight(hf_names[0])
                        # Use raw HF weight directly for HuggingFace modules
                        param_to_load = hf_weight.to(param.device, dtype=param.dtype).contiguous()
                        param.copy_(param_to_load)
                    continue

                # hf format to mcore format (for Megatron-Core weights)
                if set(to_load_from_disk) & set(hf_names):
                    if not memory_efficient:
                        hf_weights = [hf_weights_map[x] for x in hf_names]
                    else:
                        hf_weights = [self.safetensor_io.load_one_hf_weight(x) for x in hf_names]
                    mcore_weight = self._weight_to_mcore_format(local_name, hf_weights)
                else:
                    mcore_weight = None
                if hf_names[0] in {"lm_head.weight", "model.embed_tokens.weight"}:
                    if param.shape[0] == 1 and (mcore_weight is None or mcore_weight.shape[0] != 1):
                        # skip lm_head.weight when the model is a value model
                        continue

                param_to_load = torch.empty_like(param)

                # Check if this is an expert weight
                if ".mlp.experts." in local_name and ("linear_fc1" in local_name or "linear_fc2" in local_name):
                    # Extract expert ID from name (handles both formats)
                    local_expert_id, keyword = self._extract_expert_id_from_name(local_name)

                    ep_rank = torch.distributed.get_rank(self.mpu.ep_group)
                    ep_size = torch.distributed.get_world_size(self.mpu.ep_group)
                    num_local_experts = self.config.num_moe_experts // ep_size
                    global_expert_id = ep_rank * num_local_experts + local_expert_id

                    # Convert from HF interleaved format to Megatron concatenated format
                    if keyword == "weight" and "linear_fc1" in local_name:
                        # HF: (num_experts, hidden_size, 2*expert_dim) with interleaved gate/up
                        # Megatron: (hidden_size, 2*expert_dim) then transposed to (2*expert_dim, hidden_size)
                        mcore_weight_gate = mcore_weight[global_expert_id][..., ::2].contiguous()
                        mcore_weight_up = mcore_weight[global_expert_id][..., 1::2].contiguous()
                        mcore_weight = torch.cat([mcore_weight_gate, mcore_weight_up], dim=1).t()
                    elif keyword == "bias" and "linear_fc1" in local_name:
                        # HF: (num_experts, 2*expert_dim) with interleaved gate/up
                        # Megatron: (2*expert_dim) concatenated
                        mcore_weight_gate = mcore_weight[global_expert_id][..., ::2].contiguous()
                        mcore_weight_up = mcore_weight[global_expert_id][..., 1::2].contiguous()
                        mcore_weight = torch.cat([mcore_weight_gate, mcore_weight_up], dim=0)
                    else:
                        # linear_fc2 weights/biases - no special handling needed
                        mcore_weight = mcore_weight[global_expert_id]
                        if "weight" in local_name:
                            mcore_weight = mcore_weight.t()

                    # split mcore weights across etp
                    if self.mpu.etp_rank == 0:
                        mcore_weights_tp_split = self._weight_split_across_tp(
                            local_name, mcore_weight, param, self.mpu.etp_size
                        )
                        mcore_weights_tp_split = list(mcore_weights_tp_split)
                        mcore_weights_tp_split = [
                            t.to(param.device, dtype=param.dtype).contiguous() for t in mcore_weights_tp_split
                        ]
                    else:
                        mcore_weights_tp_split = None

                    torch.distributed.scatter(
                        param_to_load,
                        mcore_weights_tp_split,
                        src=torch.distributed.get_global_rank(self.mpu.etp_group, 0),
                        group=self.mpu.etp_group,
                    )
                else:
                    # Non-expert weights - regular handling
                    # split mcore weights across tp
                    if self.mpu.tp_rank == 0:
                        mcore_weights_tp_split = self._weight_split_across_tp(
                            local_name, mcore_weight, param, self.mpu.tp_size
                        )
                        mcore_weights_tp_split = list(mcore_weights_tp_split)
                        mcore_weights_tp_split = [
                            t.to(param.device, dtype=param.dtype).contiguous() for t in mcore_weights_tp_split
                        ]
                    else:
                        mcore_weights_tp_split = None
                    torch.distributed.scatter(
                        param_to_load,
                        mcore_weights_tp_split,
                        src=torch.distributed.get_global_rank(self.mpu.tp_group, 0),
                        group=self.mpu.tp_group,
                    )
                # load
                param.copy_(param_to_load)

            torch.cuda.empty_cache()

    def export_weights(self, models: list[torch.nn.Module]) -> Generator[tuple[str, torch.Tensor], None, None]:
        models = [unwrap_model(model) for model in models]

        def get_model_chunk_generator():
            for model in models:
                existing_keys = set()
                for name, param in model.named_parameters():
                    existing_keys.add(name)
                    yield name, param

                extra_keys = [
                    x
                    for x in model.state_dict().keys()
                    if "_extra_state" not in x and "expert_bias" in x and x not in existing_keys
                ]
                for name in extra_keys:
                    yield name, model.state_dict()[name].to(torch.cuda.current_device())

        weights_names = []
        for vpp_rank, model in enumerate(models):
            existing_keys = set()
            for name, param in model.named_parameters():
                existing_keys.add(name)
                weights_names.append((self.mpu.pp_rank, vpp_rank, name))
            extra_keys = [
                x
                for x in model.state_dict().keys()
                if "_extra_state" not in x and "expert_bias" in x and x not in existing_keys
            ]
            for name in extra_keys:
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

        # Track expert weights for concatenation
        expert_weights_buffer = {}
        i = 0

        while i < len(weights_names_all_pp):
            iter_pp_rank, iter_vpp_rank, iter_name = weights_names_all_pp[i]
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

            # Check if this is an expert weight that needs concatenation
            is_expert_weight = False
            if self._is_expert_weight(name):
                if ".weight" in name or ".bias" in name:
                    is_expert_weight = True

            if is_expert_weight:
                try:
                    local_expert_id, keyword = self._extract_expert_id_from_name(name)
                except ValueError:
                    is_expert_weight = False

            if is_expert_weight:
                # Extract base name
                if "local_experts" in name:
                    parts = name.split("local_experts.")
                    before_experts = parts[0]
                    after_expert_id = ".".join(parts[1].split(".")[1:])
                    base_name = before_experts.rstrip(".") + "." + after_expert_id.replace(f".{keyword}", "")
                else:
                    base_name = name.rsplit(f".{keyword}", 1)[0]

                buffer_key = f"{base_name}.{keyword}"
                if buffer_key not in expert_weights_buffer:
                    expert_weights_buffer[buffer_key] = {}

                num_experts = self.config.num_moe_experts if hasattr(self.config, "num_moe_experts") else 32

                # Handle EP case: all_gather across EP ranks
                if self.mpu.ep_size > 1:
                    num_experts_per_rank = num_experts // self.mpu.ep_size

                    infer_params = [torch.empty_like(broad_pp_param) for _ in range(self.mpu.ep_size)]
                    torch.distributed.all_gather(infer_params, broad_pp_param, group=self.mpu.ep_group)

                    for ep_rank, param in enumerate(infer_params):
                        global_expert_id = num_experts_per_rank * ep_rank + local_expert_id

                        if self.mpu.etp_size > 1:
                            if ep_rank == self.mpu.ep_rank:
                                etp_params = [torch.empty_like(broad_pp_param) for _ in range(self.mpu.etp_size)]
                                torch.distributed.all_gather(etp_params, broad_pp_param, group=self.mpu.etp_group)
                                params = etp_params
                            else:
                                params = [param]
                        else:
                            params = [param]

                        expert_name = f"{base_name}.{keyword}{global_expert_id}"
                        merge_params = self._weight_merge_across_tp(expert_name, params, broad_pp_param)

                        expert_weights_buffer[buffer_key][global_expert_id] = merge_params
                else:
                    expert_weights_buffer[buffer_key][local_expert_id] = broad_pp_param

                # Check if we have all experts
                if len(expert_weights_buffer[buffer_key]) == num_experts:
                    sorted_experts = sorted(expert_weights_buffer[buffer_key].items())
                    concat_weights = torch.stack([w for _, w in sorted_experts])

                    # Convert from Megatron concatenated format to HF interleaved format
                    if "bias" not in buffer_key and "linear_fc1" in buffer_key:
                        gate, up_proj = concat_weights.chunk(2, dim=1)
                        gate = gate.permute(0, 2, 1)
                        up_proj = up_proj.permute(0, 2, 1)

                        converted_weights = torch.empty(
                            (concat_weights.shape[0], concat_weights.shape[2], concat_weights.shape[1]),
                            dtype=concat_weights.dtype,
                            device=concat_weights.device,
                        )
                        converted_weights[..., ::2] = gate
                        converted_weights[..., 1::2] = up_proj

                    elif "bias" in buffer_key and "linear_fc1" in buffer_key:
                        gate, up_proj = concat_weights.chunk(2, dim=1)
                        converted_weights = torch.empty_like(concat_weights)
                        converted_weights[..., ::2] = gate
                        converted_weights[..., 1::2] = up_proj

                    else:
                        converted_weights = concat_weights
                        if "weight" in buffer_key:
                            converted_weights = converted_weights.transpose(1, 2)

                    concat_weights = converted_weights
                    del expert_weights_buffer[buffer_key]

                    if self.mpu.ep_size <= 1:
                        if hasattr(concat_weights, "tensor_model_parallel") and concat_weights.tensor_model_parallel:
                            if self.mpu.tp_size <= 1:
                                infer_params = [concat_weights]
                            else:
                                infer_params = [torch.empty_like(concat_weights) for _ in range(self.mpu.tp_size)]
                                torch.distributed.all_gather(infer_params, concat_weights, group=self.mpu.tp_group)
                            infer_params = self._weight_merge_across_tp(buffer_key, infer_params, concat_weights)
                        else:
                            infer_params = concat_weights
                    else:
                        infer_params = concat_weights

                    converted_names, converted_params = self._weight_to_hf_format(buffer_key, infer_params)
                    yield from zip(converted_names, converted_params, strict=False)

                i += 1
                continue

            # Regular weight handling (non-expert)
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

            yield from zip(converted_names, converted_params, strict=False)
            i += 1

    def _is_expert_weight(self, name: str) -> bool:
        """Check if a weight name belongs to MoE experts."""
        return ".mlp.experts." in name and "router" not in name and ("linear_fc1" in name or "linear_fc2" in name)

    def _weight_merge_across_tp(
        self,
        mcore_weights_name: str,
        mcore_weights: list[torch.Tensor],
        param: torch.Tensor,
    ) -> torch.Tensor:
        """Override to handle expert biases correctly."""
        # Handle expert biases - they are not TP-sharded
        if self._is_expert_weight(mcore_weights_name) and ".bias" in mcore_weights_name:
            return mcore_weights[0]

        # Handle expert fc2 weights
        if (
            "mlp.experts.linear_fc2.weight" in mcore_weights_name
            or "mlp.experts." in mcore_weights_name
            and "linear_fc2" in mcore_weights_name
            and ".weight" in mcore_weights_name
        ):
            if len(mcore_weights) == 1:
                return mcore_weights[0]
            return torch.cat(mcore_weights, dim=1)

        # Fall back to parent implementation
        return super()._weight_merge_across_tp(mcore_weights_name, mcore_weights, param)
