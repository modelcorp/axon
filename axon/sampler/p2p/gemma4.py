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
"""Gemma4 sampler-side P2P hooks.

Lives under ``axon/sampler/p2p/`` (not alongside the actor bridge in
``axon/models/mbridge/gemma4.py``) because the sampler runs in a vLLM
Ray worker without Megatron available. Importing any mbridge file there
would fail and silently leave the hooks unregistered.

Handles two Gemma4-specific P2P quirks:
  * ``override_param`` — unfuses vLLM's packed ``self_attn.qkv_proj``
    into per-projection q/k/v views so they match the mcore actor
    which ships Q/K/V separately (Gemma4 has layer-varying head_dim and
    k_eq_v, so ``linear_qkv`` fusion isn't usable). Also filters out
    params the text-only mbridge actor never syncs (tied
    ``lm_head.weight`` and multimodal vision/audio tower weights on
    ``Gemma4ForConditionalGeneration``).
  * ``extra_buffers`` — surfaces the per-layer ``layer_scalar``
    persistent buffer, which ``named_parameters`` doesn't see but the
    actor ships through its state_dict extra-keys path.
"""

from axon.utils.p2p.routing_table import ParameterMetadata

from . import _register

# Multimodal / tied params that exist on the sampler but not on the
# text-only mbridge actor. Dropped from routing to keep the tables
# balanced.
_SKIP_PREFIXES = (
    "embed_vision.",
    "embed_audio.",
    "vision_tower.",
    "audio_tower.",
    "multi_modal_projector.",
)


def override_param(*, full_param_name, module, param, tp_size, split_idx, sampler_parameters):
    if full_param_name == "lm_head.weight" or any(full_param_name.startswith(p) for p in _SKIP_PREFIXES):
        return []

    if not full_param_name.endswith("self_attn.qkv_proj.weight"):
        return None

    # vLLM's QKVParallelLinear replicates K/V shards when
    # ``tp_size > total_num_kv_heads``. ``kv_idx = rank // max(1, tp/kvn)``
    # collapses ranks that share a KV head into the same split_idx so
    # the routing table sees them as DP replicas of one shard — and the
    # formula also yields ``kv_idx=0`` for the fully-replicated
    # ``kvn=1`` case, which matches the actor's plain nn.Linear layout.
    hn, kvn, hd = module.total_num_heads, module.total_num_kv_heads, module.head_size
    q_sz = (hn // tp_size) * hd
    kv_sz = max(1, kvn // tp_size) * hd
    kv_idx = split_idx // max(1, tp_size // kvn)

    entries = []
    for slot, start, sz, full_heads, sidx in (
        ("q", 0, q_sz, hn, split_idx),
        ("k", q_sz, kv_sz, kvn, kv_idx),
        ("v", q_sz + kv_sz, kv_sz, kvn, kv_idx),
    ):
        name = full_param_name.replace("qkv_proj", f"{slot}_proj")
        view = param[start : start + sz]
        assert view.is_contiguous(), f"{name} slice must be contiguous"
        sampler_parameters[name] = view
        entries.append(
            ParameterMetadata(
                param_name=name,
                original_param_name=name,
                param_shape=tuple(view.shape),
                full_param_shape=(full_heads * hd, view.shape[1]),
                param_dtype=view.dtype,
                split_dim=0,
                split_idx=sidx,
            )
        )
    return entries


def extra_buffers(vllm_model):
    out = []
    for mod_name, module in vllm_model.named_modules():
        buf = module._buffers.get("layer_scalar")
        if buf is None or "layer_scalar" in getattr(module, "_non_persistent_buffers_set", set()):
            continue
        out.append((f"{mod_name}.layer_scalar" if mod_name else "layer_scalar", buf))
    return out


_register(
    ["Gemma4ForCausalLM", "Gemma4ForConditionalGeneration", "Gemma4Model"],
    override_param=override_param,
    extra_buffers=extra_buffers,
)
