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
"""Gemma4 FSDP training support.

Two compatibility patches are applied here:

1. **Multimodal text-only causal mask**: Gemma4 is architecturally a multimodal
   model (``Gemma4ForConditionalGeneration``) even for text-only tasks.  Its
   ``Gemma4Model.forward`` requires ``mm_token_type_ids`` to build causal
   masks (0=text, 1=image, 2=audio).  We install a forward pre-hook on the
   inner ``Gemma4Model`` that auto-injects all-zeros ``mm_token_type_ids``.

2. **KV-sharing under FSDP2 + gradient checkpointing** (E2B / E4B variants):
   transformers >= 5.5.x threads a ``shared_kv_states`` dict between Gemma4
   decoder layers as a kwarg.  Two interactions break this:

   * **FSDP2's ``_pre_forward``** rebuilds every plain ``dict`` in kwargs via
     ``_apply_to_tensors`` (to cast inputs to ``param_dtype``), so each
     wrapped layer call gets a freshly-rebuilt empty copy.

   * **HF's ``GradientCheckpointingLayer.__call__``** captures kwargs into a
     ``functools.partial`` *before* any forward pre-hook fires; backward's
     recompute then replays from that captured reference.  A pre-hook fix
     therefore can't reach the recompute path.

   Fix: override ``Gemma4TextDecoderLayer.__call__`` at the class level so
   that kwargs[``shared_kv_states``] is swapped to a stable per-forward
   carrier *before* delegating to ``GradientCheckpointingLayer.__call__``.
   The carrier supports ``carrier[idx]`` / ``carrier[idx] = ...`` (the only
   ops Gemma4 attention performs on the dict) but is not a ``dict`` — so
   ``_apply_to_tensors`` doesn't recognize it for recursive rebuild and
   preserves the reference across layer calls and across the
   forward/backward boundary in gradient checkpointing.  A pre-hook on
   ``Gemma4TextModel`` resets the carrier at the start of each forward.
"""

from __future__ import annotations

import threading
from typing import Any

import torch

# Per-thread holder for the in-flight Gemma4TextModel.forward call.  The
# Gemma4TextModel pre-hook installs a fresh carrier here at the start of each
# forward; the patched decoder layer ``__call__`` reads it back to swap the
# rebuilt-by-FSDP2 dict before ``GradientCheckpointingLayer`` captures kwargs.
_kv_state = threading.local()


class _SharedKVStatesCarrier:
    """Stable indexable container for Gemma4's ``shared_kv_states``.

    Implements only ``__getitem__`` / ``__setitem__`` / ``__contains__`` —
    the operations Gemma4 attention performs on the dict.  Crucially it is
    **not** a ``dict`` subclass, so PyTorch's ``_apply_to_tensors`` (used by
    FSDP2 to cast forward inputs) doesn't recognize it for recursive rebuild
    and preserves the reference across layer calls and across the
    forward/backward boundary in gradient checkpointing.
    """

    __slots__ = ("_data",)

    def __init__(self) -> None:
        self._data: dict[int, Any] = {}

    def __getitem__(self, key: int) -> Any:
        return self._data[key]

    def __setitem__(self, key: int, value: Any) -> None:
        self._data[key] = value

    def __contains__(self, key: int) -> bool:
        return key in self._data

    def __repr__(self) -> str:
        return f"_SharedKVStatesCarrier(keys={sorted(self._data.keys())})"


def _begin_text_model_forward(module, args, kwargs):
    """Reset the per-forward carrier when a new Gemma4TextModel.forward starts."""
    _kv_state.carrier = _SharedKVStatesCarrier()


def _patch_decoder_layer_call(decoder_cls: type) -> None:
    """Install a class-level ``__call__`` override on ``Gemma4TextDecoderLayer``
    so the per-forward carrier replaces the plain dict in kwargs *before*
    ``GradientCheckpointingLayer.__call__`` captures them in a partial.

    Idempotent: re-patching is a no-op.
    """
    if getattr(decoder_cls, "_axon_kv_carrier_patched", False):
        return
    orig_call = decoder_cls.__call__

    def patched_call(self, *args, **kwargs):
        sks = kwargs.get("shared_kv_states")
        if sks is not None and not isinstance(sks, _SharedKVStatesCarrier):
            # Pre-hook normally provides the carrier; fall back to a fresh
            # one if forward was invoked without going through the model
            # (defensive — shouldn't happen in normal training).
            carrier = getattr(_kv_state, "carrier", None) or _SharedKVStatesCarrier()
            _kv_state.carrier = carrier
            kwargs["shared_kv_states"] = carrier
        return orig_call(self, *args, **kwargs)

    decoder_cls.__call__ = patched_call
    decoder_cls._axon_kv_carrier_patched = True


def _patch_gemma4_kv_carrier(model: torch.nn.Module) -> None:
    """Apply the KV-sharing carrier patch when the model uses KV sharing.

    No-op for variants without KV sharing (e.g. 31B, 26B-A4B).
    """
    inner = getattr(model, "model", None)
    if inner is None:
        return
    text_model = getattr(inner, "language_model", inner)
    if "Gemma4TextModel" not in type(text_model).__name__:
        return
    text_config = getattr(text_model, "config", None)
    if text_config is None or getattr(text_config, "num_kv_shared_layers", 0) == 0:
        return

    text_model.register_forward_pre_hook(_begin_text_model_forward, with_kwargs=True)
    if text_model.layers:
        _patch_decoder_layer_call(type(text_model.layers[0]))


def patch_gemma4_for_text_only_training(model: torch.nn.Module) -> None:
    """Install Gemma4 training-time compatibility patches.

    Args:
        model: A ``Gemma4ForConditionalGeneration`` instance (or any module
            whose ``.model`` child is a ``Gemma4Model``).
    """
    inner = getattr(model, "model", None)
    if inner is None or "Gemma4Model" not in type(inner).__name__:
        return

    def _inject_gemma4_training_args(module, args, kwargs):
        if kwargs.get("mm_token_type_ids") is None:
            input_ids = kwargs.get("input_ids")
            if input_ids is None and len(args) > 0:
                input_ids = args[0]
            if input_ids is not None:
                kwargs["mm_token_type_ids"] = torch.zeros_like(input_ids)
        return args, kwargs

    inner.register_forward_pre_hook(_inject_gemma4_training_args, with_kwargs=True)
    _patch_gemma4_kv_carrier(model)
