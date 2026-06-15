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
Monkey patch for Megatron-Core v0.15.0rc7 to add MoE replay support.

This patch enables passing pre-computed routing decisions (existing_topk_ids) through
the forward pass for deterministic MoE routing during inference or replay scenarios.

Usage:
    from axon.monkey_patches.megatron.moe_replay import apply_router_replay_patches, moe_routing_context, MoERoutingContext
    apply_router_replay_patches()  # Call BEFORE importing megatron modules that use MoE

    # Then use routing context:
    routing_ctx = MoERoutingContext(layer_routing_maps={layer_num: topk_ids, ...})
    set_moe_routing_context(routing_ctx)
    # ... forward pass ...
    set_moe_routing_context(None)  # Clear after use
"""

import collections
import os
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import wraps

import torch

_DEBUG = os.environ.get("DEBUG_MOE_REPLAY", "0") == "1"


def _debug_print(*args, **kwargs):
    if _DEBUG:
        print(*args, **kwargs, flush=True)


# ============================================================================
# MoE Routing Context
# ============================================================================


@dataclass
class MoERoutingContext:
    """
    Context object for holding MoE routing metadata during forward pass.

    This allows passing pre-computed routing decisions (existing_topk_ids) through
    the forward pass instead of setting them directly on the router modules.
    """

    layer_routing_maps: dict[int, torch.Tensor] = field(default_factory=dict)
    is_recompute: bool = False
    saved_routing_maps: dict[int, torch.Tensor] = field(default_factory=dict)

    def get_routing_map(self, layer_number: int) -> torch.Tensor | None:
        """Get the routing map for a specific layer."""
        routing_map = self.layer_routing_maps.get(layer_number, None)
        if routing_map is not None:
            return routing_map.clone()
        return None

    def set_routing_map(self, layer_number: int, routing_map: torch.Tensor):
        """Set the routing map for a specific layer."""
        self.layer_routing_maps[layer_number] = routing_map

    def save_routing_decision(self, layer_number: int, routing_map: torch.Tensor):
        """Save routing decision during forward for potential replay during recompute."""
        if not self.is_recompute:
            self.saved_routing_maps[layer_number] = routing_map.detach().clone() if routing_map is not None else None

    def get_saved_routing_decision(self, layer_number: int) -> torch.Tensor | None:
        """Get saved routing decision for recompute pass."""
        return self.saved_routing_maps.get(layer_number, None)

    def clear(self):
        """Clear all routing data."""
        self.layer_routing_maps.clear()
        self.saved_routing_maps.clear()
        self.is_recompute = False


# Thread-local storage for the routing context
_moe_routing_context_local = threading.local()


def get_moe_routing_context() -> MoERoutingContext | None:
    """Get the current MoE routing context for this thread."""
    return getattr(_moe_routing_context_local, "context", None)


def set_moe_routing_context(context: MoERoutingContext | None):
    """Set the MoE routing context for this thread."""
    _moe_routing_context_local.context = context


def clear_moe_layer_routing_queues(model_chunks):
    """Clear per-layer recompute routing FIFOs across all model chunks."""
    total_cleared = 0
    for chunk in model_chunks:
        for module in chunk.modules():
            if hasattr(module, "_moe_recompute_fifo"):
                n = len(module._moe_recompute_fifo)
                if n > 0:
                    total_cleared += n
                module._moe_recompute_fifo.clear()
                module._moe_routing_fresh = False
    if total_cleared > 0:
        _debug_print(f"[MoE FIFO] Cleared {total_cleared} entries from recompute FIFOs")


@contextmanager
def moe_routing_context(layer_routing_maps: dict[int, torch.Tensor] = None, is_recompute: bool = False):
    """
    Context manager for setting MoE routing context during forward pass.

    Usage:
        routing_maps = {1: tensor1, 2: tensor2, ...}  # layer_number -> existing_topk_ids
        with moe_routing_context(routing_maps):
            output = model(input_ids, ...)
    """
    old_context = get_moe_routing_context()

    if layer_routing_maps is not None:
        new_context = MoERoutingContext(
            layer_routing_maps=layer_routing_maps,
            is_recompute=is_recompute,
        )
    else:
        new_context = None

    set_moe_routing_context(new_context)
    try:
        yield new_context
    finally:
        set_moe_routing_context(old_context)


# ============================================================================
# Patching Functions
# ============================================================================

_patches_applied = False
_original_methods = {}


def _patch_topk_routing_with_score_function():
    """Patch moe_utils.topk_routing_with_score_function to support existing_topk_ids."""
    from megatron.core.transformer.moe import moe_utils

    if hasattr(moe_utils, "_original_topk_routing_with_score_function"):
        _debug_print("[MoE Patch] topk_routing_with_score_function already patched")
        return

    _original_methods["topk_routing_with_score_function"] = moe_utils.topk_routing_with_score_function

    def patched_topk_routing_with_score_function(
        logits: torch.Tensor,
        topk: int,
        use_pre_softmax: bool,
        num_groups: int,
        group_topk: int,
        scaling_factor: float,
        score_function: str = "softmax",
        expert_bias: torch.Tensor | None = None,
        fused: bool = False,
        existing_topk_ids: torch.Tensor | None = None,
    ):
        """Patched version that supports existing_topk_ids for MoE replay."""
        # Handle 3D logits by reshaping to 2D
        original_shape = logits.shape
        if logits.dim() == 3:
            # Shape is [seq_len, batch, num_experts] -> [seq_len * batch, num_experts]
            logits = logits.view(-1, logits.shape[-1])
            _debug_print(f"[MoE Patch] Reshaped 3D logits from {original_shape} to {logits.shape}")

        assert logits.dim() == 2, f"Expected 2D logits [num_tokens, num_experts], got {logits.dim()}."
        num_tokens, num_experts = logits.shape

        # If no existing_topk_ids, use original function (potentially fused)
        if existing_topk_ids is None:
            # Reshape back if needed for original function
            if len(original_shape) == 3:
                logits = logits.view(original_shape)
            return _original_methods["topk_routing_with_score_function"](
                logits=logits,
                topk=topk,
                use_pre_softmax=use_pre_softmax,
                num_groups=num_groups,
                group_topk=group_topk,
                scaling_factor=scaling_factor,
                score_function=score_function,
                expert_bias=expert_bias,
                fused=fused,
            )

        _debug_print(f"[MoE Patch] Using existing_topk_ids, shape: {existing_topk_ids.shape}")

        # With existing_topk_ids: compute probabilities for the given routing
        def compute_topk(scores, topk, num_groups, group_topk):
            if num_groups is not None and group_topk is not None:
                top_scores, top_indices = moe_utils.group_limited_topk(
                    scores=scores,
                    topk=topk,
                    num_tokens=scores.shape[0],
                    num_experts=num_experts,
                    num_groups=num_groups,
                    group_topk=group_topk,
                )
                return top_scores, top_indices
            else:
                return torch.topk(scores, k=topk, dim=1)

        top_indices = existing_topk_ids.clone()

        # Check if any fallback is needed (for -1 values)
        has_any_neg1 = (existing_topk_ids == -1).any(dim=1)
        needs_fallback = has_any_neg1.any().item()

        # Compute scores
        if score_function == "softmax":
            if use_pre_softmax:
                full_scores = torch.softmax(logits, dim=-1, dtype=torch.float32).type_as(logits)
            else:
                full_scores = logits
        elif score_function == "sigmoid":
            full_scores = torch.sigmoid(logits.float()).type_as(logits)
            if expert_bias is not None:
                full_scores_for_routing = full_scores + expert_bias
            else:
                full_scores_for_routing = full_scores
        else:
            raise ValueError(f"Invalid score_function: {score_function}")

        if needs_fallback:
            _debug_print(f"[MoE Patch] Fallback needed for {has_any_neg1.sum().item()} rows with -1")
            # Compute original top-k for fallback
            if score_function == "softmax":
                _, orig_top_indices = compute_topk(full_scores, topk, num_groups, group_topk)
            else:
                if expert_bias is not None:
                    _, orig_top_indices = compute_topk(full_scores_for_routing, topk, num_groups, group_topk)
                else:
                    _, orig_top_indices = compute_topk(full_scores, topk, num_groups, group_topk)

            # Replace rows with -1 with original top-k
            top_indices = torch.where(has_any_neg1.unsqueeze(1), orig_top_indices, existing_topk_ids)

        # Compute probabilities based on final top_indices
        if score_function == "softmax":
            if use_pre_softmax:
                probs = torch.gather(full_scores, dim=1, index=top_indices)
            else:
                selected_logits = torch.gather(logits, dim=1, index=top_indices)
                probs = torch.softmax(selected_logits, dim=-1, dtype=torch.float32).type_as(logits)
        elif score_function == "sigmoid":
            selected_scores = torch.gather(full_scores, dim=1, index=top_indices)
            if topk > 1:
                denom = selected_scores.sum(dim=-1, keepdim=True) + 1e-20
                probs = selected_scores / denom
            else:
                probs = selected_scores

        # Apply scaling
        if scaling_factor is not None:
            probs = probs * scaling_factor

        # Build sparse routing
        routing_probs = torch.zeros_like(logits).scatter(1, top_indices, probs)
        routing_map = torch.zeros_like(logits).int().scatter(1, top_indices, 1).bool()

        return routing_probs, routing_map

    moe_utils.topk_routing_with_score_function = patched_topk_routing_with_score_function
    moe_utils._original_topk_routing_with_score_function = _original_methods["topk_routing_with_score_function"]
    _debug_print("[MoE Patch] Patched topk_routing_with_score_function")


def _patch_router():
    """Patch TopKRouter to support existing_topk_ids parameter."""
    from megatron.core.transformer.moe import router as router_module

    if hasattr(router_module.TopKRouter, "_original_routing"):
        _debug_print("[MoE Patch] TopKRouter already patched")
        return

    _original_methods["TopKRouter.routing"] = router_module.TopKRouter.routing
    _original_methods["TopKRouter.forward"] = router_module.TopKRouter.forward

    def patched_routing(self, logits: torch.Tensor, existing_topk_ids: torch.Tensor = None):
        """Patched routing that supports existing_topk_ids."""
        from megatron.core.transformer.moe.moe_utils import (
            apply_router_token_dropping,
            compute_routing_scores_for_aux_loss,
            topk_routing_with_score_function,
        )

        _debug_print(
            f"[MoE Patch] TopKRouter.routing called, existing_topk_ids: {existing_topk_ids is not None}, logits shape: {logits.shape}"
        )

        # Original routing does this reshape from [seq_length, bsz, num_experts] to [num_tokens, num_experts]
        seq_length, bsz = logits.shape[:2]
        logits = logits.view(-1, self.config.num_moe_experts)

        # Apply Z-Loss (same as original)
        logits = self.apply_z_loss(logits)

        # Calculate probs and routing_map for token dispatching
        if self.routing_type == "sinkhorn":
            probs, routing_map = self.sinkhorn_load_balancing(logits)
        elif existing_topk_ids is not None:
            # MoE replay path: use pre-computed routing
            probs, routing_map = topk_routing_with_score_function(
                logits,
                self.topk,
                use_pre_softmax=self.config.moe_router_pre_softmax,
                num_groups=self.config.moe_router_num_groups,
                group_topk=self.config.moe_router_group_topk,
                scaling_factor=self.config.moe_router_topk_scaling_factor,
                score_function=self.score_function,
                expert_bias=self.expert_bias,
                fused=False,  # Can't use fused with existing_topk_ids
                existing_topk_ids=existing_topk_ids,
            )
        else:
            # Normal path
            probs, routing_map = topk_routing_with_score_function(
                logits,
                self.topk,
                use_pre_softmax=self.config.moe_router_pre_softmax,
                num_groups=self.config.moe_router_num_groups,
                group_topk=self.config.moe_router_group_topk,
                scaling_factor=self.config.moe_router_topk_scaling_factor,
                score_function=self.score_function,
                expert_bias=self.expert_bias,
                fused=self.config.moe_router_fusion,
            )

        # Apply token dropping to probs and routing_map (same as original)
        if self.config.moe_expert_capacity_factor is not None:
            probs, routing_map = apply_router_token_dropping(
                probs,
                routing_map,
                router_topk=self.topk,
                capacity_factor=self.config.moe_expert_capacity_factor,
                drop_policy=self.config.moe_token_drop_policy,
                pad_to_capacity=self.config.moe_pad_expert_input_to_capacity,
            )

        # Apply aux loss (same as original, skip in replay mode to avoid duplicate computation)
        if self.training and torch.is_grad_enabled() and self.is_aux_loss_enabled() and existing_topk_ids is None:
            routing_map_for_aux_loss, scores_for_aux_loss = compute_routing_scores_for_aux_loss(
                logits, self.topk, self.score_function, fused=self.config.moe_router_fusion
            )
            probs = self._apply_aux_loss(probs, scores_for_aux_loss, routing_map_for_aux_loss)
            probs = self._apply_seq_aux_loss(probs, scores_for_aux_loss, routing_map_for_aux_loss, seq_length, bsz)
            probs = self._apply_global_aux_loss(probs, scores_for_aux_loss, routing_map_for_aux_loss)

        # Update expert bias (skip in replay mode)
        if self.enable_expert_bias and torch.is_grad_enabled() and existing_topk_ids is None:
            with torch.no_grad():
                self.local_tokens_per_expert += routing_map.sum(dim=0)

        return probs, routing_map

    def patched_forward(self, input: torch.Tensor, existing_topk_ids: torch.Tensor = None):
        """Patched forward that supports existing_topk_ids."""
        _debug_print(f"[MoE Patch] TopKRouter.forward called, existing_topk_ids: {existing_topk_ids is not None}")

        if hasattr(self, "_maintain_float32_expert_bias"):
            self._maintain_float32_expert_bias()

        is_replay_mode = existing_topk_ids is not None

        if not is_replay_mode:
            # Normal mode: apply input jitter for regularization
            input = self.apply_input_jitter(input)
        # In replay mode: skip jitter to ensure deterministic probabilities

        logits = self.gating(input)

        if hasattr(self.config, "moe_router_force_load_balancing") and self.config.moe_router_force_load_balancing:
            if not is_replay_mode:
                from megatron.core.transformer.moe.moe_utils import apply_random_logits

                logits = apply_random_logits(logits)

        probs, routing_map = patched_routing(self, logits, existing_topk_ids)

        return probs, routing_map

    router_module.TopKRouter._original_routing = _original_methods["TopKRouter.routing"]
    router_module.TopKRouter._original_forward = _original_methods["TopKRouter.forward"]
    router_module.TopKRouter.routing = patched_routing
    router_module.TopKRouter.forward = patched_forward
    _debug_print("[MoE Patch] Patched TopKRouter.routing and TopKRouter.forward")


def _patch_moe_layer():
    """Patch MoELayer to read from routing context."""
    from megatron.core.transformer.moe import moe_layer as moe_layer_module

    if hasattr(moe_layer_module.MoELayer, "_original_router_and_preprocess"):
        _debug_print("[MoE Patch] MoELayer already patched")
        return

    _original_methods["MoELayer.__init__"] = moe_layer_module.MoELayer.__init__
    _original_methods["MoELayer.router_and_preprocess"] = moe_layer_module.MoELayer.router_and_preprocess
    _original_methods["MoELayer.forward"] = moe_layer_module.MoELayer.forward

    original_init = _original_methods["MoELayer.__init__"]
    original_router_and_preprocess = _original_methods["MoELayer.router_and_preprocess"]
    original_forward = _original_methods["MoELayer.forward"]

    @wraps(original_init)
    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        # Add MoE replay support attributes
        self.recompute = False
        self.saved_existing_topk_ids = None
        # Recompute-safe routing: FIFO queue fed during first-forward, consumed during recompute.
        self._moe_recompute_fifo = collections.deque()
        self._moe_routing_fresh = False  # set True by setup_moe_routing, consumed on first read

    @wraps(original_router_and_preprocess)
    def patched_router_and_preprocess(
        self, hidden_states: torch.Tensor, existing_topk_ids_override: torch.Tensor = None
    ):
        """Patched to support MoE replay via routing context.

        Bug fix: with recompute_granularity=full + PP>=2, the 1F1B schedule
        interleaves forward(mb_i+1) before backward(mb_i).  The global
        MoERoutingContext is overwritten by mb_i+1's forward, so mb_i's
        activation recompute would read the *wrong* routing.

        Fix: on first-forward (``_moe_routing_fresh=True``, set by
        ``setup_moe_routing``), read from the (correct) global context and
        push to a per-layer FIFO.  On recompute (flag consumed), pop from
        the FIFO.  Forwards and recomputes are both in micro-batch order,
        so FIFO ordering is correct.
        """
        residual = hidden_states
        existing_topk_ids = None

        if existing_topk_ids_override is not None:
            existing_topk_ids = existing_topk_ids_override
        elif getattr(self, "_moe_routing_fresh", False):
            # ---- FIRST FORWARD: global context is correct ----
            self._moe_routing_fresh = False
            routing_context = get_moe_routing_context()
            if routing_context is not None and self.layer_number is not None:
                existing_topk_ids = routing_context.get_routing_map(self.layer_number)
            if existing_topk_ids is None and hasattr(self.router, "existing_topk_ids"):
                existing_topk_ids = getattr(self.router, "existing_topk_ids", None)
                if existing_topk_ids is not None:
                    existing_topk_ids = existing_topk_ids.detach()
            # Save for recompute
            self._moe_recompute_fifo.append(existing_topk_ids.clone() if existing_topk_ids is not None else None)
        elif hasattr(self, "_moe_recompute_fifo") and self._moe_recompute_fifo:
            # ---- ACTIVATION RECOMPUTE: pop correct routing from FIFO ----
            existing_topk_ids = self._moe_recompute_fifo.popleft()
        else:
            # Fallback (no replay, or FIFO empty): read global context as before
            routing_context = get_moe_routing_context()
            if routing_context is not None and self.layer_number is not None:
                existing_topk_ids = routing_context.get_routing_map(self.layer_number)
            if existing_topk_ids is None and hasattr(self.router, "existing_topk_ids"):
                existing_topk_ids = getattr(self.router, "existing_topk_ids", None)
                if existing_topk_ids is not None:
                    existing_topk_ids = existing_topk_ids.detach()

        if existing_topk_ids is not None and existing_topk_ids.requires_grad:
            existing_topk_ids = existing_topk_ids.detach()

        probs, routing_map = self.router(hidden_states, existing_topk_ids)

        hidden_states, probs = self.token_dispatcher.dispatch_preprocess(hidden_states, routing_map, probs)
        return hidden_states, probs, residual

    @wraps(original_forward)
    def patched_forward(self, hidden_states: torch.Tensor, existing_topk_ids: torch.Tensor = None):
        """Patched forward with MoE replay support."""
        from megatron.core import parallel_state, tensor_parallel

        if (
            self.config.tensor_model_parallel_size > 1
            and self.config.moe_extended_tp
            and not self.config.sequence_parallel
        ):
            raise ValueError(
                "Extended TP for MoE is not supported when TP is enabled without also enabling sequence parallelism."
            )

        _debug_print(f"[MoE Patch] MoELayer.forward called for layer {self.layer_number}")

        # For FP8 mode, get routing decisions BEFORE entering checkpoint
        # This ensures MoE replay works correctly during activation recompute
        if self.config.fp8 and existing_topk_ids is None:
            existing_topk_ids = _get_existing_topk_ids_for_fp8(self, hidden_states)

        # MoE forward: route -> dispatch -> compute -> combine
        def custom_forward(hidden_states, existing_topk_ids=None):
            hidden_states, probs, residual = patched_router_and_preprocess(self, hidden_states, existing_topk_ids)
            dispatched_input, probs = self.dispatch(hidden_states, probs)
            output, shared_expert_output, mlp_bias = self.experts_compute(dispatched_input, probs, residual)
            output = self.combine(output, shared_expert_output)
            return output, mlp_bias

        if self.moe_layer_recompute:
            if self.config.fp8:
                try:
                    from megatron.core.extensions.transformer_engine import te_checkpoint

                    output, mlp_bias = te_checkpoint(
                        custom_forward,
                        False,
                        tensor_parallel.random.get_cuda_rng_tracker,
                        parallel_state.get_tensor_model_parallel_group(),
                        hidden_states,
                        existing_topk_ids,
                    )
                except ImportError:
                    # Fallback if TE not available
                    output, mlp_bias = tensor_parallel.checkpoint(
                        custom_forward, False, hidden_states, existing_topk_ids
                    )
            else:
                output, mlp_bias = tensor_parallel.checkpoint(custom_forward, False, hidden_states, existing_topk_ids)
        else:
            output, mlp_bias = custom_forward(hidden_states, existing_topk_ids)

        return output, mlp_bias

    moe_layer_module.MoELayer._original_init = original_init
    moe_layer_module.MoELayer._original_router_and_preprocess = original_router_and_preprocess
    moe_layer_module.MoELayer._original_forward = original_forward
    moe_layer_module.MoELayer.__init__ = patched_init
    moe_layer_module.MoELayer.router_and_preprocess = patched_router_and_preprocess
    moe_layer_module.MoELayer.forward = patched_forward
    _debug_print("[MoE Patch] Patched MoELayer.__init__, router_and_preprocess, and forward")


def _get_existing_topk_ids_for_fp8(moe_layer, hidden_states: torch.Tensor) -> torch.Tensor | None:
    """Get existing_topk_ids for MoE replay in FP8 mode.

    For FP8 mode, we need to compute routing decisions OUTSIDE the checkpoint boundary
    to ensure deterministic replay during recompute.
    """
    # Check routing context first
    routing_context = get_moe_routing_context()
    if routing_context is not None and moe_layer.layer_number is not None:
        ctx_topk_ids = routing_context.get_routing_map(moe_layer.layer_number)
        if ctx_topk_ids is not None:
            return ctx_topk_ids.detach() if ctx_topk_ids.requires_grad else ctx_topk_ids

    # Fallback to legacy method
    if hasattr(moe_layer.router, "existing_topk_ids") and moe_layer.router.existing_topk_ids is not None:
        return moe_layer.router.existing_topk_ids.detach()

    # For FP8 mode, compute routing outside checkpoint to enable MoE replay
    config = moe_layer.config
    router = moe_layer.router

    with torch.no_grad():
        input_for_routing = router.apply_input_jitter(hidden_states)
        logits = router.gating(input_for_routing)
        logits_2d = logits.view(-1, config.num_moe_experts)

        if router.score_function == "softmax":
            if config.moe_router_pre_softmax:
                scores = torch.softmax(logits_2d, dim=-1, dtype=torch.float32).type_as(logits_2d)
            else:
                scores = logits_2d
        else:  # sigmoid
            scores = torch.sigmoid(logits_2d.float()).type_as(logits_2d)
            if router.expert_bias is not None:
                scores = scores + router.expert_bias

        # Compute topk
        if config.moe_router_group_topk:
            from megatron.core.transformer.moe.moe_utils import group_limited_topk

            _, topk_ids = group_limited_topk(
                scores=scores,
                topk=router.topk,
                num_tokens=logits_2d.shape[0],
                num_experts=config.num_moe_experts,
                num_groups=config.moe_router_num_groups,
                group_topk=config.moe_router_group_topk,
            )
        else:
            _, topk_ids = torch.topk(scores, k=router.topk, dim=1)

        return topk_ids

    return None


def _patch_moe_module_exports():
    """Add routing context exports to megatron.core.transformer.moe module."""
    try:
        import megatron.core.transformer.moe as moe_module

        moe_module.MoERoutingContext = MoERoutingContext
        moe_module.get_moe_routing_context = get_moe_routing_context
        moe_module.set_moe_routing_context = set_moe_routing_context
        moe_module.moe_routing_context = moe_routing_context
        _debug_print("[MoE Patch] Added routing context exports to moe module")
    except Exception as e:
        _debug_print(f"[MoE Patch] Failed to patch moe module exports: {e}")


def _patch_moe_utils_exports():
    """Add routing context exports to megatron.core.transformer.moe.moe_utils module."""
    try:
        from megatron.core.transformer.moe import moe_utils

        moe_utils.MoERoutingContext = MoERoutingContext
        moe_utils.get_moe_routing_context = get_moe_routing_context
        moe_utils.set_moe_routing_context = set_moe_routing_context
        moe_utils.moe_routing_context = moe_routing_context
        _debug_print("[MoE Patch] Added routing context exports to moe_utils module")
    except Exception as e:
        _debug_print(f"[MoE Patch] Failed to patch moe_utils exports: {e}")


# ============================================================================
# Main Patch Application
# ============================================================================


def apply_router_replay_patches(force: bool = False):
    """Apply all monkey patches to Megatron-Core.

    Args:
        force: If True, apply patches even if already applied.

    This should be called BEFORE importing megatron modules that use MoE.
    """
    global _patches_applied

    if _patches_applied and not force:
        _debug_print("[MoE Patch] Patches already applied, skipping")
        return

    print("[MoE Patch] Applying Megatron MoE replay patches...")

    try:
        _patch_topk_routing_with_score_function()
    except Exception as e:
        print(f"[MoE Patch] WARNING: Failed to patch topk_routing_with_score_function: {e}")

    try:
        _patch_router()
    except Exception as e:
        print(f"[MoE Patch] WARNING: Failed to patch router: {e}")

    try:
        _patch_moe_layer()
    except Exception as e:
        print(f"[MoE Patch] WARNING: Failed to patch moe_layer: {e}")

    try:
        _patch_moe_utils_exports()
    except Exception as e:
        print(f"[MoE Patch] WARNING: Failed to patch moe_utils exports: {e}")

    try:
        _patch_moe_module_exports()
    except Exception as e:
        print(f"[MoE Patch] WARNING: Failed to patch moe module exports: {e}")

    _patches_applied = True
    print("[MoE Patch] Megatron MoE replay patches applied successfully")


def is_patched() -> bool:
    """Check if patches have been applied."""
    return _patches_applied


def get_original_methods() -> dict:
    """Get dictionary of original methods that were patched."""
    return _original_methods.copy()
