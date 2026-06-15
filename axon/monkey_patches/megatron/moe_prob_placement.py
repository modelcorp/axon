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
Monkey patch to move MoE routing-probability multiplication from BEFORE
the down-projection (FC2) to AFTER it, matching vLLM's computation order.

Background
----------
Megatron-Core multiplies each token's expert output by its routing
probability **before** the FC2 matmul::

    intermediate = activation(FC1(x)) * prob   # prob applied here
    output = FC2(intermediate)

vLLM's Triton fused-MoE kernel applies the routing probability **after**
the FC2 matmul::

    intermediate = activation(FC1(x))
    output = FC2(intermediate) * prob           # prob applied here

Mathematically these are equivalent (scalar × matrix commutes), but in
bf16 the different operand magnitudes cause ~0.5 max-diff per expert per
layer, compounding to several logprob-units over 48 layers.

Usage::

    from axon.monkey_patches.megatron.moe_prob_placement import (
        apply_moe_post_fc2_prob_patch,
    )
    apply_moe_post_fc2_prob_patch()
"""

import torch

_patched = False


def apply_moe_post_fc2_prob_patch():
    """Patch TEGroupedMLP.forward so probs are applied after FC2."""
    global _patched
    if _patched:
        return
    _patched = True

    from megatron.core.transformer.moe.experts import TEGroupedMLP

    if getattr(TEGroupedMLP, "_post_fc2_prob_patched", False):
        return
    TEGroupedMLP._post_fc2_prob_patched = True

    _orig_forward = TEGroupedMLP.forward

    def _patched_forward(
        self,
        permuted_local_hidden_states: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        permuted_probs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """TEGroupedMLP forward with probs applied AFTER FC2 (matching vLLM).

        Delegates to the original forward with probs=1 so all fused kernels,
        activation checkpointing, FP8 paths, etc. are fully preserved.
        The only overhead is a single elementwise multiply on the FC2 output.
        """
        # If top-k == 1 and moe_apply_probs_on_input, keep original behavior
        if self.config.moe_apply_probs_on_input:
            return _orig_forward(self, permuted_local_hidden_states, tokens_per_expert, permuted_probs)

        # Run the full original forward with probs=1 (preserves all fused
        # kernels, activation recompute, FP8 padding, TE activation, etc.)
        dummy_probs = torch.ones_like(permuted_probs)
        output, output_bias = _orig_forward(self, permuted_local_hidden_states, tokens_per_expert, dummy_probs)

        # Apply routing probs AFTER FC2, matching vLLM's Triton kernel
        original_dtype = output.dtype
        output = output * permuted_probs.unsqueeze(-1)
        output = output.to(original_dtype)

        return output, output_bias

    TEGroupedMLP.forward = _patched_forward
    print("[MoE Patch] Applied post-FC2 prob placement patch to TEGroupedMLP", flush=True)
