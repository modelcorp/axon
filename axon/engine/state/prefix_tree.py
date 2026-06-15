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
from __future__ import annotations

import uuid

import torch


def _is_all_neg1(moe_entry) -> bool:
    """Check if a moe_routermap entry is all -1 (padding placeholder).

    The moe_routermap values may be floats (float16/float32), so we compare
    with a small tolerance.
    """
    if moe_entry is None:
        return False
    try:
        # Handle torch tensors directly (most common case)
        if isinstance(moe_entry, torch.Tensor):
            if moe_entry.numel() == 0:
                return False
            # Check if all values are close to -1
            return torch.all(torch.abs(moe_entry - (-1)) < 1e-5).item()

        def is_neg1(v):
            # Handle both int and float -1 values
            if isinstance(v, int | float):
                return abs(v - (-1)) < 1e-5
            return False

        # Handle nested lists (e.g., [num_layers, num_experts] shape)
        if isinstance(moe_entry, list | tuple):
            if len(moe_entry) == 0:
                return False
            has_any_value = False
            for row in moe_entry:
                if isinstance(row, list | tuple):
                    if len(row) == 0:
                        continue
                    has_any_value = True
                    if not all(is_neg1(v) for v in row):
                        return False
                else:
                    has_any_value = True
                    if not is_neg1(row):
                        return False
            return has_any_value
        # Handle numpy arrays
        elif hasattr(moe_entry, "tolist"):
            return _is_all_neg1(moe_entry.tolist())
        return False
    except (TypeError, AttributeError):
        return False


class PrefixTreeNode:
    def __init__(
        self,
        token_id: int,
        token_str: str,
        mask: int,
        logprob: float,
        moe_routermap: list | None = None,
        step_idx: int = -1,
        multi_modal_data=None,
    ):
        self.uid = uuid.uuid4().int
        self.token_id = token_id
        self.token_str = token_str
        self.masks = {mask}  # set of zero or one
        self.logprob = logprob
        self.moe_routermap = moe_routermap if moe_routermap is not None else []
        self.step_idx = step_idx  # Smallest step index seen so far
        self.children: dict[int, PrefixTreeNode] = {}
        self.multi_modal_data = multi_modal_data
        # MM expansion: only set on first token of an expanded image/video region.
        # mm_expansion_count=N means this node starts a run of N consecutive pad
        # tokens that resulted from processor-expanding a single placeholder.
        self.mm_expansion_count: int = 0
        self.mm_content_hash: str = ""

    def __hash__(self):
        return self.uid

    def get_child(self, token_id: int) -> PrefixTreeNode | None:
        return self.children.get(token_id, None)

    def add_child(
        self,
        token_id: int,
        token_str: str,
        mask: int,
        logprob: float,
        moe_routermap: list | None = None,
        step_idx: int = -1,
        multi_modal_data=None,  # Only stored at the deepest children to save memory
    ) -> PrefixTreeNode:
        if token_id not in self.children:
            self.children[token_id] = PrefixTreeNode(
                token_id,
                token_str,
                mask,
                logprob,
                moe_routermap if moe_routermap is not None else [],
                step_idx,
                multi_modal_data,
            )
        else:
            node = self.children[token_id]
            # Update metadata if provided
            node.masks.add(mask)
            # Keep the smallest step_idx seen so far
            if step_idx != -1:
                if node.step_idx == -1:
                    node.step_idx = step_idx
                else:
                    node.step_idx = min(node.step_idx, step_idx)

            # Backfill token_str for nodes that were inserted without one
            if token_str and not node.token_str:
                node.token_str = token_str
            # Only update multi_modal_data if the new value is not None.
            # During insertion, multi_modal_data is only set on the deepest
            # (last) token of each step — all other positions pass None.
            # Without this guard, a subsequent insertion that re-traverses an
            # existing node as a non-leaf would overwrite the stored mm_data
            # with None, destroying the original step's multimodal context.
            if multi_modal_data is not None:
                node.multi_modal_data = multi_modal_data
            # Update moe_routermap if:
            # 1. The existing one is empty, OR
            # 2. The existing one is all -1 (padding) AND the new one is valid (not all -1)
            # This handles:
            # - Prefix cache hit cases where first insertion has no moe_routermap
            # - Multi-turn cases where last token of previous turn has -1 padding
            #   but next turn's vLLM returns the correct moe_routermap for that position
            # Check if moe_routermap is valid
            has_new_moe = (isinstance(moe_routermap, torch.Tensor) and moe_routermap.numel() > 0) or (
                isinstance(moe_routermap, list) and len(moe_routermap) > 0
            )
            if has_new_moe:
                should_update = False
                has_existing_moe = (
                    isinstance(node.moe_routermap, torch.Tensor) and node.moe_routermap.numel() > 0
                ) or (isinstance(node.moe_routermap, list) and len(node.moe_routermap) > 0)
                if not has_existing_moe:
                    # Case 1: existing is empty
                    should_update = True
                elif _is_all_neg1(node.moe_routermap) and not _is_all_neg1(moe_routermap):
                    # Case 2: existing is all -1 padding, new one has valid data
                    should_update = True

                if should_update:
                    node.moe_routermap = moe_routermap
        return self.children[token_id]

    def is_leaf(self) -> bool:
        return len(self.children) == 0


class PrefixTree:
    """Trie to track token ids and per-step metadata (mask, logprob, moe_routermap)."""

    def __init__(self):
        self.root = PrefixTreeNode(token_id=-1, token_str="", mask=-1, logprob=0.0)

    def clear(self):
        self.root = PrefixTreeNode(token_id=-1, token_str="", mask=-1, logprob=0.0)

    def insert(
        self,
        token_ids: list[int],
        token_strs: list[str],
        masks: list[int],
        logprobs: list[float],
        moe_routermaps: list | None = None,
        step_idx: int = -1,
        multi_modal_data=None,  # Only stored at the deepest children to save memory
        mm_expansions: list[tuple] | None = None,
    ):
        """Insert a token sequence into the trie.

        Args:
            mm_expansions: Optional list of (token_position, count, content_hash) tuples
                that mark MM expansion regions.  The node at ``token_position`` will be
                annotated with ``mm_expansion_count=count`` and ``mm_content_hash=hash``
                so that future :meth:`longest_text_prefix` calls can skip over the
                expanded pad tokens when matching against ``input_key`` text.
        """
        # Handle tensor format: if moe_routermaps is a tensor [seq_len, num_layers, num_experts],
        # index into it for each token to get [num_layers, num_experts] per node
        if isinstance(moe_routermaps, torch.Tensor):
            assert len(moe_routermaps) == len(token_ids), (
                f"moe_routermaps tensor length {len(moe_routermaps)} must match token_ids length {len(token_ids)}"
            )
            moe_routermaps_list = [moe_routermaps[i] for i in range(len(moe_routermaps))]
        else:
            moe_routermaps_list = moe_routermaps if moe_routermaps else [None for _ in token_ids]

        assert len(token_ids) == len(token_strs) == len(masks) == len(logprobs) == len(moe_routermaps_list), (
            "All inputs must be same length"
        )

        # Build position lookup for MM expansions
        expansion_at: dict[int, tuple[int, str]] = {}
        if mm_expansions:
            for tok_pos, count, chash in mm_expansions:
                expansion_at[tok_pos] = (count, chash)

        node = self.root
        for i, (token_id, token_str, mask, logprob, moe_routermap) in enumerate(
            zip(token_ids, token_strs, masks, logprobs, moe_routermaps_list, strict=False)
        ):
            node = node.add_child(
                token_id,
                token_str,
                mask,
                logprob,
                moe_routermap,
                step_idx,
                multi_modal_data if i == len(token_ids) - 1 else None,
            )
            # Annotate MM expansion start nodes
            if i in expansion_at:
                count, chash = expansion_at[i]
                node.mm_expansion_count = count
                node.mm_content_hash = chash
        return node

    def size(self) -> int:
        """
        Return the total number of nodes in the trie.

        This does not include the root node.

        Returns:
            int: The total count of nodes in the trie.
        """
        count = -1
        stack = [self.root]

        while stack:
            node = stack.pop()
            count += 1
            for child in node.children.values():
                stack.append(child)

        return count

    def longest_prefix(self, token_ids: list[int]) -> list[int]:
        """
        Return the sequence of token ids and token strings that forms the
        longest prefix of `token_ids` present in the trie.

        If no token matches, returns ([], []).

        Args:
            token_ids: List of token ids to match against the trie

        Returns:
            tuple[list[int], list[str]]: Matched token ids and their corresponding strings
        """
        matched_ids: list[int] = []

        node = self.root
        for token_id in token_ids:
            child = node.get_child(token_id)
            if child is None:
                # No match for this token, return what we have so far
                break
            matched_ids.append(child.token_id)
            node = child

        return matched_ids

    def longest_text_prefix(
        self, text: str, mm_regions: list[tuple] | None = None
    ) -> tuple[list[int], list[str], int, list[tuple]]:
        """Walk the trie matching by ``token_str`` against *text*.

        Each node stores the exact string its token represents (populated from
        vLLM per-token strings for response tokens, or from tokenizer offsets
        for prompt tokens).  This method greedily follows children whose
        ``token_str`` matches the next characters in *text*, accumulating the
        ground-truth token ids and per-token strings.

        For multimodal inputs, the tree may contain expanded pad token regions
        (N consecutive pad tokens from a single placeholder).  When *mm_regions*
        is provided, the walk detects these expansions via ``mm_expansion_count``
        on the first pad node and skips over them, advancing *text* past the
        corresponding placeholder string.

        Args:
            text: The rendered prompt string (``input_key``).
            mm_regions: Optional list of ``(text_start, text_end, modality,
                content_hash, data)`` tuples describing where MM placeholders
                appear in *text*.  ``None`` or ``[]`` → pure text walk.

        Returns:
            (matched_token_ids, matched_token_strs, text_pos, recovered_mm):
            * ``recovered_mm`` is a list of ``(modality, data)`` tuples for each
              MM region successfully matched in the prefix.  Empty for text-only.
        """
        # Build position lookup for O(1) "is this char position an MM start?"
        mm_at: dict[int, tuple[int, str, str, object]] = {}
        if mm_regions:
            for start, end, modality, chash, data in mm_regions:
                mm_at[start] = (end, modality, chash, data)

        node = self.root
        pos = 0
        matched_ids: list[int] = []
        matched_strs: list[str] = []
        recovered_mm: list[tuple] = []

        while pos < len(text) and node.children:
            # ── Check for MM expansion region first ───────────────
            mm_child = None
            if pos in mm_at:
                for child in node.children.values():
                    if child.mm_expansion_count > 0:
                        mm_child = child
                        break

            if mm_child is not None:
                end, modality, chash, data = mm_at[pos]

                # Content changed → stop matching (suffix will re-tokenize)
                if mm_child.mm_content_hash and chash != mm_child.mm_content_hash:
                    break

                # Walk through all N expanded pad tokens in the tree
                current = mm_child
                matched_ids.append(current.token_id)
                matched_strs.append(current.token_str)
                for j in range(1, mm_child.mm_expansion_count):
                    next_node = current.get_child(current.token_id)
                    if next_node is None:
                        break  # tree truncated — stop here
                    current = next_node
                    matched_ids.append(current.token_id)
                    matched_strs.append(current.token_str)

                pos = end  # advance past full placeholder text in input_key
                node = current
                recovered_mm.append((modality, data))
                continue

            # ── Normal text token: pick longest matching child ────
            best_child = None
            best_len = 0
            for child in node.children.values():
                s = child.token_str
                if s and text[pos : pos + len(s)] == s and len(s) > best_len:
                    best_child = child
                    best_len = len(s)
            if best_child is None:
                break
            matched_ids.append(best_child.token_id)
            matched_strs.append(best_child.token_str)
            pos += best_len
            node = best_child

        return matched_ids, matched_strs, pos, recovered_mm

    def __repr__(self) -> str:
        """
        Return a tree-like string representation of the PrefixTree.
        Each node shows: (token_id, token_str, logprob, masks)

        Linear paths (no branching) are displayed vertically.
        Branches expand horizontally.

        Uses iterative approach to avoid RecursionError on deep trees.
        """
        lines = []
        lines.append("PrefixTree:")

        def format_node(node: PrefixTreeNode) -> str:
            """Format a node's data for display."""
            if node.token_id == -1:  # root node
                return "ROOT"
            masks_str = "{" + ",".join(map(str, sorted(node.masks))) + "}"
            return f"(id={node.token_id}, str={repr(node.token_str)}, logprob={node.logprob:.4f}, masks={masks_str}, step_idx={node.step_idx})"

        # Start with root
        lines.append(format_node(self.root))

        # Use iterative approach with explicit stack to avoid recursion limit
        # Stack items: (node, prefix, children_index, children_list)
        # children_index: current index into children_list, or -1 for "check children" state
        # children_list: list of children when branching, or None for linear check
        stack: list[tuple[PrefixTreeNode, str, int, list[PrefixTreeNode] | None]] = []

        # Initialize: push root in "check children" state
        stack.append((self.root, "", -1, None))

        while stack:
            node, prefix, child_idx, children_list = stack.pop()

            if child_idx == -1:
                # "Check children" state - determine how to display this node's children
                node_children = list(node.children.values())

                if len(node_children) == 0:
                    # Leaf node - nothing to display below
                    continue
                elif len(node_children) == 1:
                    # Single child - display vertically and continue linearly
                    child = node_children[0]
                    lines.append(prefix + "  ↓")
                    lines.append(prefix + format_node(child))
                    # Push child in "check children" state (same prefix for linear)
                    stack.append((child, prefix, -1, None))
                else:
                    # Multiple children - branch horizontally
                    # Push this node back with child_idx=0 to start processing children
                    stack.append((node, prefix, 0, node_children))
            else:
                # Processing branching children
                assert children_list is not None

                if child_idx >= len(children_list):
                    # All children processed
                    continue

                child = children_list[child_idx]
                is_last_child = child_idx == len(children_list) - 1

                if is_last_child:
                    connector = "  └─→ "
                    extension = "      "
                else:
                    connector = "  ├─→ "
                    extension = "  │   "

                lines.append(prefix + connector + format_node(child))

                # Push remaining siblings first (so they're processed after current child's subtree)
                if child_idx + 1 < len(children_list):
                    stack.append((node, prefix, child_idx + 1, children_list))

                # Push this child in "check children" state
                stack.append((child, prefix + extension, -1, None))

        return "\n".join(lines)
