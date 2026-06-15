import torch

from axon.engine.state.prefix_tree import PrefixTree, _is_all_neg1


class TestPrefixTree:
    """Test suite for PrefixTree (Trie) implementation."""

    def test_empty_tree(self):
        """Test that an empty tree has size 0."""
        trie = PrefixTree()
        assert trie.size() == 0, f"Expected size 0 for empty trie, got {trie.size()}"

    def test_simple_two_branch_tree(self):
        """Test a simple tree with two branches from the same root."""
        trie = PrefixTree()
        trie.insert(
            token_ids=[101, 102],
            token_strs=["He", "llo"],
            masks=[0, 0],
            logprobs=[-1.2, -0.5],
        )
        trie.insert(
            token_ids=[101, 201],
            token_strs=["He", "y"],
            masks=[0, 0],
            logprobs=[-1.2, -0.8],
        )
        assert trie.size() == 3, f"Expected size 3, got {trie.size()}"

    def test_deep_linear_path(self):
        """Test a deep linear path (10 tokens deep)."""
        trie = PrefixTree()
        deep_tokens = list(range(100, 110))
        deep_strs = [f"tok{i}" for i in range(10)]
        deep_masks = [0] * 10
        deep_logprobs = [-0.1 * i for i in range(10)]
        trie.insert(deep_tokens, deep_strs, deep_masks, deep_logprobs)

        assert trie.size() == 10, f"Expected size 10, got {trie.size()}"

        # Test longest_prefix with token ids
        ids = trie.longest_prefix(deep_tokens)
        assert ids == deep_tokens, f"Expected {deep_tokens}, got {ids}"

    def test_wide_branching(self):
        """Test wide branching with 5 branches from root."""
        trie = PrefixTree()
        branches = [
            ([1], ["A"], [0], [-1.0]),
            ([2], ["B"], [0], [-1.1]),
            ([3], ["C"], [0], [-1.2]),
            ([4], ["D"], [0], [-1.3]),
            ([5], ["E"], [0], [-1.4]),
        ]
        for token_ids, token_strs, masks, logprobs in branches:
            trie.insert(token_ids, token_strs, masks, logprobs)

        assert trie.size() == 5, f"Expected size 5, got {trie.size()}"

    def test_complex_multi_level_branching(self):
        """Test complex tree with multiple levels of branching."""
        trie = PrefixTree()

        # Create a tree like this:
        #        ROOT
        #         |
        #       "The"
        #       /   \
        #    "quick" "slow"
        #     /  \      |
        #  "brown""red" "fox"
        #    |      |     |
        #  "fox"  "fox" "jumps"

        sequences = [
            # The quick brown fox
            ([1, 2, 3, 4], ["The", "quick", "brown", "fox"], [0, 0, 0, 0], [-0.1, -0.2, -0.3, -0.4]),
            # The quick red fox
            ([1, 2, 5, 6], ["The", "quick", "red", "fox"], [1, 1, 1, 1], [-0.1, -0.2, -0.5, -0.6]),
            # The slow fox jumps
            ([1, 7, 8, 9], ["The", "slow", "fox", "jumps"], [2, 2, 2, 2], [-0.1, -0.3, -0.4, -0.5]),
        ]

        for token_ids, token_strs, masks, logprobs in sequences:
            trie.insert(token_ids, token_strs, masks, logprobs)

        # Expected nodes: The, quick, slow, brown, red, fox (under brown), fox (under red), fox (under slow), jumps
        expected_size = 9
        assert trie.size() == expected_size, f"Expected size {expected_size}, got {trie.size()}"

        # Test various prefix matches
        ids = trie.longest_prefix([1, 2, 3, 4])
        assert ids == [1, 2, 3, 4], f"Expected [1, 2, 3, 4], got {ids}"

        ids = trie.longest_prefix([1, 2, 5, 6])
        assert ids == [1, 2, 5, 6], f"Expected [1, 2, 5, 6], got {ids}"

        ids = trie.longest_prefix([1, 7, 8, 9])
        assert ids == [1, 7, 8, 9], f"Expected [1, 7, 8, 9], got {ids}"

        ids = trie.longest_prefix([1])
        assert ids == [1], f"Expected [1], got {ids}"

    def test_very_deep_tree_with_multiple_branch_levels(self):
        """Test very deep tree with branching at various levels."""
        trie = PrefixTree()

        # Create paths that branch at different depths
        sequences = [
            # Main path: A-B-C-D-E-F-G-H-I-J-K-L-M-N-O
            (
                [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
                ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O"],
                [0] * 15,
                [-0.1] * 15,
            ),
            # Branch at depth 3: A-B-C-X-Y-Z
            ([10, 11, 12, 30, 31, 32], ["A", "B", "C", "X", "Y", "Z"], [1] * 6, [-0.2] * 6),
            # Branch at depth 5: A-B-C-D-E-P-Q-R
            ([10, 11, 12, 13, 14, 40, 41, 42], ["A", "B", "C", "D", "E", "P", "Q", "R"], [2] * 8, [-0.3] * 8),
            # Branch at depth 8: A-B-C-D-E-F-G-H-S-T
            (
                [10, 11, 12, 13, 14, 15, 16, 17, 50, 51],
                ["A", "B", "C", "D", "E", "F", "G", "H", "S", "T"],
                [3] * 10,
                [-0.4] * 10,
            ),
        ]

        for token_ids, token_strs, masks, logprobs in sequences:
            trie.insert(token_ids, token_strs, masks, logprobs)

        # Expected nodes:
        # A, B, C (shared by all)
        # D, E (shared by paths 1 and 3)
        # X, Y, Z (path 2 branch)
        # F, G, H (shared by paths 1 and 4)
        # P, Q, R (path 3 branch)
        # I, J, K, L, M, N, O (path 1 continuation)
        # S, T (path 4 branch)
        # Total: 3 + 2 + 3 + 3 + 3 + 7 + 2 = 23
        expected_size = 23
        actual_size = trie.size()
        assert actual_size == expected_size, f"Expected size {expected_size}, got {actual_size}"

    def test_mask_merging(self):
        """Test that multiple insertions of the same path merge masks correctly."""
        trie = PrefixTree()

        # Insert same path with different masks
        trie.insert([100, 101], ["Hello", "World"], [0, 0], [-1.0, -1.0])
        trie.insert([100, 101], ["Hello", "World"], [1, 1], [-1.0, -1.0])
        trie.insert([100, 101], ["Hello", "World"], [2, 2], [-1.0, -1.0])

        assert trie.size() == 2, f"Expected size 2 after mask merging, got {trie.size()}"

        hello_node = trie.root.children[100]
        world_node = hello_node.children[101]

        assert {0, 1, 2} == hello_node.masks, f"Expected masks {{0,1,2}} for 'Hello', got {hello_node.masks}"
        assert {0, 1, 2} == world_node.masks, f"Expected masks {{0,1,2}} for 'World', got {world_node.masks}"

    def test_binary_tree_structure(self):
        """Test binary tree structure (complete binary tree of depth 4)."""
        trie = PrefixTree()

        # Create a complete binary tree where each node has 2 children
        # Level 0: 1 node (root)
        # Level 1: 2 nodes
        # Level 2: 4 nodes
        # Level 3: 8 nodes
        # Total paths: 8 (each leaf represents one path)

        base_tokens = [1, 2, 3, 4]  # depth 4
        for i in range(8):  # 2^3 = 8 different paths
            # Convert i to binary and use to determine path
            path_tokens = base_tokens.copy()
            path_strs = []
            for depth in range(4):
                bit = (i >> (3 - depth)) & 1
                path_tokens[depth] = path_tokens[depth] * 10 + bit
                path_strs.append(f"L{depth}_{bit}")

            trie.insert(path_tokens, path_strs, [i] * 4, [-0.1] * 4)

        # Expected: 1 + 2 + 4 + 8 = 15 nodes (complete binary tree)
        expected_size = 15
        actual_size = trie.size()
        assert actual_size == expected_size, f"Expected size {expected_size}, got {actual_size}"

    def test_longest_prefix_matching_with_overlaps(self):
        """Test longest prefix matching with complex overlapping paths."""
        trie = PrefixTree()

        sequences = [
            ([1, 2, 3], ["over", "lap", "ping"], [0] * 3, [-0.1] * 3),
            ([1, 2, 3, 4], ["over", "lap", "ping", "word"], [1] * 4, [-0.1] * 4),
            ([1, 2, 5], ["over", "lap", "ped"], [2] * 3, [-0.1] * 3),
            ([1, 6], ["over", "head"], [3] * 2, [-0.1] * 2),
        ]

        for token_ids, token_strs, masks, logprobs in sequences:
            trie.insert(token_ids, token_strs, masks, logprobs)

        # Test exact matches
        ids = trie.longest_prefix([1, 2, 3])
        assert ids == [1, 2, 3], f"Expected [1, 2, 3], got {ids}"

        ids = trie.longest_prefix([1, 2, 3, 4])
        assert ids == [1, 2, 3, 4], f"Expected [1, 2, 3, 4], got {ids}"

        ids = trie.longest_prefix([1, 2, 5])
        assert ids == [1, 2, 5], f"Expected [1, 2, 5], got {ids}"

        ids = trie.longest_prefix([1, 6])
        assert ids == [1, 6], f"Expected [1, 6], got {ids}"

        # Test partial matches
        ids = trie.longest_prefix([1, 2])
        assert ids == [1, 2], f"Expected [1, 2], got {ids}"

        ids = trie.longest_prefix([1])
        assert ids == [1], f"Expected [1], got {ids}"

        # Test no match
        ids = trie.longest_prefix([99])
        assert ids == [], f"Expected [], got {ids}"

    def test_star_shaped_tree(self):
        """Test star-shaped tree with many branches from single parent."""
        trie = PrefixTree()

        # Common prefix
        trie.insert([1000], ["Common"], [0], [-1.0])

        # Add 10 different branches
        for i in range(10):
            trie.insert([1000, 2000 + i], ["Common", f"Branch{i}"], [i, i], [-1.0, -1.0 - i * 0.1])

        expected_size = 11  # 1 common + 10 branches
        assert trie.size() == expected_size, f"Expected size {expected_size}, got {trie.size()}"

    def test_clear(self):
        """Test that clear() resets the tree to empty state."""
        trie = PrefixTree()
        trie.insert([1, 2, 3], ["a", "b", "c"], [0, 0, 0], [-1.0, -1.0, -1.0])
        assert trie.size() == 3

        trie.clear()
        assert trie.size() == 0

        # Verify we can insert after clearing
        trie.insert([4, 5], ["d", "e"], [0, 0], [-1.0, -1.0])
        assert trie.size() == 2

    def test_repr(self):
        """Test that __repr__ returns a string representation."""
        trie = PrefixTree()
        repr_empty = repr(trie)
        assert "PrefixTree:" in repr_empty
        assert "ROOT" in repr_empty

        trie.insert([1, 2], ["Hello", "World"], [0, 0], [-1.0, -1.0])
        repr_with_data = repr(trie)
        assert "PrefixTree:" in repr_with_data
        assert "ROOT" in repr_with_data
        # Should contain token information
        assert "Hello" in repr_with_data or "id=1" in repr_with_data

    # -- hardened edge cases --

    def test_insert_empty_sequence(self):
        """Inserting an empty sequence should not add any nodes."""
        trie = PrefixTree()
        trie.insert([], [], [], [])
        assert trie.size() == 0

    def test_insert_single_token(self):
        """Single token insert should create exactly one node."""
        trie = PrefixTree()
        trie.insert([42], ["tok"], [1], [-0.5])
        assert trie.size() == 1

    def test_longest_prefix_empty_input(self):
        """Empty input to longest_prefix should return empty list."""
        trie = PrefixTree()
        trie.insert([1, 2], ["a", "b"], [0, 0], [-1.0, -1.0])
        ids = trie.longest_prefix([])
        assert ids == []

    def test_longest_prefix_no_match(self):
        """Token id not in trie should return empty."""
        trie = PrefixTree()
        trie.insert([1, 2], ["a", "b"], [0, 0], [-1.0, -1.0])
        ids = trie.longest_prefix([999])
        assert ids == []

    def test_longest_text_prefix_empty_text(self):
        """Empty text should return empty results."""
        trie = PrefixTree()
        trie.insert([1], ["a"], [0], [-1.0])
        ids, strs, _, _ = trie.longest_text_prefix("")
        assert ids == []
        assert strs == []

    def test_longest_text_prefix_no_match(self):
        """Text that doesn't match any token strings should return empty."""
        trie = PrefixTree()
        trie.insert([1, 2], ["Hello", "World"], [0, 0], [-1.0, -1.0])
        ids, strs, _, _ = trie.longest_text_prefix("xyz")
        assert ids == []
        assert strs == []

    def test_longest_text_prefix_prefers_longest_matching_child(self):
        """longest_text_prefix should prefer the longest matching token string."""
        trie = PrefixTree()
        # Insert "a" first, then "ab" as separate branches from root
        trie.insert([1], ["a"], [0], [-1.0])
        trie.insert([2], ["ab"], [0], [-1.0])
        ids, strs, _, _ = trie.longest_text_prefix("ab")
        assert "".join(strs) == "ab", (
            f"Expected longest text match 'ab' but got '{''.join(strs)}'. "
            f"Prefix matching should not depend on insertion order."
        )

    def test_insert_updates_step_idx_to_minimum(self):
        """Re-inserting same path with different step_idx should keep the minimum."""
        trie = PrefixTree()
        trie.insert([1, 2], ["a", "b"], [0, 0], [-1.0, -1.0], step_idx=5)
        trie.insert([1, 2], ["a", "b"], [1, 1], [-1.0, -1.0], step_idx=2)
        node_a = trie.root.children[1]
        node_b = node_a.children[2]
        assert node_a.step_idx == 2, f"Expected step_idx=2, got {node_a.step_idx}"
        assert node_b.step_idx == 2, f"Expected step_idx=2, got {node_b.step_idx}"

    def test_moe_routermap_update_replaces_all_neg1_with_valid(self):
        """When existing moe_routermap is all -1, new valid data should replace it."""
        trie = PrefixTree()
        neg1_map = [[-1, -1], [-1, -1]]
        trie.insert([1], ["a"], [0], [-1.0], moe_routermaps=[neg1_map])
        valid_map = [[0.5, 0.3], [0.2, 0.8]]
        trie.insert([1], ["a"], [1], [-1.0], moe_routermaps=[valid_map])
        node = trie.root.children[1]
        assert node.moe_routermap == valid_map, "Valid routermap should replace all-(-1) padding"

    def test_moe_routermap_tensor_update(self):
        """Tensor moe_routermap insertion should work correctly."""
        trie = PrefixTree()
        rm = torch.randn(3, 2, 4)  # [seq_len=3, layers=2, experts=4]
        trie.insert([10, 20, 30], ["a", "b", "c"], [1, 1, 1], [-0.1, -0.2, -0.3], moe_routermaps=rm)
        assert trie.size() == 3
        # Walk the tree and verify routermaps are stored
        node = trie.root.children[10]
        assert isinstance(node.moe_routermap, torch.Tensor)


class TestIsAllNeg1:
    """Tests for the _is_all_neg1 helper function."""

    def test_none_returns_false(self):
        assert _is_all_neg1(None) is False

    def test_empty_list_returns_false(self):
        assert _is_all_neg1([]) is False

    def test_all_neg1_flat_list(self):
        assert _is_all_neg1([-1, -1, -1]) is True

    def test_mixed_values_flat_list(self):
        assert _is_all_neg1([-1, 0, -1]) is False

    def test_all_neg1_nested_list(self):
        assert _is_all_neg1([[-1, -1], [-1, -1]]) is True

    def test_mixed_nested_list(self):
        assert _is_all_neg1([[-1, 0], [-1, -1]]) is False

    def test_nested_empty_list_should_return_false(self):
        """[[]] has no actual -1 values — all() on empty iterator returns True, which is wrong."""
        assert _is_all_neg1([[]]) is False, (
            "_is_all_neg1([[]]) returned True because all(empty) is True in Python, but there are no actual -1 values."
        )

    def test_all_neg1_tensor(self):
        t = torch.full((2, 3), -1.0)
        assert _is_all_neg1(t) is True

    def test_mixed_tensor(self):
        t = torch.tensor([[-1.0, 0.0], [-1.0, -1.0]])
        assert _is_all_neg1(t) is False

    def test_empty_tensor(self):
        t = torch.tensor([])
        assert _is_all_neg1(t) is False
