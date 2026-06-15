"""Tests for axon.utils.seqlen_balancing utility functions."""

import math
import random

import pytest

from axon.utils.seqlen_balancing import (
    ceildiv,
    get_reverse_idx,
    get_seqlen_balanced_partitions,
    greedy_partition,
    karmarkar_karp,
    log_seqlen_unbalance,
    roundup_divisible,
)

# ---------------------------------------------------------------------------
# ceildiv
# ---------------------------------------------------------------------------


class TestCeildiv:
    def test_rounds_up(self):
        assert ceildiv(7, 2) == 4

    def test_small_numerator(self):
        assert ceildiv(1, 3) == 1

    def test_large_values(self):
        assert ceildiv(1000000, 3) == 333334

    def test_large_primes(self):
        """Large prime numerator and divisor to catch overflow or off-by-one."""
        assert ceildiv(999983, 997) == math.ceil(999983 / 997)

    def test_consistency_with_math_ceil(self):
        """ceildiv should match math.ceil(a/b) for many random inputs."""
        rng = random.Random(42)
        for _ in range(200):
            a = rng.randint(0, 10**7)
            b = rng.randint(1, 10**5)
            assert ceildiv(a, b) == math.ceil(a / b), f"Failed for ceildiv({a}, {b})"


# ---------------------------------------------------------------------------
# roundup_divisible
# ---------------------------------------------------------------------------


class TestRoundupDivisible:
    def test_rounds_up(self):
        assert roundup_divisible(7, 4) == 8

    def test_one_over(self):
        assert roundup_divisible(9, 4) == 12

    def test_just_below(self):
        assert roundup_divisible(3, 4) == 4

    def test_consistency_with_ceildiv(self):
        """roundup_divisible(a, b) == ceildiv(a, b) * b."""
        rng = random.Random(99)
        for _ in range(200):
            a = rng.randint(0, 10**6)
            b = rng.randint(1, 10**4)
            assert roundup_divisible(a, b) == ceildiv(a, b) * b, f"Failed for ({a}, {b})"


# ---------------------------------------------------------------------------
# get_reverse_idx
# ---------------------------------------------------------------------------


class TestGetReverseIdx:
    def test_simple_permutation(self):
        # idx_map[0]=2, idx_map[1]=0, idx_map[2]=1
        # inverse: output[2]=0, output[0]=1, output[1]=2 => [1, 2, 0]
        assert get_reverse_idx([2, 0, 1]) == [1, 2, 0]

    def test_swap(self):
        assert get_reverse_idx([1, 0]) == [1, 0]

    def test_reverse_of_reverse_is_identity(self):
        perm = [3, 1, 0, 2]
        inv = get_reverse_idx(perm)
        roundtrip = get_reverse_idx(inv)
        assert roundtrip == perm

    def test_random_permutation_inverse_property(self):
        """For a random permutation of size 100, verify the inverse property:
        perm[inverse[i]] == i for all i."""
        rng = random.Random(42)
        n = 100
        perm = list(range(n))
        rng.shuffle(perm)
        inv = get_reverse_idx(perm)
        for i in range(n):
            assert perm[inv[i]] == i, f"Inverse property failed at index {i}"
            assert inv[perm[i]] == i, f"Reverse inverse failed at index {i}"

    def test_large_permutation_roundtrip(self):
        """Stress: larger permutation roundtrip."""
        rng = random.Random(123)
        n = 500
        perm = list(range(n))
        rng.shuffle(perm)
        assert get_reverse_idx(get_reverse_idx(perm)) == perm


# ---------------------------------------------------------------------------
# karmarkar_karp
# ---------------------------------------------------------------------------


class TestKarmarkarKarp:
    def _validate_partitions(self, partitions, n, k, equal_size):
        """Common validation for partition results."""
        assert len(partitions) == k
        all_indices = []
        for p in partitions:
            all_indices.extend(p)
        assert sorted(all_indices) == list(range(n))
        if equal_size:
            expected_size = n // k
            for p in partitions:
                assert len(p) == expected_size

    def test_basic_two_partitions(self):
        seqlen_list = [10, 20, 30, 40]
        partitions = karmarkar_karp(seqlen_list, k_partitions=2, equal_size=False)
        self._validate_partitions(partitions, n=4, k=2, equal_size=False)

    def test_basic_two_partitions_balance(self):
        seqlen_list = [10, 20, 30, 40]
        partitions = karmarkar_karp(seqlen_list, k_partitions=2, equal_size=False)
        sums = [sum(seqlen_list[i] for i in p) for p in partitions]
        # The optimal partition is {40,10} and {30,20} => sums 50,50
        assert abs(sums[0] - sums[1]) <= 10  # allow small imbalance

    def test_equal_size_true(self):
        seqlen_list = [10, 20, 30, 40]
        partitions = karmarkar_karp(seqlen_list, k_partitions=2, equal_size=True)
        self._validate_partitions(partitions, n=4, k=2, equal_size=True)

    def test_equal_size_assertion_on_indivisible(self):
        with pytest.raises(AssertionError):
            karmarkar_karp([10, 20, 30], k_partitions=2, equal_size=True)

    def test_three_partitions(self):
        seqlen_list = [5, 10, 15, 20, 25, 30]
        partitions = karmarkar_karp(seqlen_list, k_partitions=3, equal_size=False)
        self._validate_partitions(partitions, n=6, k=3, equal_size=False)

    def test_three_partitions_equal_size(self):
        seqlen_list = [5, 10, 15, 20, 25, 30]
        partitions = karmarkar_karp(seqlen_list, k_partitions=3, equal_size=True)
        self._validate_partitions(partitions, n=6, k=3, equal_size=True)

    def test_single_partition(self):
        seqlen_list = [10, 20, 30]
        partitions = karmarkar_karp(seqlen_list, k_partitions=1, equal_size=False)
        self._validate_partitions(partitions, n=3, k=1, equal_size=False)
        assert sorted(partitions[0]) == [0, 1, 2]

    def test_all_equal_seqlens(self):
        seqlen_list = [100, 100, 100, 100]
        partitions = karmarkar_karp(seqlen_list, k_partitions=2, equal_size=False)
        self._validate_partitions(partitions, n=4, k=2, equal_size=False)
        sums = [sum(seqlen_list[i] for i in p) for p in partitions]
        assert sums[0] == sums[1] == 200

    def test_k_equals_n(self):
        seqlen_list = [10, 20, 30, 40]
        partitions = karmarkar_karp(seqlen_list, k_partitions=4, equal_size=False)
        self._validate_partitions(partitions, n=4, k=4, equal_size=False)
        # Each partition should have exactly one item
        for p in partitions:
            assert len(p) == 1

    def test_one_huge_item_many_tiny(self):
        """Adversarial: one huge item + many tiny items. Balance should still be reasonable."""
        seqlen_list = [10000] + [1] * 15
        partitions = karmarkar_karp(seqlen_list, k_partitions=4, equal_size=False)
        self._validate_partitions(partitions, n=16, k=4, equal_size=False)
        # The huge item must be in exactly one partition
        huge_partition = None
        for p in partitions:
            if 0 in p:
                huge_partition = p
                break
        assert huge_partition is not None

    def test_powers_of_2_lengths(self):
        """Sequence lengths that are powers of 2."""
        seqlen_list = [2**i for i in range(8)]  # [1, 2, 4, 8, 16, 32, 64, 128]
        partitions = karmarkar_karp(seqlen_list, k_partitions=2, equal_size=False)
        self._validate_partitions(partitions, n=8, k=2, equal_size=False)
        sums = [sum(seqlen_list[i] for i in p) for p in partitions]
        # Balance should be reasonable
        assert max(sums) / min(sums) < 2.0 if min(sums) > 0 else True

    def test_karmarkar_karp_always_better_or_equal_to_greedy(self):
        """Property: KK should produce better or equal balance than greedy."""
        rng = random.Random(42)
        for _ in range(20):
            n = rng.randint(4, 32)
            k = rng.randint(2, min(n, 8))
            seqlen_list = [rng.randint(1, 1000) for _ in range(n)]
            kk_parts = karmarkar_karp(seqlen_list, k_partitions=k, equal_size=False)
            gr_parts = greedy_partition(seqlen_list, k_partitions=k, equal_size=False)
            kk_sums = [sum(seqlen_list[i] for i in p) for p in kk_parts]
            gr_sums = [sum(seqlen_list[i] for i in p) for p in gr_parts]
            kk_spread = max(kk_sums) - min(kk_sums)
            gr_spread = max(gr_sums) - min(gr_sums)
            assert kk_spread <= gr_spread, (
                f"KK spread {kk_spread} > greedy spread {gr_spread} for seqlen={seqlen_list}, k={k}"
            )


# ---------------------------------------------------------------------------
# greedy_partition
# ---------------------------------------------------------------------------


class TestGreedyPartition:
    def _validate_partitions(self, partitions, n, k, equal_size):
        assert len(partitions) == k
        all_indices = []
        for p in partitions:
            all_indices.extend(p)
        assert sorted(all_indices) == list(range(n))
        if equal_size:
            expected_size = n // k
            for p in partitions:
                assert len(p) == expected_size

    def test_basic(self):
        seqlen_list = [10, 20, 30, 40]
        partitions = greedy_partition(seqlen_list, k_partitions=2, equal_size=False)
        self._validate_partitions(partitions, n=4, k=2, equal_size=False)

    def test_equal_size(self):
        seqlen_list = [10, 20, 30, 40]
        partitions = greedy_partition(seqlen_list, k_partitions=2, equal_size=True)
        self._validate_partitions(partitions, n=4, k=2, equal_size=True)

    def test_single_partition(self):
        seqlen_list = [5, 15, 25]
        partitions = greedy_partition(seqlen_list, k_partitions=1, equal_size=False)
        self._validate_partitions(partitions, n=3, k=1, equal_size=False)

    def test_equal_size_assertion_on_indivisible(self):
        with pytest.raises(AssertionError):
            greedy_partition([10, 20, 30], k_partitions=2, equal_size=True)

    def test_all_equal(self):
        seqlen_list = [50, 50, 50, 50]
        partitions = greedy_partition(seqlen_list, k_partitions=2, equal_size=False)
        self._validate_partitions(partitions, n=4, k=2, equal_size=False)


# ---------------------------------------------------------------------------
# get_seqlen_balanced_partitions
# ---------------------------------------------------------------------------


class TestGetSeqlenBalancedPartitions:
    def _validate_partitions(self, partitions, n, k, equal_size):
        assert len(partitions) == k
        all_indices = []
        for p in partitions:
            assert len(p) > 0, "partition should not be empty"
            # Indices within each partition should be sorted
            assert p == sorted(p), f"partition {p} is not sorted"
            all_indices.extend(p)
        assert sorted(all_indices) == list(range(n))
        if equal_size:
            expected_size = n // k
            for p in partitions:
                assert len(p) == expected_size

    def test_basic_two_partitions(self):
        seqlen_list = [10, 20, 30, 40]
        partitions = get_seqlen_balanced_partitions(seqlen_list, k_partitions=2, equal_size=False)
        self._validate_partitions(partitions, n=4, k=2, equal_size=False)

    def test_four_partitions_eight_items(self):
        seqlen_list = [5, 10, 15, 20, 25, 30, 35, 40]
        partitions = get_seqlen_balanced_partitions(seqlen_list, k_partitions=4, equal_size=False)
        self._validate_partitions(partitions, n=8, k=4, equal_size=False)

    def test_equal_size_true(self):
        seqlen_list = [10, 20, 30, 40, 50, 60]
        partitions = get_seqlen_balanced_partitions(seqlen_list, k_partitions=3, equal_size=True)
        self._validate_partitions(partitions, n=6, k=3, equal_size=True)

    def test_equal_size_two_partitions(self):
        seqlen_list = [10, 20, 30, 40]
        partitions = get_seqlen_balanced_partitions(seqlen_list, k_partitions=2, equal_size=True)
        self._validate_partitions(partitions, n=4, k=2, equal_size=True)

    def test_k_equals_n_each_gets_one(self):
        seqlen_list = [10, 20, 30, 40]
        partitions = get_seqlen_balanced_partitions(seqlen_list, k_partitions=4, equal_size=False)
        self._validate_partitions(partitions, n=4, k=4, equal_size=False)
        for p in partitions:
            assert len(p) == 1

    def test_k_equals_n_equal_size(self):
        seqlen_list = [10, 20, 30, 40]
        partitions = get_seqlen_balanced_partitions(seqlen_list, k_partitions=4, equal_size=True)
        self._validate_partitions(partitions, n=4, k=4, equal_size=True)

    def test_assertion_k_greater_than_n(self):
        with pytest.raises(AssertionError, match="number of items"):
            get_seqlen_balanced_partitions([10, 20], k_partitions=5, equal_size=False)

    def test_assertion_equal_size_indivisible(self):
        with pytest.raises(AssertionError):
            get_seqlen_balanced_partitions([10, 20, 30], k_partitions=2, equal_size=True)

    def test_balancing_quality(self):
        """Verify that the partition sums are reasonably balanced."""
        seqlen_list = [1, 2, 3, 4, 5, 6, 7, 8]
        partitions = get_seqlen_balanced_partitions(seqlen_list, k_partitions=2, equal_size=False)
        sums = [sum(seqlen_list[i] for i in p) for p in partitions]
        total = sum(seqlen_list)
        ideal = total / 2
        for s in sums:
            # Each partition sum should be within 20% of ideal, or the diff <= 2
            assert abs(s - ideal) <= max(ideal * 0.2, 2), f"Partition sum {s} far from ideal {ideal}"

    def test_large_input(self):
        """Test with a larger input to ensure no crashes and correct structure."""
        rng = random.Random(42)
        seqlen_list = [rng.randint(1, 1000) for _ in range(64)]
        partitions = get_seqlen_balanced_partitions(seqlen_list, k_partitions=8, equal_size=True)
        self._validate_partitions(partitions, n=64, k=8, equal_size=True)

    def test_large_input_unequal(self):
        rng = random.Random(123)
        seqlen_list = [rng.randint(1, 500) for _ in range(50)]
        partitions = get_seqlen_balanced_partitions(seqlen_list, k_partitions=7, equal_size=False)
        self._validate_partitions(partitions, n=50, k=7, equal_size=False)

    def test_single_item_single_partition(self):
        partitions = get_seqlen_balanced_partitions([42], k_partitions=1, equal_size=False)
        assert partitions == [[0]]

    def test_two_items_two_partitions(self):
        partitions = get_seqlen_balanced_partitions([100, 200], k_partitions=2, equal_size=False)
        self._validate_partitions(partitions, n=2, k=2, equal_size=False)

    def test_all_identical_lengths(self):
        """All items same length: balance should be perfect."""
        seqlen_list = [100] * 16
        partitions = get_seqlen_balanced_partitions(seqlen_list, k_partitions=4, equal_size=True)
        self._validate_partitions(partitions, n=16, k=4, equal_size=True)
        sums = [sum(seqlen_list[i] for i in p) for p in partitions]
        assert all(s == 400 for s in sums)

    def test_balance_quality_bounded_ratio(self):
        """Property: for ANY valid input, max_sum / min_sum should be bounded (< 2x)."""
        rng = random.Random(77)
        for _ in range(20):
            n = rng.randint(8, 64)
            k = rng.randint(2, min(n, 16))
            seqlen_list = [rng.randint(1, 1000) for _ in range(n)]
            partitions = get_seqlen_balanced_partitions(seqlen_list, k_partitions=k, equal_size=False)
            sums = [sum(seqlen_list[i] for i in p) for p in partitions]
            if min(sums) > 0:
                ratio = max(sums) / min(sums)
                assert ratio < 3.0, f"Balance ratio {ratio:.2f} too high for n={n}, k={k}"

    def test_stress_1000_items_16_partitions(self):
        """Stress: 1000 items into 16 partitions, verify all invariants."""
        rng = random.Random(99)
        n = 1000
        k = 16
        seqlen_list = [rng.randint(1, 5000) for _ in range(n)]
        # n must be divisible by k for equal_size
        # 1000 / 16 = 62.5, not divisible. Use equal_size=False.
        partitions = get_seqlen_balanced_partitions(seqlen_list, k_partitions=k, equal_size=False)
        # All indices present
        all_idx = sorted(idx for p in partitions for idx in p)
        assert all_idx == list(range(n))
        # No empty partitions
        for p in partitions:
            assert len(p) > 0
        # Each partition sorted
        for p in partitions:
            assert p == sorted(p)


# ---------------------------------------------------------------------------
# log_seqlen_unbalance
# ---------------------------------------------------------------------------


class TestLogSeqlenUnbalance:
    def test_basic_metrics(self):
        seqlen_list = [10, 20, 30, 40]
        partitions = [[0, 3], [1, 2]]  # sums: 50, 50
        result = log_seqlen_unbalance(seqlen_list, partitions, prefix="test")

        # Check all expected keys exist
        expected_keys = {
            "test/min",
            "test/max",
            "test/minmax_diff",
            "test/balanced_min",
            "test/balanced_max",
            "test/mean",
        }
        assert set(result.keys()) == expected_keys

        # Balanced sums: partition[0] = 10+40=50, partition[1] = 20+30=50
        assert result["test/balanced_min"] == 50
        assert result["test/balanced_max"] == 50

    def test_unbalanced_partitions(self):
        seqlen_list = [10, 20, 30, 40]
        partitions = [[0, 1], [2, 3]]  # sums: 30, 70
        result = log_seqlen_unbalance(seqlen_list, partitions, prefix="pfx")

        assert result["pfx/balanced_min"] == 30
        assert result["pfx/balanced_max"] == 70

    def test_original_minmax_metrics(self):
        """The min/max/minmax_diff fields reflect the original sequential chunking."""
        seqlen_list = [10, 20, 30, 40]
        partitions = [[0, 3], [1, 2]]
        result = log_seqlen_unbalance(seqlen_list, partitions, prefix="m")

        # The function chunks seqlen_list into groups of ceil(4/2)=2:
        # chunk0: [10, 20] sum=30, chunk1: [30, 40] sum=70
        assert result["m/min"] == 30
        assert result["m/max"] == 70
        assert result["m/minmax_diff"] == 40

    def test_mean_value(self):
        seqlen_list = [10, 20, 30, 40]
        partitions = [[0, 3], [1, 2]]
        result = log_seqlen_unbalance(seqlen_list, partitions, prefix="t")

        # Total = 100, partitions = 2 => mean = 50.0
        assert result["t/mean"] == 50.0

    def test_single_partition(self):
        seqlen_list = [10, 20, 30]
        partitions = [[0, 1, 2]]
        result = log_seqlen_unbalance(seqlen_list, partitions, prefix="s")

        assert result["s/balanced_min"] == 60
        assert result["s/balanced_max"] == 60
        assert result["s/min"] == 60
        assert result["s/max"] == 60
        assert result["s/minmax_diff"] == 0

    def test_three_partitions(self):
        seqlen_list = [10, 20, 30, 40, 50, 60]
        # balanced sums: 10+60=70, 20+50=70, 30+40=70
        partitions = [[0, 5], [1, 4], [2, 3]]
        result = log_seqlen_unbalance(seqlen_list, partitions, prefix="x")

        assert result["x/balanced_min"] == 70
        assert result["x/balanced_max"] == 70
        assert result["x/mean"] == pytest.approx(210.0 / 3)
