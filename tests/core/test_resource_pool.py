"""Tests for axon.core.resource_pool module."""

from axon.core.resource_pool import ResourcePool


# ---------------------------------------------------------------------------
# ResourcePool.__init__ defaults and custom args
# ---------------------------------------------------------------------------
class TestResourcePoolInit:
    def test_with_process_on_nodes(self):
        pool = ResourcePool(process_on_nodes=[2, 4, 2])
        assert pool.world_size == 8

    def test_custom_name_prefix(self):
        pool = ResourcePool(name_prefix="my_pool")
        assert pool.name_prefix == "my_pool"

    def test_auto_generated_name_prefix_length(self):
        pool = ResourcePool()
        assert len(pool.name_prefix) == 8

    def test_auto_generated_name_prefix_unique(self):
        pool1 = ResourcePool()
        pool2 = ResourcePool()
        assert pool1.name_prefix != pool2.name_prefix

    def test_store_is_same_reference_as_input(self):
        """Mutating the input list mutates the pool."""
        nodes = [2, 4]
        pool = ResourcePool(process_on_nodes=nodes)
        nodes.append(8)
        assert pool.world_size == 14  # aliased, not copied

    def test_negative_process_counts_not_rejected(self):
        """No validation on negative values - world_size can go negative."""
        pool = ResourcePool(process_on_nodes=[-1, 5])
        assert pool.world_size == 4


# ---------------------------------------------------------------------------
# ResourcePool.world_size
# ---------------------------------------------------------------------------
class TestResourcePoolWorldSize:
    def test_multiple_nodes(self):
        pool = ResourcePool(process_on_nodes=[2, 4, 2])
        assert pool.world_size == 8

    def test_heterogeneous_nodes(self):
        pool = ResourcePool(process_on_nodes=[1, 8, 2, 4])
        assert pool.world_size == 15


# ---------------------------------------------------------------------------
# ResourcePool.store
# ---------------------------------------------------------------------------
class TestResourcePoolStore:
    def test_store_is_same_object(self):
        nodes = [1, 2, 3]
        pool = ResourcePool(process_on_nodes=nodes)
        assert pool.store is nodes


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------
class TestResourcePoolEdgeCases:
    def test_large_numbers(self):
        pool = ResourcePool(process_on_nodes=[1000, 2000, 3000])
        assert pool.world_size == 6000

    def test_all_zeros(self):
        pool = ResourcePool(process_on_nodes=[0, 0, 0])
        assert pool.world_size == 0
        assert pool.store == [0, 0, 0]

    def test_mixed_zero_and_nonzero(self):
        pool = ResourcePool(process_on_nodes=[0, 4, 0, 2])
        assert pool.world_size == 6

    def test_name_prefix_empty_string_uses_uuid(self):
        pool = ResourcePool(name_prefix="")
        assert len(pool.name_prefix) == 8

    def test_name_prefix_none_uses_uuid(self):
        pool = ResourcePool(name_prefix=None)
        assert len(pool.name_prefix) == 8

    def test_very_many_nodes(self):
        pool = ResourcePool(process_on_nodes=[1] * 10000)
        assert pool.world_size == 10000

    def test_empty_string_name_prefix_treated_as_falsy(self):
        pool = ResourcePool(name_prefix="")
        assert len(pool.name_prefix) == 8  # generates UUID

    def test_two_pools_different_uuid_prefixes(self):
        pools = [ResourcePool() for _ in range(100)]
        prefixes = {p.name_prefix for p in pools}
        assert len(prefixes) == 100  # all unique
