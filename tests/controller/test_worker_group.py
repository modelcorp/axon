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
Unit tests for WorkerGroup and ResourcePool classes.
"""

import unittest
from unittest.mock import MagicMock, patch

from axon.controller.decorator import MAGIC_ATTR, Dispatch, Execute
from axon.controller.worker_group import WorkerGroup
from axon.core import ResourcePool


class TestWorkerGroup(unittest.TestCase):
    """Tests for the base WorkerGroup class."""

    def test_init(self):
        """Test WorkerGroup initialization."""
        resource_pool = ResourcePool(process_on_nodes=[2, 2])
        wg = WorkerGroup(resource_pool=resource_pool)

        self.assertEqual(wg.resource_pool, resource_pool)
        self.assertEqual(wg._workers, [])
        self.assertEqual(wg._dispatch_info, {})
        self.assertEqual(wg._collect_info, {})

    def test_world_size(self):
        """Test world_size property."""
        wg = WorkerGroup(resource_pool=None)
        self.assertEqual(wg.world_size, 0)

        wg._workers = [MagicMock() for _ in range(5)]
        self.assertEqual(wg.world_size, 5)

    def test_is_worker_alive_not_implemented(self):
        """Test that _is_worker_alive raises NotImplementedError in base class."""
        wg = WorkerGroup(resource_pool=None)
        with self.assertRaises(NotImplementedError):
            wg._is_worker_alive(MagicMock())

    def test_bind_worker_method_binds_decorated_methods(self):
        """Test _bind_worker_method binds methods with MAGIC_ATTR."""
        wg = WorkerGroup(resource_pool=None)
        wg.execute_all = MagicMock()

        class MockWorkerClass:
            def regular_method(self):
                pass

            def decorated_method(self):
                pass

        setattr(
            MockWorkerClass.decorated_method,
            MAGIC_ATTR,
            {
                "dispatch_mode": Dispatch.ONE_TO_ALL,
                "execute_mode": Execute.ALL,
                "blocking": True,
                "disable_collective": False,
            },
        )

        def mock_func_generator(wg, method_name, **kwargs):
            return lambda: f"called_{method_name}"

        method_names = wg._bind_worker_method(MockWorkerClass, mock_func_generator)

        self.assertIn("decorated_method", method_names)
        self.assertNotIn("regular_method", method_names)
        self.assertEqual(wg.decorated_method(), "called_decorated_method")

    def test_bind_worker_method_with_rank_zero_execute(self):
        """Test _bind_worker_method correctly routes to execute_rank_zero."""
        wg = WorkerGroup(resource_pool=None)
        wg.execute_all = MagicMock()
        wg.execute_rank_zero = MagicMock()

        class MockWorkerClass:
            def rank_zero_method(self):
                pass

        setattr(
            MockWorkerClass.rank_zero_method,
            MAGIC_ATTR,
            {
                "dispatch_mode": Dispatch.RANK_ZERO,
                "execute_mode": Execute.RANK_ZERO,
                "blocking": True,
                "disable_collective": False,
            },
        )

        captured = {}

        def mock_func_generator(wg, method_name, execute_fn=None, **kwargs):
            captured["execute_fn"] = execute_fn
            return lambda: "result"

        wg._bind_worker_method(MockWorkerClass, mock_func_generator)
        self.assertEqual(captured["execute_fn"], wg.execute_rank_zero)

    def test_bind_worker_method_with_custom_dispatch(self):
        """Test _bind_worker_method with custom dispatch/collect functions."""
        wg = WorkerGroup(resource_pool=None)
        wg.execute_all = MagicMock()

        custom_dispatch = lambda wg, *args, **kwargs: (args, kwargs)
        custom_collect = lambda wg, output: output

        class MockWorkerClass:
            def custom_method(self):
                pass

        setattr(
            MockWorkerClass.custom_method,
            MAGIC_ATTR,
            {
                "dispatch_mode": {"dispatch_fn": custom_dispatch, "collect_fn": custom_collect},
                "execute_mode": Execute.ALL,
                "blocking": True,
                "disable_collective": False,
            },
        )

        captured = {}

        def mock_func_generator(wg, method_name, dispatch_fn=None, collect_fn=None, **kwargs):
            captured["dispatch_fn"] = dispatch_fn
            captured["collect_fn"] = collect_fn
            return lambda: "result"

        wg._bind_worker_method(MockWorkerClass, mock_func_generator)

        self.assertEqual(captured["dispatch_fn"], custom_dispatch)
        self.assertEqual(captured["collect_fn"], custom_collect)

    def test_bind_worker_method_skips_properties(self):
        """Test _bind_worker_method skips class properties."""
        wg = WorkerGroup(resource_pool=None)
        wg.execute_all = MagicMock()

        class MockWorkerClass:
            @property
            def my_property(self):
                return "value"

            def decorated_method(self):
                pass

        setattr(
            MockWorkerClass.decorated_method,
            MAGIC_ATTR,
            {
                "dispatch_mode": Dispatch.ALL_TO_ALL,
                "execute_mode": Execute.ALL,
                "blocking": True,
                "disable_collective": False,
            },
        )

        def mock_func_generator(wg, method_name, **kwargs):
            return lambda: method_name

        method_names = wg._bind_worker_method(MockWorkerClass, mock_func_generator)

        self.assertNotIn("my_property", method_names)
        self.assertIn("decorated_method", method_names)

    def test_bind_worker_method_raises_on_missing_execute_fn(self):
        """Test _bind_worker_method raises when execute function doesn't exist."""
        wg = WorkerGroup(resource_pool=None)
        # Don't set execute_all

        class MockWorkerClass:
            def decorated_method(self):
                pass

        setattr(
            MockWorkerClass.decorated_method,
            MAGIC_ATTR,
            {
                "dispatch_mode": Dispatch.ALL_TO_ALL,
                "execute_mode": Execute.ALL,
                "blocking": True,
                "disable_collective": False,
            },
        )

        def mock_func_generator(wg, method_name, **kwargs):
            return lambda: method_name

        with self.assertRaises(AttributeError):
            wg._bind_worker_method(MockWorkerClass, mock_func_generator)

    @patch("axon.controller.worker_group.threading.Thread")
    def test_start_worker_aliveness_check_creates_thread(self, mock_thread):
        """Test that aliveness check creates a background thread."""
        wg = WorkerGroup(resource_pool=None)
        wg._workers = [MagicMock()]
        wg._is_worker_alive = MagicMock(return_value=True)

        mock_thread_instance = MagicMock()
        mock_thread.return_value = mock_thread_instance

        wg.start_worker_aliveness_check(every_n_seconds=5)

        mock_thread.assert_called_once()
        mock_thread_instance.start.assert_called_once()


class TestResourcePool(unittest.TestCase):
    """Tests for ResourcePool."""

    def test_world_size(self):
        """Test ResourcePool world_size calculation."""
        self.assertEqual(ResourcePool(process_on_nodes=[2, 4, 2]).world_size, 8)
        self.assertEqual(ResourcePool(process_on_nodes=[]).world_size, 0)
        self.assertEqual(ResourcePool(process_on_nodes=None).world_size, 0)
        self.assertEqual(ResourcePool(process_on_nodes=[8, 0, 4]).world_size, 12)

    def test_name_prefix(self):
        """Test ResourcePool name prefix generation."""
        pool1 = ResourcePool(process_on_nodes=[2], name_prefix="custom")
        self.assertEqual(pool1.name_prefix, "custom")

        # Auto-generated prefixes should be unique
        pool2 = ResourcePool(process_on_nodes=[2])
        pool3 = ResourcePool(process_on_nodes=[2])
        self.assertNotEqual(pool2.name_prefix, pool3.name_prefix)

    def test_max_colocate_count(self):
        """Test ResourcePool max_colocate_count."""
        self.assertEqual(ResourcePool(process_on_nodes=[2]).max_colocate_count, 10)
        self.assertEqual(ResourcePool(process_on_nodes=[2], max_colocate_count=5).max_colocate_count, 5)


class TestWorkerGroupEdgeCases(unittest.TestCase):
    """Additional edge case tests for WorkerGroup."""

    def test_bind_worker_method_with_multiple_decorated_methods(self):
        """Test _bind_worker_method binds all decorated methods."""
        wg = WorkerGroup(resource_pool=None)
        wg.execute_all = MagicMock()
        wg.execute_rank_zero = MagicMock()

        class MockWorkerClass:
            def method_a(self):
                pass

            def method_b(self):
                pass

            def method_c(self):
                pass

        for method_name, dispatch_mode, execute_mode in [
            ("method_a", Dispatch.ONE_TO_ALL, Execute.ALL),
            ("method_b", Dispatch.ALL_TO_ALL, Execute.ALL),
            ("method_c", Dispatch.RANK_ZERO, Execute.RANK_ZERO),
        ]:
            setattr(
                getattr(MockWorkerClass, method_name),
                MAGIC_ATTR,
                {
                    "dispatch_mode": dispatch_mode,
                    "execute_mode": execute_mode,
                    "blocking": True,
                    "disable_collective": False,
                },
            )

        def mock_func_generator(wg, method_name, **kwargs):
            return lambda: f"called_{method_name}"

        method_names = wg._bind_worker_method(MockWorkerClass, mock_func_generator)

        self.assertEqual(sorted(method_names), ["method_a", "method_b", "method_c"])
        self.assertEqual(wg.method_a(), "called_method_a")
        self.assertEqual(wg.method_b(), "called_method_b")
        self.assertEqual(wg.method_c(), "called_method_c")

    def test_bind_worker_method_returns_list_of_names(self):
        """Test _bind_worker_method returns the list of bound method names."""
        wg = WorkerGroup(resource_pool=None)
        wg.execute_all = MagicMock()

        class EmptyWorkerClass:
            pass

        def mock_func_generator(wg, method_name, **kwargs):
            return lambda: None

        names = wg._bind_worker_method(EmptyWorkerClass, mock_func_generator)
        self.assertEqual(names, [])

    def test_dispatch_info_and_collect_info_initially_empty(self):
        """Test that dispatch_info and collect_info are empty dicts on init."""
        wg = WorkerGroup(resource_pool=None)
        self.assertEqual(wg._dispatch_info, {})
        self.assertEqual(wg._collect_info, {})

    def test_master_addr_and_port_initially_none(self):
        """Test that master_addr and master_port are None on init."""
        wg = WorkerGroup(resource_pool=None)
        self.assertIsNone(wg._master_addr)
        self.assertIsNone(wg._master_port)

    def test_checker_thread_initially_none(self):
        """Test that checker thread is None on init."""
        wg = WorkerGroup(resource_pool=None)
        self.assertIsNone(wg._checker_thread)

    def test_bind_worker_method_skips_dunder_methods(self):
        """Test that __dunder__ methods without MAGIC_ATTR are skipped."""
        wg = WorkerGroup(resource_pool=None)
        wg.execute_all = MagicMock()

        class MockWorkerClass:
            def __init__(self):
                pass

            def __repr__(self):
                return "MockWorker"

            def decorated(self):
                pass

        setattr(
            MockWorkerClass.decorated,
            MAGIC_ATTR,
            {
                "dispatch_mode": Dispatch.ALL_TO_ALL,
                "execute_mode": Execute.ALL,
                "blocking": True,
                "disable_collective": False,
            },
        )

        def mock_func_generator(wg, method_name, **kwargs):
            return lambda: method_name

        method_names = wg._bind_worker_method(MockWorkerClass, mock_func_generator)
        self.assertIn("decorated", method_names)
        self.assertNotIn("__init__", method_names)
        self.assertNotIn("__repr__", method_names)


class TestResourcePoolEdgeCases(unittest.TestCase):
    """Additional edge case tests for ResourcePool."""

    def test_single_node(self):
        """Test ResourcePool with a single node."""
        pool = ResourcePool(process_on_nodes=[8])
        self.assertEqual(pool.world_size, 8)

    def test_many_nodes(self):
        """Test ResourcePool with many nodes."""
        pool = ResourcePool(process_on_nodes=[4, 4, 4, 4])
        self.assertEqual(pool.world_size, 16)

    def test_single_process_per_node(self):
        """Test ResourcePool with 1 process per node."""
        pool = ResourcePool(process_on_nodes=[1, 1, 1])
        self.assertEqual(pool.world_size, 3)


if __name__ == "__main__":
    unittest.main()
