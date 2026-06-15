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
Unit tests for RayWorkerGroup class.
"""

import unittest
from unittest.mock import MagicMock, patch

from axon.core import ResourcePool


class TestRayWorkerGroup(unittest.TestCase):
    """Unit tests for RayWorkerGroup."""

    @patch("axon.controller.ray.worker_group.ray")
    def test_init_with_existing_workers(self, mock_ray):
        """Test initialization with existing worker handles."""
        from axon.controller.ray.worker_group import RayWorkerGroup

        mock_workers = [MagicMock() for _ in range(4)]
        worker_names = ["w0", "w1", "w2", "w3"]

        wg = RayWorkerGroup(
            resource_pool=None,
            worker_names=worker_names,
            worker_handles=mock_workers,
        )

        self.assertEqual(wg._world_size, 4)
        self.assertEqual(wg._workers, mock_workers)
        self.assertEqual(wg._worker_names, worker_names)

    @patch("axon.controller.ray.worker_group.ray")
    def test_init_fetches_actors_by_name_when_handles_not_provided(self, mock_ray):
        """Test that actors are fetched by name when handles not provided."""
        from axon.controller.ray.worker_group import RayWorkerGroup

        mock_actor = MagicMock()
        mock_ray.get_actor.return_value = mock_actor

        wg = RayWorkerGroup(
            resource_pool=None,
            worker_names=["named_worker"],
            worker_handles=None,
        )

        mock_ray.get_actor.assert_called_once_with(name="named_worker")
        self.assertEqual(wg._workers, [mock_actor])

    @patch("axon.controller.ray.worker_group.ray")
    def test_execute_broadcasts_args_to_all_workers(self, mock_ray):
        """Test execute broadcasts same args to all workers."""
        from axon.controller.ray.worker_group import RayWorkerGroup

        mock_workers = [MagicMock() for _ in range(3)]
        for w in mock_workers:
            w.method = MagicMock()
            w.method.remote.return_value = MagicMock()

        wg = RayWorkerGroup(
            resource_pool=None,
            worker_names=["w0", "w1", "w2"],
            worker_handles=mock_workers,
        )

        wg.execute("method", "arg1", key="val")

        for w in mock_workers:
            w.method.remote.assert_called_once_with("arg1", key="val")

    @patch("axon.controller.ray.worker_group.ray")
    def test_execute_on_specific_ranks(self, mock_ray):
        """Test execute only calls specified ranks."""
        from axon.controller.ray.worker_group import RayWorkerGroup

        mock_workers = [MagicMock() for _ in range(4)]
        for w in mock_workers:
            w.method = MagicMock()
            w.method.remote.return_value = MagicMock()

        wg = RayWorkerGroup(
            resource_pool=None,
            worker_names=["w0", "w1", "w2", "w3"],
            worker_handles=mock_workers,
        )

        wg.execute("method", ranks=[0, 2])

        mock_workers[0].method.remote.assert_called_once()
        mock_workers[1].method.remote.assert_not_called()
        mock_workers[2].method.remote.assert_called_once()
        mock_workers[3].method.remote.assert_not_called()

    @patch("axon.controller.ray.worker_group.ray")
    def test_execute_single_rank_unwraps_result(self, mock_ray):
        """Test execute on single rank returns unwrapped result, not list."""
        from axon.controller.ray.worker_group import RayWorkerGroup

        mock_workers = [MagicMock() for _ in range(3)]
        mock_ref = MagicMock()
        mock_workers[1].method = MagicMock()
        mock_workers[1].method.remote.return_value = mock_ref

        wg = RayWorkerGroup(
            resource_pool=None,
            worker_names=["w0", "w1", "w2"],
            worker_handles=mock_workers,
        )

        result = wg.execute("method", ranks=[1])
        self.assertEqual(result, mock_ref)

    @patch("axon.controller.ray.worker_group.ray")
    def test_execute_blocking_calls_ray_get(self, mock_ray):
        """Test execute with blocking=True waits for results."""
        from axon.controller.ray.worker_group import RayWorkerGroup

        mock_workers = [MagicMock() for _ in range(2)]
        for w in mock_workers:
            w.method = MagicMock()
            w.method.remote.return_value = MagicMock()

        mock_ray.get.return_value = ["result_0", "result_1"]

        wg = RayWorkerGroup(
            resource_pool=None,
            worker_names=["w0", "w1"],
            worker_handles=mock_workers,
        )

        result = wg.execute("method", blocking=True)

        mock_ray.get.assert_called()
        self.assertEqual(result, ["result_0", "result_1"])

    @patch("axon.controller.ray.worker_group.ray")
    def test_execute_distributes_list_args_to_workers(self, mock_ray):
        """Test execute distributes list args when lengths match worker count."""
        from axon.controller.ray.worker_group import RayWorkerGroup

        mock_workers = [MagicMock() for _ in range(3)]
        for w in mock_workers:
            w.process = MagicMock()
            w.process.remote.return_value = MagicMock()

        wg = RayWorkerGroup(
            resource_pool=None,
            worker_names=["w0", "w1", "w2"],
            worker_handles=mock_workers,
        )

        wg.execute("process", ["d0", "d1", "d2"])

        mock_workers[0].process.remote.assert_called_once_with("d0")
        mock_workers[1].process.remote.assert_called_once_with("d1")
        mock_workers[2].process.remote.assert_called_once_with("d2")

    @patch("axon.controller.ray.worker_group.ray")
    def test_execute_distributes_list_kwargs(self, mock_ray):
        """Test execute distributes list kwargs when lengths match."""
        from axon.controller.ray.worker_group import RayWorkerGroup

        mock_workers = [MagicMock() for _ in range(2)]
        for w in mock_workers:
            w.process = MagicMock()
            w.process.remote.return_value = MagicMock()

        wg = RayWorkerGroup(
            resource_pool=None,
            worker_names=["w0", "w1"],
            worker_handles=mock_workers,
        )

        wg.execute("process", data=["d0", "d1"], cfg=["c0", "c1"])

        mock_workers[0].process.remote.assert_called_once_with(data="d0", cfg="c0")
        mock_workers[1].process.remote.assert_called_once_with(data="d1", cfg="c1")

    @patch("axon.controller.ray.worker_group.ray")
    def test_execute_broadcasts_when_list_length_mismatches(self, mock_ray):
        """Test execute broadcasts list args when length doesn't match worker count."""
        from axon.controller.ray.worker_group import RayWorkerGroup

        mock_workers = [MagicMock() for _ in range(3)]
        for w in mock_workers:
            w.process = MagicMock()
            w.process.remote.return_value = MagicMock()

        wg = RayWorkerGroup(
            resource_pool=None,
            worker_names=["w0", "w1", "w2"],
            worker_handles=mock_workers,
        )

        # List has 2 elements but 3 workers - broadcasts entire list
        wg.execute("process", ["d0", "d1"])

        for w in mock_workers:
            w.process.remote.assert_called_once_with(["d0", "d1"])

    @patch("axon.controller.ray.worker_group.ray")
    def test_execute_rank_zero_only_calls_rank_0(self, mock_ray):
        """Test execute_rank_zero only executes on rank 0."""
        from axon.controller.ray.worker_group import RayWorkerGroup

        mock_workers = [MagicMock() for _ in range(3)]
        for w in mock_workers:
            w.method = MagicMock()
            w.method.remote.return_value = MagicMock()

        wg = RayWorkerGroup(
            resource_pool=None,
            worker_names=["w0", "w1", "w2"],
            worker_handles=mock_workers,
        )

        wg.execute_rank_zero("method", "arg")

        mock_workers[0].method.remote.assert_called_once_with("arg")
        mock_workers[1].method.remote.assert_not_called()
        mock_workers[2].method.remote.assert_not_called()

    @patch("axon.controller.ray.worker_group.get_actor")
    @patch("axon.controller.ray.worker_group.ray")
    def test_is_worker_alive(self, mock_ray, mock_get_actor):
        """Test _is_worker_alive checks actor state."""
        from axon.controller.ray.worker_group import RayWorkerGroup

        mock_worker = MagicMock()
        mock_worker._actor_id.hex.return_value = "abc123"

        wg = RayWorkerGroup(
            resource_pool=None,
            worker_names=["w0"],
            worker_handles=[mock_worker],
        )

        mock_get_actor.return_value = {"state": "ALIVE"}
        self.assertTrue(wg._is_worker_alive(mock_worker))

        mock_get_actor.return_value = {"state": "DEAD"}
        self.assertFalse(wg._is_worker_alive(mock_worker))

        mock_get_actor.return_value = None
        self.assertFalse(wg._is_worker_alive(mock_worker))

    @patch("axon.controller.ray.worker_group.ray")
    def test_collective_backend_selection(self, mock_ray):
        """Test collective backend is NCCL for CUDA, Gloo otherwise."""
        from axon.controller.ray.worker_group import RayWorkerGroup

        wg_cuda = RayWorkerGroup(
            resource_pool=None,
            worker_names=["w0"],
            worker_handles=[MagicMock()],
            device_name="cuda",
        )
        self.assertEqual(wg_cuda._ray_collective_backend, "nccl")

        wg_cpu = RayWorkerGroup(
            resource_pool=None,
            worker_names=["w0"],
            worker_handles=[MagicMock()],
            device_name="cpu",
        )
        self.assertEqual(wg_cpu._ray_collective_backend, "gloo")


class TestRayActorWithInitArgs(unittest.TestCase):
    """Tests for RayActorWithInitArgs."""

    def test_update_options_merges(self):
        """Test update_options merges and overwrites options."""
        from axon.controller.ray.class_init import RayActorWithInitArgs

        ray_cls = RayActorWithInitArgs(cls=MagicMock())

        ray_cls.update_options({"num_gpus": 1, "name": "test"})
        ray_cls.update_options({"num_gpus": 2, "num_cpus": 4})

        self.assertEqual(ray_cls._options, {"num_gpus": 2, "name": "test", "num_cpus": 4})

    @patch("axon.controller.ray.class_init.PlacementGroupSchedulingStrategy")
    def test_call_creates_actor_with_resources(self, mock_pg_strategy):
        """Test __call__ creates actor with correct resource options."""
        from axon.controller.ray.class_init import RayActorWithInitArgs

        mock_cls = MagicMock()
        mock_actor = MagicMock()
        mock_cls.options.return_value.remote.return_value = mock_actor

        ray_cls = RayActorWithInitArgs(cls=mock_cls, model_path="/path")
        ray_cls.update_options({"name": "test_actor"})

        result = ray_cls(
            placement_group=MagicMock(),
            placement_group_bundle_idx=0,
            resource_dict={"CPU": 2, "GPU": 1, "NPU": 4},
        )

        call_kwargs = mock_cls.options.call_args[1]
        self.assertEqual(call_kwargs["num_gpus"], 1)
        self.assertEqual(call_kwargs["num_cpus"], 2)
        self.assertEqual(call_kwargs["resources"], {"NPU": 4})
        self.assertEqual(call_kwargs["name"], "test_actor")
        self.assertEqual(call_kwargs["max_concurrency"], 2048)
        mock_cls.options.return_value.remote.assert_called_once_with(model_path="/path")
        self.assertEqual(result, mock_actor)


class TestInitWorkerGroup(unittest.TestCase):
    """Tests for init_worker_group function."""

    @patch("axon.controller.ray.worker_group.RayWorkerGroup")
    def test_single_class_returns_worker_group(self, mock_rwg_class):
        """Test single class dict returns RayWorkerGroup directly."""
        from axon.controller.ray.worker_group import init_worker_group

        mock_wg = MagicMock()
        mock_rwg_class.return_value = mock_wg

        resource_pool = ResourcePool(process_on_nodes=[2])
        mock_ray_cls = MagicMock()

        result = init_worker_group({"actor": mock_ray_cls}, resource_pool)

        self.assertEqual(result, {"actor": mock_wg})
        mock_rwg_class.assert_called_once_with(
            resource_pool=resource_pool,
            ray_cls_with_init=mock_ray_cls,
        )

    @patch("axon.controller.ray.worker_group.RoleProxy")
    @patch("axon.controller.ray.worker_group.fuse_worker_cls")
    @patch("axon.controller.ray.worker_group.RayWorkerGroup")
    def test_multiple_classes_fuses_and_returns_proxies(self, mock_rwg, mock_fuse, mock_proxy):
        """Test multiple classes creates fused worker and role proxies."""
        from axon.controller.ray.worker_group import init_worker_group

        mock_rwg.return_value = MagicMock()
        mock_fuse.return_value = MagicMock()

        result = init_worker_group(
            {"actor": MagicMock(), "critic": MagicMock()},
            ResourcePool(process_on_nodes=[2]),
        )

        mock_fuse.assert_called_once()
        self.assertEqual(mock_proxy.call_count, 2)
        self.assertIn("actor", result)
        self.assertIn("critic", result)


class TestMakeWgView(unittest.TestCase):
    """Tests for _make_wg_view helper."""

    @patch("axon.controller.ray.worker_group.ray")
    def test_view_overrides_specified_attributes(self, mock_ray):
        """Test view overrides only specified attributes, delegates rest to base."""
        from axon.controller.ray.worker_group import RayWorkerGroup, _make_wg_view

        mock_workers = [MagicMock()]
        wg = RayWorkerGroup(
            resource_pool=None,
            worker_names=["w0"],
            worker_handles=mock_workers,
        )
        wg._ray_collective_initialized = True

        view = _make_wg_view(wg, _ray_collective_initialized=False)

        # Override takes effect on view
        self.assertFalse(view._ray_collective_initialized)
        # Original unchanged
        self.assertTrue(wg._ray_collective_initialized)
        # Non-overridden attributes delegate to base
        self.assertEqual(view._workers, mock_workers)

    @patch("axon.controller.ray.worker_group.ray")
    def test_view_setattr_updates_correct_location(self, mock_ray):
        """Test setattr on view updates override or base appropriately."""
        from axon.controller.ray.worker_group import RayWorkerGroup, _make_wg_view

        mock_workers = [MagicMock()]
        wg = RayWorkerGroup(
            resource_pool=None,
            worker_names=["w0"],
            worker_handles=mock_workers,
        )

        view = _make_wg_view(wg, _ray_collective_initialized=False)

        # Setting overridden attr updates view only
        view._ray_collective_initialized = True
        self.assertTrue(view._ray_collective_initialized)

        # Setting non-overridden attr updates base
        new_workers = [MagicMock(), MagicMock()]
        view._workers = new_workers
        self.assertEqual(wg._workers, new_workers)


class TestFuncGenerator(unittest.TestCase):
    """Tests for func_generator and the Functor class."""

    @patch("axon.controller.ray.worker_group.ray")
    def test_functor_calls_dispatch_execute_collect(self, mock_ray):
        """Test that the Functor calls dispatch, execute, and collect in order."""
        from axon.controller.ray.worker_group import RayWorkerGroup, func_generator

        mock_workers = [MagicMock() for _ in range(2)]
        wg = RayWorkerGroup(
            resource_pool=None,
            worker_names=["w0", "w1"],
            worker_handles=mock_workers,
        )
        wg._enable_ray_collective = False

        dispatch_fn = MagicMock(return_value=(("dispatched_args",), {}))
        collect_fn = MagicMock(return_value="collected_result")
        execute_fn = MagicMock(return_value=["ref0", "ref1"])
        mock_ray.get.return_value = ["result0", "result1"]

        func = func_generator(
            wg,
            "test_method",
            dispatch_fn=dispatch_fn,
            collect_fn=collect_fn,
            execute_fn=execute_fn,
            blocking=True,
            disable_collective=False,
        )

        result = func("input_data")

        dispatch_fn.assert_called_once()
        execute_fn.assert_called_once_with("test_method", "dispatched_args")
        mock_ray.get.assert_called_once()
        collect_fn.assert_called_once()
        self.assertEqual(result, "collected_result")

    @patch("axon.controller.ray.worker_group.ray")
    def test_functor_non_blocking_skips_ray_get(self, mock_ray):
        """Test Functor with blocking=False doesn't call ray.get."""
        from axon.controller.ray.worker_group import RayWorkerGroup, func_generator

        mock_workers = [MagicMock()]
        wg = RayWorkerGroup(
            resource_pool=None,
            worker_names=["w0"],
            worker_handles=mock_workers,
        )
        wg._enable_ray_collective = False

        dispatch_fn = MagicMock(return_value=((), {}))
        collect_fn = MagicMock(return_value="collected")
        execute_fn = MagicMock(return_value=["future_ref"])

        func = func_generator(
            wg,
            "test_method",
            dispatch_fn=dispatch_fn,
            collect_fn=collect_fn,
            execute_fn=execute_fn,
            blocking=False,
            disable_collective=False,
        )

        func()
        mock_ray.get.assert_not_called()

    @patch("axon.controller.ray.worker_group.ray")
    def test_functor_class_name_matches_method_name(self, mock_ray):
        """Test that generated Functor class is named after the method for observability."""
        from axon.controller.ray.worker_group import RayWorkerGroup, func_generator

        mock_workers = [MagicMock()]
        wg = RayWorkerGroup(
            resource_pool=None,
            worker_names=["w0"],
            worker_handles=mock_workers,
        )
        wg._enable_ray_collective = False

        func = func_generator(
            wg,
            "my_cool_method",
            dispatch_fn=MagicMock(return_value=((), {})),
            collect_fn=MagicMock(return_value=None),
            execute_fn=MagicMock(return_value=[]),
            blocking=False,
            disable_collective=False,
        )

        self.assertEqual(type(func).__name__, "my_cool_method")


class TestRayWorkerGroupProperties(unittest.TestCase):
    """Tests for RayWorkerGroup properties."""

    @patch("axon.controller.ray.worker_group.ray")
    def test_worker_names_property(self, mock_ray):
        """Test worker_names property returns names."""
        from axon.controller.ray.worker_group import RayWorkerGroup

        names = ["w0", "w1", "w2"]
        wg = RayWorkerGroup(
            resource_pool=None,
            worker_names=names,
            worker_handles=[MagicMock() for _ in range(3)],
        )
        self.assertEqual(wg.worker_names, names)

    @patch("axon.controller.ray.worker_group.ray")
    def test_master_address_and_port_properties(self, mock_ray):
        """Test master_address and master_port properties."""
        from axon.controller.ray.worker_group import RayWorkerGroup

        wg = RayWorkerGroup(
            resource_pool=None,
            worker_names=["w0"],
            worker_handles=[MagicMock()],
            master_addr="192.168.1.1",
            master_port="29500",
        )
        self.assertEqual(wg.master_address, "192.168.1.1")
        self.assertEqual(wg.master_port, "29500")

    @patch("axon.controller.ray.worker_group.ray")
    def test_workers_property(self, mock_ray):
        """Test workers property returns worker list."""
        from axon.controller.ray.worker_group import RayWorkerGroup

        handles = [MagicMock(), MagicMock()]
        wg = RayWorkerGroup(
            resource_pool=None,
            worker_names=["w0", "w1"],
            worker_handles=handles,
        )
        self.assertEqual(wg.workers, handles)

    @patch("axon.controller.ray.worker_group.ray")
    def test_world_size_property_uses_world_size_attr(self, mock_ray):
        """Test world_size comes from _world_size, not len(_workers)."""
        from axon.controller.ray.worker_group import RayWorkerGroup

        wg = RayWorkerGroup(
            resource_pool=None,
            worker_names=["w0", "w1"],
            worker_handles=[MagicMock(), MagicMock()],
        )
        self.assertEqual(wg.world_size, 2)


class TestRayWorkerGroupEdgeCases(unittest.TestCase):
    """Edge case tests for RayWorkerGroup."""

    @patch("axon.controller.ray.worker_group.ray")
    def test_execute_all_delegates_to_execute(self, mock_ray):
        """Test execute_all is an alias that calls execute."""
        from axon.controller.ray.worker_group import RayWorkerGroup

        mock_workers = [MagicMock() for _ in range(2)]
        for w in mock_workers:
            w.method = MagicMock()
            w.method.remote.return_value = MagicMock()

        wg = RayWorkerGroup(
            resource_pool=None,
            worker_names=["w0", "w1"],
            worker_handles=mock_workers,
        )

        wg.execute_all("method", "arg1")

        for w in mock_workers:
            w.method.remote.assert_called_once_with("arg1")

    @patch("axon.controller.ray.worker_group.ray")
    def test_execute_with_no_args(self, mock_ray):
        """Test execute with no args or kwargs."""
        from axon.controller.ray.worker_group import RayWorkerGroup

        mock_workers = [MagicMock() for _ in range(2)]
        for w in mock_workers:
            w.method = MagicMock()
            w.method.remote.return_value = MagicMock()

        wg = RayWorkerGroup(
            resource_pool=None,
            worker_names=["w0", "w1"],
            worker_handles=mock_workers,
        )

        wg.execute("method")

        for w in mock_workers:
            w.method.remote.assert_called_once_with()

    @patch("axon.controller.ray.worker_group.ray")
    def test_execute_kwargs_only_distribution(self, mock_ray):
        """Test execute distributes kwargs-only args matching worker count."""
        from axon.controller.ray.worker_group import RayWorkerGroup

        mock_workers = [MagicMock() for _ in range(2)]
        for w in mock_workers:
            w.process = MagicMock()
            w.process.remote.return_value = MagicMock()

        wg = RayWorkerGroup(
            resource_pool=None,
            worker_names=["w0", "w1"],
            worker_handles=mock_workers,
        )

        # kwargs-only with matching lengths triggers distribution
        wg.execute("process", data=["a", "b"])

        mock_workers[0].process.remote.assert_called_once_with(data="a")
        mock_workers[1].process.remote.assert_called_once_with(data="b")

    @patch("axon.controller.ray.worker_group.ray")
    def test_execute_mixed_list_and_nonlist_broadcasts(self, mock_ray):
        """Test execute broadcasts when args are a mix of list and non-list."""
        from axon.controller.ray.worker_group import RayWorkerGroup

        mock_workers = [MagicMock() for _ in range(2)]
        for w in mock_workers:
            w.process = MagicMock()
            w.process.remote.return_value = MagicMock()

        wg = RayWorkerGroup(
            resource_pool=None,
            worker_names=["w0", "w1"],
            worker_handles=mock_workers,
        )

        # One list arg and one non-list arg -> all() on isinstance fails -> broadcast
        wg.execute("process", ["d0", "d1"], "config")

        for w in mock_workers:
            w.process.remote.assert_called_once_with(["d0", "d1"], "config")

    @patch("axon.controller.ray.worker_group.ray")
    def test_collective_group_name_includes_prefix(self, mock_ray):
        """Test that collective group name is derived from name_prefix."""
        from axon.controller.ray.worker_group import RayWorkerGroup

        wg = RayWorkerGroup(
            resource_pool=None,
            worker_names=["w0"],
            worker_handles=[MagicMock()],
            name_prefix="my_prefix",
        )
        self.assertEqual(wg._ray_collective_group_name, "my_prefix_ray_cc")

    @patch("axon.controller.ray.worker_group.ray")
    def test_enable_ray_collective_flag(self, mock_ray):
        """Test enable_ray_collective initialises flag correctly."""
        from axon.controller.ray.worker_group import RayWorkerGroup

        wg_enabled = RayWorkerGroup(
            resource_pool=None,
            worker_names=["w0"],
            worker_handles=[MagicMock()],
            enable_ray_collective=True,
        )
        self.assertTrue(wg_enabled._enable_ray_collective)

        wg_disabled = RayWorkerGroup(
            resource_pool=None,
            worker_names=["w0"],
            worker_handles=[MagicMock()],
            enable_ray_collective=False,
        )
        self.assertFalse(wg_disabled._enable_ray_collective)


class TestRayActorWithInitArgsEdgeCases(unittest.TestCase):
    """Additional edge case tests for RayActorWithInitArgs."""

    @patch("axon.controller.ray.class_init.PlacementGroupSchedulingStrategy")
    def test_call_without_gpu(self, mock_pg_strategy):
        """Test __call__ with no GPU in resource_dict."""
        from axon.controller.ray.class_init import RayActorWithInitArgs

        mock_cls = MagicMock()
        mock_cls.options.return_value.remote.return_value = MagicMock()

        ray_cls = RayActorWithInitArgs(cls=mock_cls)
        ray_cls(
            placement_group=MagicMock(),
            placement_group_bundle_idx=0,
            resource_dict={"CPU": 4},
        )

        call_kwargs = mock_cls.options.call_args[1]
        self.assertNotIn("num_gpus", call_kwargs)
        self.assertEqual(call_kwargs["num_cpus"], 4)

    @patch("axon.controller.ray.class_init.PlacementGroupSchedulingStrategy")
    def test_call_without_cpu(self, mock_pg_strategy):
        """Test __call__ with no CPU in resource_dict."""
        from axon.controller.ray.class_init import RayActorWithInitArgs

        mock_cls = MagicMock()
        mock_cls.options.return_value.remote.return_value = MagicMock()

        ray_cls = RayActorWithInitArgs(cls=mock_cls)
        ray_cls(
            placement_group=MagicMock(),
            placement_group_bundle_idx=0,
            resource_dict={"GPU": 2},
        )

        call_kwargs = mock_cls.options.call_args[1]
        self.assertNotIn("num_cpus", call_kwargs)
        self.assertEqual(call_kwargs["num_gpus"], 2)

    @patch("axon.controller.ray.class_init.PlacementGroupSchedulingStrategy")
    def test_call_with_empty_resource_dict(self, mock_pg_strategy):
        """Test __call__ with empty resource_dict."""
        from axon.controller.ray.class_init import RayActorWithInitArgs

        mock_cls = MagicMock()
        mock_cls.options.return_value.remote.return_value = MagicMock()

        ray_cls = RayActorWithInitArgs(cls=mock_cls)
        ray_cls(
            placement_group=MagicMock(),
            placement_group_bundle_idx=0,
            resource_dict={},
        )

        call_kwargs = mock_cls.options.call_args[1]
        self.assertNotIn("num_gpus", call_kwargs)
        self.assertNotIn("num_cpus", call_kwargs)
        self.assertEqual(call_kwargs["resources"], {})

    def test_update_options_empty(self):
        """Test update_options with empty dict is a no-op."""
        from axon.controller.ray.class_init import RayActorWithInitArgs

        ray_cls = RayActorWithInitArgs(cls=MagicMock())
        ray_cls.update_options({})
        self.assertEqual(ray_cls._options, {})

    def test_initial_options_empty(self):
        """Test initial _options is empty dict."""
        from axon.controller.ray.class_init import RayActorWithInitArgs

        ray_cls = RayActorWithInitArgs(cls=MagicMock())
        self.assertEqual(ray_cls._options, {})


class TestMakeWgViewEdgeCases(unittest.TestCase):
    """Additional edge case tests for _make_wg_view."""

    @patch("axon.controller.ray.worker_group.ray")
    def test_view_isinstance_check(self, mock_ray):
        """Test that the view is an instance of the original class type."""
        from axon.controller.ray.worker_group import RayWorkerGroup, _make_wg_view

        wg = RayWorkerGroup(
            resource_pool=None,
            worker_names=["w0"],
            worker_handles=[MagicMock()],
        )
        view = _make_wg_view(wg, _ray_collective_initialized=False)
        self.assertIsInstance(view, RayWorkerGroup)

    @patch("axon.controller.ray.worker_group.ray")
    def test_view_with_no_overrides(self, mock_ray):
        """Test view with no overrides delegates everything to base."""
        from axon.controller.ray.worker_group import RayWorkerGroup, _make_wg_view

        handles = [MagicMock(), MagicMock()]
        wg = RayWorkerGroup(
            resource_pool=None,
            worker_names=["w0", "w1"],
            worker_handles=handles,
        )
        view = _make_wg_view(wg)
        self.assertEqual(view._workers, handles)
        self.assertEqual(view._world_size, 2)

    @patch("axon.controller.ray.worker_group.ray")
    def test_view_multiple_overrides(self, mock_ray):
        """Test view with multiple attribute overrides."""
        from axon.controller.ray.worker_group import RayWorkerGroup, _make_wg_view

        wg = RayWorkerGroup(
            resource_pool=None,
            worker_names=["w0"],
            worker_handles=[MagicMock()],
        )
        wg._ray_collective_initialized = True
        wg._enable_ray_collective = True

        view = _make_wg_view(
            wg,
            _ray_collective_initialized=False,
            _enable_ray_collective=False,
        )

        self.assertFalse(view._ray_collective_initialized)
        self.assertFalse(view._enable_ray_collective)
        # Originals unchanged
        self.assertTrue(wg._ray_collective_initialized)
        self.assertTrue(wg._enable_ray_collective)


if __name__ == "__main__":
    unittest.main()
