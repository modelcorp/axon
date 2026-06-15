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
Unit tests for axon.controller.decorator module.

Tests cover:
- DynamicEnum (Dispatch / Execute) registration and lookup
- Core helpers: _chunk_item, _broadcast_item, _remap
- dispatch() and collect() functions
- Predefined dispatch/execute fn lookups
- _materialize_futures helper
- _check_dispatch_mode / _check_execute_mode validation
- @register decorator attribute stamping and wrapper behaviour
"""

import asyncio
import unittest
from unittest.mock import MagicMock

import torch
from tensordict import TensorDict

from axon.controller.decorator import (
    MAGIC_ATTR,
    Dispatch,
    Execute,
    _broadcast_item,
    _check_dispatch_mode,
    _check_execute_mode,
    _chunk_item,
    _materialize_futures,
    _passthrough_collect,
    _passthrough_dispatch,
    _remap,
    collect,
    dispatch,
    get_predefined_dispatch_fn,
    get_predefined_execute_fn,
    register,
)

# ---------------------------------------------------------------------------
# DynamicEnum (Dispatch / Execute)
# ---------------------------------------------------------------------------


class TestDispatchEnum(unittest.TestCase):
    """Tests for the Dispatch DynamicEnum."""

    def test_predefined_members_exist(self):
        self.assertIsNotNone(Dispatch.RANK_ZERO)
        self.assertIsNotNone(Dispatch.ONE_TO_ALL)
        self.assertIsNotNone(Dispatch.ALL_TO_ALL)
        self.assertIsNotNone(Dispatch.DP_COMPUTE_PROTO_WITH_FUNC)
        self.assertIsNotNone(Dispatch.DIRECT_SAMPLER_METHOD)

    def test_members_are_unique(self):
        members = [
            Dispatch.RANK_ZERO,
            Dispatch.ONE_TO_ALL,
            Dispatch.ALL_TO_ALL,
            Dispatch.DP_COMPUTE_PROTO_WITH_FUNC,
            Dispatch.DIRECT_SAMPLER_METHOD,
        ]
        values = [m.value for m in members]
        self.assertEqual(len(values), len(set(values)))

    def test_dispatch_repr(self):
        r = repr(Dispatch.RANK_ZERO)
        self.assertIn("Dispatch", r)
        self.assertIn("RANK_ZERO", r)

    def test_dispatch_contains(self):
        self.assertIn("RANK_ZERO", Dispatch)
        self.assertIn(Dispatch.ONE_TO_ALL, Dispatch)
        self.assertNotIn("NONEXISTENT", Dispatch)

    def test_dispatch_iter(self):
        members = list(Dispatch)
        self.assertGreaterEqual(len(members), 5)

    def test_dispatch_names_and_values(self):
        self.assertIn("RANK_ZERO", Dispatch.names())
        self.assertIn(Dispatch.ALL_TO_ALL, Dispatch.values())


class TestExecuteEnum(unittest.TestCase):
    """Tests for the Execute DynamicEnum."""

    def test_predefined_members_exist(self):
        self.assertIsNotNone(Execute.ALL)
        self.assertIsNotNone(Execute.RANK_ZERO)

    def test_members_are_unique(self):
        self.assertNotEqual(Execute.ALL.value, Execute.RANK_ZERO.value)


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------


class TestChunkItem(unittest.TestCase):
    """Tests for _chunk_item."""

    def test_chunk_tensordict(self):
        td = TensorDict({"a": torch.arange(6)}, batch_size=[6])
        chunks = _chunk_item(td, 3)
        self.assertEqual(len(chunks), 3)
        for c in chunks:
            self.assertEqual(len(c), 2)

    def test_chunk_tensor(self):
        t = torch.arange(8)
        chunks = _chunk_item(t, 4)
        self.assertEqual(len(chunks), 4)
        for c in chunks:
            self.assertEqual(len(c), 2)

    def test_chunk_non_chunkable_broadcasts(self):
        """Items without .chunk() are broadcast to all chunks."""
        result = _chunk_item("hello", 3)
        self.assertEqual(result, ["hello", "hello", "hello"])

    def test_chunk_single(self):
        t = torch.arange(4)
        chunks = _chunk_item(t, 1)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(len(chunks[0]), 4)


class TestBroadcastItem(unittest.TestCase):
    """Tests for _broadcast_item."""

    def test_broadcast_normal(self):
        result = _broadcast_item("data", 4)
        self.assertEqual(result, ["data", "data", "data", "data"])

    def test_broadcast_collective(self):
        """With collective=True, only rank 0 gets real data; rest get None."""
        result = _broadcast_item("data", 3, collective=True)
        self.assertEqual(result, ["data", None, None])

    def test_broadcast_single_worker(self):
        result = _broadcast_item(42, 1)
        self.assertEqual(result, [42])

    def test_broadcast_collective_single_worker(self):
        result = _broadcast_item(42, 1, collective=True)
        self.assertEqual(result, [42])


class TestRemap(unittest.TestCase):
    """Tests for _remap."""

    def test_identity_mapping(self):
        items = [[10, 20, 30]]
        result = _remap(items, [0, 1, 2], 3)
        self.assertEqual(result, [[10, 20, 30]])

    def test_reverse_mapping(self):
        items = [[10, 20, 30]]
        result = _remap(items, [2, 1, 0], 3)
        self.assertEqual(result, [[30, 20, 10]])

    def test_duplicate_mapping(self):
        """Multiple workers map to same dp rank."""
        items = [["a", "b"]]
        result = _remap(items, [0, 0, 1, 1], 4)
        self.assertEqual(result, [["a", "a", "b", "b"]])

    def test_multiple_items(self):
        items = [[10, 20], [100, 200]]
        result = _remap(items, [1, 0], 2)
        self.assertEqual(result, [[20, 10], [200, 100]])


# ---------------------------------------------------------------------------
# dispatch() function
# ---------------------------------------------------------------------------


class TestDispatchFunction(unittest.TestCase):
    """Tests for the unified dispatch() function."""

    def _make_wg(self, world_size, collective=False):
        wg = MagicMock()
        wg.world_size = world_size
        wg._ray_collective_initialized = collective
        return wg

    def test_broadcast_mode(self):
        """replicate=True broadcasts args to all workers."""
        wg = self._make_wg(3)
        args, kwargs = dispatch(wg, "x", key="v", replicate=True)
        self.assertEqual(args, (["x", "x", "x"],))
        self.assertEqual(kwargs, {"key": ["v", "v", "v"]})

    def test_scatter_mode_tensor(self):
        """replicate=False chunks tensor across workers."""
        wg = self._make_wg(2)
        t = torch.arange(4)
        args, kwargs = dispatch(wg, t, replicate=False)
        self.assertEqual(len(args), 1)
        self.assertEqual(len(args[0]), 2)
        torch.testing.assert_close(args[0][0], torch.tensor([0, 1]))
        torch.testing.assert_close(args[0][1], torch.tensor([2, 3]))

    def test_scatter_with_split_first_arg_function(self):
        """split_first_arg=True replicates function, splits data."""
        wg = self._make_wg(2)
        fn = lambda x: x  # noqa: E731
        data = torch.arange(4)
        args, kwargs = dispatch(wg, fn, data, replicate=False, split_first_arg=True)
        # First arg (function) should be replicated
        self.assertIs(args[0][0], fn)
        self.assertIs(args[0][1], fn)
        # Second arg (data) should be chunked
        self.assertEqual(len(args[1]), 2)

    def test_rank_mapping(self):
        """rank_mapping dispatches using dp remapping."""
        wg = self._make_wg(4, collective=True)
        t = torch.arange(4)
        # 2 dp groups, mapping: workers 0,1 -> dp0, workers 2,3 -> dp1
        args, kwargs = dispatch(wg, t, rank_mapping=[0, 0, 1, 1])
        self.assertEqual(len(args), 1)
        self.assertEqual(len(args[0]), 4)

    def test_broadcast_with_kwargs_only(self):
        wg = self._make_wg(2)
        args, kwargs = dispatch(wg, key1="a", key2="b", replicate=True)
        self.assertEqual(args, ())
        self.assertEqual(kwargs, {"key1": ["a", "a"], "key2": ["b", "b"]})


# ---------------------------------------------------------------------------
# collect() function
# ---------------------------------------------------------------------------


class TestCollectFunction(unittest.TestCase):
    """Tests for the unified collect() function."""

    def _make_wg(self, collective=False):
        wg = MagicMock()
        wg._ray_collective_initialized = collective
        wg._enable_ray_collective = collective
        return wg

    def test_passthrough(self):
        wg = self._make_wg()
        output = [1, 2, 3]
        result = collect(wg, output)
        self.assertEqual(result, [1, 2, 3])

    def test_collect_mask(self):
        """collect_mask filters outputs to specific ranks."""
        wg = self._make_wg()
        output = ["a", "b", "c", "d"]
        result = collect(wg, output, collect_mask=[True, False, True, False])
        self.assertEqual(result, ["a", "c"])

    def test_collective_unwrap(self):
        """With collective enabled, unwraps rank-0 gathered list."""
        wg = self._make_wg(collective=True)
        # In collective mode, rank 0 has a gathered list, others are None
        output = [["result_0", "result_1"], None]
        result = collect(wg, output)
        self.assertEqual(result, ["result_0", "result_1"])

    def test_collective_no_unwrap_when_multiple_non_none(self):
        """Don't unwrap if multiple non-None values exist."""
        wg = self._make_wg(collective=True)
        output = ["result_0", "result_1"]
        result = collect(wg, output)
        self.assertEqual(result, ["result_0", "result_1"])


# ---------------------------------------------------------------------------
# Predefined dispatch/execute fn lookups
# ---------------------------------------------------------------------------


class TestPredefinedFns(unittest.TestCase):
    """Tests for get_predefined_dispatch_fn and get_predefined_execute_fn."""

    def test_rank_zero_dispatch(self):
        fns = get_predefined_dispatch_fn(Dispatch.RANK_ZERO)
        self.assertIs(fns["dispatch_fn"], _passthrough_dispatch)
        self.assertIs(fns["collect_fn"], _passthrough_collect)

    def test_one_to_all_dispatch(self):
        fns = get_predefined_dispatch_fn(Dispatch.ONE_TO_ALL)
        self.assertIn("dispatch_fn", fns)
        self.assertIn("collect_fn", fns)

    def test_all_to_all_dispatch(self):
        fns = get_predefined_dispatch_fn(Dispatch.ALL_TO_ALL)
        self.assertIs(fns["dispatch_fn"], _passthrough_dispatch)

    def test_dp_compute_proto_with_func_dispatch(self):
        fns = get_predefined_dispatch_fn(Dispatch.DP_COMPUTE_PROTO_WITH_FUNC)
        self.assertIn("dispatch_fn", fns)
        self.assertIn("collect_fn", fns)

    def test_direct_sampler_method_raises(self):
        fns = get_predefined_dispatch_fn(Dispatch.DIRECT_SAMPLER_METHOD)
        with self.assertRaises(NotImplementedError):
            fns["dispatch_fn"]()
        with self.assertRaises(NotImplementedError):
            fns["collect_fn"]()

    def test_execute_all(self):
        result = get_predefined_execute_fn(Execute.ALL)
        self.assertEqual(result["execute_fn_name"], "execute_all")

    def test_execute_rank_zero(self):
        result = get_predefined_execute_fn(Execute.RANK_ZERO)
        self.assertEqual(result["execute_fn_name"], "execute_rank_zero")

    def test_invalid_dispatch_mode_raises(self):
        with self.assertRaises(KeyError):
            get_predefined_dispatch_fn("not_a_dispatch_mode")


# ---------------------------------------------------------------------------
# _check_dispatch_mode / _check_execute_mode
# ---------------------------------------------------------------------------


class TestCheckModes(unittest.TestCase):
    """Tests for _check_dispatch_mode and _check_execute_mode."""

    def test_check_dispatch_with_enum(self):
        # Should not raise
        _check_dispatch_mode(Dispatch.RANK_ZERO)
        _check_dispatch_mode(Dispatch.ALL_TO_ALL)

    def test_check_dispatch_with_dict(self):
        # Should not raise when dict has both keys
        _check_dispatch_mode({"dispatch_fn": lambda: None, "collect_fn": lambda: None})

    def test_check_dispatch_with_incomplete_dict_raises(self):
        with self.assertRaises(AssertionError):
            _check_dispatch_mode({"dispatch_fn": lambda: None})

    def test_check_dispatch_with_invalid_type_raises(self):
        with self.assertRaises(ValueError):
            _check_dispatch_mode("invalid")
        with self.assertRaises(ValueError):
            _check_dispatch_mode(42)

    def test_check_execute_with_enum(self):
        _check_execute_mode(Execute.ALL)
        _check_execute_mode(Execute.RANK_ZERO)

    def test_check_execute_with_invalid_type_raises(self):
        with self.assertRaises(AssertionError):
            _check_execute_mode("ALL")
        with self.assertRaises(AssertionError):
            _check_execute_mode(42)


# ---------------------------------------------------------------------------
# _materialize_futures
# ---------------------------------------------------------------------------


class TestMaterializeFutures(unittest.TestCase):
    """Tests for _materialize_futures."""

    def test_no_futures(self):
        args, kwargs = _materialize_futures(1, "hello", key="val")
        self.assertEqual(args, (1, "hello"))
        self.assertEqual(kwargs, {"key": "val"})

    def test_with_data_proto_future(self):
        mock_future = MagicMock()
        mock_future.get.return_value = "resolved_value"
        # Need to make isinstance check work
        from axon.protocol import DataProtoFuture

        mock_future.__class__ = DataProtoFuture

        args, kwargs = _materialize_futures(mock_future, "normal")
        self.assertEqual(args[0], "resolved_value")
        self.assertEqual(args[1], "normal")


# ---------------------------------------------------------------------------
# @register decorator
# ---------------------------------------------------------------------------


class TestRegisterDecorator(unittest.TestCase):
    """Tests for the @register decorator."""

    def test_stamps_magic_attr(self):
        @register(dispatch_mode=Dispatch.ONE_TO_ALL, execute_mode=Execute.ALL)
        def my_method(self):
            pass

        self.assertTrue(hasattr(my_method, MAGIC_ATTR))
        attr = getattr(my_method, MAGIC_ATTR)
        self.assertEqual(attr["dispatch_mode"], Dispatch.ONE_TO_ALL)
        self.assertEqual(attr["execute_mode"], Execute.ALL)
        self.assertTrue(attr["blocking"])
        self.assertFalse(attr["disable_collective"])

    def test_custom_blocking_and_collective(self):
        @register(
            dispatch_mode=Dispatch.ALL_TO_ALL,
            execute_mode=Execute.RANK_ZERO,
            blocking=False,
            disable_collective=True,
        )
        def my_method(self):
            pass

        attr = getattr(my_method, MAGIC_ATTR)
        self.assertFalse(attr["blocking"])
        self.assertTrue(attr["disable_collective"])

    def test_custom_dispatch_dict(self):
        custom_dispatch = {"dispatch_fn": lambda: None, "collect_fn": lambda: None}

        @register(dispatch_mode=custom_dispatch, execute_mode=Execute.ALL)
        def my_method(self):
            pass

        attr = getattr(my_method, MAGIC_ATTR)
        self.assertEqual(attr["dispatch_mode"], custom_dispatch)

    def test_decorated_sync_function_is_sync(self):
        @register()
        def my_method(self):
            return "result"

        self.assertFalse(asyncio.iscoroutinefunction(my_method))

    def test_decorated_async_function_is_async(self):
        @register()
        async def my_method(self):
            return "result"

        self.assertTrue(asyncio.iscoroutinefunction(my_method))

    def test_preserves_function_name(self):
        @register()
        def original_name(self):
            pass

        self.assertEqual(original_name.__name__, "original_name")

    def test_invalid_dispatch_mode_raises(self):
        with self.assertRaises(ValueError):

            @register(dispatch_mode="invalid")
            def bad_method(self):
                pass

    def test_invalid_execute_mode_raises(self):
        with self.assertRaises(AssertionError):

            @register(execute_mode="invalid")
            def bad_method(self):
                pass

    def test_all_dispatch_modes_accepted(self):
        """All predefined Dispatch modes should be accepted by @register."""
        for mode in [
            Dispatch.RANK_ZERO,
            Dispatch.ONE_TO_ALL,
            Dispatch.ALL_TO_ALL,
            Dispatch.DP_COMPUTE_PROTO_WITH_FUNC,
            Dispatch.DIRECT_SAMPLER_METHOD,
        ]:

            @register(dispatch_mode=mode)
            def method(self):
                pass

            self.assertTrue(hasattr(method, MAGIC_ATTR))

    def test_sync_wrapper_calls_function(self):
        """Sync decorated function calls original logic correctly."""
        call_log = []

        @register(dispatch_mode=Dispatch.ALL_TO_ALL, execute_mode=Execute.ALL)
        def my_func(wg, x):
            call_log.append(x)
            return x * 2

        mock_wg = MagicMock()
        mock_wg._enable_ray_collective = False

        result = my_func(mock_wg, 5)
        self.assertEqual(result, 10)
        self.assertEqual(call_log, [5])

    def test_async_wrapper_calls_function(self):
        """Async decorated function calls original logic correctly."""

        @register(dispatch_mode=Dispatch.ALL_TO_ALL, execute_mode=Execute.ALL)
        async def my_func(wg, x):
            return x * 3

        mock_wg = MagicMock()
        mock_wg._enable_ray_collective = False

        result = asyncio.get_event_loop().run_until_complete(my_func(mock_wg, 7))
        self.assertEqual(result, 21)


# ---------------------------------------------------------------------------
# Passthrough functions
# ---------------------------------------------------------------------------


class TestPassthroughFunctions(unittest.TestCase):
    """Tests for _passthrough_dispatch and _passthrough_collect."""

    def test_passthrough_dispatch(self):
        args, kwargs = _passthrough_dispatch(MagicMock(), 1, 2, key="val")
        self.assertEqual(args, (1, 2))
        self.assertEqual(kwargs, {"key": "val"})

    def test_passthrough_collect(self):
        output = [1, 2, 3]
        result = _passthrough_collect(MagicMock(), output)
        self.assertEqual(result, [1, 2, 3])


# ---------------------------------------------------------------------------
# ONE_TO_ALL dispatch integration
# ---------------------------------------------------------------------------


class TestOneToAllDispatch(unittest.TestCase):
    """Integration tests for ONE_TO_ALL dispatch via predefined fn."""

    def test_one_to_all_broadcasts_tensor(self):
        fns = get_predefined_dispatch_fn(Dispatch.ONE_TO_ALL)
        wg = MagicMock()
        wg.world_size = 3
        wg._ray_collective_initialized = False

        t = torch.tensor([1, 2, 3])
        args, kwargs = fns["dispatch_fn"](wg, t)

        self.assertEqual(len(args), 1)
        self.assertEqual(len(args[0]), 3)
        for chunk in args[0]:
            torch.testing.assert_close(chunk, t)

    def test_one_to_all_collect_returns_list(self):
        fns = get_predefined_dispatch_fn(Dispatch.ONE_TO_ALL)
        wg = MagicMock()
        wg._ray_collective_initialized = False
        wg._enable_ray_collective = False

        result = fns["collect_fn"](wg, [10, 20, 30])
        self.assertEqual(result, [10, 20, 30])


# ---------------------------------------------------------------------------
# DP_COMPUTE_PROTO_WITH_FUNC dispatch integration
# ---------------------------------------------------------------------------


class TestDPComputeProtoWithFuncDispatch(unittest.TestCase):
    """Integration tests for DP_COMPUTE_PROTO_WITH_FUNC dispatch."""

    def test_splits_data_and_replicates_function(self):
        fns = get_predefined_dispatch_fn(Dispatch.DP_COMPUTE_PROTO_WITH_FUNC)
        wg = MagicMock()
        wg.world_size = 2
        wg._ray_collective_initialized = False

        fn = lambda x: x  # noqa: E731
        data = torch.arange(4)
        args, kwargs = fns["dispatch_fn"](wg, fn, data)

        # Function replicated
        self.assertIs(args[0][0], fn)
        self.assertIs(args[0][1], fn)
        # Data chunked
        self.assertEqual(len(args[1]), 2)
        torch.testing.assert_close(args[1][0], torch.tensor([0, 1]))
        torch.testing.assert_close(args[1][1], torch.tensor([2, 3]))


if __name__ == "__main__":
    unittest.main()
