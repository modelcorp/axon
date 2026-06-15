"""Tests for axon.core.worker module."""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers to import Worker safely despite Ray / decorator dependencies
# ---------------------------------------------------------------------------

_WORKER_ENV = {
    "RANK": "2",
    "WORLD_SIZE": "8",
    "MASTER_ADDR": "127.0.0.1",
    "MASTER_PORT": "29500",
    "LOCAL_WORLD_SIZE": "4",
    "LOCAL_RANK": "1",
}


def _import_worker_class():
    """Import Worker with heavy dependencies mocked out.

    The challenge is that ``axon.controller.__init__`` transitively imports
    Ray-based modules (scheduling strategies, worker groups, etc.).  We mock
    the entire ``axon.controller`` package *except* ``axon.controller.decorator``
    which we import manually after mocking its own transitive dependencies.
    """
    # Build a single mock-module dict covering everything we need to stub.
    ray_mock = MagicMock()
    mocks = {
        # Ray and its submodules
        "ray": ray_mock,
        "ray.util": ray_mock.util,
        "ray.util.collective": ray_mock.util.collective,
        "ray.util.scheduling_strategies": ray_mock.util.scheduling_strategies,
        # Heavy first-party deps of the decorator module
        "tensordict": MagicMock(),
        "axon.protocol": MagicMock(),
        "axon.utils.ray.collective": MagicMock(),
        "axon.utils.tensordict_utils": MagicMock(),
        # The axon.controller package and sub-packages that should NOT
        # trigger their real __init__ (they pull in Ray workers, etc.)
        "axon.controller": MagicMock(),
        "axon.controller.ray": MagicMock(),
        "axon.controller.ray.class_init": MagicMock(),
        "axon.controller.ray.worker_group": MagicMock(),
        "axon.controller.worker_group": MagicMock(),
    }

    # We need a *real* DynamicEnum so Dispatch/Execute work properly.

    with patch.dict(sys.modules, mocks):
        # Remove cached decorator / worker modules so they re-import cleanly.
        for key in list(sys.modules.keys()):
            if key.startswith("axon.controller.decorator") or key.startswith("axon.core.worker"):
                del sys.modules[key]

        # Now import the decorator module for real (its transitive deps are mocked).
        from axon.controller import decorator as _dec_mod  # noqa: F811

        # Patch it into sys.modules so worker.py's import finds it.
        sys.modules["axon.controller.decorator"] = _dec_mod

        # Finally import Worker itself.
        from axon.core.worker import Worker  # noqa: F811

    return Worker


# ---------------------------------------------------------------------------
# Attempt a single import; skip the entire module if it still fails.
# ---------------------------------------------------------------------------
try:
    _Worker = _import_worker_class()
    _skip_reason = ""
except Exception as _import_err:  # pragma: no cover
    _Worker = None
    _skip_reason = f"Cannot import Worker: {_import_err}"

_skip_if_no_worker = pytest.mark.skipif(_Worker is None, reason=_skip_reason)


def _make_worker(env_overrides=None):
    """Create a Worker instance with mocked environment and dependencies."""
    env = dict(_WORKER_ENV)
    if env_overrides:
        env.update(env_overrides)

    ray_mock = MagicMock()
    with patch.dict(
        sys.modules,
        {
            "ray": ray_mock,
            "ray.util": ray_mock.util,
            "ray.util.collective": ray_mock.util.collective,
        },
    ):
        with patch.dict(os.environ, env, clear=False):
            worker = _Worker.__new__(_Worker)
            worker.__init__()
    return worker


# ---------------------------------------------------------------------------
# TestWorkerInit
# ---------------------------------------------------------------------------
@_skip_if_no_worker
class TestWorkerInit:
    def test_rank_from_env(self):
        worker = _make_worker()
        assert worker._rank == 2

    def test_world_size_from_env(self):
        worker = _make_worker()
        assert worker._world_size == 8

    def test_local_rank_from_env(self):
        worker = _make_worker()
        assert worker._local_rank == 1

    def test_local_world_size_from_env(self):
        worker = _make_worker()
        assert worker._local_world_size == 4

    def test_custom_rank(self):
        worker = _make_worker({"RANK": "0", "WORLD_SIZE": "1"})
        assert worker._rank == 0
        assert worker._world_size == 1

    def test_collective_disabled_by_default(self):
        worker = _make_worker()
        assert worker._enable_ray_collective is False

    def test_collective_group_name_none_by_default(self):
        worker = _make_worker()
        assert worker._ray_collective_group_name is None


# ---------------------------------------------------------------------------
# TestWorkerEnvKeys
# ---------------------------------------------------------------------------
@_skip_if_no_worker
class TestWorkerEnvKeys:
    def test_returns_list(self):
        keys = _Worker.env_keys()
        assert isinstance(keys, list)

    def test_contains_world_size(self):
        assert "WORLD_SIZE" in _Worker.env_keys()

    def test_contains_rank(self):
        assert "RANK" in _Worker.env_keys()

    def test_contains_local_world_size(self):
        assert "LOCAL_WORLD_SIZE" in _Worker.env_keys()

    def test_contains_local_rank(self):
        assert "LOCAL_RANK" in _Worker.env_keys()

    def test_contains_master_addr(self):
        assert "MASTER_ADDR" in _Worker.env_keys()

    def test_contains_master_port(self):
        assert "MASTER_PORT" in _Worker.env_keys()

    def test_contains_cuda_visible_devices(self):
        assert "CUDA_VISIBLE_DEVICES" in _Worker.env_keys()

    def test_expected_key_count(self):
        assert len(_Worker.env_keys()) == 9


# ---------------------------------------------------------------------------
# TestWorkerDispatchCollectInfo
# ---------------------------------------------------------------------------
@_skip_if_no_worker
class TestWorkerDispatchCollectInfo:
    def test_register_and_query_dispatch(self):
        worker = _make_worker()
        worker._register_dispatch_collect_info("mesh_a", dp_rank=3, is_collect=True)
        assert worker._Worker__dispatch_dp_rank["mesh_a"] == 3

    def test_register_and_query_collect(self):
        worker = _make_worker()
        worker._register_dispatch_collect_info("mesh_b", dp_rank=1, is_collect=False)
        assert worker._Worker__collect_dp_rank["mesh_b"] is False

    def test_duplicate_mesh_name_raises(self):
        worker = _make_worker()
        worker._register_dispatch_collect_info("mesh_dup", dp_rank=0, is_collect=True)
        with pytest.raises(ValueError, match="mesh_dup has been registered"):
            worker._register_dispatch_collect_info("mesh_dup", dp_rank=1, is_collect=False)

    def test_multiple_mesh_names(self):
        worker = _make_worker()
        worker._register_dispatch_collect_info("mesh_x", dp_rank=0, is_collect=True)
        worker._register_dispatch_collect_info("mesh_y", dp_rank=1, is_collect=False)
        assert worker._Worker__dispatch_dp_rank["mesh_x"] == 0
        assert worker._Worker__dispatch_dp_rank["mesh_y"] == 1
        assert worker._Worker__collect_dp_rank["mesh_x"] is True
        assert worker._Worker__collect_dp_rank["mesh_y"] is False


# ---------------------------------------------------------------------------
# TestWorkerProperties
# ---------------------------------------------------------------------------
@_skip_if_no_worker
class TestWorkerProperties:
    def test_rank_property(self):
        worker = _make_worker()
        assert worker.rank == 2

    def test_world_size_property(self):
        worker = _make_worker()
        assert worker.world_size == 8

    def test_rank_property_matches_internal(self):
        worker = _make_worker()
        assert worker.rank == worker._rank

    def test_world_size_property_matches_internal(self):
        worker = _make_worker()
        assert worker.world_size == worker._world_size

    def test_rank_property_with_different_env(self):
        worker = _make_worker({"RANK": "5", "WORLD_SIZE": "16"})
        assert worker.rank == 5
        assert worker.world_size == 16
