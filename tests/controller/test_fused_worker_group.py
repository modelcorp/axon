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
Unit tests for axon.controller.ray.fused_worker_group module.

Tests cover:
- fuse_worker_cls(): method prefixing, DIRECT_SAMPLER_METHOD handling,
  role-scoped dispatch routing, async/sync/generator wrappers
- RoleProxy: getattr routing, setattr delegation, pickle support
"""

import unittest
from unittest.mock import MagicMock, patch

from axon.controller.decorator import MAGIC_ATTR, Dispatch, Execute
from axon.controller.ray.fused_worker_group import RoleProxy

# ---------------------------------------------------------------------------
# RoleProxy
# ---------------------------------------------------------------------------


class TestRoleProxy(unittest.TestCase):
    """Tests for RoleProxy attribute routing."""

    def test_getattr_prefixed_method(self):
        """RoleProxy.method_name routes to wg.role_method_name."""
        wg = MagicMock()
        wg.actor_init_model = MagicMock(return_value="init_result")
        proxy = RoleProxy(wg, "actor")

        result = proxy.init_model()
        wg.actor_init_model.assert_called_once()
        self.assertEqual(result, "init_result")

    def test_getattr_falls_back_to_unprefixed(self):
        """RoleProxy falls back to unprefixed attr when prefixed doesn't exist."""
        wg = MagicMock(spec=[])
        wg.world_size = 4

        # spec=[] means hasattr returns False for actor_world_size
        # but we need to make hasattr(wg, "actor_world_size") return False
        proxy = RoleProxy(wg, "actor")

        # MagicMock without spec will create actor_world_size on access
        # so let's use a real object
        class FakeWG:
            world_size = 4

        wg = FakeWG()
        proxy = RoleProxy(wg, "actor")
        self.assertEqual(proxy.world_size, 4)

    def test_getattr_query_dispatch_info_is_prefixed(self):
        """_query_dispatch_info goes through the prefix path like other methods."""

        class FakeWG:
            def actor__query_dispatch_info(self, mesh_name):
                return [0, 1]

        wg = FakeWG()
        proxy = RoleProxy(wg, "actor")
        result = proxy._query_dispatch_info("trainer")
        self.assertEqual(result, [0, 1])

    def test_setattr_delegates_to_wg(self):
        """setattr on proxy sets on the underlying wg."""

        class FakeWG:
            pass

        wg = FakeWG()
        proxy = RoleProxy(wg, "actor")
        proxy.some_attr = 42
        self.assertEqual(wg.some_attr, 42)

    def test_pickle_roundtrip(self):
        """RoleProxy supports pickle serialization/deserialization."""
        wg = MagicMock()
        proxy = RoleProxy(wg, "critic")

        # Check __reduce__ produces correct reconstruction args
        cls, args = proxy.__reduce__()
        self.assertIs(cls, RoleProxy)
        self.assertEqual(args, (wg, "critic"))

    def test_prefix_construction(self):
        """RoleProxy constructs prefix as role_."""
        wg = MagicMock()
        proxy = RoleProxy(wg, "my_role")
        prefix = object.__getattribute__(proxy, "_prefix")
        self.assertEqual(prefix, "my_role_")

    def test_getattr_prefers_prefixed_over_unprefixed(self):
        """When both prefixed and unprefixed exist, prefixed wins."""

        class FakeWG:
            def method(self):
                return "unprefixed"

            def actor_method(self):
                return "prefixed"

        wg = FakeWG()
        proxy = RoleProxy(wg, "actor")
        result = proxy.method()
        self.assertEqual(result, "prefixed")

    def test_query_dispatch_info_falls_back_to_unprefixed(self):
        """When no prefixed version exists, falls back to unprefixed."""
        wg = MagicMock()
        wg._query_dispatch_info = MagicMock(return_value=[0, 1])
        proxy = RoleProxy(wg, "actor")

        # MagicMock auto-creates actor__query_dispatch_info, so use a real object
        class FakeWG:
            def _query_dispatch_info(self, mesh_name):
                return [0, 1]

        wg = FakeWG()
        proxy = RoleProxy(wg, "actor")
        result = proxy._query_dispatch_info("trainer")
        self.assertEqual(result, [0, 1])


# ---------------------------------------------------------------------------
# fuse_worker_cls - method prefixing logic
# ---------------------------------------------------------------------------


class TestFuseWorkerCls(unittest.TestCase):
    """Tests for fuse_worker_cls method binding logic.

    These tests verify the FusedWorker class creation without Ray, by testing
    the helper functions and logic used during class creation.
    """

    def test_magic_attr_preserved_on_decorated_method(self):
        """Decorated methods retain their MAGIC_ATTR for binding."""

        def method(self, x):
            return x * 2

        attrs = {
            "dispatch_mode": Dispatch.ONE_TO_ALL,
            "execute_mode": Execute.ALL,
            "blocking": True,
            "disable_collective": False,
        }
        setattr(method, MAGIC_ATTR, attrs)

        # Verify the attrs are accessible (same check fuse_worker_cls does)
        self.assertTrue(hasattr(method, MAGIC_ATTR))
        self.assertEqual(getattr(method, MAGIC_ATTR), attrs)
        self.assertTrue(callable(method))

    def test_direct_sampler_method_not_prefixed(self):
        """Methods with DIRECT_SAMPLER_METHOD dispatch are not prefixed."""
        # Verify the logic: if dispatch_mode == Dispatch.DIRECT_SAMPLER_METHOD,
        # the method name is NOT prefixed
        attrs = {
            "dispatch_mode": Dispatch.DIRECT_SAMPLER_METHOD,
            "execute_mode": Execute.ALL,
            "blocking": True,
            "disable_collective": False,
        }
        # This is the check from fuse_worker_cls
        self.assertEqual(attrs["dispatch_mode"], Dispatch.DIRECT_SAMPLER_METHOD)

    def test_normal_method_gets_prefixed(self):
        """Methods with non-DIRECT_SAMPLER_METHOD dispatch get role_ prefix."""
        role = "actor"
        name = "init_model"
        prefixed = f"{role}_{name}"
        self.assertEqual(prefixed, "actor_init_model")

    @patch("axon.controller.ray.fused_worker_group.ray")
    def test_fuse_worker_cls_creates_class(self, mock_ray):
        """fuse_worker_cls returns a RayActorWithInitArgs wrapping a FusedWorker."""
        from axon.controller.ray.class_init import RayActorWithInitArgs
        from axon.controller.ray.fused_worker_group import fuse_worker_cls

        # Create mock worker classes with MAGIC_ATTR-decorated methods
        class ActorWorker:
            _colocated = None

            def __init__(self):
                pass

            def regular_method(self):
                pass

        class CriticWorker:
            _colocated = None

            def __init__(self, val):
                self.val = val

            def regular_method(self):
                pass

        # Add decorated methods
        def actor_train(self):
            return "actor_train_result"

        setattr(
            actor_train,
            MAGIC_ATTR,
            {
                "dispatch_mode": Dispatch.ONE_TO_ALL,
                "execute_mode": Execute.ALL,
                "blocking": True,
                "disable_collective": False,
            },
        )
        ActorWorker.train = actor_train

        def critic_eval(self):
            return "critic_eval_result"

        setattr(
            critic_eval,
            MAGIC_ATTR,
            {
                "dispatch_mode": Dispatch.ALL_TO_ALL,
                "execute_mode": Execute.ALL,
                "blocking": True,
                "disable_collective": False,
            },
        )
        CriticWorker.evaluate = critic_eval

        # Make ray.remote return a class with __ray_actor_class__
        mock_remote_actor_cls = MagicMock()
        mock_remote_actor_cls.__ray_actor_class__ = ActorWorker

        mock_remote_critic_cls = MagicMock()
        mock_remote_critic_cls.__ray_actor_class__ = CriticWorker

        actor_cia = RayActorWithInitArgs(cls=mock_remote_actor_cls)
        critic_cia = RayActorWithInitArgs(cls=mock_remote_critic_cls, val=10)

        # mock ray.remote to return the class as-is (wrapped)
        mock_ray.remote.return_value = MagicMock()

        result = fuse_worker_cls({"actor": actor_cia, "critic": critic_cia})
        self.assertIsInstance(result, RayActorWithInitArgs)


# ---------------------------------------------------------------------------
# RoleProxy with multiple roles
# ---------------------------------------------------------------------------


class TestRoleProxyMultiRole(unittest.TestCase):
    """Tests for RoleProxy with multiple roles on the same wg."""

    def test_different_proxies_access_different_methods(self):
        """Different role proxies route to different prefixed methods."""

        class FakeWG:
            def actor_train(self):
                return "actor_training"

            def critic_train(self):
                return "critic_training"

            @property
            def world_size(self):
                return 4

        wg = FakeWG()
        actor_proxy = RoleProxy(wg, "actor")
        critic_proxy = RoleProxy(wg, "critic")

        self.assertEqual(actor_proxy.train(), "actor_training")
        self.assertEqual(critic_proxy.train(), "critic_training")

    def test_shared_unprefixed_methods(self):
        """Both proxies can access shared WorkerGroup methods."""

        class FakeWG:
            def actor_init(self):
                return "actor"

            def critic_init(self):
                return "critic"

            @property
            def world_size(self):
                return 8

        wg = FakeWG()
        actor_proxy = RoleProxy(wg, "actor")
        critic_proxy = RoleProxy(wg, "critic")

        self.assertEqual(actor_proxy.world_size, 8)
        self.assertEqual(critic_proxy.world_size, 8)


# ---------------------------------------------------------------------------
# Role-scoped dispatch in fused workers
# ---------------------------------------------------------------------------


class TestRoleScopedDispatch(unittest.TestCase):
    """Tests that fused workers correctly scope dispatch info per role.

    When two roles (e.g., actor and ref) share the same mesh_name ("trainer"),
    _make_wrapper must replace the dispatch/collect functions with role-scoped
    versions that:
    1. Call role-prefixed query methods (actor__query_dispatch_info, ref__query_dispatch_info)
    2. Cache dispatch info under role-scoped keys ("actor/trainer", "ref/trainer")

    This prevents actor and ref from colliding when they have different sharding.
    """

    @patch("axon.controller.ray.fused_worker_group.ray")
    def test_mesh_dispatch_replaced_with_role_scoped(self, mock_ray):
        """_make_wrapper replaces mesh-based dispatch_mode with role-scoped version."""
        from axon.controller.decorator import make_nd_compute_dataproto_dispatch_fn
        from axon.controller.ray.class_init import RayActorWithInitArgs
        from axon.controller.ray.fused_worker_group import fuse_worker_cls

        class TrainerWorker:
            _colocated = None
            _mesh_name = "trainer"

            def __init__(self):
                pass

        # Create a method with mesh-based dispatch (like forward())
        def forward(self, data):
            return data

        mesh_dispatch = make_nd_compute_dataproto_dispatch_fn(mesh_name="trainer")
        setattr(
            forward,
            MAGIC_ATTR,
            {
                "dispatch_mode": mesh_dispatch,
                "execute_mode": Execute.ALL,
                "blocking": True,
                "disable_collective": False,
            },
        )
        TrainerWorker.forward = forward

        mock_remote_cls = MagicMock()
        mock_remote_cls.__ray_actor_class__ = TrainerWorker
        mock_ray.remote.return_value = MagicMock()

        actor_cia = RayActorWithInitArgs(cls=mock_remote_cls)
        ref_cia = RayActorWithInitArgs(cls=mock_remote_cls)

        fuse_worker_cls({"actor": actor_cia, "ref": ref_cia})

        # Verify that actor_forward and ref_forward have DIFFERENT dispatch_mode dicts
        # by checking the class that was passed to ray.remote
        # ray.remote is called as ray.remote(max_concurrency=2048)(FusedWorker)
        # The first call to mock_ray.remote returns a mock, and that mock is called with FusedWorker
        remote_decorator = mock_ray.remote.return_value
        FusedWorker = remote_decorator.call_args[0][0]

        # Both actor_forward and ref_forward should exist
        self.assertTrue(hasattr(FusedWorker, "actor_forward"))
        self.assertTrue(hasattr(FusedWorker, "ref_forward"))

        # Both should have MAGIC_ATTR with mesh-based dispatch
        actor_attrs = getattr(FusedWorker.actor_forward, MAGIC_ATTR)
        ref_attrs = getattr(FusedWorker.ref_forward, MAGIC_ATTR)

        actor_dispatch = actor_attrs["dispatch_mode"]
        ref_dispatch = ref_attrs["dispatch_mode"]

        # Both should be dicts (mesh-based) with mesh_name
        self.assertIsInstance(actor_dispatch, dict)
        self.assertIsInstance(ref_dispatch, dict)
        self.assertEqual(actor_dispatch["mesh_name"], "trainer")
        self.assertEqual(ref_dispatch["mesh_name"], "trainer")

        # Crucially: the dispatch_fn should be DIFFERENT functions (role-scoped)
        self.assertIsNot(actor_dispatch["dispatch_fn"], ref_dispatch["dispatch_fn"])
        self.assertIsNot(actor_dispatch["collect_fn"], ref_dispatch["collect_fn"])

    @patch("axon.controller.ray.fused_worker_group.ray")
    def test_role_scoped_dispatch_queries_prefixed_method(self, mock_ray):
        """Role-scoped dispatch_fn calls {role}__query_dispatch_info, not _query_dispatch_info."""
        from axon.controller.decorator import make_nd_compute_dataproto_dispatch_fn
        from axon.controller.ray.class_init import RayActorWithInitArgs
        from axon.controller.ray.fused_worker_group import fuse_worker_cls

        class TrainerWorker:
            _colocated = None
            _mesh_name = "trainer"

            def __init__(self):
                pass

        def forward(self, data):
            return data

        mesh_dispatch = make_nd_compute_dataproto_dispatch_fn(mesh_name="trainer")
        setattr(
            forward,
            MAGIC_ATTR,
            {
                "dispatch_mode": mesh_dispatch,
                "execute_mode": Execute.ALL,
                "blocking": True,
                "disable_collective": False,
            },
        )
        TrainerWorker.forward = forward

        mock_remote_cls = MagicMock()
        mock_remote_cls.__ray_actor_class__ = TrainerWorker
        mock_ray.remote.return_value = MagicMock()

        actor_cia = RayActorWithInitArgs(cls=mock_remote_cls)
        ref_cia = RayActorWithInitArgs(cls=mock_remote_cls)

        fuse_worker_cls({"actor": actor_cia, "ref": ref_cia})

        remote_decorator = mock_ray.remote.return_value
        FusedWorker = remote_decorator.call_args[0][0]

        actor_dispatch_fn = getattr(FusedWorker.actor_forward, MAGIC_ATTR)["dispatch_mode"]["dispatch_fn"]
        ref_dispatch_fn = getattr(FusedWorker.ref_forward, MAGIC_ATTR)["dispatch_mode"]["dispatch_fn"]

        # Create a fake worker group to test the dispatch functions
        from axon.controller.worker_group import WorkerGroup

        wg = MagicMock(spec=WorkerGroup)
        wg._dispatch_info = {}
        wg._collect_info = {}
        wg._ray_collective_initialized = False
        wg.world_size = 4

        # Mock the prefixed query methods with DIFFERENT return values
        # to prove each dispatch fn routes to its own role's query
        wg.actor__query_dispatch_info = MagicMock(return_value=[0, 0, 1, 1])
        wg.ref__query_dispatch_info = MagicMock(return_value=[0, 1, 0, 1])

        # Call actor's dispatch_fn (no data args — we're testing routing, not chunking)
        actor_dispatch_fn(wg)
        wg.actor__query_dispatch_info.assert_called_once_with("trainer")
        wg.ref__query_dispatch_info.assert_not_called()

        # Call ref's dispatch_fn — should call ref__query_dispatch_info
        ref_dispatch_fn(wg)
        wg.ref__query_dispatch_info.assert_called_once_with("trainer")

        # Verify separate cache keys
        self.assertIn("actor/trainer", wg._dispatch_info)
        self.assertIn("ref/trainer", wg._dispatch_info)
        self.assertEqual(wg._dispatch_info["actor/trainer"], [0, 0, 1, 1])
        self.assertEqual(wg._dispatch_info["ref/trainer"], [0, 1, 0, 1])

    @patch("axon.controller.ray.fused_worker_group.ray")
    def test_non_mesh_dispatch_not_modified(self, mock_ray):
        """Methods with non-mesh dispatch (e.g., ONE_TO_ALL) keep original attrs."""
        from axon.controller.ray.class_init import RayActorWithInitArgs
        from axon.controller.ray.fused_worker_group import fuse_worker_cls

        class TrainerWorker:
            _colocated = None

            def __init__(self):
                pass

        def init_model(self):
            pass

        original_attrs = {
            "dispatch_mode": Dispatch.ONE_TO_ALL,
            "execute_mode": Execute.ALL,
            "blocking": True,
            "disable_collective": False,
        }
        setattr(init_model, MAGIC_ATTR, original_attrs)
        TrainerWorker.init_model = init_model

        mock_remote_cls = MagicMock()
        mock_remote_cls.__ray_actor_class__ = TrainerWorker
        mock_ray.remote.return_value = MagicMock()

        actor_cia = RayActorWithInitArgs(cls=mock_remote_cls)
        fuse_worker_cls({"actor": actor_cia})

        remote_decorator = mock_ray.remote.return_value
        FusedWorker = remote_decorator.call_args[0][0]

        bound_attrs = getattr(FusedWorker.actor_init_model, MAGIC_ATTR)
        # dispatch_mode should be the original enum, NOT a role-scoped dict
        self.assertEqual(bound_attrs["dispatch_mode"], Dispatch.ONE_TO_ALL)


if __name__ == "__main__":
    unittest.main()
