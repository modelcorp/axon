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
"""Fused worker implementation for Axon.

This module provides utilities for fusing multiple worker classes into a single Ray actor,
enabling efficient colocation of different worker types (e.g., actor, critic) within the
same Ray process on GPU. This reduces inter-process communication overhead and enables direct
access between colocated workers.
"""

import inspect

import ray

from axon.controller.decorator import MAGIC_ATTR, Dispatch, collect, dispatch

from .class_init import RayActorWithInitArgs


def fuse_worker_cls(class_dict: dict[str, RayActorWithInitArgs]):
    """Create a Ray actor class that holds multiple workers.

    This function takes a dictionary of worker classes and creates a single fused Ray actor
    that contains instances of all the workers. The fused actor automatically routes method
    calls to the appropriate inner worker based on method prefixes or special routing rules.

    Args:
        class_dict: Dictionary mapping role names (e.g., "actor", "critic") to
                   RayActorWithInitArgs instances containing the worker class and init args.

    Returns:
        RayActorWithInitArgs: A Ray actor class that contains all the fused workers.

    Example:
        >>> actor_cls = RayActorWithInitArgs(cls=TrainerWorker, args=(), kwargs={})
        >>> critic_cls = RayActorWithInitArgs(cls=CriticWorker, args=(), kwargs={})
        >>> fused_cls = fuse_worker_cls({"actor": actor_cls, "critic": critic_cls})
        >>> # The fused actor will have methods like actor_init_model(), critic_init_model()
    """
    from axon.core.worker import Worker

    cls_info = {str(k): (v.cls.__ray_actor_class__, v.args, v.kwargs) for k, v in class_dict.items()}
    first_cls = next(iter(cls_info.values()))[0]
    base = next((c for c in first_cls.__mro__ if c.__name__ == "Worker"), Worker)

    class FusedWorker(base):
        """Dynamically created class that holds multiple worker instances."""

        def __init__(self):
            """Initialize the fused worker with all inner workers."""
            super().__init__()
            # Create instances of all worker classes
            self._w = {r: cls(*a, **kw) for r, (cls, a, kw) in cls_info.items()}
            # Enable inter-worker access: worker._colocated["critic"] accesses sibling
            for worker in self._w.values():
                worker._colocated = self._w

        def _query_dispatch_info(self, mesh_name):
            raise RuntimeError(
                f"_query_dispatch_info('{mesh_name}') must not be called directly on a FusedWorker. "
                "Use the role-prefixed version (e.g. actor__query_dispatch_info) via RoleProxy instead."
            )

        def _query_collect_info(self, mesh_name):
            raise RuntimeError(
                f"_query_collect_info('{mesh_name}') must not be called directly on a FusedWorker. "
                "Use the role-prefixed version (e.g. actor__query_collect_info) via RoleProxy instead."
            )

    def _make_role_scoped_dispatch(role, mesh_name):
        """Create role-scoped dispatch/collect functions for mesh-based dispatch.

        In a fused worker, multiple roles may share the same mesh_name (e.g., both
        actor and ref use "trainer"). The standard dispatch function calls
        ``_query_dispatch_info(mesh_name)`` which can't distinguish between them.

        This creates dispatch/collect functions that call the role-prefixed query
        methods instead (e.g., ``actor__query_dispatch_info``), so each role gets
        its own dispatch info even when the mesh_name is the same.
        """
        cache_key = f"{role}/{mesh_name}"
        query_dispatch_name = f"{role}__query_dispatch_info"
        query_collect_name = f"{role}__query_collect_info"

        def dispatch_fn(worker_group, *args, **kwargs):
            from axon.controller.worker_group import WorkerGroup

            assert isinstance(worker_group, WorkerGroup)
            if cache_key not in worker_group._dispatch_info:
                query_fn = getattr(worker_group, query_dispatch_name)
                worker_group._dispatch_info[cache_key] = query_fn(mesh_name)
                assert len(worker_group._dispatch_info[cache_key]) == worker_group.world_size
            rank_mapping = worker_group._dispatch_info[cache_key]
            use_ray = not getattr(worker_group, "_ray_collective_initialized", False)
            return dispatch(worker_group, *args, rank_mapping=rank_mapping, use_ray_object_store=use_ray, **kwargs)

        def collect_fn(worker_group, output):
            from axon.controller.worker_group import WorkerGroup

            assert isinstance(worker_group, WorkerGroup)
            assert cache_key in worker_group._dispatch_info
            if cache_key not in worker_group._collect_info:
                query_fn = getattr(worker_group, query_collect_name)
                worker_group._collect_info[cache_key] = query_fn(mesh_name)
                assert len(worker_group._collect_info[cache_key]) == worker_group.world_size
            collect_mask = worker_group._collect_info[cache_key]
            return collect(worker_group, output, collect_mask=collect_mask, concat=True)

        return {"dispatch_fn": dispatch_fn, "collect_fn": collect_fn, "mesh_name": mesh_name}

    def _make_wrapper(role, name, method):
        """Create wrapper that forwards to inner worker, handling sync/async/gen.

        This function creates a wrapper method that forwards calls to the appropriate
        inner worker while preserving the original method's async/generator semantics.

        For mesh-based dispatch, the dispatch/collect functions are replaced with
        role-scoped versions that route dispatch queries through the prefixed path
        (e.g., ``actor__query_dispatch_info``), allowing multiple roles to share the
        same mesh_name without collision.

        Args:
            role: The role name (e.g., "actor", "critic")
            name: The method name to wrap
            method: The original method from the worker class

        Returns:
            Wrapped function that forwards to the inner worker with correct semantics
        """
        attrs = getattr(method, MAGIC_ATTR)
        is_coro = inspect.iscoroutinefunction(method)
        is_gen = inspect.isgeneratorfunction(method)
        is_async_gen = inspect.isasyncgenfunction(method)

        if is_async_gen:

            async def f(self, *a, **kw):
                async for item in getattr(self._w[role], name)(*a, **kw):
                    yield item
        elif is_coro:

            async def f(self, *a, **kw):
                return await getattr(self._w[role], name)(*a, **kw)
        elif is_gen:

            def f(self, *a, **kw):
                yield from getattr(self._w[role], name)(*a, **kw)
        else:

            def f(self, *a, **kw):
                return getattr(self._w[role], name)(*a, **kw)

        # For mesh-based dispatch, replace with role-scoped dispatch/collect
        # so that each role queries its own worker's dispatch info.
        new_attrs = dict(attrs)
        dispatch_mode = new_attrs.get("dispatch_mode")
        if isinstance(dispatch_mode, dict) and "mesh_name" in dispatch_mode:
            new_attrs["dispatch_mode"] = _make_role_scoped_dispatch(role, dispatch_mode["mesh_name"])
        setattr(f, MAGIC_ATTR, new_attrs)
        return f

    # Bind methods for each role to the FusedWorker class
    for role, (cls, _, _) in cls_info.items():
        for name in dir(cls):
            method = getattr(cls, name)
            # Only process methods that have dispatch metadata (MAGIC_ATTR)
            if not (callable(method) and hasattr(method, MAGIC_ATTR)):
                continue
            attrs = getattr(method, MAGIC_ATTR)

            # Handle DIRECT_SAMPLER_METHOD (no prefix) vs normal (prefixed)
            # DIRECT_SAMPLER_METHOD methods are called directly without role prefix
            if attrs.get("dispatch_mode") == Dispatch.DIRECT_SAMPLER_METHOD:
                setattr(FusedWorker, name, _make_wrapper(role, name, method))
            else:
                # Normal methods get prefixed with role name (e.g., actor_init_model)
                prefixed = f"{role}_{name}"
                print(f"Binding FusedWorker.{prefixed} to {cls.__name__}.{name}.")
                setattr(FusedWorker, prefixed, _make_wrapper(role, name, method))

    return RayActorWithInitArgs(cls=ray.remote(max_concurrency=2048)(FusedWorker))


class RoleProxy:
    """Thin view that strips prefix: view.init_model() -> wg.actor_init_model().

    This class provides a convenient interface for accessing methods on a specific role
    within a fused worker group. It automatically handles method name prefixing and
    special routing for certain methods.

    The proxy allows you to call methods on a specific role without having to manually
    add the role prefix. For example, if you have an "actor" role, calling
    proxy.init_model() will automatically route to wg.actor_init_model().
    """

    __slots__ = ("_wg", "_prefix")

    def __init__(self, wg, role: str):
        """Initialize the role proxy.

        Args:
            wg: The RayWorkerGroup containing the fused workers
            role: The role name (e.g., "actor", "critic") to proxy for
        """
        object.__setattr__(self, "_wg", wg)
        object.__setattr__(self, "_prefix", f"{role}_")

    def __getattr__(self, name):
        """Get attribute from the worker group, handling prefixing automatically.

        This method implements the core proxy logic:
        1. Try the prefixed version first (role-specific methods)
        2. Fall back to unprefixed version (WorkerGroup methods)

        Args:
            name: The attribute/method name to access

        Returns:
            The requested attribute/method from the worker group
        """
        # Use object.__getattribute__ to avoid recursion during pickle
        wg = object.__getattribute__(self, "_wg")
        prefix = object.__getattribute__(self, "_prefix")
        # Try prefixed first (worker methods), fall back to unprefixed (WorkerGroup methods)
        prefixed = prefix + name
        if hasattr(wg, prefixed):
            return getattr(wg, prefixed)
        return getattr(wg, name)

    def __setattr__(self, name, value):
        """Set attribute on the underlying worker group.

        Args:
            name: The attribute name to set
            value: The value to set
        """
        wg = object.__getattribute__(self, "_wg")
        setattr(wg, name, value)

    # Support pickle serialization
    def __reduce__(self):
        """Support for pickle serialization.

        Returns:
            Tuple containing the class and arguments needed to reconstruct this proxy
        """
        wg = object.__getattribute__(self, "_wg")
        prefix = object.__getattribute__(self, "_prefix")
        role = prefix.rstrip("_")
        return (RoleProxy, (wg, role))
