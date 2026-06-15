# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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

import functools
import threading
from collections.abc import Callable

import nvtx
import torch

from .config import NsightToolConfig
from .profile import DistProfiler, ProfilerConfig

# Cache NVTX Domain handles by name. Passing a string to nvtx.start_range /
# nvtx.annotate constructs a Domain on each call; marked_timer fires several
# times per step inside the training loop, so the per-call cost compounds.
_domain_cache: dict[str, "nvtx.Domain"] = {}
_domain_cache_lock = threading.Lock()


def _resolve_domain(domain):
    """Return the cached NVTX Domain for a given name, constructing it once."""
    if domain is None or not isinstance(domain, str):
        return domain
    cached = _domain_cache.get(domain)
    if cached is not None:
        return cached
    with _domain_cache_lock:
        cached = _domain_cache.get(domain)
        if cached is None:
            cached = nvtx.Domain(domain)
            _domain_cache[domain] = cached
        return cached


def mark_start_range(
    message: str | None = None,
    color: str | None = None,
    domain: str | None = None,
    category: str | None = None,
) -> None:
    """Start a mark range in the profiler.

    Args:
        message (str, optional):
            The message to be displayed in the profiler. Defaults to None.
        color (str, optional):
            The color of the range. Defaults to None.
        domain (str, optional):
            The domain of the range. Defaults to None.
        category (str, optional):
            The category of the range. Defaults to None.
    """
    return nvtx.start_range(message=message, color=color, domain=_resolve_domain(domain), category=category)


def mark_end_range(range_id: str) -> None:
    """End a mark range in the profiler.

    Args:
        range_id (str):
            The id of the mark range to end.
    """
    return nvtx.end_range(range_id)


def mark_annotate(
    message: str | None = None,
    color: str | None = None,
    domain: str | None = None,
    category: str | None = None,
) -> Callable:
    """Decorate a function to annotate a mark range along with the function life cycle.

    Args:
        message (str, optional):
            The message to be displayed in the profiler. Defaults to None.
        color (str, optional):
            The color of the range. Defaults to None.
        domain (str, optional):
            The domain of the range. Defaults to None.
        category (str, optional):
            The category of the range. Defaults to None.
    """

    def decorator(func):
        profile_message = message or func.__name__
        return nvtx.annotate(profile_message, color=color, domain=_resolve_domain(domain), category=category)(func)

    return decorator


class NsightSystemsProfiler(DistProfiler):
    """Nsight system profiler. Installed in a worker to control the Nsight system profiler."""

    def __init__(self, rank: int, config: ProfilerConfig | None, tool_config: NsightToolConfig | None, **kwargs):
        """Initialize the NsightSystemsProfiler.

        Args:
            rank (int): The rank of the current process.
            config (Optional[ProfilerConfig]): Configuration for the profiler. If None, a default configuration is used.
        """
        # If no configuration is provided, create a default ProfilerConfig with an empty list of ranks
        if not config:
            config = ProfilerConfig(ranks=[])
        if not tool_config:
            assert not config.enable, "tool_config must be provided when profiler is enabled"
        self.enable = config.enable
        if not config.enable:
            return
        self.this_step: bool = False
        self.discrete: bool = tool_config.discrete
        self.this_rank: bool = False
        if config.all_ranks:
            self.this_rank = True
        elif config.ranks:
            self.this_rank = rank in config.ranks

    def start(self, **kwargs):
        if self.enable and self.this_rank:
            self.this_step = True
            if not self.discrete:
                torch.cuda.profiler.start()

    def stop(self):
        if self.enable and self.this_rank:
            self.this_step = False
            if not self.discrete:
                torch.cuda.profiler.stop()

    def annotate(
        self,
        message: str | None = None,
        color: str | None = None,
        domain: str | None = None,
        category: str | None = None,
        **kwargs_outer,
    ) -> Callable:
        """Decorate a Worker member function to profile the current rank in the current training step.

        Requires the target function to be a member function of a Worker, which has a member field `profiler` with
        NightSystemsProfiler type.

        Args:
            message (str, optional):
                The message to be displayed in the profiler. Defaults to None.
            color (str, optional):
                The color of the range. Defaults to None.
            domain (str, optional):
                The domain of the range. Defaults to None.
            category (str, optional):
                The category of the range. Defaults to None.
        """

        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs_inner):
                if not self.enable:
                    return func(*args, **kwargs_inner)

                profile_name = message or func.__name__

                if self.this_step:
                    if self.discrete:
                        torch.cuda.profiler.start()
                    mark_range = mark_start_range(message=profile_name, color=color, domain=domain, category=category)

                result = func(*args, **kwargs_inner)

                if self.this_step:
                    mark_end_range(mark_range)
                    if self.discrete:
                        torch.cuda.profiler.stop()

                return result

            return wrapper

        return decorator
