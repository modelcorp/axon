# Copyright 2025 Model AI Corp.
# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
#
# Adapted from verl single_controller/ray/base.py — RayClassWithInitArgs (github.com/volcengine/verl), Apache-2.0.
from typing import Any

from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from axon.core import ClassWithInitArgs


class RayActorWithInitArgs(ClassWithInitArgs):
    """A wrapper class for Ray actors with initialization arguments.

    This class extends ClassWithInitArgs to provide additional functionality for
    configuring and creating Ray actors with specific resource requirements and
    scheduling strategies.
    """

    def __init__(self, cls, *args, max_concurrency: int = 2048, **kwargs) -> None:
        super().__init__(cls, *args, **kwargs)
        self._options = {}
        self._max_concurrency = max_concurrency

    def update_options(self, options: dict):
        """Update the Ray actor creation options.

        Args:
            options: Dictionary of options to update
        """
        self._options.update(options)

    def __call__(
        self,
        placement_group,
        placement_group_bundle_idx,
        resource_dict: dict[str, Any],
    ) -> Any:
        """Create and return a Ray actor with the configured options.

        Args:
            placement_group: Ray placement group for scheduling
            placement_group_bundle_idx: Index of the bundle in the placement group
            resource_dict: Dictionary of resources to allocate

        Returns:
            A Ray actor handle with the configured options
        """
        options = {
            "scheduling_strategy": PlacementGroupSchedulingStrategy(
                placement_group=placement_group, placement_group_bundle_index=placement_group_bundle_idx
            )
        }
        options.update(self._options)

        resource_dict = dict(resource_dict)
        num_gpus = resource_dict.pop("GPU", None)
        num_cpus = resource_dict.pop("CPU", None)

        # Ray requires CPU and GPU resources to be specified separately.
        if num_gpus:
            options.update({"num_gpus": num_gpus})
        if num_cpus:
            options.update({"num_cpus": num_cpus})

        # Specify other resources, such as NPU, etc.
        options.update({"resources": resource_dict})
        # Ensure Ray actor can be called concurrently.
        if "max_concurrency" not in options:
            options.update({"max_concurrency": self._max_concurrency})
        # Launch Ray actor with the configured options
        return self.cls.options(**options).remote(*self.args, **self.kwargs)
