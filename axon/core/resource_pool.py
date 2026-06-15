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
# Adapted from verl single_controller/base/worker_group.py (github.com/volcengine/verl), Apache-2.0.
from uuid import uuid4


class ResourcePool:
    """
    Manages a pool of resources across multiple nodes, tracking process counts and GPU allocations.

    The ResourcePool class is used to define the distribution of processes across nodes in a
    distributed computing environment. It tracks how many processes should run on each node
    and provides utilities for calculating global and local process information.

    Examples:
        >>> pool = ResourcePool(process_on_nodes=[2, 4, 2])
        >>> pool.world_size
        8
        >>> pool.store
        [2, 4, 2]
    """

    def __init__(
        self,
        process_on_nodes: list[int] | None = None,
        max_colocate_count: int = 10,
        name_prefix: str = None,
    ) -> None:
        """Initialize the ResourcePool with node processes and GPU configuration.

        Args:
            process_on_nodes (List[int], optional): List of process counts per node.
                Each integer represents the number of processes to run on that node.
                For example, [2, 4, 2] means 2 processes on node 0, 4 processes on node 1,
                and 2 processes on node 2. Defaults to empty list.
            max_colocate_count (int, optional): Maximum number of processes that can be
                colocated per GPU or resource unit. This is used for resource allocation
                and scheduling decisions. Defaults to 10.
            name_prefix (str, optional): Custom prefix for naming resources created from
                this pool. If not provided, a random 8-character UUID will be generated.
        """
        if process_on_nodes is None:
            process_on_nodes = []
        self._store = process_on_nodes
        self.max_colocate_count = max_colocate_count
        self.name_prefix = name_prefix if name_prefix else str(uuid4())[:8]

    @property
    def world_size(self):
        """Total number of processes across all nodes in the pool.

        Returns:
            int: Sum of all processes across all nodes.
        """
        return sum(self._store)

    @property
    def store(self):
        """Get the internal storage of process counts per node.

        Returns:
            list[int]: List of process counts for each node.
        """
        return self._store
