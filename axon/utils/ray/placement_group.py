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

import ray
from ray.util.placement_group import PlacementGroup, placement_group

from axon.core import ResourcePool

########################################################
# Ray Placement Groups (for RayWorkerGroup)            #
########################################################


def create_placement_groups(
    resource_pool: ResourcePool, strategy: str = "STRICT_PACK", name: str | None = None, device_name: str = "cuda"
) -> list[PlacementGroup]:
    """Create and return Ray placement groups for the resource pool.

    Args:
        strategy: Ray placement strategy (e.g., "STRICT_PACK", "PACK", "SPREAD")
        name: Optional custom name for placement groups
        device_name: Type of device to allocate ("cuda", "npu", None)

    Returns:
        List of Ray placement groups, sorted by node IP
    """
    pg_name_prefix = (
        name
        if name
        else f"{resource_pool.name_prefix}_pg_[{','.join([str(count) for count in resource_pool._store])}]:"
    )

    # Normalize device name for Ray resource allocation
    if device_name == "npu":
        device_name = "NPU"
    elif device_name == "cuda":
        device_name = "GPU"

    # Create resource bundle specification
    bundle = {"CPU": resource_pool.max_colocate_count}
    if device_name is not None:
        bundle[device_name] = 1

    # Create placement group scheme: one PG per node with bundles for each process
    pg_scheme = [[bundle.copy() for _ in range(process_count)] for process_count in resource_pool._store]

    # Create placement groups
    pgs = [
        placement_group(
            bundles=bundles,
            strategy=strategy,
            name=pg_name_prefix + str(idx),
        )
        for idx, bundles in enumerate(pg_scheme)
    ]

    # Wait for all placement groups to be ready
    ray.get([pg.ready() for pg in pgs])
    return pgs, pg_scheme


def sort_placement_group_by_node_ip(pgs: list[PlacementGroup]) -> list[PlacementGroup]:
    """Sort placement groups by node IP address.

    All bundles in a single placement group should be on the same node.
    FSDPStateManager saves sharded model states and optimizer states in local storage,
    which requires RANK to be consistent across nodes when resuming from state.

    With this function, if there's only one resource pool and there's no node change,
    RANK should be consistent across nodes in multiple ray jobs, even if the whole
    ray cluster is restarted.

    Args:
        pgs: List of placement groups to sort

    Returns:
        List of placement groups sorted by node IP address
    """
    node_ip = {node["NodeID"]: node["NodeManagerAddress"] for node in ray.nodes()}
    pg_ip = {}
    for pg in pgs:
        specs = ray._private.state.state.placement_group_table(pg.id)
        # all bundles should be on the same node
        node_id = specs["bundles_to_node_id"][0]
        pg_ip[pg.id] = node_ip[node_id]
    return sorted(pgs, key=lambda pg: pg_ip[pg.id])
