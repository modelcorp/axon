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
from collections import defaultdict
from dataclasses import dataclass, field

import torch


@dataclass
class ParameterMetadata:
    param_name: str
    original_param_name: str
    param_shape: tuple[int, ...]
    full_param_shape: tuple[int, ...]
    param_dtype: torch.dtype
    # Dimension of where the parameter was split (-1 for replicas)
    split_dim: int = -1
    # For TP, where a parameter is split across multiple ranks.
    split_idx: int = 0
    # Expert idx
    expert_idx: int = -1
    # Metadata
    metadata: dict[str, any] = field(default_factory=dict)

    def __hash__(self):
        return hash((self.param_name, self.split_idx))


@dataclass
class FusedParameterMetadata:
    param_name: str
    param_shape: tuple[int, ...]
    param_dtype: torch.dtype
    full_param_shape: tuple[int, ...]
    split_dim: int = -1
    split_idx: int = 0
    params: list[ParameterMetadata] = field(default_factory=list)

    def __hash__(self):
        return hash((self.param_name, self.split_idx))


@dataclass
class RankMapping:
    rank: int
    params: list[ParameterMetadata] = field(default_factory=list)


class RoutingTable:
    """
    Creates a load-balanced routing table for parameter synchronization between
    actor and sampler workers in distributed training.

    The routing table maps parameter keys from actor ranks to sampler ranks,
    ensuring that no single rank becomes a bottleneck by balancing the transfer load.
    """

    def __init__(self, actor_rank_mapping: list[RankMapping], sampler_rank_mapping: list[RankMapping]):
        """
        Initialize the routing table with parameter keys from actor and sampler workers.

        Args:
            actor_rank_mapping: List of RankMapping from actor ranks
            sampler_rank_mapping: List of RankMapping from sampler ranks
        """
        # Filter out empty params (ranks that don't hold parameters)
        self.actor_rank_mapping = [x for x in actor_rank_mapping if x.params]
        self.sampler_rank_mapping = [x for x in sampler_rank_mapping if x.params]

        # Build index structures
        self.actor_param_to_rank = self._build_param_to_rank_map(self.actor_rank_mapping)
        self.sampler_param_to_rank = self._build_param_to_rank_map(self.sampler_rank_mapping)

        # Build the routing map
        self.routing_map = self._build_routing_map()

    def _build_param_to_rank_map(
        self, param_keys_list: list[RankMapping]
    ) -> dict[tuple[str, int], list[tuple[int, any]]]:
        """
        Build a mapping from (param_name, split_idx) to list of (rank, metadata) tuples.
        This handles both ParameterMetadata and FusedParameterMetadata.

        Args:
            param_keys_list: List of RankMapping objects

        Returns:
            Dict mapping (param_name, split_idx) to list of (rank, metadata) tuples
        """
        param_to_ranks = defaultdict(list)
        for rank_mapping in param_keys_list:
            rank = rank_mapping.rank
            for param_metadata in rank_mapping.params:
                key = (param_metadata.param_name, param_metadata.split_idx)
                param_to_ranks[key].append((rank, param_metadata))
        return dict(param_to_ranks)

    def _build_routing_map(self) -> dict[tuple[int, int], list[dict[str, any]]]:
        """
        Build a load-balanced routing map from actor ranks to sampler ranks.

        The routing map is structured as:
        {(actor_rank, sampler_rank): [list of {'actor': actor_meta, 'sampler': sampler_meta} dicts]}

        For each parameter that exists on multiple ranks (FSDP sharding),
        we create rank-to-rank mappings based on matching indices.

        Tensor Parallel (TP) handling:
        - One actor parameter shard may map to multiple sampler shards (actor TP < sampler TP)
        - Or multiple actor shards may map to one sampler shard (actor TP > sampler TP)

        Expert Tensor Parallel (ETP) handling:
        - Expert parameters use ETP for sharding instead of TP
        - ETP size may differ from TP size
        - Each expert parameter is treated independently

        Data Parallel (DP) handling:
        - Multiple actor ranks may hold the same parameter shard (DP replicas)
        - Each sampler rank receives from exactly ONE actor rank (no redundant transfers)
        - Round-robin assignment distributes load evenly across actor ranks

        Assumptions:
        - TP/ETP sizes between actor and sampler are multiples of each other
        - Matched weights have the same sharding dimensions (row-parallel with row-parallel, etc.)

        Returns:
            Dict mapping (actor_rank, sampler_rank) tuples to lists of {'actor': ..., 'sampler': ...} dicts
        """
        routing_map = defaultdict(list)

        # Track assignment counts for load balancing
        # Key: (actor_rank, sampler_rank), Value: number of parameters assigned
        assignment_counts = defaultdict(int)

        # Track total load per actor rank (O(1) lookup instead of O(R) recalculation)
        actor_total_loads = defaultdict(int)

        # Track which sampler params have been assigned to avoid duplicates
        # This is needed when actor_tp > sampler_tp (multiple actor shards map to one sampler shard)
        # Key: (sampler_rank, param_name, split_idx)
        assigned_sampler_params = set()

        # Get all parameter keys (param_name, split_idx)
        actor_keys = set(self.actor_param_to_rank.keys())
        sampler_keys = set(self.sampler_param_to_rank.keys())

        # Group by parameter name only (ignoring split_idx for now)
        actor_param_names = set(key[0] for key in actor_keys)
        sampler_param_names = set(key[0] for key in sampler_keys)
        common_param_names = actor_param_names & sampler_param_names

        print(f"[RoutingTable] Actor has {len(actor_param_names)} unique parameter names")
        print(f"[RoutingTable] Sampler has {len(sampler_param_names)} unique parameter names")

        if common_param_names != actor_param_names or common_param_names != sampler_param_names:
            actor_only = actor_param_names - sampler_param_names
            sampler_only = sampler_param_names - actor_param_names
            error_lines = [
                f"Parameter name mismatch! Actor: {len(actor_param_names)}, "
                f"Sampler: {len(sampler_param_names)}, Common: {len(common_param_names)}"
            ]
            for label, params in [("Actor-only", actor_only), ("Sampler-only", sampler_only)]:
                if params:
                    error_lines.append(f"  {label} ({len(params)}):")
                    for name in sorted(params)[:10]:
                        error_lines.append(f"    - {name}")
                    if len(params) > 10:
                        error_lines.append(f"    ... and {len(params) - 10} more")
            raise ValueError("\n".join(error_lines))

        if not common_param_names:
            raise ValueError("No common parameters found between actor and sampler workers!")

        # Pre-group keys by parameter name to avoid O(n²) lookups
        actor_keys_by_name = defaultdict(list)
        for name, idx in actor_keys:
            actor_keys_by_name[name].append((name, idx))

        sampler_keys_by_name = defaultdict(list)
        for name, idx in sampler_keys:
            sampler_keys_by_name[name].append((name, idx))

        # For each unique parameter name, handle TP mapping
        for param_name in sorted(common_param_names):
            # Get all keys for this parameter name (now O(1) lookup)
            actor_keys_for_param = actor_keys_by_name[param_name]
            sampler_keys_for_param = sampler_keys_by_name[param_name]

            # Sort by split_idx to ensure consistent ordering
            actor_keys_sorted = sorted(actor_keys_for_param, key=lambda x: x[1])
            sampler_keys_sorted = sorted(sampler_keys_for_param, key=lambda x: x[1])

            # Get TP/ETP sizes (number of shards for this parameter)
            actor_tp_size = len(actor_keys_sorted)
            sampler_tp_size = len(sampler_keys_sorted)

            # Map actor shards to sampler shards
            for actor_key in actor_keys_sorted:
                _, actor_split_idx = actor_key
                actor_rank_meta_list = self.actor_param_to_rank[actor_key]

                # Determine which sampler shards this actor shard should map to
                if actor_tp_size == sampler_tp_size:
                    # 1-to-1 mapping: same TP size
                    target_sampler_keys = [sampler_keys_sorted[actor_split_idx]]
                elif actor_tp_size < sampler_tp_size:
                    # One actor shard maps to multiple sampler shards
                    # E.g., actor TP=2, sampler TP=4: actor shard 0 -> sampler shards 0,1
                    ratio = sampler_tp_size // actor_tp_size
                    start_idx = actor_split_idx * ratio
                    end_idx = start_idx + ratio
                    target_sampler_keys = sampler_keys_sorted[start_idx:end_idx]
                else:
                    # Multiple actor shards map to one sampler shard
                    # E.g., actor TP=4, sampler TP=2: actor shards 0,1 -> sampler shard 0
                    ratio = actor_tp_size // sampler_tp_size
                    target_idx = actor_split_idx // ratio
                    target_sampler_keys = [sampler_keys_sorted[target_idx]]

                # Create mappings for sampler ranks to actor ranks
                # Handle Data Parallel (DP) replicas:
                # - actor_rank_meta_list contains all actor ranks with this shard (DP replicas)
                # - Each target_sampler_key may have multiple sampler ranks (DP replicas)
                # - Collect ALL sampler ranks that need this actor shard
                # - Each sampler rank receives from exactly ONE actor rank (no redundant transfers)
                # - Round-robin distributes load evenly across actor ranks

                # Collect all sampler ranks across all target keys
                all_sampler_ranks = []
                for sampler_key in target_sampler_keys:
                    sampler_rank_meta_list = self.sampler_param_to_rank[sampler_key]
                    all_sampler_ranks.extend(sampler_rank_meta_list)

                # Sort for deterministic assignment
                sorted_actor_ranks = sorted(actor_rank_meta_list, key=lambda x: x[0])
                sorted_sampler_ranks = sorted(all_sampler_ranks, key=lambda x: x[0])

                # Greedy load-balanced assignment with global load awareness:
                # For each sampler rank, assign the actor rank that minimizes the maximum transfer size
                # This helps reduce synchronization bottlenecks by keeping transfer sizes more uniform
                for sampler_rank, sampler_meta in sorted_sampler_ranks:
                    # Check if this sampler param was already assigned (happens when actor_tp > sampler_tp)
                    sampler_param_key = (sampler_rank, sampler_meta.param_name, sampler_meta.split_idx, actor_split_idx)
                    if sampler_param_key in assigned_sampler_params:
                        continue  # Skip - already assigned from another actor shard

                    # Find the actor rank that results in the smallest maximum transfer size
                    # We consider both the specific (actor, sampler) pair load AND the global actor load
                    best_actor_rank = None
                    best_actor_meta = None
                    best_score = (float("inf"), float("inf"))

                    for actor_rank, actor_meta in sorted_actor_ranks:
                        # Current load for this specific transfer
                        pair_load = assignment_counts[(actor_rank, sampler_rank)]

                        # Total load for this actor across all sampler ranks (O(1) lookup)
                        actor_total_load = actor_total_loads[actor_rank]

                        # Score: prioritize balancing pair load, with actor total as tiebreaker.
                        score = (pair_load, actor_total_load)

                        if score < best_score:
                            best_score = score
                            best_actor_rank = actor_rank
                            best_actor_meta = actor_meta

                    # Assign this parameter to the selected actor rank
                    routing_map[(best_actor_rank, sampler_rank)].append(
                        {
                            "actor": best_actor_meta,
                            "sampler": sampler_meta,
                        }
                    )

                    # Mark this sampler param as assigned
                    assigned_sampler_params.add(sampler_param_key)

                    # Update load counters
                    assignment_counts[(best_actor_rank, sampler_rank)] += 1
                    actor_total_loads[best_actor_rank] += 1

        # Verify that every sampler parameter gets assigned
        self._verify_routing_completeness(routing_map)

        return dict(routing_map)

    def _verify_routing_completeness(self, routing_map: dict[tuple[int, int], list[dict[str, any]]]):
        """
        Verify that every parameter in sampler workers gets assigned to receive from the correct actor worker(s).

        When actor_tp > sampler_tp, multiple actor shards must send to one sampler shard.
        When actor_tp <= sampler_tp, each sampler shard receives from exactly one actor shard.

        Args:
            routing_map: The routing map to verify

        Raises:
            ValueError: If any sampler parameter has incorrect assignments
        """
        # Track which actor shards send to each sampler param
        # Key: (sampler_rank, param_name, sampler_split_idx)
        # Value: set of actor_split_idx that send to this sampler param
        sampler_param_to_actor_shards = defaultdict(set)

        # Also track the actor ranks for error reporting
        sampler_param_to_actor_ranks = defaultdict(set)

        for (actor_rank, sampler_rank), param_dicts in routing_map.items():
            for param_dict in param_dicts:
                actor_meta = param_dict["actor"]
                sampler_meta = param_dict["sampler"]
                base_key = (sampler_rank, sampler_meta.param_name, sampler_meta.split_idx)

                sampler_param_to_actor_shards[base_key].add(actor_meta.split_idx)
                sampler_param_to_actor_ranks[base_key].add(actor_rank)

        # Pre-group keys by parameter name in O(K) instead of O(P*K) repeated scans
        # This is the key optimization - we build the index once instead of scanning for each param
        actor_split_indices_by_name = defaultdict(set)
        for param_name, split_idx in self.actor_param_to_rank.keys():
            actor_split_indices_by_name[param_name].add(split_idx)

        sampler_split_indices_by_name = defaultdict(set)
        for param_name, split_idx in self.sampler_param_to_rank.keys():
            sampler_split_indices_by_name[param_name].add(split_idx)

        # For each unique parameter name, determine expected number of actor shards per sampler shard
        # Now O(P) instead of O(P*K)
        param_name_to_tp_ratio = {}
        for param_name in sampler_split_indices_by_name:
            actor_tp_size = len(actor_split_indices_by_name[param_name])
            sampler_tp_size = len(sampler_split_indices_by_name[param_name])

            if actor_tp_size > sampler_tp_size:
                # Multiple actor shards map to one sampler shard
                expected_actor_shards = actor_tp_size // sampler_tp_size
            else:
                # One actor shard maps to one or more sampler shards
                expected_actor_shards = 1

            param_name_to_tp_ratio[param_name] = {
                "actor_tp": actor_tp_size,
                "sampler_tp": sampler_tp_size,
                "expected_actor_shards": expected_actor_shards,
            }

        # Collect all expected (sampler_rank, param_name, split_idx) tuples
        expected_sampler_params = set()
        for param_key, rank_meta_list in self.sampler_param_to_rank.items():
            param_name, split_idx = param_key
            for sampler_rank, sampler_meta in rank_meta_list:
                expected_sampler_params.add((sampler_rank, param_name, split_idx))

        # Check for missing assignments
        assigned_sampler_params = set(sampler_param_to_actor_shards.keys())
        missing_params = expected_sampler_params - assigned_sampler_params

        if missing_params:
            missing_by_rank = defaultdict(list)
            for sampler_rank, param_name, split_idx in missing_params:
                missing_by_rank[sampler_rank].append((param_name, split_idx))

            error_msg = "Routing verification failed! Some sampler parameters are not assigned:\n"
            for rank in sorted(missing_by_rank.keys()):
                params = missing_by_rank[rank]
                error_msg += f"  Sampler Rank {rank}: {len(params)} missing parameters\n"
                for param_name, split_idx in params[:3]:
                    error_msg += f"    - {param_name} (split_idx={split_idx})\n"
                if len(params) > 3:
                    error_msg += f"    ... and {len(params) - 3} more\n"
            raise ValueError(error_msg)

        # Check for extra assignments
        extra_params = assigned_sampler_params - expected_sampler_params
        if extra_params:
            error_msg = f"Routing verification failed! {len(extra_params)} extra parameters assigned that sampler doesn't have.\n"
            for sampler_rank, param_name, split_idx in list(extra_params)[:5]:
                error_msg += f"  - Rank {sampler_rank}: {param_name} (split_idx={split_idx})\n"
            raise ValueError(error_msg)

        # Verify each sampler param receives from the correct NUMBER of actor shards
        # AND verify correct actor shard indices when actor_tp > sampler_tp
        # Combined into a single pass for efficiency
        incorrect_shard_count = []
        incorrect_shard_mapping = []

        for base_key in expected_sampler_params:
            sampler_rank, param_name, sampler_split_idx = base_key

            actual_actor_shards = sampler_param_to_actor_shards.get(base_key, set())
            tp_info = param_name_to_tp_ratio[param_name]
            expected_count = tp_info["expected_actor_shards"]
            actual_count = len(actual_actor_shards)

            # Check shard count
            if actual_count != expected_count:
                incorrect_shard_count.append(
                    {
                        "sampler_rank": sampler_rank,
                        "param_name": param_name,
                        "sampler_split_idx": sampler_split_idx,
                        "expected": expected_count,
                        "actual": actual_count,
                        "actor_shards": sorted(actual_actor_shards),
                        "actor_ranks": sorted(sampler_param_to_actor_ranks.get(base_key, set())),
                    }
                )
            # Check shard mapping when actor_tp > sampler_tp
            elif tp_info["actor_tp"] > tp_info["sampler_tp"]:
                ratio = tp_info["actor_tp"] // tp_info["sampler_tp"]
                expected_actor_shard_indices = set(range(sampler_split_idx * ratio, (sampler_split_idx + 1) * ratio))
                if actual_actor_shards != expected_actor_shard_indices:
                    incorrect_shard_mapping.append(
                        {
                            "sampler_rank": sampler_rank,
                            "param_name": param_name,
                            "sampler_split_idx": sampler_split_idx,
                            "expected_shards": sorted(expected_actor_shard_indices),
                            "actual_shards": sorted(actual_actor_shards),
                        }
                    )

        if incorrect_shard_count:
            error_msg = "Routing verification failed! Incorrect number of actor shards for some sampler parameters:\n"
            for info in incorrect_shard_count[:10]:
                param_display = info["param_name"][:50] + "..." if len(info["param_name"]) > 50 else info["param_name"]
                error_msg += (
                    f"  Sampler Rank {info['sampler_rank']}, {param_display} (split_idx={info['sampler_split_idx']}):\n"
                    f"    Expected {info['expected']} actor shard(s), got {info['actual']}\n"
                    f"    Actor shards received: {info['actor_shards']}\n"
                    f"    From actor ranks: {info['actor_ranks']}\n"
                )
            if len(incorrect_shard_count) > 10:
                error_msg += f"  ... and {len(incorrect_shard_count) - 10} more\n"
            raise ValueError(error_msg)

        if incorrect_shard_mapping:
            error_msg = "Routing verification failed! Wrong actor shards assigned to sampler parameters:\n"
            for info in incorrect_shard_mapping[:10]:
                param_display = info["param_name"][:50] + "..." if len(info["param_name"]) > 50 else info["param_name"]
                error_msg += (
                    f"  Sampler Rank {info['sampler_rank']}, {param_display} (split_idx={info['sampler_split_idx']}):\n"
                    f"    Expected actor shards: {info['expected_shards']}\n"
                    f"    Got actor shards: {info['actual_shards']}\n"
                )
            if len(incorrect_shard_mapping) > 10:
                error_msg += f"  ... and {len(incorrect_shard_mapping) - 10} more\n"
            raise ValueError(error_msg)

        # Calculate statistics for logging
        total_transfers = sum(len(shards) for shards in sampler_param_to_actor_shards.values())
        unique_sampler_params = len(expected_sampler_params)

        # Count params by TP ratio type - use pre-computed tp_ratio instead of re-lookup
        one_to_one_params = sum(
            1 for key in expected_sampler_params if param_name_to_tp_ratio[key[1]]["expected_actor_shards"] == 1
        )
        multi_shard_params = unique_sampler_params - one_to_one_params

        print(
            f"[RoutingTable] ✓ Verified: {unique_sampler_params} sampler parameters, {total_transfers} total transfers\n"
            f"              - {one_to_one_params} params with 1:1 actor→sampler mapping\n"
            f"              - {multi_shard_params} params with N:1 actor→sampler mapping (actor_tp > sampler_tp)"
        )

    def get_transfers_for_actor_rank(self, actor_rank: int) -> dict[int, list[any]]:
        """
        Get all parameter transfers that an actor rank needs to send.

        Args:
            actor_rank: The actor rank to query

        Returns:
            Dict mapping sampler_rank to list of metadata to send
        """
        transfers = {}
        for (src_rank, dst_rank), params in self.routing_map.items():
            if src_rank == actor_rank:
                transfers[dst_rank] = params
        return transfers

    def get_transfers_for_sampler_rank(self, sampler_rank: int) -> dict[int, list[any]]:
        """
        Get all parameter transfers that a sampler rank needs to receive.

        Args:
            sampler_rank: The sampler rank to query

        Returns:
            Dict mapping actor_rank to list of metadata to receive
        """
        transfers = {}
        for (src_rank, dst_rank), params in self.routing_map.items():
            if dst_rank == sampler_rank:
                transfers[src_rank] = params
        return transfers

    def get_all_transfers(self) -> dict[tuple[int, int], list[any]]:
        """
        Get the complete routing map.

        Returns:
            Dict mapping (actor_rank, sampler_rank) to list of metadata
        """
        return self.routing_map

    def get_load_statistics(self) -> dict:
        """
        Get statistics about the load distribution across ranks.

        Returns:
            Dict containing load statistics for actor and sampler ranks
        """
        actor_send_load = defaultdict(int)
        sampler_recv_load = defaultdict(int)
        transfer_sizes = []

        for (actor_rank, sampler_rank), param_dicts in self.routing_map.items():
            param_count = len(param_dicts)
            actor_send_load[actor_rank] += param_count
            sampler_recv_load[sampler_rank] += param_count
            transfer_sizes.append(param_count)

        # Calculate transfer size statistics
        if transfer_sizes:
            min_transfer = min(transfer_sizes)
            max_transfer = max(transfer_sizes)
            avg_transfer = sum(transfer_sizes) / len(transfer_sizes)
            # Calculate imbalance ratio
            imbalance_ratio = (max_transfer - min_transfer) / avg_transfer if avg_transfer > 0 else 0
        else:
            min_transfer = max_transfer = avg_transfer = imbalance_ratio = 0

        return {
            "actor_send_load": dict(actor_send_load),
            "sampler_recv_load": dict(sampler_recv_load),
            "total_params": sum(len(param_dicts) for param_dicts in self.routing_map.values()),
            "num_transfers": len(self.routing_map),
            "transfer_size_min": min_transfer,
            "transfer_size_max": max_transfer,
            "transfer_size_avg": avg_transfer,
            "transfer_imbalance_ratio": imbalance_ratio,
        }

    def print_routing_table(self):
        """Print a human-readable representation of the routing table."""
        print("\n" + "=" * 80)
        print("ROUTING TABLE")
        print("=" * 80)

        stats = self.get_load_statistics()
        print(f"\nTotal parameters to transfer: {stats['total_params']}")
        print(f"Number of rank-to-rank transfers: {stats['num_transfers']}")

        print("\n--- Transfer Size Distribution ---")
        print(f"  Min transfer size: {stats['transfer_size_min']} parameters")
        print(f"  Max transfer size: {stats['transfer_size_max']} parameters")
        print(f"  Avg transfer size: {stats['transfer_size_avg']:.1f} parameters")
        print(f"  Imbalance ratio: {stats['transfer_imbalance_ratio']:.2f} (lower is better)")
        if stats["transfer_imbalance_ratio"] > 0.5:
            print("  ⚠ WARNING: High transfer imbalance detected!")
            print("  This may be due to ETP size < TP size (e.g., ETP=2, TP=4).")
            print("  Some sampler ranks may only receive non-expert parameters.")

        # Show parameter distribution info
        print("\n--- Parameter Distribution ---")
        actor_keys = set(self.actor_param_to_rank.keys())
        sampler_keys = set(self.sampler_param_to_rank.keys())
        common_keys = actor_keys & sampler_keys

        if common_keys:
            sample_key = sorted(common_keys, key=lambda x: (x[0], x[1]))[0]  # Sort by name, split_idx
            actor_rank_meta_list = self.actor_param_to_rank[sample_key]
            sampler_rank_meta_list = self.sampler_param_to_rank[sample_key]

            _, sample_meta = actor_rank_meta_list[0]
            param_name, split_idx = sample_key
            param_display = param_name[:50] + "..." if len(param_name) > 50 else param_name

            actor_ranks = [rank for rank, _ in actor_rank_meta_list]
            sampler_ranks = [rank for rank, _ in sampler_rank_meta_list]

            print(f"  Sample parameter '{param_display}' (split_idx={split_idx}, split_dim={sample_meta.split_dim}):")
            print(f"    Shape: {sample_meta.param_shape}, Dtype: {sample_meta.param_dtype}")
            print(f"    Actor ranks: {sorted(actor_ranks)}")
            print(f"    Sampler ranks: {sorted(sampler_ranks)}")

        # Show TP/ETP configuration if detected
        param_names = set(key[0] for key in actor_keys) & set(key[0] for key in sampler_keys)
        if param_names:
            # Check regular (non-expert) parameter TP size
            non_expert_params = [name for name in param_names if ".mlp.experts." not in name]
            expert_params = [name for name in param_names if ".mlp.experts." in name]

            if non_expert_params:
                sample_non_expert = sorted(non_expert_params)[0]
                non_expert_tp_size = len([key for key in actor_keys if key[0] == sample_non_expert])

                print("\n--- Tensor Parallel Configuration ---")
                print(f"  Non-expert parameter TP size: {non_expert_tp_size}")
                print(f"    (sample: {sample_non_expert[:50]})")

            if expert_params:
                sample_expert = sorted(expert_params)[0]
                expert_tp_size = len([key for key in actor_keys if key[0] == sample_expert])

                if non_expert_params:
                    print(f"  Expert parameter ETP size: {expert_tp_size}")
                    print(f"    (sample: {sample_expert[:50]})")

                    if expert_tp_size != non_expert_tp_size:
                        print(f"\n  ⚠ NOTE: ETP ({expert_tp_size}) ≠ TP ({non_expert_tp_size})")
                        print("  This causes inevitable transfer imbalance:")
                        print(f"    - Ranks with split_idx < {expert_tp_size}: receive expert + non-expert params")
                        print(f"    - Ranks with split_idx >= {expert_tp_size}: receive only non-expert params")
                else:
                    print("\n--- Tensor Parallel Configuration ---")
                    print(f"  Expert parameter ETP size: {expert_tp_size}")
                    print(f"    (sample: {sample_expert[:50]})")

        print("\n--- Actor Send Load ---")
        for rank, load in sorted(stats["actor_send_load"].items()):
            print(f"  Actor Rank {rank}: {load} parameters")

        print("\n--- Sampler Receive Load ---")
        for rank, load in sorted(stats["sampler_recv_load"].items()):
            print(f"  Sampler Rank {rank}: {load} parameters")

        print("\n--- Transfer Details (first 10 rank pairs) ---")
        for idx, ((actor_rank, sampler_rank), param_dicts) in enumerate(sorted(self.routing_map.items())):
            if idx >= 10:
                print(f"  ... and {len(self.routing_map) - 10} more rank pairs")
                break
            print(f"  Actor[{actor_rank}] -> Sampler[{sampler_rank}]: {len(param_dicts)} parameters")
            if len(param_dicts) <= 3:
                for param_dict in param_dicts:
                    sampler_meta = param_dict["sampler"]
                    param_display = (
                        sampler_meta.param_name[:40] + "..."
                        if len(sampler_meta.param_name) > 40
                        else sampler_meta.param_name
                    )
                    print(f"    - {param_display} (split_idx={sampler_meta.split_idx})")
            else:
                for param_dict in param_dicts[:2]:
                    sampler_meta = param_dict["sampler"]
                    param_display = (
                        sampler_meta.param_name[:40] + "..."
                        if len(sampler_meta.param_name) > 40
                        else sampler_meta.param_name
                    )
                    print(f"    - {param_display} (split_idx={sampler_meta.split_idx})")
                print(f"    ... and {len(param_dicts) - 2} more")

        print("=" * 80 + "\n")
