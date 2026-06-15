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
#!/usr/bin/env python3
"""
Verify P2P operation order between actor sends and sampler receives.

This script reads the P2P operation logs generated during weight transfer and
verifies that:
1. For each send operation from an actor to a sampler rank, there's a matching receive
2. The dtype and shape of corresponding send/recv operations match
3. The order of operations is consistent

Usage:
    python verify_p2p_order.py [--log-dir ~/p2p_logs] [--verbose] [--stats]
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path


def load_p2p_logs(log_dir):
    """Load all P2P logs from the directory."""
    log_dir = Path(log_dir)

    actor_sends = {}  # actor_rank -> list of send ops
    sampler_recvs = {}  # sampler_rank -> list of recv ops

    # Load actor send logs
    for log_file in sorted(log_dir.glob("actor_rank_*_send.json")):
        try:
            rank = int(log_file.stem.split("_")[2])
            with open(log_file) as f:
                data = json.load(f)
                if not isinstance(data, list):
                    print(f"⚠️  Warning: {log_file} does not contain a list, skipping")
                    continue
                actor_sends[rank] = data
        except (ValueError, IndexError, json.JSONDecodeError) as e:
            print(f"⚠️  Warning: Failed to load {log_file}: {e}")
            continue

    # Load sampler receive logs
    for log_file in sorted(log_dir.glob("sampler_rank_*_recv.json")):
        try:
            rank = int(log_file.stem.split("_")[2])
            with open(log_file) as f:
                data = json.load(f)
                if not isinstance(data, list):
                    print(f"⚠️  Warning: {log_file} does not contain a list, skipping")
                    continue
                sampler_recvs[rank] = data
        except (ValueError, IndexError, json.JSONDecodeError) as e:
            print(f"⚠️  Warning: Failed to load {log_file}: {e}")
            continue

    return actor_sends, sampler_recvs


def build_transfer_map(actor_sends, actor_world_size):
    """
    Build a map of (actor_rank, sampler_rank) -> list of send operations.

    Args:
        actor_sends: Dict of actor_rank -> list of send ops
        actor_world_size: Number of actor ranks

    Returns:
        Dict of (actor_rank, sampler_rank) -> list of send ops
    """
    transfer_map = defaultdict(list)

    for actor_rank, send_ops in actor_sends.items():
        for op in send_ops:
            # peer_rank in send log is the global rank in bridge PG
            # sampler ranks start at actor_world_size
            sampler_rank = op["peer_rank"] - actor_world_size
            transfer_map[(actor_rank, sampler_rank)].append(op)

    return transfer_map


def verify_p2p_order(actor_sends, sampler_recvs, actor_world_size, verbose=False):
    """
    Verify that P2P send and receive operations match in order, dtype, and shape.

    Args:
        actor_sends: Dict of actor_rank -> list of send ops
        sampler_recvs: Dict of sampler_rank -> list of recv ops
        actor_world_size: Number of actor ranks
        verbose: Print detailed information

    Returns:
        Tuple of (success: bool, errors: list of error messages)
    """
    errors = []

    # Build transfer map: (actor_rank, sampler_rank) -> list of sends
    transfer_map = build_transfer_map(actor_sends, actor_world_size)

    print(f"\n{'=' * 80}")
    print("P2P Operation Verification")
    print(f"{'=' * 80}")
    print(f"Actor ranks: {sorted(actor_sends.keys())}")
    print(f"Sampler ranks: {sorted(sampler_recvs.keys())}")
    print(f"Actor world size: {actor_world_size}")
    print(f"{'=' * 80}\n")

    # For each sampler rank, verify receives match sends
    for sampler_rank, recv_ops in sorted(sampler_recvs.items()):
        if verbose:
            print(f"\n--- Checking Sampler Rank {sampler_rank} ---")

        # Group receives by source actor rank
        recv_by_actor = defaultdict(list)
        for op in recv_ops:
            actor_rank = op["peer_rank"]
            recv_by_actor[actor_rank].append(op)

        # Verify each actor -> sampler connection
        for actor_rank, recvs in sorted(recv_by_actor.items()):
            key = (actor_rank, sampler_rank)
            sends = transfer_map.get(key, [])

            if verbose:
                print(f"  Actor {actor_rank} -> Sampler {sampler_rank}: {len(sends)} sends, {len(recvs)} recvs")

            # Check operation count matches
            if len(sends) != len(recvs):
                error = (
                    f"❌ Mismatch in op count: Actor {actor_rank} -> Sampler {sampler_rank}: "
                    f"{len(sends)} sends vs {len(recvs)} recvs"
                )
                errors.append(error)
                print(error)

                # Show sample operations for debugging
                if verbose and (sends or recvs):
                    print("  Sample send ops (first 3):")
                    for i, op in enumerate(sends[:3]):
                        print(f"    [{i}] shape={op['shape']}, dtype={op['dtype']}, numel={op['numel']}")
                    print("  Sample recv ops (first 3):")
                    for i, op in enumerate(recvs[:3]):
                        print(f"    [{i}] shape={op['shape']}, dtype={op['dtype']}, numel={op['numel']}")
                continue

            # Check each send/recv pair
            mismatches = []
            max_mismatches_to_show = 10
            for idx, (send_op, recv_op) in enumerate(zip(sends, recvs, strict=False)):
                # Check dtype
                if send_op["dtype"] != recv_op["dtype"]:
                    mismatches.append(
                        f"    Op {idx}: dtype mismatch - send {send_op['dtype']} vs recv {recv_op['dtype']}"
                    )

                # Check shape
                if send_op["shape"] != recv_op["shape"]:
                    mismatches.append(
                        f"    Op {idx}: shape mismatch - send {send_op['shape']} vs recv {recv_op['shape']}"
                    )

                # Check numel
                if send_op["numel"] != recv_op["numel"]:
                    mismatches.append(
                        f"    Op {idx}: numel mismatch - send {send_op['numel']} vs recv {recv_op['numel']}"
                    )

            if mismatches:
                # Limit output to avoid overwhelming the console
                total_mismatches = len(mismatches)
                mismatches_to_show = mismatches[:max_mismatches_to_show]

                error_msg = f"❌ Mismatches in Actor {actor_rank} -> Sampler {sampler_rank} ({total_mismatches} total):"
                if total_mismatches > max_mismatches_to_show:
                    error_msg += f"\n    (Showing first {max_mismatches_to_show} of {total_mismatches} mismatches)"
                error_msg += "\n" + "\n".join(mismatches_to_show)

                errors.append(error_msg)
                print(error_msg)
            elif verbose:
                print(f"  ✓ All {len(sends)} operations match")

    # Check for sends without corresponding receives
    all_recv_pairs = set()
    for sampler_rank, recv_ops in sampler_recvs.items():
        for op in recv_ops:
            all_recv_pairs.add((op["peer_rank"], sampler_rank))

    for (actor_rank, sampler_rank), sends in transfer_map.items():
        if (actor_rank, sampler_rank) not in all_recv_pairs:
            error = f"❌ Actor {actor_rank} sent {len(sends)} ops to Sampler {sampler_rank} but no receives found"
            errors.append(error)
            print(error)

    print(f"\n{'=' * 80}")
    if errors:
        print(f"❌ Verification FAILED with {len(errors)} error(s)")
        print(f"{'=' * 80}")

        # Print summary of total operations
        total_sends = sum(len(ops) for ops in actor_sends.values())
        total_recvs = sum(len(ops) for ops in sampler_recvs.values())
        print("\nSummary:")
        print(f"  Total send operations: {total_sends}")
        print(f"  Total receive operations: {total_recvs}")
        print(f"  Actor ranks: {len(actor_sends)}")
        print(f"  Sampler ranks: {len(sampler_recvs)}")
        print()
        return False, errors
    else:
        print("✓ All P2P operations verified successfully!")

        # Print summary
        total_sends = sum(len(ops) for ops in actor_sends.values())
        total_recvs = sum(len(ops) for ops in sampler_recvs.values())
        num_pairs = len(transfer_map)
        print("\nSummary:")
        print(f"  Total send operations: {total_sends}")
        print(f"  Total receive operations: {total_recvs}")
        print(f"  Unique actor→sampler pairs: {num_pairs}")
        print(f"  Actor ranks: {len(actor_sends)}")
        print(f"  Sampler ranks: {len(sampler_recvs)}")
        print(f"{'=' * 80}\n")
        return True, []


def print_statistics(actor_sends, sampler_recvs, actor_world_size):
    """Print statistics about P2P operations."""
    transfer_map = build_transfer_map(actor_sends, actor_world_size)

    total_sends = sum(len(ops) for ops in actor_sends.values())
    total_recvs = sum(len(ops) for ops in sampler_recvs.values())

    total_send_bytes = sum(sum(op["numel"] * dtype_size(op["dtype"]) for op in ops) for ops in actor_sends.values())

    print(f"\n{'=' * 80}")
    print("P2P Statistics")
    print(f"{'=' * 80}")
    print(f"Total send operations: {total_sends}")
    print(f"Total receive operations: {total_recvs}")
    print(f"Total data transferred: {total_send_bytes / (1024**3):.2f} GB")
    print(f"Unique actor->sampler pairs: {len(transfer_map)}")

    # Per-rank statistics
    print("\nPer-rank send counts:")
    for actor_rank in sorted(actor_sends.keys()):
        print(f"  Actor {actor_rank}: {len(actor_sends[actor_rank])} sends")

    print("\nPer-rank receive counts:")
    for sampler_rank in sorted(sampler_recvs.keys()):
        print(f"  Sampler {sampler_rank}: {len(sampler_recvs[sampler_rank])} recvs")

    print(f"{'=' * 80}\n")


def dtype_size(dtype_str):
    """Get size in bytes for a dtype string."""
    if "float32" in dtype_str or "int32" in dtype_str:
        return 4
    elif "float16" in dtype_str or "bfloat16" in dtype_str or "int16" in dtype_str:
        return 2
    elif "float64" in dtype_str or "int64" in dtype_str:
        return 8
    elif "int8" in dtype_str or "uint8" in dtype_str:
        return 1
    else:
        return 4  # default


def main():
    parser = argparse.ArgumentParser(description="Verify P2P operation order between actor and sampler workers")

    # Default to ~/p2p_logs
    default_log_dir = os.path.join(str(Path.home()), "p2p_logs")

    parser.add_argument(
        "--log-dir",
        type=str,
        default=default_log_dir,
        help=f"Directory containing P2P log files (default: {default_log_dir})",
    )
    parser.add_argument(
        "--actor-world-size", type=int, help="Number of actor ranks (auto-detected from logs if not provided)"
    )
    parser.add_argument("--verbose", action="store_true", help="Print detailed verification information")
    parser.add_argument("--stats", action="store_true", help="Print statistics about P2P operations")
    parser.add_argument("--output-report", type=str, help="Save detailed error report to file")

    args = parser.parse_args()

    # Expand ~ in log directory path
    args.log_dir = os.path.expanduser(args.log_dir)

    # Check if log directory exists
    if not os.path.exists(args.log_dir):
        print(f"❌ Error: Log directory '{args.log_dir}' does not exist")
        print("\nMake sure your training has run and generated P2P logs.")
        print(f"Logs should be in: {args.log_dir}")
        print("\nOr set P2P_LOG_DIR environment variable before training:")
        print(f"  export P2P_LOG_DIR={args.log_dir}")
        sys.exit(1)

    # Load logs
    print(f"Loading P2P logs from {args.log_dir}...")
    actor_sends, sampler_recvs = load_p2p_logs(args.log_dir)

    if not actor_sends:
        print(f"❌ Error: No actor send logs found in {args.log_dir}")
        print("\nExpected files like: actor_rank_0_send.json, actor_rank_1_send.json, ...")
        sys.exit(1)

    if not sampler_recvs:
        print(f"❌ Error: No sampler receive logs found in {args.log_dir}")
        print("\nExpected files like: sampler_rank_0_recv.json, sampler_rank_1_recv.json, ...")
        sys.exit(1)

    print(f"✓ Loaded {len(actor_sends)} actor send logs and {len(sampler_recvs)} sampler receive logs")

    # Determine actor world size
    if args.actor_world_size:
        actor_world_size = args.actor_world_size
    else:
        # Auto-detect from logs
        actor_world_size = len(actor_sends)
        print(f"Auto-detected actor_world_size: {actor_world_size}")

    # Print statistics if requested
    if args.stats:
        print_statistics(actor_sends, sampler_recvs, actor_world_size)

    # Verify P2P operations
    success, errors = verify_p2p_order(actor_sends, sampler_recvs, actor_world_size, verbose=args.verbose)

    # Save detailed report if requested
    if args.output_report:
        report_path = os.path.expanduser(args.output_report)
        with open(report_path, "w") as f:
            f.write("P2P Operation Verification Report\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Log directory: {args.log_dir}\n")
            f.write(f"Actor ranks: {sorted(actor_sends.keys())}\n")
            f.write(f"Sampler ranks: {sorted(sampler_recvs.keys())}\n")
            f.write(f"Actor world size: {actor_world_size}\n")
            f.write(f"Status: {'PASSED' if success else 'FAILED'}\n\n")

            if errors:
                f.write(f"Errors ({len(errors)}):\n")
                f.write("=" * 80 + "\n\n")
                for i, error in enumerate(errors, 1):
                    f.write(f"Error {i}:\n{error}\n\n")
            else:
                f.write("All P2P operations verified successfully!\n")

        print(f"\n📝 Detailed report saved to: {report_path}")

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
