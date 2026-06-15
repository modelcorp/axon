import os
import socket
import subprocess
import sys
import time
from datetime import timedelta
from pathlib import Path

import ray
import torch
import torch.distributed as dist
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

RAY_ENV_VARS = {
    "HF_HOME": "/root/.cache/huggingface",
    "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE", "0"),
    "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE", "0"),
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "NCCL_SOCKET_IFNAME": os.environ.get("NCCL_SOCKET_IFNAME", "eth0"),
    "NCCL_IB_DISABLE": os.environ.get("NCCL_IB_DISABLE", "1"),
    "NCCL_ASYNC_ERROR_HANDLING": os.environ.get("NCCL_ASYNC_ERROR_HANDLING", "1"),
    "NCCL_BLOCKING_WAIT": os.environ.get("NCCL_BLOCKING_WAIT", "1"),
}

if os.environ.get("AXON_ROOT"):
    AXON_ROOT = Path(os.environ["AXON_ROOT"])
else:
    AXON_ROOT = Path(__file__).resolve().parents[2]  # tests/smoke -> tests -> axon (project root)

if os.environ.get("AXON_DATA_DIR"):
    DATA_DIR = Path(os.environ["AXON_DATA_DIR"])
else:
    DATA_DIR = AXON_ROOT / "data" / "frozenlake"


def _get_int_env(name, default):
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        print(f"==> [Smoke Test] {name}='{raw}' is invalid; defaulting to {default}")
        return default


def _get_expected_nodes():
    raw = os.environ.get("AWS_BATCH_JOB_NUM_NODES")
    if raw is None or raw == "":
        print("==> [Smoke Test] AWS_BATCH_JOB_NUM_NODES not set; defaulting to 1")
        return 1
    try:
        return int(raw)
    except ValueError:
        print(f"==> [Smoke Test] AWS_BATCH_JOB_NUM_NODES='{raw}' is invalid; defaulting to 1")
        return 1


def check_ray_cluster():
    """
    Initialize Ray and verify cluster resources.
    Waits for up to 5 minutes for expected GPUs to be registered.
    """
    print("==> [Smoke Test] Initializing Ray...")
    try:
        ray.init(address="auto", runtime_env={"env_vars": RAY_ENV_VARS})
    except Exception as e:
        print(f"Error initializing Ray: {e}")
        # Fallback for single-node non-batch runs if needed, though address="auto" usually works if ray start was called.
        ray.init(runtime_env={"env_vars": RAY_ENV_VARS})

    # Determine expected resources
    # Default to 1 node if not specified
    expected_nodes = _get_expected_nodes()
    # Assume 1 GPU per node for now (g5/p5 instances usually have at least 1, p5.48xlarge has 8 but we request 1 in job def)
    # The job def requests "GPU": "1", so container sees 1.
    # If using p5.48xlarge with full GPUs, we'd see 8.
    # Let's enforce we see at least expected_nodes * 1 GPU.
    expected_gpus = expected_nodes

    print(f"==> [Smoke Test] Waiting for {expected_nodes} nodes and {expected_gpus} GPUs...")
    timeout = _get_int_env(
        "AXON_SMOKE_RESOURCE_TIMEOUT_SEC",
        300 if expected_nodes <= 1 else 900,
    )
    start_time = time.time()

    while True:
        resources = ray.cluster_resources()
        gpu_count = resources.get("GPU", 0)
        node_count = len(ray.nodes())

        print(f"    Current resources: Nodes={node_count}, GPUs={gpu_count}")

        if gpu_count >= expected_gpus and node_count >= expected_nodes:
            print("    ✓ Expected resources detected.")
            break

        if time.time() - start_time > timeout:
            print(f"Error: Timeout waiting for resources. Expected {expected_nodes} nodes, {expected_gpus} GPUs.")
            sys.exit(1)

        time.sleep(10)


def check_nccl():
    """
    Verify NCCL communication via a simple all_reduce.
    This runs a Ray task on each GPU to participate in a torch.distributed group.
    """
    print("==> [Smoke Test] Verifying NCCL communication...")

    resources = ray.cluster_resources()
    total_gpus = int(resources.get("GPU", 0))

    if total_gpus == 0:
        print("Skipping NCCL check (no GPUs).")
        return

    @ray.remote(num_gpus=1)
    def run_all_reduce(rank, world_size, master_addr, master_port):
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = str(master_port)
        os.environ.setdefault("NCCL_SOCKET_IFNAME", "eth0")
        os.environ.setdefault("NCCL_IB_DISABLE", "1")

        local_ip = socket.gethostbyname(socket.gethostname())
        node_ip = ray.util.get_node_ip_address()
        print(f"    [NCCL] rank={rank} host={socket.gethostname()} ip={local_ip} node_ip={node_ip}")

        # Initialize Process Group
        nccl_timeout = _get_int_env("AXON_SMOKE_NCCL_TIMEOUT_SEC", 180)
        dist.init_process_group(
            backend="nccl",
            rank=rank,
            world_size=world_size,
            timeout=timedelta(seconds=nccl_timeout),
        )

        # Create tensor and all_reduce
        tensor = torch.ones(1).cuda()
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

        result = tensor.item()
        dist.destroy_process_group()
        return result

    # Get Head IP for master addr
    master_addr = os.environ.get("AWS_BATCH_JOB_MAIN_NODE_PRIVATE_IPV4_ADDRESS")
    if not master_addr:
        master_addr = ray.util.get_node_ip_address()
    master_port = 29500  # Free port

    nodes = [n for n in ray.nodes() if n.get("Alive")]
    head_ip = ray.util.get_node_ip_address()
    head_node_id = None
    for node in nodes:
        if node.get("NodeManagerAddress") == head_ip:
            head_node_id = node.get("NodeID")
            break
    if not head_node_id and nodes:
        head_node_id = nodes[0].get("NodeID")

    world_size = min(total_gpus, len(nodes))

    print(f"    Launching {world_size} workers for all_reduce (Master: {master_addr}:{master_port})...")

    refs = []
    for rank in range(world_size):
        options = {}
        if world_size > 1 and rank == 0 and head_node_id:
            # Pin rank 0 to the head node so the rendezvous address is local to rank 0.
            options = {
                "scheduling_strategy": NodeAffinitySchedulingStrategy(
                    node_id=head_node_id,
                    soft=False,
                )
            }
        refs.append(run_all_reduce.options(**options).remote(rank, world_size, master_addr, master_port))

    try:
        results = ray.get(refs, timeout=_get_int_env("AXON_SMOKE_NCCL_TIMEOUT_SEC", 180))
        print(f"    Results: {results}")
        assert all(r == world_size for r in results), f"NCCL Check Failed: Expected {world_size}, got {results}"
        print("    ✓ NCCL communication successful.")
    except Exception as e:
        print(f"Error during NCCL check: {e}")
        sys.exit(1)


def run_frozenlake_ppo():
    """
    Run a minimal FrozenLake PPO training loop.
    """
    print("==> [Smoke Test] Running FrozenLake PPO...")

    # Determine num_nodes from environment
    num_nodes = _get_expected_nodes()

    sampler_name = "vllm"

    recipes_dir = AXON_ROOT / "recipes" / "frozenlake"

    cmd = [
        sys.executable,
        "-m",
        "axon.driver.train_agent_ppo",
        # Cluster
        f"num_nodes={num_nodes}",
        "num_gpus_per_node=1",
        # Training loop
        "total_training_steps=1",
        "total_epochs=1",
        "train_batch_size=1",
        "mini_batch_size=2",
        "max_prompt_length=512",
        "max_seq_length=512",
        "max_steps=1",
        "prompt_truncation=left",
        "validation.before_train=False",
        "logger=[console]",
        # Data
        f"train_files={DATA_DIR / 'train.parquet'}",
        f"val_files={DATA_DIR / 'test.parquet'}",
        # Model
        "model_path=TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        # Loss
        "advantage=reinforce_plus_plus",
        "drop_zero_advantage_samples=false",
        # Actor
        "actor.micro_batch_size_per_gpu=1",
        "actor.max_token_len_per_gpu=1024",
        "actor.forward_micro_batch_size_per_gpu=1",
        # Sampler
        f"sampler.name={sampler_name}",
        "sampler.tensor_model_parallel_size=1",
        "sampler.gpu_memory_utilization=0.2",
        "sampler.enforce_eager=True",
        "sampler.max_num_batched_tokens=2048",
        "sampler.max_num_seqs=16",
        "sampler.max_model_len=1024",
        "decoding.top_k=-1",
        # Program (FrozenLake)
        "program.name=react",
        f"+program.env_name={recipes_dir / 'env.py'}:FrozenLakeEnv",
        f"+program.agent_name={recipes_dir / 'agent.py'}:FrozenLakeAgent",
        "+program.env_args.max_turns=8",
    ]

    print(f"    Executing: {' '.join(cmd)}")

    # We use subprocess to run the actual training command
    # This ensures it runs in a separate process context if needed, but inherits env vars.
    env = os.environ.copy()
    env.setdefault("WANDB_MODE", "disabled")
    env.setdefault("WANDB_DISABLED", "true")
    env.setdefault("WANDB_SILENT", "true")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("TORCH_COMPILE_DISABLE", "1")
    env.setdefault("TORCHINDUCTOR_DISABLE", "1")
    env.setdefault("NCCL_SOCKET_IFNAME", "eth0")
    env.setdefault("NCCL_IB_DISABLE", "1")
    env.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")
    env.setdefault("NCCL_BLOCKING_WAIT", "1")
    env.setdefault("HF_HOME", "/root/.cache/huggingface")
    env.setdefault("HF_HUB_OFFLINE", "0")
    env.setdefault("TRANSFORMERS_OFFLINE", "0")
    env.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    result = subprocess.run(cmd, env=env, capture_output=False)  # stream output to stdout

    if result.returncode != 0:
        print("Error: FrozenLake PPO run failed.")
        sys.exit(result.returncode)

    print("    ✓ FrozenLake PPO run complete.")


if __name__ == "__main__":
    print("==================================================")
    print("   Starting FrozenLake Ray Smoke Test")
    print("==================================================")

    check_ray_cluster()
    check_nccl()
    run_frozenlake_ppo()

    print("==================================================")
    print("   Smoke Test PASSED")
    print("==================================================")
