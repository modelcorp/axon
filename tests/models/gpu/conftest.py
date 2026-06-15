"""Shared fixtures and helpers for GPU mbridge integration tests."""

import json
import os
import subprocess
import sys
import tempfile

import pytest
import torch

MIN_GPUS = 8
WORKER_SCRIPT = os.path.join(os.path.dirname(__file__), "..", "_gpu_mbridge_worker.py")
_BASE_PORT = 29600


@pytest.fixture(scope="module", autouse=True)
def require_gpus():
    if not torch.cuda.is_available():
        pytest.skip("No CUDA GPUs available")
    if torch.cuda.device_count() < MIN_GPUS:
        pytest.skip(f"Requires {MIN_GPUS} GPUs, found {torch.cuda.device_count()}")


def _find_free_gpu():
    """Return the GPU index with the most free memory."""
    best_idx, best_free = 0, 0
    for i in range(torch.cuda.device_count()):
        free = torch.cuda.mem_get_info(i)[0]
        if free > best_free:
            best_free = free
            best_idx = i
    return best_idx


def run_mbridge_test(model_id, port_offset=0, timeout=600, tp=1, pp=1, ep=1, etp=1, cp=1, gpus=None, extra_env=None):
    """Launch the mbridge worker via torchrun and return the results dict.

    Args:
        model_id: HuggingFace model ID.
        port_offset: Added to base port to avoid collisions between tests.
        timeout: Max seconds to wait for the worker.
        tp/pp/cp: "attention" parallel sizes — nproc = tp * pp * cp * DP.
        ep/etp: "expert" parallel sizes — orthogonal to tp/pp, sharing the world.
        gpus: Comma-separated GPU indices. If None, auto-selects.
        extra_env: Optional dict of extra env vars to pass through (e.g. to
            enable the MBRIDGE_EXPORT_COMPARE / MBRIDGE_TEST_EXPORT codepaths).
    """
    nproc = tp * pp * cp
    port = _BASE_PORT + port_offset

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        output_path = f.name

    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    if gpus is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpus
    elif nproc == 1:
        env["CUDA_VISIBLE_DEVICES"] = str(_find_free_gpu())

    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        f"--nproc_per_node={nproc}",
        "--master_port",
        str(port),
        WORKER_SCRIPT,
        "--model-id",
        model_id,
        "--output",
        output_path,
        "--tp",
        str(tp),
        "--pp",
        str(pp),
        "--ep",
        str(ep),
        "--etp",
        str(etp),
        "--cp",
        str(cp),
    ]

    try:
        proc = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        if os.path.exists(output_path):
            with open(output_path) as f:
                result = json.load(f)
        else:
            result = {
                "passed": False,
                "checks": {},
                "error": f"No output file. returncode={proc.returncode}",
                "stdout": proc.stdout[-2000:] if proc.stdout else "",
                "stderr": proc.stderr[-2000:] if proc.stderr else "",
            }

        result["_returncode"] = proc.returncode
        if proc.stderr:
            result["_stderr_tail"] = proc.stderr[-1000:]

        return result

    except subprocess.TimeoutExpired:
        return {"passed": False, "checks": {}, "error": f"Timeout after {timeout}s"}
    finally:
        if os.path.exists(output_path):
            os.unlink(output_path)
