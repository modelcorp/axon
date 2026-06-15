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
import pickle
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import ray.util.collective as collective
import torch


def _get_current_device():
    """Get the current CUDA device for this worker.

    Returns:
        torch.device: The current CUDA device
    """
    if torch.cuda.is_available():
        return torch.device(f"cuda:{torch.cuda.current_device()}")
    raise RuntimeError("CUDA is not available")


def _bytes_to_cuda_u8(buf, device):
    """Convert bytes buffer to CUDA uint8 tensor.

    Args:
        buf: Bytes buffer to convert
        device: CUDA device to place tensor on

    Returns:
        torch.Tensor: uint8 tensor on specified CUDA device
    """
    # CPU → GPU staging
    cpu_u8 = np.frombuffer(buf, dtype=np.uint8).copy()
    return torch.as_tensor(cpu_u8, dtype=torch.uint8, device=device)


def _cuda_u8_to_bytes(t):
    """Convert CUDA uint8 tensor to bytes.

    Args:
        t: CUDA uint8 tensor to convert

    Returns:
        bytes: Tensor data as bytes
    """
    # GPU → CPU (tensor should already be synchronized before calling this)
    return t.detach().cpu().numpy().tobytes()


def encode_object_to_cuda_tensor(obj, device, protocol=pickle.HIGHEST_PROTOCOL):
    """Encode a Python object to a CUDA uint8 tensor.

    Args:
        obj: Python object to encode
        device: CUDA device to place tensor on
        protocol: Pickle protocol to use (default: HIGHEST_PROTOCOL)

    Returns:
        torch.Tensor: uint8 tensor on specified CUDA device containing serialized object
    """
    payload_bytes = pickle.dumps(obj, protocol=protocol)
    return _bytes_to_cuda_u8(payload_bytes, device)


def decode_cuda_tensor_to_object(tensor):
    """Decode a CUDA uint8 tensor back to a Python object.

    Args:
        tensor: CUDA uint8 tensor containing serialized object

    Returns:
        Deserialized Python object
    """
    payload_bytes = _cuda_u8_to_bytes(tensor)
    return pickle.loads(payload_bytes)  # nosec B301


def dispatch_one_to_all_via_cc(group_name, args, kwargs):
    """Send data from rank 0 to all other ranks using Ray collective communication.

    This function implements an efficient GPU-based broadcast operation where rank 0
    sends its data to all other ranks. The data is serialized and transferred via CUDA tensors.

    Args:
        group_name (str): Name of the collective group
        args: Tuple of arguments to send to all ranks
        kwargs: Dictionary of keyword arguments to send to all ranks

    Returns:
        Tuple of (args, kwargs)
    """
    rank = collective.get_rank(group_name)
    use_cuda = torch.cuda.is_available()
    if not use_cuda:
        raise RuntimeError("GPU staging requires CUDA; fall back to CPU/GLOO path otherwise.")

    device = _get_current_device()

    # Only rank 0 prepares the payload (excluding self)
    if rank == 0:
        payload_obj = (args[1:], kwargs)
        payload_gpu = encode_object_to_cuda_tensor(payload_obj, device)
        size_gpu = torch.tensor([payload_gpu.numel()], dtype=torch.int64, device=device)
    else:
        size_gpu = torch.zeros(1, dtype=torch.int64, device=device)

    # Broadcast size then payload
    collective.broadcast(size_gpu, src_rank=0, group_name=group_name)
    num_bytes = int(size_gpu.item())

    if rank == 0:
        buf = payload_gpu
    else:
        buf = torch.empty(num_bytes, dtype=torch.uint8, device=device)

    collective.broadcast(buf, src_rank=0, group_name=group_name)

    if rank == 0:
        return args, kwargs
    else:
        b_args, b_kwargs = decode_cuda_tensor_to_object(buf)
        new_args = (args[0],) + tuple(b_args)
        return new_args, b_kwargs


def collect_one_to_all_via_cc(group_name, result):
    """Send data from all ranks to rank 0 using Ray collective communication.

    This function implements an efficient GPU-based gather operation where all ranks
    send their data to rank 0. The data is serialized and transferred via CUDA tensors.

    Args:
        group_name (str): Name of the collective group
        result: Data to be sent to rank 0

    Returns:
        If rank 0: List of results gathered from all ranks
        If other rank: None

    Raises:
        RuntimeError: If CUDA is not available
    """
    rank = collective.get_rank(group_name)
    world_size = collective.get_collective_group_size(group_name)

    use_cuda = torch.cuda.is_available()
    if not use_cuda:
        raise RuntimeError("GPU staging requires CUDA; fall back to CPU/GLOO path otherwise.")

    device = _get_current_device()

    # 1) Serialize payload and stage on GPU
    payload_gpu = encode_object_to_cuda_tensor(result, device)

    # 2) Share sizes via GPU (CUDA Long tensor)
    size_gpu = torch.tensor([payload_gpu.numel()], dtype=torch.int64, device=device)
    size_list_gpu = [torch.empty_like(size_gpu) for _ in range(world_size)]
    collective.allgather(size_list_gpu, size_gpu, group_name=group_name)

    if rank == 0:
        # 3) Receive from all ranks sequentially (NCCL is NOT thread-safe)
        # We keep NCCL operations sequential but parallelize the CPU-bound decode
        outputs = [None] * world_size
        outputs[0] = result

        # Pre-allocate buffers and receive sequentially
        buffers = []
        for src in range(1, world_size):
            n = int(size_list_gpu[src].item())
            buf = torch.empty(n, dtype=torch.uint8, device=device)
            collective.recv(buf, src_rank=src, group_name=group_name)
            buffers.append(buf)

        # 4) Parallel decode (CPU-bound pickle.loads is safe to parallelize)
        num_workers = min(len(buffers), 8) if buffers else 1
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            decoded = list(executor.map(decode_cuda_tensor_to_object, buffers))

        for i, src in enumerate(range(1, world_size)):
            outputs[src] = decoded[i]

        return outputs
    else:
        # 4) Send GPU buffer to rank 0
        collective.send(payload_gpu, dst_rank=0, group_name=group_name)
        return None


def dispatch_all_to_all_via_cc(group_name, args, kwargs):
    """Distribute different data to each rank using Ray collective communication.

    This implementation uses point-to-point communication to ensure each rank
    gets its specific data portion efficiently via GPU staging.

    Args:
        group_name (str): Name of the collective group
        args: Tuple where each element should be a list of length world_size
        kwargs: Dictionary where each value should be a list of length world_size

    Returns:
        Tuple of (args, kwargs) transformed for the current rank
    """
    rank = collective.get_rank(group_name)
    world_size = collective.get_collective_group_size(group_name)
    use_cuda = torch.cuda.is_available()
    if not use_cuda:
        raise RuntimeError("GPU staging requires CUDA; fall back to CPU/GLOO path otherwise.")

    device = _get_current_device()

    # Use scatter pattern from rank 0
    if rank == 0:
        # Rank 0 distributes data to all ranks (except itself)
        for target_rank in range(1, world_size):
            # Prepare data to send to this specific rank
            target_args = []
            # Skip the first arg (unpicklable class)
            for arg in args[1:]:
                if isinstance(arg, list) and len(arg) == world_size:
                    target_args.append(arg[target_rank])
                else:
                    target_args.append(arg)

            target_kwargs = {}
            for key, val in kwargs.items():
                if isinstance(val, list) and len(val) == world_size:
                    target_kwargs[key] = val[target_rank]
                else:
                    target_kwargs[key] = val

            payload = (target_args, target_kwargs)
            payload_gpu = encode_object_to_cuda_tensor(payload, device)
            size_gpu = torch.tensor([payload_gpu.numel()], dtype=torch.int64, device=device)

            # Send size then payload
            collective.send(size_gpu, dst_rank=target_rank, group_name=group_name)
            collective.send(payload_gpu, dst_rank=target_rank, group_name=group_name)

        # Rank 0 keeps its own data (index 0) with original first arg
        rank_0_args = []
        for arg in args[1:]:
            if isinstance(arg, list) and len(arg) == world_size:
                rank_0_args.append(arg[0])
            else:
                rank_0_args.append(arg)

        rank_0_kwargs = {}
        for key, val in kwargs.items():
            if isinstance(val, list) and len(val) == world_size:
                rank_0_kwargs[key] = val[0]
            else:
                rank_0_kwargs[key] = val
        return (args[0],) + tuple(rank_0_args), rank_0_kwargs
    else:
        # Other ranks receive from rank 0
        size_gpu = torch.empty(1, dtype=torch.int64, device=device)
        collective.recv(size_gpu, src_rank=0, group_name=group_name)

        num_bytes = int(size_gpu.item())
        buf = torch.empty(num_bytes, dtype=torch.uint8, device=device)
        collective.recv(buf, src_rank=0, group_name=group_name)

        recv_args, recv_kwargs = decode_cuda_tensor_to_object(buf)
        # Prepend the original first arg (unpicklable class)
        return (args[0],) + tuple(recv_args), recv_kwargs
