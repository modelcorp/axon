# Copyright 2025 Model AI Corp.
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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
# Adapted from Megatron-Core dist_checkpointing zarr strategy (github.com/NVIDIA/Megatron-LM), BSD-3-Clause.
import io
from functools import partial
from logging import getLogger
from pathlib import Path

import numpy as np
import torch
import zarr
from megatron.core.dist_checkpointing.core import CheckpointingException
from megatron.core.dist_checkpointing.dict_utils import nested_values
from megatron.core.dist_checkpointing.mapping import ShardedObject, ShardedStateDict, ShardedTensor, is_main_replica
from megatron.core.dist_checkpointing.strategies.async_utils import AsyncRequest
from megatron.core.dist_checkpointing.strategies.base import (
    AsyncSaveShardedStrategy,
)

logger = getLogger(__name__)

try:
    import zarr

    HAVE_ZARR = True
except ImportError:
    zarr = None
    HAVE_ZARR = False

# Dtype mappings
numpy_to_torch_dtype_dict = {
    np.dtype("bool"): torch.bool,
    np.dtype("uint8"): torch.uint8,
    np.dtype("int8"): torch.int8,
    np.dtype("int16"): torch.int16,
    np.dtype("int32"): torch.int32,
    np.dtype("int64"): torch.int64,
    np.dtype("float16"): torch.float16,
    np.dtype("float32"): torch.float32,
    np.dtype("float64"): torch.float64,
    np.dtype("complex64"): torch.complex64,
    np.dtype("complex128"): torch.complex128,
}

torch_to_numpy_dtype_dict = {v: k for k, v in numpy_to_torch_dtype_dict.items()}


class ZarrSaveContext:
    """Context object for async save operations."""

    def __init__(
        self, array_paths, sharded_tensors, sharded_objects, object_buffers, checkpoint_dir, synchronizer_enabled
    ):
        self.array_paths = array_paths
        self.sharded_tensors = sharded_tensors
        self.sharded_objects = sharded_objects
        self.object_buffers = object_buffers  # Pre-serialized object data
        self.checkpoint_dir = checkpoint_dir
        self.synchronizer_enabled = synchronizer_enabled


def _zarr_save_fn(context: ZarrSaveContext, preloaded_tensors, extra_arg=None):
    if preloaded_tensors is None:
        return

    for ten, arr_path, preloaded in zip(context.sharded_tensors, context.array_paths, preloaded_tensors, strict=False):
        if not is_main_replica(ten.replica_id):
            continue
        if preloaded is None or arr_path is None:
            continue

        # Re-open array in the worker
        open_kwargs = {}
        if context.synchronizer_enabled:
            open_kwargs["synchronizer"] = zarr.ProcessSynchronizer(str(context.checkpoint_dir / f"{ten.key}.sync"))
        arr = zarr.open(str(arr_path), mode="r+", **open_kwargs)

        if ten.flattened_range is None:
            arr[ten.global_slice()] = preloaded
        else:
            arr.set_coordinate_selection(ten.global_coordinates(), preloaded)

    # Write pre-serialized objects
    for obj, buffer in zip(context.sharded_objects, context.object_buffers, strict=False):
        if buffer is None:
            continue
        obj_dir = context.checkpoint_dir / obj.key
        obj_path = obj_dir / _get_shard_file(obj.global_offset, obj.global_shape)
        with open(obj_path, "wb") as f:
            f.write(buffer)


def _zarr_preload_fn(save_context: ZarrSaveContext):
    """Preload tensors to numpy arrays.

    This is a module-level function so it can be pickled.
    Objects are already serialized, so we only need to handle tensors here.
    """
    try:
        import tensorstore  # noqa: F401
    except Exception:
        pass
    preloaded_tensors = []
    for ten in save_context.sharded_tensors:
        if is_main_replica(ten.replica_id) and ten.data is not None:
            # Move tensor to CPU and convert to numpy
            x = ten.data.detach().cpu()

            # Convert to numpy
            if x.dtype == torch.bfloat16:
                x = x.float().numpy().astype("bfloat16")
            else:
                x = x.numpy()

            preloaded_tensors.append(x)
        else:
            preloaded_tensors.append(None)

    return preloaded_tensors


def _zarr_finalize_fn():
    """Finalize the save operation.

    This is a module-level function so it can be pickled.
    """
    # Ensure GPU operations are complete (works on both CUDA and ROCm)
    from axon.utils.torch import get_torch_device

    try:
        get_torch_device().synchronize()
    except Exception:
        pass
    # Synchronize across all ranks
    torch.distributed.barrier()
    logger.debug("Zarr async save finalized")


class ZarrAsyncSaveShardedStrategy(AsyncSaveShardedStrategy):
    """Async save strategy for Zarr backend with sharded object support."""

    def __init__(self, backend: str, version: int, thread_count: int = 2, synchronizer_enabled: bool = True):
        """Initialize async Zarr save strategy.

        Args:
            backend (str): format backend string
            version (int): format version
            thread_count (int, optional): number of threads for parallel saving. Defaults to 2.
            synchronizer_enabled (bool, optional): whether to use process synchronizers for
                concurrent writes. Defaults to True.
        """
        super().__init__(backend, version)
        self.thread_count = thread_count
        self.synchronizer_enabled = synchronizer_enabled
        logger.warning(
            "`zarr` distributed checkpoint backend is deprecated."
            " Please switch to PyTorch Distributed format (`torch_dist`)."
        )

    def async_save(self, sharded_state_dict: ShardedStateDict, checkpoint_dir: str | Path) -> AsyncRequest:
        """Async save sharded state dict to Zarr format with object support.

        This method prepares the save operation and returns an AsyncRequest that can be
        executed asynchronously. The save is split into three phases:
        1. Planning: Create/open zarr arrays and serialize objects
        2. Execution: Write tensor data to arrays and object buffers to files (async)
        3. Finalization: Synchronization and cleanup

        Args:
            sharded_state_dict (ShardedStateDict): sharded state dict to save
            checkpoint_dir (Union[str, Path]): checkpoint directory

        Returns:
            AsyncRequest: Request object containing save function and finalization callbacks
        """
        if isinstance(checkpoint_dir, str):
            checkpoint_dir = Path(checkpoint_dir)

        # Separate tensors and objects
        sharded_tensors = []
        sharded_objects = []

        for value in nested_values(sharded_state_dict):
            if isinstance(value, ShardedTensor):
                sharded_tensors.append(value)
            elif isinstance(value, ShardedObject):
                sharded_objects.append(value)

        # Planning phase - prepare arrays for tensors
        arrays = self._create_or_open_zarr_arrays_async(sharded_tensors, checkpoint_dir)

        # Create directories for sharded objects
        self._prepare_object_directories(sharded_objects, checkpoint_dir)

        # Pre-serialize objects in the main process to avoid CUDA issues
        object_buffers = self._serialize_objects(sharded_objects)

        # Convert arrays to paths to avoid pickling issues with bfloat16
        # The subprocess will reopen arrays from these paths
        array_paths = []
        for arr, ten in zip(arrays, sharded_tensors, strict=False):
            if arr is None:
                array_paths.append(None)
            else:
                # Store the path instead of the array object
                array_paths.append(checkpoint_dir / ten.key)

        # Create context object with paths instead of arrays
        save_context = ZarrSaveContext(
            array_paths=array_paths,
            sharded_tensors=sharded_tensors,
            sharded_objects=sharded_objects,
            object_buffers=object_buffers,
            checkpoint_dir=checkpoint_dir,
            synchronizer_enabled=self.synchronizer_enabled,
        )

        # Use partial to bind the context to the preload function
        preload_fn_with_context = partial(_zarr_preload_fn, save_context)

        # Return AsyncRequest with module-level functions that can be pickled
        return AsyncRequest(
            async_fn=_zarr_save_fn,  # Module-level function
            async_fn_args=(save_context, None, None),  # 3 arguments, 2nd will be replaced
            finalize_fns=[_zarr_finalize_fn],  # Module-level function
            preload_fn=preload_fn_with_context,  # Partial with bound context
        )

    def _serialize_objects(self, sharded_objects: list[ShardedObject]) -> list[bytes | None]:
        """Serialize objects to byte buffers in the main process.

        This avoids CUDA issues by serializing objects before forking.
        """
        object_buffers = []
        for obj in sharded_objects:
            if is_main_replica(obj.replica_id) and obj.data is not None:
                # Move object to CPU first
                cpu_data = _move_to_cpu(obj.data)

                # Serialize to bytes buffer
                buffer = io.BytesIO()
                torch.save(cpu_data, buffer)
                object_buffers.append(buffer.getvalue())
            else:
                object_buffers.append(None)

        return object_buffers

    def _prepare_object_directories(self, sharded_objects: list[ShardedObject], checkpoint_dir: Path):
        """Create directories for sharded objects."""
        created_dirs = set()
        for obj in sharded_objects:
            if not is_main_replica(obj.replica_id):
                continue
            obj_dir = checkpoint_dir / obj.key
            if obj_dir not in created_dirs:
                obj_dir.mkdir(parents=True, exist_ok=True)
                created_dirs.add(obj_dir)

    def _create_or_open_zarr_arrays_async(
        self, sharded_tensors: list[ShardedTensor], checkpoint_dir: Path
    ) -> list[zarr.Array | None]:
        """Async version of creating/opening zarr arrays.

        Similar to the sync version but optimized for async execution.
        """
        if not sharded_tensors:
            return []

        arrays = []

        # First pass - create arrays for first chunks
        for ten in sharded_tensors:
            arr = self._create_zarr_array_async(ten, checkpoint_dir) if self._should_create_array(ten) else None
            arrays.append(arr)

        # Synchronize to ensure all arrays are created
        torch.distributed.barrier()

        # Second pass - open arrays created by other processes
        for arr_idx, ten in enumerate(sharded_tensors):
            if arrays[arr_idx] is not None:
                # Array created by this process
                assert self._should_create_array(ten), ten
                continue
            if not is_main_replica(ten.replica_id):
                # This array won't be needed for saving
                continue

            # Open existing array
            open_kwargs = {}
            if ten.flattened_range is not None and self.synchronizer_enabled:
                open_kwargs["synchronizer"] = zarr.ProcessSynchronizer(str(checkpoint_dir / f"{ten.key}.sync"))
            arrays[arr_idx] = self._open_zarr_array_async(checkpoint_dir / ten.key, "r+", **open_kwargs)

        return arrays

    def _should_create_array(self, ten: ShardedTensor) -> bool:
        """Check if this rank should create the zarr array."""
        return (
            is_main_replica(ten.replica_id)
            and set(ten.global_offset) == {0}
            and (ten.flattened_range is None or ten.flattened_range.start == 0)
        )

    def _create_zarr_array_async(self, sharded_tensor: ShardedTensor, checkpoint_dir: Path) -> zarr.Array:
        try:
            import tensorstore  # noqa: F401

            HAS_BFLOAT16 = True
            numpy_to_torch_dtype_dict[np.dtype("bfloat16")] = torch.bfloat16
            torch_to_numpy_dtype_dict[torch.bfloat16] = np.dtype("bfloat16")
        except Exception:
            HAS_BFLOAT16 = False

        np_dtype = torch_to_numpy_dtype_dict[sharded_tensor.dtype]
        path = checkpoint_dir / sharded_tensor.key

        # Check if array already exists
        array_exists = (path / ".zarray").exists()

        # Always use a synchronizer when multiple processes may touch metadata.
        synchronizer = None
        if self.synchronizer_enabled:
            synchronizer = zarr.ProcessSynchronizer(str(checkpoint_dir / f"{sharded_tensor.key}.sync"))

        # Special handling for bfloat16
        if HAS_BFLOAT16 and np_dtype == np.dtype("bfloat16") and array_exists:
            # For existing bfloat16 arrays, open without dtype specification
            # zarr will read it as |V2 which is expected
            arr = zarr.open_array(
                str(path),
                mode="r+",
                synchronizer=synchronizer,
            )
            # Manually set the dtype attribute for our use
            arr._dtype = np_dtype
        else:
            # For new arrays or non-bfloat16, use normal creation
            arr = zarr.open_array(
                str(path),
                mode="a",
                shape=sharded_tensor.global_shape,
                dtype=np_dtype,
                chunks=sharded_tensor.max_allowed_chunks(),
                compressor=None,
                fill_value=None,
                write_empty_chunks=True,
                synchronizer=synchronizer,
            )

        # Validate shape
        if arr.shape != tuple(sharded_tensor.global_shape):
            raise CheckpointingException(
                f"Existing array at {path} has shape {arr.shape}, expected {tuple(sharded_tensor.global_shape)}"
            )

        # Skip dtype validation for bfloat16 as zarr will always report it as |V2
        if not (HAS_BFLOAT16 and np_dtype == np.dtype("bfloat16")):
            if arr.dtype != np_dtype:
                raise CheckpointingException(f"Existing array at {path} has dtype {arr.dtype}, expected {np_dtype}")

        # For new bfloat16 arrays, patch the metadata
        if HAS_BFLOAT16 and np_dtype == np.dtype("bfloat16") and not array_exists:
            arr._dtype = np_dtype
            zarray = arr.store[".zarray"]
            arr.store[".zarray"] = zarray.replace(b"<V2", b"bfloat16")

        return arr

    def _open_zarr_array_async(self, path: Path, mode: str, **open_kwargs) -> zarr.Array:
        """Open an existing zarr array for async operations."""
        try:
            return zarr.open(str(path), mode, **open_kwargs)
        except zarr.errors.PathNotFoundError as e:
            ckpt_dir = path.parent
            err_msg = f"Array {path} not found"
            if ckpt_dir.exists():
                ckpt_files = [f.name for f in ckpt_dir.iterdir()]
                logger.debug(f"{err_msg}. Checkpoint directory {ckpt_dir} content: {ckpt_files}")
            else:
                err_msg += f". Checkpoint directory {ckpt_dir} does not exist."
            raise CheckpointingException(err_msg) from e

    def can_handle_sharded_objects(self):
        """Zarr strategy now handles sharded objects."""
        return True


def _get_shard_file(global_offset: tuple[int, ...], global_shape: tuple[int, ...]) -> str:
    """Generate shard filename based on offset and shape.

    Creates filenames matching the expected pattern:
    - 1D: 'shard_0_94.pt' (offset_total)
    - 2D: 'shard_47.96_94.128.pt' (offset1.offset2_shape1.shape2)
    - ND: 'shard_o1.o2.o3_s1.s2.s3.pt'

    Uses dots (.) to separate dimensions within offset/shape,
    and underscore (_) to separate offset from shape.
    """
    # Join dimensions with dots for both offset and shape
    offset_str = ".".join(str(o) for o in global_offset)
    shape_str = ".".join(str(s) for s in global_shape)

    # Combine with underscore between offset and shape
    return f"shard_{offset_str}_{shape_str}.pt"


def _move_to_cpu(data):
    """Recursively move data to CPU and remove CUDA state."""
    if isinstance(data, torch.Tensor):
        return data.detach().cpu()
    elif isinstance(data, torch.Generator):
        # For generators, create a fresh CPU generator with a deterministic seed
        # We can't transfer CUDA generator state to CPU due to incompatible formats
        cpu_gen = torch.Generator("cpu")
        # Use a deterministic seed based on the original generator if possible
        # This ensures reproducibility even though we can't preserve exact state
        cpu_gen.manual_seed(42)  # Or derive from rank/iteration if available
        return cpu_gen
    elif isinstance(data, dict):
        # Clean dictionary, handling special objects
        clean_dict = {}
        for k, v in data.items():
            # Check for generator objects by type
            if isinstance(v, torch.Generator):
                # Handle Generator objects specially
                if v.device.type == "cuda":
                    # Can't transfer CUDA generator state to CPU
                    # Create a new CPU generator with deterministic seed
                    cpu_gen = torch.Generator("cpu")
                    cpu_gen.manual_seed(42)  # Use a deterministic seed
                    clean_dict[k] = cpu_gen
                else:
                    # Already a CPU generator, keep it
                    clean_dict[k] = v
            elif hasattr(v, "__class__") and "Generator" in str(v.__class__):
                # Skip other generator types that we can't handle
                logger.debug(f"Skipping generator object of type {v.__class__}")
                continue
            else:
                # Recursively process other data
                clean_dict[k] = _move_to_cpu(v)
        return clean_dict
    elif isinstance(data, list | tuple):
        # Process sequences recursively
        clean_data = []
        for v in data:
            if isinstance(v, torch.Generator) and v.device.type == "cuda":
                # Replace CUDA generators with CPU ones
                cpu_gen = torch.Generator("cpu")
                cpu_gen.manual_seed(42)
                clean_data.append(cpu_gen)
            else:
                clean_data.append(_move_to_cpu(v))
        return type(data)(clean_data)
    else:
        # Return other types as-is
        return data
