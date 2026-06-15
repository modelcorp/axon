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
# Adapted from verl protocol.py (github.com/volcengine/verl), Apache-2.0.
"""
Base data transfer protocol for communication between functions and modules.

This module provides DataProto, a standardized data structure for exchanging
tensor and non-tensor data between components in a distributed training system.
"""

import contextlib
import copy
import io
import logging
import math
import os
import pickle
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import ray
import tensordict
import torch
import torch.distributed
from packaging.version import parse as parse_version
from tensordict import TensorDict
from torch.utils.data import DataLoader

from axon.utils.py_utils import union_two_dict
from axon.utils.torch import get_device_id, get_torch_device
from axon.utils.torch.ops import allgather_dict_tensors

__all__ = ["DataProto", "union_tensor_dict"]

_padding_size_key = "_padding_size_key_x123d"

with contextlib.suppress(Exception):
    tensordict.set_lazy_legacy(False).set()
    if parse_version(tensordict.__version__) < parse_version("0.10.0"):
        tensordict.set_list_to_stack(True).set()


class _DataProtoConfigMeta(type):
    """Metaclass for DataProtoConfig providing class-level configuration properties."""

    _config = {}
    auto_padding_key = "_axon_auto_padding"

    @property
    def auto_padding(cls) -> bool:
        """Check if auto-padding is enabled via environment or config."""
        env_enabled = os.getenv("AXON_AUTO_PADDING", "FALSE").upper() in ("TRUE", "1")
        return env_enabled or cls._config.get(cls.auto_padding_key, False)

    @auto_padding.setter
    def auto_padding(cls, enabled: bool):
        assert isinstance(enabled, bool), f"enabled must be bool, got {type(enabled)}"
        cls._config[cls.auto_padding_key] = enabled


class DataProtoConfig(metaclass=_DataProtoConfigMeta):
    """Configuration class for DataProto behavior."""

    pass


def pad_dataproto_to_divisor(data: "DataProto", size_divisor: int) -> tuple["DataProto", int]:
    """Pad a DataProto to make its size divisible by size_divisor.

    Args:
        data: The DataProto to pad.
        size_divisor: The divisor that the final size should be divisible by.

    Returns:
        A tuple of (padded DataProto, padding size added).
    """
    assert isinstance(data, DataProto), "data must be a DataProto"
    remainder = len(data) % size_divisor
    if remainder == 0:
        if len(data) == 0:
            logging.warning("padding a DataProto with no item, no change made")
        return data, 0

    pad_size = size_divisor - remainder
    padding_protos, remaining = [], pad_size
    while remaining > 0:
        take_size = min(remaining, len(data))
        padding_protos.append(copy.deepcopy(data[:take_size]))
        remaining -= take_size
    return DataProto.concat([data] + padding_protos), pad_size


def unpad_dataproto(data: "DataProto", pad_size: int) -> "DataProto":
    """Remove padding from a DataProto.

    Args:
        data: The padded DataProto.
        pad_size: Number of padding elements to remove from the end.

    Returns:
        The unpadded DataProto.
    """
    return data[:-pad_size] if pad_size else data


def union_tensor_dict(tensor_dict1: TensorDict, tensor_dict2: TensorDict) -> TensorDict:
    """Union two TensorDicts, merging keys from tensor_dict2 into tensor_dict1.

    Args:
        tensor_dict1: The primary TensorDict to merge into (modified in place).
        tensor_dict2: The TensorDict to merge from.

    Returns:
        The merged TensorDict (same object as tensor_dict1).

    Raises:
        AssertionError: If batch sizes differ or conflicting keys have different values.
    """
    assert tensor_dict1.batch_size == tensor_dict2.batch_size, (
        f"Batch sizes must match: {tensor_dict1.batch_size} vs {tensor_dict2.batch_size}"
    )
    if tensor_dict1.is_locked:
        tensor_dict1.unlock_()
    for key in tensor_dict2.keys():
        if key in tensor_dict1.keys():
            assert _deep_equal(tensor_dict1[key], tensor_dict2[key], set()), f"Conflicting values for key '{key}'"
        else:
            tensor_dict1[key] = tensor_dict2[key]
    return tensor_dict1


def _array_equal(array1: np.ndarray, array2: np.ndarray, visited: set[int]) -> bool:
    """Compare two NumPy arrays for equality, handling object dtypes and NaN values.

    Args:
        array1: First array to compare.
        array2: Second array to compare.
        visited: Set of visited object IDs for cycle detection.

    Returns:
        True if arrays have equal dtype, shape, and all elements match.
    """
    if array1.dtype != array2.dtype or array1.shape != array2.shape:
        return False
    if array1.dtype != "object":
        return np.array_equal(array1, array2, equal_nan=True)
    return all(_deep_equal(x, y, visited) for x, y in zip(array1.flat, array2.flat, strict=False))


def _deep_equal(a: Any, b: Any, visited: set[int]) -> bool:
    """Recursively compare two objects for equality with NaN and cycle handling.

    Args:
        a: First object to compare.
        b: Second object to compare.
        visited: Set of visited object IDs for cycle detection.

    Returns:
        True if objects are deeply equal.
    """
    if type(a) is not type(b):
        return False
    obj_id = id(a)
    if obj_id in visited:
        return True
    visited.add(obj_id)
    try:
        if isinstance(a, float) and math.isnan(a) and math.isnan(b):
            return True
        if isinstance(a, torch.Tensor):
            if a.shape != b.shape or a.dtype != b.dtype:
                return False
            if a.is_floating_point():
                nan_mask_a = torch.isnan(a)
                nan_mask_b = torch.isnan(b)
                if not torch.equal(nan_mask_a, nan_mask_b):
                    return False
                non_nan = ~nan_mask_a
                return torch.equal(a[non_nan], b[non_nan]) if non_nan.any() else True
            return torch.equal(a, b)
        if isinstance(a, np.ndarray):
            return _array_equal(a, b, visited)
        return a == b
    finally:
        visited.remove(obj_id)


def union_numpy_dict(dict1: dict[str, np.ndarray], dict2: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Merge dict2 into dict1, asserting equal values for overlapping keys.

    Args:
        dict1: Primary dict to merge into (modified in place).
        dict2: Dict to merge from.

    Returns:
        The merged dict (same object as dict1).
    """
    for key, val in dict2.items():
        if key in dict1:
            assert isinstance(dict1[key], np.ndarray) and isinstance(val, np.ndarray)
            assert _deep_equal(dict1[key], val, set()), f"Conflicting values for key '{key}'"
        dict1[key] = val
    return dict1


def list_of_dict_to_dict_of_list(list_of_dict: list[dict]) -> dict:
    """Transpose a list of dicts to a dict of lists.

    Args:
        list_of_dict: List of dicts with identical keys.

    Returns:
        Dict mapping each key to a list of values from all input dicts.
    """
    if not list_of_dict:
        return {}
    keys = set().union(*(d.keys() for d in list_of_dict))
    return {key: [d[key] for d in list_of_dict if key in d] for key in keys}


def fold_batch_dim(data: "DataProto", new_batch_size: int) -> "DataProto":
    """Fold batch dimension from [bsz, ...] to [new_bsz, bsz // new_bsz, ...].

    Args:
        data: DataProto to reshape.
        new_batch_size: New first dimension size.

    Returns:
        New DataProto with folded batch dimension.
    """
    assert data.batch.batch_size[0] % new_batch_size == 0
    tensor = data.batch.view(new_batch_size, -1)
    tensor.auto_batch_size_(batch_dims=1)
    non_tensor = {k: np.reshape(v, (new_batch_size, -1, *v.shape[1:])) for k, v in data.non_tensor_batch.items()}
    return type(data)(batch=tensor, non_tensor_batch=non_tensor, meta_info=data.meta_info)


def unfold_batch_dim(data: "DataProto", batch_dims: int = 2) -> "DataProto":
    """Unfold first n dimensions into a single batch dimension.

    Args:
        data: DataProto to reshape.
        batch_dims: Number of leading dimensions to flatten.

    Returns:
        New DataProto with unfolded batch dimension.
    """
    tensor = data.batch
    tensor.auto_batch_size_(batch_dims=batch_dims)
    tensor = tensor.view(-1)
    batch_size = tensor.batch_size[0]
    non_tensor = {k: np.reshape(v, (batch_size, *v.shape[batch_dims:])) for k, v in data.non_tensor_batch.items()}
    return type(data)(batch=tensor, non_tensor_batch=non_tensor, meta_info=data.meta_info)


def serialize_single_tensor(obj: torch.Tensor) -> tuple[str, tuple[int, ...], np.ndarray]:
    """Serialize a single tensor to a tuple of (dtype_str, shape, bytes_as_numpy).

    Args:
        obj: Tensor to serialize.

    Returns:
        Tuple of (dtype string, shape tuple, numpy uint8 array of bytes).
    """
    return str(obj.dtype).removeprefix("torch."), tuple(obj.shape), obj.flatten().contiguous().view(torch.uint8).numpy()


def serialize_tensordict(batch: TensorDict) -> tuple[tuple[int, ...], str | None, dict[str, Any]]:
    """Serialize a TensorDict to a picklable tuple.

    Args:
        batch: TensorDict to serialize.

    Returns:
        Tuple of (batch_size, device_str, encoded_items_dict).
    """
    encoded = {}
    for k, v in batch.items():
        if v.is_nested:
            encoded[k] = (str(v.layout).removeprefix("torch."), [serialize_single_tensor(t) for t in v.unbind()])
        else:
            encoded[k] = serialize_single_tensor(v)
    return tuple(batch.batch_size), str(batch.device) if batch.device else None, encoded


def deserialize_single_tensor(arr: tuple) -> torch.Tensor:
    """Deserialize a tensor from its serialized tuple form.

    Args:
        arr: Tuple of (dtype_str, shape, bytes_array).

    Returns:
        Reconstructed tensor.
    """
    dtype, shape, data = arr
    torch_dtype = getattr(torch, dtype)
    return torch.frombuffer(bytearray(data), dtype=torch.uint8).view(torch_dtype).view(shape)


def deserialize_tensordict(arr: tuple) -> TensorDict:
    """Deserialize a TensorDict from its serialized tuple form.

    Args:
        arr: Tuple of (batch_size, device_str, encoded_items).

    Returns:
        Reconstructed TensorDict.
    """
    batch_size, device, encoded = arr
    decoded = {}
    for k, v in encoded.items():
        if len(v) == 2:  # nested tensor: (layout, list of tensors)
            decoded[k] = torch.nested.as_nested_tensor(
                [deserialize_single_tensor(t) for t in v[1]], layout=getattr(torch, v[0])
            )
        elif len(v) == 3:  # regular tensor
            decoded[k] = deserialize_single_tensor(v)
        else:
            raise ValueError(f"Invalid encoding format, expected length 2 or 3, got {len(v)}")
    return TensorDict(source=decoded, batch_size=batch_size, device=device)


def collate_fn(items: list["DataProtoItem"]) -> "DataProto":
    """Collate DataProtoItems into a single DataProto batch.

    Args:
        items: List of DataProtoItem objects to collate.

    Returns:
        A DataProto containing the batched data.
    """
    batch = torch.stack([item.batch for item in items]).contiguous()
    non_tensor = list_of_dict_to_dict_of_list([item.non_tensor_batch for item in items])
    non_tensor = {k: np.array(v, dtype=object) for k, v in non_tensor.items()}
    return DataProto(batch=batch, non_tensor_batch=non_tensor)


@dataclass
class DataProtoItem:
    """Single item from a DataProto, returned when indexing with an integer."""

    batch: TensorDict = None
    non_tensor_batch: dict = field(default_factory=dict)
    meta_info: dict = field(default_factory=dict)


@dataclass
class DataProto:
    """Standard protocol for data exchange between functions in distributed training.

    DataProto wraps a TensorDict (for GPU tensors) and a dict of numpy arrays
    (for non-tensor data like strings), along with metadata. It provides
    unified slicing, concatenation, and serialization across both data types.

    Attributes:
        batch: TensorDict containing tensor data with a single batch dimension.
        non_tensor_batch: Dict of numpy arrays (dtype=object) for non-tensor data.
        meta_info: Dict of metadata that doesn't vary across the batch.
    """

    batch: TensorDict = None
    non_tensor_batch: dict = field(default_factory=dict)
    meta_info: dict = field(default_factory=dict)

    def __post_init__(self):
        self.check_consistency()

    def __len__(self) -> int:
        if self.batch is not None:
            return self.batch.batch_size[0]
        if self.non_tensor_batch:
            return next(iter(self.non_tensor_batch.values())).shape[0]
        return 0

    def __getitem__(self, item) -> "DataProto | DataProtoItem":
        """Index into the DataProto.

        Args:
            item: Index specification - int, slice, list, ndarray, or Tensor.

        Returns:
            DataProtoItem for single integer index, DataProto for all other types.

        Raises:
            TypeError: If item type is not supported.
        """
        if isinstance(item, slice):
            return self.slice(item.start, item.stop, item.step)
        if isinstance(item, list | np.ndarray | torch.Tensor):
            return self.select_idxs(item)
        if isinstance(item, int | np.integer):
            return DataProtoItem(
                batch=self.batch[item] if self.batch is not None else None,
                non_tensor_batch={k: v[item] for k, v in self.non_tensor_batch.items()},
                meta_info=self.meta_info,
            )
        raise TypeError(f"Indexing with {type(item)} is not supported")

    def __getstate__(self) -> tuple:
        """Serialize for pickling, supporting both numpy and torch serialization methods."""
        batch = self.batch
        if batch is not None and parse_version(tensordict.__version__) >= parse_version("0.5.0"):
            batch = batch.contiguous().consolidate() if batch.keys() else batch

        if os.getenv("AXON_DATAPROTO_SERIALIZATION_METHOD") == "numpy":
            return (serialize_tensordict(batch) if batch else None, self.non_tensor_batch, self.meta_info)

        buffer = io.BytesIO()
        torch.save(batch, buffer)
        return buffer.getvalue(), self.non_tensor_batch, self.meta_info

    def __setstate__(self, data: tuple):
        """Deserialize from pickled state."""
        batch_data, self.non_tensor_batch, self.meta_info = data
        if os.getenv("AXON_DATAPROTO_SERIALIZATION_METHOD") == "numpy":
            self.batch = deserialize_tensordict(batch_data) if batch_data else None
        else:
            self.batch = torch.load(  # nosec B614
                io.BytesIO(batch_data),
                weights_only=False,
                map_location="cpu" if not get_torch_device().is_available() else None,
            )

    def save_to_disk(self, filepath: str):
        """Save this DataProto to disk using pickle.

        Args:
            filepath: Path to save the file.
        """
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load_from_disk(filepath: str) -> "DataProto":
        """Load a DataProto from disk.

        Args:
            filepath: Path to the saved file.

        Returns:
            The loaded DataProto.
        """
        with open(filepath, "rb") as f:
            return pickle.load(f)  # nosec B301

    def print_size(self, prefix: str = ""):
        """Print the memory size of this DataProto.

        Args:
            prefix: Optional prefix string for the output message.
        """
        tensor_gb = (
            sum(t.element_size() * t.numel() for t in self.batch.values()) / 1024**3 if self.batch is not None else 0
        )
        numpy_gb = sum(arr.nbytes for arr in self.non_tensor_batch.values()) / 1024**3
        msg = f"Size of tensordict: {tensor_gb} GB, size of non_tensor_batch: {numpy_gb} GB"
        print(f"{prefix}, {msg}" if prefix else msg)

    def check_consistency(self):
        """Validate internal consistency of batch and non_tensor_batch.

        Raises:
            AssertionError: If batch has multiple batch dims, non_tensor_batch contains
                non-arrays, or batch sizes don't match.
        """
        if self.batch is not None:
            assert len(self.batch.batch_size) == 1, "only support num_batch_dims=1"
        if self.non_tensor_batch:
            for key, val in self.non_tensor_batch.items():
                assert isinstance(val, np.ndarray), f"non_tensor_batch['{key}'] must be ndarray, got {type(val)}"
        if self.batch is not None and self.non_tensor_batch:
            batch_size = self.batch.batch_size[0]
            for key, val in self.non_tensor_batch.items():
                assert val.shape[0] == batch_size, f"'{key}' has length {val.shape[0]}, expected {batch_size}"

    @classmethod
    def from_single_dict(
        cls, data: dict[str, torch.Tensor | np.ndarray], meta_info: dict = None, auto_padding: bool = False
    ) -> "DataProto":
        """Create a DataProto from a mixed dict of tensors and numpy arrays.

        Args:
            data: Dict mapping keys to torch.Tensor or np.ndarray values.
            meta_info: Optional metadata dict.
            auto_padding: Whether to enable auto-padding for this DataProto.

        Returns:
            New DataProto with tensors and non-tensors separated.

        Raises:
            ValueError: If data contains values that aren't Tensor or ndarray.
        """
        tensors, non_tensors = {}, {}
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                tensors[key] = val
            elif isinstance(val, np.ndarray):
                non_tensors[key] = val
            else:
                raise ValueError(f"Unsupported type for key '{key}': {type(val)}")
        return cls.from_dict(tensors=tensors, non_tensors=non_tensors, meta_info=meta_info, auto_padding=auto_padding)

    @classmethod
    def from_dict(
        cls,
        tensors: dict[str, torch.Tensor] | TensorDict | None = None,
        non_tensors: dict = None,
        meta_info: dict = None,
        num_batch_dims: int = 1,
        auto_padding: bool = False,
    ) -> "DataProto":
        """Create a DataProto from separate tensor and non-tensor dicts.

        Args:
            tensors: Dict of tensors or a TensorDict. All must have same batch size.
            non_tensors: Dict of non-tensor data (converted to numpy arrays).
            meta_info: Optional metadata dict.
            num_batch_dims: Number of leading dimensions to treat as batch.
            auto_padding: Whether to enable auto-padding.

        Returns:
            New DataProto instance.
        """
        assert num_batch_dims > 0, "num_batch_dims must be > 0"
        tensors = tensors if tensors is not None else {}
        non_tensors = non_tensors if non_tensors is not None else {}
        meta_info = meta_info if meta_info is not None else {}
        if non_tensors:
            assert num_batch_dims == 1, "num_batch_dims must be 1 when non_tensors provided"

        # Validate batch sizes
        batch_size, pivot_key = None, None
        for key, tensor in tensors.items() if isinstance(tensors, dict) else []:
            curr = tensor.shape[:num_batch_dims]
            if batch_size is None:
                batch_size, pivot_key = curr, key
            else:
                assert batch_size == curr, f"Batch size mismatch: '{pivot_key}' has {batch_size}, '{key}' has {curr}"

        non_tensors = {k: v if isinstance(v, np.ndarray) else np.array(v, dtype=object) for k, v in non_tensors.items()}
        tensor_dict = (
            tensors
            if isinstance(tensors, TensorDict)
            else (TensorDict(tensors, batch_size) if len(tensors) > 0 else None)
        )
        if auto_padding:
            meta_info[DataProtoConfig.auto_padding_key] = True
        return cls(batch=tensor_dict, non_tensor_batch=non_tensors, meta_info=meta_info)

    @classmethod
    def from_tensordict(
        cls, tensor_dict: TensorDict = None, meta_info: dict = None, num_batch_dims: int = 1
    ) -> "DataProto":
        """Create a DataProto from a TensorDict containing mixed data types.

        Args:
            tensor_dict: TensorDict potentially containing NonTensorData/NonTensorStack.
            meta_info: Optional metadata dict.
            num_batch_dims: Number of leading dimensions to treat as batch.

        Returns:
            New DataProto with tensor and non-tensor data separated.
        """
        assert parse_version(tensordict.__version__) >= parse_version("0.10.0"), "Requires tensordict >= 0.10.0"
        from tensordict import NonTensorData, NonTensorStack

        assert num_batch_dims > 0, "num_batch_dims must be > 0"
        if not all(isinstance(v, torch.Tensor) for v in tensor_dict.values()):
            assert num_batch_dims == 1, "num_batch_dims must be 1 when tensor_dict contains non-tensor data"

        meta_info = meta_info or {}
        batch, non_tensor_batch, batch_size = {}, {}, None
        for key, val in tensor_dict.items():
            if isinstance(val, torch.Tensor):
                batch[key] = val
                batch_size = batch_size or val.shape[:num_batch_dims]
            elif isinstance(val, NonTensorStack):
                non_tensor_batch[key] = np.array([elem.data for elem in val], dtype=object)
            elif isinstance(val, NonTensorData):
                meta_info[key] = val.data
        return cls(batch=TensorDict(batch, batch_size), non_tensor_batch=non_tensor_batch, meta_info=meta_info)

    def to(self, device) -> "DataProto":
        """Move the batch tensors to the specified device.

        Args:
            device: Target device (torch.device or string like 'cuda:0').

        Returns:
            Self, for method chaining.
        """
        if self.batch is not None:
            self.batch = self.batch.to(device)
        return self

    def select(
        self,
        batch_keys: list[str] = None,
        non_tensor_batch_keys: list[str] = None,
        meta_info_keys: list[str] = None,
        deepcopy: bool = False,
    ) -> "DataProto":
        """Select a subset of keys from the DataProto.

        Args:
            batch_keys: Keys to keep from batch (None keeps all).
            non_tensor_batch_keys: Keys to keep from non_tensor_batch (None keeps all).
            meta_info_keys: Keys to keep from meta_info (None keeps all).
            deepcopy: Whether to deep copy non_tensor_batch and meta_info.

        Returns:
            New DataProto with only the selected keys.
        """
        sub_batch = self.batch.select(*batch_keys) if batch_keys else self.batch
        non_tensor = (
            {k: v for k, v in self.non_tensor_batch.items() if k in non_tensor_batch_keys}
            if non_tensor_batch_keys
            else self.non_tensor_batch
        )
        sub_meta = (
            {k: v for k, v in self.meta_info.items() if k in meta_info_keys} if meta_info_keys else self.meta_info
        )
        if deepcopy:
            non_tensor, sub_meta = copy.deepcopy(non_tensor), copy.deepcopy(sub_meta)
        return type(self)(batch=sub_batch, non_tensor_batch=non_tensor, meta_info=sub_meta)

    def select_idxs(self, idxs: torch.Tensor | np.ndarray | list) -> "DataProto":
        """Select specific indices from the DataProto.

        Args:
            idxs: Indices to select (list, ndarray, or tensor). Can be integer indices or boolean mask.

        Returns:
            New DataProto containing only the selected indices.
        """
        if isinstance(idxs, list):
            idxs = torch.tensor(idxs, dtype=torch.int32 if not any(isinstance(i, bool) for i in idxs) else torch.bool)
        idxs_torch = torch.from_numpy(idxs) if isinstance(idxs, np.ndarray) else idxs
        idxs_np = idxs if isinstance(idxs, np.ndarray) else idxs_torch.detach().cpu().numpy()
        batch_size = int(idxs_np.sum()) if idxs_np.dtype == bool else idxs_np.shape[0]

        selected_batch = (
            TensorDict(
                {k: v[idxs_torch] for k, v in self.batch.items()}, batch_size=(batch_size,), device=self.batch.device
            )
            if self.batch is not None
            else None
        )
        return type(self)(
            batch=selected_batch,
            non_tensor_batch={k: v[idxs_np] for k, v in self.non_tensor_batch.items()},
            meta_info=self.meta_info,
        )

    def slice(self, start: int = None, end: int = None, step: int = None) -> "DataProto":
        """Slice the DataProto along the batch dimension.

        Args:
            start: Start index (None = beginning).
            end: End index, exclusive (None = end).
            step: Step size (None = 1).

        Returns:
            New DataProto with sliced data.

        Example:
            >>> data[10:20]      # Returns DataProto
            >>> data[::2]        # Every other element
            >>> data.slice(0, 5) # First 5 elements
        """
        s = slice(start, end, step)
        return type(self)(
            batch=self.batch[s] if self.batch is not None else None,
            non_tensor_batch={k: v[s] for k, v in self.non_tensor_batch.items()},
            meta_info=self.meta_info,
        )

    def pop(
        self, batch_keys: list[str] = None, non_tensor_batch_keys: list[str] = None, meta_info_keys: list[str] = None
    ) -> "DataProto":
        """Remove and return specified keys from this DataProto.

        Args:
            batch_keys: Keys to pop from batch.
            non_tensor_batch_keys: Keys to pop from non_tensor_batch.
            meta_info_keys: Keys to pop from meta_info.

        Returns:
            New DataProto containing the popped keys.

        Raises:
            AssertionError: If any specified key does not exist in its respective dict.
        """
        batch_keys, non_tensor_batch_keys, meta_info_keys = (
            batch_keys or [],
            non_tensor_batch_keys or [],
            meta_info_keys or [],
        )
        for k in batch_keys:
            assert k in self.batch.keys(), f"Key '{k}' not found in batch"
        for k in non_tensor_batch_keys:
            assert k in self.non_tensor_batch, f"Key '{k}' not found in non_tensor_batch"
        for k in meta_info_keys:
            assert k in self.meta_info, f"Key '{k}' not found in meta_info"
        return DataProto.from_dict(
            tensors={k: self.batch.pop(k) for k in batch_keys},
            non_tensors={k: self.non_tensor_batch.pop(k) for k in non_tensor_batch_keys},
            meta_info={k: self.meta_info.pop(k) for k in meta_info_keys},
        )

    def rename(self, old_keys: str | list[str] = None, new_keys: str | list[str] = None) -> "DataProto":
        """Rename keys in the batch.

        Args:
            old_keys: Key(s) to rename (string or list of strings).
            new_keys: New name(s) for the keys (must match length of old_keys).

        Returns:
            Self, for method chaining.

        Raises:
            TypeError: If keys are not strings or lists.
            ValueError: If old_keys and new_keys have different lengths.
        """

        def to_list(keys):
            if keys is None:
                return []
            if isinstance(keys, str):
                return [keys]
            if isinstance(keys, list):
                return keys
            raise TypeError(f"keys must be str or list, got {type(keys)}")

        old_keys, new_keys = to_list(old_keys), to_list(new_keys)
        if len(old_keys) != len(new_keys):
            raise ValueError(f"Length mismatch: {len(old_keys)} old keys vs {len(new_keys)} new keys")
        if old_keys:
            self.batch.rename_key_(tuple(old_keys), tuple(new_keys))
        return self

    def union(self, other: "DataProto") -> "DataProto":
        """Merge another DataProto into this one.

        Args:
            other: DataProto to merge from.

        Returns:
            Self, with merged data.

        Raises:
            AssertionError: If batch sizes differ or conflicting keys have different values.
        """
        self.batch = union_tensor_dict(self.batch, other.batch)
        self.non_tensor_batch = union_numpy_dict(self.non_tensor_batch, other.non_tensor_batch)
        self.meta_info = union_two_dict(self.meta_info, other.meta_info)
        return self

    def make_iterator(self, mini_batch_size: int, epochs: int, seed: int = None, dataloader_kwargs: dict = None):
        """Create an iterator that yields mini-batches over multiple epochs.

        Args:
            mini_batch_size: Size of each mini-batch. Must evenly divide batch size.
            epochs: Number of epochs to iterate.
            seed: Optional random seed for shuffling.
            dataloader_kwargs: Additional kwargs passed to PyTorch DataLoader.

        Returns:
            Iterator yielding DataProto mini-batches. Total iterations = batch_size * epochs / mini_batch_size.
        """
        assert self.batch.batch_size[0] % mini_batch_size == 0, f"{self.batch.batch_size[0]} % {mini_batch_size} != 0"
        generator = torch.Generator().manual_seed(seed) if seed is not None else None
        loader = DataLoader(
            self, batch_size=mini_batch_size, collate_fn=collate_fn, generator=generator, **(dataloader_kwargs or {})
        )

        def gen():
            for _ in range(epochs):
                for batch in loader:
                    batch.meta_info = self.meta_info
                    yield batch

        return iter(gen())

    def is_padding_enabled(self) -> bool:
        """Check if auto-padding is enabled for this DataProto.

        Returns:
            True if padding is enabled via meta_info or global config.
        """
        return self.meta_info.get(DataProtoConfig.auto_padding_key, False) or DataProtoConfig.auto_padding

    def padding(self, padding_size: int, padding_candidate: str = ""):
        """Pad the DataProto in-place by repeating elements.

        Args:
            padding_size: Number of padding elements to add.
            padding_candidate: "first" or "last" - which element to repeat for padding.
        """
        if padding_size == 0:
            return
        idx = 0 if padding_candidate == "first" else len(self) - 1
        padded = DataProto.concat([self, self.select_idxs([idx]).repeat(padding_size)])
        self.batch, self.non_tensor_batch = padded.batch, padded.non_tensor_batch

    def chunk(self, chunks: int) -> list["DataProto"]:
        """Split into a fixed number of chunks along batch dimension.

        Args:
            chunks: Number of chunks to create.

        Returns:
            List of DataProto chunks. Meta_info is shared across all chunks.
        """
        if not self.is_padding_enabled():
            assert len(self) % chunks == 0, f"Size {len(self)} not divisible by {chunks} (enable padding or use split)"

        if self.batch is not None:
            batch_lst = self.batch.chunk(chunks=chunks, dim=0)
            split_indices = np.cumsum([b.batch_size[0] for b in batch_lst])[:-1].tolist()
        else:
            batch_lst = [None] * chunks
            split_indices = chunks

        non_tensor_splits = [{} for _ in range(chunks)]
        for key, val in self.non_tensor_batch.items():
            for i, arr in enumerate(np.array_split(val, split_indices)):
                non_tensor_splits[i][key] = arr

        return [
            type(self)(batch=batch_lst[i], non_tensor_batch=non_tensor_splits[i], meta_info=self.meta_info)
            for i in range(chunks)
        ]

    def split(self, split_size: int) -> list["DataProto"]:
        """Split into chunks of a fixed size along batch dimension.

        Args:
            split_size: Maximum size of each chunk.

        Returns:
            List of DataProto chunks.
        """
        return [self[i : i + split_size] for i in range(0, len(self), split_size)]

    @staticmethod
    def concat(data: list["DataProto"]) -> "DataProto":
        """Concatenate multiple DataProto objects along the batch dimension.

        Args:
            data: List of DataProto to concatenate.

        Returns:
            New DataProto with concatenated batches and merged meta_info.
        """
        if not data:
            return DataProto()

        new_batch = torch.cat([d.batch for d in data], dim=0) if data[0].batch is not None else None

        non_tensor = list_of_dict_to_dict_of_list([d.non_tensor_batch for d in data])
        for key, val in non_tensor.items():
            try:
                non_tensor[key] = np.concatenate(val, axis=0)
            except Exception:
                non_tensor[key] = np.array([item for sublist in val for item in sublist])

        # Merge meta_info, aggregating metrics specially
        merged_meta, all_metrics = {}, []
        for d in data:
            for k, v in d.meta_info.items():
                if k == "metrics" and v is not None:
                    all_metrics.extend(v if isinstance(v, list) else [v])
                elif k not in merged_meta:
                    merged_meta[k] = v
                else:
                    assert merged_meta[k] == v, f"Conflicting meta_info for '{k}'"
        if all_metrics:
            merged_meta["metrics"] = list_of_dict_to_dict_of_list(all_metrics)

        return type(data[0])(batch=new_batch, non_tensor_batch=non_tensor, meta_info=merged_meta)

    def reorder(self, indices: torch.Tensor):
        """Reorder elements in-place according to indices.

        Args:
            indices: Tensor of indices specifying new order.
        """
        self.batch = self.batch[indices]
        self.non_tensor_batch = {k: v[indices.detach().numpy()] for k, v in self.non_tensor_batch.items()}

    def repeat(self, repeat_times: int = 2, interleave: bool = True) -> "DataProto":
        """Repeat the batch data a specified number of times.

        Args:
            repeat_times: Number of times to repeat.
            interleave: If True, interleave (AABBCC). If False, tile (ABCABC).

        Returns:
            New DataProto with repeated data.
        """
        new_size = len(self) * repeat_times
        if self.batch is not None:
            if interleave:
                tensors = {k: v.repeat_interleave(repeat_times, dim=0) for k, v in self.batch.items()}
            else:
                tensors = {
                    k: v.unsqueeze(0).expand(repeat_times, *v.shape).reshape(-1, *v.shape[1:])
                    for k, v in self.batch.items()
                }
            repeated_batch = TensorDict(tensors, batch_size=(new_size,))
        else:
            repeated_batch = None

        if interleave:
            non_tensor = {k: np.repeat(v, repeat_times, axis=0) for k, v in self.non_tensor_batch.items()}
        else:
            non_tensor = {
                k: np.tile(v, (repeat_times,) + (1,) * (v.ndim - 1)) for k, v in self.non_tensor_batch.items()
            }

        return type(self)(batch=repeated_batch, non_tensor_batch=non_tensor, meta_info=self.meta_info)

    def unfold_column_chunks(self, n_split: int, split_keys: list[str] | None = None) -> "DataProto":
        """Split second dimension and unfold to first dimension.

        Useful for passing grouped tensors that shouldn't be shuffled in a dataset.
        Keys not in split_keys are repeated to match the shape.

        Args:
            n_split: Number of splits along the second dimension.
            split_keys: Keys to split (others are repeated).

        Returns:
            New DataProto with unfolded dimensions.
        """
        split_keys = split_keys or []
        new_size = len(self) * n_split

        if self.batch is not None:
            unfolded = {}
            for key, val in self.batch.items():
                if key in split_keys:
                    unfolded[key] = val.reshape(new_size, val.shape[1] // n_split, *val.shape[2:])
                else:
                    unfolded[key] = torch.repeat_interleave(val, n_split, dim=0)
            unfolded_batch = TensorDict(unfolded, batch_size=(new_size,), device=self.batch.device)
        else:
            unfolded_batch = None

        non_tensor = {}
        for key, val in self.non_tensor_batch.items():
            if key in split_keys:
                non_tensor[key] = val.reshape(new_size, val.shape[1] // n_split, *val.shape[2:])
            else:
                non_tensor[key] = np.repeat(val, n_split, axis=0)

        return type(self)(batch=unfolded_batch, non_tensor_batch=non_tensor, meta_info=self.meta_info)

    def sample_level_repeat(self, repeat_times: list | tuple | torch.Tensor | np.ndarray) -> "DataProto":
        """Repeat each row a variable number of times (interleaved).

        Args:
            repeat_times: Per-element repeat counts (1D array-like).

        Returns:
            New DataProto with repeated data.
        """
        if isinstance(repeat_times, tuple | np.ndarray):
            repeat_times = list(repeat_times) if isinstance(repeat_times, tuple) else repeat_times.tolist()
        elif isinstance(repeat_times, torch.Tensor):
            repeat_times = repeat_times.tolist()
        assert isinstance(repeat_times, list), f"repeat_times must be list-like, got {type(repeat_times)}"
        counts = torch.tensor(repeat_times)

        if self.batch is not None:
            tensors = {k: v.repeat_interleave(counts, dim=0) for k, v in self.batch.items()}
            repeated_batch = TensorDict(tensors, batch_size=(counts.sum().item(),), device=self.batch.device)
        else:
            repeated_batch = None

        non_tensor = {k: np.repeat(v, repeat_times, axis=0) for k, v in self.non_tensor_batch.items()}
        return type(self)(batch=repeated_batch, non_tensor_batch=non_tensor, meta_info=self.meta_info)

    def repeat_by_counts(self, repeat_counts: list[int], interleave: bool = True) -> "DataProto":
        """Repeat each element a variable number of times.

        Args:
            repeat_counts: Per-element repeat counts (must match batch size).
            interleave: If True, interleave results. If False, gather by index.

        Returns:
            New DataProto with repeated data.
        """
        assert all(isinstance(x, int) for x in repeat_counts), "repeat_counts must be list of ints"
        total = sum(repeat_counts)

        if self.batch is not None:
            assert len(repeat_counts) == self.batch.batch_size[0], "repeat_counts length must match batch size"
            counts_tensor = torch.tensor(repeat_counts, device=self.batch.device)
            if interleave:
                tensors = {k: v.repeat_interleave(counts_tensor, dim=0) for k, v in self.batch.items()}
            else:
                indices = torch.cat(
                    [
                        torch.full((c,), i, dtype=torch.long, device=self.batch.device)
                        for i, c in enumerate(repeat_counts)
                    ]
                )
                tensors = {k: v[indices] for k, v in self.batch.items()}
            repeated_batch = TensorDict(tensors, batch_size=(total,))
        else:
            repeated_batch = None

        if interleave:
            non_tensor = {k: np.repeat(v, repeat_counts, axis=0) for k, v in self.non_tensor_batch.items()}
        else:
            idx = np.concatenate([[i] * c for i, c in enumerate(repeat_counts)])
            non_tensor = {k: v[idx] for k, v in self.non_tensor_batch.items()}

        return type(self)(batch=repeated_batch, non_tensor_batch=non_tensor, meta_info=self.meta_info)

    def to_tensordict(self) -> TensorDict:
        """Convert this DataProto to a TensorDict (requires tensordict >= 0.10).

        Returns:
            TensorDict containing all batch, non_tensor_batch, and meta_info data.
        """
        assert parse_version(tensordict.__version__) >= parse_version("0.10"), "Requires tensordict >= 0.10"
        from tensordict.tensorclass import NonTensorData, NonTensorStack

        from axon.utils import tensordict_utils as tu

        tensor_batch = self.batch.to_dict()
        assert not (set(tensor_batch) & set(self.non_tensor_batch)), "batch and non_tensor_batch have overlapping keys"

        for key, val in self.non_tensor_batch.items():
            tensor_batch[key] = NonTensorStack.from_list([NonTensorData(item) for item in val])
        return tu.get_tensordict(tensor_dict=tensor_batch, non_tensor_dict=self.meta_info)

    def get_data_info(self) -> str:
        """Get formatted information about all stored data.

        Returns:
            Multi-line string describing batch tensors, non_tensor_batch arrays, and meta_info types.
        """
        lines = ["batch"]
        for key, tensor in self.batch.items():
            if hasattr(tensor, "device"):
                lines.append(f"  {key}: {tuple(tensor.shape)} ({tensor.dtype}) {tensor.device}")
            elif hasattr(tensor, "shape"):
                lines.append(f"  {key}: {tuple(tensor.shape)} ({tensor.dtype})")
            else:
                lines.append(f"  {key}: {type(tensor).__name__}")

        lines.append("non_tensor_batch")
        lines.extend(f"  {k}: ndarray{v.shape} ({v.dtype})" for k, v in self.non_tensor_batch.items())

        lines.append("meta_info")
        lines.extend(f"  {k}: {self._get_type_info(v)}" for k, v in self.meta_info.items())
        return "\n".join(lines)

    def _get_type_info(self, value: Any) -> str:
        """Get type description for a value, recursively handling containers.

        Args:
            value: Any Python value.

        Returns:
            String description of the type structure.
        """
        if isinstance(value, list):
            elem_types = {self._get_type_info(v) for v in value[:3]}
            return f"list[{'|'.join(elem_types) or '...'}]"
        if isinstance(value, tuple):
            return f"tuple({', '.join(self._get_type_info(v) for v in value)})"
        if isinstance(value, dict):
            if not value:
                return "dict"
            k, v = next(iter(value.items()))
            return f"dict[{self._get_type_info(k)}: {self._get_type_info(v)}]"
        if isinstance(value, np.ndarray):
            return f"ndarray{value.shape} ({value.dtype})"
        return type(value).__name__


@dataclass
class DataProtoFuture:
    """Lazy wrapper for async DataProto operations without blocking the driver.

    Enables asynchronous execution by deferring data fetching until explicitly requested.
    Contains Ray ObjectRefs that will be collected and optionally dispatched.

    Attributes:
        collect_fn: Function to reduce list of futures to a single DataProto.
        futures: List of Ray ObjectRefs to DataProto or TensorDict objects.
        dispatch_fn: Optional function to partition/select from the collected result.

    Note:
        Only supports direct passing between methods. Cannot perform operations
        on DataProtoFuture in the driver without calling get().
    """

    collect_fn: Callable
    futures: list[ray.ObjectRef]
    dispatch_fn: Callable = None

    @staticmethod
    def concat(data: list[ray.ObjectRef]) -> "DataProtoFuture":
        """Create a DataProtoFuture that will concatenate the resolved futures.

        Args:
            data: List of Ray ObjectRefs to DataProto objects.

        Returns:
            DataProtoFuture configured to concatenate results.
        """
        return DataProtoFuture(collect_fn=DataProto.concat, futures=data)

    def chunk(self, chunks: int) -> list["DataProtoFuture"]:
        """Create futures for each chunk of the eventual result.

        Args:
            chunks: Number of chunks to create.

        Returns:
            List of DataProtoFuture, one per chunk.
        """

        def make_dispatch(i, n):
            return lambda x: x.chunk(chunks=n)[i]

        return [
            DataProtoFuture(collect_fn=self.collect_fn, dispatch_fn=make_dispatch(i, chunks), futures=self.futures)
            for i in range(chunks)
        ]

    def get(self) -> DataProto | TensorDict:
        """Resolve all futures and return the collected/dispatched result.

        Returns:
            Collected DataProto or TensorDict, optionally dispatched.
        """
        output = ray.get(self.futures)
        if isinstance(output[0], DataProto):
            output = DataProto.concat(output)
        elif isinstance(output[0], TensorDict):
            from axon.utils.tensordict_utils import concat_tensordict

            output = concat_tensordict(output)
        else:
            raise TypeError(f"Unknown type {type(output[0])} in DataProtoFuture")
        return self.dispatch_fn(output) if self.dispatch_fn else output


def all_gather_data_proto(data: DataProto, process_group) -> None:
    """All-gather a DataProto across a process group (in-place).

    Args:
        data: DataProto to gather (modified in place).
        process_group: torch.distributed process group.
    """
    group_size = torch.distributed.get_world_size(group=process_group)
    prev_device = data.batch.device
    data.to(get_device_id())
    data.batch = allgather_dict_tensors(data.batch.contiguous(), size=group_size, group=process_group, dim=0)
    data.to(prev_device)

    all_non_tensor = [None] * group_size
    torch.distributed.all_gather_object(all_non_tensor, data.non_tensor_batch, group=process_group)
    data.non_tensor_batch = {k: np.concatenate([d[k] for d in all_non_tensor]) for k in data.non_tensor_batch}
