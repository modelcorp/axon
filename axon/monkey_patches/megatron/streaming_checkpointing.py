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

"""Patch FileSystemWriterAsync to clone CPU view tensors one-at-a-time.

With ``fused=True`` CPU Adam, optimizer states (exp_avg, exp_avg_sq, master_param)
are views into large flat buffers.  ``torch.save`` serialises the *entire* backing
storage of a tensor, so views must be cloned to standalone storage before writing.

The original ``prepare_write_data`` clones every CPU view upfront (~29 GB/rank for
large MoE models with full CPU optimizer offloading), causing OOM.

This patch:
  1. Replaces ``prepare_write_data`` so it resolves tensors **without** cloning.
  2. Replaces ``write_preloaded_data`` so it clones each view just before
     ``torch.save`` and frees it immediately after, using ``mmap`` for allocation
     instead of ``malloc`` to avoid fork deadlocks.

After ``fork()``, glibc malloc arena mutexes inherited from parent threads can be
permanently locked, making any ``malloc`` (and therefore ``t.clone()``) deadlock.
We bypass this by allocating tensor data via anonymous ``mmap`` (a direct syscall
to the kernel that does not touch userspace malloc state) and copying data with
``memcpy`` via ctypes.

Usage::

    from axon.monkey_patches.megatron.streaming_checkpointing import apply_streaming_checkpointing_patch
    apply_streaming_checkpointing_patch()
"""

import ctypes
import inspect
import mmap as _mmap_module
import os

import torch
from megatron.core.dist_checkpointing.strategies.async_utils import _disable_gc
from torch.distributed.checkpoint.filesystem import _write_item
from torch.distributed.checkpoint.planner import WriteItemType

# libc memcpy for raw copies that bypass Python/PyTorch allocators.
_libc = ctypes.CDLL("libc.so.6", use_errno=True)
_libc.memcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
_libc.memcpy.restype = ctypes.c_void_p


def _fork_safe_clone(t):
    """Clone a contiguous CPU view tensor using mmap (safe after fork).

    Allocates storage via anonymous mmap (kernel syscall, no malloc) and copies
    data with libc memcpy.  Returns ``(cloned_tensor, mmap_buf)`` — caller must
    keep ``mmap_buf`` alive and call ``mmap_buf.close()`` when done.
    """
    assert t.is_contiguous(), f"_fork_safe_clone requires contiguous tensor, got strides {t.stride()}"
    nbytes = t.numel() * t.element_size()
    if nbytes == 0:
        return t, None

    # Anonymous mmap — goes directly to kernel, bypasses glibc malloc entirely.
    buf = _mmap_module.mmap(-1, nbytes)
    try:
        # Get the address of the mmap buffer via a ctypes array, then memcpy.
        dst_arr = (ctypes.c_char * nbytes).from_buffer(buf)
        _libc.memcpy(ctypes.addressof(dst_arr), ctypes.c_void_p(t.data_ptr()), nbytes)

        # Wrap mmap as a torch tensor (zero-copy).
        result = torch.frombuffer(buf, dtype=t.dtype, count=t.numel()).reshape(t.shape)
    except Exception:
        buf.close()
        raise
    return result, buf


def apply_streaming_checkpointing_patch():
    """Apply the patch.  Safe to call multiple times (idempotent)."""
    import megatron.core.dist_checkpointing.strategies.filesystem_async as mod
    from megatron.core.dist_checkpointing.strategies.filesystem_async import (
        DEFAULT_SUFFIX,
        _get_write_results_queue,
        _split_by_separation_hint,
        _split_by_size_and_type,
    )

    # Guard against double-patching.
    if getattr(mod.FileSystemWriterAsync, "_streaming_ckpt_patched", False):
        return

    # Save originals for revert_streaming_checkpointing_patch().
    mod.FileSystemWriterAsync._orig_prepare_write_data = mod.FileSystemWriterAsync.prepare_write_data
    mod.FileSystemWriterAsync._orig_write_preloaded_data = mod.FileSystemWriterAsync.write_preloaded_data

    # ------------------------------------------------------------------
    # 1. prepare_write_data — resolve tensors WITHOUT cloning views.
    # ------------------------------------------------------------------
    def _prepare_write_data_no_clone(self, plan, planner):
        storage_plan = plan.storage_data
        if self.separation_hint:
            assert self.thread_count > 1
        bins = self.thread_count // 2 if self.separation_hint is not None else self.thread_count
        item_buckets = _split_by_size_and_type(bins, plan.items)

        file_count = 0

        def gen_file(prefix=""):
            nonlocal file_count
            name = f"{prefix}{storage_plan.prefix}{file_count}{DEFAULT_SUFFIX}"
            file_count += 1
            return name

        self.write_buckets = []
        for group_name, group_buckets in _split_by_separation_hint(item_buckets, self.separation_hint).items():
            for bucket in group_buckets:
                bytes_data = [
                    (item, planner.resolve_data(item)) for item in bucket if item.type == WriteItemType.BYTE_IO
                ]
                tensor_data = [
                    (item, planner.resolve_data(item).detach()) for item in bucket if item.type != WriteItemType.BYTE_IO
                ]
                if bytes_data or tensor_data:
                    fname = gen_file(prefix=group_name)
                    self.write_buckets.append(
                        (
                            os.path.join(self.checkpoint_dir, fname),
                            fname,
                            (bytes_data, tensor_data),
                        )
                    )

        if self.write_buckets:
            assert len(self.write_buckets) <= self.thread_count
            self.results_queue = _get_write_results_queue()
        else:
            self.results_queue = None

    # ------------------------------------------------------------------
    # 2. write_preloaded_data — mmap-based clone in forked child.
    # ------------------------------------------------------------------
    @staticmethod
    @_disable_gc()
    def _streaming_write_preloaded_data(
        transform_list,
        local_proc_idx,
        write_bucket,
        results_queue,
        count_queue,
        use_fsync,
        **kwargs,
    ):
        local_results = []
        try:
            file_name, storage_key, (bytes_data, tensor_data) = write_bucket
            extra_kwargs = {}
            if "serialization_format" in inspect.signature(_write_item).parameters:
                from torch.distributed.checkpoint.filesystem import SerializationFormat

                extra_kwargs["serialization_format"] = SerializationFormat.TORCH_SAVE

            use_msc = kwargs.get("use_msc", False)
            open_file = open
            if use_msc:
                import multistorageclient as msc

                open_file = msc.open

            mmap_bufs = []
            with open_file(file_name, "wb") as stream:
                for item, data in bytes_data:
                    local_results.append(_write_item(*transform_list, stream, data, item, storage_key, **extra_kwargs))
                for item, tensor in tensor_data:
                    mmap_buf = None
                    t = tensor
                    if t.device.type == "cpu":
                        is_view = t.untyped_storage().size() != t.numel() * t.element_size()
                        if is_view:
                            t, mmap_buf = _fork_safe_clone(t)
                            mmap_bufs.append(mmap_buf)
                    assert t.is_cpu
                    local_results.append(_write_item(*transform_list, stream, t, item, storage_key, **extra_kwargs))
                    del t

                if use_fsync:
                    if use_msc:
                        stream.fsync()
                    else:
                        os.fsync(stream.fileno())

            # Close mmap buffers only after the file is fully written and fsynced.
            # Closing earlier risks use-after-unmap if _write_item or torch.save
            # kept an internal reference to the tensor (whose storage points to mmap).
            for mb in mmap_bufs:
                mb.close()
            del mmap_bufs

            local_output = (local_proc_idx, local_results)
        except Exception as e:
            local_output = (local_proc_idx, e)

        results_queue.put(local_output)
        count_queue.get()
        count_queue.task_done()

    # ------------------------------------------------------------------
    # Apply.
    # ------------------------------------------------------------------
    mod.FileSystemWriterAsync.prepare_write_data = _prepare_write_data_no_clone
    mod.FileSystemWriterAsync.write_preloaded_data = _streaming_write_preloaded_data
    mod.FileSystemWriterAsync._streaming_ckpt_patched = True


def revert_streaming_checkpointing_patch():
    """Revert to the original MCore behavior (upfront clone in prepare_write_data)."""
    import megatron.core.dist_checkpointing.strategies.filesystem_async as mod

    if not getattr(mod.FileSystemWriterAsync, "_streaming_ckpt_patched", False):
        return

    mod.FileSystemWriterAsync.prepare_write_data = mod.FileSystemWriterAsync._orig_prepare_write_data
    mod.FileSystemWriterAsync.write_preloaded_data = mod.FileSystemWriterAsync._orig_write_preloaded_data
    mod.FileSystemWriterAsync._streaming_ckpt_patched = False
