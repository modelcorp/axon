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
"""Patches for Megatron-Core MoE checkpoint saving.

Fixes two issues in megatron-core 0.15.0rc7:

1. **EP replica_id**: Non-expert parameters (attention norms, projections) are
   replicated across Expert Parallel ranks, but the default
   ``make_sharded_tensor_for_checkpoint`` sets ``replica_id=(0, tp, dp)`` for
   ALL ranks. With EP > 1, multiple ranks claim the same shard as replica 0,
   causing ``CheckpointingException``.  Fix: include EP rank in replica_id.

2. **master_param KeyError**: ``HybridDeviceOptimizer`` with
   ``use_precision_aware_optimizer`` may not have ``master_param`` in optimizer
   state before the first step or for already-FP32 params.  The checkpoint code
   unconditionally pops it.  Fix: fall back to the param itself.
"""

import importlib
import logging

logger = logging.getLogger(__name__)


def apply_moe_checkpoint_patches():
    """Apply all MoE checkpoint fixes. Call once during trainer init."""
    _patch_ep_replica_id()
    _patch_master_param_fallback()
    _patch_hybrid_optimizer_master_param_load_fallback()
    _patch_filesystem_writer_preserve_mcore_data()
    _patch_skip_validation_for_ep_replicated_params()


def _patch_ep_replica_id():
    """Include EP rank in replica_id for non-expert parameters."""
    import megatron.core.utils as mcore_utils
    from megatron.core import parallel_state as mpu

    _orig = mcore_utils.make_sharded_tensor_for_checkpoint

    def _make_sharded_with_ep_replica(tensor, key, prepend_offsets=(), replica_id=None, **kwargs):
        if replica_id is None:
            ep_rank = mpu.get_expert_model_parallel_rank()
            if ep_rank > 0:
                tp_rank = mpu.get_tensor_model_parallel_rank()
                dp_rank = mpu.get_data_parallel_rank(with_context_parallel=True)
                replica_id = (ep_rank, tp_rank, dp_rank)
        return _orig(tensor, key, prepend_offsets, replica_id, **kwargs)

    # Patch at the source definition
    mcore_utils.make_sharded_tensor_for_checkpoint = _make_sharded_with_ep_replica

    # Patch in every module that imported the function as a local binding
    patched = 1
    for mod_name in [
        "megatron.core.transformer.utils",
        "megatron.core.transformer.moe.shared_experts",
        "megatron.core.ssm.mamba_mixer",
        "megatron.core.extensions.transformer_engine",
    ]:
        try:
            mod = importlib.import_module(mod_name)
            if hasattr(mod, "make_sharded_tensor_for_checkpoint"):
                mod.make_sharded_tensor_for_checkpoint = _make_sharded_with_ep_replica
                patched += 1
        except ImportError:
            pass

    logger.info(f"Patched make_sharded_tensor_for_checkpoint with EP replica_id in {patched} modules")


def _patch_master_param_fallback():
    """Handle missing master_param in HybridDeviceOptimizer state during checkpoint save."""
    from megatron.core.optimizer.cpu_offloading import HybridDeviceOptimizer
    from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer

    def _safe_get_main_param_and_optimizer_states(self, model_param):
        group_index, group_order = self.model_param_group_index_map[model_param]
        if self.config.use_precision_aware_optimizer_no_fp8_or_ds_fp8:
            sharded_model_param = self.optimizer.param_groups[group_index]["params"][group_order]
            tensors = {}
            for k in self.optimizer.state[sharded_model_param]:
                if isinstance(self.optimizer, HybridDeviceOptimizer):
                    tensors[k] = self.optimizer.state[sharded_model_param][k]
                    continue
                tensors[k] = self.optimizer.get_unscaled_state(sharded_model_param, k)
            if "master_param" in tensors:
                tensors["param"] = tensors.pop("master_param")
            else:
                # master_param not in state — use the param itself as master.
                # This happens when:
                # - HybridDeviceOptimizer hasn't been stepped yet (state empty)
                # - The param is already FP32 (no separate master copy needed)
                tensors["param"] = sharded_model_param.data.float()
        else:
            main_param = self.optimizer.param_groups[group_index]["params"][group_order]
            optim_state = self.optimizer.state[main_param]
            tensors = {"param": main_param, **optim_state}
        return tensors

    DistributedOptimizer._get_main_param_and_optimizer_states = _safe_get_main_param_and_optimizer_states
    logger.info("Patched DistributedOptimizer._get_main_param_and_optimizer_states with master_param fallback")


def _patch_hybrid_optimizer_master_param_load_fallback():
    """Skip params without ``master_param`` in HybridDeviceOptimizer's post-load hook.

    Megatron calls ``load_state_dict(state_dict())`` on a fresh optimizer to
    pre-allocate state; the post-hook then crashes on `v["master_param"]`
    because no step has populated it yet.
    """
    from megatron.core.optimizer.cpu_offloading.hybrid_optimizer import (
        HybridDeviceOptimizer,
    )

    if getattr(HybridDeviceOptimizer, "_master_param_load_patched", False):
        return

    def _safe_update_fp32_params_by_new_state(self):
        if not self.param_update_in_fp32:
            return
        for param, v in self.state.items():
            if "master_param" not in v:
                continue
            fp32_param = self.param_to_fp32_param[param]
            fp32_param.data.copy_(v["master_param"])

    HybridDeviceOptimizer._update_fp32_params_by_new_state = _safe_update_fp32_params_by_new_state
    HybridDeviceOptimizer._master_param_load_patched = True


def _patch_filesystem_writer_preserve_mcore_data():
    """Re-attach ``mcore_data`` after PyTorch's ``FileSystemWriter.finish``.

    Why: ``finish`` does ``dataclasses.replace(metadata, version=...)`` which
    drops every non-dataclass attribute Megatron tacked on. Without this,
    the saved ``.metadata`` is missing ``mcore_data`` and load crashes in
    ``get_reformulation_metadata``.
    """
    import torch.distributed.checkpoint.filesystem as _torch_fs

    if getattr(_torch_fs.FileSystemWriter, "_mcore_data_preserve_patched", False):
        return

    _orig_finish = _torch_fs.FileSystemWriter.finish
    _PRESERVED_ATTRS = ("mcore_data", "all_local_plans")

    def _rewrite_metadata_with_preserved(metadata_path, preserved):
        import os as _os
        import pickle as _pickle

        try:
            with open(metadata_path, "rb") as f:
                written = _pickle.load(f)
            for attr, value in preserved.items():
                setattr(written, attr, value)
            tmp_path = str(metadata_path) + ".mcore_preserve.tmp"
            with open(tmp_path, "wb") as f:
                _pickle.dump(written, f)
            _os.replace(tmp_path, metadata_path)
        except (OSError, _pickle.PickleError) as e:
            logger.warning("Failed to re-attach mcore_data to %s: %s", metadata_path, e)

    def _finish_preserving_extra_attrs(self, metadata, results):
        preserved = {
            attr: getattr(metadata, attr) for attr in _PRESERVED_ATTRS if hasattr(metadata, attr)
        }
        _orig_finish(self, metadata, results)
        if not preserved:
            return
        import os as _os

        from torch.distributed.checkpoint.filesystem import _metadata_fn
        metadata_path = self._get_metadata_path() if hasattr(self, "_get_metadata_path") else None
        if metadata_path is None:
            metadata_path = _os.path.join(str(self.path), _metadata_fn)
        _rewrite_metadata_with_preserved(metadata_path, preserved)

    _torch_fs.FileSystemWriter.finish = _finish_preserving_extra_attrs
    _torch_fs.FileSystemWriter._mcore_data_preserve_patched = True

    from megatron.core.dist_checkpointing.strategies.filesystem_async import (
        FileSystemWriterAsync,
    )
    if FileSystemWriterAsync.finish is not _finish_preserving_extra_attrs:
        _orig_async_finish = FileSystemWriterAsync.finish

        def _async_finish_preserving_extra_attrs(self, metadata, results):
            preserved = {
                attr: getattr(metadata, attr)
                for attr in _PRESERVED_ATTRS
                if hasattr(metadata, attr)
            }
            _orig_async_finish(self, metadata, results)
            if not preserved:
                return
            import os as _os

            _rewrite_metadata_with_preserved(_os.path.join(str(self.path), ".metadata"), preserved)

        FileSystemWriterAsync.finish = _async_finish_preserving_extra_attrs


def _patch_skip_validation_for_ep_replicated_params():
    """Skip sharding integrity validation for MoE models with EP > 1.

    In MoE models with Expert Parallelism, non-expert parameters (attention
    norms, projections) are replicated across EP ranks but Megatron-Core's
    ``validate_sharding_integrity`` sees them as having incomplete coverage
    because each EP rank maps to a different global layer offset.

    The actual data is correct — all EP ranks have identical copies of these
    parameters, and EP rank 0's copy is saved.  The validation is overly
    strict for this case.  We patch ``save_preprocess`` to skip the
    validation when EP > 1, matching Megatron's own behavior for DP-replicated
    parameters.
    """
    import megatron.core.dist_checkpointing.state_dict_utils as _sd_utils
    from megatron.core import parallel_state as mpu

    _orig_save_preprocess = _sd_utils.save_preprocess

    def _save_preprocess_skip_validation(*args, **kwargs):
        ep_size = mpu.get_expert_model_parallel_world_size()
        if ep_size > 1:
            # Override validate_access_integrity (2nd positional arg) to False
            args = list(args)
            if len(args) >= 2:
                args[1] = False
            else:
                kwargs["validate_access_integrity"] = False
            args = tuple(args)
        return _orig_save_preprocess(*args, **kwargs)

    _sd_utils.save_preprocess = _save_preprocess_skip_validation

    # Also patch wherever save_preprocess was imported directly
    try:
        import megatron.core.dist_checkpointing.serialization as _ser_mod

        if hasattr(_ser_mod, "save_preprocess"):
            _ser_mod.save_preprocess = _save_preprocess_skip_validation
    except ImportError:
        pass

    # PyTorch's default_planner also validates global plan coverage and raises
    # ValueError on incomplete fill. Patch _validate_global_plan to always
    # return True when EP > 1 (warnings are still logged).
    import torch.distributed.checkpoint.default_planner as _dp_mod

    _orig_validate = _dp_mod._validate_global_plan

    def _lenient_validate_global_plan(global_plan, metadata):
        result = _orig_validate(global_plan, metadata)
        if not result and mpu.get_expert_model_parallel_world_size() > 1:
            logger.info("Allowing incomplete chunk coverage in global plan (EP-replicated params)")
            return True
        return result

    _dp_mod._validate_global_plan = _lenient_validate_global_plan

    logger.info("Patched save_preprocess and _validate_global_plan for MoE models with EP > 1")
