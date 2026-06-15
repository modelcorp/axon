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
#
# Adapted from FlashRL (github.com/LLM360/Flash-RL), Apache-2.0; FP8 weight-refit references NVIDIA-NeMo/RL.
"""
Memory-optimized FP8 patcher for vLLM - v6
Added: FP8 verification utilities to ensure model is actually running in FP8
"""

import gc
import logging
import os
import time
import types
from dataclasses import asdict, dataclass, field
from functools import partial
from unittest.mock import patch

import torch
import vllm

try:
    from vllm.model_executor.layers.fused_moe.layer import FusedMoE
    from vllm.model_executor.layers.linear import LinearBase
except ImportError:
    FusedMoE = None
    LinearBase = None

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)


# =============================================================================
# CONFIGURATION
# =============================================================================

GC_FREQUENCY = 500
DEBUG_TIMING = os.environ.get("FLASHRL_DEBUG_TIMING", "0") == "1"

_FLASHRL_CONFIG_DATA = None


# =============================================================================
# FP8 VERIFICATION UTILITIES
# =============================================================================


def get_fp8_dtypes():
    """Get FP8 dtype constants."""
    fp8_dtypes = []
    if hasattr(torch, "float8_e4m3fn"):
        fp8_dtypes.append(torch.float8_e4m3fn)
    if hasattr(torch, "float8_e5m2"):
        fp8_dtypes.append(torch.float8_e5m2)
    if hasattr(torch, "float8_e4m3fnuz"):
        fp8_dtypes.append(torch.float8_e4m3fnuz)
    if hasattr(torch, "float8_e5m2fnuz"):
        fp8_dtypes.append(torch.float8_e5m2fnuz)
    return tuple(fp8_dtypes)


def is_fp8_dtype(dtype):
    """Check if dtype is an FP8 type."""
    fp8_dtypes = get_fp8_dtypes()
    return dtype in fp8_dtypes


def verify_fp8_model(model, verbose=True):
    """
    Verify that a model is properly configured for FP8 inference.

    Returns:
        dict: Verification results with counts and details
    """
    results = {
        "is_fp8": False,
        "total_linear_layers": 0,
        "fp8_linear_layers": 0,
        "bf16_linear_layers": 0,
        "total_moe_layers": 0,
        "fp8_moe_layers": 0,
        "total_params": 0,
        "fp8_params": 0,
        "fp8_param_names": [],
        "non_fp8_param_names": [],
        "quant_methods": {},
        "issues": [],
    }

    # Check for FP8 quantization methods on layers
    try:
        from vllm.model_executor.layers.quantization.fp8 import Fp8LinearMethod, Fp8MoEMethod

        has_fp8_classes = True
    except ImportError:
        has_fp8_classes = False
        results["issues"].append("Could not import Fp8LinearMethod/Fp8MoEMethod")

    # Scan all modules
    for name, module in model.named_modules():
        quant_method = getattr(module, "quant_method", None)

        if quant_method is not None:
            method_type = type(quant_method).__name__
            results["quant_methods"][name] = method_type

            if has_fp8_classes:
                if isinstance(quant_method, Fp8LinearMethod):
                    results["total_linear_layers"] += 1
                    results["fp8_linear_layers"] += 1
                elif isinstance(quant_method, Fp8MoEMethod):
                    results["total_moe_layers"] += 1
                    results["fp8_moe_layers"] += 1
                elif "Unquantized" in method_type:
                    results["total_linear_layers"] += 1
                    results["bf16_linear_layers"] += 1
                    results["issues"].append(f"Layer {name} using {method_type} (not FP8)")

    # Scan all parameters for FP8 weights
    for name, param in model.named_parameters():
        results["total_params"] += 1

        if is_fp8_dtype(param.dtype):
            results["fp8_params"] += 1
            results["fp8_param_names"].append(name)
        else:
            # Only flag weight params, not scales/biases
            if "weight" in name and "scale" not in name:
                results["non_fp8_param_names"].append(f"{name} ({param.dtype})")

    # Determine overall FP8 status
    if results["fp8_linear_layers"] > 0 or results["fp8_moe_layers"] > 0:
        results["is_fp8"] = True

    # Check for FP8 weight tensors (after process_weights_after_loading)
    fp8_weight_count = 0
    for name, param in model.named_parameters():
        if "weight" in name and is_fp8_dtype(param.dtype):
            fp8_weight_count += 1

    if fp8_weight_count > 0:
        results["is_fp8"] = True
        results["fp8_weight_count"] = fp8_weight_count

    if verbose:
        print("\n" + "=" * 60)
        print("FP8 VERIFICATION REPORT")
        print("=" * 60)
        print(f"Model is FP8: {results['is_fp8']}")
        print("\nQuantization Methods:")
        print(f"  - FP8 Linear layers: {results['fp8_linear_layers']}")
        print(f"  - BF16 Linear layers: {results['bf16_linear_layers']}")
        print(f"  - FP8 MoE layers: {results['fp8_moe_layers']}")
        print("\nParameters:")
        print(f"  - Total: {results['total_params']}")
        print(f"  - FP8 dtype: {results['fp8_params']}")
        print(f"  - FP8 weights: {results.get('fp8_weight_count', 0)}")

        if results["non_fp8_param_names"]:
            print("\nNon-FP8 weight parameters (first 10):")
            for name in results["non_fp8_param_names"][:10]:
                print(f"  - {name}")
            if len(results["non_fp8_param_names"]) > 10:
                print(f"  ... and {len(results['non_fp8_param_names']) - 10} more")

        if results["issues"]:
            print("\nIssues found:")
            for issue in results["issues"]:
                print(f"  ⚠ {issue}")

        print("=" * 60 + "\n")

    return results


def verify_fp8_forward(model, sample_input_ids=None, verbose=True):
    """
    Verify FP8 is used during forward pass by checking intermediate activations.

    Args:
        model: The vLLM model
        sample_input_ids: Optional input IDs for test forward pass
        verbose: Print detailed report

    Returns:
        dict: Forward pass verification results
    """
    results = {
        "forward_uses_fp8": False,
        "fp8_gemm_detected": False,
        "modules_checked": 0,
        "fp8_modules": [],
    }

    # Check if model has FP8 quantization methods
    try:
        from vllm.model_executor.layers.quantization.fp8 import Fp8LinearMethod, Fp8MoEMethod

        for name, module in model.named_modules():
            quant_method = getattr(module, "quant_method", None)
            results["modules_checked"] += 1

            if isinstance(quant_method, Fp8LinearMethod | Fp8MoEMethod):
                results["fp8_modules"].append(name)
                results["forward_uses_fp8"] = True

                # Check if weights are actually FP8
                weight = getattr(module, "weight", None)
                if weight is not None and is_fp8_dtype(weight.dtype):
                    results["fp8_gemm_detected"] = True

    except ImportError:
        results["error"] = "Could not import FP8 classes"

    if verbose:
        print("\n" + "=" * 60)
        print("FP8 FORWARD PASS VERIFICATION")
        print("=" * 60)
        print(f"Forward uses FP8: {results['forward_uses_fp8']}")
        print(f"FP8 GEMM detected: {results['fp8_gemm_detected']}")
        print(f"Modules checked: {results['modules_checked']}")
        print(f"FP8 modules found: {len(results['fp8_modules'])}")

        if results["fp8_modules"]:
            print("\nFP8 modules (first 10):")
            for name in results["fp8_modules"][:10]:
                print(f"  - {name}")
            if len(results["fp8_modules"]) > 10:
                print(f"  ... and {len(results['fp8_modules']) - 10} more")

        print("=" * 60 + "\n")

    return results


def add_fp8_forward_hooks(model, log_every_n=100):
    """
    Add hooks to monitor FP8 operations during forward pass.
    Useful for debugging to see which layers are actually using FP8.

    Args:
        model: The model to monitor
        log_every_n: Log every N forward calls

    Returns:
        list: Hook handles (call .remove() to cleanup)
    """
    hooks = []
    call_count = [0]

    def create_hook(name):
        def hook(module, input, output):
            call_count[0] += 1
            if call_count[0] % log_every_n == 0:
                quant_method = getattr(module, "quant_method", None)
                weight = getattr(module, "weight", None)

                info = f"[{call_count[0]}] {name}"
                if quant_method:
                    info += f" | quant={type(quant_method).__name__}"
                if weight is not None:
                    info += f" | weight.dtype={weight.dtype}"
                if isinstance(output, torch.Tensor):
                    info += f" | output.dtype={output.dtype}"

                logger.info(info)

        return hook

    try:
        from vllm.model_executor.layers.quantization.fp8 import Fp8LinearMethod, Fp8MoEMethod

        for name, module in model.named_modules():
            quant_method = getattr(module, "quant_method", None)
            if isinstance(quant_method, Fp8LinearMethod | Fp8MoEMethod):
                handle = module.register_forward_hook(create_hook(name))
                hooks.append(handle)

    except ImportError:
        logger.warning("Could not add FP8 forward hooks - FP8 classes not found")

    logger.info(f"Added {len(hooks)} FP8 forward hooks")
    return hooks


def assert_fp8_model(model, min_fp8_layers=1):
    """
    Assert that model is properly configured for FP8.
    Raises AssertionError if FP8 is not properly configured.

    Args:
        model: The model to check
        min_fp8_layers: Minimum number of FP8 layers required
    """
    results = verify_fp8_model(model, verbose=False)

    total_fp8_layers = results["fp8_linear_layers"] + results["fp8_moe_layers"]

    assert results["is_fp8"], (
        f"Model is not configured for FP8!\n"
        f"FP8 Linear: {results['fp8_linear_layers']}, "
        f"FP8 MoE: {results['fp8_moe_layers']}, "
        f"BF16 Linear: {results['bf16_linear_layers']}\n"
        f"Issues: {results['issues']}"
    )

    assert total_fp8_layers >= min_fp8_layers, (
        f"Expected at least {min_fp8_layers} FP8 layers, but found {total_fp8_layers}"
    )

    logger.info(f"✓ FP8 assertion passed: {total_fp8_layers} FP8 layers found")


# =============================================================================
# UTILITIES
# =============================================================================


def safe_gc():
    try:
        gc.collect()
    except Exception:
        pass


def timed_log(msg, start_time=None):
    if start_time is not None:
        logger.info(f"{msg} (took {time.time() - start_time:.2f}s)")
    else:
        logger.info(msg)


class LazyParamMetadata:
    __slots__ = ["shape", "stride", "dtype", "nbytes"]

    def __init__(self, param):
        self.shape = param.shape
        self.stride = param.stride()
        self.dtype = param.dtype
        self.nbytes = param.untyped_storage().nbytes()


def check_updated(name, updated_params, quant_fn_name):
    if name in updated_params:
        return True
    if (
        quant_fn_name in ["fp8", "fp8_vllm", "fp8_vllm_fast", "fp8_fast"]
        and name.endswith("weight_scale")
        and name[:-6] in updated_params
    ):
        return True
    return False


def bond_method_to_cls(func, obj):
    if hasattr(func, "__self__") or not callable(func):
        return func
    return types.MethodType(func, obj)


RECORDED_LOADER_KEYS = [
    "weight_loader",
    "load_qkv_weight",
    "load_row_parallel_weight",
    "load_merged_column_weight",
    "load_column_parallel_weight",
    "output_dim",
    "input_dim",
    "_assert_and_load",
    "tp_rank",
    "tp_size",
]


def _get_config_data(quantization: str = "fp8"):
    global _FLASHRL_CONFIG_DATA
    if _FLASHRL_CONFIG_DATA is not None:
        return _FLASHRL_CONFIG_DATA
    from axon.monkey_patches.vllm.fp8.configs import get_default_config

    cfg = get_default_config(quantization)
    _FLASHRL_CONFIG_DATA = asdict(cfg)
    return _FLASHRL_CONFIG_DATA


# =============================================================================
# QUANTIZATION WRAPPER
# =============================================================================


def create_passthrough_quantize_fn():
    """Pass through weights unchanged - let vLLM handle FP8 quantization."""

    def passthrough(weights_iter, profile):
        yield from weights_iter

    return passthrough


# =============================================================================
# WEIGHT PROCESSING FUNCTIONS
# =============================================================================


def _swap_to_hacked_data_optimized(model, hacked_data_dict, updated_params, quant_fn_name):
    start_time = time.time()
    skipped_params = []
    param_list = list(model.named_parameters())
    total = len(param_list)

    logger.info(f"Swapping {total} parameters...")

    for i, (name, p) in enumerate(param_list):
        if check_updated(name, updated_params, quant_fn_name):
            strided_data = torch.as_strided(p.data, hacked_data_dict[name].shape, hacked_data_dict[name].stride())
            hacked_data_dict[name].copy_(strided_data)
        else:
            skipped_params.append(name)

        tmp_data = p.data
        p.data = hacked_data_dict[name]
        del tmp_data

        if (i + 1) % GC_FREQUENCY == 0:
            if DEBUG_TIMING:
                logger.info(f"Swap progress: {i + 1}/{total}")
            safe_gc()

    timed_log(f"Swap complete. Skipped: {len(skipped_params)}", start_time)


def hacked_process_weights_after_loading_optimized(
    original_process_weights_after_loading,
    model,
    model_config,
    target_device,
    hacked_data_dict=None,
    updated_params=None,
) -> None:
    start_time = time.time()
    logger.info(">>> Entering process_weights_after_loading")

    if model_config is None and target_device is None:
        model_config = getattr(model, "hacked_model_config", None)
        target_device = getattr(model, "hacked_target_device", None)
    else:
        model.hacked_model_config = model_config
        model.hacked_target_device = target_device

    if getattr(model, "hacked_not_need_process_weights_after_loading", False):
        logger.info("<<< Already processed, skipping")
        return

    if not hasattr(model, "hacked_original_weights_rebuild_keys"):
        logger.info("Capturing parameter metadata...")
        model.hacked_original_weights_rebuild_keys = {}
        for name, p in model.named_parameters():
            model.hacked_original_weights_rebuild_keys[name] = LazyParamMetadata(p)
        logger.info(f"Captured {len(model.hacked_original_weights_rebuild_keys)} params")

    logger.info("Recording weight loaders...")
    recorded_loader = {k: {} for k in RECORDED_LOADER_KEYS}
    for name, p in model.named_parameters():
        for k in RECORDED_LOADER_KEYS:
            if hasattr(p, k):
                attr = getattr(p, k)
                if not callable(attr):
                    recorded_loader[k][name] = attr
                elif p is attr.__self__:
                    recorded_loader[k][name] = attr.__func__
                else:
                    recorded_loader[k][name] = attr

    quant_fn_name = getattr(model, "flashrl_quant_fn", "int8")

    if "fast" in quant_fn_name and hacked_data_dict is not None:
        logger.info(f"Using fast path for {quant_fn_name}")
        _process_fast_path_optimized(model, target_device, hacked_data_dict, updated_params, quant_fn_name)
    else:
        logger.info("Calling original process_weights_after_loading...")
        logger.info(">>> This is where vLLM converts weights to FP8 <<<")
        proc_start = time.time()
        original_process_weights_after_loading(model, model_config, target_device)
        timed_log("Original complete (FP8 quantization done)", proc_start)

        if hacked_data_dict is not None:
            _swap_to_hacked_data_optimized(model, hacked_data_dict, updated_params, quant_fn_name)

    # VERIFY FP8 after processing
    logger.info("Verifying FP8 quantization after process_weights_after_loading...")
    _quick_fp8_check(model)

    model.hacked_recorded_loader = recorded_loader
    timed_log("<<< process_weights_after_loading complete", start_time)


def _quick_fp8_check(model):
    """Quick check to verify FP8 is applied."""
    fp8_linear = 0
    fp8_moe = 0
    fp8_weights = 0

    try:
        from vllm.model_executor.layers.quantization.fp8 import Fp8LinearMethod, Fp8MoEMethod

        for name, module in model.named_modules():
            qm = getattr(module, "quant_method", None)
            if isinstance(qm, Fp8LinearMethod):
                fp8_linear += 1
            elif isinstance(qm, Fp8MoEMethod):
                fp8_moe += 1

        for name, param in model.named_parameters():
            if is_fp8_dtype(param.dtype):
                fp8_weights += 1

        logger.info(f"FP8 Quick Check: {fp8_linear} Fp8Linear, {fp8_moe} Fp8MoE, {fp8_weights} FP8 weight tensors")

        if fp8_linear == 0 and fp8_moe == 0:
            logger.warning("⚠ WARNING: No FP8 layers detected! Model may be running in BF16")
        else:
            logger.info("✓ FP8 layers detected")

    except ImportError:
        logger.warning("Could not verify FP8 (import error)")


def _process_fast_path_optimized(model, target_device, hacked_data_dict, updated_params, quant_fn_name):
    start_time = time.time()
    logger.info("Fast path processing...")

    from vllm.model_executor.layers.linear import QKVCrossParallelLinear
    from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
    from vllm.model_executor.layers.quantization.compressed_tensors.schemes import CompressedTensorsW8A8Int8
    from vllm.model_executor.layers.quantization.fp8 import Fp8LinearMethod

    try:
        from vllm.model_executor.model_loader.loader import device_loading_context
    except ImportError:
        from vllm.model_executor.model_loader.utils import device_loading_context

    for name, module in model.named_modules():
        if isinstance(module, QKVCrossParallelLinear):
            module.process_weights_after_loading()
            continue
        quant_method = getattr(module, "quant_method", None)
        if isinstance(quant_method, QuantizeMethodBase):
            if isinstance(quant_method, Fp8LinearMethod | CompressedTensorsW8A8Int8):
                continue
            with device_loading_context(module, target_device):
                quant_method.process_weights_after_loading(module)

    skipped_params = []
    param_list = list(model.named_parameters())

    if "fp8" in quant_fn_name:
        for i, (name, p) in enumerate(param_list):
            if "weight_scale" not in name:
                if name in updated_params:
                    if p.dtype != hacked_data_dict[name].dtype:
                        weight_output = hacked_data_dict[name].t()
                        scale_output = hacked_data_dict[name + "_scale"]
                        torch.ops._C.dynamic_scaled_fp8_quant(
                            weight_output,
                            p.to(target_device),
                            scale_output,
                        )
                        all_params = dict(model.named_parameters())
                        pscale = all_params[name + "_scale"]
                        tmp = pscale.data
                        pscale.data = hacked_data_dict[name + "_scale"]
                        del tmp
                    else:
                        strided = torch.as_strided(
                            p.data, hacked_data_dict[name].shape, hacked_data_dict[name].stride()
                        )
                        hacked_data_dict[name].copy_(strided)
                    tmp = p.data
                    p.data = hacked_data_dict[name]
                    del tmp
                else:
                    skipped_params.append(name)
                    tmp = p.data
                    p.data = hacked_data_dict[name]
                    del tmp
            if (i + 1) % GC_FREQUENCY == 0:
                safe_gc()
    else:
        for i, (name, p) in enumerate(param_list):
            if name in updated_params:
                strided = torch.as_strided(p.data, hacked_data_dict[name].shape, hacked_data_dict[name].stride())
                hacked_data_dict[name].copy_(strided)
            else:
                skipped_params.append(name)
            tmp = p.data
            p.data = hacked_data_dict[name]
            del tmp
            if (i + 1) % GC_FREQUENCY == 0:
                safe_gc()

    timed_log(f"Fast path done. Skipped {len(skipped_params)}", start_time)


# =============================================================================
# OPTIMIZED LOAD_WEIGHTS
# =============================================================================


def _create_optimized_load_weights(model, original_load_weights, flash_quantize_fn, config_data):
    """Create the optimized load_weights function."""

    module_attrs = config_data.get("module_attribute_to_preserve", []) or []
    profile = getattr(model, "flash_rl_profile", None)
    quant_fn_name = config_data.get("fn", "fp8")

    # For FP8 with vLLM, use passthrough - vLLM handles FP8 in process_weights_after_loading
    if quant_fn_name in ["fp8", "fp8_vllm"]:
        logger.info("Using passthrough (vLLM handles FP8 in process_weights_after_loading)")
        effective_quantize_fn = create_passthrough_quantize_fn()
    else:
        effective_quantize_fn = flash_quantize_fn

    def optimized_load_weights(weights):
        overall_start = time.time()
        logger.info("=" * 60)
        logger.info(">>> ENTERING optimized_load_weights")
        logger.info(f"    Quantization mode: {quant_fn_name}")
        logger.info("=" * 60)

        model.hacked_not_need_process_weights_after_loading = False

        # First load - no rebuild keys yet
        if not hasattr(model, "hacked_original_weights_rebuild_keys"):
            logger.info("First load path")

            weight_count = [0]

            def counting_weights(weights_iter):
                for name, weight in weights_iter:
                    weight_count[0] += 1
                    if weight_count[0] % 100 == 0:
                        logger.info(f"Loading weight {weight_count[0]}: {name} ({weight.dtype})")
                    yield name, weight

            logger.info("Calling original_load_weights (weights as BF16)...")
            load_start = time.time()
            result = original_load_weights(effective_quantize_fn(counting_weights(weights), profile))
            timed_log(f"original_load_weights complete ({weight_count[0]} weights)", load_start)

            logger.info("NOTE: FP8 conversion happens in process_weights_after_loading")
            timed_log("<<< optimized_load_weights (first) complete", overall_start)
            return result

        logger.info("Subsequent load path")

        # Preserve attrs
        if module_attrs:
            for _, m in model.named_modules():
                for attr in module_attrs:
                    if torch.is_tensor(getattr(m, attr, None)):
                        setattr(m, f"hacked_{attr}", getattr(m, attr))

        # Build data dict
        logger.info("Building hacked_data_dict...")
        start = time.time()
        hacked_data_dict = {name: p.data for name, p in model.named_parameters()}
        param_names = list(hacked_data_dict.keys())
        timed_log(f"Built dict for {len(param_names)} params", start)

        # Reallocate
        logger.info("Reallocating parameter storages...")
        start = time.time()
        metadata = model.hacked_original_weights_rebuild_keys
        for name in param_names:
            if name in metadata:
                meta = metadata[name]
                p = dict(model.named_parameters())[name]
                p.data = torch.empty(meta.shape, dtype=meta.dtype, device=p.device)
        timed_log("Reallocation complete", start)
        safe_gc()

        # Reattach loaders
        logger.info("Reattaching loaders...")
        start = time.time()
        existing = dict(model.named_parameters())
        for k, loader_k in getattr(model, "hacked_recorded_loader", {}).items():
            for n, loader in loader_k.items():
                if n in existing and not hasattr(existing[n], k):
                    setattr(existing[n], k, bond_method_to_cls(loader, existing[n]))
        del existing
        timed_log("Loaders reattached", start)

        # Load
        logger.info("Loading weights...")
        start = time.time()
        updated_params = original_load_weights(effective_quantize_fn(weights, profile))
        timed_log("original_load_weights complete", start)

        del weights

        # Post-process (this is where FP8 conversion happens)
        logger.info("Post-processing (FP8 conversion)...")
        start = time.time()
        if hasattr(model, "hacked_model_config") and hasattr(model, "hacked_target_device"):
            try:
                from vllm.model_executor.model_loader import loader

                loader._process_weights_after_loading(
                    model, None, None, hacked_data_dict=hacked_data_dict, updated_params=updated_params
                )
            except ImportError:
                from vllm.model_executor.model_loader import utils

                utils.process_weights_after_loading(
                    model, None, None, hacked_data_dict=hacked_data_dict, updated_params=updated_params
                )
            model.hacked_not_need_process_weights_after_loading = True
        else:
            model.hacked_not_need_process_weights_after_loading = False
            quant_fn = getattr(model, "flashrl_quant_fn", "int8")
            _swap_to_hacked_data_optimized(model, hacked_data_dict, updated_params, quant_fn)
        timed_log("Post-processing complete", start)

        del hacked_data_dict
        safe_gc()
        torch.cuda.empty_cache()

        # Restore attrs
        if module_attrs:
            for _, m in model.named_modules():
                for attr in module_attrs:
                    if hasattr(m, f"hacked_{attr}"):
                        setattr(m, attr, getattr(m, f"hacked_{attr}"))
                        delattr(m, f"hacked_{attr}")

        timed_log("<<< optimized_load_weights (subsequent) complete", overall_start)
        return updated_params

    return optimized_load_weights


def _setup_model_for_flashrl(model, config_data):
    """Setup a model with FlashRL."""

    if hasattr(model, "beforeflashrl_load_weights"):
        logger.debug("Model already setup for FlashRL")
        return

    from axon.monkey_patches.vllm.fp8.flash_quantization import get_quantize_fn

    quant_fn_name = config_data.get("fn", "fp8")
    logger.info(f"Setting up model for FlashRL with quant_fn={quant_fn_name}")

    model.flashrl_quant_fn = quant_fn_name
    model.flash_rl_module_attribute_to_preserve = config_data.get("module_attribute_to_preserve", []) or []

    if quant_fn_name not in ["fp8", "fp8_vllm", "fp8_fast", "fp8_vllm_fast"]:
        profile_path = config_data.get("profile")
        if profile_path:
            model.flash_rl_profile = _load_profile(profile_path)
        else:
            model.flash_rl_profile = None
    else:
        model.flash_rl_profile = None

    flash_quantize_fn = get_quantize_fn(quant_fn_name)

    model.beforeflashrl_load_weights = model.load_weights
    model.load_weights = _create_optimized_load_weights(
        model, model.beforeflashrl_load_weights, flash_quantize_fn, config_data
    )

    logger.info("Model load_weights patched")


def _load_profile(profile_path):
    try:
        if not os.path.exists(profile_path):
            from huggingface_hub import hf_hub_download

            parts = profile_path.split("/")
            assert len(parts) >= 3
            profile_path = hf_hub_download(  # nosec B615
                repo_id="/".join(parts[:2]),
                filename="/".join(parts[2:]),
                revision=os.environ.get("HF_HUB_REVISION"),
            )
        # Loads trusted internal profiling artifacts.
        return torch.load(profile_path, map_location="cpu")  # nosec B614
    except Exception as e:
        logger.warning(f"Failed to load profile: {e}")
        return None


# =============================================================================
# PATCHES
# =============================================================================


def patch_vllm_initialize_model(quantization):
    config_data = _get_config_data(quantization)

    try:
        from vllm.model_executor.model_loader import utils

        if hasattr(utils, "beforeflashrl_initialize_model"):
            return True

        original = utils.initialize_model
        utils.beforeflashrl_initialize_model = original

        def hacked_initialize_model(*args, **kwargs):
            logger.info(">>> Entering patched initialize_model")
            start = time.time()
            model = original(*args, **kwargs)
            timed_log("Model initialized", start)
            _setup_model_for_flashrl(model, config_data)
            logger.info("<<< Exiting patched initialize_model")
            return model

        utils.initialize_model = hacked_initialize_model
        logger.info("Patched utils.initialize_model")
        return True

    except ImportError as e:
        logger.warning(f"Could not patch initialize_model: {e}")
        return False


def patch_vllm_process_weights_after_loading():
    try:
        succeeded = False

        try:
            from vllm.model_executor.model_loader import loader

            if not hasattr(loader, "beforeflashrl_process_weights_after_loading"):
                original = loader._process_weights_after_loading
                loader.beforeflashrl_process_weights_after_loading = original
                loader._process_weights_after_loading = partial(
                    hacked_process_weights_after_loading_optimized, original
                )
                logger.info("Patched loader._process_weights_after_loading")
                succeeded = True
        except (ImportError, AttributeError):
            pass

        if not succeeded:
            try:
                from vllm.model_executor.model_loader import utils

                if not hasattr(utils, "beforeflashrl_process_weights_after_loading"):
                    original = utils.process_weights_after_loading
                    utils.beforeflashrl_process_weights_after_loading = original
                    utils.process_weights_after_loading = partial(
                        hacked_process_weights_after_loading_optimized, original
                    )
                    logger.info("Patched utils.process_weights_after_loading")
                    succeeded = True
            except (ImportError, AttributeError) as e:
                logger.warning(f"Could not patch: {e}")

        try:
            from vllm.model_executor.layers.quantization.kv_cache import BaseKVCacheMethod

            if not hasattr(BaseKVCacheMethod, "beforeflashrl_process_weights_after_loading"):
                original = BaseKVCacheMethod.process_weights_after_loading
                BaseKVCacheMethod.beforeflashrl_process_weights_after_loading = original

                def hacked_kvcache(self, layer):
                    for attr in ["k_scale", "v_scale", "q_scale", "prob_scale"]:
                        if not hasattr(layer, attr):
                            setattr(layer, attr, -1.0)
                    return original(self, layer)

                BaseKVCacheMethod.process_weights_after_loading = hacked_kvcache
        except Exception:
            pass

        return succeeded
    except Exception as e:
        logger.error(f"Error: {e}")
        return False


def patch_vllm_fp8_create_weight():
    try:
        from vllm.model_executor.layers.quantization.fp8 import Fp8LinearMethod

        if not hasattr(Fp8LinearMethod, "beforeflashrl_create_weights"):
            original = Fp8LinearMethod.create_weights
            Fp8LinearMethod.beforeflashrl_create_weights = original
            from axon.monkey_patches.vllm.fp8.fp8loader import disable_mem_pool

            def hacked(self, *args, **kwargs):
                with disable_mem_pool(disable=True):
                    return original(self, *args, **kwargs)

            Fp8LinearMethod.create_weights = hacked
            logger.info("Patched Fp8LinearMethod.create_weights")
        return True
    except Exception as e:
        logger.error(f"Error: {e}")
        return False


def patch_vllm_fp8_moe_create_weight():
    try:
        from vllm.model_executor.layers.quantization.fp8 import Fp8MoEMethod

        if not hasattr(Fp8MoEMethod, "beforeflashrl_create_weights"):
            original = Fp8MoEMethod.create_weights
            Fp8MoEMethod.beforeflashrl_create_weights = original
            from axon.monkey_patches.vllm.fp8.fp8loader import disable_mem_pool

            def hacked(self, *args, **kwargs):
                with disable_mem_pool(disable=True):
                    return original(self, *args, **kwargs)

            Fp8MoEMethod.create_weights = hacked
            logger.info("Patched Fp8MoEMethod.create_weights")
        return True
    except Exception as e:
        logger.error(f"Error: {e}")
        return False


def patch_vllm_engine_args(quantization):
    config_data = _get_config_data(quantization)
    quant_fn_name = config_data.get("fn", "fp8")

    if quant_fn_name == "bf16":
        return True

    patched = False

    try:
        from vllm.engine.arg_utils import EngineArgs

        if not hasattr(EngineArgs, "beforeflashrl_init"):
            EngineArgs.beforeflashrl_init = EngineArgs.__init__

            def hacked_init(self, *args, **kwargs):
                EngineArgs.beforeflashrl_init(self, *args, **kwargs)
                if quant_fn_name.startswith("fp8"):
                    if getattr(self, "quantization", None) in (None, "auto"):
                        logger.info("Setting EngineArgs.quantization='fp8'")
                        self.quantization = "fp8"

            EngineArgs.__init__ = hacked_init
            logger.info("Patched EngineArgs.__init__")
            patched = True
    except ImportError:
        pass

    try:
        from vllm.engine.arg_utils import AsyncEngineArgs

        if not hasattr(AsyncEngineArgs, "beforeflashrl_init"):
            AsyncEngineArgs.beforeflashrl_init = AsyncEngineArgs.__init__

            def hacked_async_init(self, *args, **kwargs):
                AsyncEngineArgs.beforeflashrl_init(self, *args, **kwargs)
                if quant_fn_name.startswith("fp8"):
                    if getattr(self, "quantization", None) in (None, "auto"):
                        logger.info("Setting AsyncEngineArgs.quantization='fp8'")
                        self.quantization = "fp8"

            AsyncEngineArgs.__init__ = hacked_async_init
            patched = True

        if not hasattr(AsyncEngineArgs, "beforeflashrl_create_engine_config"):
            AsyncEngineArgs.beforeflashrl_create_engine_config = AsyncEngineArgs.create_engine_config

            def hacked_config(self, *args, **kwargs):
                vllm_config = AsyncEngineArgs.beforeflashrl_create_engine_config(self, *args, **kwargs)
                if quant_fn_name.startswith("fp8"):
                    try:
                        if getattr(vllm_config.model_config, "quantization", None) != "fp8":
                            logger.info("Setting model_config.quantization='fp8'")
                            vllm_config.model_config.quantization = "fp8"
                    except Exception:
                        pass
                return vllm_config

            AsyncEngineArgs.create_engine_config = hacked_config
    except ImportError:
        pass

    return patched


def patch_vllm_kernel_warmup():
    try:
        from vllm.model_executor.warmup import kernel_warmup as warmup_module

        if not hasattr(warmup_module, "beforeflashrl_flashinfer_autotune"):
            original = warmup_module.flashinfer_autotune
            warmup_module.beforeflashrl_flashinfer_autotune = original

            def patched(runner):
                orig_max = runner.scheduler_config.max_num_batched_tokens
                reduced = max(4096, orig_max // 4)
                logger.info(f"Reducing warmup batch {orig_max} -> {reduced}")
                runner.scheduler_config.max_num_batched_tokens = reduced
                try:
                    safe_gc()
                    torch.cuda.empty_cache()
                    return original(runner)
                finally:
                    runner.scheduler_config.max_num_batched_tokens = orig_max

            warmup_module.flashinfer_autotune = patched
        return True
    except Exception:
        return False


def patch_vllm_fp8_moe_apply_for_routermap():
    try:
        from vllm.model_executor.layers.quantization.fp8 import Fp8MoEMethod

        if not hasattr(Fp8MoEMethod, "beforeflashrl_apply"):
            original = Fp8MoEMethod.apply
            Fp8MoEMethod.beforeflashrl_apply = original

            def hacked_apply(
                self,
                layer,
                x,
                router_logits,
                top_k,
                renormalize,
                use_grouped_topk=False,
                topk_group=None,
                num_expert_group=None,
                global_num_experts=-1,
                expert_map=None,
                custom_routing_function=None,
                scoring_func="softmax",
                routed_scaling_factor=1.0,
                e_score_correction_bias=None,
                apply_router_weight_on_input=False,
                activation="silu",
                enable_eplb=False,
                expert_load_view=None,
                logical_to_physical_map=None,
                logical_replica_count=None,
                **kw,
            ):
                if custom_routing_function is None or callable(custom_routing_function):
                    try:
                        from vllm.model_executor.layers.fused_moe.layer import FusedMoE

                        _, topk_ids = FusedMoE.select_experts(
                            hidden_states=x,
                            router_logits=router_logits,
                            use_grouped_topk=use_grouped_topk,
                            top_k=top_k,
                            renormalize=renormalize,
                            topk_group=topk_group,
                            num_expert_group=num_expert_group,
                            custom_routing_function=custom_routing_function,
                            scoring_func=scoring_func,
                            routed_scaling_factor=routed_scaling_factor,
                            e_score_correction_bias=e_score_correction_bias,
                            indices_type=getattr(self, "topk_indices_dtype", torch.int32),
                            enable_eplb=enable_eplb,
                            expert_map=expert_map,
                            expert_load_view=expert_load_view,
                            logical_to_physical_map=logical_to_physical_map,
                            logical_replica_count=logical_replica_count,
                        )
                        n = topk_ids.shape[0]
                        if hasattr(layer, "last_topk_ids") and layer.last_topk_ids.shape[0] >= n:
                            layer.last_topk_ids[:n].copy_(topk_ids)
                        if hasattr(layer, "num_tokens"):
                            layer.num_tokens.fill_(n)
                    except Exception:
                        pass

                return original(
                    self,
                    layer=layer,
                    x=x,
                    router_logits=router_logits,
                    top_k=top_k,
                    renormalize=renormalize,
                    use_grouped_topk=use_grouped_topk,
                    topk_group=topk_group,
                    num_expert_group=num_expert_group,
                    global_num_experts=global_num_experts,
                    expert_map=expert_map,
                    custom_routing_function=custom_routing_function,
                    scoring_func=scoring_func,
                    routed_scaling_factor=routed_scaling_factor,
                    e_score_correction_bias=e_score_correction_bias,
                    apply_router_weight_on_input=apply_router_weight_on_input,
                    activation=activation,
                    enable_eplb=enable_eplb,
                    expert_load_view=expert_load_view,
                    logical_to_physical_map=logical_to_physical_map,
                    logical_replica_count=logical_replica_count,
                    **kw,
                )

            Fp8MoEMethod.apply = hacked_apply
        return True
    except Exception:
        return False


# =============================================================================
# Axon FP8 WEIGHT REFIT PATCHES
# (Ref: https://github.com/NVIDIA-NeMo/RL/commit/bc24887c72a6e1b2699a228bc87c588546dfe6b7)
# Patches Fp8LinearMethod/Fp8MoEMethod.process_weights_after_loading to preserve
# weight_loader attributes needed for weight refit during RL training.
# =============================================================================

FP8_BLOCK_QUANT_KWARGS = {
    "activation_scheme": "dynamic",
    "fmt": "e4m3",
    "quant_method": "fp8",
    "weight_block_size": [128, 128],
}


@dataclass()
class FP8State:
    # A cache of fp8 parameter names, we can check this cache to see if a
    # param name corresponds to a fp8 weight
    seen_params: set = field(default_factory=lambda: set())
    fp8_param_names: set = field(default_factory=lambda: set())
    vllm_patches: list = field(default_factory=lambda: [])


fp8_state: FP8State = FP8State()


def is_fp8_model(vllm_config):
    from vllm.model_executor.layers.quantization.fp8 import Fp8Config

    if hasattr(vllm_config, "quant_config") and isinstance(vllm_config.quant_config, Fp8Config):
        return True

    return False


def get_module_from_param_name(model, name: str):
    # Split the name into parts (e.g., 'layers', '0', 'self_attn', 'q_proj', 'weight')
    # The module path is all but the last part (the parameter's own name)
    path_parts = name.split(".")
    module_path = path_parts[:-1]
    # Replace with the fused model name
    packed_modules_mapping = model.packed_modules_mapping
    reversed_mapping = {
        original_name: fused_name
        for fused_name, original_names_list in packed_modules_mapping.items()
        for original_name in original_names_list
    }
    if module_path[-1] in reversed_mapping.keys():
        module_path[-1] = reversed_mapping[module_path[-1]]

    current_module = model
    try:
        # Traverse the model hierarchy
        for part in module_path:
            if isinstance(current_module, FusedMoE):
                return current_module
            elif isinstance(current_module, torch.nn.ModuleList):
                current_module = current_module[int(part)]
            else:
                current_module = getattr(current_module, part)
    except (AttributeError, IndexError, ValueError) as e:
        print(f"Warning: Could not find module for parameter '{name}'. Error: {e}")
    return current_module


def is_fp8_weight(name, model):
    if name not in fp8_state.seen_params:
        fp8_state.seen_params.add(name)
        # Filter out bias params
        if name.endswith("weight"):
            module = get_module_from_param_name(model, name)
            # We currently only quantize linear layers

            if (isinstance(module, LinearBase) and module.weight.dtype == torch.float8_e4m3fn) or (
                isinstance(module, FusedMoE)
                and module.w13_weight.dtype == torch.float8_e4m3fn
                and module.w2_weight.dtype == torch.float8_e4m3fn
            ):
                fp8_state.fp8_param_names.add(name)
    return name in fp8_state.fp8_param_names


def scaled_fp8_blockwise(
    data_hp,
    weight_block_size,
):
    # cast tensor from high precision to FP8 with 128*128 blockwise quantization.
    assert len(data_hp.shape) == 2, "Only 2d input tensor is supported"

    block_size1 = weight_block_size[1]
    block_size0 = weight_block_size[0]
    assert data_hp.shape[1] % block_size1 == 0, (
        f"data_hp.shape[1] {data_hp.shape[1]}  must be a multiple of block_size1: {block_size1}."
    )
    assert data_hp.shape[0] % block_size0 == 0, (
        f"data_hp.shape[0] {data_hp.shape[0]} must be a multiple of block_size0: {block_size0}."
    )

    # FP8
    max_dtype = torch.finfo(torch.float8_e4m3fn).max

    original_shape = data_hp.shape
    blk_m, blk_n = data_hp.shape[0] // block_size0, data_hp.shape[1] // block_size1

    assert block_size1 == block_size0
    data_hp = data_hp.reshape(blk_m, block_size0, blk_n, block_size1)

    # Permute to (BLK_M, BLK_N, BLOCK_SIZE_M, BLOCK_SIZE_N)
    data_hp = data_hp.permute(0, 2, 1, 3)
    # Flatten to (BLK_M, BLK_N, BLOCK_SIZE_M * BLOCK_SIZE_N)
    data_hp = data_hp.to(torch.float32).contiguous().flatten(start_dim=2)

    # Calculate max absolute value per block
    max_abs = torch.amax(torch.abs(data_hp), dim=-1, keepdim=True)

    # Use FP32 scale
    scale_fp = max_dtype / max_abs
    scale_fp = torch.where(max_abs == 0, 1.0, scale_fp)
    # preserve the behavior for 0 amax case
    scale_fp = torch.where(max_abs == torch.inf, 1.0, scale_fp)

    descale_fp = torch.reciprocal(scale_fp)

    # Scale and saturate cast the data elements to max of target dtype
    data_lp = torch.clamp(data_hp * scale_fp, min=-1 * max_dtype, max=max_dtype)

    fp_data = data_lp.to(torch.float8_e4m3fn)

    # (BLK_M, BLK_N, BLOCK_SIZE_M * BLOCK_SIZE_N) to (M, N)
    fp_data = fp_data.reshape(blk_m, blk_n, block_size0, block_size1).permute(0, 2, 1, 3).reshape(original_shape)

    # Convert to target format, but still in original precision container
    return fp_data, descale_fp


def quant_weights(weights, model, quant_config, dtype=torch.bfloat16):
    weights_quantized = []
    for k, v in weights:
        if not is_fp8_weight(k, model):
            weights_quantized.append((k, v))
            continue
        # Cast the weight into fp8 and its scale factor
        if quant_config.weight_block_size is not None:
            logger.info("Using blockwise quantization")
            param_lp, param_scale = scaled_fp8_blockwise(
                v.to(dtype),
                weight_block_size=quant_config.weight_block_size,
            )
            param_scale = param_scale.squeeze(-1)
            weights_quantized.append([k, param_lp])
            if vllm.__version__ >= "0.11.0":
                if "expert" in k:
                    weights_quantized.append([k + "_scale_inv", param_scale])
                else:
                    weights_quantized.append([k + "_scale", param_scale])
            else:
                weights_quantized.append([k + "_scale_inv", param_scale])

        else:
            raise ValueError(
                "Currently only support blockwise quantization, please set weight_block_size in quant_config"
            )

    return weights_quantized


def load_quanted_weights(weights, model_runner):
    model = model_runner.model
    quant_config = model_runner.vllm_config.quant_config
    vllm_dtype = model_runner.vllm_config.model_config.dtype

    weights_quantized = quant_weights(weights, model, quant_config, dtype=vllm_dtype)

    # Monkey patch the param class to their subclass, as certain models
    # will check the param type to call the proper weightloader
    for name, param in model.named_parameters():
        if hasattr(param, "subclass_type"):
            param.orig_type = param.__class__
            param.__class__ = param.subclass_type
    # Finally load the weights into vllm
    loaded_params = model.load_weights(weights_quantized)
    # Undo the type change above to the original type
    for name, param in model.named_parameters():
        if hasattr(param, "subclass_type"):
            param.__class__ = param.orig_type
    return loaded_params


def _save_weight_loaders(layer):
    """Save weight_loader attributes from all parameters before processing."""
    saved = {}
    for name, param in layer.named_parameters(recurse=False):
        wl = getattr(param, "weight_loader", None)
        if wl is not None:
            saved[name] = wl
    return saved


def _restore_weight_loaders(layer, saved):
    """Restore weight_loader attributes after processing."""
    for name, wl in saved.items():
        param = getattr(layer, name, None)
        if param is not None and isinstance(param, torch.nn.Parameter):
            param.weight_loader = wl


def process_weights_after_loading_for_vllm16(self, layer) -> None:
    """v0.16 wrapper: call native process_weights_after_loading but preserve
    weight_loader attributes needed for axon weight refit."""
    from vllm.model_executor.layers.quantization.fp8 import Fp8LinearMethod

    saved = _save_weight_loaders(layer)
    Fp8LinearMethod._original_process_weights_after_loading(self, layer)
    _restore_weight_loaders(layer, saved)


def process_weights_after_loading_moe_for_vllm16(self, layer) -> None:
    """v0.16 wrapper: call native process_weights_after_loading but preserve
    weight_loader attributes needed for axon weight refit."""
    from vllm.model_executor.layers.quantization.fp8 import Fp8MoEMethod

    saved = _save_weight_loaders(layer)
    Fp8MoEMethod._original_process_weights_after_loading(self, layer)
    _restore_weight_loaders(layer, saved)


def apply_vllm_fp8_patches():
    """Wrap vllm's FP8 process_weights_after_loading to preserve weight_loader
    attributes needed for axon online weight refit."""
    logger.info("Applying vllm fp8 patches for blockwise quantization")

    from vllm.model_executor.layers.quantization.fp8 import Fp8LinearMethod, Fp8MoEMethod

    Fp8LinearMethod._original_process_weights_after_loading = Fp8LinearMethod.process_weights_after_loading
    Fp8MoEMethod._original_process_weights_after_loading = Fp8MoEMethod.process_weights_after_loading

    patcher1 = patch(
        "vllm.model_executor.layers.quantization.fp8.Fp8LinearMethod.process_weights_after_loading",
        process_weights_after_loading_for_vllm16,
    )
    patcher1.start()
    patcher2 = patch(
        "vllm.model_executor.layers.quantization.fp8.Fp8MoEMethod.process_weights_after_loading",
        process_weights_after_loading_moe_for_vllm16,
    )
    patcher2.start()


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================


def patch_vllm_fp8_optimized(quantization: str = "fp8"):
    """Memory-optimized FP8 patching for vLLM - v6 with verification.

    Args:
        quantization: Quantization mode string, e.g. "fp8", "fp8_fast".
    """
    import importlib.util

    if importlib.util.find_spec("vllm") is None:
        logger.info("vLLM not installed")
        return

    logger.info("=" * 60)
    logger.info("FlashRL vLLM FP8 Patching v6 (with verification)")
    logger.info("=" * 60)

    config_data = _get_config_data(quantization)
    quant_fn = config_data.get("fn", "fp8")
    logger.info(f"Quantization: {quant_fn}")

    if quant_fn == "bf16":
        logger.info("BF16 mode - minimal patching")
        return

    s = patch_vllm_engine_args(quantization)
    logger.info(f"[1/8] EngineArgs: {s}")

    s = patch_vllm_initialize_model(quantization)
    logger.info(f"[2/8] initialize_model: {s}")

    s = patch_vllm_process_weights_after_loading()
    logger.info(f"[3/8] process_weights_after_loading: {s}")

    s = patch_vllm_fp8_create_weight()
    logger.info(f"[4/8] Fp8LinearMethod.create_weights: {s}")

    s = patch_vllm_fp8_moe_create_weight()
    logger.info(f"[5/8] Fp8MoEMethod.create_weights: {s}")

    s = patch_vllm_fp8_moe_apply_for_routermap()
    logger.info(f"[6/8] Fp8MoEMethod.apply: {s}")

    s = patch_vllm_kernel_warmup()
    logger.info(f"[7/8] kernel_warmup: {s}")

    apply_vllm_fp8_patches()
    logger.info("[8/8] Fp8LinearMethod/Fp8MoEMethod.process_weights_after_loading (weight refit)")

    logger.info("=" * 60)
    logger.info("FlashRL patching complete")
    logger.info("=" * 60)


patch_vllm_fp8 = patch_vllm_fp8_optimized
