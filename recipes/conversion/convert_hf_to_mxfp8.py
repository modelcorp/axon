#!/usr/bin/env python3
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
# Ported from miles convert_hf_to_mxfp8.py (github.com/radixark/miles), Apache-2.0.

"""Convert a HuggingFace model checkpoint to MxFP8 format.

MxFP8 uses group-wise FP8 (e4m3) quantization with UE8M0 (uint8) scale
factors.  Groups of 32 elements along the last dimension share one scale.

Usage:
    python scripts/convert_hf_to_mxfp8.py \
        --model-dir /path/to/hf-model \
        --save-dir /path/to/output-mxfp8

The output checkpoint is compatible with SGLang's MxFP8 loader.

Ported from miles/tools/convert_hf_to_mxfp8.py.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import shutil

import torch
from safetensors.torch import load_file, save_file

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_MXFP8_GROUP_SIZE = 32

# Layer filtering: matches miles' SKIP_WEIGHT_SUBSTRINGS exactly
_SKIP_WEIGHT_SUBSTRINGS = (
    "layernorm",
    "embed",
    "router",
    "mlp.gate.",
    "norm",
    "lm_head",
    "eh_proj",
    "weights_proj",
)


def _should_quantize(name: str, weight: torch.Tensor) -> bool:
    """Determine whether to quantize a weight tensor to MxFP8.

    Matches miles' should_quantize logic exactly.
    """
    if not name.endswith(".weight"):
        return False
    if any(substr in name for substr in _SKIP_WEIGHT_SUBSTRINGS):
        return False
    if weight.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return False
    if weight.dim() < 2:
        return False
    if weight.shape[-1] % _MXFP8_GROUP_SIZE != 0:
        return False
    return True


def parse_args():
    parser = argparse.ArgumentParser(description="Convert HF model to MxFP8 format")
    parser.add_argument("--model-dir", type=str, required=True, help="Path to HF model directory")
    parser.add_argument("--save-dir", type=str, required=True, help="Path to save MxFP8 model")
    parser.add_argument("--device", type=str, default="cuda", help="Device for quantization (default: cuda)")
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = os.path.abspath(args.model_dir)
    output_path = os.path.abspath(args.save_dir)
    os.makedirs(output_path, exist_ok=True)

    from axon.utils.sglang.mxfp8 import quantize_weight_mxfp8

    # Copy non-safetensors files first (config, tokenizer, etc.)
    for filename in os.listdir(input_path):
        if not filename.endswith(".safetensors") and not os.path.isdir(os.path.join(input_path, filename)):
            shutil.copyfile(os.path.join(input_path, filename), os.path.join(output_path, filename))

    st_files = sorted(f for f in os.listdir(input_path) if f.endswith(".safetensors"))
    if not st_files:
        raise FileNotFoundError(f"No .safetensors files found in {input_path}")

    logger.info(f"Found {len(st_files)} safetensors files in {input_path}")

    # Track results for index and config generation (matches miles' ConversionResult)
    weight_map: dict[str, str] = {}
    total_size: int = 0
    modules_to_not_convert: list[str] = []

    for st_file in st_files:
        logger.info(f"Processing {st_file}...")
        weights = load_file(os.path.join(input_path, st_file), device=args.device)
        output_weights: dict[str, torch.Tensor] = {}

        file_modules_skipped: list[str] = []
        for name, param in weights.items():
            if _should_quantize(name, param):
                qweight, scale = quantize_weight_mxfp8(param)
                output_weights[name] = qweight
                scale_name = name.replace(".weight", ".weight_scale_inv")
                output_weights[scale_name] = scale
            else:
                if name.endswith(".weight"):
                    file_modules_skipped.append(name.replace(".weight", ""))
                output_weights[name] = param

        save_file(output_weights, os.path.join(output_path, st_file), metadata={"format": "pt"})

        # Accumulate index and tracking info
        for key, tensor in output_weights.items():
            weight_map[key] = st_file
            total_size += tensor.numel() * tensor.element_size()
        modules_to_not_convert.extend(file_modules_skipped)

        logger.info(f"  Saved {st_file}")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Write quantization config to config.json (matches miles format exactly)
    quantization_config = {
        "activation_scheme": "dynamic",
        "fmt": "e4m3",
        "quant_method": "mxfp8",
        "weight_block_size": [1, _MXFP8_GROUP_SIZE],
        "scale_fmt": "ue8m0",
    }
    if modules_to_not_convert:
        quantization_config["modules_to_not_convert"] = sorted(set(modules_to_not_convert))

    config_path = os.path.join(input_path, "config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            cfg = json.load(f)
        cfg["quantization_config"] = quantization_config
        with open(os.path.join(output_path, "config.json"), "w") as f:
            json.dump(cfg, f, indent=2)

    # Regenerate model.safetensors.index.json
    index_dict = {
        "metadata": {"total_size": total_size},
        "weight_map": weight_map,
    }
    with open(os.path.join(output_path, "model.safetensors.index.json"), "w") as f:
        json.dump(index_dict, f, indent=2)

    logger.info(f"Done! {len(weight_map)} weight entries, {len(modules_to_not_convert)} modules not converted")
    logger.info(f"Output saved to {output_path}")


if __name__ == "__main__":
    main()
