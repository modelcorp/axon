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
# Ported from miles convert_hf_to_int4.py (github.com/radixark/miles), Apache-2.0.

"""Convert a HuggingFace model checkpoint to INT4 compressed-tensors format.

Usage:
    python scripts/convert_hf_to_int4.py \
        --model-dir /path/to/hf-model \
        --save-dir /path/to/output-int4 \
        --group-size 32 \
        --symmetric

The output checkpoint is compatible with SGLang and vLLM's compressed-tensors
quantization loader.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

# noqa: E402 – lazy-imported below: axon.utils.sglang.int4

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# Default ignore rules matching miles' convert_hf_to_int4_direct.py.
# These are tuned for MoE models (DeepSeek, Qwen-MoE) where only MoE expert
# weights are quantized to INT4.  For dense models, pass --ignore-rules
# to override with a simpler set like "re:.*lm_head.* re:.*norm.* re:.*embed.*".
_DEFAULT_IGNORE_RULES = [
    "re:.*lm_head.*",
    "re:.*norm.*",
    "re:.*embed.*",
    "re:.*self_attn.*",
    "re:.*shared_experts.*",
    r"re:.*mlp\.(gate|up|gate_up|down)_proj.*",
    r"re:.*mlp\.gate\..*",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Convert HF model to INT4 compressed-tensors format")
    parser.add_argument("--model-dir", type=str, required=True, help="Path to HF model directory")
    parser.add_argument("--save-dir", type=str, required=True, help="Path to save INT4 model")
    parser.add_argument(
        "--group-size", type=int, default=32, help="Quantization group size (default: 32, matching miles)"
    )
    parser.add_argument("--symmetric", action="store_true", default=True, help="Use symmetric quantization")
    parser.add_argument("--no-symmetric", dest="symmetric", action="store_false", help="Use asymmetric quantization")
    parser.add_argument(
        "--ignore-rules",
        type=str,
        nargs="*",
        default=None,
        help="Regex ignore rules (prefix with re:). Default: lm_head, norm, embed",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device for quantization (default: cuda)")
    return parser.parse_args()


def _matches_ignore(name: str, ignore_rules: list[str]) -> bool:
    """Check if a parameter name matches any ignore rule.

    Rules prefixed with ``re:`` are treated as regex patterns.
    Other rules match as simple prefixes.
    """
    import re

    for rule in ignore_rules:
        if rule.startswith("re:"):
            if re.match(rule[3:], name):
                return True
        elif rule == name or name.startswith(rule):
            return True
    return False


def main():
    args = parse_args()
    model_dir = Path(args.model_dir)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    from axon.utils.sglang.int4 import pack_layer

    ignore_rules = args.ignore_rules if args.ignore_rules is not None else _DEFAULT_IGNORE_RULES

    # Find all safetensors files
    st_files = sorted(model_dir.glob("*.safetensors"))
    if not st_files:
        raise FileNotFoundError(f"No .safetensors files found in {model_dir}")

    logger.info(f"Found {len(st_files)} safetensors files in {model_dir}")
    logger.info(f"Config: group_size={args.group_size}, symmetric={args.symmetric}")
    logger.info(f"Ignore rules: {ignore_rules}")

    total_quantized = 0
    total_skipped = 0

    for st_file in st_files:
        logger.info(f"Processing {st_file.name}...")
        weights = load_file(str(st_file), device=args.device)
        output_weights = {}

        for name, param in weights.items():
            is_ignored = _matches_ignore(name, ignore_rules)
            can_quantize = (
                not is_ignored
                and name.endswith(".weight")
                and param.ndim >= 2
                and param.shape[-1] % args.group_size == 0
            )
            if can_quantize:
                w2d = param.reshape(-1, param.shape[-1])
                packed, scales, packed_zp = pack_layer(w2d, args.group_size, args.symmetric)

                base = name.replace(".weight", "")
                output_weights[base + ".weight_packed"] = packed.cpu()
                output_weights[base + ".weight_scale"] = scales.cpu()
                # vLLM expects weight_shape as int64
                output_weights[base + ".weight_shape"] = torch.tensor(list(param.shape), dtype=torch.int64)
                if packed_zp is not None:
                    output_weights[base + ".weight_zero_point"] = packed_zp.cpu()
                total_quantized += 1
                logger.debug(f"  Quantized: {name} {list(param.shape)} -> packed {list(packed.shape)}")
            else:
                output_weights[name] = param.cpu()
                total_skipped += 1

        out_path = save_dir / st_file.name
        save_file(output_weights, str(out_path), metadata={"format": "pt"})
        logger.info(f"  Saved {out_path.name}")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Copy non-safetensors files (config, tokenizer, etc.), but skip stale index
    for f in model_dir.iterdir():
        if f.suffix != ".safetensors" and f.name not in (".git", "model.safetensors.index.json"):
            dst = save_dir / f.name
            if f.is_file() and not dst.exists():
                shutil.copy2(f, dst)

    # Regenerate model.safetensors.index.json with correct weight names
    weight_map = {}
    total_size = 0
    for st_file in sorted(save_dir.glob("*.safetensors")):
        with safe_open(str(st_file), framework="pt") as f:
            for key in f.keys():
                weight_map[key] = st_file.name
                t = f.get_tensor(key)
                total_size += t.numel() * t.element_size()
    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    index_path = save_dir / "model.safetensors.index.json"
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)
    logger.info(f"Regenerated {index_path.name} with {len(weight_map)} entries")

    # Update config.json with quantization metadata (matches miles format exactly)
    config_path = save_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
    else:
        config = {}

    config["quantization_config"] = {
        "config_groups": {
            "group_0": {
                "input_activations": None,
                "output_activations": None,
                "targets": ["Linear"],
                "weights": {
                    "actorder": None,
                    "block_structure": None,
                    "dynamic": False,
                    "group_size": args.group_size,
                    "num_bits": 4,
                    "observer": "minmax",
                    "observer_kwargs": {},
                    "strategy": "group",
                    "symmetric": args.symmetric,
                    "type": "int",
                },
            }
        },
        "format": "pack-quantized",
        "ignore": ignore_rules,
        "kv_cache_scheme": None,
        "quant_method": "compressed-tensors",
        "quantization_status": "compressed",
    }

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    logger.info(f"Done! Quantized {total_quantized} layers, skipped {total_skipped}")
    logger.info(f"Output saved to {save_dir}")


if __name__ == "__main__":
    main()
