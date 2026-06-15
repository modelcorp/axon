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
"""Data sampler module for data loading."""

from enum import Enum
import inspect

from omegaconf import DictConfig
import torch
from torch.utils.data import Dataset, Sampler

from axon.utils.module_loader import load_module

__all__ = ["SamplerType", "create_data_sampler"]


class SamplerType(str, Enum):
    SEQUENTIAL = "sequential"
    RANDOM = "random"
    EXP_WEIGHTED_CURRICULUM = "exp_weighted_curriculum"
    THRESHOLD_MASK_CURRICULUM = "threshold_masking_curriculum"


def create_data_sampler(config: DictConfig, dataset: Dataset) -> Sampler:
    """Create a sampler from config.

    Args:
        config: Config with `data_sampler` (SamplerType or path) and optional `data_sampler_args`.
        dataset: The dataset to sample from.
    """

    def _load_class(spec: str) -> type:
        """Load a class from pkg://module:Class or /path/file.py:Class."""

        if spec.startswith("pkg://"):
            rest = spec[len("pkg://") :]
            module_path, class_name = rest.rsplit(":", 1) if ":" in rest else (rest, rest.rsplit(".", 1)[-1])
            module_path = f"pkg://{module_path}"
        elif spec.startswith("file://"):
            module_path, class_name = spec[len("file://") :].rsplit(":", 1)
        else:
            module_path, class_name = spec.rsplit(":", 1)

        module = load_module(module_path)
        return getattr(module, class_name)


    def _filter_args(cls: type, args: dict) -> dict:
        """Filter args to only include parameters accepted by cls.__init__."""
        sig = inspect.signature(cls.__init__)
        valid_params = set(sig.parameters.keys()) - {"self"}

        # Check if **kwargs is accepted
        has_var_keyword = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
        if has_var_keyword:
            return args

        return {k: v for k, v in args.items() if k in valid_params}

    sampler_type = config.get("data_sampler", "sequential")
    sampler_args = config.get("data_sampler_args", {}) or {}

    # data_mix requires random sampling — force it and warn
    sample_weights = getattr(dataset, "sample_weights", None)
    if sample_weights is not None and sampler_type not in (SamplerType.RANDOM, "random"):
        import warnings
        warnings.warn(
            f"data_mix is set but data_sampler is '{sampler_type}'. "
            f"Overriding to 'random' (data_mix is incompatible with other samplers).",
            stacklevel=2,
        )
        sampler_type = "random"

    if sampler_type == SamplerType.SEQUENTIAL or sampler_type == "sequential":
        from torch.utils.data import SequentialSampler
        return SequentialSampler(data_source=dataset)

    if sampler_type == SamplerType.RANDOM or sampler_type == "random":
        # Use WeightedRandomSampler when the dataset provides per-file sampling weights
        sample_weights = getattr(dataset, "sample_weights", None)
        if sample_weights is not None:
            generator = torch.Generator()
            seed = sampler_args.get("seed")
            if seed is not None:
                generator.manual_seed(seed)
            return torch.utils.data.WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(dataset),
                replacement=True,
                generator=generator,
            )
        from torchdata.stateful_dataloader.sampler import RandomSampler
        generator = torch.Generator()
        seed = sampler_args.get("seed")
        if seed is not None:
            generator.manual_seed(seed)
        return RandomSampler(data_source=dataset, generator=generator)

    if sampler_type == SamplerType.EXP_WEIGHTED_CURRICULUM or sampler_type == "exp_weighted_curriculum":
        from axon.data.data_sampler.curriculum_samplers import ExpWeightedCurriculumSampler
        sampler_cls = ExpWeightedCurriculumSampler
    elif sampler_type == SamplerType.THRESHOLD_MASK_CURRICULUM or sampler_type == "threshold_masking_curriculum":
        from axon.data.data_sampler.curriculum_samplers import ThresholdMaskingSampler
        sampler_cls = ThresholdMaskingSampler
    else:
        # Load external sampler class from path
        sampler_cls = _load_class(sampler_type)
    filtered_args = _filter_args(sampler_cls, sampler_args)
    return sampler_cls(data_source=dataset, **filtered_args)
