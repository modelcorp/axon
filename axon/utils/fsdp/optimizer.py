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
import importlib


def build_optimizer(parameters, optimizer_name, optimizer_args_config):
    """Build an optimizer by dynamically importing and instantiating the class.

    Args:
        parameters: Model parameters to optimize.
        optimizer_name: Optimizer class name (e.g. "AdamW", "AdamW8bit", "_AdamW").
        optimizer_args_config: Dict/OmegaConf with optimizer_impl, lr, weight_decay,
            betas, override_optimizer_args.

    Returns:
        Optimizer instance.
    """

    def _get(cfg, key, default=None):
        if hasattr(cfg, "get"):
            return cfg.get(key, default)
        return getattr(cfg, key, default)

    optimizer_impl = _get(optimizer_args_config, "optimizer_impl", "torch.optim")
    lr = _get(optimizer_args_config, "lr", 1e-3)
    weight_decay = _get(optimizer_args_config, "weight_decay", 0.1)
    betas = _get(optimizer_args_config, "betas", (0.9, 0.999))
    override_optimizer_args = _get(optimizer_args_config, "override_optimizer_args", None)

    optimizer_args = {
        "lr": lr,
        "weight_decay": weight_decay,
    }

    optimizer_name_lower = optimizer_name.lower()
    if "adam" in optimizer_name_lower or "ademamix" in optimizer_name_lower:
        optimizer_args["betas"] = betas

    if override_optimizer_args is not None:
        optimizer_args.update(override_optimizer_args)

    try:
        module = importlib.import_module(optimizer_impl)
        optimizer_cls = getattr(module, optimizer_name)
    except ImportError as e:
        raise ImportError(
            f"Failed to import module '{optimizer_impl}'. Make sure the package is installed. Error: {e}"
        ) from e
    except AttributeError as e:
        raise AttributeError(
            f"Optimizer '{optimizer_name}' not found in module '{optimizer_impl}'. Available optimizers: {dir(module)}"
        ) from e

    return optimizer_cls(parameters, **optimizer_args)
