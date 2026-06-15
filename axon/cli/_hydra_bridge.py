"""Bridge between Click CLI arguments and Hydra config composition."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig

# ---------------------------------------------------------------------------
# Absolute path to the built-in Hydra config directory
# ---------------------------------------------------------------------------
_CONFIG_DIR = str(Path(__file__).resolve().parent.parent / "config")


# ---------------------------------------------------------------------------
# YAML → Hydra overrides
# ---------------------------------------------------------------------------


def _to_hydra_literal(value: Any) -> str:
    """Convert a Python scalar to its Hydra override literal."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _flatten(prefix: str, obj: Any) -> list[str]:
    """Recursively flatten a nested dict into Hydra dot-notation overrides.

    Every leaf override is emitted with the ``++`` (force-add) prefix so that
    it works regardless of whether the key already exists in the base config.
    Users never need to worry about Hydra's ``+`` / ``++`` semantics.

    Lists are serialised as inline YAML (``[a, b, c]``).
    """
    if isinstance(obj, dict):
        items: list[str] = []
        for key, value in obj.items():
            new_prefix = f"{prefix}.{key}" if prefix else str(key)
            items.extend(_flatten(new_prefix, value))
        return items

    # Scalar or list — serialise to a Hydra-safe string
    if isinstance(obj, list):
        rendered = "[" + ", ".join(_to_hydra_literal(v) for v in obj) + "]"
        return [f"++{prefix}={rendered}"]

    # Scalar value
    return [f"++{prefix}={_to_hydra_literal(obj)}"]


def flatten_yaml_to_overrides(yaml_path: str | Path) -> list[str]:
    """Load a user YAML file and return a list of Hydra override strings."""
    path = Path(yaml_path)
    with path.open() as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping at top level, got {type(data).__name__}")
    return _flatten("", data)


# ---------------------------------------------------------------------------
# CLI flags → Hydra overrides
# ---------------------------------------------------------------------------

# Maps (Click kwarg name → Hydra config key).
_FLAG_MAP: dict[str, str] = {
    "model": "model_path",
    "train_data": "train_files",
    "val_data": "val_files",
    "gpus": "num_gpus_per_node",
    "nodes": "num_nodes",
    "experiment_name": "experiment_name",
    "output_dir": "output_dir",
    "resume": "resume_from_checkpoint",
}


def build_hydra_overrides(**kwargs: Any) -> list[str]:
    """Convert explicit CLI flags to Hydra override strings.

    Only flags that were actually provided (not ``None``) are emitted.
    """
    overrides: list[str] = []
    for flag, config_key in _FLAG_MAP.items():
        value = kwargs.get(flag)
        if value is not None:
            overrides.append(f"{config_key}={value}")
    return overrides


# ---------------------------------------------------------------------------
# Compose the final OmegaConf config
# ---------------------------------------------------------------------------


def compose_config(config_name: str, overrides: list[str]) -> DictConfig:
    """Use Hydra Compose API to build a fully-resolved ``DictConfig``.

    Parameters
    ----------
    config_name:
        Name of the base config (e.g. ``config``).
    overrides:
        List of Hydra-style overrides (``key=value``).
    """
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=_CONFIG_DIR, version_base=None):
        cfg = compose(config_name=config_name, overrides=overrides)
    return cfg
