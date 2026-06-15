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
"""
A unified tracking interface that supports logging data to different backend
"""

import dataclasses
import json
import logging
import os
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class Tracking:
    """A unified tracking interface for logging experiment data to multiple backends.

    This class provides a centralized way to log experiment metrics, parameters, and artifacts
    to various tracking backends including WandB, MLflow, SwanLab, TensorBoard, and console.

    Attributes:
        supported_backend: List of supported tracking backends.
        logger: Dictionary of initialized logger instances for each backend.
    """

    supported_backend = [
        "wandb",
        "mlflow",
        "swanlab",
        "vemlp_wandb",
        "tensorboard",
        "console",
        "clearml",
        "trackio",
        "file",
    ]

    def __init__(self, project_name, experiment_name, default_backend: str | list[str] = "console", config=None):
        if isinstance(default_backend, str):
            default_backend = [default_backend]
        for backend in default_backend:
            if backend == "tracking":
                import warnings

                warnings.warn("`tracking` logger is deprecated. use `wandb` instead.", DeprecationWarning, stacklevel=2)
            else:
                assert backend in self.supported_backend, f"{backend} is not supported"

        self.logger = {}

        if "tracking" in default_backend or "wandb" in default_backend:
            import os

            import wandb

            settings = None
            entity = os.environ.get("WANDB_ENTITY", None)
            wandb.init(
                project=project_name,
                name=experiment_name,
                entity=entity,
                config=config,
                settings=settings,
                resume="allow",
            )
            self.logger["wandb"] = wandb

        if "trackio" in default_backend:
            import trackio

            trackio.init(project=project_name, name=experiment_name, config=config)
            self.logger["trackio"] = trackio

        if "mlflow" in default_backend:
            import os

            import mlflow

            MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:////tmp/mlruns.db")
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

            # Project_name is actually experiment_name in MLFlow
            # If experiment does not exist, will create a new experiment
            experiment = mlflow.set_experiment(project_name)
            mlflow.start_run(experiment_id=experiment.experiment_id, run_name=experiment_name)
            mlflow.log_params(_compute_mlflow_params_from_objects(config))
            self.logger["mlflow"] = _MlflowLoggingAdapter()

        if "tensorboard" in default_backend:
            self.logger["tensorboard"] = _TensorboardAdapter(project_name, experiment_name)

        if "console" in default_backend:
            from axon.utils.logger import LocalLogger

            self.console_logger = LocalLogger(print_to_console=True)
            self.logger["console"] = self.console_logger

        if "clearml" in default_backend:
            self.logger["clearml"] = ClearMLLogger(project_name, experiment_name, config)

        if "file" in default_backend:
            self.logger["file"] = FileLogger(project_name, experiment_name)

    def log(self, data, step, backend=None, title=None):
        for default_backend, logger_instance in self.logger.items():
            if backend is None or default_backend in backend:
                if title is not None and hasattr(logger_instance, "log_with_title"):
                    logger_instance.log_with_title(data=data, step=step, title=title)
                else:
                    logger_instance.log(data=data, step=step)

    def finish(self):
        """Tear down all logging backends. Idempotent.

        Prefer this (or the context-manager form) over relying on ``__del__``:
        ``__del__`` is not guaranteed to run during interpreter shutdown, and
        modules a backend depends on (e.g. ``mlflow``) may already be torn down
        by the time the destructor fires.
        """
        if not getattr(self, "logger", None):
            return
        for name in list(self.logger.keys()):
            backend = self.logger.get(name)
            if backend is None:
                continue
            try:
                if name in ("wandb", "vemlp_wandb"):
                    backend.finish(exit_code=0)
                elif name == "mlflow":
                    import mlflow

                    mlflow.end_run()
                elif name in ("swanlab", "tensorboard", "clearml", "trackio", "file"):
                    backend.finish()
            except Exception:  # noqa: BLE001
                logger.exception("Error finishing tracking backend %r", name)
            finally:
                self.logger.pop(name, None)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finish()
        return False

    def __del__(self):
        # Defensive fallback; callers should invoke ``finish()`` explicitly.
        # Wrapped in try/except because ``__del__`` may fire during interpreter
        # teardown when modules have already been removed.
        try:
            self.finish()
        except Exception:  # noqa: BLE001
            pass


class ClearMLLogger:
    def __init__(self, project_name: str, experiment_name: str, config):
        self.project_name = project_name
        self.experiment_name = experiment_name

        import clearml

        self._task: clearml.Task = clearml.Task.init(
            task_name=experiment_name,
            project_name=project_name,
            continue_last_task=True,
            output_uri=False,
        )

        self._task.connect_configuration(config, name="Hyperparameters")

    def _get_logger(self):
        return self._task.get_logger()

    def log(self, data, step):
        import numpy as np
        import pandas as pd

        # logs = self._rewrite_logs(data)
        logger = self._get_logger()
        for k, v in data.items():
            title, series = k.split("/", 1)

            if isinstance(v, int | float | np.floating | np.integer):
                logger.report_scalar(
                    title=title,
                    series=series,
                    value=v,
                    iteration=step,
                )
            elif isinstance(v, pd.DataFrame):
                logger.report_table(
                    title=title,
                    series=series,
                    table_plot=v,
                    iteration=step,
                )
            else:
                logger.warning(
                    f'Trainer is attempting to log a value of "{v}" of type {type(v)} for key "{k}". This '
                    f"invocation of ClearML logger's function is incorrect so this attribute was dropped. "
                )

    def finish(self):
        self._task.close()


class FileLogger:
    def __init__(self, project_name: str, experiment_name: str):
        self.project_name = project_name
        self.experiment_name = experiment_name

        self.filepath = os.getenv("AXON_FILE_LOGGER_PATH", None)
        if self.filepath is None:
            root_path = os.path.expanduser(os.getenv("AXON_FILE_LOGGER_ROOT", "."))
            directory = os.path.join(root_path, self.project_name)
            os.makedirs(directory, exist_ok=True)
            self.filepath = os.path.join(directory, f"{self.experiment_name}.jsonl")
            logger.info("Creating file logger at %s", self.filepath)
        self.fp = open(self.filepath, "w")

    def log(self, data, step):
        data = {"step": step, "data": data}
        self.fp.write(json.dumps(data) + "\n")

    def finish(self):
        self.fp.close()


class _TensorboardAdapter:
    def __init__(self, project_name, experiment_name):
        import os

        from torch.utils.tensorboard import SummaryWriter

        tensorboard_dir = os.environ.get("TENSORBOARD_DIR", f"tensorboard_log/{project_name}/{experiment_name}")
        os.makedirs(tensorboard_dir, exist_ok=True)
        logger.info("Saving tensorboard log to %s.", tensorboard_dir)
        self.writer = SummaryWriter(tensorboard_dir)

    def log(self, data, step):
        for key in data:
            self.writer.add_scalar(key, data[key], step)

    def finish(self):
        self.writer.close()


class _MlflowLoggingAdapter:
    def __init__(self):
        import logging
        import re

        self.logger = logging.getLogger(__name__)
        # MLflow metric key validation logic:
        # https://github.com/mlflow/mlflow/blob/master/mlflow/utils/validation.py#L157C12-L157C44
        # Only characters allowed: slashes, alphanumerics, underscores, periods, dashes, colons,
        # and spaces.
        self._invalid_chars_pattern = re.compile(
            r"[^/\w.\- :]"
        )  # Allowed: slashes, alphanumerics, underscores, periods, dashes, colons, and spaces.
        self._consecutive_slashes_pattern = re.compile(r"/+")

    def log(self, data, step):
        import mlflow

        def sanitize_key(key):
            # First replace @ with _at_ for backward compatibility
            sanitized = key.replace("@", "_at_")
            # Replace consecutive slashes with a single slash (MLflow treats them as file paths)
            sanitized = self._consecutive_slashes_pattern.sub("/", sanitized)
            # Then replace any other invalid characters with _
            sanitized = self._invalid_chars_pattern.sub("_", sanitized)
            if sanitized != key:
                self.logger.warning(
                    "[MLflow] Metric key '%s' sanitized to '%s' due to invalid characters.", key, sanitized
                )
            return sanitized

        results = {sanitize_key(k): v for k, v in data.items()}
        mlflow.log_metrics(metrics=results, step=step)


def _compute_mlflow_params_from_objects(params) -> dict[str, Any]:
    if params is None:
        return {}

    return _flatten_dict(_transform_params_to_json_serializable(params, convert_list_to_dict=True), sep="/")


def _transform_params_to_json_serializable(x, convert_list_to_dict: bool):
    _transform = partial(_transform_params_to_json_serializable, convert_list_to_dict=convert_list_to_dict)

    if dataclasses.is_dataclass(x):
        return _transform(dataclasses.asdict(x))
    if isinstance(x, dict):
        return {k: _transform(v) for k, v in x.items()}
    if isinstance(x, list):
        if convert_list_to_dict:
            return {"list_len": len(x)} | {f"{i}": _transform(v) for i, v in enumerate(x)}
        else:
            return [_transform(v) for v in x]
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, Enum):
        return x.value

    return x


def _flatten_dict(raw: dict[str, Any], *, sep: str) -> dict[str, Any]:
    import pandas as pd

    ans = pd.json_normalize(raw, sep=sep).to_dict(orient="records")[0]
    assert isinstance(ans, dict)
    return ans
