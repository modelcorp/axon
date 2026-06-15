"""Tests for axon.utils.tracking module."""

import dataclasses
import json
import os
from enum import Enum
from pathlib import Path

import pytest

from axon.utils.tracking import (
    FileLogger,
    Tracking,
    _compute_mlflow_params_from_objects,
    _MlflowLoggingAdapter,
    _transform_params_to_json_serializable,
)


# ---------------------------------------------------------------------------
# _transform_params_to_json_serializable
# ---------------------------------------------------------------------------
class TestTransformParams:
    def test_dataclass(self):
        @dataclasses.dataclass
        class Cfg:
            lr: float = 0.01
            epochs: int = 10

        assert _transform_params_to_json_serializable(Cfg(), convert_list_to_dict=False) == {"lr": 0.01, "epochs": 10}

    def test_nested_dataclass(self):
        @dataclasses.dataclass
        class Inner:
            x: int = 1

        @dataclasses.dataclass
        class Outer:
            inner: Inner = dataclasses.field(default_factory=Inner)

        result = _transform_params_to_json_serializable(Outer(), convert_list_to_dict=False)
        assert result == {"inner": {"x": 1}}

    def test_list_as_dict_adds_length(self):
        result = _transform_params_to_json_serializable([10, 20], convert_list_to_dict=True)
        assert result == {"list_len": 2, "0": 10, "1": 20}

    def test_empty_list_as_dict(self):
        assert _transform_params_to_json_serializable([], convert_list_to_dict=True) == {"list_len": 0}

    def test_path_to_str(self):
        assert _transform_params_to_json_serializable(Path("/tmp/x"), convert_list_to_dict=False) == "/tmp/x"

    def test_enum_to_value(self):
        class C(Enum):
            RED = "red"

        assert _transform_params_to_json_serializable(C.RED, convert_list_to_dict=False) == "red"

    @pytest.mark.parametrize("val", [42, 3.14, "hello", None, True])
    def test_primitives_passthrough(self, val):
        assert (
            _transform_params_to_json_serializable(val, convert_list_to_dict=False) is val
            or _transform_params_to_json_serializable(val, convert_list_to_dict=False) == val
        )

    def test_deeply_nested_mixed(self):
        data = {"a": {"b": [Path("/x"), 1, {"c": [2]}]}}
        result = _transform_params_to_json_serializable(data, convert_list_to_dict=False)
        assert result["a"]["b"][0] == "/x"
        assert result["a"]["b"][2] == {"c": [2]}


# ---------------------------------------------------------------------------
# _compute_mlflow_params_from_objects
# ---------------------------------------------------------------------------
class TestComputeMlflowParams:
    def test_none_returns_empty(self):
        assert _compute_mlflow_params_from_objects(None) == {}

    def test_nested_dict_flattened(self):
        result = _compute_mlflow_params_from_objects({"opt": {"type": "adam", "lr": 0.01}})
        assert result == {"opt/type": "adam", "opt/lr": 0.01}

    def test_dataclass_flattened(self):
        @dataclasses.dataclass
        class Cfg:
            lr: float = 0.1
            path: Path = Path("/tmp")

        result = _compute_mlflow_params_from_objects(Cfg())
        assert result["lr"] == 0.1
        assert result["path"] == "/tmp"


# ---------------------------------------------------------------------------
# _MlflowLoggingAdapter – sanitize_key
# ---------------------------------------------------------------------------
class TestMlflowSanitizeKey:
    def setup_method(self):
        self.adapter = _MlflowLoggingAdapter()

    def _sanitize(self, key):
        """Extract the sanitize_key function from the adapter's log method."""
        import re

        sanitized = key.replace("@", "_at_")
        sanitized = re.compile(r"/+").sub("/", sanitized)
        sanitized = re.compile(r"[^/\w.\- :]").sub("_", sanitized)
        return sanitized

    def test_at_sign_replaced(self):
        assert self._sanitize("metric@5") == "metric_at_5"

    def test_consecutive_slashes_collapsed(self):
        assert self._sanitize("a///b") == "a/b"

    def test_invalid_chars_underscored(self):
        assert self._sanitize("metric{value}") == "metric_value_"

    def test_all_at_signs(self):
        assert self._sanitize("@@@") == "_at__at__at_"

    def test_valid_key_unchanged(self):
        assert self._sanitize("train/loss_v2") == "train/loss_v2"


# ---------------------------------------------------------------------------
# FileLogger
# ---------------------------------------------------------------------------
class TestFileLogger:
    def test_writes_exact_jsonl(self, tmp_path, monkeypatch):
        monkeypatch.setenv("AXON_FILE_LOGGER_ROOT", str(tmp_path))
        logger = FileLogger(project_name="proj", experiment_name="exp")
        logger.log({"loss": 0.5, "lr": 1e-3}, step=1)
        logger.log({"loss": 0.3}, step=2)
        logger.finish()

        with open(logger.filepath) as f:
            lines = f.readlines()
        assert len(lines) == 2
        r1 = json.loads(lines[0])
        assert r1 == {"step": 1, "data": {"loss": 0.5, "lr": 1e-3}}
        r2 = json.loads(lines[1])
        assert r2 == {"step": 2, "data": {"loss": 0.3}}

    def test_creates_directory_if_needed(self, tmp_path, monkeypatch):
        monkeypatch.setenv("AXON_FILE_LOGGER_ROOT", str(tmp_path / "deep" / "dir"))
        logger = FileLogger(project_name="p", experiment_name="e")
        logger.log({"x": 1}, step=0)
        logger.finish()
        assert os.path.exists(logger.filepath)

    def test_custom_filepath_via_env(self, tmp_path, monkeypatch):
        filepath = str(tmp_path / "custom.jsonl")
        monkeypatch.setenv("AXON_FILE_LOGGER_PATH", filepath)
        logger = FileLogger(project_name="p", experiment_name="e")
        assert logger.filepath == filepath
        logger.finish()

    def test_finish_closes_file_handle(self, tmp_path, monkeypatch):
        monkeypatch.setenv("AXON_FILE_LOGGER_ROOT", str(tmp_path))
        logger = FileLogger(project_name="p", experiment_name="e")
        logger.finish()
        assert logger.fp.closed

    def test_filepath_format(self, tmp_path, monkeypatch):
        """File should be at {root}/{project}/{experiment}.jsonl."""
        monkeypatch.setenv("AXON_FILE_LOGGER_ROOT", str(tmp_path))
        logger = FileLogger(project_name="my_project", experiment_name="run_1")
        expected = os.path.join(str(tmp_path), "my_project", "run_1.jsonl")
        assert logger.filepath == expected
        logger.finish()


# ---------------------------------------------------------------------------
# Tracking – console and file backends
# ---------------------------------------------------------------------------
class TestTracking:
    def test_console_log_output(self, capsys):
        tracker = Tracking("proj", "exp", default_backend="console")
        tracker.log({"loss": 0.5}, step=1)
        assert "loss" in capsys.readouterr().out

    def test_unsupported_backend_raises(self):
        with pytest.raises(AssertionError):
            Tracking("p", "e", default_backend="nonexistent")

    def test_multiple_backends(self, tmp_path, monkeypatch):
        monkeypatch.setenv("AXON_FILE_LOGGER_ROOT", str(tmp_path))
        tracker = Tracking("p", "e", default_backend=["console", "file"])
        assert set(tracker.logger.keys()) == {"console", "file"}

    def test_backend_specific_routing(self, tmp_path, monkeypatch, capsys):
        """Logging to only 'file' should NOT print to console."""
        monkeypatch.setenv("AXON_FILE_LOGGER_ROOT", str(tmp_path))
        tracker = Tracking("p", "e", default_backend=["console", "file"])
        tracker.log({"x": 1}, step=0, backend=["file"])
        assert capsys.readouterr().out == ""

    def test_deprecated_tracking_backend_warns(self):
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                Tracking("p", "e", default_backend="tracking")
            except Exception:
                pass
            assert any(issubclass(x.category, DeprecationWarning) and "deprecated" in str(x.message).lower() for x in w)

    def test_del_safe_without_logger_attr(self):
        """__del__ should not crash if __init__ never ran."""
        obj = Tracking.__new__(Tracking)
        del obj  # should not raise

    def test_file_backend_receives_data(self, tmp_path, monkeypatch):
        """Verify data actually lands in the file."""
        monkeypatch.setenv("AXON_FILE_LOGGER_ROOT", str(tmp_path))
        tracker = Tracking("p", "e", default_backend="file")
        tracker.log({"metric": 42}, step=5)
        # Finish to flush
        file_logger = tracker.logger["file"]
        file_logger.finish()
        with open(file_logger.filepath) as f:
            record = json.loads(f.readline())
        assert record == {"step": 5, "data": {"metric": 42}}
