"""Tests for axon.cli._runs module."""

from __future__ import annotations

import io
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import pytest

import axon.cli._runs as runs_mod
from axon.cli._runs import (
    _load_meta,
    _pid_alive,
    _TeeStream,
    create_run,
    generate_run_id,
    get_run,
    list_runs,
    resolve_run_id,
    tee_to_file,
    update_meta_status,
)

# ---------------------------------------------------------------------------
# Fixture: redirect RUNS_DIR to a temp directory
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolate_runs_dir(tmp_path, monkeypatch):
    """Point RUNS_DIR at a temporary directory so tests never touch ~/.axon/runs/."""
    monkeypatch.setattr(runs_mod, "RUNS_DIR", tmp_path / "runs")


# ---------------------------------------------------------------------------
# generate_run_id
# ---------------------------------------------------------------------------


class TestGenerateRunId:
    def test_returns_unique_8_char_hex_ids(self):
        ids = {generate_run_id() for _ in range(50)}
        # All 50 must be unique
        assert len(ids) == 50
        for rid in ids:
            assert len(rid) == 8
            assert all(c in "0123456789abcdef" for c in rid)


class TestDefaultRunsDir:
    def test_defaults_to_home_axon_runs(self, monkeypatch, tmp_path):
        monkeypatch.delenv("AXON_RUNS_DIR", raising=False)
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        assert runs_mod._default_runs_dir() == tmp_path / ".axon" / "runs"

    def test_env_override(self, monkeypatch, tmp_path):
        custom = tmp_path / "custom-runs"
        monkeypatch.setenv("AXON_RUNS_DIR", str(custom))
        assert runs_mod._default_runs_dir() == custom


# ---------------------------------------------------------------------------
# resolve_run_id
# ---------------------------------------------------------------------------


class TestResolveRunId:
    def test_exact_match(self, tmp_path):
        run_dir = tmp_path / "runs" / "abcd1234"
        run_dir.mkdir(parents=True)
        assert resolve_run_id("abcd1234") == "abcd1234"

    def test_prefix_match_unique(self, tmp_path):
        run_dir = tmp_path / "runs" / "abcd1234"
        run_dir.mkdir(parents=True)
        assert resolve_run_id("abcd") == "abcd1234"

    def test_prefix_match_ambiguous_returns_none(self, tmp_path):
        (tmp_path / "runs" / "abcd1234").mkdir(parents=True)
        (tmp_path / "runs" / "abcd5678").mkdir(parents=True)
        assert resolve_run_id("abcd") is None

    def test_no_matches_returns_none(self, tmp_path):
        (tmp_path / "runs").mkdir(parents=True)
        assert resolve_run_id("zzzz") is None

    def test_runs_dir_does_not_exist_returns_none(self):
        # The autouse fixture points RUNS_DIR at tmp_path/"runs" which doesn't exist yet
        assert resolve_run_id("anything") is None

    def test_empty_prefix_multiple_dirs_returns_none(self, tmp_path):
        """Empty string matches every directory name; multiple matches -> None."""
        (tmp_path / "runs" / "aaaa0000").mkdir(parents=True)
        (tmp_path / "runs" / "bbbb1111").mkdir(parents=True)
        assert resolve_run_id("") is None

    def test_empty_prefix_single_dir_returns_that_dir(self, tmp_path):
        """Empty string with exactly one directory should return its name."""
        (tmp_path / "runs" / "only_one").mkdir(parents=True)
        assert resolve_run_id("") == "only_one"

    def test_ignores_regular_files_in_runs_dir(self, tmp_path):
        """Only directories should be considered, not plain files."""
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir(parents=True)
        (runs_dir / "abcd1234").mkdir()
        # Create a file (not a directory) that would also match the prefix
        (runs_dir / "abcd9999").write_text("I am a file, not a dir")
        assert resolve_run_id("abcd") == "abcd1234"


# ---------------------------------------------------------------------------
# _pid_alive
# ---------------------------------------------------------------------------


class TestPidAlive:
    def test_current_process_is_alive(self):
        """os.getpid() is the running process; _pid_alive must return True."""
        assert _pid_alive(os.getpid()) is True

    def test_nonexistent_pid_is_not_alive(self):
        """PID 2**30 is almost certainly not running; _pid_alive must return False."""
        assert _pid_alive(2**30) is False


# ---------------------------------------------------------------------------
# create_run
# ---------------------------------------------------------------------------


class TestCreateRun:
    def test_creates_directory_and_meta(self, tmp_path):
        run_dir = create_run("aabbccdd", pid=12345)
        assert run_dir.is_dir()
        meta_path = run_dir / "meta.json"
        assert meta_path.exists()

    def test_meta_has_expected_keys(self, tmp_path):
        run_dir = create_run("aabbccdd", pid=12345)
        meta = json.loads((run_dir / "meta.json").read_text())
        assert meta["run_id"] == "aabbccdd"
        assert meta["pid"] == 12345
        assert meta["status"] == "running"
        assert "started" in meta
        assert "log_path" in meta

    def test_custom_status(self, tmp_path):
        run_dir = create_run("aabbccdd", pid=1, status="queued")
        meta = json.loads((run_dir / "meta.json").read_text())
        assert meta["status"] == "queued"

    def test_meta_extras_included(self, tmp_path):
        extras = {"model": "gpt-test", "epochs": 3}
        run_dir = create_run("aabbccdd", pid=1, meta_extras=extras)
        meta = json.loads((run_dir / "meta.json").read_text())
        assert meta["model"] == "gpt-test"
        assert meta["epochs"] == 3

    def test_meta_extras_can_override_defaults(self, tmp_path):
        """meta_extras is applied after defaults, so it can override them."""
        run_dir = create_run("aabbccdd", pid=1, meta_extras={"status": "custom"})
        meta = json.loads((run_dir / "meta.json").read_text())
        assert meta["status"] == "custom"

    def test_duplicate_run_id_overwrites_meta(self, tmp_path):
        """Calling create_run twice with the same ID should overwrite meta.json."""
        run_dir_1 = create_run("deadbeef", pid=100, status="running")
        run_dir_2 = create_run("deadbeef", pid=200, status="queued")
        assert run_dir_1 == run_dir_2
        meta = json.loads((run_dir_2 / "meta.json").read_text())
        assert meta["pid"] == 200
        assert meta["status"] == "queued"

    def test_started_is_valid_iso_timestamp(self, tmp_path):
        """The 'started' field must be a parseable ISO 8601 timestamp."""
        run_dir = create_run("aabbccdd", pid=1)
        meta = json.loads((run_dir / "meta.json").read_text())
        parsed = datetime.fromisoformat(meta["started"])
        assert isinstance(parsed, datetime)


# ---------------------------------------------------------------------------
# update_meta_status
# ---------------------------------------------------------------------------


class TestUpdateMetaStatus:
    def test_updates_status_field(self, tmp_path):
        run_dir = create_run("aabbccdd", pid=1, status="running")
        update_meta_status(run_dir, "completed")
        meta = json.loads((run_dir / "meta.json").read_text())
        assert meta["status"] == "completed"
        # Other fields should be preserved
        assert meta["run_id"] == "aabbccdd"
        assert meta["pid"] == 1

    def test_nonexistent_file_raises(self, tmp_path):
        """update_meta_status on a directory without meta.json should raise."""
        bogus_dir = tmp_path / "runs" / "no_meta"
        bogus_dir.mkdir(parents=True)
        with pytest.raises(FileNotFoundError):
            update_meta_status(bogus_dir, "completed")


# ---------------------------------------------------------------------------
# get_run
# ---------------------------------------------------------------------------


class TestGetRun:
    def test_returns_meta_for_existing_run(self, tmp_path):
        create_run("aabbccdd", pid=99999, status="completed")
        meta = get_run("aabbccdd")
        assert meta is not None
        assert meta["run_id"] == "aabbccdd"
        assert meta["status"] == "completed"

    def test_returns_none_for_nonexistent(self, tmp_path):
        assert get_run("nonexistent") is None


# ---------------------------------------------------------------------------
# list_runs
# ---------------------------------------------------------------------------


class TestListRuns:
    def test_returns_runs_sorted_newest_first(self, tmp_path):
        # Create runs with explicit started timestamps to control order
        d1 = create_run("run_old", pid=1)
        d2 = create_run("run_new", pid=2)
        # Patch started timestamps so ordering is deterministic
        meta1 = json.loads((d1 / "meta.json").read_text())
        meta1["started"] = "2025-01-01T00:00:00+00:00"
        (d1 / "meta.json").write_text(json.dumps(meta1, indent=2) + "\n")
        meta2 = json.loads((d2 / "meta.json").read_text())
        meta2["started"] = "2025-06-01T00:00:00+00:00"
        (d2 / "meta.json").write_text(json.dumps(meta2, indent=2) + "\n")

        result = list_runs()
        assert len(result) == 2
        assert result[0]["run_id"] == "run_new"
        assert result[1]["run_id"] == "run_old"

    def test_returns_empty_list_when_no_runs(self):
        # RUNS_DIR does not exist at all
        assert list_runs() == []

    def test_skips_directories_without_meta(self, tmp_path):
        create_run("good_run", pid=1)
        (tmp_path / "runs" / "bad_run").mkdir(parents=True)
        # bad_run has no meta.json
        result = list_runs()
        assert len(result) == 1
        assert result[0]["run_id"] == "good_run"

    def test_ignores_regular_files_mixed_with_directories(self, tmp_path):
        """list_runs should only process directories, not regular files."""
        create_run("real_run", pid=1)
        runs_dir = tmp_path / "runs"
        # Plant some non-directory files that should be ignored
        (runs_dir / "stray_file.txt").write_text("not a run")
        (runs_dir / "another_file").write_text("{}")
        result = list_runs()
        assert len(result) == 1
        assert result[0]["run_id"] == "real_run"

    def test_sort_correctness_with_many_runs(self, tmp_path):
        """Verify sort with 5+ runs to exercise ordering beyond a trivial pair."""
        timestamps = [
            ("run_c", "2025-03-01T00:00:00+00:00"),
            ("run_a", "2025-01-01T00:00:00+00:00"),
            ("run_e", "2025-05-01T00:00:00+00:00"),
            ("run_b", "2025-02-01T00:00:00+00:00"),
            ("run_d", "2025-04-01T00:00:00+00:00"),
        ]
        for run_id, ts in timestamps:
            d = create_run(run_id, pid=1)
            meta = json.loads((d / "meta.json").read_text())
            meta["started"] = ts
            (d / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")

        result = list_runs()
        assert len(result) == 5
        ids = [r["run_id"] for r in result]
        assert ids == ["run_e", "run_d", "run_c", "run_b", "run_a"]


# ---------------------------------------------------------------------------
# _load_meta
# ---------------------------------------------------------------------------


class TestLoadMeta:
    def test_returns_none_for_missing_meta_json(self, tmp_path):
        run_dir = tmp_path / "runs" / "nofile"
        run_dir.mkdir(parents=True)
        assert _load_meta(run_dir) is None

    def test_returns_none_for_corrupt_json(self, tmp_path):
        run_dir = tmp_path / "runs" / "corrupt"
        run_dir.mkdir(parents=True)
        (run_dir / "meta.json").write_text("NOT VALID JSON {{{")
        assert _load_meta(run_dir) is None

    def test_marks_crashed_when_pid_dead(self, tmp_path, monkeypatch):
        run_dir = tmp_path / "runs" / "deadpid"
        run_dir.mkdir(parents=True)
        meta = {"run_id": "deadpid", "pid": -99999, "status": "running", "started": "t"}
        (run_dir / "meta.json").write_text(json.dumps(meta))

        # Ensure _pid_alive returns False for the dead pid
        monkeypatch.setattr(runs_mod, "_pid_alive", lambda pid: False)

        result = _load_meta(run_dir)
        assert result["status"] == "crashed"
        # Verify it was also written back to disk
        on_disk = json.loads((run_dir / "meta.json").read_text())
        assert on_disk["status"] == "crashed"

    def test_does_not_mark_crashed_when_pid_alive(self, tmp_path, monkeypatch):
        run_dir = tmp_path / "runs" / "alivepid"
        run_dir.mkdir(parents=True)
        meta = {"run_id": "alivepid", "pid": 1, "status": "running", "started": "t"}
        (run_dir / "meta.json").write_text(json.dumps(meta))

        monkeypatch.setattr(runs_mod, "_pid_alive", lambda pid: True)

        result = _load_meta(run_dir)
        assert result["status"] == "running"

    def test_does_not_touch_completed_status(self, tmp_path, monkeypatch):
        """Only status=='running' triggers the pid-alive check."""
        run_dir = tmp_path / "runs" / "done"
        run_dir.mkdir(parents=True)
        meta = {"run_id": "done", "pid": -99999, "status": "completed", "started": "t"}
        (run_dir / "meta.json").write_text(json.dumps(meta))

        monkeypatch.setattr(runs_mod, "_pid_alive", lambda pid: False)

        result = _load_meta(run_dir)
        assert result["status"] == "completed"

    def test_extra_unexpected_keys_passed_through(self, tmp_path):
        """Unknown keys in meta.json should be preserved in the returned dict."""
        run_dir = tmp_path / "runs" / "extras"
        run_dir.mkdir(parents=True)
        meta = {
            "run_id": "extras",
            "pid": 1,
            "status": "completed",
            "started": "t",
            "custom_field": [1, 2, 3],
            "nested": {"a": "b"},
        }
        (run_dir / "meta.json").write_text(json.dumps(meta))

        result = _load_meta(run_dir)
        assert result is not None
        assert result["custom_field"] == [1, 2, 3]
        assert result["nested"] == {"a": "b"}

    def test_empty_json_object_does_not_crash(self, tmp_path):
        """An empty JSON object {} has no 'status' key; _load_meta should not crash."""
        run_dir = tmp_path / "runs" / "empty_obj"
        run_dir.mkdir(parents=True)
        (run_dir / "meta.json").write_text("{}")

        result = _load_meta(run_dir)
        assert result is not None
        assert result == {}

    def test_crashed_detection_when_meta_is_readonly(self, tmp_path, monkeypatch):
        """When meta.json is read-only and process is dead, _load_meta should still
        return 'crashed' status even though the write-back fails (try/except OSError: pass)."""
        run_dir = tmp_path / "runs" / "readonly"
        run_dir.mkdir(parents=True)
        meta = {"run_id": "readonly", "pid": -99999, "status": "running", "started": "t"}
        meta_path = run_dir / "meta.json"
        meta_path.write_text(json.dumps(meta))
        # Make the file read-only
        meta_path.chmod(0o444)

        monkeypatch.setattr(runs_mod, "_pid_alive", lambda pid: False)

        result = _load_meta(run_dir)
        # The in-memory dict should still show crashed
        assert result["status"] == "crashed"
        # But on disk it should remain "running" because the write failed
        meta_path.chmod(0o644)  # restore perms so we can read
        on_disk = json.loads(meta_path.read_text())
        assert on_disk["status"] == "running"


# ---------------------------------------------------------------------------
# _TeeStream and tee_to_file
# ---------------------------------------------------------------------------


class TestTeeStream:
    def test_write_and_flush_go_to_both_streams(self):
        original = io.StringIO()
        log_fh = io.StringIO()
        tee = _TeeStream(original, log_fh)
        tee.write("hello")
        tee.flush()  # should not raise
        assert original.getvalue() == "hello"
        assert log_fh.getvalue() == "hello"

    def test_getattr_delegates_to_original(self):
        original = io.StringIO()
        log_fh = io.StringIO()
        tee = _TeeStream(original, log_fh)
        # StringIO has a 'getvalue' method; __getattr__ should delegate
        tee.write("delegated")
        assert tee.getvalue() == "delegated"

    def test_unicode_content(self):
        """Emoji and CJK characters should pass through correctly."""
        original = io.StringIO()
        log_fh = io.StringIO()
        tee = _TeeStream(original, log_fh)
        text = "\u2764\ufe0f \U0001f680 \u4f60\u597d\u4e16\u754c"
        tee.write(text)
        assert original.getvalue() == text
        assert log_fh.getvalue() == text

    def test_empty_string_write(self):
        """Writing an empty string should not crash and should propagate."""
        original = io.StringIO()
        log_fh = io.StringIO()
        tee = _TeeStream(original, log_fh)
        tee.write("")
        tee.write("after")
        assert original.getvalue() == "after"
        assert log_fh.getvalue() == "after"


class TestTeeToFile:
    def test_restores_streams_after_exception(self, tmp_path):
        log_path = tmp_path / "test.log"
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        with pytest.raises(RuntimeError):
            with tee_to_file(log_path):
                raise RuntimeError("boom")
        assert sys.stdout is old_stdout
        assert sys.stderr is old_stderr

    def test_stdout_and_stderr_both_captured_in_same_file(self, tmp_path):
        """Both stdout and stderr should end up in the same log file."""
        log_path = tmp_path / "combined.log"
        with tee_to_file(log_path):
            print("from-stdout", end="", file=sys.stdout)
            print("from-stderr", end="", file=sys.stderr)
        contents = log_path.read_text()
        assert "from-stdout" in contents
        assert "from-stderr" in contents
