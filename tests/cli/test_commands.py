"""Tests for axon.cli._commands.

Integration-style tests that use real directories via the _runs module,
plus edge-case tests for error handling in cancel/logs.
"""

import json
import signal
from pathlib import Path
from unittest.mock import patch

import click
import pytest
from click.testing import CliRunner

import axon.cli._runs as runs_mod
from axon.cli._commands import _format_time, _resolve_or_fail, cancel, logs, status


@pytest.fixture(autouse=True)
def _isolate_runs_dir(tmp_path, monkeypatch):
    """Redirect RUNS_DIR to tmp so tests never touch ~/.axon/runs/."""
    monkeypatch.setattr(runs_mod, "RUNS_DIR", tmp_path / "runs")
    # Also patch the _commands module's imported references
    monkeypatch.setattr("axon.cli._commands.RUNS_DIR", tmp_path / "runs")


# =========================================================================
# _format_time
# =========================================================================


class TestFormatTime:
    @pytest.mark.parametrize(
        "inp,expected",
        [
            (None, "\u2014"),
            ("", "\u2014"),
            ("2025-01-15T12:30:45.123+00:00", "2025-01-15 12:30"),
            ("2025-01-15", "2025-01-15"),
        ],
    )
    def test_format_time(self, inp, expected):
        assert _format_time(inp) == expected


# =========================================================================
# _resolve_or_fail
# =========================================================================


class TestResolveOrFail:
    def test_resolves_partial_id_against_real_directory(self, tmp_path):
        runs_mod.create_run("abcd1234", pid=1)
        full_id, meta = _resolve_or_fail("abcd")
        assert full_id == "abcd1234"
        assert meta["run_id"] == "abcd1234"

    def test_exact_8char_id(self, tmp_path):
        runs_mod.create_run("abcd1234", pid=1)
        full_id, _ = _resolve_or_fail("abcd1234")
        assert full_id == "abcd1234"

    def test_no_match_raises(self, tmp_path):
        with pytest.raises(click.ClickException, match="No run matching"):
            _resolve_or_fail("zzz")

    def test_ambiguous_prefix_raises(self, tmp_path):
        runs_mod.create_run("abcd1111", pid=1)
        runs_mod.create_run("abcd2222", pid=2)
        with pytest.raises(click.ClickException, match="No run matching"):
            _resolve_or_fail("abcd")


# =========================================================================
# status — integration with real runs
# =========================================================================


class TestStatusIntegration:
    def test_no_runs(self):
        result = CliRunner().invoke(status, [])
        assert result.exit_code == 0
        assert "No runs found" in result.output

    def test_lists_multiple_real_runs(self, tmp_path):
        runs_mod.create_run("aaa11111", pid=1, status="running")
        runs_mod.create_run("bbb22222", pid=2, status="completed")
        result = CliRunner().invoke(status, [])
        assert result.exit_code == 0
        assert "aaa11111" in result.output
        assert "bbb22222" in result.output

    def test_single_run_detail(self, tmp_path):
        runs_mod.create_run("aaa11111", pid=1234, status="completed")
        result = CliRunner().invoke(status, ["aaa11111"])
        assert result.exit_code == 0
        assert "completed" in result.output
        assert "1234" in result.output

    def test_status_detects_crashed_run(self, tmp_path, monkeypatch):
        """A 'running' run whose PID is dead should show as 'crashed'."""
        runs_mod.create_run("dead0000", pid=2**30, status="running")
        monkeypatch.setattr(runs_mod, "_pid_alive", lambda pid: False)
        result = CliRunner().invoke(status, ["dead0000"])
        assert "crashed" in result.output


# =========================================================================
# logs — integration with real files
# =========================================================================


class TestLogsIntegration:
    def test_shows_last_n_lines(self, tmp_path):
        d = runs_mod.create_run("log00001", pid=1)
        log_path = Path(d) / "train.log"
        log_path.write_text("\n".join(f"line {i}" for i in range(200)))
        result = CliRunner().invoke(logs, ["log00001", "-n", "3"])
        assert result.exit_code == 0
        assert "line 197" in result.output
        assert "line 199" in result.output
        assert "line 50" not in result.output

    def test_empty_log_file(self, tmp_path):
        d = runs_mod.create_run("log00002", pid=1)
        Path(d, "train.log").write_text("")
        result = CliRunner().invoke(logs, ["log00002"])
        assert result.exit_code == 0

    def test_missing_log_file_errors(self, tmp_path):
        runs_mod.create_run("log00003", pid=1)
        # Don't create the log file
        result = CliRunner().invoke(logs, ["log00003"])
        assert result.exit_code != 0


# =========================================================================
# cancel — integration with real runs
# =========================================================================


class TestCancelIntegration:
    def test_not_running_refuses(self, tmp_path):
        runs_mod.create_run("can00001", pid=1, status="completed")
        result = CliRunner().invoke(cancel, ["can00001"])
        assert result.exit_code != 0

    def test_cancel_sends_sigterm_and_updates_meta(self, tmp_path):
        d = runs_mod.create_run("can00002", pid=99999, status="running")
        with patch("os.kill") as mock_kill:
            result = CliRunner().invoke(cancel, ["can00002"])
        assert result.exit_code == 0
        # os.kill may be called twice: once for _pid_alive(pid, 0) and once for SIGTERM
        sigterm_calls = [c for c in mock_kill.call_args_list if c == ((99999, signal.SIGTERM),)]
        assert len(sigterm_calls) == 1
        # Verify meta.json updated on disk
        meta = json.loads((Path(d) / "meta.json").read_text())
        assert meta["status"] == "cancelled"

    def test_cancel_already_dead_process(self, tmp_path, monkeypatch):
        runs_mod.create_run("can00003", pid=99999, status="running")
        # Keep _pid_alive returning True so _load_meta doesn't flip to 'crashed'
        monkeypatch.setattr(runs_mod, "_pid_alive", lambda pid: True)

        def kill_side_effect(pid, sig):
            if sig == 0:
                return  # _pid_alive probe
            raise ProcessLookupError  # actual SIGTERM — process died between check and kill

        with patch("os.kill", side_effect=kill_side_effect):
            result = CliRunner().invoke(cancel, ["can00003"])
        assert result.exit_code == 0
        assert "dead" in result.output.lower()

    def test_cancel_permission_denied(self, tmp_path):
        runs_mod.create_run("can00004", pid=1, status="running")
        with patch("os.kill", side_effect=PermissionError("nope")):
            result = CliRunner().invoke(cancel, ["can00004"])
        assert result.exit_code != 0

    def test_cancel_zero_pid_refuses(self, tmp_path):
        runs_mod.create_run("can00005", pid=0, status="running")
        result = CliRunner().invoke(cancel, ["can00005"])
        assert result.exit_code != 0

    def test_full_lifecycle_create_then_cancel(self, tmp_path, monkeypatch):
        """Integration: create a run, verify it's running, cancel it, verify cancelled."""
        runs_mod.create_run("life0001", pid=99999, status="running")
        monkeypatch.setattr(runs_mod, "_pid_alive", lambda pid: True)

        # Status should show running
        r1 = CliRunner().invoke(status, ["life0001"])
        assert "running" in r1.output

        # Cancel
        with patch("os.kill"):
            r2 = CliRunner().invoke(cancel, ["life0001"])
        assert r2.exit_code == 0

        # Status should now show cancelled
        r3 = CliRunner().invoke(status, ["life0001"])
        assert "cancelled" in r3.output
