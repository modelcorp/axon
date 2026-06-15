"""Run lifecycle: metadata, launching, and stdout tee.

By default, run data lives under ``~/.axon/runs/<run_id>/``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from omegaconf import OmegaConf

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


def _default_runs_dir() -> Path:
    """Return the directory used for local run metadata.

    ``AXON_RUNS_DIR`` is useful for shared clusters and CI where writing under
    the user's home directory is inconvenient.
    """
    override = os.environ.get("AXON_RUNS_DIR")
    return Path(override).expanduser() if override else Path.home() / ".axon" / "runs"


RUNS_DIR = _default_runs_dir()

# ---------------------------------------------------------------------------
# Run ID helpers
# ---------------------------------------------------------------------------


def generate_run_id() -> str:
    """Return an 8-char hex run ID."""
    return uuid4().hex[:8]


def resolve_run_id(partial: str) -> str | None:
    """Prefix-match *partial* against known run IDs.

    Returns the full ID if exactly one match, else ``None``.
    """
    if not RUNS_DIR.exists():
        return None
    matches = [d.name for d in RUNS_DIR.iterdir() if d.is_dir() and d.name.startswith(partial)]
    return matches[0] if len(matches) == 1 else None


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _load_meta(run_dir: Path) -> dict | None:
    meta_file = run_dir / "meta.json"
    if not meta_file.exists():
        return None
    try:
        meta = json.loads(meta_file.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    if meta.get("status") == "running" and not _pid_alive(meta.get("pid", -1)):
        meta["status"] = "crashed"
        try:
            meta_file.write_text(json.dumps(meta, indent=2) + "\n")
        except OSError:
            pass
    return meta


def update_meta_status(run_dir: Path, status: str) -> None:
    """Read meta.json from *run_dir*, set *status*, and write it back."""
    meta_path = run_dir / "meta.json"
    meta = json.loads(meta_path.read_text())
    meta["status"] = status
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")


def create_run(run_id: str, pid: int, status: str = "running", meta_extras: dict | None = None) -> Path:
    """Create ``~/.axon/runs/<run_id>/`` and write initial ``meta.json``."""
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "run_id": run_id,
        "pid": pid,
        "status": status,
        "started": datetime.now(timezone.utc).isoformat(),
        "log_path": str(run_dir / "train.log"),
    }
    if meta_extras:
        meta.update(meta_extras)
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return run_dir


def get_run(run_id: str) -> dict | None:
    """Return metadata for a single run (exact ID)."""
    run_dir = RUNS_DIR / run_id
    if not run_dir.is_dir():
        return None
    return _load_meta(run_dir)


def list_runs() -> list[dict]:
    """Return metadata for every run, newest first."""
    if not RUNS_DIR.exists():
        return []
    runs = []
    for entry in RUNS_DIR.iterdir():
        if entry.is_dir():
            meta = _load_meta(entry)
            if meta is not None:
                runs.append(meta)
    runs.sort(key=lambda m: m.get("started", ""), reverse=True)
    return runs


# ---------------------------------------------------------------------------
# Launcher (async / background)
# ---------------------------------------------------------------------------


def launch_training(cfg, run_id: str, run_dir: Path) -> int:
    """Write config, spawn a detached child, and return its PID."""
    OmegaConf.save(cfg, str(run_dir / "config.yaml"))
    log_fh = (run_dir / "train.log").open("w")

    bootstrap = textwrap.dedent(f"""\
        import sys, importlib
        sys.argv = ["axon-train"]
        mod = importlib.import_module("axon.cli._runs")
        mod._train_entry({str(run_dir)!r})
    """)

    proc = subprocess.Popen(
        [sys.executable, "-c", bootstrap],
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    log_fh.close()
    return proc.pid


def _train_entry(run_dir_str: str) -> None:
    """Child-process entry point: load config, train, update meta."""
    run_dir = Path(run_dir_str)
    try:
        cfg = OmegaConf.load(str(run_dir / "config.yaml"))
        from axon.driver.train_agent_ppo import run_ppo_agent

        run_ppo_agent(cfg)
        update_meta_status(run_dir, "completed")
    except BaseException:
        import traceback

        traceback.print_exc()
        update_meta_status(run_dir, "crashed")
        raise SystemExit(1) from None


# ---------------------------------------------------------------------------
# Tee (foreground logging)
# ---------------------------------------------------------------------------


class _TeeStream:
    """Wraps a stream so writes go to both the original and a file."""

    def __init__(self, original, log_fh):
        self._original = original
        self._log_fh = log_fh

    def write(self, data):
        self._original.write(data)
        self._log_fh.write(data)
        self._log_fh.flush()

    def flush(self):
        self._original.flush()
        self._log_fh.flush()

    def __getattr__(self, name):
        return getattr(self._original, name)


@contextmanager
def tee_to_file(log_path: Path):
    """Context manager that tees stdout and stderr to *log_path*."""
    fh = log_path.open("w")
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout = _TeeStream(old_stdout, fh)
    sys.stderr = _TeeStream(old_stderr, fh)
    try:
        yield
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        fh.close()
