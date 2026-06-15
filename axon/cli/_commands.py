"""Subcommands: ``axon status``, ``axon logs``, ``axon cancel``."""

from __future__ import annotations

import json
import os
import signal
import sys
import time
from pathlib import Path

import click

from axon.cli._runs import RUNS_DIR, get_run, list_runs, resolve_run_id

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_or_fail(run_id: str) -> tuple[str, dict]:
    """Resolve a (possibly partial) run ID and return (full_id, meta)."""
    full_id = resolve_run_id(run_id) if len(run_id) < 8 else run_id
    if full_id is None:
        raise click.ClickException(f"No run matching prefix '{run_id}'")
    meta = get_run(full_id)
    if meta is None:
        raise click.ClickException(f"Run '{full_id}' not found")
    return full_id, meta


def _format_time(iso: str | None) -> str:
    if not iso:
        return "—"
    return iso.replace("T", " ")[:16]


_STATUS_COLORS = {"running": "green", "completed": "blue", "crashed": "red", "cancelled": "yellow"}

# ---------------------------------------------------------------------------
# axon status
# ---------------------------------------------------------------------------


@click.command("status")
@click.argument("run_id", default="", required=False)
def status(run_id: str) -> None:
    """Show status of training runs."""
    if run_id:
        full_id, meta = _resolve_or_fail(run_id)
        for key, val in meta.items():
            click.echo(f"  {key:15s} {val}")
        return

    runs = list_runs()
    if not runs:
        click.echo("No runs found.")
        return

    header = f"{'ID':10s} {'STATUS':11s} {'BASE':24s} {'EXPERIMENT':20s} {'STARTED':18s}"
    click.echo(header)
    click.echo("—" * len(header))

    for m in runs:
        base = (m.get("base", "—") or "—")[:24]
        experiment = (m.get("experiment", "—") or "—")[:20]
        status_str = m.get("status", "?")
        styled = click.style(f"{status_str:11s}", fg=_STATUS_COLORS.get(status_str, "white"))
        click.echo(f"{m['run_id']:10s} {styled} {base:24s} {experiment:20s} {_format_time(m.get('started')):18s}")


# ---------------------------------------------------------------------------
# axon logs
# ---------------------------------------------------------------------------


@click.command("logs")
@click.argument("run_id")
@click.option("-f", "--follow", is_flag=True, default=False, help="Follow log output (tail -f).")
@click.option("-n", "--lines", default=50, show_default=True, help="Number of trailing lines to show.")
def logs(run_id: str, follow: bool, lines: int) -> None:
    """View training logs for a run."""
    _, meta = _resolve_or_fail(run_id)

    log_path = Path(meta["log_path"])
    if not log_path.exists():
        raise click.ClickException(f"Log file not found: {log_path}")

    all_lines = log_path.read_text().splitlines()
    for line in all_lines[-lines:]:
        click.echo(line)

    if not follow:
        return

    try:
        with log_path.open() as fh:
            fh.seek(0, 2)
            while True:
                line = fh.readline()
                if line:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                else:
                    time.sleep(0.3)
    except KeyboardInterrupt:
        pass


# ---------------------------------------------------------------------------
# axon cancel
# ---------------------------------------------------------------------------


@click.command("cancel")
@click.argument("run_id")
def cancel(run_id: str) -> None:
    """Cancel a running training job."""
    full_id, meta = _resolve_or_fail(run_id)

    if meta["status"] != "running":
        raise click.ClickException(f"Run {full_id} is not running (status: {meta['status']})")

    pid = meta["pid"]
    if pid <= 0:
        raise click.ClickException(f"Run has not started yet (PID not assigned, got {pid})")
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        click.echo(f"Process {pid} already dead.")
    except PermissionError as err:
        raise click.ClickException(f"Permission denied killing PID {pid}") from err
    else:
        click.echo(f"Sent SIGTERM to PID {pid}.")

    meta_path = RUNS_DIR / full_id / "meta.json"
    meta["status"] = "cancelled"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    click.echo(f"Run {full_id} cancelled.")
