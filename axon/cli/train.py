"""``axon train`` — launch Agent PPO training."""

from __future__ import annotations

from typing import Any

import click
from omegaconf import OmegaConf

from axon.cli._hydra_bridge import (
    build_hydra_overrides,
    compose_config,
    flatten_yaml_to_overrides,
)

_BASE_CONFIG = "config"


@click.command("train")
@click.option(
    "--config",
    "-c",
    "config_file",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to a user YAML file with config overrides.",
)
@click.option("--model", "-m", default=None, help="Model path (model_path).")
@click.option("--train-data", default=None, help="Training data path (train_files).")
@click.option("--val-data", default=None, help="Validation data path (val_files).")
@click.option("--gpus", default=None, type=int, help="GPUs per node (num_gpus_per_node).")
@click.option("--nodes", default=None, type=int, help="Number of nodes (num_nodes).")
@click.option("--experiment-name", default=None, help="Experiment name (experiment_name).")
@click.option("--output-dir", default=None, help="Output directory (output_dir).")
@click.option("--resume", default=None, help="Checkpoint path to resume from (resume_from_checkpoint).")
@click.option("--dry-run", is_flag=True, default=False, help="Resolve and print the final config, then exit.")
@click.option(
    "--foreground", "--fg", "foreground", is_flag=True, default=False, help="Run training in the foreground (blocking)."
)
@click.argument("hydra_overrides", nargs=-1, type=click.UNPROCESSED)
def train(
    config_file: str | None,
    dry_run: bool,
    foreground: bool,
    hydra_overrides: tuple[str, ...],
    **cli_flags: Any,
) -> None:
    """Launch Agent PPO training.

    Override precedence (last wins):

        base config → YAML file → CLI flags → raw Hydra overrides (after --)
    """
    base = _BASE_CONFIG
    # 1. YAML file overrides
    overrides: list[str] = []
    if config_file is not None:
        overrides.extend(flatten_yaml_to_overrides(config_file))

    # 2. Explicit CLI flag overrides
    overrides.extend(build_hydra_overrides(**cli_flags))

    # 3. Raw Hydra overrides passed after ``--``
    overrides.extend(hydra_overrides)

    # 4. Compose the final config
    cfg = compose_config(base, overrides)

    # 5. Validate before launching workers so the user gets a fast, in-process error.
    from axon.config import validate_axon_config

    validate_axon_config(OmegaConf.to_container(cfg, resolve=True))

    # 6. Dry-run: print resolved config and exit
    if dry_run:
        click.echo(OmegaConf.to_yaml(cfg, resolve=True))
        raise SystemExit(0)

    import json
    import os

    from axon.cli._runs import create_run, generate_run_id

    run_id = generate_run_id()
    experiment = OmegaConf.select(cfg, "experiment_name", default=None) or ""
    meta_extras = {"base": base, "experiment": experiment}

    # 6a. Foreground mode — run in-process, tee stdout+stderr to log file
    if foreground:
        run_dir = create_run(run_id, pid=os.getpid(), meta_extras=meta_extras)
        OmegaConf.save(cfg, str(run_dir / "config.yaml"))
        log_path = run_dir / "train.log"

        click.echo(f"Run {run_id} (foreground, PID {os.getpid()})")
        click.echo(f"  Log: {log_path}")

        from axon.cli._runs import tee_to_file

        meta_path = run_dir / "meta.json"
        with tee_to_file(log_path):
            try:
                from axon.driver.train_agent_ppo import run_ppo_agent

                run_ppo_agent(cfg)
                meta = json.loads(meta_path.read_text())
                meta["status"] = "completed"
                meta_path.write_text(json.dumps(meta, indent=2) + "\n")
            except BaseException:
                meta = json.loads(meta_path.read_text())
                meta["status"] = "crashed"
                meta_path.write_text(json.dumps(meta, indent=2) + "\n")
                raise
        return

    # 6b. Async mode (default) — spawn detached process and return
    from axon.cli._runs import launch_training

    run_dir = create_run(run_id, pid=-1, status="starting", meta_extras=meta_extras)
    pid = launch_training(cfg, run_id, run_dir)

    # Update meta with real PID and mark as running.
    meta_path = run_dir / "meta.json"
    meta = json.loads(meta_path.read_text())
    meta["pid"] = pid
    meta["status"] = "running"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")

    click.echo(f"Run {run_id} started (PID {pid})")
    click.echo(f"  Log: {run_dir / 'train.log'}")
    click.echo(f"  axon status {run_id[:4]}  |  axon logs -f {run_id[:4]}")
