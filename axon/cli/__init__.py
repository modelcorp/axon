"""Axon CLI for launching and inspecting program-to-gradient training runs."""

import click

from axon.cli._commands import cancel, logs, status
from axon.cli.train import train


@click.group()
@click.version_option(package_name="axon")
def main() -> None:
    """Axon: the trace is the contract."""


main.add_command(train)
main.add_command(status)
main.add_command(logs)
main.add_command(cancel)
