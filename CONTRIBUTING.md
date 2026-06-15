# Contributing

Thanks for helping make Axon better. This project is still moving quickly, so the
best contributions are small, well-scoped, and easy to validate.

## Development Setup

Use the Python version from the installation docs for full-stack work; the
launch recipes are developed against Python 3.10.

```bash
pip install -r install/dependencies/requirements-dev.txt
pre-commit install
```

For full GPU development, follow `docs/getting-started/installation.md`; most
unit tests and docs edits do not need the compiled CUDA stack.

## Before Opening a PR

```bash
ruff check .
ruff format --check .
pytest
mkdocs build --strict
```

`pytest` excludes tests marked `gpu` by default. Run GPU tests explicitly only on
an appropriate machine:

```bash
pytest -m gpu
```

For narrow changes, a focused test command is fine in the PR description. Please
name what you ran and why it covers the changed behavior.

## Pull Request Guidelines

- Keep changes focused on one behavior or documentation area.
- Add or update tests when behavior changes.
- Update docs when user-facing APIs, config keys, recipes, or installation steps
  change.
- Do not include generated datasets, checkpoints, model weights, logs, or W&B output.
- Preserve existing copyright and upstream attribution headers.

## Reporting Issues

Please include:

- The exact command or recipe you ran.
- The model, GPU type/count, backend (`fsdp` or `megatron`), and topology
  (`hybrid_engine` true/false).
- Relevant config overrides and the first useful error stack.
- Whether the issue reproduces with `recipes/frozenlake/`.
