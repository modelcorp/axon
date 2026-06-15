# Developer Guide

This guide covers development setup, linting, and code quality tools for the Axon codebase.

## Prerequisites

- Python 3.10+
- Git

## Development Setup

### 1. Install Development Dependencies

```bash
pip install -r install/dependencies/requirements-dev.txt
```

This installs:
- `ruff` - Fast Python linter and formatter
- `pre-commit` - Git hook framework
- `pytest` / `pytest-asyncio` - Testing framework
- `mypy` - Static type checker
- `py-spy` - Profiler

### 2. Set Up Pre-commit Hooks

Install the pre-commit hooks to automatically check code on each commit:

```bash
pre-commit install
```

This will run ruff linting and formatting checks automatically before each commit.

---

## Linting & Formatting with Ruff

[Ruff](https://docs.astral.sh/ruff/) is our primary linter and formatter. Configuration is in `pyproject.toml`.

### Running via Pre-commit (Recommended)

**Run all pre-commit hooks:**
```bash
pre-commit run --all-files
```

**Run only ruff linting (with auto-fix):**
```bash
pre-commit run ruff --all-files
```

**Run only ruff formatting:**
```bash
pre-commit run ruff-format --all-files
```

### Running Ruff Directly

**Check and auto-fix linting issues:**
```bash
ruff check --fix .
```

**Format code:**
```bash
ruff format .
```

**Check without fixing (CI mode):**
```bash
ruff check .
ruff format --check .
```

**Check a specific file or directory:**
```bash
ruff check axon/engine/
ruff format axon/engine/
```

### Ruff Configuration

Configuration is in `pyproject.toml`:

```toml
[tool.ruff]
line-length = 120
target-version = "py310"
extend-exclude = [
    "checkpoints",
    "outputs",
    "logs",
    "data",
]

[tool.ruff.lint]
select = ["E", "F", "UP", "B", "I", "G"]  # pycodestyle, pyflakes, pyupgrade, bugbear, isort
```

### Common Ruff Commands

| Command | Description |
|---------|-------------|
| `ruff check .` | Check for linting errors |
| `ruff check --fix .` | Auto-fix linting errors |
| `ruff format .` | Format code |
| `ruff format --check .` | Check formatting without changing files |
| `ruff check --select=I --fix .` | Fix only import sorting |
| `ruff check --show-fixes .` | Show what fixes would be applied |

---

## Type Checking with MyPy

```bash
mypy axon/
```

Configuration is in `pyproject.toml` under `[tool.mypy]`.

---

## Testing

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/engine/test_engine.py

# Run with verbose output
pytest -v

# Run async tests
pytest tests/ -v
```

---

## Pre-commit Hooks Reference

The `.pre-commit-config.yaml` defines these hooks:

| Hook | Description |
|------|-------------|
| `ruff` | Linting with auto-fix |
| `ruff-format` | Code formatting |

### Skipping Pre-commit (Emergency Only)

If you need to bypass pre-commit hooks temporarily:

```bash
git commit --no-verify -m "your message"
```

⚠️ **Warning:** Only use this in emergencies. CI will still run checks.

---

## Troubleshooting

### Pre-commit Not Running

```bash
# Reinstall hooks
pre-commit uninstall
pre-commit install
```

### Ruff Version Mismatch

If you get version conflicts, update ruff:

```bash
pip install --upgrade ruff
```

Or use pre-commit's cached version:

```bash
pre-commit clean
pre-commit run --all-files
```

### Ruff Cache Issues

Clear the ruff cache if you see stale results:

```bash
ruff clean
```

---

## IDE Integration

### VS Code

Install the [Ruff extension](https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff) and add to `.vscode/settings.json`:

```json
{
    "[python]": {
        "editor.defaultFormatter": "charliermarsh.ruff",
        "editor.formatOnSave": true,
        "editor.codeActionsOnSave": {
            "source.fixAll.ruff": "explicit",
            "source.organizeImports.ruff": "explicit"
        }
    },
    "ruff.lineLength": 120
}
```

### PyCharm

Install the [Ruff plugin](https://plugins.jetbrains.com/plugin/20574-ruff) from the marketplace.

---

## Quick Reference

```bash
# One-time setup
pip install -r install/dependencies/requirements-dev.txt
pre-commit install

# Before committing (or let pre-commit do it automatically)
pre-commit run --all-files

# Or run ruff directly
ruff check --fix . && ruff format .
```

