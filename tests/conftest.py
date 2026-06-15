"""Root conftest for the axon test suite.

Provides ``@pytest.mark.gpu`` for tests requiring a CUDA GPU.

Usage:
    pytest              # CPU-only tests (default)
    pytest -m gpu       # GPU tests only
    pytest -m all       # everything (CPU + GPU)
"""

import pytest
import torch


def pytest_collection_modifyitems(config, items):
    marker_expr = config.getoption("-m", default="")

    # -m all: run everything, but skip gpu tests if no CUDA
    if marker_expr == "all":
        if not torch.cuda.is_available():
            skip = pytest.mark.skip(reason="No CUDA GPU available")
            for item in items:
                if "gpu" in item.keywords:
                    item.add_marker(skip)
        # Clear the marker expression so pytest doesn't try to filter on "all"
        config.option.markexpr = ""
        return

    # Explicit -m expression (e.g. -m gpu): respect it, auto-skip if no CUDA
    if marker_expr:
        if not torch.cuda.is_available():
            skip = pytest.mark.skip(reason="No CUDA GPU available")
            for item in items:
                if "gpu" in item.keywords:
                    item.add_marker(skip)
        return

    # Default (no -m): exclude gpu tests
    for item in list(items):
        if "gpu" in item.keywords:
            items.remove(item)
