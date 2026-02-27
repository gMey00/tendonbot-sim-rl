"""Shared pytest fixtures and configuration for the tensegrity test suite.

Isaac Sim fixtures are marked with ``requires_isaac`` and skipped automatically
when the simulator is not available (e.g. on a CI runner without a GPU).
"""

from __future__ import annotations

import subprocess
import sys

import pytest

_ISAAC_AVAILABLE: bool | None = None


def _check_isaac_runtime() -> bool:
    """Check whether the Isaac Sim runtime is reachable without blocking."""
    global _ISAAC_AVAILABLE  # noqa: PLW0603
    if _ISAAC_AVAILABLE is not None:
        return _ISAAC_AVAILABLE
    try:
        result = subprocess.run(
            [sys.executable, "-c", "import omni.timeline"],
            capture_output=True,
            timeout=10,
        )
        _ISAAC_AVAILABLE = result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        _ISAAC_AVAILABLE = False
    return _ISAAC_AVAILABLE


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "requires_isaac: test needs Isaac Sim runtime")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip ``requires_isaac`` tests when Isaac Sim runtime is not available."""
    if _check_isaac_runtime():
        return
    skip_marker = pytest.mark.skip(reason="Isaac Sim runtime not available")
    for item in items:
        if "requires_isaac" in item.keywords:
            item.add_marker(skip_marker)
