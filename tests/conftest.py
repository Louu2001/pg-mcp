"""Pytest configuration and shared fixtures.

This module provides shared fixtures and configuration for all tests.
"""

import os

import pytest

from pg_mcp.config.settings import reset_settings


@pytest.fixture(autouse=True)
def reset_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reset global settings and keep tests off the project .env file."""
    monkeypatch.delenv("DATABASES", raising=False)
    monkeypatch.setattr("pg_mcp.config.settings.load_env_file", lambda: None)
    reset_settings()


@pytest.fixture(autouse=True)
def disable_metrics_for_tests():
    """Disable metrics for tests to avoid port conflicts."""
    os.environ["OBSERVABILITY_METRICS_ENABLED"] = "false"
    yield
    # Clean up
    if "OBSERVABILITY_METRICS_ENABLED" in os.environ:
        del os.environ["OBSERVABILITY_METRICS_ENABLED"]
