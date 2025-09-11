"""Pytest configuration and fixtures for lspeak tests."""

import os
import tempfile
from pathlib import Path
from collections.abc import Generator

import pytest


@pytest.fixture(autouse=True)
def patch_daemon_paths(monkeypatch) -> None:
    """Automatically patch daemon paths for all tests to use test-specific locations."""

    def get_test_socket_path() -> Path:
        """Get test-specific socket path."""
        test_dir = Path(tempfile.gettempdir()) / f"lspeak-test-{os.getpid()}"
        test_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        return test_dir / "daemon.sock"

    def get_test_runtime_dir() -> Path:
        """Get test-specific runtime directory."""
        test_dir = Path(tempfile.gettempdir()) / f"lspeak-test-{os.getpid()}"
        test_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        return test_dir

    # Patch the daemon.paths module functions
    monkeypatch.setattr("lspeak.daemon.paths.get_socket_path", get_test_socket_path)
    monkeypatch.setattr("lspeak.daemon.paths.get_runtime_dir", get_test_runtime_dir)

    # Also patch any direct imports that might have already happened
    import lspeak.daemon.server
    import lspeak.daemon.client
    import lspeak.daemon.spawn
    import lspeak.daemon.control

    # These modules cache the socket path at import time, so we need to reinitialize
    # Actually, they call get_socket_path() at runtime now, so the monkeypatch will work


@pytest.fixture
def clean_daemon() -> Generator[None]:
    """Fixture to ensure daemon is cleaned up before and after tests."""
    import subprocess
    from pathlib import Path

    # Clean up before test
    test_dir = Path(tempfile.gettempdir()) / f"lspeak-test-{os.getpid()}"
    socket_path = test_dir / "daemon.sock"
    subprocess.run(["pkill", "-f", "lspeak.daemon"], capture_output=True)
    if socket_path.exists():
        try:
            socket_path.unlink()
        except OSError:
            pass

    yield

    # Clean up after test
    subprocess.run(["pkill", "-f", "lspeak.daemon"], capture_output=True)
    if socket_path.exists():
        try:
            socket_path.unlink()
        except OSError:
            pass
