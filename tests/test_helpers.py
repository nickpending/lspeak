"""Test helpers for consistent test socket paths."""

import os
import tempfile
from pathlib import Path


def get_test_socket_path(test_name: str = "test") -> Path:
    """Get a test-specific socket path in temp directory.

    Args:
        test_name: Name to include in socket path

    Returns:
        Path to test socket file
    """
    # Use system temp dir for tests (isolated from production)
    return Path(tempfile.gettempdir()) / f"lspeak-test-{os.getpid()}-{test_name}.sock"
