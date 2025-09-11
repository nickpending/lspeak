"""Lock file implementation to prevent multiple daemon spawns."""

import fcntl
import os
from contextlib import contextmanager
from pathlib import Path
from collections.abc import Generator

from lspeak.daemon.paths import get_runtime_dir


def get_lock_path() -> Path:
    """Get path to daemon lock file."""
    return get_runtime_dir() / "daemon.lock"


@contextmanager
def daemon_lock(timeout: float = 1.0) -> Generator[bool]:
    """Context manager for exclusive daemon spawn lock.

    Args:
        timeout: Max time to wait for lock in seconds

    Yields:
        True if lock acquired, False if timeout
    """
    lock_path = get_lock_path()
    lock_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)

    # Open or create lock file
    lock_fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o600)

    try:
        # Try to acquire exclusive lock (non-blocking)
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield True
    except BlockingIOError:
        # Lock is held by another process
        yield False
    finally:
        # Always close the file descriptor
        os.close(lock_fd)
