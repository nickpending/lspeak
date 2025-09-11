"""XDG-compliant directory paths for daemon operations."""

import os
from pathlib import Path


def get_runtime_dir() -> Path:
    """Get XDG-compliant runtime directory for sockets.

    Priority:
    1. $XDG_RUNTIME_DIR/lspeak/ (best - auto-cleaned on logout)
    2. $XDG_CACHE_HOME/lspeak/ (fallback)
    3. ~/.cache/lspeak/ (fallback)
    4. /tmp/lspeak-{uid}/ (last resort with proper permissions)

    Returns:
        Path to runtime directory for daemon operations
    """
    uid = os.getuid()

    # First choice: XDG_RUNTIME_DIR (best for sockets)
    runtime_dir = os.environ.get("XDG_RUNTIME_DIR")
    if runtime_dir:
        path = Path(runtime_dir) / "lspeak"
        path.mkdir(parents=True, exist_ok=True, mode=0o700)
        return path

    # Second choice: XDG_CACHE_HOME
    cache_home = os.environ.get("XDG_CACHE_HOME")
    if cache_home:
        path = Path(cache_home) / "lspeak"
        path.mkdir(parents=True, exist_ok=True, mode=0o700)
        return path

    # Third choice: ~/.cache
    home = Path.home()
    if home.exists():
        path = home / ".cache" / "lspeak"
        path.mkdir(parents=True, exist_ok=True, mode=0o700)
        return path

    # Last resort: /tmp with proper permissions
    path = Path(f"/tmp/lspeak-{uid}")
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    return path


def get_socket_path() -> Path:
    """Get the daemon socket path using XDG directories.

    Returns:
        Path to daemon socket file
    """
    return get_runtime_dir() / "daemon.sock"


def get_config_dir() -> Path:
    """Get XDG-compliant configuration directory.

    Priority:
    1. $XDG_CONFIG_HOME/lspeak/
    2. ~/.config/lspeak/

    Returns:
        Path to configuration directory
    """
    config_home = os.environ.get("XDG_CONFIG_HOME")
    if config_home:
        path = Path(config_home) / "lspeak"
    else:
        path = Path.home() / ".config" / "lspeak"

    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    return path


def get_data_dir() -> Path:
    """Get XDG-compliant data directory.

    Priority:
    1. $XDG_DATA_HOME/lspeak/
    2. ~/.local/share/lspeak/

    Returns:
        Path to data directory
    """
    data_home = os.environ.get("XDG_DATA_HOME")
    if data_home:
        path = Path(data_home) / "lspeak"
    else:
        path = Path.home() / ".local" / "share" / "lspeak"

    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    return path
