"""Daemon control functions for managing lspeak daemon process."""

import asyncio
import os
import signal
import time
from typing import Any

from lspeak.daemon.paths import get_socket_path

from .client import DaemonClient
from .spawn import ensure_daemon_running


async def daemon_status() -> dict[str, Any]:
    """Get current daemon status.

    Queries the daemon via Unix socket to get status information.

    Returns:
        Dict containing daemon status:
        - running: bool - Whether daemon is running and responding
        - pid: int|None - Process ID if running, None if not
        - models_loaded: bool|None - Whether models are loaded, None if not running
        - socket_path: str - Path to daemon socket
    """
    socket_path = get_socket_path()

    # Check if socket exists first
    if not socket_path.exists():
        return {
            "running": False,
            "pid": None,
            "models_loaded": None,
            "socket_path": str(socket_path),
        }

    try:
        client = DaemonClient()
        response = await client.send_request("status", {})

        if response.get("status") == "success" and "result" in response:
            result = response["result"]
            return {
                "running": True,
                "pid": result.get("pid"),
                "models_loaded": result.get("models_loaded", False),
                "socket_path": str(socket_path),
            }
        else:
            # Daemon responded but with error
            return {
                "running": False,
                "pid": None,
                "models_loaded": None,
                "socket_path": str(socket_path),
                "error": response.get("error", "Unknown daemon error"),
            }

    except Exception as e:
        # Communication failed - socket exists but daemon not responding
        return {
            "running": False,
            "pid": None,
            "models_loaded": None,
            "socket_path": str(socket_path),
            "error": f"Communication failed: {e}",
        }


def daemon_stop() -> bool:
    """Stop daemon gracefully using SIGTERM.

    Gets daemon PID from status and sends SIGTERM signal.
    Waits up to 1 second for socket to disappear.

    Returns:
        True if daemon was stopped successfully, False otherwise
    """
    try:
        # Get daemon status to find PID
        status = asyncio.run(daemon_status())

        if not status["running"] or not status["pid"]:
            # Daemon not running
            return True

        pid = status["pid"]
        socket_path = get_socket_path()

        # Send SIGTERM for graceful shutdown
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            # Process already dead
            pass
        except PermissionError:
            # Not our process or permission denied
            return False

        # Wait for socket to disappear (up to 1 second)
        for _ in range(10):
            if not socket_path.exists():
                return True
            time.sleep(0.1)

        # Socket still exists, try to remove it
        try:
            if socket_path.exists():
                socket_path.unlink()
        except Exception:
            pass

        return True

    except Exception:
        # Unexpected error during stop
        return False


def daemon_restart() -> bool:
    """Restart daemon by stopping then starting.

    Calls daemon_stop() then ensure_daemon_running() with a brief pause.

    Returns:
        True if daemon restarted successfully, False otherwise
    """
    try:
        # Stop existing daemon
        daemon_stop()

        # Brief pause to ensure cleanup
        time.sleep(0.5)

        # Start daemon again
        return ensure_daemon_running()

    except Exception:
        return False
