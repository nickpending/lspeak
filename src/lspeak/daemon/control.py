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
    """Stop daemon gracefully using SIGTERM, with SIGKILL fallback.

    First tries to get daemon PID from socket status.
    If that fails, scans running processes for lspeak.daemon.
    Sends SIGTERM, then SIGKILL if needed.

    Returns:
        True if daemon was stopped successfully, False otherwise
    """
    import subprocess

    daemon_pids = []

    try:
        # Method 1: Get daemon status via socket to find PID
        status = asyncio.run(daemon_status())
        if status["running"] and status["pid"]:
            daemon_pids.append(status["pid"])
            print(f"Found daemon via socket: PID {status['pid']}")
    except Exception:
        # Socket communication failed, continue to process scanning
        pass

    # Method 2: Scan running processes for orphaned lspeak.daemon processes
    try:
        result = subprocess.run(
            ["ps", "aux"], capture_output=True, text=True, check=True
        )
        found_via_ps = []
        for line in result.stdout.splitlines():
            if "lspeak.daemon" in line and "grep" not in line:
                parts = line.split()
                if len(parts) > 1:
                    try:
                        pid = int(parts[1])
                        if pid not in daemon_pids:
                            daemon_pids.append(pid)
                            found_via_ps.append(pid)
                    except ValueError:
                        continue
        if found_via_ps:
            print(f"Found orphaned daemon processes: {found_via_ps}")
    except Exception:
        # Process scanning failed, continue with what we have
        pass

    if not daemon_pids:
        # No daemon processes found
        print("No daemon processes found")
        return True

    print(f"Sending SIGTERM to PIDs: {daemon_pids}")
    # Send SIGTERM to all found daemon processes
    for pid in daemon_pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            # Process already dead
            print(f"PID {pid} already dead")
            continue
        except PermissionError:
            # Not our process or permission denied
            print(f"Permission denied for PID {pid}")
            return False

    # Wait for processes to terminate (give SIGTERM time to work)
    socket_path = get_socket_path()
    print("Waiting for graceful shutdown...")
    for i in range(20):  # Wait up to 2 seconds for graceful shutdown
        still_running = []
        for pid in daemon_pids:
            try:
                os.kill(pid, 0)  # Check if process exists
                still_running.append(pid)
            except ProcessLookupError:
                # Process terminated
                continue

        if not still_running:
            print(f"SIGTERM successful - processes stopped after {(i + 1) * 0.1:.1f}s")
            break
        time.sleep(0.1)
    else:
        # Loop completed without break - processes still running
        print(f"SIGTERM failed - {len(still_running)} processes still running after 2s")

    # If processes still running, force kill with SIGKILL
    killed_pids = []
    for pid in daemon_pids:
        try:
            os.kill(pid, 0)  # Check if still exists
            os.kill(pid, signal.SIGKILL)  # Force kill
            killed_pids.append(pid)
        except ProcessLookupError:
            # Process already dead
            continue
        except PermissionError:
            # Not our process
            print(f"Permission denied for SIGKILL on PID {pid}")
            return False

    if killed_pids:
        print(f"Used SIGKILL on PIDs: {killed_pids}")

    # Clean up stale socket if it exists
    try:
        if socket_path.exists():
            socket_path.unlink()
            print("Cleaned up stale socket")
    except Exception:
        pass

    return True


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
