"""Auto-spawn logic for lspeak daemon."""

import asyncio
import fcntl
import logging
import os
import socket as sock
import subprocess
import sys
import time

from lspeak.daemon.paths import get_runtime_dir, get_socket_path

from .client import DaemonClient

logger = logging.getLogger(__name__)


def check_socket_exists() -> bool:
    """Check if daemon socket file exists.

    Returns:
        True if socket file exists, False otherwise
    """
    socket_path = get_socket_path()
    return socket_path.exists()


async def is_daemon_alive() -> bool:
    """Check if daemon is alive and responding.

    Attempts to send a status request to verify daemon is running.

    Returns:
        True if daemon responds successfully, False otherwise
    """
    if not check_socket_exists():
        return False

    try:
        client = DaemonClient()
        response = await client.send_request("status", {})
        return response.get("status") == "success"
    except Exception as e:
        logger.debug(f"Daemon not responding: {e}")
        return False


def spawn_daemon() -> bool:
    """Spawn daemon process with proper detachment.

    Starts daemon as a background process that survives parent exit.

    Returns:
        True if spawn command executed, False on error
    """
    try:
        # Always use the current Python executable directly
        # This avoids creating unnecessary parent processes
        cmd = [sys.executable, "-m", "lspeak.daemon"]

        logger.debug(f"Spawning daemon with command: {' '.join(cmd)}")

        # Start daemon with proper detachment
        subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            start_new_session=True,  # Detach from parent process group
        )

        return True

    except Exception as e:
        logger.error(f"Failed to spawn daemon: {e}")
        return False


async def ensure_daemon_running_async(force_restart: bool = False) -> bool:
    """Ensure daemon is running, spawn if needed (async version).

    Checks if daemon is alive and starts it if necessary.
    Waits up to 3 seconds for daemon to be ready.

    Args:
        force_restart: If True, restart daemon even if running

    Returns:
        True if daemon is running and ready, False otherwise
    """
    socket_path = get_socket_path()

    # Handle force restart
    if force_restart:
        logger.debug("Force restart requested, removing socket")
        if socket_path.exists():
            try:
                socket_path.unlink()
            except Exception as e:
                logger.debug(f"Failed to remove socket: {e}")

    # Quick check if already running
    if await is_daemon_alive():
        logger.debug("Daemon already running and responsive")
        return True

    # Remove stale socket if exists but daemon is dead
    if socket_path.exists():
        logger.debug("Removing stale socket file")
        try:
            socket_path.unlink()
        except Exception as e:
            logger.debug(f"Failed to remove stale socket: {e}")

    # Spawn daemon
    logger.debug("Starting daemon...")
    if not spawn_daemon():
        return False

    # Wait for daemon to be ready (up to 30 seconds for model loading)
    for i in range(300):  # 300 * 0.1 = 30 seconds
        await asyncio.sleep(0.1)

        # Check if socket appears
        if socket_path.exists():
            # Give it a moment to fully initialize
            await asyncio.sleep(0.1)

            # Verify it's actually responding
            if await is_daemon_alive():
                logger.debug(f"Daemon ready after {(i + 2) * 0.1:.1f} seconds")
                return True

    logger.error("Daemon failed to start within 30 seconds")
    return False


def ensure_daemon_running(force_restart: bool = False) -> bool:
    """Ensure daemon is running, spawn if needed (sync version).

    Checks if daemon is alive and starts it if necessary.
    Waits up to 3 seconds for daemon to be ready.

    Args:
        force_restart: If True, restart daemon even if running

    Returns:
        True if daemon is running and ready, False otherwise
    """
    try:
        # Try to get the current event loop
        loop = asyncio.get_running_loop()
        # If we're in an async context, we can't use asyncio.run
        # So we'll do a sync version
        socket_path = get_socket_path()

        # Handle force restart
        if force_restart:
            logger.debug("Force restart requested, removing socket")
            if socket_path.exists():
                try:
                    socket_path.unlink()
                except Exception as e:
                    logger.debug(f"Failed to remove socket: {e}")

        # Check if daemon is already running by trying to acquire its lock
        lock_path = get_runtime_dir() / "daemon.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)

        try:
            # Try to acquire the daemon lock (non-blocking)
            lock_fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o600)
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                # We got the lock, so no daemon is running - need to spawn
                logger.debug("No daemon lock held, spawning daemon...")
                os.close(lock_fd)  # Release immediately, daemon will acquire its own

                # Spawn daemon process
                if not spawn_daemon():
                    logger.error("Failed to spawn daemon process")
                    return False

                # Wait for daemon to be ready (daemon will acquire its own lock)
                for i in range(100):  # 100 * 0.1 = 10 seconds
                    time.sleep(0.1)

                    if socket_path.exists():
                        time.sleep(0.1)  # Brief pause for socket initialization
                        try:
                            client_socket = sock.socket(sock.AF_UNIX, sock.SOCK_STREAM)
                            client_socket.settimeout(0.5)
                            client_socket.connect(str(socket_path))
                            client_socket.close()
                            logger.debug(
                                f"Daemon ready after {(i + 2) * 0.1:.1f} seconds"
                            )
                            return True
                        except (OSError, ConnectionError, TimeoutError):
                            continue  # Socket not ready yet

                logger.error("Daemon failed to become responsive within 10 seconds")
                return False

            except BlockingIOError:
                # Lock is held by daemon - daemon is running
                os.close(lock_fd)
                logger.debug("Daemon lock held, checking if responsive...")

                # Verify daemon is actually responding
                if socket_path.exists():
                    try:
                        client_socket = sock.socket(sock.AF_UNIX, sock.SOCK_STREAM)
                        client_socket.settimeout(0.5)
                        client_socket.connect(str(socket_path))
                        client_socket.close()
                        logger.debug("Daemon already running and responsive")
                        return True
                    except (OSError, ConnectionError, TimeoutError):
                        logger.warning(
                            "Daemon lock held but socket unresponsive, waiting..."
                        )
                        # Wait a bit for daemon to become ready
                        time.sleep(1.0)
                        return True  # Assume daemon is starting up
                else:
                    logger.debug(
                        "Daemon lock held but no socket yet, daemon starting up"
                    )
                    return True

        except Exception as e:
            logger.error(f"Error checking daemon lock: {e}")
            return False

    except RuntimeError:
        # No running loop, use asyncio.run
        return asyncio.run(ensure_daemon_running_async(force_restart))
