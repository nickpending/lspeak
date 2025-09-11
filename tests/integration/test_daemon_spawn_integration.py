"""Integration tests for daemon spawn functionality with real processes."""

import asyncio
import os
import sys
import time
from pathlib import Path

import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from lspeak.daemon.spawn import (
    check_socket_exists,
    is_daemon_alive,
    spawn_daemon,
    ensure_daemon_running,
)
from lspeak.daemon.client import DaemonClient


class TestDaemonSpawnIntegration:
    """Test complete daemon spawning workflow with real processes."""

    def setup_method(self) -> None:
        """Clean up before each test."""
        self.cleanup_daemon()

    def teardown_method(self) -> None:
        """Clean up after each test."""
        self.cleanup_daemon()

    def cleanup_daemon(self) -> None:
        """Kill any existing daemon and remove socket."""
        import subprocess

        # Kill any existing daemon processes
        try:
            subprocess.run(
                ["pkill", "-f", "lspeak.daemon"], capture_output=True, timeout=5
            )
            time.sleep(0.5)  # Give processes time to die
        except (subprocess.SubprocessError, subprocess.TimeoutExpired):
            pass

        # Remove socket file
        socket_path = Path(f"/tmp/lspeak-{os.getuid()}.sock")
        if socket_path.exists():
            try:
                socket_path.unlink()
            except OSError:
                pass

    def test_complete_auto_spawn_workflow(self) -> None:
        """Test complete auto-spawn workflow from clean state."""
        # Verify clean initial state
        assert check_socket_exists() is False
        assert asyncio.run(is_daemon_alive()) is False

        # Auto-spawn daemon
        result = ensure_daemon_running()
        assert result is True, "ensure_daemon_running should succeed"

        # Verify daemon is running
        assert check_socket_exists() is True, "Socket should exist after spawn"
        assert asyncio.run(is_daemon_alive()) is True, "Daemon should be alive"

        # Verify daemon actually responds
        client = DaemonClient()
        response = asyncio.run(client.send_request("status", {}))
        assert response["status"] == "success"
        assert response["result"]["models_loaded"] is True
        assert isinstance(response["result"]["pid"], int)

    def test_ensure_daemon_running_is_idempotent(self) -> None:
        """Test subsequent calls don't spawn duplicate daemons."""
        # First call - spawn daemon
        start_time = time.time()
        result1 = ensure_daemon_running()
        first_call_time = time.time() - start_time

        assert result1 is True
        assert check_socket_exists() is True

        # Get initial daemon PID
        client = DaemonClient()
        response1 = asyncio.run(client.send_request("status", {}))
        original_pid = response1["result"]["pid"]

        # Second call - should be fast and not spawn new daemon
        start_time = time.time()
        result2 = ensure_daemon_running()
        second_call_time = time.time() - start_time

        assert result2 is True
        assert second_call_time < first_call_time, "Second call should be faster"

        # Verify same daemon is running
        response2 = asyncio.run(client.send_request("status", {}))
        assert response2["result"]["pid"] == original_pid, "Should be same daemon"

    def test_force_restart_functionality(self) -> None:
        """Test force_restart parameter restarts existing daemon."""
        # Start initial daemon
        result1 = ensure_daemon_running()
        assert result1 is True

        # Get initial PID
        client = DaemonClient()
        response1 = asyncio.run(client.send_request("status", {}))
        original_pid = response1["result"]["pid"]

        # Force restart
        result2 = ensure_daemon_running(force_restart=True)
        assert result2 is True

        # Verify new daemon with different PID
        response2 = asyncio.run(client.send_request("status", {}))
        new_pid = response2["result"]["pid"]

        assert new_pid != original_pid, "Should have new daemon process"
        assert response2["result"]["models_loaded"] is True

    def test_stale_socket_cleanup(self) -> None:
        """Test cleanup of stale socket when daemon is dead."""
        # Create a stale socket file (no daemon behind it)
        socket_path = Path(f"/tmp/lspeak-{os.getuid()}.sock")
        socket_path.touch()  # Create empty socket file

        assert check_socket_exists() is True
        assert asyncio.run(is_daemon_alive()) is False

        # ensure_daemon_running should clean up stale socket and start new daemon
        result = ensure_daemon_running()
        assert result is True

        # Verify new daemon is actually running
        assert check_socket_exists() is True
        assert asyncio.run(is_daemon_alive()) is True

        # Verify daemon responds properly
        client = DaemonClient()
        response = asyncio.run(client.send_request("status", {}))
        assert response["status"] == "success"

    @pytest.mark.asyncio
    async def test_is_daemon_alive_with_real_daemon(self) -> None:
        """Test is_daemon_alive with actual daemon process."""
        # Start daemon
        assert ensure_daemon_running() is True

        # Test daemon is alive
        result = await is_daemon_alive()
        assert result is True

        # Kill daemon process
        self.cleanup_daemon()

        # Test daemon is dead
        result = await is_daemon_alive()
        assert result is False

    def test_spawn_daemon_creates_detached_process(self) -> None:
        """Test spawn_daemon creates properly detached process."""
        # Spawn daemon
        result = spawn_daemon()
        assert result is True

        # Wait for daemon to start up
        for _ in range(30):  # 3 seconds max
            if check_socket_exists():
                break
            time.sleep(0.1)

        assert check_socket_exists() is True, "Socket should be created"

        # Verify daemon is actually running and responsive
        assert asyncio.run(is_daemon_alive()) is True

        # Verify daemon process exists
        import subprocess

        result = subprocess.run(
            ["pgrep", "-f", "lspeak.daemon"], capture_output=True, text=True
        )
        assert result.returncode == 0, "Daemon process should exist"
        assert result.stdout.strip(), "Should find daemon PID"

    def test_daemon_survives_parent_process_exit(self) -> None:
        """Test daemon continues running after spawning process exits."""
        import subprocess

        # Spawn daemon in a subprocess that exits immediately
        spawn_script = """
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from lspeak.daemon.spawn import spawn_daemon
spawn_daemon()
"""

        result = subprocess.run(
            ["uv", "run", "python", "-c", spawn_script],
            cwd=Path(__file__).parent.parent.parent,
            capture_output=True,
            timeout=10,
        )

        assert result.returncode == 0, "Spawn script should succeed"

        # Wait for daemon to start
        for _ in range(30):  # 3 seconds max
            if check_socket_exists():
                break
            time.sleep(0.1)

        # Verify daemon is still running after parent process exited
        assert check_socket_exists() is True
        assert asyncio.run(is_daemon_alive()) is True

        # Verify daemon responds
        client = DaemonClient()
        response = asyncio.run(client.send_request("status", {}))
        assert response["status"] == "success"

    def test_timeout_behavior_when_daemon_fails_to_start(self) -> None:
        """Test ensure_daemon_running timeout when daemon fails to start."""
        # Mock spawn_daemon to fail
        import unittest.mock

        with unittest.mock.patch("lspeak.daemon.spawn.spawn_daemon") as mock_spawn:
            mock_spawn.return_value = False  # Simulate spawn failure

            result = ensure_daemon_running()
            assert result is False

            mock_spawn.assert_called_once()

    def test_sequential_ensure_daemon_running_calls(self) -> None:
        """Test multiple sequential ensure_daemon_running calls work correctly."""
        # First call - should start daemon
        result1 = ensure_daemon_running()
        assert result1 is True

        # Get initial daemon info
        client = DaemonClient()
        response1 = asyncio.run(client.send_request("status", {}))
        original_pid = response1["result"]["pid"]

        # Second call - should be idempotent
        result2 = ensure_daemon_running()
        assert result2 is True

        # Third call - should still be idempotent
        result3 = ensure_daemon_running()
        assert result3 is True

        # Verify same daemon throughout
        response2 = asyncio.run(client.send_request("status", {}))
        assert response2["result"]["pid"] == original_pid
        assert response2["status"] == "success"
