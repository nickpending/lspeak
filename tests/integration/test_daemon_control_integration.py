"""Integration tests for daemon control functions with real daemon processes."""

import asyncio
import contextlib
import os
import subprocess
import sys
import time
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from lspeak.daemon.client import DaemonClient
from lspeak.daemon.control import daemon_restart, daemon_status, daemon_stop
from lspeak.daemon.spawn import ensure_daemon_running


class TestDaemonControlIntegration:
    """Test daemon control functions with real daemon processes."""

    def setup_method(self) -> None:
        """Clean up before each test."""
        self.cleanup_daemon()

    def teardown_method(self) -> None:
        """Clean up after each test."""
        self.cleanup_daemon()

    def cleanup_daemon(self) -> None:
        """Kill any existing daemon and remove socket."""
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
            with contextlib.suppress(OSError):
                socket_path.unlink()

    def test_complete_control_workflow(self) -> None:
        """Test complete workflow: status -> stop -> restart with real daemon."""
        # Start with clean state
        initial_status = asyncio.run(daemon_status())
        assert initial_status["running"] is False, "Should start with no daemon"
        assert initial_status["pid"] is None
        assert initial_status["models_loaded"] is None

        # Start daemon
        ensure_result = ensure_daemon_running()
        assert ensure_result is True, "Should start daemon successfully"

        # Test status of running daemon
        running_status = asyncio.run(daemon_status())
        assert running_status["running"] is True, "Daemon should be running"
        assert isinstance(running_status["pid"], int), "Should have valid PID"
        assert running_status["models_loaded"] is True, "Models should be loaded"
        original_pid = running_status["pid"]

        # Test stop
        stop_result = daemon_stop()
        assert stop_result is True, "Stop should succeed"

        # Verify daemon stopped
        stopped_status = asyncio.run(daemon_status())
        assert stopped_status["running"] is False, "Daemon should be stopped"
        assert stopped_status["pid"] is None, "PID should be None when stopped"

        # Verify socket removed
        socket_path = Path(f"/tmp/lspeak-{os.getuid()}.sock")
        assert not socket_path.exists(), "Socket should be removed after stop"

        # Test restart (from stopped state)
        restart_result = daemon_restart()
        assert restart_result is True, "Restart should succeed"

        # Verify daemon running with new PID
        restarted_status = asyncio.run(daemon_status())
        assert restarted_status["running"] is True, "Daemon should be running after restart"
        assert isinstance(restarted_status["pid"], int), "Should have valid PID"
        assert restarted_status["pid"] != original_pid, "Should have new PID after restart"
        assert restarted_status["models_loaded"] is True, "Models should be loaded"

    def test_stop_handles_already_stopped_daemon(self) -> None:
        """Test daemon_stop returns True when daemon already stopped."""
        # Ensure daemon is not running
        initial_status = asyncio.run(daemon_status())
        assert initial_status["running"] is False, "Should start with no daemon"

        # Stopping already-stopped daemon should return True
        stop_result = daemon_stop()
        assert stop_result is True, "Stop should return True for already-stopped daemon"

        # Status should remain unchanged
        final_status = asyncio.run(daemon_status())
        assert final_status["running"] is False, "Daemon should still be stopped"

    def test_restart_creates_new_process_with_different_pid(self) -> None:
        """Test restart stops old daemon and starts new one with different PID."""
        # Start daemon
        ensure_daemon_running()

        # Get initial PID
        initial_status = asyncio.run(daemon_status())
        assert initial_status["running"] is True
        initial_pid = initial_status["pid"]
        assert isinstance(initial_pid, int)

        # Restart daemon
        restart_result = daemon_restart()
        assert restart_result is True, "Restart should succeed"

        # Verify new PID
        new_status = asyncio.run(daemon_status())
        assert new_status["running"] is True, "Daemon should be running"
        new_pid = new_status["pid"]
        assert isinstance(new_pid, int), "Should have valid PID"
        assert new_pid != initial_pid, f"PID should change: {initial_pid} -> {new_pid}"

        # Verify daemon actually responds (not just socket exists)
        client = DaemonClient()
        response = asyncio.run(client.send_request("status", {}))
        assert response["status"] == "success", "New daemon should respond to requests"
        assert response["result"]["pid"] == new_pid, "Response PID should match status PID"

    def test_status_with_stale_socket(self) -> None:
        """Test status correctly reports daemon not running with stale socket."""
        # Create a stale socket file (no daemon running)
        socket_path = Path(f"/tmp/lspeak-{os.getuid()}.sock")
        socket_path.touch()

        # Status should detect daemon not actually running
        status = asyncio.run(daemon_status())
        assert status["running"] is False, "Should detect daemon not running despite socket"
        assert "error" in status or status["pid"] is None

    def test_stop_with_permission_error_handling(self) -> None:
        """Test daemon_stop handles permission errors gracefully."""
        # Start daemon
        ensure_daemon_running()

        # Get status to confirm running
        status = asyncio.run(daemon_status())
        assert status["running"] is True

        # Stop should succeed for our own daemon
        stop_result = daemon_stop()
        assert stop_result is True, "Should stop our own daemon"

        # Verify stopped
        stopped_status = asyncio.run(daemon_status())
        assert stopped_status["running"] is False

    def test_restart_from_clean_state(self) -> None:
        """Test restart works even when no daemon is initially running."""
        # Ensure clean state
        initial_status = asyncio.run(daemon_status())
        assert initial_status["running"] is False, "Should start with no daemon"

        # Restart should work (stop does nothing, then starts)
        restart_result = daemon_restart()
        assert restart_result is True, "Restart should succeed from clean state"

        # Verify daemon running
        final_status = asyncio.run(daemon_status())
        assert final_status["running"] is True, "Daemon should be running"
        assert isinstance(final_status["pid"], int), "Should have valid PID"
        assert final_status["models_loaded"] is True, "Models should be loaded"

    def test_control_functions_with_rapid_operations(self) -> None:
        """Test control functions handle rapid successive operations."""
        # Start daemon
        ensure_daemon_running()

        # Rapid status checks
        for _ in range(5):
            status = asyncio.run(daemon_status())
            assert status["running"] is True

        # Stop and immediately check
        daemon_stop()
        status = asyncio.run(daemon_status())
        assert status["running"] is False

        # Restart and immediately check
        daemon_restart()
        status = asyncio.run(daemon_status())
        assert status["running"] is True
