"""Integration tests for daemon control CLI flags with real CLI execution."""

import contextlib
import os
import subprocess
import sys
import time
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Project root for running tests
PROJECT_ROOT = Path(__file__).parent.parent.parent


class TestDaemonControlCLI:
    """Test daemon control CLI flags with real CLI subprocess execution."""

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

    def run_cli_command(self, args: list[str]) -> subprocess.CompletedProcess:
        """Run CLI command and return result."""
        cmd = ["uv", "run", "python", "-m", "lspeak"] + args
        
        return subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            env={"PYTHONPATH": "src", **os.environ},
            capture_output=True,
            text=True,
        )

    def test_daemon_status_flag_shows_help_option(self) -> None:
        """Test that --daemon-status flag appears in help output."""
        result = self.run_cli_command(["--help"])
        
        assert result.returncode == 0
        assert "--daemon-status" in result.stdout
        assert "Show daemon status and exit" in result.stdout

    def test_daemon_stop_flag_shows_help_option(self) -> None:
        """Test that --daemon-stop flag appears in help output."""
        result = self.run_cli_command(["--help"])
        
        assert result.returncode == 0
        assert "--daemon-stop" in result.stdout
        assert "Stop the daemon and exit" in result.stdout

    def test_daemon_restart_flag_shows_help_option(self) -> None:
        """Test that --daemon-restart flag appears in help output."""
        result = self.run_cli_command(["--help"])
        
        assert result.returncode == 0
        assert "--daemon-restart" in result.stdout
        assert "Restart the daemon and exit" in result.stdout

    def test_daemon_status_when_not_running(self) -> None:
        """Test --daemon-status shows not running when daemon stopped."""
        # Ensure daemon is not running
        assert not Path(f"/tmp/lspeak-{os.getuid()}.sock").exists()
        
        result = self.run_cli_command(["--daemon-status"])
        
        assert result.returncode == 0
        assert "✗ Daemon not running" in result.stdout
        assert result.stderr == "" or "pygame" in result.stderr  # pygame warnings ok

    def test_daemon_status_when_running(self) -> None:
        """Test --daemon-status shows running status when daemon active."""
        # Start daemon first using existing mechanism
        start_result = self.run_cli_command(["--daemon-restart"])
        assert start_result.returncode == 0
        assert "Daemon restarted" in start_result.stdout
        
        # Wait a moment for daemon to be fully ready
        time.sleep(0.5)
        
        # Check status
        result = self.run_cli_command(["--daemon-status"])
        
        assert result.returncode == 0
        assert "✓ Daemon running" in result.stdout
        assert "PID:" in result.stdout
        assert "Models:" in result.stdout
        assert result.stderr == "" or "pygame" in result.stderr

    def test_daemon_stop_when_not_running(self) -> None:
        """Test --daemon-stop succeeds even when daemon not running."""
        # Ensure daemon is not running
        assert not Path(f"/tmp/lspeak-{os.getuid()}.sock").exists()
        
        result = self.run_cli_command(["--daemon-stop"])
        
        assert result.returncode == 0
        assert "Daemon stopped" in result.stdout or "Failed to stop daemon (may not be running)" in result.stdout
        assert result.stderr == "" or "pygame" in result.stderr

    def test_daemon_stop_when_running(self) -> None:
        """Test --daemon-stop stops running daemon."""
        # Start daemon first
        start_result = self.run_cli_command(["--daemon-restart"])
        assert start_result.returncode == 0
        
        # Verify daemon is running
        socket_path = Path(f"/tmp/lspeak-{os.getuid()}.sock")
        assert socket_path.exists()
        
        # Stop daemon
        result = self.run_cli_command(["--daemon-stop"])
        
        assert result.returncode == 0
        assert "Daemon stopped" in result.stdout
        assert result.stderr == "" or "pygame" in result.stderr
        
        # Verify daemon is stopped
        time.sleep(0.5)  # Give time for cleanup
        assert not socket_path.exists()

    def test_daemon_restart_from_stopped_state(self) -> None:
        """Test --daemon-restart starts daemon when not running."""
        # Ensure daemon is not running
        assert not Path(f"/tmp/lspeak-{os.getuid()}.sock").exists()
        
        result = self.run_cli_command(["--daemon-restart"])
        
        assert result.returncode == 0
        assert "Daemon restarted" in result.stdout
        assert result.stderr == "" or "pygame" in result.stderr
        
        # Verify daemon is now running
        time.sleep(0.5)
        socket_path = Path(f"/tmp/lspeak-{os.getuid()}.sock")
        assert socket_path.exists()

    def test_daemon_restart_from_running_state(self) -> None:
        """Test --daemon-restart restarts running daemon with new PID."""
        # Start daemon first
        start_result = self.run_cli_command(["--daemon-restart"])
        assert start_result.returncode == 0
        
        # Get initial PID
        status_result = self.run_cli_command(["--daemon-status"])
        assert status_result.returncode == 0
        initial_output = status_result.stdout
        
        # Extract PID from status output (format: "✓ Daemon running (PID: 12345, ...)")
        import re
        pid_match = re.search(r"PID: (\d+)", initial_output)
        assert pid_match
        initial_pid = int(pid_match.group(1))
        
        # Wait to ensure different timestamps
        time.sleep(0.5)
        
        # Restart daemon
        restart_result = self.run_cli_command(["--daemon-restart"])
        assert restart_result.returncode == 0
        assert "Daemon restarted" in restart_result.stdout
        
        # Get new PID
        time.sleep(0.5)
        new_status_result = self.run_cli_command(["--daemon-status"])
        assert new_status_result.returncode == 0
        new_output = new_status_result.stdout
        
        # Extract new PID
        new_pid_match = re.search(r"PID: (\d+)", new_output)
        assert new_pid_match
        new_pid = int(new_pid_match.group(1))
        
        # PIDs should be different
        assert new_pid != initial_pid
        assert "✓ Daemon running" in new_output

    def test_complete_daemon_control_workflow_via_cli(self) -> None:
        """Test complete workflow: status -> restart -> status -> stop -> status via CLI."""
        # 1. Check initial status (should not be running)
        result1 = self.run_cli_command(["--daemon-status"])
        assert result1.returncode == 0
        assert "✗ Daemon not running" in result1.stdout
        
        # 2. Start/restart daemon
        result2 = self.run_cli_command(["--daemon-restart"])
        assert result2.returncode == 0
        assert "Daemon restarted" in result2.stdout
        
        # 3. Check status again (should be running)
        time.sleep(0.5)
        result3 = self.run_cli_command(["--daemon-status"])
        assert result3.returncode == 0
        assert "✓ Daemon running" in result3.stdout
        assert "PID:" in result3.stdout
        
        # 4. Stop daemon
        result4 = self.run_cli_command(["--daemon-stop"])
        assert result4.returncode == 0
        assert "Daemon stopped" in result4.stdout
        
        # 5. Check final status (should not be running)
        time.sleep(0.5)
        result5 = self.run_cli_command(["--daemon-status"])
        assert result5.returncode == 0
        assert "✗ Daemon not running" in result5.stdout

    def test_daemon_control_flags_exit_early(self) -> None:
        """Test that daemon control flags exit without processing text arguments."""
        # Test status flag exits early (doesn't try to process "some text")
        result = self.run_cli_command(["--daemon-status", "some text"])
        
        assert result.returncode == 0
        assert "✗ Daemon not running" in result.stdout or "✓ Daemon running" in result.stdout
        # Should not see any TTS processing messages for "some text"
        assert "some text" not in result.stdout

    def test_daemon_control_flags_with_debug_option(self) -> None:
        """Test that daemon control flags work with debug option."""
        # Test with debug flag
        result = self.run_cli_command(["--debug", "--daemon-status"])
        
        assert result.returncode == 0
        assert "✗ Daemon not running" in result.stdout or "✓ Daemon running" in result.stdout
        # Should succeed with or without debug output
        assert result.stderr == "" or "pygame" in result.stderr or "DEBUG" in result.stderr

    def test_daemon_control_error_handling_with_debug(self) -> None:
        """Test that daemon control errors show debug information when debug flag used."""
        # This test would require a scenario where daemon control fails
        # For now, test that debug mode at least doesn't break the commands
        
        result = self.run_cli_command(["--debug", "--daemon-status"])
        
        # Should succeed regardless of debug mode
        assert result.returncode == 0
        assert "✗ Daemon not running" in result.stdout or "✓ Daemon running" in result.stdout

    def test_daemon_control_flags_are_mutually_exclusive_in_behavior(self) -> None:
        """Test that only the first daemon control flag is processed."""
        # When multiple flags provided, should process them in order defined
        # Since they all exit early, only the first should execute
        
        result = self.run_cli_command(["--daemon-status", "--daemon-stop", "--daemon-restart"])
        
        assert result.returncode == 0
        # Should only show status, not execute stop or restart
        assert "✗ Daemon not running" in result.stdout or "✓ Daemon running" in result.stdout
        assert "Daemon stopped" not in result.stdout
        assert "Daemon restarted" not in result.stdout