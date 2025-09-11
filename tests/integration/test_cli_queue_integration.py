"""Integration tests for CLI queue options with real CLI execution."""

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


class TestCLIQueueOptions:
    """Test CLI queue options with real subprocess execution."""

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
            timeout=10,
        )

    def test_no_queue_flag_shows_in_help(self) -> None:
        """Test that --no-queue flag appears in help output."""
        result = self.run_cli_command(["--help"])

        assert result.returncode == 0
        assert "--no-queue" in result.stdout
        # Check for the help text (may be wrapped)
        assert "Skip queue" in result.stdout and "immediate" in result.stdout

    def test_queue_status_flag_shows_in_help(self) -> None:
        """Test that --queue-status flag appears in help output."""
        result = self.run_cli_command(["--help"])

        assert result.returncode == 0
        assert "--queue-status" in result.stdout
        # Check for the help text (may be wrapped)
        assert "queue status" in result.stdout and "exit" in result.stdout

    def test_queue_status_with_daemon_not_running(self) -> None:
        """Test --queue-status when daemon is not running."""
        # Ensure daemon is not running
        self.cleanup_daemon()

        result = self.run_cli_command(["--queue-status"])

        assert result.returncode == 0
        assert "Daemon not running" in result.stdout

    def test_queue_status_with_daemon_running(self) -> None:
        """Test --queue-status output format when daemon is running."""
        # Start daemon first
        start_result = self.run_cli_command(["--daemon-restart"])
        assert start_result.returncode == 0
        time.sleep(2)  # Give daemon time to start

        # Check queue status
        result = self.run_cli_command(["--queue-status"])

        assert result.returncode == 0
        assert "Queue Status" in result.stdout or "Queue size" in result.stdout
        # Should show queue information
        if "Queue Status" in result.stdout:
            assert "Queue size:" in result.stdout

    def test_no_queue_flag_with_text(self) -> None:
        """Test that --no-queue flag works with text input."""
        # This will use the daemon if running, or fall back to direct execution
        result = self.run_cli_command(["test message", "--no-queue", "--debug"])

        # Should complete without error
        assert result.returncode == 0
        # Debug output should mention queue parameter
        # Either daemon handled it or it fell back to direct execution

    def test_queue_status_exits_immediately(self) -> None:
        """Test that --queue-status exits after showing status."""
        result = self.run_cli_command(["--queue-status", "some text"])

        # Should exit after showing status, not process the text
        assert result.returncode == 0
        # Should show queue status, not try to speak
        assert "Queue" in result.stdout or "Daemon not running" in result.stdout

    def test_combined_daemon_and_queue_flags(self) -> None:
        """Test that daemon flags work alongside queue flags."""
        # Test --daemon-status with other flags in help
        result = self.run_cli_command(["--help"])

        # Should show all daemon and queue control flags
        assert "--daemon-status" in result.stdout
        assert "--daemon-stop" in result.stdout
        assert "--daemon-restart" in result.stdout
        assert "--no-queue" in result.stdout
        assert "--queue-status" in result.stdout
