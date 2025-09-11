"""Unit tests for daemon spawn logic and validation."""

import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from lspeak.daemon.spawn import check_socket_exists, spawn_daemon


class TestSocketPathLogic:
    """Test socket path generation logic consistency."""

    def test_socket_path_uses_user_id(self) -> None:
        """Test socket path includes current user ID for isolation."""
        # Test check_socket_exists uses correct path format
        expected_path = f"/tmp/lspeak-{os.getuid()}.sock"

        # We can't easily test the internal logic without mocking,
        # but we can verify the pattern is consistent with other components
        from lspeak.daemon.client import DaemonClient
        from lspeak.daemon.server import LspeakDaemon

        client = DaemonClient()
        daemon = LspeakDaemon()

        # All should use same socket path pattern
        assert str(client.socket_path) == expected_path
        assert str(daemon.socket_path) == expected_path

    def test_socket_path_is_absolute(self) -> None:
        """Test socket path is always absolute for reliability."""
        from lspeak.daemon.client import DaemonClient

        client = DaemonClient()
        assert client.socket_path.is_absolute()
        assert str(client.socket_path).startswith("/tmp/lspeak-")


class TestSpawnCommandGeneration:
    """Test spawn_daemon command generation logic."""

    @patch("subprocess.Popen")
    @patch("sys.executable", "/usr/bin/python")
    def test_uses_direct_python_when_not_in_uv(self, mock_popen: MagicMock) -> None:
        """Test uses direct Python when not running under uv."""
        mock_popen.return_value = MagicMock()

        result = spawn_daemon()

        assert result is True
        mock_popen.assert_called_once()
        args, kwargs = mock_popen.call_args

        # Should use direct python command
        command = args[0]
        assert command == ["/usr/bin/python", "-m", "lspeak.daemon"]

    @patch("subprocess.Popen")
    @patch("sys.executable", "/path/to/.venv/bin/python")
    def test_uses_uv_run_when_in_venv(self, mock_popen: MagicMock) -> None:
        """Test uses uv run when running in virtual environment."""
        mock_popen.return_value = MagicMock()

        result = spawn_daemon()

        assert result is True
        mock_popen.assert_called_once()
        args, kwargs = mock_popen.call_args

        # Should use uv run command
        command = args[0]
        assert command == ["uv", "run", "python", "-m", "lspeak.daemon"]

    @patch("subprocess.Popen")
    @patch("sys.executable", "/usr/bin/uv-python")
    def test_uses_uv_run_when_uv_in_executable(self, mock_popen: MagicMock) -> None:
        """Test uses uv run when 'uv' is in sys.executable."""
        mock_popen.return_value = MagicMock()

        result = spawn_daemon()

        assert result is True
        mock_popen.assert_called_once()
        args, kwargs = mock_popen.call_args

        # Should use uv run command
        command = args[0]
        assert command == ["uv", "run", "python", "-m", "lspeak.daemon"]

    @patch("subprocess.Popen")
    def test_subprocess_called_with_correct_detachment_options(
        self, mock_popen: MagicMock
    ) -> None:
        """Test subprocess is called with proper detachment options."""
        mock_popen.return_value = MagicMock()

        result = spawn_daemon()

        assert result is True
        mock_popen.assert_called_once()
        args, kwargs = mock_popen.call_args

        # Verify detachment options
        assert kwargs["stdout"] == __import__("subprocess").DEVNULL
        assert kwargs["stderr"] == __import__("subprocess").DEVNULL
        assert kwargs["stdin"] == __import__("subprocess").DEVNULL
        assert kwargs["start_new_session"] is True

    @patch("subprocess.Popen")
    def test_returns_false_on_subprocess_error(self, mock_popen: MagicMock) -> None:
        """Test returns False when subprocess fails to start."""
        mock_popen.side_effect = OSError("Failed to start process")

        result = spawn_daemon()

        assert result is False
        mock_popen.assert_called_once()

    @patch("subprocess.Popen")
    def test_logs_spawn_command_in_debug(self, mock_popen: MagicMock) -> None:
        """Test logs the spawn command for debugging."""
        mock_popen.return_value = MagicMock()

        with patch("lspeak.daemon.spawn.logger.debug") as mock_debug:
            result = spawn_daemon()

            assert result is True
            # Should log the command being executed
            mock_debug.assert_called()
            debug_calls = [call.args[0] for call in mock_debug.call_args_list]
            assert any("Spawning daemon with command:" in call for call in debug_calls)


class TestParameterValidation:
    """Test parameter handling and validation logic."""

    def test_check_socket_exists_returns_boolean(self) -> None:
        """Test check_socket_exists always returns boolean."""
        # This tests the return type contract
        result = check_socket_exists()
        assert isinstance(result, bool)

    def test_spawn_daemon_returns_boolean(self) -> None:
        """Test spawn_daemon always returns boolean."""
        with patch("subprocess.Popen") as mock_popen:
            mock_popen.return_value = MagicMock()

            result = spawn_daemon()
            assert isinstance(result, bool)

        # Test error case too
        with patch("subprocess.Popen") as mock_popen:
            mock_popen.side_effect = Exception("Test error")

            result = spawn_daemon()
            assert isinstance(result, bool)
            assert result is False
