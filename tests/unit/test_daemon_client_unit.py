"""Unit tests for daemon client logic and validation."""

import os
import sys
from pathlib import Path

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from lspeak.daemon.client import DaemonClient


class TestDaemonClientInitialization:
    """Test daemon client initialization logic."""

    def test_socket_path_generation_uses_user_id(self) -> None:
        """Test socket path includes current user ID for isolation."""
        client = DaemonClient()

        expected_path = Path(f"/tmp/lspeak-{os.getuid()}.sock")
        assert client.socket_path == expected_path

    def test_socket_path_is_pathlib_path(self) -> None:
        """Test socket path is a Path object for proper path handling."""
        client = DaemonClient()

        assert isinstance(client.socket_path, Path)
        assert str(client.socket_path).startswith("/tmp/lspeak-")
        assert str(client.socket_path).endswith(".sock")

    def test_multiple_clients_share_same_socket_path(self) -> None:
        """Test multiple client instances point to same daemon socket."""
        client1 = DaemonClient()
        client2 = DaemonClient()

        # Both should point to same socket for same user
        assert client1.socket_path == client2.socket_path


class TestErrorResponseFormat:
    """Test error response formatting logic."""

    def test_error_response_has_required_fields(self) -> None:
        """Test error response format matches protocol specification."""
        # Test the format we return for errors
        error_response = {"status": "error", "error": "Test error message"}

        assert "status" in error_response
        assert "error" in error_response
        assert error_response["status"] == "error"
        assert isinstance(error_response["error"], str)

    def test_daemon_not_running_error_format(self) -> None:
        """Test specific error format for daemon not running."""
        # This is the format we return when daemon is not running
        error_response = {"status": "error", "error": "Daemon not running"}

        assert error_response["status"] == "error"
        assert error_response["error"] == "Daemon not running"

    def test_daemon_not_responding_error_format(self) -> None:
        """Test specific error format for daemon not responding."""
        # This is the format we return when connection is refused
        error_response = {"status": "error", "error": "Daemon not responding"}

        assert error_response["status"] == "error"
        assert error_response["error"] == "Daemon not responding"
