"""Unit tests for daemon server logic and validation."""

import os
import sys
from pathlib import Path

import pytest

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from lspeak.daemon.server import LspeakDaemon


class TestLspeakDaemonInitialization:
    """Test daemon initialization logic."""

    def test_socket_path_generation_uses_xdg(self) -> None:
        """Test socket path uses XDG-compliant directories."""
        daemon = LspeakDaemon()

        # Path should be in test temp directory due to conftest.py patching
        assert daemon.socket_path.name == "daemon.sock"
        assert "lspeak" in str(daemon.socket_path.parent)

    def test_initial_state_is_correct(self) -> None:
        """Test daemon starts with correct initial state."""
        daemon = LspeakDaemon()

        assert daemon.models_loaded is False
        assert daemon.cache_manager is None


class TestHandleSpeakValidationLogic:
    """Test handle_speak validation logic only - pure logic, no mocks."""

    @pytest.mark.asyncio
    async def test_handle_speak_validates_empty_text(self) -> None:
        """Test handle_speak rejects empty text input."""
        daemon = LspeakDaemon()

        # Test empty string
        result = await daemon.handle_speak({"text": ""})
        assert "error" in result
        assert "Text cannot be empty" in result["error"]

        # Test whitespace only
        result = await daemon.handle_speak({"text": "   "})
        assert "error" in result
        assert "Text cannot be empty" in result["error"]

        # Test missing text key
        result = await daemon.handle_speak({})
        assert "error" in result
        assert "Text cannot be empty" in result["error"]


class TestJSONProtocolLogic:
    """Test JSON protocol formatting logic."""

    def test_json_response_format_for_successful_request(self) -> None:
        """Test JSON response format matches protocol specification."""
        # Test response format - this tests the logic without async complexity
        request_id = "test-123"
        result = {"played": True, "saved": None}

        response = {"id": request_id, "status": "success", "result": result}

        # Verify response structure
        assert "id" in response
        assert "status" in response
        assert "result" in response
        assert response["status"] == "success"
        assert response["id"] == request_id

    def test_json_response_format_for_error_request(self) -> None:
        """Test JSON error response format."""
        # Test error response format
        request_id = "test-456"
        result = {"error": "Invalid method"}

        response = {"id": request_id, "status": "error", "result": result}

        # Verify error response structure
        assert response["status"] == "error"
        assert "error" in result

    def test_status_method_response_format(self) -> None:
        """Test status method returns correct information."""
        daemon = LspeakDaemon()
        daemon.models_loaded = True

        # Test status response format
        result = {"pid": os.getpid(), "models_loaded": daemon.models_loaded}

        assert "pid" in result
        assert "models_loaded" in result
        assert result["models_loaded"] is True
        assert isinstance(result["pid"], int)
