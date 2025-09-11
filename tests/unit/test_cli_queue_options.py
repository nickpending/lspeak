"""Unit tests for CLI queue options functionality."""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from lspeak.cli import get_queue_status


@pytest.mark.asyncio
async def test_get_queue_status_returns_daemon_response() -> None:
    """Test that get_queue_status returns the daemon client response."""
    mock_response = {
        "queue_size": 2,
        "current": {"text": "Current message", "id": "123"},
        "waiting": [{"text": "Waiting 1"}, {"text": "Waiting 2"}],
    }

    with patch("lspeak.daemon.client.DaemonClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client.send_request.return_value = mock_response
        mock_client_class.return_value = mock_client

        result = await get_queue_status()

        assert result == mock_response
        mock_client.send_request.assert_called_once_with("queue_status", {})


@pytest.mark.asyncio
async def test_get_queue_status_handles_daemon_error() -> None:
    """Test that get_queue_status returns error dict when daemon not running."""
    error_response = {"status": "error", "error": "Daemon not running"}

    with patch("lspeak.daemon.client.DaemonClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client.send_request.return_value = error_response
        mock_client_class.return_value = mock_client

        result = await get_queue_status()

        assert result == error_response
        assert result["status"] == "error"
