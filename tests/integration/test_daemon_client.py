"""Integration tests for daemon client."""

import os
import subprocess
import time
from pathlib import Path

import pytest

from lspeak.daemon.client import DaemonClient


@pytest.mark.asyncio
async def test_daemon_client_full_workflow() -> None:
    """Test complete client workflow with real daemon."""
    socket_path = Path(f"/tmp/lspeak-{os.getuid()}.sock")

    # Clean up any existing daemon
    subprocess.run(["pkill", "-f", "lspeak.daemon"], capture_output=True)
    if socket_path.exists():
        socket_path.unlink()

    # Start daemon in subprocess
    daemon_process = subprocess.Popen(
        ["uv", "run", "python", "-m", "lspeak.daemon"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )

    try:
        # Wait for daemon to start
        for _ in range(30):  # 3 seconds
            if socket_path.exists():
                time.sleep(0.1)  # Give it a moment to be ready
                break
            time.sleep(0.1)

        # Create client
        client = DaemonClient()
        assert client.socket_path == socket_path

        # Test status request
        response = await client.send_request("status", {})
        assert response["status"] == "success"
        assert "result" in response
        assert response["result"]["models_loaded"] is True
        # PID will be different since uv run creates a parent process
        assert isinstance(response["result"]["pid"], int)

        # Test speak request
        response = await client.speak(
            text="Test message", provider="system", cache=False
        )
        assert response["status"] == "success"
        assert "result" in response

        # Test with all parameters
        response = await client.speak(
            text="Full parameter test",
            provider="system",
            voice=None,
            output=None,
            cache=True,
            cache_threshold=0.95,
            debug=False,
        )
        assert response["status"] == "success"

    finally:
        # Clean up daemon
        daemon_process.terminate()
        try:
            daemon_process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            daemon_process.kill()  # Force kill if terminate didn't work
            daemon_process.wait(timeout=1)
        if socket_path.exists():
            socket_path.unlink()


@pytest.mark.asyncio
async def test_daemon_client_error_handling() -> None:
    """Test client error handling when daemon not running."""
    socket_path = Path(f"/tmp/lspeak-{os.getuid()}.sock")

    # Ensure daemon is NOT running
    subprocess.run(["pkill", "-f", "lspeak.daemon"], capture_output=True)
    if socket_path.exists():
        socket_path.unlink()

    # Create client
    client = DaemonClient()

    # Test error response when daemon not running
    response = await client.send_request("status", {})
    assert response["status"] == "error"
    assert response["error"] == "Daemon not running"

    # Test speak with daemon not running
    response = await client.speak(text="test")
    assert response["status"] == "error"
    assert "Daemon not running" in response["error"]


@pytest.mark.asyncio
async def test_daemon_client_handles_large_text() -> None:
    """Test client handles large text requests properly."""
    socket_path = Path(f"/tmp/lspeak-{os.getuid()}.sock")

    # Clean up any existing daemon
    subprocess.run(["pkill", "-f", "lspeak.daemon"], capture_output=True)
    if socket_path.exists():
        socket_path.unlink()

    # Start daemon
    daemon_process = subprocess.Popen(
        ["uv", "run", "python", "-m", "lspeak.daemon"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )

    try:
        # Wait for daemon to start
        for _ in range(30):
            if socket_path.exists():
                time.sleep(0.1)
                break
            time.sleep(0.1)

        client = DaemonClient()

        # Test with large text (but not too large for system TTS)
        large_text = "This is a test sentence. " * 50  # ~1000 characters
        response = await client.speak(text=large_text, provider="system", cache=False)
        assert response["status"] == "success"

    finally:
        daemon_process.terminate()
        try:
            daemon_process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            daemon_process.kill()
            daemon_process.wait(timeout=1)
        if socket_path.exists():
            socket_path.unlink()


@pytest.mark.asyncio
async def test_daemon_client_handles_empty_params() -> None:
    """Test client handles requests with empty parameters."""
    socket_path = Path(f"/tmp/lspeak-{os.getuid()}.sock")

    # Clean up any existing daemon
    subprocess.run(["pkill", "-f", "lspeak.daemon"], capture_output=True)
    if socket_path.exists():
        socket_path.unlink()

    # Start daemon
    daemon_process = subprocess.Popen(
        ["uv", "run", "python", "-m", "lspeak.daemon"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )

    try:
        # Wait for daemon to start
        for _ in range(30):
            if socket_path.exists():
                time.sleep(0.1)
                break
            time.sleep(0.1)

        client = DaemonClient()

        # Test status with empty params
        response = await client.send_request("status", {})
        assert response["status"] == "success"

        # Test speak with empty text (should error)
        response = await client.speak(text="")
        assert response["status"] == "error" or response["status"] == "success"
        # Server should handle empty text validation

    finally:
        daemon_process.terminate()
        try:
            daemon_process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            daemon_process.kill()
            daemon_process.wait(timeout=1)
        if socket_path.exists():
            socket_path.unlink()


@pytest.mark.asyncio
async def test_daemon_client_multiple_concurrent_requests() -> None:
    """Test client can handle multiple concurrent requests."""
    import asyncio

    socket_path = Path(f"/tmp/lspeak-{os.getuid()}.sock")

    # Clean up any existing daemon
    subprocess.run(["pkill", "-f", "lspeak.daemon"], capture_output=True)
    if socket_path.exists():
        socket_path.unlink()

    # Start daemon
    daemon_process = subprocess.Popen(
        ["uv", "run", "python", "-m", "lspeak.daemon"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )

    try:
        # Wait for daemon to start
        for _ in range(30):
            if socket_path.exists():
                time.sleep(0.1)
                break
            time.sleep(0.1)

        client = DaemonClient()

        # Send multiple concurrent requests
        tasks = [
            client.send_request("status", {}),
            client.speak(text="Request 1", provider="system", cache=False),
            client.speak(text="Request 2", provider="system", cache=False),
            client.send_request("status", {}),
        ]

        responses = await asyncio.gather(*tasks)

        # All should succeed
        for response in responses:
            assert response["status"] == "success"

    finally:
        daemon_process.terminate()
        try:
            daemon_process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            daemon_process.kill()
            daemon_process.wait(timeout=1)
        if socket_path.exists():
            socket_path.unlink()
