"""Integration tests for daemon server with real Unix socket communication."""

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from lspeak.daemon.server import LspeakDaemon


class TestDaemonServerIntegration:
    """Test daemon server with real Unix socket and services."""

    @pytest.mark.asyncio
    async def test_daemon_startup_and_socket_creation(self) -> None:
        """Test daemon starts up and creates Unix socket."""
        daemon = LspeakDaemon()

        # Use a test socket path
        test_socket = Path(f"/tmp/test-lspeak-{os.getpid()}.sock")
        daemon.socket_path = test_socket

        # Ensure socket doesn't exist
        if test_socket.exists():
            test_socket.unlink()

        # Start daemon in background
        daemon_task = asyncio.create_task(daemon.start())

        try:
            # Wait for socket to be created
            for _ in range(30):  # 3 seconds max
                if test_socket.exists():
                    break
                await asyncio.sleep(0.1)

            # Verify socket was created
            assert test_socket.exists(), "Daemon should create Unix socket"

            # Verify daemon reports models loaded
            assert daemon.models_loaded is True
            assert daemon.cache_manager is not None

        finally:
            # Clean up
            daemon_task.cancel()
            try:
                await daemon_task
            except asyncio.CancelledError:
                pass

            if test_socket.exists():
                test_socket.unlink()

    @pytest.mark.asyncio
    async def test_json_status_request_over_unix_socket(self) -> None:
        """Test complete JSON request/response over Unix socket."""
        daemon = LspeakDaemon()

        # Use a test socket path
        test_socket = Path(f"/tmp/test-lspeak-{os.getpid()}.sock")
        daemon.socket_path = test_socket

        # Ensure socket doesn't exist
        if test_socket.exists():
            test_socket.unlink()

        # Start daemon in background
        daemon_task = asyncio.create_task(daemon.start())

        try:
            # Wait for socket to be created
            for _ in range(30):  # 3 seconds max
                if test_socket.exists():
                    break
                await asyncio.sleep(0.1)

            assert test_socket.exists(), "Socket should be created"

            # Connect to socket and send status request
            reader, writer = await asyncio.open_unix_connection(str(test_socket))

            # Send JSON status request
            request = {"method": "status", "id": "test-status-1"}
            writer.write(json.dumps(request).encode())
            await writer.drain()

            # Read response
            response_data = await reader.read(4096)
            response = json.loads(response_data.decode())

            # Verify response
            assert response["id"] == "test-status-1"
            assert response["status"] == "success"
            assert "result" in response
            assert response["result"]["models_loaded"] is True
            assert response["result"]["pid"] == os.getpid()

            # Close connection
            writer.close()
            await writer.wait_closed()

        finally:
            # Clean up
            daemon_task.cancel()
            try:
                await daemon_task
            except asyncio.CancelledError:
                pass

            if test_socket.exists():
                test_socket.unlink()

    @pytest.mark.asyncio
    async def test_speak_request_with_system_provider(self) -> None:
        """Test speak request using system provider (no external API)."""
        daemon = LspeakDaemon()

        # Use a test socket path
        test_socket = Path(f"/tmp/test-lspeak-{os.getpid()}.sock")
        daemon.socket_path = test_socket

        # Ensure socket doesn't exist
        if test_socket.exists():
            test_socket.unlink()

        # Start daemon in background
        daemon_task = asyncio.create_task(daemon.start())

        try:
            # Wait for socket to be created
            for _ in range(30):  # 3 seconds max
                if test_socket.exists():
                    break
                await asyncio.sleep(0.1)

            assert test_socket.exists(), "Socket should be created"

            # Connect to socket and send speak request
            reader, writer = await asyncio.open_unix_connection(str(test_socket))

            # Use temp file for output
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_path = temp_file.name

            try:
                # Send JSON speak request with system provider
                request = {
                    "method": "speak",
                    "id": "test-speak-1",
                    "params": {
                        "text": "Test daemon integration",
                        "provider": "system",
                        "output": temp_path,
                        "cache": False,
                    },
                }
                writer.write(json.dumps(request).encode())
                await writer.drain()

                # Read response
                response_data = await reader.read(4096)
                response = json.loads(response_data.decode())

                # Verify response
                assert response["id"] == "test-speak-1"
                assert response["status"] == "success"
                assert "result" in response
                assert response["result"]["played"] is False
                assert response["result"]["saved"] == temp_path

                # Verify file was created
                assert Path(temp_path).exists(), "Audio file should be created"
                assert Path(temp_path).stat().st_size > 0, (
                    "Audio file should have content"
                )

            finally:
                # Clean up temp file
                if Path(temp_path).exists():
                    Path(temp_path).unlink()

            # Close connection
            writer.close()
            await writer.wait_closed()

        finally:
            # Clean up
            daemon_task.cancel()
            try:
                await daemon_task
            except asyncio.CancelledError:
                pass

            if test_socket.exists():
                test_socket.unlink()

    @pytest.mark.asyncio
    async def test_invalid_json_error_handling(self) -> None:
        """Test daemon handles invalid JSON gracefully."""
        daemon = LspeakDaemon()

        # Use a test socket path
        test_socket = Path(f"/tmp/test-lspeak-{os.getpid()}.sock")
        daemon.socket_path = test_socket

        # Ensure socket doesn't exist
        if test_socket.exists():
            test_socket.unlink()

        # Start daemon in background
        daemon_task = asyncio.create_task(daemon.start())

        try:
            # Wait for socket to be created
            for _ in range(30):  # 3 seconds max
                if test_socket.exists():
                    break
                await asyncio.sleep(0.1)

            assert test_socket.exists(), "Socket should be created"

            # Connect to socket and send invalid JSON
            reader, writer = await asyncio.open_unix_connection(str(test_socket))

            # Send invalid JSON
            writer.write(b"not valid json{]")
            await writer.drain()

            # Read error response
            response_data = await reader.read(4096)
            response = json.loads(response_data.decode())

            # Verify error response
            assert response["status"] == "error"
            assert "result" in response
            assert "error" in response["result"]
            assert "Invalid JSON" in response["result"]["error"]

            # Close connection
            writer.close()
            await writer.wait_closed()

        finally:
            # Clean up
            daemon_task.cancel()
            try:
                await daemon_task
            except asyncio.CancelledError:
                pass

            if test_socket.exists():
                test_socket.unlink()

    @pytest.mark.asyncio
    async def test_unknown_method_error_handling(self) -> None:
        """Test daemon handles unknown method gracefully."""
        daemon = LspeakDaemon()

        # Use a test socket path
        test_socket = Path(f"/tmp/test-lspeak-{os.getpid()}.sock")
        daemon.socket_path = test_socket

        # Ensure socket doesn't exist
        if test_socket.exists():
            test_socket.unlink()

        # Start daemon in background
        daemon_task = asyncio.create_task(daemon.start())

        try:
            # Wait for socket to be created
            for _ in range(30):  # 3 seconds max
                if test_socket.exists():
                    break
                await asyncio.sleep(0.1)

            assert test_socket.exists(), "Socket should be created"

            # Connect to socket and send unknown method
            reader, writer = await asyncio.open_unix_connection(str(test_socket))

            # Send request with unknown method
            request = {"method": "unknown_method", "id": "test-unknown-1"}
            writer.write(json.dumps(request).encode())
            await writer.drain()

            # Read error response
            response_data = await reader.read(4096)
            response = json.loads(response_data.decode())

            # Verify error response
            assert response["id"] == "test-unknown-1"
            assert response["status"] == "error"
            assert "result" in response
            assert "error" in response["result"]
            assert "Unknown method" in response["result"]["error"]

            # Close connection
            writer.close()
            await writer.wait_closed()

        finally:
            # Clean up
            daemon_task.cancel()
            try:
                await daemon_task
            except asyncio.CancelledError:
                pass

            if test_socket.exists():
                test_socket.unlink()
