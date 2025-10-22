"""Integration tests for HTTP API - protecting envelope format invariants."""

import asyncio
import os
import sys
from pathlib import Path

import httpx
import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from lspeak.daemon.server import LspeakDaemon


class TestHTTPAPIIntegration:
    """Test HTTP API with real daemon and network requests."""

    @pytest.mark.asyncio
    async def test_http_server_starts_with_valid_port(self) -> None:
        """
        INVARIANT: HTTP server starts when LSPEAK_HTTP_PORT is set
        BREAKS: Daemon continues but HTTP API unavailable
        """
        # Set HTTP port BEFORE creating daemon (read in __init__)
        test_port = 17777
        os.environ["LSPEAK_HTTP_PORT"] = str(test_port)

        daemon = LspeakDaemon()

        # Use test socket and lock paths
        test_id = f"{os.getpid()}-17777"
        test_socket = Path(f"/tmp/test-lspeak-{test_id}.sock")
        test_lock = Path(f"/tmp/test-lspeak-{test_id}.lock")
        daemon.socket_path = test_socket
        daemon.lock_path = test_lock

        # Ensure socket doesn't exist
        if test_socket.exists():
            test_socket.unlink()

        # Start daemon in background
        daemon_task = asyncio.create_task(daemon.start())

        try:
            # Wait for socket AND HTTP server to be created
            # Daemon needs time to load models (~10s) before HTTP server starts
            http_ready = False
            for _ in range(200):  # 20 seconds max (model loading + HTTP startup)
                # Check if daemon task failed
                if daemon_task.done():
                    await daemon_task  # Re-raise any exception

                if test_socket.exists():
                    # Try to connect to HTTP server
                    try:
                        async with httpx.AsyncClient() as client:
                            response = await client.get(
                                f"http://localhost:{test_port}/status", timeout=1.0
                            )
                            if response.status_code == 200:
                                http_ready = True
                                break
                    except (httpx.ConnectError, httpx.TimeoutException):
                        pass
                await asyncio.sleep(0.1)

            # Verify HTTP server is responding
            assert http_ready, "HTTP server did not start within timeout"

        finally:
            # Clean up
            daemon_task.cancel()
            try:
                await daemon_task
            except asyncio.CancelledError:
                pass

            if test_socket.exists():
                test_socket.unlink()

            # Clean up env var
            os.environ.pop("LSPEAK_HTTP_PORT", None)

    @pytest.mark.asyncio
    async def test_speak_endpoint_queued_response_format(self) -> None:
        """
        INVARIANT: Queued response has {success: true, message: str, data: {queue_id, queue_position, timestamp}}
        BREAKS: Clients can't parse response or track queue position
        """
        test_port = 17778
        os.environ["LSPEAK_HTTP_PORT"] = str(test_port)

        daemon = LspeakDaemon()
        test_id = f"{os.getpid()}-17778"
        test_socket = Path(f"/tmp/test-lspeak-{test_id}.sock")
        test_lock = Path(f"/tmp/test-lspeak-{test_id}.lock")
        daemon.socket_path = test_socket
        daemon.lock_path = test_lock

        if test_socket.exists():
            test_socket.unlink()

        daemon_task = asyncio.create_task(daemon.start())

        try:
            # Wait for HTTP server (model loading + HTTP startup)
            http_ready = False
            for _ in range(200):  # 20 seconds max
                if daemon_task.done():
                    await daemon_task  # Re-raise any exception
                try:
                    async with httpx.AsyncClient() as client:
                        response = await client.get(
                            f"http://localhost:{test_port}/status", timeout=1.0
                        )
                        if response.status_code == 200:
                            http_ready = True
                            break
                except (httpx.ConnectError, httpx.TimeoutException):
                    pass
                await asyncio.sleep(0.1)

            assert http_ready, "HTTP server did not start within timeout"

            # Test queued speak request (queue=true is default)
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"http://localhost:{test_port}/speak",
                    json={
                        "text": "Test queued speech",
                        "provider": "system",
                        "queue": True,
                    },
                    timeout=5.0,
                )

                assert response.status_code == 200
                data = response.json()

                # Verify envelope structure
                assert "success" in data, "Response must have 'success' field"
                assert "message" in data, "Response must have 'message' field"
                assert "data" in data, "Response must have 'data' field"

                # Verify success response
                assert data["success"] is True, "Queued request should succeed"
                assert "queued" in data["message"].lower(), (
                    "Message should mention queuing"
                )

                # Verify queue data structure
                assert data["data"] is not None, "Queued response must have data"
                assert "queue_id" in data["data"], "Must include queue_id"
                assert "queue_position" in data["data"], "Must include queue_position"
                assert "timestamp" in data["data"], "Must include timestamp"

        finally:
            daemon_task.cancel()
            try:
                await daemon_task
            except asyncio.CancelledError:
                pass
            if test_socket.exists():
                test_socket.unlink()
            os.environ.pop("LSPEAK_HTTP_PORT", None)

    @pytest.mark.asyncio
    async def test_speak_endpoint_immediate_response_format(self) -> None:
        """
        INVARIANT: Immediate response has {success: true, message: str, data: {played, saved, cached}}
        BREAKS: Clients can't determine if speech was processed
        """
        test_port = 17779
        os.environ["LSPEAK_HTTP_PORT"] = str(test_port)

        daemon = LspeakDaemon()
        test_id = f"{os.getpid()}-17779"
        test_socket = Path(f"/tmp/test-lspeak-{test_id}.sock")
        test_lock = Path(f"/tmp/test-lspeak-{test_id}.lock")
        daemon.socket_path = test_socket
        daemon.lock_path = test_lock

        if test_socket.exists():
            test_socket.unlink()

        daemon_task = asyncio.create_task(daemon.start())

        try:
            # Wait for HTTP server (model loading + HTTP startup)
            http_ready = False
            for _ in range(200):  # 20 seconds max
                if daemon_task.done():
                    await daemon_task  # Re-raise any exception
                try:
                    async with httpx.AsyncClient() as client:
                        response = await client.get(
                            f"http://localhost:{test_port}/status", timeout=1.0
                        )
                        if response.status_code == 200:
                            http_ready = True
                            break
                except (httpx.ConnectError, httpx.TimeoutException):
                    pass
                await asyncio.sleep(0.1)

            assert http_ready, "HTTP server did not start within timeout"

            # Test immediate speak request (queue=false)
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"http://localhost:{test_port}/speak",
                    json={
                        "text": "Test immediate speech",
                        "provider": "system",
                        "queue": False,
                        "cache": False,
                    },
                    timeout=10.0,
                )

                assert response.status_code == 200
                data = response.json()

                # Verify envelope structure
                assert "success" in data, "Response must have 'success' field"
                assert "message" in data, "Response must have 'message' field"
                assert "data" in data, "Response must have 'data' field"

                # Verify success response
                assert data["success"] is True, "Immediate request should succeed"
                assert "processed" in data["message"].lower(), (
                    "Message should mention processing"
                )

                # Verify processing data structure
                assert data["data"] is not None, "Immediate response must have data"
                assert "played" in data["data"], "Must include 'played' field"
                assert "saved" in data["data"], "Must include 'saved' field"
                assert "cached" in data["data"], "Must include 'cached' field"

        finally:
            daemon_task.cancel()
            try:
                await daemon_task
            except asyncio.CancelledError:
                pass
            if test_socket.exists():
                test_socket.unlink()
            os.environ.pop("LSPEAK_HTTP_PORT", None)

    @pytest.mark.asyncio
    async def test_speak_endpoint_validation_error_format(self) -> None:
        """
        INVARIANT: Validation errors return {success: false, message: str, data: null} with 422
        BREAKS: Clients can't distinguish validation errors from server errors
        """
        test_port = 17780
        os.environ["LSPEAK_HTTP_PORT"] = str(test_port)

        daemon = LspeakDaemon()
        test_id = f"{os.getpid()}-17780"
        test_socket = Path(f"/tmp/test-lspeak-{test_id}.sock")
        test_lock = Path(f"/tmp/test-lspeak-{test_id}.lock")
        daemon.socket_path = test_socket
        daemon.lock_path = test_lock

        if test_socket.exists():
            test_socket.unlink()

        daemon_task = asyncio.create_task(daemon.start())

        try:
            # Wait for HTTP server (model loading + HTTP startup)
            http_ready = False
            for _ in range(200):  # 20 seconds max
                if daemon_task.done():
                    await daemon_task  # Re-raise any exception
                try:
                    async with httpx.AsyncClient() as client:
                        response = await client.get(
                            f"http://localhost:{test_port}/status", timeout=1.0
                        )
                        if response.status_code == 200:
                            http_ready = True
                            break
                except (httpx.ConnectError, httpx.TimeoutException):
                    pass
                await asyncio.sleep(0.1)

            assert http_ready, "HTTP server did not start within timeout"

            # Test validation error (missing required field)
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"http://localhost:{test_port}/speak",
                    json={"provider": "system"},  # Missing 'text' field
                    timeout=5.0,
                )

                # Verify 422 status code
                assert response.status_code == 422, "Validation error should return 422"
                data = response.json()

                # Verify error envelope structure
                assert "success" in data, "Error must have 'success' field"
                assert "message" in data, "Error must have 'message' field"
                assert "data" in data, "Error must have 'data' field"

                # Verify error format
                assert data["success"] is False, (
                    "Validation error must have success=False"
                )
                assert len(data["message"]) > 0, "Must have error message"
                assert data["data"] is None, "Validation error should have data=None"

        finally:
            daemon_task.cancel()
            try:
                await daemon_task
            except asyncio.CancelledError:
                pass
            if test_socket.exists():
                test_socket.unlink()
            os.environ.pop("LSPEAK_HTTP_PORT", None)

    @pytest.mark.asyncio
    async def test_status_endpoint_response_format(self) -> None:
        """
        INVARIANT: Status returns {success: true, message: str, data: {pid, models_loaded, uptime}}
        BREAKS: Monitoring tools can't check daemon health
        """
        test_port = 17781
        os.environ["LSPEAK_HTTP_PORT"] = str(test_port)

        daemon = LspeakDaemon()
        test_id = f"{os.getpid()}-17781"
        test_socket = Path(f"/tmp/test-lspeak-{test_id}.sock")
        test_lock = Path(f"/tmp/test-lspeak-{test_id}.lock")
        daemon.socket_path = test_socket
        daemon.lock_path = test_lock

        if test_socket.exists():
            test_socket.unlink()

        daemon_task = asyncio.create_task(daemon.start())

        try:
            # Wait for HTTP server (model loading + HTTP startup)
            http_ready = False
            for _ in range(200):  # 20 seconds max
                if daemon_task.done():
                    await daemon_task  # Re-raise any exception
                try:
                    async with httpx.AsyncClient() as client:
                        response = await client.get(
                            f"http://localhost:{test_port}/status", timeout=1.0
                        )
                        if response.status_code == 200:
                            http_ready = True
                            break
                except (httpx.ConnectError, httpx.TimeoutException):
                    pass
                await asyncio.sleep(0.1)

            assert http_ready, "HTTP server did not start within timeout"

            # Test status endpoint
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"http://localhost:{test_port}/status", timeout=5.0
                )

                assert response.status_code == 200
                data = response.json()

                # Verify envelope structure
                assert "success" in data, "Response must have 'success' field"
                assert "message" in data, "Response must have 'message' field"
                assert "data" in data, "Response must have 'data' field"

                # Verify success response
                assert data["success"] is True, "Status check should succeed"

                # Verify status data structure
                assert data["data"] is not None, "Status must have data"
                assert "pid" in data["data"], "Must include 'pid'"
                assert "models_loaded" in data["data"], "Must include 'models_loaded'"
                assert "uptime" in data["data"], "Must include 'uptime'"

                # Verify data types
                assert isinstance(data["data"]["pid"], int), "PID must be integer"
                assert isinstance(data["data"]["models_loaded"], bool), (
                    "models_loaded must be boolean"
                )
                assert isinstance(data["data"]["uptime"], (int, float)), (
                    "uptime must be numeric"
                )

        finally:
            daemon_task.cancel()
            try:
                await daemon_task
            except asyncio.CancelledError:
                pass
            if test_socket.exists():
                test_socket.unlink()
            os.environ.pop("LSPEAK_HTTP_PORT", None)

    @pytest.mark.asyncio
    async def test_queue_endpoint_response_format(self) -> None:
        """
        INVARIANT: Queue returns {success: true, message: str, data: {queue_size, current, waiting}}
        BREAKS: Clients can't monitor queue status
        """
        test_port = 17782
        os.environ["LSPEAK_HTTP_PORT"] = str(test_port)

        daemon = LspeakDaemon()
        test_id = f"{os.getpid()}-17782"
        test_socket = Path(f"/tmp/test-lspeak-{test_id}.sock")
        test_lock = Path(f"/tmp/test-lspeak-{test_id}.lock")
        daemon.socket_path = test_socket
        daemon.lock_path = test_lock

        if test_socket.exists():
            test_socket.unlink()

        daemon_task = asyncio.create_task(daemon.start())

        try:
            # Wait for HTTP server (model loading + HTTP startup)
            http_ready = False
            for _ in range(200):  # 20 seconds max
                if daemon_task.done():
                    await daemon_task  # Re-raise any exception
                try:
                    async with httpx.AsyncClient() as client:
                        response = await client.get(
                            f"http://localhost:{test_port}/status", timeout=1.0
                        )
                        if response.status_code == 200:
                            http_ready = True
                            break
                except (httpx.ConnectError, httpx.TimeoutException):
                    pass
                await asyncio.sleep(0.1)

            assert http_ready, "HTTP server did not start within timeout"

            # Test queue endpoint
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"http://localhost:{test_port}/queue", timeout=5.0
                )

                assert response.status_code == 200
                data = response.json()

                # Verify envelope structure
                assert "success" in data, "Response must have 'success' field"
                assert "message" in data, "Response must have 'message' field"
                assert "data" in data, "Response must have 'data' field"

                # Verify success response
                assert data["success"] is True, "Queue status check should succeed"

                # Verify queue data structure
                assert data["data"] is not None, "Queue status must have data"
                assert "queue_size" in data["data"], "Must include 'queue_size'"
                assert "current" in data["data"], "Must include 'current'"
                assert "waiting" in data["data"], "Must include 'waiting'"

                # Verify data types
                assert isinstance(data["data"]["queue_size"], int), (
                    "queue_size must be integer"
                )
                assert isinstance(data["data"]["waiting"], list), "waiting must be list"

        finally:
            daemon_task.cancel()
            try:
                await daemon_task
            except asyncio.CancelledError:
                pass
            if test_socket.exists():
                test_socket.unlink()
            os.environ.pop("LSPEAK_HTTP_PORT", None)
