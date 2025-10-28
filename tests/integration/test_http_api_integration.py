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

    @pytest.mark.asyncio
    async def test_authentication_required_when_api_key_set(self) -> None:
        """
        INVARIANT: When LSPEAK_API_KEY is set, requests without X-API-Key header return 401
        BREAKS: Unauthorized access to TTS API
        """
        test_port = 17783
        os.environ["LSPEAK_HTTP_PORT"] = str(test_port)
        os.environ["LSPEAK_API_KEY"] = "test-secret-key-123"

        daemon = LspeakDaemon()
        test_id = f"{os.getpid()}-17783"
        test_socket = Path(f"/tmp/test-lspeak-{test_id}.sock")
        test_lock = Path(f"/tmp/test-lspeak-{test_id}.lock")
        daemon.socket_path = test_socket
        daemon.lock_path = test_lock

        if test_socket.exists():
            test_socket.unlink()

        daemon_task = asyncio.create_task(daemon.start())

        try:
            # Wait for HTTP server
            http_ready = False
            for _ in range(200):  # 20 seconds max
                if daemon_task.done():
                    await daemon_task
                try:
                    async with httpx.AsyncClient() as client:
                        response = await client.get(
                            f"http://localhost:{test_port}/status", timeout=1.0
                        )
                        # Will get 401 but server is ready
                        if response.status_code in (200, 401):
                            http_ready = True
                            break
                except (httpx.ConnectError, httpx.TimeoutException):
                    pass
                await asyncio.sleep(0.1)

            assert http_ready, "HTTP server did not start within timeout"

            # Test all endpoints without API key - should return 401
            async with httpx.AsyncClient() as client:
                # Test /speak endpoint
                response = await client.post(
                    f"http://localhost:{test_port}/speak",
                    json={"text": "test", "provider": "system"},
                    timeout=5.0,
                )
                assert response.status_code == 401, (
                    "/speak should return 401 without API key"
                )
                data = response.json()
                assert data["success"] is False
                assert "API key" in data["message"]

                # Test /status endpoint
                response = await client.get(
                    f"http://localhost:{test_port}/status", timeout=5.0
                )
                assert response.status_code == 401, (
                    "/status should return 401 without API key"
                )
                data = response.json()
                assert data["success"] is False
                assert "API key" in data["message"]

                # Test /queue endpoint
                response = await client.get(
                    f"http://localhost:{test_port}/queue", timeout=5.0
                )
                assert response.status_code == 401, (
                    "/queue should return 401 without API key"
                )
                data = response.json()
                assert data["success"] is False
                assert "API key" in data["message"]

        finally:
            daemon_task.cancel()
            try:
                await daemon_task
            except asyncio.CancelledError:
                pass
            if test_socket.exists():
                test_socket.unlink()
            os.environ.pop("LSPEAK_HTTP_PORT", None)
            os.environ.pop("LSPEAK_API_KEY", None)

    @pytest.mark.asyncio
    async def test_authentication_succeeds_with_valid_api_key(self) -> None:
        """
        INVARIANT: Requests with valid X-API-Key header return 200 when auth enabled
        BREAKS: Legitimate clients can't access API
        """
        test_port = 17784
        api_key = "test-valid-key-456"
        os.environ["LSPEAK_HTTP_PORT"] = str(test_port)
        os.environ["LSPEAK_API_KEY"] = api_key

        daemon = LspeakDaemon()
        test_id = f"{os.getpid()}-17784"
        test_socket = Path(f"/tmp/test-lspeak-{test_id}.sock")
        test_lock = Path(f"/tmp/test-lspeak-{test_id}.lock")
        daemon.socket_path = test_socket
        daemon.lock_path = test_lock

        if test_socket.exists():
            test_socket.unlink()

        daemon_task = asyncio.create_task(daemon.start())

        try:
            # Wait for HTTP server
            http_ready = False
            for _ in range(200):
                if daemon_task.done():
                    await daemon_task
                try:
                    async with httpx.AsyncClient() as client:
                        response = await client.get(
                            f"http://localhost:{test_port}/status",
                            headers={"X-API-Key": api_key},
                            timeout=1.0,
                        )
                        if response.status_code == 200:
                            http_ready = True
                            break
                except (httpx.ConnectError, httpx.TimeoutException):
                    pass
                await asyncio.sleep(0.1)

            assert http_ready, "HTTP server did not start within timeout"

            # Test all endpoints with valid API key - should return 200
            async with httpx.AsyncClient() as client:
                # Test /status endpoint
                response = await client.get(
                    f"http://localhost:{test_port}/status",
                    headers={"X-API-Key": api_key},
                    timeout=5.0,
                )
                assert response.status_code == 200, (
                    "/status should return 200 with valid API key"
                )
                data = response.json()
                assert data["success"] is True

                # Test /queue endpoint
                response = await client.get(
                    f"http://localhost:{test_port}/queue",
                    headers={"X-API-Key": api_key},
                    timeout=5.0,
                )
                assert response.status_code == 200, (
                    "/queue should return 200 with valid API key"
                )
                data = response.json()
                assert data["success"] is True

                # Test /speak endpoint (queued to avoid blocking)
                response = await client.post(
                    f"http://localhost:{test_port}/speak",
                    headers={"X-API-Key": api_key},
                    json={"text": "test", "provider": "system", "queue": True},
                    timeout=5.0,
                )
                assert response.status_code == 200, (
                    "/speak should return 200 with valid API key"
                )
                data = response.json()
                assert data["success"] is True

        finally:
            daemon_task.cancel()
            try:
                await daemon_task
            except asyncio.CancelledError:
                pass
            if test_socket.exists():
                test_socket.unlink()
            os.environ.pop("LSPEAK_HTTP_PORT", None)
            os.environ.pop("LSPEAK_API_KEY", None)

    @pytest.mark.asyncio
    async def test_authentication_fails_with_invalid_api_key(self) -> None:
        """
        INVARIANT: Requests with invalid X-API-Key header return 401
        BREAKS: Unauthorized access with wrong key
        """
        test_port = 17785
        os.environ["LSPEAK_HTTP_PORT"] = str(test_port)
        os.environ["LSPEAK_API_KEY"] = "correct-key-789"

        daemon = LspeakDaemon()
        test_id = f"{os.getpid()}-17785"
        test_socket = Path(f"/tmp/test-lspeak-{test_id}.sock")
        test_lock = Path(f"/tmp/test-lspeak-{test_id}.lock")
        daemon.socket_path = test_socket
        daemon.lock_path = test_lock

        if test_socket.exists():
            test_socket.unlink()

        daemon_task = asyncio.create_task(daemon.start())

        try:
            # Wait for HTTP server
            http_ready = False
            for _ in range(200):
                if daemon_task.done():
                    await daemon_task
                try:
                    async with httpx.AsyncClient() as client:
                        response = await client.get(
                            f"http://localhost:{test_port}/status", timeout=1.0
                        )
                        if response.status_code in (200, 401):
                            http_ready = True
                            break
                except (httpx.ConnectError, httpx.TimeoutException):
                    pass
                await asyncio.sleep(0.1)

            assert http_ready, "HTTP server did not start within timeout"

            # Test with wrong API key - should return 401
            async with httpx.AsyncClient() as client:
                wrong_key = "wrong-key-999"
                response = await client.get(
                    f"http://localhost:{test_port}/status",
                    headers={"X-API-Key": wrong_key},
                    timeout=5.0,
                )
                assert response.status_code == 401, (
                    "Should return 401 with invalid API key"
                )
                data = response.json()
                assert data["success"] is False
                assert "Invalid API key" in data["message"]

        finally:
            daemon_task.cancel()
            try:
                await daemon_task
            except asyncio.CancelledError:
                pass
            if test_socket.exists():
                test_socket.unlink()
            os.environ.pop("LSPEAK_HTTP_PORT", None)
            os.environ.pop("LSPEAK_API_KEY", None)

    @pytest.mark.asyncio
    async def test_cors_headers_present_on_localhost_origin(self) -> None:
        """
        INVARIANT: CORS headers allow localhost origins (any port)
        BREAKS: Browser-based clients can't access API
        """
        test_port = 17786
        os.environ["LSPEAK_HTTP_PORT"] = str(test_port)

        daemon = LspeakDaemon()
        test_id = f"{os.getpid()}-17786"
        test_socket = Path(f"/tmp/test-lspeak-{test_id}.sock")
        test_lock = Path(f"/tmp/test-lspeak-{test_id}.lock")
        daemon.socket_path = test_socket
        daemon.lock_path = test_lock

        if test_socket.exists():
            test_socket.unlink()

        daemon_task = asyncio.create_task(daemon.start())

        try:
            # Wait for HTTP server
            http_ready = False
            for _ in range(200):
                if daemon_task.done():
                    await daemon_task
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

            # Test CORS headers with Origin header
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"http://localhost:{test_port}/status",
                    headers={"Origin": "http://localhost:3000"},
                    timeout=5.0,
                )

                assert response.status_code == 200

                # Verify CORS headers present
                assert "access-control-allow-origin" in response.headers, (
                    "CORS headers should be present with Origin header"
                )
                assert "access-control-allow-credentials" in response.headers

                # Verify localhost origin is allowed
                assert (
                    response.headers["access-control-allow-origin"]
                    == "http://localhost:3000"
                )

                # Test OPTIONS preflight request
                response = await client.options(
                    f"http://localhost:{test_port}/speak",
                    headers={
                        "Origin": "http://localhost:8080",
                        "Access-Control-Request-Method": "POST",
                    },
                    timeout=5.0,
                )

                # OPTIONS should return 200 and include CORS headers
                assert response.status_code == 200
                assert "access-control-allow-methods" in response.headers

                # Test private network IPs (RFC1918) are allowed
                private_ips = [
                    "http://192.168.1.100:3000",  # 192.168.0.0/16
                    "http://10.0.0.50:8080",  # 10.0.0.0/8
                    "http://172.16.5.10:5000",  # 172.16.0.0/12
                ]

                for origin in private_ips:
                    response = await client.get(
                        f"http://localhost:{test_port}/status",
                        headers={"Origin": origin},
                        timeout=5.0,
                    )
                    assert response.status_code == 200, f"Failed for origin {origin}"
                    assert "access-control-allow-origin" in response.headers, (
                        f"CORS headers missing for private IP {origin}"
                    )
                    assert response.headers["access-control-allow-origin"] == origin

                # Test that external IPs are rejected (should not have CORS headers)
                response = await client.get(
                    f"http://localhost:{test_port}/status",
                    headers={"Origin": "http://93.184.216.34:3000"},  # example.com IP
                    timeout=5.0,
                )
                # Request succeeds but origin should not be in allowed list
                assert response.status_code == 200
                # CORS middleware returns null for disallowed origins
                assert (
                    "access-control-allow-origin" not in response.headers
                    or response.headers.get("access-control-allow-origin") == "null"
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
