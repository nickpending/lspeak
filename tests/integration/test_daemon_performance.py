"""Integration tests for daemon performance with pre-loaded models."""

import asyncio
import json
import os
import sys
import time
from pathlib import Path

import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from lspeak.daemon.server import LspeakDaemon


class TestDaemonPerformanceIntegration:
    """Test daemon performance with real pre-loaded models."""

    @pytest.mark.asyncio
    async def test_daemon_uses_preloaded_models_for_fast_response(self) -> None:
        """Test daemon achieves fast response times with pre-loaded models."""
        daemon = LspeakDaemon()

        # Use a test socket path
        test_socket = Path(f"/tmp/test-perf-{os.getpid()}.sock")
        daemon.socket_path = test_socket

        # Ensure socket doesn't exist
        if test_socket.exists():
            test_socket.unlink()

        # Start daemon in background
        daemon_task = asyncio.create_task(daemon.start())

        try:
            # Wait for socket to be created and models loaded
            for _ in range(100):  # 10 seconds max for model loading
                if test_socket.exists() and daemon.models_loaded:
                    break
                await asyncio.sleep(0.1)

            assert test_socket.exists(), "Socket should be created"
            assert daemon.models_loaded is True, "Models should be loaded"
            assert daemon.cache_manager is not None, "Cache manager should exist"

            # First speak request - should use pre-loaded models
            reader1, writer1 = await asyncio.open_unix_connection(str(test_socket))
            start_time = time.time()
            request1 = {
                "method": "speak",
                "id": "perf-test-1",
                "params": {
                    "text": "Performance test one",
                    "provider": "system",
                    "output": "/tmp/perf-test-1.mp3",
                    "cache": True
                }
            }
            writer1.write(json.dumps(request1).encode())
            await writer1.drain()

            response1_data = await reader1.read(65536)
            response1 = json.loads(response1_data.decode())
            first_call_time = time.time() - start_time

            writer1.close()
            await writer1.wait_closed()

            assert response1["status"] == "success"
            assert response1["result"]["saved"] == "/tmp/perf-test-1.mp3"

            # Second speak request - should also be fast
            reader2, writer2 = await asyncio.open_unix_connection(str(test_socket))
            start_time = time.time()
            request2 = {
                "method": "speak",
                "id": "perf-test-2",
                "params": {
                    "text": "Performance test two",
                    "provider": "system",
                    "output": "/tmp/perf-test-2.mp3",
                    "cache": True
                }
            }
            writer2.write(json.dumps(request2).encode())
            await writer2.drain()

            response2_data = await reader2.read(65536)
            response2 = json.loads(response2_data.decode())
            second_call_time = time.time() - start_time

            writer2.close()
            await writer2.wait_closed()

            assert response2["status"] == "success"

            # Both calls should be reasonably fast (models pre-loaded)
            # System TTS adds some overhead, so we allow up to 1 second
            assert first_call_time < 1.5, f"First call too slow: {first_call_time:.2f}s"
            assert second_call_time < 1.0, f"Second call too slow: {second_call_time:.2f}s"

            # Second should be faster than first (cache warmup effect)
            assert second_call_time < first_call_time, "Second call should be faster"

        finally:
            # Clean up daemon
            daemon_task.cancel()
            try:
                await daemon_task
            except asyncio.CancelledError:
                pass

            if test_socket.exists():
                test_socket.unlink()

            # Clean up test files
            for f in ["/tmp/perf-test-1.mp3", "/tmp/perf-test-2.mp3"]:
                if Path(f).exists():
                    Path(f).unlink()

    @pytest.mark.asyncio
    async def test_daemon_cache_hit_performance(self) -> None:
        """Test daemon cache hits are near-instant with pre-loaded models."""
        daemon = LspeakDaemon()

        # Use a test socket path
        test_socket = Path(f"/tmp/test-cache-{os.getpid()}.sock")
        daemon.socket_path = test_socket

        # Ensure socket doesn't exist
        if test_socket.exists():
            test_socket.unlink()

        # Start daemon in background
        daemon_task = asyncio.create_task(daemon.start())

        try:
            # Wait for socket to be created and models loaded
            for _ in range(50):  # 5 seconds max
                if test_socket.exists() and daemon.models_loaded:
                    break
                await asyncio.sleep(0.1)

            assert daemon.models_loaded is True

            # First request - will cache
            reader1, writer1 = await asyncio.open_unix_connection(str(test_socket))
            request1 = {
                "method": "speak",
                "id": "cache-test-1",
                "params": {
                    "text": "Cache this text for testing",
                    "provider": "system",
                    "output": "/tmp/cache-test-1.mp3",
                    "cache": True
                }
            }
            writer1.write(json.dumps(request1).encode())
            await writer1.drain()

            response1_data = await reader1.read(65536)
            response1 = json.loads(response1_data.decode())

            assert response1["status"] == "success"
            # First call shouldn't be cached
            assert response1["result"]["cached"] is False

            writer1.close()
            await writer1.wait_closed()

            # Second request - same text, should hit cache
            reader2, writer2 = await asyncio.open_unix_connection(str(test_socket))
            start_time = time.time()
            request2 = {
                "method": "speak",
                "id": "cache-test-2",
                "params": {
                    "text": "Cache this text for testing",  # Same text
                    "provider": "system",
                    "output": "/tmp/cache-test-2.mp3",
                    "cache": True
                }
            }
            writer2.write(json.dumps(request2).encode())
            await writer2.drain()

            response2_data = await reader2.read(65536)
            response2 = json.loads(response2_data.decode())
            cache_hit_time = time.time() - start_time

            assert response2["status"] == "success"
            # Second call should be cached
            assert response2["result"]["cached"] is True
            # Cache hit should be very fast (just file I/O)
            assert cache_hit_time < 0.2, f"Cache hit too slow: {cache_hit_time:.3f}s"

            # Clean up connection
            writer2.close()
            await writer2.wait_closed()

        finally:
            # Clean up daemon
            daemon_task.cancel()
            try:
                await daemon_task
            except asyncio.CancelledError:
                pass

            if test_socket.exists():
                test_socket.unlink()

            # Clean up test files
            for f in ["/tmp/cache-test-1.mp3", "/tmp/cache-test-2.mp3"]:
                if Path(f).exists():
                    Path(f).unlink()

    @pytest.mark.asyncio
    async def test_daemon_maintains_performance_across_multiple_requests(self) -> None:
        """Test daemon maintains fast response times across many requests."""
        daemon = LspeakDaemon()

        # Use a test socket path
        test_socket = Path(f"/tmp/test-multi-{os.getpid()}.sock")
        daemon.socket_path = test_socket

        # Ensure socket doesn't exist
        if test_socket.exists():
            test_socket.unlink()

        # Start daemon in background
        daemon_task = asyncio.create_task(daemon.start())

        try:
            # Wait for socket to be created and models loaded
            for _ in range(50):  # 5 seconds max
                if test_socket.exists() and daemon.models_loaded:
                    break
                await asyncio.sleep(0.1)

            assert daemon.models_loaded is True

            # Make multiple requests and track times
            request_times = []

            for i in range(5):
                # New connection for each request
                reader, writer = await asyncio.open_unix_connection(str(test_socket))

                start_time = time.time()
                request = {
                    "method": "speak",
                    "id": f"multi-test-{i}",
                    "params": {
                        "text": f"Request number {i}",
                        "provider": "system",
                        "output": f"/tmp/multi-test-{i}.mp3",
                        "cache": True
                    }
                }
                writer.write(json.dumps(request).encode())
                await writer.drain()

                response_data = await reader.read(65536)
                response = json.loads(response_data.decode())
                request_time = time.time() - start_time

                writer.close()
                await writer.wait_closed()

                assert response["status"] == "success"
                request_times.append(request_time)

            # All requests should be fast (models stay loaded)
            for i, req_time in enumerate(request_times):
                assert req_time < 1.5, f"Request {i} too slow: {req_time:.2f}s"

            # Average should be under 1 second
            avg_time = sum(request_times) / len(request_times)
            assert avg_time < 1.0, f"Average time too high: {avg_time:.2f}s"

        finally:
            # Clean up daemon
            daemon_task.cancel()
            try:
                await daemon_task
            except asyncio.CancelledError:
                pass

            if test_socket.exists():
                test_socket.unlink()

            # Clean up test files
            for i in range(5):
                f = f"/tmp/multi-test-{i}.mp3"
                if Path(f).exists():
                    Path(f).unlink()
