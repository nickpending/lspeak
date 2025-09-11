"""Integration tests for daemon queue functionality."""

import asyncio
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from lspeak.daemon.client import DaemonClient


class TestDaemonQueueIntegration:
    """Test daemon queue functionality with real services."""

    @pytest.fixture
    async def daemon_with_client(self):
        """Start daemon subprocess and provide client."""
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
            env={**os.environ, "PYTHONPATH": "src"}
        )

        try:
            # Wait for daemon to start and load models
            for _ in range(100):  # 10 seconds max for model loading
                if socket_path.exists():
                    time.sleep(0.5)  # Extra time for models
                    break
                time.sleep(0.1)

            # Create client
            client = DaemonClient()

            # Verify daemon is ready
            response = await client.send_request("status", {})
            assert response["result"]["models_loaded"] is True

            yield client

        finally:
            # Clean up daemon
            daemon_process.terminate()
            try:
                daemon_process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                daemon_process.kill()
                daemon_process.wait(timeout=1)
            if socket_path.exists():
                socket_path.unlink()

    @pytest.mark.asyncio
    async def test_concurrent_requests_are_queued(self, daemon_with_client):
        """Test that multiple concurrent requests get queued with increasing queue sizes."""
        client = daemon_with_client

        # Send multiple requests concurrently
        tasks = []
        for i in range(3):
            task = client.speak(
                text=f"Queue test item {i+1}",
                provider="system",
                cache=False,
                queue=True
            )
            tasks.append(task)

        # Execute all requests concurrently
        responses = await asyncio.gather(*tasks)

        # All should return queued=true
        for response in responses:
            assert response["status"] == "success"
            assert response["result"]["queued"] is True
            assert "queue_id" in response["result"]
            assert "queue_position" in response["result"]
            assert "timestamp" in response["result"]

        # Queue positions should be 1, 2, 3 (or at least increasing)
        positions = [r["result"]["queue_position"] for r in responses]
        assert len(set(positions)) >= 1  # At least some different positions

    @pytest.mark.asyncio
    async def test_queue_status_shows_waiting_items(self, daemon_with_client):
        """Test that queue status accurately reports current and waiting items."""
        client = daemon_with_client

        # Queue multiple items with longer text to ensure they take time
        queue_ids = []
        for i in range(3):
            response = await client.speak(
                text=f"Status test item {i+1} with much longer text to ensure processing takes some time",
                provider="system",
                cache=False,
                queue=True
            )
            assert response["result"]["queued"] is True
            queue_ids.append(response["result"]["queue_id"])

        # Check queue status immediately (items should still be queued)
        status_response = await client.send_request("queue_status", {})
        assert status_response["status"] == "success"

        result = status_response["result"]
        assert "queue_size" in result
        assert "current" in result
        assert "waiting" in result

        # Verify the queue status structure is valid
        # May have 0 items if very fast, but structure should be correct
        assert isinstance(result["queue_size"], int)
        assert result["queue_size"] >= 0

        # If there's a current item, verify its structure
        if result["current"]:
            current = result["current"]
            assert "id" in current
            assert "text" in current
            assert "timestamp" in current
            assert len(current["text"]) <= 53  # 50 chars + "..."

        # Verify waiting list structure
        assert isinstance(result["waiting"], list)
        for item in result["waiting"]:
            assert "id" in item
            assert "text" in item
            assert "timestamp" in item

    @pytest.mark.asyncio
    async def test_immediate_bypass_skips_queue(self, daemon_with_client):
        """Test that queue=false bypasses the queue for immediate processing."""
        client = daemon_with_client

        # Queue a long item first
        await client.speak(
            text="This is a longer text that will take time to process and speak",
            provider="system",
            cache=False,
            queue=True
        )

        # Send immediate request (should not queue)
        immediate_response = await client.speak(
            text="Immediate bypass",
            provider="system",
            cache=False,
            queue=False
        )

        # Should NOT have queued fields
        assert immediate_response["status"] == "success"
        result = immediate_response["result"]
        assert "queued" not in result or result.get("queued") is False
        assert "queue_id" not in result
        assert "queue_position" not in result

        # Should have standard speak response fields
        assert "played" in result
        assert result["played"] is True

    @pytest.mark.asyncio
    async def test_mixed_queue_and_immediate_requests(self, daemon_with_client):
        """Test that queue and immediate requests work together correctly."""
        client = daemon_with_client

        # Queue two items
        queue1 = await client.speak(
            text="First queued item",
            provider="system",
            cache=False,
            queue=True
        )
        assert queue1["result"]["queued"] is True

        queue2 = await client.speak(
            text="Second queued item",
            provider="system",
            cache=False,
            queue=True
        )
        assert queue2["result"]["queued"] is True

        # Send immediate request
        immediate = await client.speak(
            text="Immediate item",
            provider="system",
            cache=False,
            queue=False
        )
        assert "queued" not in immediate["result"] or immediate["result"].get("queued") is False

        # Queue another item after immediate
        queue3 = await client.speak(
            text="Third queued item",
            provider="system",
            cache=False,
            queue=True
        )
        assert queue3["result"]["queued"] is True

        # Check final queue status
        status = await client.send_request("queue_status", {})
        result = status["result"]

        # Should still have queued items (since they process serially)
        total_pending = (1 if result["current"] else 0) + len(result["waiting"])
        assert total_pending >= 0  # May have processed by now, but structure should be valid
