"""Integration tests for core speak_text daemon functionality with real services."""

import asyncio
import os
import signal
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from lspeak.core import speak_text


class TestSpeakTextDaemonIntegration:
    """Test speak_text daemon integration with real Unix socket and services."""

    @pytest.mark.asyncio
    async def test_speak_text_with_daemon_enabled_default(self) -> None:
        """Test speak_text uses daemon by default for fast response."""
        # Kill any existing daemon to start fresh
        os.system("pkill -f 'lspeak.daemon' 2>/dev/null")
        await asyncio.sleep(0.5)
        
        # First call should spawn daemon and use it
        start_time = asyncio.get_event_loop().time()
        await speak_text(
            "Testing daemon default behavior",
            provider="system",
            debug=True
        )
        first_call_time = asyncio.get_event_loop().time() - start_time
        
        # Second call should be faster (daemon already running)
        start_time = asyncio.get_event_loop().time()
        await speak_text(
            "Second call should be faster",
            provider="system",
            debug=False  # No debug to avoid output noise
        )
        second_call_time = asyncio.get_event_loop().time() - start_time
        
        # Daemon should be running now
        socket_path = Path(f"/tmp/lspeak-{os.getuid()}.sock")
        assert socket_path.exists(), "Daemon socket should exist after speak_text"
        
        # Second call should generally be faster (models already loaded)
        # We can't guarantee timing but socket should exist
        print(f"First call: {first_call_time:.2f}s, Second call: {second_call_time:.2f}s")

    @pytest.mark.asyncio
    async def test_speak_text_with_use_daemon_false_bypasses_daemon(self) -> None:
        """Test speak_text bypasses daemon entirely when use_daemon=False."""
        # Ensure daemon is not running
        os.system("pkill -f 'lspeak.daemon' 2>/dev/null")
        await asyncio.sleep(0.5)
        
        socket_path = Path(f"/tmp/lspeak-{os.getuid()}.sock")
        if socket_path.exists():
            socket_path.unlink()
        
        # Call with use_daemon=False
        await speak_text(
            "Bypass daemon test",
            provider="system",
            use_daemon=False,
            debug=True
        )
        
        # Daemon socket should NOT be created
        assert not socket_path.exists(), "Daemon should not be spawned when use_daemon=False"

    @pytest.mark.asyncio
    async def test_speak_text_falls_back_when_daemon_unavailable(self) -> None:
        """Test speak_text falls back to direct execution when daemon can't start."""
        # Kill any existing daemon
        os.system("pkill -f 'lspeak.daemon' 2>/dev/null")
        await asyncio.sleep(0.5)
        
        # Make socket path unwriteable to prevent daemon startup
        socket_path = Path(f"/tmp/lspeak-{os.getuid()}.sock")
        if socket_path.exists():
            socket_path.unlink()
        
        # Create a directory with the socket name to prevent socket creation
        socket_path.mkdir(exist_ok=True)
        
        try:
            # This should fall back to direct execution
            await speak_text(
                "Fallback test when daemon fails",
                provider="system",
                debug=True
            )
            # Should complete without error via fallback
            
        finally:
            # Clean up the directory
            if socket_path.is_dir():
                socket_path.rmdir()

    @pytest.mark.asyncio
    async def test_speak_text_restarts_daemon_on_error(self) -> None:
        """Test speak_text restarts daemon and retries once on daemon error."""
        # Start with a clean state
        os.system("pkill -f 'lspeak.daemon' 2>/dev/null")
        await asyncio.sleep(0.5)
        
        socket_path = Path(f"/tmp/lspeak-{os.getuid()}.sock")
        if socket_path.exists():
            socket_path.unlink()
        
        # First call to start daemon
        await speak_text(
            "Initial daemon start",
            provider="system",
            debug=False
        )
        
        # Get the daemon PID
        from lspeak.daemon.client import DaemonClient
        client = DaemonClient()
        response = await client.send_request("status", {})
        original_pid = response.get("result", {}).get("pid")
        
        # Force kill the daemon to simulate a crash
        if original_pid:
            try:
                os.kill(original_pid, signal.SIGKILL)
                await asyncio.sleep(0.5)
            except ProcessLookupError:
                pass  # Already dead
        
        # Remove the stale socket
        if socket_path.exists():
            socket_path.unlink()
        
        # This call should detect daemon failure and restart it
        await speak_text(
            "Should restart daemon after crash",
            provider="system",
            debug=True
        )
        
        # Verify daemon is running again with different PID
        response = await client.send_request("status", {})
        new_pid = response.get("result", {}).get("pid")
        
        assert new_pid is not None, "Daemon should be running after restart"
        if original_pid:
            assert new_pid != original_pid, "Daemon should have new PID after restart"

    @pytest.mark.asyncio
    async def test_speak_text_saves_file_through_daemon(self) -> None:
        """Test speak_text can save audio files when using daemon."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "daemon_test.mp3"
            
            # Use daemon to save audio
            await speak_text(
                "Save through daemon",
                provider="system",
                output_file=str(output_path),
                use_daemon=True
            )
            
            # Verify file was created
            assert output_path.exists(), "Audio file should be created via daemon"
            
            # Verify file has content
            audio_data = output_path.read_bytes()
            assert len(audio_data) > 100, "Audio file should have content"

    @pytest.mark.asyncio  
    async def test_speak_text_with_debug_logs_daemon_activity(self, caplog) -> None:
        """Test debug mode logs daemon routing activity."""
        import logging
        
        # Ensure fresh daemon state
        os.system("pkill -f 'lspeak.daemon' 2>/dev/null")
        await asyncio.sleep(0.5)
        
        # Enable debug logging
        with caplog.at_level(logging.DEBUG):
            # Call with debug=True
            await speak_text(
                "Debug message test",
                provider="system",
                debug=True,
                use_daemon=True
            )
        
        # Check for daemon-related debug messages in logs
        debug_messages = [record.message for record in caplog.records if record.levelname == "DEBUG"]
        daemon_messages = [msg for msg in debug_messages if "daemon" in msg.lower()]
        
        assert len(daemon_messages) > 0, (
            f"Debug mode should log daemon activity. Got messages: {debug_messages}"
        )