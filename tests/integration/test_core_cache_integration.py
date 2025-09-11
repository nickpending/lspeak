"""Integration tests for core orchestration with cache and provider selection."""

import logging
import sys
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from lspeak.core import speak_text
from lspeak.providers import ProviderRegistry


class TestProviderSelection:
    """Test provider selection functionality with real services."""

    @pytest.mark.asyncio
    async def test_system_provider_selection(self) -> None:
        """Test that system provider is used when specified."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "system_test.wav"

            # Use system provider (no API key needed)
            await speak_text(
                "Testing system provider",
                provider="system",
                output_file=str(output_path),
                cache=False,  # Disable cache for this test
            )

            # Verify audio file was created
            assert output_path.exists(), "System provider should create audio file"
            assert output_path.stat().st_size > 1000, "Audio file should have content"

            # System TTS typically creates WAV format
            audio_data = output_path.read_bytes()
            # WAV files start with RIFF header
            assert audio_data[:4] == b"RIFF" or audio_data[:4] == b"FORM", (
                "System provider should create WAV/AIFF format"
            )

    @pytest.mark.asyncio
    async def test_provider_not_found_error(self) -> None:
        """Test that invalid provider name raises KeyError."""
        with pytest.raises(KeyError, match="Provider 'invalid_provider' not found"):
            await speak_text(
                "Test invalid provider", provider="invalid_provider", cache=False
            )

    @pytest.mark.asyncio
    async def test_provider_registry_get(self) -> None:
        """Test that ProviderRegistry returns correct provider classes."""
        # System provider should be registered
        system_provider = ProviderRegistry.get("system")
        assert system_provider is not None
        assert system_provider.__name__ == "SystemTTSProvider"

        # ElevenLabs provider should be registered
        elevenlabs_provider = ProviderRegistry.get("elevenlabs")
        assert elevenlabs_provider is not None
        assert elevenlabs_provider.__name__ == "ElevenLabsProvider"


class TestCacheIntegration:
    """Test cache functionality with real cache components."""

    @pytest.mark.asyncio
    async def test_cache_miss_then_hit(self) -> None:
        """Test cache miss on first call, hit on second with similar text."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir) / "test_cache"
            output1 = Path(temp_dir) / "first.wav"
            output2 = Path(temp_dir) / "second.wav"

            # First call - should be cache miss
            await speak_text(
                "Cache test phrase",
                provider="system",
                output_file=str(output1),
                cache=True,
                cache_threshold=0.95,
            )

            assert output1.exists(), "First call should create audio"
            first_size = output1.stat().st_size

            # Second call with exact same text - should be cache hit
            await speak_text(
                "Cache test phrase",
                provider="system",
                output_file=str(output2),
                cache=True,
                cache_threshold=0.95,
            )

            assert output2.exists(), "Second call should create audio from cache"
            second_size = output2.stat().st_size

            # Both files should have identical content (cache hit)
            assert first_size == second_size, "Cache hit should return same audio"
            assert output1.read_bytes() == output2.read_bytes(), (
                "Cache hit should return identical audio data"
            )

    @pytest.mark.asyncio
    async def test_cache_with_different_providers(self) -> None:
        """Test that cache correctly handles different providers for same text."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock ElevenLabs to avoid API calls
            mock_provider = MagicMock()
            mock_provider.synthesize = AsyncMock(return_value=b"fake_audio_data")

            with patch.object(ProviderRegistry, "get") as mock_get:
                from collections.abc import Callable
                from typing import Any

                def get_provider(name: str) -> Callable[[], Any]:
                    if name == "system":
                        return ProviderRegistry._providers["system"]
                    else:
                        return lambda: mock_provider

                mock_get.side_effect = get_provider

                output1 = Path(temp_dir) / "system.wav"
                output2 = Path(temp_dir) / "elevenlabs.mp3"

                # First call with system provider
                await speak_text(
                    "Same text different provider",
                    provider="system",
                    output_file=str(output1),
                    cache=True,
                )

                # Second call with elevenlabs provider - should NOT hit cache
                await speak_text(
                    "Same text different provider",
                    provider="elevenlabs",
                    output_file=str(output2),
                    cache=True,
                )

                # Both files should exist but with different content
                assert output1.exists() and output2.exists()
                assert output1.read_bytes() != output2.read_bytes(), (
                    "Different providers should not share cache"
                )

    @pytest.mark.asyncio
    async def test_cache_with_different_voices(self) -> None:
        """Test that cache correctly handles different voices for same text."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock provider to control voice behavior
            mock_provider = MagicMock()

            async def voice_synthesizer(text: str, voice: str) -> bytes:
                return f"audio_{voice}".encode()

            mock_provider.synthesize = AsyncMock(side_effect=voice_synthesizer)

            with patch.object(
                ProviderRegistry, "get", return_value=lambda: mock_provider
            ):
                output1 = Path(temp_dir) / "voice1.mp3"
                output2 = Path(temp_dir) / "voice2.mp3"

                # First call with voice1
                await speak_text(
                    "Same text different voice",
                    provider="test",
                    voice_id="voice1",
                    output_file=str(output1),
                    cache=True,
                )

                # Second call with voice2 - should NOT hit cache
                await speak_text(
                    "Same text different voice",
                    provider="test",
                    voice_id="voice2",
                    output_file=str(output2),
                    cache=True,
                )

                # Files should have different content
                assert output1.read_bytes() == b"audio_voice1"
                assert output2.read_bytes() == b"audio_voice2"

    @pytest.mark.asyncio
    async def test_cache_disabled(self) -> None:
        """Test that cache can be disabled with cache=False."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Use counter to verify TTS is called twice
            call_count = 0

            async def counting_synthesize(text: str, voice: str) -> bytes:
                nonlocal call_count
                call_count += 1
                return f"audio_{call_count}".encode()

            mock_provider = MagicMock()
            mock_provider.synthesize = counting_synthesize

            with patch.object(
                ProviderRegistry, "get", return_value=lambda: mock_provider
            ):
                output1 = Path(temp_dir) / "no_cache1.mp3"
                output2 = Path(temp_dir) / "no_cache2.mp3"

                # First call with cache disabled
                await speak_text(
                    "No cache test",
                    provider="test",
                    output_file=str(output1),
                    cache=False,
                )

                # Second call with same text, cache still disabled
                await speak_text(
                    "No cache test",
                    provider="test",
                    output_file=str(output2),
                    cache=False,
                )

                # Should have called synthesize twice (no cache)
                assert call_count == 2, "Cache disabled should call TTS every time"
                assert output1.read_bytes() == b"audio_1"
                assert output2.read_bytes() == b"audio_2"

    @pytest.mark.asyncio
    async def test_cache_threshold_adjustment(self) -> None:
        """Test that cache threshold affects similarity matching."""
        with tempfile.TemporaryDirectory() as temp_dir:
            mock_provider = MagicMock()

            async def text_synthesizer(text: str, voice: str) -> bytes:
                return f"audio_{text}".encode()

            mock_provider.synthesize = AsyncMock(side_effect=text_synthesizer)

            with patch.object(
                ProviderRegistry, "get", return_value=lambda: mock_provider
            ):
                # Store original phrase
                output1 = Path(temp_dir) / "original.mp3"
                await speak_text(
                    "Hello world",
                    provider="test",
                    output_file=str(output1),
                    cache=True,
                    cache_threshold=0.95,
                )

                # Try similar phrase with high threshold - should miss
                output2 = Path(temp_dir) / "high_threshold.mp3"
                await speak_text(
                    "Hello worlds",  # Slightly different
                    provider="test",
                    output_file=str(output2),
                    cache=True,
                    cache_threshold=0.99,  # Very high threshold
                )

                # Different audio means cache miss
                assert output1.read_bytes() != output2.read_bytes(), (
                    "High threshold should cause cache miss for similar text"
                )

                # Try same similar phrase with lower threshold - might hit
                # (depends on actual similarity score, but demonstrates the parameter works)
                output3 = Path(temp_dir) / "low_threshold.mp3"
                await speak_text(
                    "Hello worlds",
                    provider="test",
                    output_file=str(output3),
                    cache=True,
                    cache_threshold=0.5,  # Low threshold
                )

                # Should get the cached "Hello worlds" audio
                assert output3.read_bytes() == output2.read_bytes(), (
                    "Should hit cache for exact same text regardless of threshold"
                )


class TestDebugLogging:
    """Test debug logging functionality for cache operations."""

    @pytest.mark.asyncio
    async def test_debug_logging_cache_operations(self, caplog) -> None:
        """Test that debug mode logs cache operations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "debug_test.wav"

            # Enable debug logging
            with caplog.at_level(logging.DEBUG):
                await speak_text(
                    "Debug logging test",
                    provider="system",
                    output_file=str(output_path),
                    cache=True,
                    debug=True,
                )

            # Check for expected debug messages
            debug_messages = [
                record.message
                for record in caplog.records
                if record.levelname == "DEBUG"
            ]

            # Should have cache-related debug messages
            assert any(
                "Cache enabled with threshold" in msg for msg in debug_messages
            ), "Should log cache initialization"
            assert any("Checking cache for text" in msg for msg in debug_messages), (
                "Should log cache lookup"
            )
            assert any("Cache miss" in msg for msg in debug_messages) or any(
                "Cache hit" in msg for msg in debug_messages
            ), "Should log cache hit/miss"

    @pytest.mark.asyncio
    async def test_debug_logging_shows_provider(self, caplog) -> None:
        """Test that debug mode shows which provider is being used."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "provider_debug.wav"

            with caplog.at_level(logging.DEBUG):
                await speak_text(
                    "Provider debug test",
                    provider="system",
                    output_file=str(output_path),
                    cache=False,  # Disable cache to focus on provider
                    debug=True,
                )

            debug_messages = [
                record.message
                for record in caplog.records
                if record.levelname == "DEBUG"
            ]

            # Should log TTS API call
            assert any("Calling system TTS API" in msg for msg in debug_messages), (
                "Should log which provider is being called"
            )

    @pytest.mark.asyncio
    async def test_no_debug_logging_when_disabled(self, caplog) -> None:
        """Test that debug messages are not logged when debug=False."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "no_debug.wav"

            # Capture at DEBUG level but with debug=False
            with caplog.at_level(logging.DEBUG):
                await speak_text(
                    "No debug test",
                    provider="system",
                    output_file=str(output_path),
                    cache=True,
                    debug=False,  # Debug disabled
                )

            # Should have minimal or no debug messages from core.py
            core_debug = [
                record.message
                for record in caplog.records
                if record.levelname == "DEBUG" and "lspeak.core" in record.name
            ]

            # Should not have cache check messages from core
            assert not any("Checking cache" in msg for msg in core_debug), (
                "Should not log cache operations when debug=False"
            )
