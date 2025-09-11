"""Integration tests for provider abstraction with real services."""

import sys
from pathlib import Path

import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from lspeak.core import list_available_voices, speak_text
from lspeak.providers import ProviderRegistry
from lspeak.providers.base import TTSProvider
from lspeak.providers.elevenlabs import ElevenLabsProvider
from lspeak.tts.errors import TTSAuthError


class TestElevenLabsProviderRealAPIIntegration:
    """Test ElevenLabsProvider with real ElevenLabs API integration."""

    @pytest.mark.asyncio
    async def test_elevenlabs_provider_initializes_with_real_api(self) -> None:
        """Test ElevenLabsProvider initializes successfully with real ElevenLabs API."""
        # This uses the real API key from environment
        provider = ElevenLabsProvider()

        # Verify provider is initialized properly
        assert provider._api_key is not None
        assert provider._client is not None
        assert provider._voices_cache is None  # Should start empty

    @pytest.mark.asyncio
    async def test_list_voices_returns_real_voices(self) -> None:
        """Test list_voices returns actual voices from ElevenLabs API."""
        provider = ElevenLabsProvider()

        voices = await provider.list_voices()

        # Verify we get real voice data
        assert len(voices) > 0, "Should return at least one voice"

        # Check first voice has required fields in dict format
        first_voice = voices[0]
        assert isinstance(first_voice, dict)
        assert "id" in first_voice
        assert "name" in first_voice
        assert "provider" in first_voice
        assert first_voice["provider"] == "elevenlabs"
        assert first_voice["id"], "Voice ID should not be empty"
        assert first_voice["name"], "Voice name should not be empty"

        # Verify voice_id format (ElevenLabs uses specific patterns)
        assert len(first_voice["id"]) > 10, "Voice ID should be a substantial string"

    @pytest.mark.asyncio
    async def test_list_voices_caching_behavior(self) -> None:
        """Test that list_voices caches results after first call."""
        provider = ElevenLabsProvider()

        # First call should populate cache
        voices1 = await provider.list_voices()
        assert provider._voices_cache is not None, (
            "Cache should be populated after first call"
        )

        # Second call should return cached results
        voices2 = await provider.list_voices()

        # Should return identical results (same objects due to caching)
        assert len(voices1) == len(voices2)
        assert voices1[0]["id"] == voices2[0]["id"]

    @pytest.mark.asyncio
    async def test_synthesize_with_real_api_generates_audio(self) -> None:
        """Test synthesize method generates actual audio with real API."""
        provider = ElevenLabsProvider()

        # Get a voice to use
        voices = await provider.list_voices()
        test_voice = voices[0]["id"]

        # Generate audio with specific voice
        audio_bytes = await provider.synthesize("Hello world", test_voice)

        # Verify we got actual audio data
        assert isinstance(audio_bytes, bytes)
        assert len(audio_bytes) > 1000, "Audio file should be substantial (>1KB)"

        # MP3 files typically start with specific bytes
        # Check for MP3 header or general audio file markers
        assert len(audio_bytes) > 10, "Should have audio header"

    @pytest.mark.asyncio
    async def test_synthesize_with_empty_voice_uses_default(self) -> None:
        """Test synthesize works when voice parameter is empty string."""
        provider = ElevenLabsProvider()

        # Generate audio with empty voice (should use first available)
        audio_bytes = await provider.synthesize("Testing default voice", "")

        # Verify audio was generated
        assert isinstance(audio_bytes, bytes)
        assert len(audio_bytes) > 1000, "Audio should be generated with default voice"

    @pytest.mark.asyncio
    async def test_synthesize_different_text_lengths(self) -> None:
        """Test synthesize handles different text lengths appropriately."""
        provider = ElevenLabsProvider()
        voices = await provider.list_voices()
        test_voice = voices[0]["id"]

        # Test short text
        short_audio = await provider.synthesize("Hi", test_voice)
        assert len(short_audio) > 500, "Even short text should generate some audio"

        # Test longer text
        long_text = "This is a longer sentence to test how the text-to-speech system handles more substantial content."
        long_audio = await provider.synthesize(long_text, test_voice)
        assert len(long_audio) > len(short_audio), (
            "Longer text should generate more audio data"
        )


class TestProviderRegistryIntegration:
    """Test provider registry integration with real providers."""

    def test_elevenlabs_provider_is_registered(self) -> None:
        """Test that ElevenLabs provider is registered in the registry."""
        # Verify provider is registered
        assert "elevenlabs" in ProviderRegistry._providers

        # Get provider class
        provider_class = ProviderRegistry.get("elevenlabs")
        assert provider_class is ElevenLabsProvider

        # Verify it's a TTSProvider
        assert issubclass(provider_class, TTSProvider)

    def test_get_and_instantiate_provider(self) -> None:
        """Test getting and instantiating provider from registry."""
        # Get provider class from registry
        provider_class = ProviderRegistry.get("elevenlabs")

        # Instantiate it
        provider = provider_class()

        # Verify it's the right type
        assert isinstance(provider, ElevenLabsProvider)
        assert isinstance(provider, TTSProvider)

    def test_registry_error_for_nonexistent_provider(self) -> None:
        """Test registry gives helpful error for non-existent provider."""
        with pytest.raises(
            KeyError,
            match="Provider 'nonexistent' not found. Available providers: elevenlabs",
        ):
            ProviderRegistry.get("nonexistent")


class TestCoreIntegrationWithProviders:
    """Test core.py integration with provider abstraction."""

    @pytest.mark.asyncio
    async def test_list_available_voices_with_provider(self) -> None:
        """Test list_available_voices works with provider parameter."""
        # Capture output
        import io
        import sys

        captured_output = io.StringIO()
        sys.stdout = captured_output

        try:
            # List voices using provider
            await list_available_voices(provider="elevenlabs")

            # Get output
            output = captured_output.getvalue()

            # Verify output format
            assert output, "Should have output"
            lines = output.strip().split("\n")
            assert len(lines) > 0, "Should have voice lines"

            # Each line should be "Name: voice_id"
            first_line = lines[0]
            assert ": " in first_line, "Should have name: id format"
            name, voice_id = first_line.split(": ", 1)
            assert name, "Should have voice name"
            assert voice_id, "Should have voice ID"

        finally:
            sys.stdout = sys.__stdout__

    @pytest.mark.asyncio
    async def test_speak_text_with_provider(self) -> None:
        """Test speak_text works with provider parameter."""
        # Test with file output to avoid audio playback
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            output_file = tmp.name

        try:
            # Synthesize audio using provider
            await speak_text(
                "Testing provider integration",
                provider="elevenlabs",
                output_file=output_file,
            )

            # Verify file was created with content
            output_path = Path(output_file)
            assert output_path.exists(), "Output file should exist"
            assert output_path.stat().st_size > 1000, "Audio file should have content"

        finally:
            # Clean up
            Path(output_file).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_speak_text_with_specific_voice(self) -> None:
        """Test speak_text works with specific voice ID."""
        # Get available voices first
        provider = ElevenLabsProvider()
        voices = await provider.list_voices()
        test_voice_id = voices[0]["id"]

        # Test with file output
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            output_file = tmp.name

        try:
            # Synthesize audio with specific voice
            await speak_text(
                "Testing specific voice",
                provider="elevenlabs",
                voice_id=test_voice_id,
                output_file=output_file,
            )

            # Verify file was created
            output_path = Path(output_file)
            assert output_path.exists(), "Output file should exist"
            assert output_path.stat().st_size > 1000, "Audio file should have content"

        finally:
            # Clean up
            Path(output_file).unlink(missing_ok=True)


class TestCompleteProviderWorkflow:
    """Test complete provider workflow integration."""

    @pytest.mark.asyncio
    async def test_complete_provider_workflow(self) -> None:
        """
        Test complete provider workflow:
        - Get provider from registry
        - List available voices
        - Select a voice
        - Generate audio
        - Use through core API
        """
        # Step 1: Get provider from registry
        provider_class = ProviderRegistry.get("elevenlabs")
        provider = provider_class()
        assert isinstance(provider, TTSProvider)

        # Step 2: List voices
        voices = await provider.list_voices()
        assert len(voices) > 0, "Should have available voices"

        # Step 3: Select a voice
        test_voice = voices[0]
        assert test_voice["id"]
        assert test_voice["name"]
        assert test_voice["provider"] == "elevenlabs"

        # Step 4: Generate audio directly
        audio_data = await provider.synthesize(
            "This is a complete test of the provider workflow", test_voice["id"]
        )
        assert isinstance(audio_data, bytes)
        assert len(audio_data) > 2000, (
            "Complete workflow should generate substantial audio"
        )

        # Step 5: Use through core API
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            output_file = tmp.name

        try:
            await speak_text(
                "Testing through core API",
                provider="elevenlabs",
                voice_id=test_voice["id"],
                output_file=output_file,
            )

            # Verify file exists
            assert Path(output_file).exists()
            assert Path(output_file).stat().st_size > 1000

        finally:
            Path(output_file).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_provider_error_handling_integration(self) -> None:
        """Test error handling works in integrated environment."""
        # Test with invalid API key
        import os

        original_key = os.environ.get("ELEVENLABS_API_KEY")

        try:
            # Set invalid key
            os.environ["ELEVENLABS_API_KEY"] = "invalid_key_test"

            # Create provider with invalid key
            provider = ElevenLabsProvider()

            # Should fail on API calls
            with pytest.raises(TTSAuthError):
                await provider.list_voices()

        finally:
            # Restore original key
            if original_key:
                os.environ["ELEVENLABS_API_KEY"] = original_key
            else:
                os.environ.pop("ELEVENLABS_API_KEY", None)
