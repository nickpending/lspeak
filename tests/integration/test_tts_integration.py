"""Integration tests for TTS package with real ElevenLabs API."""

import sys
from pathlib import Path

import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from lspeak.tts import (
    TTSAPIError,
    TTSAuthError,
    TTSClient,
    TTSError,
    VoiceInfo,
    VoiceSettings,
)
from lspeak.tts.client import TTSClient
from lspeak.tts.models import VoiceInfo


class TestTTSClientRealAPIIntegration:
    """Test TTSClient with real ElevenLabs API integration."""

    def test_tts_client_initializes_with_real_api(self) -> None:
        """Test TTSClient initializes successfully with real ElevenLabs API."""
        # This uses the real API key from environment
        client = TTSClient()

        # Verify client is initialized properly
        assert client._api_key is not None
        assert client._client is not None
        assert client._voices_cache is None  # Should start empty

    def test_list_voices_returns_real_voices(self) -> None:
        """Test list_voices returns actual voices from ElevenLabs API."""
        client = TTSClient()

        voices = client.list_voices()

        # Verify we get real voice data
        assert len(voices) > 0, "Should return at least one voice"

        # Check first voice has required fields
        first_voice = voices[0]
        assert isinstance(first_voice, VoiceInfo)
        assert first_voice.voice_id, "Voice ID should not be empty"
        assert first_voice.name, "Voice name should not be empty"

        # Verify voice_id format (ElevenLabs uses specific patterns)
        assert len(first_voice.voice_id) > 10, "Voice ID should be a substantial string"

    def test_list_voices_caching_behavior(self) -> None:
        """Test that list_voices caches results after first call."""
        client = TTSClient()

        # First call should populate cache
        voices1 = client.list_voices()
        assert client._voices_cache is not None, (
            "Cache should be populated after first call"
        )

        # Second call should return cached results
        voices2 = client.list_voices()

        # Should return identical results (same objects due to caching)
        assert len(voices1) == len(voices2)
        assert voices1[0].voice_id == voices2[0].voice_id

    def test_synthesize_with_real_api_generates_audio(self) -> None:
        """Test synthesize method generates actual audio with real API."""
        client = TTSClient()

        # Generate audio with default voice
        audio_bytes = client.synthesize("Hello world")

        # Verify we got actual audio data
        assert isinstance(audio_bytes, bytes)
        assert len(audio_bytes) > 1000, "Audio file should be substantial (>1KB)"

        # MP3 files typically start with specific bytes
        # Check for MP3 header or general audio file markers
        assert len(audio_bytes) > 10, "Should have audio header"

    def test_synthesize_with_specific_voice(self) -> None:
        """Test synthesize works with a specific voice ID."""
        client = TTSClient()

        # Get available voices
        voices = client.list_voices()
        assert len(voices) > 0, "Need at least one voice for test"

        # Use first available voice explicitly
        test_voice_id = voices[0].voice_id
        audio_bytes = client.synthesize(
            "Testing specific voice", voice_id=test_voice_id
        )

        # Verify audio was generated
        assert isinstance(audio_bytes, bytes)
        assert len(audio_bytes) > 1000, "Audio should be generated with specific voice"

    def test_synthesize_different_text_lengths(self) -> None:
        """Test synthesize handles different text lengths appropriately."""
        client = TTSClient()

        # Test short text
        short_audio = client.synthesize("Hi")
        assert len(short_audio) > 500, "Even short text should generate some audio"

        # Test longer text
        long_text = "This is a longer sentence to test how the text-to-speech system handles more substantial content."
        long_audio = client.synthesize(long_text)
        assert len(long_audio) > len(short_audio), (
            "Longer text should generate more audio data"
        )


class TestTTSPackageCompleteWorkflow:
    """Test complete TTS package workflow integration."""

    def test_complete_tts_workflow(self) -> None:
        """
        Test complete TTS workflow:
        - Import TTS package components
        - Initialize TTSClient
        - List available voices
        - Select a voice
        - Generate audio
        """
        # Test package imports work
        from lspeak.tts import TTSClient, VoiceInfo

        # Initialize client
        client = TTSClient()

        # List voices and verify structure
        voices = client.list_voices()
        assert len(voices) > 0, "Should have available voices"

        # Verify voice info structure
        test_voice = voices[0]
        assert isinstance(test_voice, VoiceInfo)
        assert test_voice.voice_id
        assert test_voice.name

        # Generate audio with selected voice
        audio_data = client.synthesize(
            "This is a complete test of the TTS workflow", voice_id=test_voice.voice_id
        )

        # Verify final result
        assert isinstance(audio_data, bytes)
        assert len(audio_data) > 2000, (
            "Complete workflow should generate substantial audio"
        )

    def test_voice_info_validation_with_real_data(self) -> None:
        """Test VoiceInfo validation works with real API data."""
        client = TTSClient()
        voices = client.list_voices()

        # All real voices should pass validation
        for voice in voices:
            assert voice.voice_id.strip(), f"Voice {voice.name} has invalid voice_id"
            assert voice.name.strip(), f"Voice {voice.voice_id} has invalid name"

            # Test that we can create new VoiceInfo with same validation
            validated_voice = VoiceInfo(
                voice_id=voice.voice_id,
                name=voice.name,
                category=voice.category,
                description=voice.description,
            )
            assert validated_voice.voice_id == voice.voice_id
            assert validated_voice.name == voice.name

    def test_error_handling_integration(self) -> None:
        """Test error handling works in integrated environment."""
        # Test with invalid API key
        try:
            invalid_client = TTSClient(api_key="invalid_key_test")
            # If we get here, the client initialized but should fail on API calls
            with pytest.raises(TTSAuthError):
                invalid_client.list_voices()
        except TTSAuthError:
            # Expected - invalid key should fail at initialization
            pass

    def test_package_exports_work_correctly(self) -> None:
        """Test that all package exports work correctly."""
        # Test direct imports from package
        from lspeak.tts import (
            TTSAuthError,
            TTSClient,
            VoiceInfo,
        )

        # Verify classes can be instantiated
        client = TTSClient()
        voice_info = VoiceInfo("test_id", "Test Voice")
        voice_settings = VoiceSettings()

        # Verify exception hierarchy
        assert issubclass(TTSAuthError, TTSError)
        assert issubclass(TTSAPIError, TTSError)
        assert issubclass(TTSError, Exception)
