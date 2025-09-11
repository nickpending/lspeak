"""Unit tests for TTSClient error handling logic."""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from lspeak.tts.client import TTSClient
from lspeak.tts.errors import TTSAPIError, TTSAuthError


class TestTTSClientInitialization:
    """Test TTSClient initialization and authentication error handling."""

    def test_initialization_with_provided_api_key(self) -> None:
        """Test TTSClient initializes successfully with provided API key."""
        with patch("lspeak.tts.client.ElevenLabs") as mock_elevenlabs:
            mock_client = MagicMock()
            mock_elevenlabs.return_value = mock_client

            client = TTSClient(api_key="test_key")

            assert client._api_key == "test_key"
            mock_elevenlabs.assert_called_once_with(api_key="test_key")
            assert client._client == mock_client

    def test_initialization_with_env_var_api_key(self) -> None:
        """Test TTSClient reads API key from environment variable."""
        with patch.dict(os.environ, {"ELEVENLABS_API_KEY": "env_test_key"}):
            with patch("lspeak.tts.client.ElevenLabs") as mock_elevenlabs:
                mock_client = MagicMock()
                mock_elevenlabs.return_value = mock_client

                client = TTSClient()

                assert client._api_key == "env_test_key"
                mock_elevenlabs.assert_called_once_with(api_key="env_test_key")

    def test_initialization_no_api_key_raises_auth_error(self) -> None:
        """Test TTSClient raises TTSAuthError when no API key provided."""
        # Clear environment variable if it exists
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(TTSAuthError, match="ElevenLabs API key not found"):
                TTSClient()

    def test_initialization_elevenlabs_client_failure_raises_auth_error(self) -> None:
        """Test TTSClient raises TTSAuthError when ElevenLabs client fails to initialize."""
        with patch("lspeak.tts.client.ElevenLabs") as mock_elevenlabs:
            mock_elevenlabs.side_effect = Exception("Invalid API key")

            with pytest.raises(
                TTSAuthError, match="Failed to initialize ElevenLabs client"
            ):
                TTSClient(api_key="invalid_key")


class TestTTSClientSynthesizeErrorHandling:
    """Test TTSClient synthesize method error handling logic."""

    def setup_method(self) -> None:
        """Set up test client with mocked ElevenLabs client."""
        with patch("lspeak.tts.client.ElevenLabs") as mock_elevenlabs:
            self.mock_elevenlabs_client = MagicMock()
            mock_elevenlabs.return_value = self.mock_elevenlabs_client
            self.client = TTSClient(api_key="test_key")

    def test_synthesize_empty_text_raises_value_error(self) -> None:
        """Test synthesize raises ValueError for empty text."""
        with pytest.raises(ValueError, match="Text cannot be empty"):
            self.client.synthesize("")

    def test_synthesize_whitespace_only_text_raises_value_error(self) -> None:
        """Test synthesize raises ValueError for whitespace-only text."""
        with pytest.raises(ValueError, match="Text cannot be empty"):
            self.client.synthesize("   ")

    def test_synthesize_unauthorized_error_raises_tts_auth_error(self) -> None:
        """Test synthesize maps 401/unauthorized errors to TTSAuthError."""
        # Mock list_voices to return test voice
        with patch.object(self.client, "list_voices") as mock_list_voices:
            mock_voice = MagicMock()
            mock_voice.voice_id = "test_voice"
            mock_list_voices.return_value = [mock_voice]

            # Mock text_to_speech.convert to raise unauthorized error
            self.mock_elevenlabs_client.text_to_speech.convert.side_effect = Exception(
                "401 unauthorized"
            )

            with pytest.raises(TTSAuthError, match="Authentication failed"):
                self.client.synthesize("test text")

    def test_synthesize_rate_limit_error_raises_tts_api_error_with_status(self) -> None:
        """Test synthesize maps 429 rate limit errors to TTSAPIError with status code."""
        # Mock list_voices to return test voice
        with patch.object(self.client, "list_voices") as mock_list_voices:
            mock_voice = MagicMock()
            mock_voice.voice_id = "test_voice"
            mock_list_voices.return_value = [mock_voice]

            # Mock text_to_speech.convert to raise rate limit error
            self.mock_elevenlabs_client.text_to_speech.convert.side_effect = Exception(
                "429 rate limit"
            )

            with pytest.raises(TTSAPIError, match="Rate limit exceeded") as exc_info:
                self.client.synthesize("test text")

            assert exc_info.value.status_code == 429

    def test_synthesize_server_error_raises_tts_api_error(self) -> None:
        """Test synthesize maps 5xx server errors to TTSAPIError."""
        # Mock list_voices to return test voice
        with patch.object(self.client, "list_voices") as mock_list_voices:
            mock_voice = MagicMock()
            mock_voice.voice_id = "test_voice"
            mock_list_voices.return_value = [mock_voice]

            # Mock text_to_speech.convert to raise server error
            self.mock_elevenlabs_client.text_to_speech.convert.side_effect = Exception(
                "500 server error"
            )

            with pytest.raises(TTSAPIError, match="Server error"):
                self.client.synthesize("test text")

    def test_synthesize_generic_error_raises_tts_api_error(self) -> None:
        """Test synthesize maps other errors to TTSAPIError."""
        # Mock list_voices to return test voice
        with patch.object(self.client, "list_voices") as mock_list_voices:
            mock_voice = MagicMock()
            mock_voice.voice_id = "test_voice"
            mock_list_voices.return_value = [mock_voice]

            # Mock text_to_speech.convert to raise generic error
            self.mock_elevenlabs_client.text_to_speech.convert.side_effect = Exception(
                "network error"
            )

            with pytest.raises(TTSAPIError, match="API call failed"):
                self.client.synthesize("test text")

    def test_synthesize_no_audio_data_raises_tts_api_error(self) -> None:
        """Test synthesize raises TTSAPIError when no audio data received."""
        # Mock list_voices to return test voice
        with patch.object(self.client, "list_voices") as mock_list_voices:
            mock_voice = MagicMock()
            mock_voice.voice_id = "test_voice"
            mock_list_voices.return_value = [mock_voice]

            # Mock text_to_speech.convert to return empty generator
            self.mock_elevenlabs_client.text_to_speech.convert.return_value = iter([])

            with pytest.raises(TTSAPIError, match="No audio data received from API"):
                self.client.synthesize("test text")


class TestTTSClientListVoicesErrorHandling:
    """Test TTSClient list_voices method error handling logic."""

    def setup_method(self) -> None:
        """Set up test client with mocked ElevenLabs client."""
        with patch("lspeak.tts.client.ElevenLabs") as mock_elevenlabs:
            self.mock_elevenlabs_client = MagicMock()
            mock_elevenlabs.return_value = self.mock_elevenlabs_client
            self.client = TTSClient(api_key="test_key")

    def test_list_voices_unauthorized_error_raises_tts_auth_error(self) -> None:
        """Test list_voices maps 401/unauthorized errors to TTSAuthError."""
        self.mock_elevenlabs_client.voices.get_all.side_effect = Exception(
            "401 unauthorized"
        )

        with pytest.raises(TTSAuthError, match="Authentication failed"):
            self.client.list_voices()

    def test_list_voices_rate_limit_error_raises_tts_api_error_with_status(
        self,
    ) -> None:
        """Test list_voices maps 429 rate limit errors to TTSAPIError with status code."""
        self.mock_elevenlabs_client.voices.get_all.side_effect = Exception(
            "429 rate limit"
        )

        with pytest.raises(TTSAPIError, match="Rate limit exceeded") as exc_info:
            self.client.list_voices()

        assert exc_info.value.status_code == 429

    def test_list_voices_server_error_raises_tts_api_error(self) -> None:
        """Test list_voices maps 5xx server errors to TTSAPIError."""
        self.mock_elevenlabs_client.voices.get_all.side_effect = Exception(
            "500 server error"
        )

        with pytest.raises(TTSAPIError, match="Server error"):
            self.client.list_voices()

    def test_list_voices_generic_error_raises_tts_api_error(self) -> None:
        """Test list_voices maps other errors to TTSAPIError."""
        self.mock_elevenlabs_client.voices.get_all.side_effect = Exception(
            "network error"
        )

        with pytest.raises(TTSAPIError, match="Failed to list voices"):
            self.client.list_voices()


class TestTTSClientListVoicesCaching:
    """Test TTSClient list_voices caching behavior."""

    def setup_method(self) -> None:
        """Set up test client with mocked ElevenLabs client."""
        with patch("lspeak.tts.client.ElevenLabs") as mock_elevenlabs:
            self.mock_elevenlabs_client = MagicMock()
            mock_elevenlabs.return_value = self.mock_elevenlabs_client
            self.client = TTSClient(api_key="test_key")

    def test_list_voices_cache_hit_returns_cached_data(self) -> None:
        """Test list_voices returns cached data without API call on cache hit."""
        # Create mock voice data
        mock_voice1 = MagicMock()
        mock_voice1.voice_id = "voice_123"
        mock_voice1.name = "Test Voice"
        mock_voice1.category = "general"
        mock_voice1.description = "A test voice"

        # First call - should hit API
        mock_response = MagicMock()
        mock_response.voices = [mock_voice1]
        self.mock_elevenlabs_client.voices.get_all.return_value = mock_response

        voices1 = self.client.list_voices()

        # Verify API was called once
        assert self.mock_elevenlabs_client.voices.get_all.call_count == 1
        assert len(voices1) == 1
        assert voices1[0].voice_id == "voice_123"

        # Second call - should use cache
        voices2 = self.client.list_voices()

        # Verify API was NOT called again
        assert self.mock_elevenlabs_client.voices.get_all.call_count == 1
        # Should return same data
        assert len(voices2) == 1
        assert voices2[0].voice_id == "voice_123"
        # Should be the exact same list object (not a copy)
        assert voices1 is voices2

    def test_list_voices_transforms_api_response_to_voice_info(self) -> None:
        """Test list_voices correctly transforms API response to VoiceInfo objects."""
        # Create mock voices with various attributes
        mock_voice1 = MagicMock()
        mock_voice1.voice_id = "voice_abc"
        mock_voice1.name = "Alice"
        mock_voice1.category = "conversational"
        mock_voice1.description = "Friendly voice"

        mock_voice2 = MagicMock()
        mock_voice2.voice_id = "voice_xyz"
        mock_voice2.name = "Bob"
        # Simulate missing optional attributes
        delattr(mock_voice2, "category")
        delattr(mock_voice2, "description")

        mock_response = MagicMock()
        mock_response.voices = [mock_voice1, mock_voice2]
        self.mock_elevenlabs_client.voices.get_all.return_value = mock_response

        voices = self.client.list_voices()

        # Verify correct transformation
        assert len(voices) == 2

        # Check first voice with all attributes
        assert voices[0].voice_id == "voice_abc"
        assert voices[0].name == "Alice"
        assert voices[0].category == "conversational"
        assert voices[0].description == "Friendly voice"

        # Check second voice with missing optional attributes
        assert voices[1].voice_id == "voice_xyz"
        assert voices[1].name == "Bob"
        assert voices[1].category is None
        assert voices[1].description is None

        # Verify all are VoiceInfo instances
        from lspeak.tts.models import VoiceInfo

        for voice in voices:
            assert isinstance(voice, VoiceInfo)


class TestTTSClientVoiceSelection:
    """Test TTSClient voice selection logic."""

    def setup_method(self) -> None:
        """Set up test client with mocked ElevenLabs client."""
        with patch("lspeak.tts.client.ElevenLabs") as mock_elevenlabs:
            self.mock_elevenlabs_client = MagicMock()
            mock_elevenlabs.return_value = self.mock_elevenlabs_client
            self.client = TTSClient(api_key="test_key")

    def test_synthesize_no_voices_available_raises_tts_api_error(self) -> None:
        """Test synthesize raises TTSAPIError when no voices available."""
        # Mock list_voices to return empty list
        with patch.object(self.client, "list_voices") as mock_list_voices:
            mock_list_voices.return_value = []

            with pytest.raises(TTSAPIError, match="No voices available"):
                self.client.synthesize("test text")
