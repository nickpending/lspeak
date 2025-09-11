"""Unit tests for ElevenLabsProvider error handling and logic."""

import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from lspeak.providers.elevenlabs import ElevenLabsProvider
from lspeak.tts.errors import TTSAPIError, TTSAuthError


class TestElevenLabsProviderInitialization:
    """Test ElevenLabsProvider initialization and authentication error handling."""

    def test_initialization_with_provided_api_key(self) -> None:
        """Test ElevenLabsProvider initializes successfully with provided API key."""
        with patch("lspeak.providers.elevenlabs.ElevenLabs") as mock_elevenlabs:
            mock_client = MagicMock()
            mock_elevenlabs.return_value = mock_client

            provider = ElevenLabsProvider(api_key="test_key")

            assert provider._api_key == "test_key"
            mock_elevenlabs.assert_called_once_with(api_key="test_key")
            assert provider._client == mock_client

    def test_initialization_with_env_var_api_key(self) -> None:
        """Test ElevenLabsProvider reads API key from environment variable."""
        with patch.dict(os.environ, {"ELEVENLABS_API_KEY": "env_test_key"}):
            with patch("lspeak.providers.elevenlabs.ElevenLabs") as mock_elevenlabs:
                mock_client = MagicMock()
                mock_elevenlabs.return_value = mock_client

                provider = ElevenLabsProvider()

                assert provider._api_key == "env_test_key"
                mock_elevenlabs.assert_called_once_with(api_key="env_test_key")

    def test_initialization_no_api_key_raises_auth_error(self) -> None:
        """Test ElevenLabsProvider raises TTSAuthError when no API key provided."""
        # Clear environment variable if it exists
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(TTSAuthError, match="ElevenLabs API key not found"):
                ElevenLabsProvider()

    def test_initialization_elevenlabs_client_failure_raises_auth_error(self) -> None:
        """Test ElevenLabsProvider raises TTSAuthError when ElevenLabs client fails to initialize."""
        with patch("lspeak.providers.elevenlabs.ElevenLabs") as mock_elevenlabs:
            mock_elevenlabs.side_effect = Exception("Invalid API key")

            with pytest.raises(
                TTSAuthError, match="Failed to initialize ElevenLabs client"
            ):
                ElevenLabsProvider(api_key="invalid_key")


class TestElevenLabsProviderSynthesizeErrorHandling:
    """Test ElevenLabsProvider synthesize method error handling logic."""

    def setup_method(self) -> None:
        """Set up test provider with mocked ElevenLabs client."""
        with patch("lspeak.providers.elevenlabs.ElevenLabs") as mock_elevenlabs:
            self.mock_elevenlabs_client = MagicMock()
            mock_elevenlabs.return_value = self.mock_elevenlabs_client
            self.provider = ElevenLabsProvider(api_key="test_key")

    @pytest.mark.asyncio
    async def test_synthesize_empty_text_raises_value_error(self) -> None:
        """Test synthesize raises ValueError for empty text."""
        with pytest.raises(ValueError, match="Text cannot be empty"):
            await self.provider.synthesize("", "test_voice")

    @pytest.mark.asyncio
    async def test_synthesize_whitespace_only_text_raises_value_error(self) -> None:
        """Test synthesize raises ValueError for whitespace-only text."""
        with pytest.raises(ValueError, match="Text cannot be empty"):
            await self.provider.synthesize("   ", "test_voice")

    @pytest.mark.asyncio
    async def test_synthesize_unauthorized_error_raises_tts_auth_error(self) -> None:
        """Test synthesize maps 401/unauthorized errors to TTSAuthError."""
        # Mock text_to_speech.convert to raise unauthorized error
        self.mock_elevenlabs_client.text_to_speech.convert.side_effect = Exception(
            "401 unauthorized"
        )

        with pytest.raises(TTSAuthError, match="Authentication failed"):
            await self.provider.synthesize("test text", "test_voice")

    @pytest.mark.asyncio
    async def test_synthesize_rate_limit_error_raises_tts_api_error_with_status(
        self,
    ) -> None:
        """Test synthesize maps 429 rate limit errors to TTSAPIError with status code."""
        # Mock text_to_speech.convert to raise rate limit error
        self.mock_elevenlabs_client.text_to_speech.convert.side_effect = Exception(
            "429 rate limit"
        )

        with pytest.raises(TTSAPIError, match="Rate limit exceeded") as exc_info:
            await self.provider.synthesize("test text", "test_voice")

        assert exc_info.value.status_code == 429

    @pytest.mark.asyncio
    async def test_synthesize_server_error_raises_tts_api_error(self) -> None:
        """Test synthesize maps 5xx server errors to TTSAPIError."""
        # Mock text_to_speech.convert to raise server error
        self.mock_elevenlabs_client.text_to_speech.convert.side_effect = Exception(
            "500 server error"
        )

        with pytest.raises(TTSAPIError, match="Server error"):
            await self.provider.synthesize("test text", "test_voice")

    @pytest.mark.asyncio
    async def test_synthesize_generic_error_raises_tts_api_error(self) -> None:
        """Test synthesize maps other errors to TTSAPIError."""
        # Mock text_to_speech.convert to raise generic error
        self.mock_elevenlabs_client.text_to_speech.convert.side_effect = Exception(
            "network error"
        )

        with pytest.raises(TTSAPIError, match="API call failed"):
            await self.provider.synthesize("test text", "test_voice")

    @pytest.mark.asyncio
    async def test_synthesize_no_audio_data_raises_tts_api_error(self) -> None:
        """Test synthesize raises TTSAPIError when no audio data received."""
        # Mock text_to_speech.convert to return empty generator
        self.mock_elevenlabs_client.text_to_speech.convert.return_value = iter([])

        with pytest.raises(TTSAPIError, match="No audio data received from API"):
            await self.provider.synthesize("test text", "test_voice")

    @pytest.mark.asyncio
    async def test_synthesize_default_voice_selection(self) -> None:
        """Test synthesize selects first available voice when no voice specified."""
        # Mock list_voices to return test voices
        self.provider.list_voices = AsyncMock(
            return_value=[
                {"id": "default_voice", "name": "Default", "provider": "elevenlabs"},
                {"id": "other_voice", "name": "Other", "provider": "elevenlabs"},
            ]
        )

        # Mock successful synthesis
        self.mock_elevenlabs_client.text_to_speech.convert.return_value = iter(
            [b"audio_data"]
        )

        result = await self.provider.synthesize("test text", "")

        # Verify it used the first voice
        self.mock_elevenlabs_client.text_to_speech.convert.assert_called_once_with(
            text="test text", voice_id="default_voice", model_id="eleven_monolingual_v1"
        )
        assert result == b"audio_data"


class TestElevenLabsProviderListVoicesErrorHandling:
    """Test ElevenLabsProvider list_voices method error handling logic."""

    def setup_method(self) -> None:
        """Set up test provider with mocked ElevenLabs client."""
        with patch("lspeak.providers.elevenlabs.ElevenLabs") as mock_elevenlabs:
            self.mock_elevenlabs_client = MagicMock()
            mock_elevenlabs.return_value = self.mock_elevenlabs_client
            self.provider = ElevenLabsProvider(api_key="test_key")

    @pytest.mark.asyncio
    async def test_list_voices_unauthorized_error_raises_tts_auth_error(self) -> None:
        """Test list_voices maps 401/unauthorized errors to TTSAuthError."""
        self.mock_elevenlabs_client.voices.get_all.side_effect = Exception(
            "401 unauthorized"
        )

        with pytest.raises(TTSAuthError, match="Authentication failed"):
            await self.provider.list_voices()

    @pytest.mark.asyncio
    async def test_list_voices_rate_limit_error_raises_tts_api_error_with_status(
        self,
    ) -> None:
        """Test list_voices maps 429 rate limit errors to TTSAPIError with status code."""
        self.mock_elevenlabs_client.voices.get_all.side_effect = Exception(
            "429 rate limit"
        )

        with pytest.raises(TTSAPIError, match="Rate limit exceeded") as exc_info:
            await self.provider.list_voices()

        assert exc_info.value.status_code == 429

    @pytest.mark.asyncio
    async def test_list_voices_server_error_raises_tts_api_error(self) -> None:
        """Test list_voices maps 5xx server errors to TTSAPIError."""
        self.mock_elevenlabs_client.voices.get_all.side_effect = Exception(
            "500 server error"
        )

        with pytest.raises(TTSAPIError, match="Server error"):
            await self.provider.list_voices()

    @pytest.mark.asyncio
    async def test_list_voices_generic_error_raises_tts_api_error(self) -> None:
        """Test list_voices maps other errors to TTSAPIError."""
        self.mock_elevenlabs_client.voices.get_all.side_effect = Exception(
            "network error"
        )

        with pytest.raises(TTSAPIError, match="Failed to list voices"):
            await self.provider.list_voices()


class TestElevenLabsProviderVoiceTransformation:
    """Test ElevenLabsProvider voice transformation logic."""

    def setup_method(self) -> None:
        """Set up test provider with mocked ElevenLabs client."""
        with patch("lspeak.providers.elevenlabs.ElevenLabs") as mock_elevenlabs:
            self.mock_elevenlabs_client = MagicMock()
            mock_elevenlabs.return_value = self.mock_elevenlabs_client
            self.provider = ElevenLabsProvider(api_key="test_key")

    @pytest.mark.asyncio
    async def test_list_voices_transforms_api_response_to_dict_format(self) -> None:
        """Test list_voices correctly transforms API response to dict format."""
        # Create mock voices with various attributes
        mock_voice1 = MagicMock()
        mock_voice1.voice_id = "voice_abc"
        mock_voice1.name = "Alice"

        mock_voice2 = MagicMock()
        mock_voice2.voice_id = "voice_xyz"
        mock_voice2.name = "Bob"

        mock_response = MagicMock()
        mock_response.voices = [mock_voice1, mock_voice2]
        self.mock_elevenlabs_client.voices.get_all.return_value = mock_response

        voices = await self.provider.list_voices()

        # Verify correct transformation to dict format
        assert len(voices) == 2

        # Check first voice transformed correctly
        assert voices[0] == {
            "id": "voice_abc",
            "name": "Alice",
            "provider": "elevenlabs",
        }

        # Check second voice transformed correctly
        assert voices[1] == {"id": "voice_xyz", "name": "Bob", "provider": "elevenlabs"}

        # Verify all are dicts
        for voice in voices:
            assert isinstance(voice, dict)
            assert "id" in voice
            assert "name" in voice
            assert "provider" in voice
            assert voice["provider"] == "elevenlabs"

    @pytest.mark.asyncio
    async def test_list_voices_caching_behavior(self) -> None:
        """Test list_voices caches results after first call."""
        # Create mock voice data
        mock_voice1 = MagicMock()
        mock_voice1.voice_id = "voice_123"
        mock_voice1.name = "Test Voice"

        # First call - should hit API
        mock_response = MagicMock()
        mock_response.voices = [mock_voice1]
        self.mock_elevenlabs_client.voices.get_all.return_value = mock_response

        voices1 = await self.provider.list_voices()

        # Verify API was called once
        assert self.mock_elevenlabs_client.voices.get_all.call_count == 1
        assert len(voices1) == 1
        assert voices1[0]["id"] == "voice_123"

        # Second call - should use cache
        voices2 = await self.provider.list_voices()

        # Verify API was NOT called again
        assert self.mock_elevenlabs_client.voices.get_all.call_count == 1
        # Should return same data
        assert len(voices2) == 1
        assert voices2[0]["id"] == "voice_123"
        # Should be the exact same list object (not a copy)
        assert voices1 is voices2


class TestElevenLabsProviderNoVoicesAvailable:
    """Test ElevenLabsProvider behavior when no voices available."""

    def setup_method(self) -> None:
        """Set up test provider with mocked ElevenLabs client."""
        with patch("lspeak.providers.elevenlabs.ElevenLabs") as mock_elevenlabs:
            self.mock_elevenlabs_client = MagicMock()
            mock_elevenlabs.return_value = self.mock_elevenlabs_client
            self.provider = ElevenLabsProvider(api_key="test_key")

    @pytest.mark.asyncio
    async def test_synthesize_no_voices_available_raises_tts_api_error(self) -> None:
        """Test synthesize raises TTSAPIError when no voices available."""
        # Mock list_voices to return empty list
        self.provider.list_voices = AsyncMock(return_value=[])

        with pytest.raises(TTSAPIError, match="No voices available"):
            await self.provider.synthesize("test text", "")
