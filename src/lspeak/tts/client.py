"""TTS client for ElevenLabs API integration."""

import os

from elevenlabs.client import ElevenLabs

from .errors import TTSAPIError, TTSAuthError
from .models import VoiceInfo


class TTSClient:
    """Client for ElevenLabs text-to-speech API.

    Provides methods to synthesize speech from text and manage voices.
    """

    def __init__(self, api_key: str | None = None) -> None:
        """Initialize TTS client.

        Args:
            api_key: ElevenLabs API key. If not provided, reads from
                    ELEVENLABS_API_KEY environment variable.

        Raises:
            TTSAuthError: If API key is not provided or invalid.
        """
        self._api_key = api_key or os.getenv("ELEVENLABS_API_KEY")
        if not self._api_key:
            raise TTSAuthError(
                "ElevenLabs API key not found. Set ELEVENLABS_API_KEY environment "
                "variable or provide api_key parameter."
            )

        try:
            self._client = ElevenLabs(api_key=self._api_key)
        except Exception as e:
            raise TTSAuthError(f"Failed to initialize ElevenLabs client: {e}") from e

        # Cache for voices to avoid repeated API calls
        self._voices_cache: list[VoiceInfo] | None = None

    def synthesize(
        self,
        text: str,
        voice_id: str | None = None,
        model_id: str = "eleven_monolingual_v1",
    ) -> bytes:
        """Convert text to speech audio bytes.

        Args:
            text: Text to convert to speech
            voice_id: Voice ID to use. If not provided, uses first available voice
            model_id: ElevenLabs model ID to use

        Returns:
            Audio data as bytes (MP3 format)

        Raises:
            TTSAPIError: If API call fails
            TTSAuthError: If authentication fails
            ValueError: If text is empty
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        # Use first available voice if not specified
        if not voice_id:
            voices = self.list_voices()
            if not voices:
                raise TTSAPIError("No voices available")
            voice_id = voices[0].voice_id

        try:
            audio_generator = self._client.text_to_speech.convert(
                text=text.strip(), voice_id=voice_id, model_id=model_id
            )

            # Collect all audio chunks
            audio_bytes = b"".join(audio_generator)

            if not audio_bytes:
                raise TTSAPIError("No audio data received from API")

            return audio_bytes

        except Exception as e:
            if "unauthorized" in str(e).lower() or "401" in str(e):
                raise TTSAuthError(f"Authentication failed: {e}") from e
            elif "429" in str(e):
                raise TTSAPIError(f"Rate limit exceeded: {e}", 429) from e
            elif "5" in str(e)[:1]:  # 5xx server errors
                raise TTSAPIError(f"Server error: {e}") from e
            else:
                raise TTSAPIError(f"API call failed: {e}") from e

    def list_voices(self) -> list[VoiceInfo]:
        """Get list of available voices.

        Results are cached after first call to avoid repeated API requests.

        Returns:
            List of VoiceInfo objects with voice details

        Raises:
            TTSAPIError: If API call fails
            TTSAuthError: If authentication fails
        """
        if self._voices_cache is not None:
            return self._voices_cache

        try:
            response = self._client.voices.get_all()

            voices = []
            for voice in response.voices:
                voice_info = VoiceInfo(
                    voice_id=voice.voice_id,
                    name=voice.name,
                    category=getattr(voice, "category", None),
                    description=getattr(voice, "description", None),
                )
                voices.append(voice_info)

            # Cache the results
            self._voices_cache = voices
            return voices

        except Exception as e:
            if "unauthorized" in str(e).lower() or "401" in str(e):
                raise TTSAuthError(f"Authentication failed: {e}") from e
            elif "429" in str(e):
                raise TTSAPIError(f"Rate limit exceeded: {e}", 429) from e
            elif "5" in str(e)[:1]:  # 5xx server errors
                raise TTSAPIError(f"Server error: {e}") from e
            else:
                raise TTSAPIError(f"Failed to list voices: {e}") from e
