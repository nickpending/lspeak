"""ElevenLabs text-to-speech provider implementation."""

import asyncio
import os

from elevenlabs.client import ElevenLabs

from ..tts.errors import TTSAPIError, TTSAuthError
from ..tts.models import VoiceSettings
from .base import TTSProvider


class ElevenLabsProvider(TTSProvider):
    """ElevenLabs TTS provider implementation.

    Provides methods to synthesize speech from text and manage voices
    using the ElevenLabs API.
    """

    def __init__(self, api_key: str | None = None) -> None:
        """Initialize ElevenLabs provider.

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
        self._voices_cache: list[dict] | None = None

    async def synthesize(
        self,
        text: str,
        voice: str,
        model_id: str = "eleven_turbo_v2_5",
    ) -> bytes:
        """Convert text to speech audio bytes.

        Args:
            text: Text to convert to speech
            voice: Voice ID to use for synthesis
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
        if not voice:
            voices = await self.list_voices()
            if not voices:
                raise TTSAPIError("No voices available")
            voice = voices[0]["id"]

        try:
            # Create voice settings optimized for natural speech
            voice_settings = VoiceSettings(
                stability=0.65,  # More natural variation than 0.79
                similarity_boost=0.75,  # Balance voice likeness with variation
                style=0.4,  # More expressive than 0.25
                use_speaker_boost=True,  # Boost volume for clearer audio
                speaking_rate=0.87,  # Faster, more natural pace
            )

            # Run synchronous ElevenLabs client in thread to avoid blocking event loop
            def _sync_convert():
                audio_generator = self._client.text_to_speech.convert(
                    text=text.strip(),
                    voice_id=voice,
                    model_id=model_id,
                    voice_settings=voice_settings.to_dict()
                    if hasattr(voice_settings, "to_dict")
                    else {
                        "stability": voice_settings.stability,
                        "similarity_boost": voice_settings.similarity_boost,
                        "style": voice_settings.style,
                        "use_speaker_boost": voice_settings.use_speaker_boost,
                    },
                )
                # Collect all audio chunks
                return b"".join(audio_generator)

            # Execute in thread pool to avoid blocking
            audio_bytes = await asyncio.to_thread(_sync_convert)

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

    async def list_voices(self) -> list[dict]:
        """Get list of available voices.

        Results are cached after first call to avoid repeated API requests.

        Returns:
            List of voice dictionaries with id, name, and provider fields

        Raises:
            TTSAPIError: If API call fails
            TTSAuthError: If authentication fails
        """
        if self._voices_cache is not None:
            return self._voices_cache

        try:
            # Run synchronous voice listing in thread
            def _sync_get_voices():
                response = self._client.voices.get_all()
                voices = []
                for voice in response.voices:
                    voice_dict = {
                        "id": voice.voice_id,
                        "name": voice.name,
                        "provider": "elevenlabs",
                    }
                    voices.append(voice_dict)
                return voices

            # Execute in thread pool to avoid blocking
            voices = await asyncio.to_thread(_sync_get_voices)

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
