"""Kokoro text-to-speech provider implementation."""

import asyncio
import io

import soundfile as sf
from kokoro import KPipeline

from .base import TTSProvider


class KokoroProvider(TTSProvider):
    """Kokoro TTS provider using local neural speech synthesis.

    Uses the Kokoro-82M model for CPU-based text-to-speech generation.
    No API key required — runs entirely locally.
    """

    def __init__(self, lang_code: str = "a") -> None:
        """Initialize Kokoro provider.

        Args:
            lang_code: Language code for phonemizer. 'a' = American English.
        """
        self._lang_code = lang_code
        self._pipeline = KPipeline(lang_code=lang_code)

    async def synthesize(self, text: str, voice: str = "af_heart") -> bytes:
        """Convert text to speech using Kokoro neural TTS.

        Args:
            text: Text to convert to speech
            voice: Kokoro voice ID (e.g., af_heart, am_adam)

        Returns:
            Audio data as bytes (WAV format, 24kHz)

        Raises:
            ValueError: If text is empty
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        voice = voice or "af_heart"

        def _generate() -> bytes:
            buf = io.BytesIO()
            for _, _, audio in self._pipeline(text, voice=voice):
                if audio is not None:
                    sf.write(buf, audio, 24000, format="WAV")
            buf.seek(0)
            return buf.read()

        return await asyncio.to_thread(_generate)

    async def list_voices(self) -> list[dict]:
        """List available Kokoro voices.

        Returns:
            List of voice dictionaries with id, name, and provider fields.
        """
        return [
            {"id": v, "name": v, "provider": "kokoro"}
            for v in [
                "af_heart",
                "af_alloy",
                "af_aoede",
                "af_bella",
                "af_jessica",
                "af_kore",
                "af_nicole",
                "af_nova",
                "af_river",
                "af_sarah",
                "af_sky",
                "am_adam",
                "am_echo",
                "am_eric",
                "am_fenrir",
                "am_liam",
                "am_michael",
                "am_onyx",
                "am_puck",
                "am_santa",
            ]
        ]
