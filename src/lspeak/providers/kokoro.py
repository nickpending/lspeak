"""Kokoro text-to-speech provider implementation."""

import asyncio
import io
import logging
import re

import soundfile as sf
import torch
from kokoro import KPipeline

from ..config import load_config
from .base import TTSProvider

logger = logging.getLogger(__name__)

# Voice prefix -> lang_code mapping
LANG_CODES = {"a": "a", "b": "b"}


class KokoroProvider(TTSProvider):
    """Kokoro TTS provider using local neural speech synthesis.

    Uses the Kokoro-82M model for GPU-accelerated (MPS/CUDA) or CPU-based
    text-to-speech generation. No API key required — runs entirely locally.

    Automatically selects American or British phonemizer based on voice prefix.
    """

    def __init__(self) -> None:
        """Initialize Kokoro provider with lazy pipeline creation."""
        self._device = self._resolve_device()
        self._pipelines: dict[str, KPipeline] = {}
        logger.info(f"Kokoro using device: {self._device}")

    def _get_pipeline(self, voice: str) -> KPipeline:
        """Get or create a KPipeline for the given voice's language.

        Pipelines are cached by lang_code so switching between American
        and British voices doesn't reload the model unnecessarily.
        """
        lang_code = LANG_CODES.get(voice[0], "a") if voice else "a"
        if lang_code not in self._pipelines:
            logger.info(f"Creating Kokoro pipeline for lang_code='{lang_code}'")
            self._pipelines[lang_code] = KPipeline(
                lang_code=lang_code, device=self._device
            )
        return self._pipelines[lang_code]

    @staticmethod
    def _resolve_device() -> str:
        """Resolve compute device from config.

        Config value 'auto' selects the best available device.
        Explicit values ('mps', 'cuda', 'cpu') are passed through.
        """
        config = load_config()
        device = config.tts.device

        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            if torch.backends.mps.is_available():
                return "mps"
            return "cpu"

        return device

    @staticmethod
    def _preprocess_text(text: str) -> str:
        """Apply pronunciation overrides from config.

        Replaces words with misaki's inline IPA syntax: [word](/IPA/)
        """
        config = load_config()
        pronunciation = config.tts.pronunciation
        if not pronunciation:
            return text
        for word, ipa in pronunciation.items():
            text = re.sub(rf"\b{re.escape(word)}\b", f"[{word}](/{ipa}/)", text)
        return text

    async def synthesize(self, text: str, voice: str = "bf_isabella") -> bytes:
        """Convert text to speech using Kokoro neural TTS.

        Args:
            text: Text to convert to speech
            voice: Kokoro voice ID (e.g., bf_isabella, af_heart, am_adam)

        Returns:
            Audio data as bytes (WAV format, 24kHz)

        Raises:
            ValueError: If text is empty
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        voice = voice or "bf_isabella"
        text = self._preprocess_text(text)
        pipeline = self._get_pipeline(voice)

        def _generate() -> bytes:
            buf = io.BytesIO()
            for _, _, audio in pipeline(text, voice=voice):
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
                "bf_alice",
                "bf_emma",
                "bf_isabella",
                "bf_lily",
                "bm_daniel",
                "bm_fable",
                "bm_george",
                "bm_lewis",
            ]
        ]
