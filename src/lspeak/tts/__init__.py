"""TTS (Text-to-Speech) package for lspeak.

This package provides text-to-speech functionality using ElevenLabs API.
"""

from .client import TTSClient
from .errors import TTSAPIError, TTSAuthError, TTSError
from .models import VoiceInfo, VoiceSettings

__all__ = [
    "TTSAPIError",
    "TTSAuthError",
    "TTSClient",
    "TTSError",
    "VoiceInfo",
    "VoiceSettings",
]
