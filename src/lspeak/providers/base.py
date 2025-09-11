"""Abstract base class for text-to-speech providers.

This module defines the interface that all TTS providers must implement,
ensuring consistent behavior across different TTS backends.
"""

from abc import ABC, abstractmethod


class TTSProvider(ABC):
    """Abstract base class for text-to-speech providers.

    All TTS providers must inherit from this class and implement
    the required methods for synthesizing speech and listing voices.

    Voice Dictionary Structure:
        Each voice returned by list_voices() should follow this structure:
        {
            "id": str,       # Unique identifier for the voice
            "name": str,     # Human-readable name for the voice
            "provider": str  # Name of the provider (e.g., "elevenlabs", "system")
        }
    """

    @abstractmethod
    async def synthesize(self, text: str, voice: str) -> bytes:
        """Convert text to audio bytes.

        Args:
            text: The text to convert to speech
            voice: Voice ID or name to use for synthesis

        Returns:
            Audio data as bytes (MP3 or WAV format)

        Raises:
            Exception: If synthesis fails
        """
        pass

    @abstractmethod
    async def list_voices(self) -> list[dict]:
        """Return available voices for this provider.

        Returns:
            List of voice dictionaries, each containing:
            - id: Unique voice identifier
            - name: Human-readable voice name
            - provider: Provider name

        Raises:
            Exception: If voice listing fails
        """
        pass
