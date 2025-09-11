"""Audio player for cross-platform audio playback using pygame."""

# ruff: noqa: E402
import os

# Suppress pygame's annoying welcome message BEFORE any pygame import
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

import warnings

# Suppress pygame's pkg_resources deprecation warning spam
warnings.filterwarnings("ignore", category=UserWarning, module="pygame.pkgdata")

import asyncio
import io
from pathlib import Path

import pygame


class AudioPlayer:
    """Cross-platform audio player using pygame.

    Provides methods to play audio from bytes or save to file.
    """

    def __init__(self) -> None:
        """Initialize the audio player with pygame mixer.

        Raises:
            RuntimeError: If pygame mixer fails to initialize.
        """
        try:
            pygame.mixer.init()
        except pygame.error as e:
            raise RuntimeError(f"Failed to initialize pygame audio mixer: {e}") from e

    def play_bytes(self, audio_data: bytes) -> None:
        """Play audio from bytes through system speakers (blocking).

        Args:
            audio_data: Audio data in MP3 or WAV format.

        Raises:
            RuntimeError: If audio playback fails.
        """
        if not audio_data:
            raise ValueError("No audio data provided")

        try:
            # Create in-memory file-like object
            audio_file = io.BytesIO(audio_data)

            # Load and play audio
            pygame.mixer.music.load(audio_file)
            pygame.mixer.music.play()

            # Wait for playback to complete
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)

        except pygame.error as e:
            raise RuntimeError(f"Failed to play audio: {e}") from e

    async def play_bytes_async(self, audio_data: bytes) -> None:
        """Play audio from bytes through system speakers (async).

        Args:
            audio_data: Audio data in MP3 or WAV format.

        Raises:
            RuntimeError: If audio playback fails.
        """
        if not audio_data:
            raise ValueError("No audio data provided")

        def _play_audio():
            """Synchronous audio playback in thread."""
            try:
                # Create in-memory file-like object
                audio_file = io.BytesIO(audio_data)

                # Load and play audio
                pygame.mixer.music.load(audio_file)
                pygame.mixer.music.play()

                # Wait for playback to complete
                while pygame.mixer.music.get_busy():
                    pygame.time.Clock().tick(10)

            except pygame.error as e:
                raise RuntimeError(f"Failed to play audio: {e}") from e

        try:
            # Run pygame operations in thread to avoid blocking event loop
            await asyncio.to_thread(_play_audio)
        except Exception as e:
            raise RuntimeError(f"Failed to play audio: {e}") from e

    def save_to_file(self, audio_data: bytes, filepath: str | Path) -> None:
        """Save audio bytes to a file.

        Args:
            audio_data: Audio data to save.
            filepath: Path where the audio file should be saved.

        Raises:
            ValueError: If no audio data provided.
            OSError: If file cannot be written.
        """
        if not audio_data:
            raise ValueError("No audio data provided")

        # Convert to Path object if string
        filepath = Path(filepath)

        try:
            # Create parent directories if they don't exist
            filepath.parent.mkdir(parents=True, exist_ok=True)

            # Write audio data to file
            filepath.write_bytes(audio_data)

        except OSError as e:
            raise OSError(f"Failed to save audio to {filepath}: {e}") from e
