"""Cache management for lspeak TTS system."""

from pathlib import Path


def get_cache_dir() -> Path:
    """Get or create the lspeak cache directory.

    Creates ~/.cache/lspeak/ and ~/.cache/lspeak/audio/ directories
    if they don't exist.

    Returns:
        Path to the cache directory
    """
    cache_dir = Path.home() / ".cache" / "lspeak"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Create audio subdirectory for audio files
    audio_dir = cache_dir / "audio"
    audio_dir.mkdir(exist_ok=True)

    return cache_dir
