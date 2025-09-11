"""Data models for cache storage."""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class CacheEntry:
    """Cache entry containing TTS metadata and audio file reference.

    Attributes:
        text: Original input text for TTS
        provider: TTS provider name (e.g., "elevenlabs", "system")
        voice: Voice identifier used for synthesis
        audio_path: Path to the cached audio file
        embedding_idx: Index in FAISS similarity search array
        timestamp: When this entry was created
    """

    text: str
    provider: str
    voice: str
    audio_path: Path
    embedding_idx: int
    timestamp: datetime
