"""TTS data models with validation."""

from dataclasses import dataclass


@dataclass
class VoiceInfo:
    """Information about an available voice.

    Args:
        voice_id: Unique identifier for the voice
        name: Human-readable name of the voice
        category: Optional voice category (e.g., "premade", "cloned")
        description: Optional voice description
    """

    voice_id: str
    name: str
    category: str | None = None
    description: str | None = None

    def __post_init__(self) -> None:
        """Validate voice information."""
        if not self.voice_id or not self.voice_id.strip():
            raise ValueError("voice_id cannot be empty")
        if not self.name or not self.name.strip():
            raise ValueError("name cannot be empty")


@dataclass
class VoiceSettings:
    """Voice generation settings.

    Args:
        stability: Voice stability (0.0-1.0)
        similarity_boost: Voice similarity boost (0.0-1.0)
        style: Voice style exaggeration (0.0-1.0)
        use_speaker_boost: Whether to use speaker boost
        speaking_rate: Speaking rate (0.25-4.0)
    """

    stability: float = 0.79
    similarity_boost: float = 0.85
    style: float = 0.25
    use_speaker_boost: bool = True
    speaking_rate: float = 0.79

    def __post_init__(self) -> None:
        """Validate voice settings."""
        if not 0.0 <= self.stability <= 1.0:
            raise ValueError("stability must be between 0.0 and 1.0")
        if not 0.0 <= self.similarity_boost <= 1.0:
            raise ValueError("similarity_boost must be between 0.0 and 1.0")
        if not 0.0 <= self.style <= 1.0:
            raise ValueError("style must be between 0.0 and 1.0")
        if not 0.25 <= self.speaking_rate <= 4.0:
            raise ValueError("speaking_rate must be between 0.25 and 4.0")
