"""Unit tests for TTS data models validation logic."""

import sys
from pathlib import Path

import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from lspeak.tts.models import VoiceInfo, VoiceSettings


class TestVoiceInfo:
    """Test VoiceInfo validation logic."""

    def test_valid_voice_info_creation(self) -> None:
        """Test creating valid VoiceInfo with required fields."""
        voice = VoiceInfo(voice_id="voice_123", name="Rachel")

        assert voice.voice_id == "voice_123"
        assert voice.name == "Rachel"
        assert voice.category is None
        assert voice.description is None

    def test_valid_voice_info_with_optional_fields(self) -> None:
        """Test creating VoiceInfo with all optional fields."""
        voice = VoiceInfo(
            voice_id="voice_456",
            name="Adam",
            category="premade",
            description="Professional narrator voice",
        )

        assert voice.voice_id == "voice_456"
        assert voice.name == "Adam"
        assert voice.category == "premade"
        assert voice.description == "Professional narrator voice"

    def test_empty_voice_id_raises_error(self) -> None:
        """Test that empty voice_id raises ValueError."""
        with pytest.raises(ValueError, match="voice_id cannot be empty"):
            VoiceInfo(voice_id="", name="Rachel")

    def test_whitespace_only_voice_id_raises_error(self) -> None:
        """Test that whitespace-only voice_id raises ValueError."""
        with pytest.raises(ValueError, match="voice_id cannot be empty"):
            VoiceInfo(voice_id="   ", name="Rachel")

    def test_empty_name_raises_error(self) -> None:
        """Test that empty name raises ValueError."""
        with pytest.raises(ValueError, match="name cannot be empty"):
            VoiceInfo(voice_id="voice_123", name="")

    def test_whitespace_only_name_raises_error(self) -> None:
        """Test that whitespace-only name raises ValueError."""
        with pytest.raises(ValueError, match="name cannot be empty"):
            VoiceInfo(voice_id="voice_123", name="   ")


class TestVoiceSettings:
    """Test VoiceSettings validation logic."""

    def test_default_voice_settings(self) -> None:
        """Test creating VoiceSettings with default values."""
        settings = VoiceSettings()

        assert settings.stability == 0.79
        assert settings.similarity_boost == 0.85
        assert settings.style == 0.25
        assert settings.use_speaker_boost is True
        assert settings.speaking_rate == 0.79

    def test_valid_custom_settings(self) -> None:
        """Test creating VoiceSettings with valid custom values."""
        settings = VoiceSettings(
            stability=0.5,
            similarity_boost=0.6,
            style=0.4,
            use_speaker_boost=False,
            speaking_rate=1.2,
        )

        assert settings.stability == 0.5
        assert settings.similarity_boost == 0.6
        assert settings.style == 0.4
        assert settings.use_speaker_boost is False
        assert settings.speaking_rate == 1.2

    def test_stability_boundary_values(self) -> None:
        """Test stability accepts boundary values 0.0 and 1.0."""
        # Test minimum boundary
        settings_min = VoiceSettings(stability=0.0)
        assert settings_min.stability == 0.0

        # Test maximum boundary
        settings_max = VoiceSettings(stability=1.0)
        assert settings_max.stability == 1.0

    def test_stability_below_range_raises_error(self) -> None:
        """Test that stability below 0.0 raises ValueError."""
        with pytest.raises(ValueError, match="stability must be between 0.0 and 1.0"):
            VoiceSettings(stability=-0.1)

    def test_stability_above_range_raises_error(self) -> None:
        """Test that stability above 1.0 raises ValueError."""
        with pytest.raises(ValueError, match="stability must be between 0.0 and 1.0"):
            VoiceSettings(stability=1.1)

    def test_similarity_boost_boundary_values(self) -> None:
        """Test similarity_boost accepts boundary values 0.0 and 1.0."""
        # Test minimum boundary
        settings_min = VoiceSettings(similarity_boost=0.0)
        assert settings_min.similarity_boost == 0.0

        # Test maximum boundary
        settings_max = VoiceSettings(similarity_boost=1.0)
        assert settings_max.similarity_boost == 1.0

    def test_similarity_boost_below_range_raises_error(self) -> None:
        """Test that similarity_boost below 0.0 raises ValueError."""
        with pytest.raises(
            ValueError, match="similarity_boost must be between 0.0 and 1.0"
        ):
            VoiceSettings(similarity_boost=-0.1)

    def test_similarity_boost_above_range_raises_error(self) -> None:
        """Test that similarity_boost above 1.0 raises ValueError."""
        with pytest.raises(
            ValueError, match="similarity_boost must be between 0.0 and 1.0"
        ):
            VoiceSettings(similarity_boost=1.1)

    def test_style_boundary_values(self) -> None:
        """Test style accepts boundary values 0.0 and 1.0."""
        # Test minimum boundary
        settings_min = VoiceSettings(style=0.0)
        assert settings_min.style == 0.0

        # Test maximum boundary
        settings_max = VoiceSettings(style=1.0)
        assert settings_max.style == 1.0

    def test_style_below_range_raises_error(self) -> None:
        """Test that style below 0.0 raises ValueError."""
        with pytest.raises(ValueError, match="style must be between 0.0 and 1.0"):
            VoiceSettings(style=-0.1)

    def test_style_above_range_raises_error(self) -> None:
        """Test that style above 1.0 raises ValueError."""
        with pytest.raises(ValueError, match="style must be between 0.0 and 1.0"):
            VoiceSettings(style=1.1)

    def test_speaking_rate_boundary_values(self) -> None:
        """Test speaking_rate accepts boundary values 0.25 and 4.0."""
        # Test minimum boundary
        settings_min = VoiceSettings(speaking_rate=0.25)
        assert settings_min.speaking_rate == 0.25

        # Test maximum boundary
        settings_max = VoiceSettings(speaking_rate=4.0)
        assert settings_max.speaking_rate == 4.0

    def test_speaking_rate_below_range_raises_error(self) -> None:
        """Test that speaking_rate below 0.25 raises ValueError."""
        with pytest.raises(
            ValueError, match="speaking_rate must be between 0.25 and 4.0"
        ):
            VoiceSettings(speaking_rate=0.24)

    def test_speaking_rate_above_range_raises_error(self) -> None:
        """Test that speaking_rate above 4.0 raises ValueError."""
        with pytest.raises(
            ValueError, match="speaking_rate must be between 0.25 and 4.0"
        ):
            VoiceSettings(speaking_rate=4.1)
