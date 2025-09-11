"""Integration tests for core module with real API."""

import sys
import tempfile
from pathlib import Path

import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from lspeak.core import list_available_voices, speak_text
from lspeak.tts.errors import TTSAuthError


class TestListAvailableVoicesIntegration:
    """Test list_available_voices function with real ElevenLabs API."""

    def test_list_available_voices_prints_correct_format(self, capsys) -> None:
        """Test list_available_voices prints voices in 'Name: voice_id' format."""
        # Call the function which prints to stdout
        list_available_voices()

        # Capture the printed output
        captured = capsys.readouterr()

        # Verify output format
        output_lines = captured.out.strip().split("\n")
        assert len(output_lines) > 0, "Should print at least one voice"

        # Check each line follows the expected format
        for line in output_lines:
            assert ": " in line, f"Line should contain ': ' separator: {line}"
            parts = line.split(": ", 1)
            assert len(parts) == 2, f"Line should have name and ID: {line}"
            name, voice_id = parts
            assert name.strip(), f"Voice name should not be empty: {line}"
            assert voice_id.strip(), f"Voice ID should not be empty: {line}"
            assert len(voice_id) > 10, f"Voice ID should be substantial: {line}"

    def test_list_available_voices_handles_missing_api_key(self, monkeypatch) -> None:
        """Test list_available_voices raises clear error when API key missing."""
        # Remove API key from environment
        monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)

        # Should raise TTSAuthError with clear message
        with pytest.raises(TTSAuthError) as exc_info:
            list_available_voices()

        assert "ElevenLabs API key not found" in str(exc_info.value)

    def test_list_available_voices_output_matches_api_data(self, capsys) -> None:
        """Test that printed output matches actual API voice data."""
        from lspeak.tts.client import TTSClient

        # Get voices directly from API
        client = TTSClient()
        voices = client.list_voices()

        # Call list_available_voices
        list_available_voices()
        captured = capsys.readouterr()

        # Parse output
        output_lines = captured.out.strip().split("\n")

        # Should have same number of voices
        assert len(output_lines) == len(voices), "Output should have one line per voice"

        # Build expected output for comparison
        expected_lines = [f"{voice.name}: {voice.voice_id}" for voice in voices]

        # Compare actual vs expected
        for actual, expected in zip(sorted(output_lines), sorted(expected_lines), strict=False):
            assert actual == expected, f"Output mismatch: {actual} != {expected}"

    def test_list_available_voices_consistent_output(self, capsys) -> None:
        """Test that multiple calls produce consistent output."""
        # First call
        list_available_voices()
        first_output = capsys.readouterr().out

        # Second call (should use cached data)
        list_available_voices()
        second_output = capsys.readouterr().out

        # Output should be identical
        assert first_output == second_output, "Output should be consistent across calls"
        assert first_output.strip(), "Output should not be empty"


class TestSpeakTextIntegration:
    """Test speak_text function with real ElevenLabs API and pygame."""

    def test_speak_text_with_playback(self) -> None:
        """Test speak_text synthesizes and plays audio through speakers."""
        # This test will actually play sound through speakers
        # In CI/CD environments without audio, pygame may initialize but not play
        speak_text("Testing audio playback")

        # If we get here without exception, the function executed successfully
        # We can't easily verify audio was played without audio capture hardware
        # But we've verified the full integration path works

    def test_speak_text_saves_to_file(self) -> None:
        """Test speak_text saves audio to file when output_file provided."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_speech.mp3"

            # Generate and save audio
            speak_text("Hello from integration test", output_file=str(output_path))

            # Verify file was created
            assert output_path.exists(), "Audio file should be created"

            # Verify file has content (MP3 audio)
            audio_data = output_path.read_bytes()
            assert len(audio_data) > 1000, "Audio file should have substantial content"

            # Basic MP3 validation - files often start with ID3 tag or FF FB/FF FA
            assert audio_data[:3] == b"ID3" or audio_data[0] == 0xFF, (
                "Should be valid MP3 data"
            )

    def test_speak_text_with_specific_voice(self) -> None:
        """Test speak_text works with a specific voice ID."""
        from lspeak.tts.client import TTSClient

        # Get a real voice ID
        client = TTSClient()
        voices = client.list_voices()
        assert len(voices) > 0, "Need at least one voice for test"
        test_voice_id = voices[0].voice_id

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "voice_test.mp3"

            # Generate with specific voice
            speak_text(
                "Testing specific voice selection",
                voice_id=test_voice_id,
                output_file=str(output_path),
            )

            # Verify file was created with audio
            assert output_path.exists()
            assert output_path.stat().st_size > 1000, (
                "Should generate audio with specific voice"
            )

    def test_speak_text_different_text_lengths(self) -> None:
        """Test speak_text handles different text lengths appropriately."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test short text
            short_path = Path(temp_dir) / "short.mp3"
            speak_text("Hi", output_file=str(short_path))
            short_size = short_path.stat().st_size

            # Test longer text
            long_path = Path(temp_dir) / "long.mp3"
            speak_text(
                "This is a much longer text to test how the system handles more content",
                output_file=str(long_path),
            )
            long_size = long_path.stat().st_size

            # Verify both files created
            assert short_path.exists() and long_path.exists()
            assert short_size > 500, "Even short text should generate audio"
            assert long_size > short_size, "Longer text should generate larger file"

    def test_speak_text_handles_missing_api_key(self, monkeypatch) -> None:
        """Test speak_text raises clear error when API key missing."""
        # Remove API key from environment
        monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)

        # Should raise TTSAuthError with clear message
        with pytest.raises(TTSAuthError) as exc_info:
            speak_text("Test without API key")

        assert "ElevenLabs API key not found" in str(exc_info.value)

    def test_speak_text_empty_text_raises_error(self) -> None:
        """Test speak_text raises ValueError for empty text."""
        with pytest.raises(ValueError) as exc_info:
            speak_text("")

        assert "Text cannot be empty" in str(exc_info.value)

    def test_speak_text_creates_parent_directories(self) -> None:
        """Test speak_text creates parent directories for output file if needed."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create nested path that doesn't exist
            output_path = Path(temp_dir) / "nested" / "dirs" / "speech.mp3"

            # Parent directories should not exist
            assert not output_path.parent.exists()

            # Generate audio to nested path
            speak_text("Test nested directory creation", output_file=str(output_path))

            # Verify directories were created and file exists
            assert output_path.parent.exists(), "Parent directories should be created"
            assert output_path.exists(), "Audio file should be created"
            assert output_path.stat().st_size > 1000, "File should contain audio data"
