"""Integration tests for AudioPlayer with real pygame and file I/O."""

import sys
import tempfile
from pathlib import Path

import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from lspeak.audio.player import AudioPlayer


class TestAudioPlayerRealIntegration:
    """Test AudioPlayer with real pygame and file system."""

    def test_audio_player_initializes_with_real_pygame(self) -> None:
        """Test AudioPlayer initializes with real pygame mixer."""
        # This uses real pygame initialization
        player = AudioPlayer()

        # If we get here without exception, pygame initialized successfully
        assert player is not None

    def test_save_to_file_creates_real_file(self) -> None:
        """Test save_to_file creates actual file on disk."""
        player = AudioPlayer()
        test_audio = b"This is test audio data"

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_output.mp3"

            # Save audio to real file
            player.save_to_file(test_audio, output_path)

            # Verify file exists and contains correct data
            assert output_path.exists()
            assert output_path.read_bytes() == test_audio

    def test_save_to_file_creates_parent_directories(self) -> None:
        """Test save_to_file creates parent directories if they don't exist."""
        player = AudioPlayer()
        test_audio = b"Audio data for nested directory"

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a nested path that doesn't exist
            output_path = Path(temp_dir) / "nested" / "dirs" / "audio.mp3"

            # Parent directories should not exist yet
            assert not output_path.parent.exists()

            # Save file - should create all parent directories
            player.save_to_file(test_audio, output_path)

            # Verify directories were created and file saved
            assert output_path.parent.exists()
            assert output_path.exists()
            assert output_path.read_bytes() == test_audio

    def test_save_to_file_handles_permission_errors(self) -> None:
        """Test save_to_file raises appropriate error for permission issues."""
        player = AudioPlayer()
        test_audio = b"Test audio"

        # Try to save to a path we can't write to
        # Note: This test might not work on all systems, particularly Windows
        import platform

        if platform.system() != "Windows":
            with pytest.raises(OSError, match="Failed to save audio"):
                player.save_to_file(test_audio, "/root/test.mp3")

    def test_play_bytes_with_invalid_audio_data(self) -> None:
        """Test play_bytes handles invalid audio data gracefully."""
        player = AudioPlayer()

        # pygame should raise an error for invalid audio data
        with pytest.raises(RuntimeError, match="Failed to play audio"):
            player.play_bytes(b"This is not valid audio data")

    def test_complete_audio_workflow(self) -> None:
        """Test complete workflow: initialize, save, and verify."""
        # Initialize player
        player = AudioPlayer()

        # Create some test data
        test_audio = b"Complete workflow test audio data"

        with tempfile.TemporaryDirectory() as temp_dir:
            # Define output path
            output_path = Path(temp_dir) / "workflow_test.mp3"

            # Save audio
            player.save_to_file(test_audio, output_path)

            # Verify saved correctly
            assert output_path.exists()
            saved_data = output_path.read_bytes()
            assert saved_data == test_audio
            assert len(saved_data) == len(test_audio)

    def test_multiple_audio_players_can_coexist(self) -> None:
        """Test multiple AudioPlayer instances can be created."""
        # Create multiple players - each initializes pygame
        player1 = AudioPlayer()
        player2 = AudioPlayer()

        # Both should work independently
        test_audio = b"Multi-player test"

        with tempfile.TemporaryDirectory() as temp_dir:
            path1 = Path(temp_dir) / "player1.mp3"
            path2 = Path(temp_dir) / "player2.mp3"

            player1.save_to_file(test_audio, path1)
            player2.save_to_file(test_audio, path2)

            assert path1.exists()
            assert path2.exists()
            assert path1.read_bytes() == path2.read_bytes()
