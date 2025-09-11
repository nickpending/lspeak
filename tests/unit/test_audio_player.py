"""Unit tests for AudioPlayer validation and error handling logic."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from lspeak.audio.player import AudioPlayer


class TestAudioPlayerValidation:
    """Test AudioPlayer validation logic."""

    def test_play_bytes_with_empty_data_raises_value_error(self) -> None:
        """Test play_bytes raises ValueError when audio data is empty."""
        with patch("lspeak.audio.player.pygame.mixer.init"):
            player = AudioPlayer()

            with pytest.raises(ValueError, match="No audio data provided"):
                player.play_bytes(b"")

    def test_save_to_file_with_empty_data_raises_value_error(self) -> None:
        """Test save_to_file raises ValueError when audio data is empty."""
        with patch("lspeak.audio.player.pygame.mixer.init"):
            player = AudioPlayer()

            with pytest.raises(ValueError, match="No audio data provided"):
                player.save_to_file(b"", "output.mp3")

    def test_save_to_file_converts_string_to_path(self) -> None:
        """Test save_to_file handles string filepath correctly."""
        with patch("lspeak.audio.player.pygame.mixer.init"):
            player = AudioPlayer()

            # Mock Path operations
            with patch("lspeak.audio.player.Path") as mock_path_class:
                mock_path = MagicMock()
                mock_path.parent.mkdir = MagicMock()
                mock_path.write_bytes = MagicMock()
                mock_path_class.return_value = mock_path

                player.save_to_file(b"test data", "output.mp3")

                # Verify string was converted to Path
                mock_path_class.assert_called_once_with("output.mp3")
                mock_path.write_bytes.assert_called_once_with(b"test data")

    def test_save_to_file_accepts_path_object(self) -> None:
        """Test save_to_file works with Path object directly."""
        with patch("lspeak.audio.player.pygame.mixer.init"):
            player = AudioPlayer()

            # Mock the pathlib.Path class methods
            with patch("pathlib.Path.mkdir") as mock_mkdir:
                with patch("pathlib.Path.write_bytes") as mock_write:
                    test_path = Path("output.mp3")
                    player.save_to_file(b"test data", test_path)

                    # Verify parent directory creation was attempted
                    mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
                    # Verify data was written
                    mock_write.assert_called_once_with(b"test data")


class TestAudioPlayerErrorHandling:
    """Test AudioPlayer error handling."""

    def test_initialization_pygame_error_raises_runtime_error(self) -> None:
        """Test initialization converts pygame.error to RuntimeError."""
        import pygame

        with patch("lspeak.audio.player.pygame.mixer.init") as mock_init:
            mock_init.side_effect = pygame.error("Audio system unavailable")

            with pytest.raises(
                RuntimeError, match="Failed to initialize pygame audio mixer"
            ):
                AudioPlayer()

    def test_play_bytes_pygame_error_raises_runtime_error(self) -> None:
        """Test play_bytes converts pygame.error to RuntimeError."""
        import pygame

        with patch("lspeak.audio.player.pygame.mixer.init"):
            player = AudioPlayer()

            with patch("lspeak.audio.player.pygame.mixer.music.load") as mock_load:
                mock_load.side_effect = pygame.error("Invalid audio format")

                with pytest.raises(RuntimeError, match="Failed to play audio"):
                    player.play_bytes(b"invalid audio data")

    def test_save_to_file_os_error_handling(self) -> None:
        """Test save_to_file handles OSError correctly."""
        with patch("lspeak.audio.player.pygame.mixer.init"):
            player = AudioPlayer()

            with patch("lspeak.audio.player.Path") as mock_path_class:
                mock_path = MagicMock()
                mock_path.parent.mkdir = MagicMock()
                mock_path.write_bytes.side_effect = OSError("Permission denied")
                mock_path_class.return_value = mock_path

                with pytest.raises(OSError, match="Failed to save audio"):
                    player.save_to_file(b"test data", "output.mp3")

    def test_play_bytes_waits_for_completion(self) -> None:
        """Test play_bytes waits for audio to finish playing."""
        with patch("lspeak.audio.player.pygame.mixer.init"):
            player = AudioPlayer()

            # Mock the pygame components
            with patch("lspeak.audio.player.io.BytesIO") as mock_bytesio:
                with patch("lspeak.audio.player.pygame.mixer.music.load") as mock_load:
                    with patch(
                        "lspeak.audio.player.pygame.mixer.music.play"
                    ) as mock_play:
                        with patch(
                            "lspeak.audio.player.pygame.mixer.music.get_busy"
                        ) as mock_busy:
                            with patch(
                                "lspeak.audio.player.pygame.time.Clock"
                            ) as mock_clock_class:
                                # Simulate audio playing then stopping
                                mock_busy.side_effect = [True, True, False]
                                mock_clock = MagicMock()
                                mock_clock_class.return_value = mock_clock

                                player.play_bytes(b"test audio")

                                # Verify playback started
                                mock_play.assert_called_once()
                                # Verify we waited (tick called twice for True, True)
                                assert mock_clock.tick.call_count == 2
