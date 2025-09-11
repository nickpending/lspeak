"""Unit tests for API module logic."""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from lspeak.api import speak


class TestSpeakParameterHandling:
    """Test speak function parameter processing logic."""

    @pytest.mark.asyncio
    @patch("lspeak.api.speak_text")
    async def test_speak_converts_path_to_string(
        self, mock_speak_text: AsyncMock
    ) -> None:
        """Test speak converts Path object to string for output parameter."""
        mock_speak_text.return_value = None

        # Call with Path object
        test_path = Path("/tmp/test.mp3")
        await speak(text="test", provider="elevenlabs", output=test_path)

        # Verify string conversion and default parameters
        mock_speak_text.assert_called_once_with(
            text="test",
            provider="elevenlabs",
            voice_id=None,
            output_file=str(test_path),
            cache=True,
            cache_threshold=0.95,
            use_daemon=True,
            queue=True,
            debug=False,
        )

    @pytest.mark.asyncio
    @patch("lspeak.api.speak_text")
    async def test_speak_preserves_string_output(
        self, mock_speak_text: AsyncMock
    ) -> None:
        """Test speak preserves string output parameter as-is."""
        mock_speak_text.return_value = None

        # Call with string
        output_str = "/tmp/test.mp3"
        await speak(text="test", provider="elevenlabs", output=output_str)

        # Verify string passed through with defaults
        mock_speak_text.assert_called_once_with(
            text="test",
            provider="elevenlabs",
            voice_id=None,
            output_file=output_str,
            cache=True,
            cache_threshold=0.95,
            use_daemon=True,
            queue=True,
            debug=False,
        )

    @pytest.mark.asyncio
    @patch("lspeak.api.speak_text")
    async def test_speak_handles_none_output(self, mock_speak_text: AsyncMock) -> None:
        """Test speak handles None output parameter correctly."""
        mock_speak_text.return_value = None

        # Call with None output
        await speak(text="test", provider="elevenlabs", output=None)

        # Verify None passed through with defaults
        mock_speak_text.assert_called_once_with(
            text="test",
            provider="elevenlabs",
            voice_id=None,
            output_file=None,
            cache=True,
            cache_threshold=0.95,
            use_daemon=True,
            queue=True,
            debug=False,
        )

    @pytest.mark.asyncio
    @patch("lspeak.api.speak_text")
    async def test_speak_maps_voice_parameter(self, mock_speak_text: AsyncMock) -> None:
        """Test speak maps voice parameter to voice_id in core function."""
        mock_speak_text.return_value = None

        # Call with voice parameter
        await speak(text="test", voice="Rachel")

        # Verify voice mapped to voice_id with defaults
        mock_speak_text.assert_called_once_with(
            text="test",
            provider="elevenlabs",
            voice_id="Rachel",
            output_file=None,
            cache=True,
            cache_threshold=0.95,
            use_daemon=True,
            queue=True,
            debug=False,
        )


class TestSpeakReturnValueLogic:
    """Test speak function return value logic."""

    @pytest.mark.asyncio
    @patch("lspeak.api.speak_text")
    async def test_speak_returns_none_when_no_output_file(
        self, mock_speak_text: AsyncMock
    ) -> None:
        """Test speak returns None when no output file specified (plays audio)."""
        mock_speak_text.return_value = None

        result = await speak("test", output=None)

        assert result is None

    @pytest.mark.asyncio
    @patch("lspeak.api.speak_text")
    @patch("pathlib.Path.read_bytes")
    @patch("pathlib.Path.exists")
    async def test_speak_returns_bytes_when_output_file_exists(
        self,
        mock_exists: MagicMock,
        mock_read_bytes: MagicMock,
        mock_speak_text: AsyncMock,
    ) -> None:
        """Test speak returns audio bytes when output file exists."""
        mock_speak_text.return_value = None
        mock_exists.return_value = True
        mock_read_bytes.return_value = b"fake audio data"

        result = await speak("test", output="/tmp/test.mp3")

        assert result == b"fake audio data"
        mock_read_bytes.assert_called_once()

    @pytest.mark.asyncio
    @patch("lspeak.api.speak_text")
    @patch("pathlib.Path.exists")
    async def test_speak_returns_none_when_output_file_missing(
        self, mock_exists: MagicMock, mock_speak_text: AsyncMock
    ) -> None:
        """Test speak returns None when output file doesn't exist (creation failed)."""
        mock_speak_text.return_value = None
        mock_exists.return_value = False

        result = await speak("test", output="/tmp/nonexistent.mp3")

        assert result is None

    @pytest.mark.asyncio
    @patch("lspeak.api.speak_text")
    @patch("pathlib.Path.read_bytes")
    @patch("pathlib.Path.exists")
    async def test_speak_handles_path_object_for_file_reading(
        self,
        mock_exists: MagicMock,
        mock_read_bytes: MagicMock,
        mock_speak_text: AsyncMock,
    ) -> None:
        """Test speak correctly handles Path object for file reading."""
        mock_speak_text.return_value = None
        mock_exists.return_value = True
        mock_read_bytes.return_value = b"audio bytes"

        test_path = Path("/tmp/test.mp3")
        result = await speak("test", output=test_path)

        assert result == b"audio bytes"
        # Verify Path object was used for reading
        mock_exists.assert_called_once()
        mock_read_bytes.assert_called_once()


class TestSpeakParameterDefaults:
    """Test speak function parameter defaults."""

    @pytest.mark.asyncio
    @patch("lspeak.api.speak_text")
    async def test_speak_uses_default_provider(
        self, mock_speak_text: AsyncMock
    ) -> None:
        """Test speak uses 'elevenlabs' as default provider."""
        mock_speak_text.return_value = None

        await speak("test")

        # Verify default provider and other defaults used
        mock_speak_text.assert_called_once_with(
            text="test",
            provider="elevenlabs",
            voice_id=None,
            output_file=None,
            cache=True,
            cache_threshold=0.95,
            use_daemon=True,
            queue=True,
            debug=False,
        )

    @pytest.mark.asyncio
    @patch("lspeak.api.speak_text")
    async def test_speak_cache_parameters_accepted(
        self, mock_speak_text: AsyncMock
    ) -> None:
        """Test speak accepts cache parameters even though not implemented yet."""
        mock_speak_text.return_value = None

        # Should not raise error
        await speak("test", cache=True, cache_threshold=0.8)

        # Function should complete successfully
        mock_speak_text.assert_called_once()
