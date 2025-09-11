"""Integration tests for API module with real services."""

import sys
import tempfile
from pathlib import Path

import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from lspeak import speak
from lspeak.tts.errors import TTSAuthError


class TestSpeakFunctionIntegration:
    """Test speak function with real TTS API and file system."""

    @pytest.mark.asyncio
    async def test_speak_function_complete_workflow_with_file_output(self) -> None:
        """Test complete library workflow: import, call, get bytes."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "library_test.mp3"

            # Test complete library workflow
            audio_bytes = await speak(
                "Testing library interface integration",
                provider="elevenlabs",
                output=output_path,
            )

            # Verify file was created
            assert output_path.exists(), "Audio file should be created"

            # Verify audio bytes returned
            assert audio_bytes is not None, "Should return audio bytes"
            assert len(audio_bytes) > 1000, "Audio should have substantial content"

            # Verify file content matches returned bytes
            file_content = output_path.read_bytes()
            assert audio_bytes == file_content, (
                "Returned bytes should match file content"
            )

            # Basic MP3 validation
            assert audio_bytes[:3] == b"ID3" or audio_bytes[0] == 0xFF, (
                "Should be valid MP3 data"
            )

    @pytest.mark.asyncio
    async def test_speak_function_with_string_output_path(self) -> None:
        """Test speak function works with string output path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = str(Path(temp_dir) / "string_path_test.mp3")

            # Call with string path
            audio_bytes = await speak(
                "Testing string path", provider="elevenlabs", output=output_path
            )

            # Verify results
            assert Path(output_path).exists(), "File should be created"
            assert audio_bytes is not None, "Should return audio bytes"
            assert len(audio_bytes) > 1000, "Should contain audio data"

    @pytest.mark.asyncio
    async def test_speak_function_with_path_object_output(self) -> None:
        """Test speak function works with Path object output."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "path_object_test.mp3"

            # Call with Path object
            audio_bytes = await speak(
                "Testing Path object", provider="elevenlabs", output=output_path
            )

            # Verify results
            assert output_path.exists(), "File should be created"
            assert audio_bytes is not None, "Should return audio bytes"
            assert len(audio_bytes) > 1000, "Should contain audio data"

    @pytest.mark.asyncio
    async def test_speak_function_plays_audio_when_no_output(self) -> None:
        """Test speak function plays audio when no output file specified."""
        # This will actually attempt to play through speakers
        # In CI environments, pygame may not have audio but should not crash
        result = await speak(
            "Testing audio playback from library", provider="elevenlabs"
        )

        # When no output file, should return None
        assert result is None, "Should return None when playing audio"

    @pytest.mark.asyncio
    async def test_speak_function_with_specific_voice(self) -> None:
        """Test speak function works with specific voice selection."""
        from lspeak.tts.client import TTSClient

        # Get a real voice ID
        client = TTSClient()
        voices = client.list_voices()
        assert len(voices) > 0, "Need at least one voice for test"
        test_voice_id = voices[0].voice_id

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "voice_test.mp3"

            # Test with specific voice
            audio_bytes = await speak(
                "Testing specific voice selection",
                provider="elevenlabs",
                voice=test_voice_id,
                output=output_path,
            )

            # Verify file and bytes
            assert output_path.exists(), "Audio file should be created"
            assert audio_bytes is not None, "Should return audio bytes"
            assert len(audio_bytes) > 1000, "Should generate substantial audio"

    @pytest.mark.asyncio
    async def test_speak_function_cache_parameters_accepted(self) -> None:
        """Test speak function accepts cache parameters without error."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "cache_params_test.mp3"

            # Should not raise error even though cache not implemented
            audio_bytes = await speak(
                "Testing cache parameters",
                provider="elevenlabs",
                output=output_path,
                cache=True,
                cache_threshold=0.9,
            )

            # Verify function completed successfully
            assert output_path.exists(), "Audio file should be created"
            assert audio_bytes is not None, "Should return audio bytes"

    @pytest.mark.asyncio
    async def test_speak_function_creates_nested_directories(self) -> None:
        """Test speak function creates parent directories if needed."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create nested path that doesn't exist
            output_path = Path(temp_dir) / "nested" / "dirs" / "test.mp3"

            # Parent directories should not exist initially
            assert not output_path.parent.exists()

            # Call speak function
            audio_bytes = await speak(
                "Testing nested directory creation",
                provider="elevenlabs",
                output=output_path,
            )

            # Verify directories were created and file exists
            assert output_path.parent.exists(), "Parent directories should be created"
            assert output_path.exists(), "Audio file should be created"
            assert audio_bytes is not None, "Should return audio bytes"
            assert len(audio_bytes) > 1000, "File should contain audio data"

    @pytest.mark.asyncio
    async def test_speak_function_handles_missing_api_key(self, monkeypatch) -> None:
        """Test speak function raises clear error when API key missing."""
        # Remove API key from environment
        monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)

        # Should raise TTSAuthError with clear message
        with pytest.raises(TTSAuthError) as exc_info:
            await speak("Test without API key", provider="elevenlabs")

        assert "ElevenLabs API key not found" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_speak_function_different_text_lengths(self) -> None:
        """Test speak function handles different text lengths appropriately."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test short text
            short_path = Path(temp_dir) / "short.mp3"
            short_bytes = await speak("Hi", provider="elevenlabs", output=short_path)

            # Test longer text
            long_path = Path(temp_dir) / "long.mp3"
            long_bytes = await speak(
                "This is a much longer text to test how the library interface handles more content",
                provider="elevenlabs",
                output=long_path,
            )

            # Verify both work and longer text produces more audio
            assert short_bytes is not None and long_bytes is not None
            assert len(short_bytes) > 500, "Even short text should generate audio"
            assert len(long_bytes) > len(short_bytes), (
                "Longer text should generate more audio"
            )

            # Verify files exist
            assert short_path.exists() and long_path.exists()


class TestSpeakFunctionImport:
    """Test speak function import behavior."""

    def test_speak_can_be_imported_from_lspeak(self) -> None:
        """Test that speak function can be imported from main package."""
        # This import should work
        from lspeak import speak

        # Function should be available
        assert speak is not None
        assert callable(speak)
        assert speak.__name__ == "speak"

    def test_speak_function_has_correct_signature(self) -> None:
        """Test speak function has expected signature and type hints."""
        import inspect

        from lspeak import speak

        # Get function signature
        sig = inspect.signature(speak)

        # Check parameter names
        param_names = list(sig.parameters.keys())
        expected_params = [
            "text",
            "provider",
            "voice",
            "output",
            "cache",
            "cache_threshold",
        ]
        assert param_names == expected_params, (
            f"Expected {expected_params}, got {param_names}"
        )

        # Check all parameters have type annotations
        for param in sig.parameters.values():
            assert param.annotation != inspect.Parameter.empty, (
                f"Parameter {param.name} missing type hint"
            )

        # Check return type annotation
        assert sig.return_annotation != inspect.Signature.empty, (
            "Missing return type annotation"
        )

    def test_speak_function_is_async(self) -> None:
        """Test speak function is properly async."""
        import inspect

        from lspeak import speak

        assert inspect.iscoroutinefunction(speak), "speak function should be async"
