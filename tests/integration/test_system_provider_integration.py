"""Integration tests for SystemTTSProvider with real OS commands."""

import platform
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from lspeak.core import list_available_voices, speak_text
from lspeak.providers import ProviderRegistry
from lspeak.providers.base import TTSProvider
from lspeak.providers.system import SystemTTSProvider


class TestSystemProviderRealTTSIntegration:
    """Test SystemTTSProvider with real OS TTS commands."""

    @pytest.mark.asyncio
    async def test_system_provider_initializes_successfully(self) -> None:
        """Test SystemTTSProvider initializes on supported platforms."""
        provider = SystemTTSProvider()

        # Verify provider initialized with correct platform
        assert provider.platform in ["Darwin", "Linux", "Windows"]
        assert isinstance(provider, TTSProvider)

    @pytest.mark.asyncio
    async def test_list_voices_returns_real_system_voices(self) -> None:
        """Test list_voices returns actual system voices."""
        provider = SystemTTSProvider()

        voices = await provider.list_voices()

        # Should have at least one voice (or default)
        assert len(voices) > 0, "Should return at least one voice"

        # Check first voice has required fields
        first_voice = voices[0]
        assert isinstance(first_voice, dict)
        assert "id" in first_voice
        assert "name" in first_voice
        assert "provider" in first_voice
        assert first_voice["provider"] == "system"
        assert first_voice["id"], "Voice ID should not be empty"
        assert first_voice["name"], "Voice name should not be empty"

    @pytest.mark.asyncio
    async def test_synthesize_generates_real_audio(self) -> None:
        """Test synthesize generates actual audio with OS commands."""
        provider = SystemTTSProvider()

        # Get a voice to use (or use default)
        voices = await provider.list_voices()
        test_voice = voices[0]["id"] if voices[0]["id"] != "default" else None

        # Generate audio
        audio_bytes = await provider.synthesize("Hello from system TTS", test_voice)

        # Verify we got actual audio data
        assert isinstance(audio_bytes, bytes)
        assert len(audio_bytes) > 1000, "Audio file should be substantial (>1KB)"

        # Check for audio file headers
        if platform.system() == "Darwin":
            # WAV file header after conversion
            assert audio_bytes[:4] == b"RIFF", "Should be WAV format on macOS"
        elif platform.system() == "Linux":
            # WAV file from espeak
            assert audio_bytes[:4] == b"RIFF", "Should be WAV format on Linux"
        elif platform.system() == "Windows":
            # WAV file from SAPI
            assert audio_bytes[:4] == b"RIFF", "Should be WAV format on Windows"

    @pytest.mark.asyncio
    async def test_synthesize_with_empty_voice_works(self) -> None:
        """Test synthesize works without specifying a voice."""
        provider = SystemTTSProvider()

        # Generate audio without voice (uses default)
        audio_bytes = await provider.synthesize("Testing default voice")

        # Verify audio was generated
        assert isinstance(audio_bytes, bytes)
        assert len(audio_bytes) > 1000, "Should generate audio with default voice"

    @pytest.mark.asyncio
    async def test_synthesize_different_text_lengths(self) -> None:
        """Test synthesize handles different text lengths."""
        provider = SystemTTSProvider()

        # Test short text
        short_audio = await provider.synthesize("Hi")
        assert len(short_audio) > 500, "Even short text should generate some audio"

        # Test longer text
        long_text = "This is a longer sentence to test how the system text-to-speech handles more substantial content."
        long_audio = await provider.synthesize(long_text)
        assert len(long_audio) > len(short_audio), (
            "Longer text should generate more audio"
        )

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        platform.system() != "Linux", reason="Linux-specific test for missing espeak"
    )
    async def test_linux_espeak_not_installed_error(self) -> None:
        """Test helpful error when espeak not installed on Linux."""
        # Check if espeak is actually missing
        try:
            subprocess.run(["which", "espeak"], check=True, capture_output=True)
            pytest.skip("espeak is installed, cannot test missing espeak error")
        except subprocess.CalledProcessError:
            # espeak is missing, test should work
            pass

        provider = SystemTTSProvider()

        with pytest.raises(
            RuntimeError,
            match="espeak not found. Install it with: sudo apt-get install espeak",
        ):
            await provider.synthesize("Test")


class TestSystemProviderRegistryIntegration:
    """Test system provider integration with provider registry."""

    def test_system_provider_is_registered(self) -> None:
        """Test that system provider is registered in the registry."""
        # Verify provider is registered
        assert "system" in ProviderRegistry._providers

        # Get provider class
        provider_class = ProviderRegistry.get("system")
        assert provider_class is SystemTTSProvider

        # Verify it's a TTSProvider
        assert issubclass(provider_class, TTSProvider)

    def test_get_and_instantiate_system_provider(self) -> None:
        """Test getting and instantiating system provider from registry."""
        # Get provider class from registry
        provider_class = ProviderRegistry.get("system")

        # Instantiate it
        provider = provider_class()

        # Verify it's the right type
        assert isinstance(provider, SystemTTSProvider)
        assert isinstance(provider, TTSProvider)

    def test_registry_lists_system_provider(self) -> None:
        """Test registry lists system as available provider."""
        # Get available providers
        available = list(ProviderRegistry._providers.keys())

        # Should include both providers
        assert "system" in available
        assert "elevenlabs" in available


class TestCoreIntegrationWithSystemProvider:
    """Test core.py integration with system provider."""

    @pytest.mark.asyncio
    async def test_list_available_voices_with_system_provider(self) -> None:
        """Test list_available_voices works with system provider."""
        # Capture output
        import io
        import sys

        captured_output = io.StringIO()
        sys.stdout = captured_output

        try:
            # List voices using system provider
            await list_available_voices(provider="system")

            # Get output
            output = captured_output.getvalue()

            # Verify output format
            assert output, "Should have output"
            lines = output.strip().split("\n")
            assert len(lines) > 0, "Should have voice lines"

            # Each line should be "Name: voice_id" or just the default
            first_line = lines[0]
            if ": " in first_line:
                name, voice_id = first_line.split(": ", 1)
                assert name, "Should have voice name"
                assert voice_id, "Should have voice ID"
            else:
                # Might just be default voice
                assert "Default" in first_line or len(lines) > 0

        finally:
            sys.stdout = sys.__stdout__

    @pytest.mark.asyncio
    async def test_speak_text_with_system_provider(self) -> None:
        """Test speak_text works with system provider."""
        # Test with file output to verify audio generation
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            output_file = tmp.name

        try:
            # Synthesize audio using system provider
            await speak_text(
                "Testing system provider integration",
                provider="system",
                output_file=output_file,
            )

            # Verify file was created with content
            output_path = Path(output_file)
            assert output_path.exists(), "Output file should exist"
            assert output_path.stat().st_size > 1000, "Audio file should have content"

            # Verify it's a valid audio file
            with open(output_file, "rb") as f:
                header = f.read(4)
                assert header == b"RIFF", "Should be WAV format"

        finally:
            # Clean up
            Path(output_file).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_speak_text_with_system_voice_selection(self) -> None:
        """Test speak_text works with specific system voice."""
        # Get available voices first
        provider = SystemTTSProvider()
        voices = await provider.list_voices()

        # Skip if only default voice available
        if len(voices) == 1 and voices[0]["id"] == "default":
            pytest.skip("No specific voices available on this system")

        test_voice_id = voices[0]["id"]

        # Test with file output
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            output_file = tmp.name

        try:
            # Synthesize audio with specific voice
            await speak_text(
                "Testing specific system voice",
                provider="system",
                voice_id=test_voice_id,
                output_file=output_file,
            )

            # Verify file was created
            output_path = Path(output_file)
            assert output_path.exists(), "Output file should exist"
            assert output_path.stat().st_size > 1000, "Audio file should have content"

        finally:
            # Clean up
            Path(output_file).unlink(missing_ok=True)


class TestCompleteSystemProviderWorkflow:
    """Test complete system provider workflow integration."""

    @pytest.mark.asyncio
    async def test_complete_system_provider_workflow(self) -> None:
        """
        Test complete system provider workflow:
        - Get provider from registry
        - List available voices
        - Generate audio with a voice
        - Use through core API
        - Verify no API key needed
        """
        # Ensure no ElevenLabs API key is set
        import os

        original_key = os.environ.pop("ELEVENLABS_API_KEY", None)

        try:
            # Step 1: Get provider from registry
            provider_class = ProviderRegistry.get("system")
            provider = provider_class()
            assert isinstance(provider, TTSProvider)

            # Step 2: List voices
            voices = await provider.list_voices()
            assert len(voices) > 0, "Should have available voices"

            # Step 3: Select a voice
            test_voice = voices[0]
            assert test_voice["id"]
            assert test_voice["name"]
            assert test_voice["provider"] == "system"

            # Step 4: Generate audio directly
            audio_data = await provider.synthesize(
                "This is a complete test of the system provider workflow",
                test_voice["id"] if test_voice["id"] != "default" else None,
            )
            assert isinstance(audio_data, bytes)
            assert len(audio_data) > 2000, "Should generate substantial audio"

            # Step 5: Use through core API
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                output_file = tmp.name

            try:
                await speak_text(
                    "Testing system provider through core API",
                    provider="system",
                    voice_id=test_voice["id"]
                    if test_voice["id"] != "default"
                    else None,
                    output_file=output_file,
                )

                # Verify file exists and has content
                assert Path(output_file).exists()
                assert Path(output_file).stat().st_size > 1000

            finally:
                Path(output_file).unlink(missing_ok=True)

        finally:
            # Restore API key if it was set
            if original_key:
                os.environ["ELEVENLABS_API_KEY"] = original_key

    @pytest.mark.asyncio
    async def test_system_provider_platform_specific_features(self) -> None:
        """Test platform-specific features work correctly."""
        provider = SystemTTSProvider()
        current_platform = platform.system()

        if current_platform == "Darwin":
            # macOS should have many voices
            voices = await provider.list_voices()
            # macOS typically has 50+ voices
            assert len(voices) > 10, "macOS should have many system voices"

            # Test AIFF to WAV conversion works
            audio = await provider.synthesize("Testing macOS audio conversion")
            assert audio[:4] == b"RIFF", "Should convert AIFF to WAV"

        elif current_platform == "Linux":
            # Linux with espeak
            voices = await provider.list_voices()
            # espeak has language codes as voices
            if len(voices) > 1:  # If espeak is installed
                assert any(v["id"] in ["en", "en-us", "en-gb"] for v in voices), (
                    "Should have English voice options"
                )

        elif current_platform == "Windows":
            # Windows SAPI voices
            voices = await provider.list_voices()
            if len(voices) > 1:  # If not just default
                assert any("Microsoft" in v["id"] for v in voices), (
                    "Windows should have Microsoft voices"
                )

    @pytest.mark.asyncio
    async def test_provider_selection_without_api_key(self) -> None:
        """Test that system provider works when no API key is available."""
        # Remove API key temporarily
        import os

        original_key = os.environ.pop("ELEVENLABS_API_KEY", None)

        try:
            # System provider should work without API key
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                output_file = tmp.name

            try:
                await speak_text(
                    "System TTS works without API key",
                    provider="system",
                    output_file=output_file,
                )

                # Verify file was created
                assert Path(output_file).exists()
                assert Path(output_file).stat().st_size > 1000

            finally:
                Path(output_file).unlink(missing_ok=True)

        finally:
            # Restore API key
            if original_key:
                os.environ["ELEVENLABS_API_KEY"] = original_key
