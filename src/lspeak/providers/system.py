"""System TTS provider using native OS text-to-speech commands.

This module provides text-to-speech functionality using the built-in
TTS capabilities of the operating system (say on macOS, espeak on Linux,
SAPI on Windows).
"""

import asyncio
import logging
import platform
import subprocess
import tempfile
from pathlib import Path

from .base import TTSProvider

logger = logging.getLogger(__name__)


class SystemTTSProvider(TTSProvider):
    """System TTS provider using native OS commands.

    Provides text-to-speech functionality without requiring external APIs,
    using the built-in TTS capabilities of the operating system.

    Note: Audio quality will be robotic compared to AI-powered voices.
    """

    def __init__(self) -> None:
        """Initialize system TTS provider and detect platform."""
        self.platform = platform.system()

        # Log quality warning
        logger.warning(
            "Using system TTS - quality will be robotic compared to AI voices. "
            "For better quality, set ELEVENLABS_API_KEY and use --provider=elevenlabs"
        )

        # Validate platform is supported
        if self.platform not in ["Darwin", "Linux", "Windows"]:
            raise RuntimeError(f"Unsupported platform: {self.platform}")

    async def synthesize(self, text: str, voice: str | None = None) -> bytes:
        """Convert text to speech using native OS commands.

        Args:
            text: Text to convert to speech
            voice: Optional voice ID/name (platform-specific)

        Returns:
            Audio data as bytes in WAV or AIFF format

        Raises:
            RuntimeError: If TTS command fails
        """
        # Create temporary file for audio output
        # For macOS, we'll generate AIFF then convert to WAV for compatibility
        if self.platform == "Darwin":
            aiff_file = tempfile.NamedTemporaryFile(suffix=".aiff", delete=False)
            aiff_path = aiff_file.name
            aiff_file.close()

            wav_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            output_path = wav_file.name
            wav_file.close()
        else:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                output_path = tmp.name

        try:
            if self.platform == "Darwin":  # macOS
                # First generate AIFF
                cmd = ["say", "-o", aiff_path]
                if voice:
                    cmd.extend(["-v", voice])
                cmd.append(text)

                # Use async subprocess to avoid blocking event loop
                proc = await asyncio.create_subprocess_exec(
                    *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
                )
                stdout, stderr = await proc.communicate()

                if proc.returncode != 0:
                    raise RuntimeError(
                        f"System TTS failed with code {proc.returncode}: {stderr.decode()}"
                    )

                # Convert AIFF to WAV using afconvert
                convert_cmd = [
                    "afconvert",
                    "-f",
                    "WAVE",
                    "-d",
                    "LEI16",
                    aiff_path,
                    output_path,
                ]
                # Use async subprocess for conversion too
                proc = await asyncio.create_subprocess_exec(
                    *convert_cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, stderr = await proc.communicate()

                if proc.returncode != 0:
                    raise RuntimeError(
                        f"Audio conversion failed with code {proc.returncode}: {stderr.decode()}"
                    )

                # Clean up AIFF file
                Path(aiff_path).unlink(missing_ok=True)

            elif self.platform == "Linux":
                # Check if espeak is available (async)
                proc = await asyncio.create_subprocess_exec(
                    "which",
                    "espeak",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                await proc.communicate()

                if proc.returncode != 0:
                    raise RuntimeError(
                        "espeak not found. Install it with: sudo apt-get install espeak"
                    )

                cmd = ["espeak", "-w", output_path]
                if voice:
                    cmd.extend(["-v", voice])
                cmd.append(text)

                # Use async subprocess for espeak
                proc = await asyncio.create_subprocess_exec(
                    *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
                )
                stdout, stderr = await proc.communicate()

                if proc.returncode != 0:
                    raise RuntimeError(
                        f"System TTS failed with code {proc.returncode}: {stderr.decode()}"
                    )

            else:  # Windows
                # Use PowerShell with SAPI
                ps_script = f'''
                Add-Type -AssemblyName System.Speech
                $speak = New-Object System.Speech.Synthesis.SpeechSynthesizer
                $speak.SetOutputToWaveFile("{output_path}")
                '''
                if voice:
                    ps_script += f'$speak.SelectVoice("{voice}")\n'
                ps_script += f'$speak.Speak("{text}")\n$speak.Dispose()'

                cmd = ["powershell", "-Command", ps_script]

                # Use async subprocess for PowerShell
                proc = await asyncio.create_subprocess_exec(
                    *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
                )
                stdout, stderr = await proc.communicate()

                if proc.returncode != 0:
                    raise RuntimeError(
                        f"System TTS failed with code {proc.returncode}: {stderr.decode()}"
                    )

            # Read the generated audio file
            with open(output_path, "rb") as f:
                audio_data = f.read()

            return audio_data

        finally:
            # Clean up temporary file
            Path(output_path).unlink(missing_ok=True)

    async def list_voices(self) -> list[dict]:
        """List available system voices.

        Returns:
            List of voice dictionaries with id, name, and provider fields

        Raises:
            RuntimeError: If voice listing fails
        """
        voices = []

        if self.platform == "Darwin":  # macOS
            try:
                result = subprocess.run(
                    ["say", "-v", "?"], capture_output=True, text=True, check=True
                )

                # Parse macOS voice output
                # Format: "Voice Name     Language  # Description"
                for line in result.stdout.strip().split("\n"):
                    if line and not line.startswith("#"):
                        # Extract voice name (first field)
                        parts = line.split()
                        if parts:
                            voice_name = parts[0]
                            voices.append(
                                {
                                    "id": voice_name,
                                    "name": voice_name,
                                    "provider": "system",
                                }
                            )

            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to list macOS voices: {e}")

        elif self.platform == "Linux":
            try:
                result = subprocess.run(
                    ["espeak", "--voices"], capture_output=True, text=True, check=True
                )

                # Parse espeak voice output
                # Skip the header line
                lines = result.stdout.strip().split("\n")
                if lines:
                    for line in lines[1:]:  # Skip header
                        if line:
                            parts = line.split()
                            if len(parts) >= 2:
                                # Voice ID is in the second column
                                voice_id = parts[1]
                                voices.append(
                                    {
                                        "id": voice_id,
                                        "name": voice_id,
                                        "provider": "system",
                                    }
                                )

            except subprocess.CalledProcessError:
                logger.warning("espeak not found - no voices available")

        elif self.platform == "Windows":
            # Use PowerShell to list SAPI voices
            ps_script = """
            Add-Type -AssemblyName System.Speech
            $speak = New-Object System.Speech.Synthesis.SpeechSynthesizer
            $speak.GetInstalledVoices() | ForEach-Object {
                $_.VoiceInfo.Name
            }
            """

            try:
                result = subprocess.run(
                    ["powershell", "-Command", ps_script],
                    capture_output=True,
                    text=True,
                    check=True,
                )

                for line in result.stdout.strip().split("\n"):
                    if line:
                        voices.append({"id": line, "name": line, "provider": "system"})

            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to list Windows voices: {e}")

        # If no voices found, add a default
        if not voices:
            voices.append(
                {"id": "default", "name": "Default System Voice", "provider": "system"}
            )

        return voices
