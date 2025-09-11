"""Integration tests for CLI execution."""

import os
import subprocess
from pathlib import Path

import pytest

# Project root for running tests
PROJECT_ROOT = Path(__file__).parent.parent.parent


def test_cli_executes_with_text_argument() -> None:
    """Test that CLI executes successfully with text argument."""
    cmd = ["uv", "run", "python", "-m", "lspeak", "Hello world"]

    result = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        env={"PYTHONPATH": "src", **os.environ},
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "Hello world" in result.stdout
    assert result.stderr == ""


def test_cli_shows_help() -> None:
    """Test that CLI shows help when --help flag used."""
    cmd = ["uv", "run", "python", "-m", "lspeak", "--help"]

    result = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        env={"PYTHONPATH": "src", **os.environ},
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "Convert text to speech using AI voices" in result.stdout
    assert "TEXT" in result.stdout


def test_cli_waits_for_stdin_when_no_arguments() -> None:
    """Test that CLI waits for stdin input when no arguments provided."""
    cmd = ["uv", "run", "python", "-m", "lspeak"]

    # With empty input, it should process successfully
    result = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        env={"PYTHONPATH": "src", **os.environ},
        input="",
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert result.stdout == "\n"


def test_cli_handles_multiline_text() -> None:
    """Test that CLI handles multiline text correctly."""
    multiline_text = "Line 1\\nLine 2\\nLine 3"
    cmd = ["uv", "run", "python", "-m", "lspeak", multiline_text]

    result = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        env={"PYTHONPATH": "src", **os.environ},
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert multiline_text in result.stdout


def test_cli_handles_special_characters() -> None:
    """Test that CLI handles special characters correctly."""
    special_text = "Hello! @#$%^&*()_+ 🎉"
    cmd = ["uv", "run", "python", "-m", "lspeak", special_text]

    result = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        env={"PYTHONPATH": "src", **os.environ},
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert special_text in result.stdout


def test_cli_reads_from_stdin() -> None:
    """Test that CLI reads text from stdin when no argument provided."""
    stdin_text = "Hello from stdin"
    cmd = ["uv", "run", "python", "-m", "lspeak"]

    result = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        env={"PYTHONPATH": "src", **os.environ},
        input=stdin_text,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert stdin_text in result.stdout


def test_cli_handles_multiline_stdin() -> None:
    """Test that CLI handles multiline text from stdin correctly."""
    multiline_text = "Line 1\nLine 2\nLine 3"
    cmd = ["uv", "run", "python", "-m", "lspeak"]

    result = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        env={"PYTHONPATH": "src", **os.environ},
        input=multiline_text,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "Line 1" in result.stdout
    assert "Line 2" in result.stdout
    assert "Line 3" in result.stdout


def test_cli_handles_empty_stdin() -> None:
    """Test that CLI handles empty stdin input."""
    cmd = ["uv", "run", "python", "-m", "lspeak"]

    result = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        env={"PYTHONPATH": "src", **os.environ},
        input="",
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert result.stdout == "\n"


def test_cli_reads_from_file() -> None:
    """Test that CLI reads text from file with -f option (Task 2.5 smoke test)."""
    # Create temporary test file
    test_file = PROJECT_ROOT / "test_file_input.txt"
    test_content = "Hello from test file"
    test_file.write_text(test_content)

    try:
        cmd = ["uv", "run", "python", "-m", "lspeak", "-f", str(test_file)]

        result = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            env={"PYTHONPATH": "src", **os.environ},
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert test_content in result.stdout
        assert result.stderr == ""
    finally:
        # Clean up test file
        if test_file.exists():
            test_file.unlink()


def test_cli_file_input_error_handling() -> None:
    """Test that CLI handles file input errors gracefully."""
    # Test 1: Missing file
    cmd = ["uv", "run", "python", "-m", "lspeak", "-f", "nonexistent_file.txt"]

    result = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        env={"PYTHONPATH": "src", **os.environ},
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "Error: File not found: nonexistent_file.txt" in result.stderr
    assert result.stdout == ""


def test_cli_file_input_precedence() -> None:
    """Test that file input takes precedence over stdin but not over text argument."""
    # Create test file
    test_file = PROJECT_ROOT / "test_precedence.txt"
    file_content = "Content from file"
    test_file.write_text(file_content)

    try:
        # Test 1: File takes precedence over stdin
        cmd = ["uv", "run", "python", "-m", "lspeak", "-f", str(test_file)]

        result = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            env={"PYTHONPATH": "src", **os.environ},
            input="Content from stdin",
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert file_content in result.stdout
        assert "Content from stdin" not in result.stdout

        # Test 2: Text argument takes precedence over file
        cmd = [
            "uv",
            "run",
            "python",
            "-m",
            "lspeak",
            "Content from argument",
            "-f",
            str(test_file),
        ]

        result = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            env={"PYTHONPATH": "src", **os.environ},
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert "Content from argument" in result.stdout
        assert file_content not in result.stdout

    finally:
        # Clean up test file
        if test_file.exists():
            test_file.unlink()


def test_cli_file_input_with_multiline_content() -> None:
    """Test that CLI correctly handles multiline file content."""
    # Create test file with multiline content
    test_file = PROJECT_ROOT / "test_multiline.txt"
    multiline_content = "Line 1\nLine 2\nLine 3"
    test_file.write_text(multiline_content)

    try:
        cmd = ["uv", "run", "python", "-m", "lspeak", "-f", str(test_file)]

        result = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            env={"PYTHONPATH": "src", **os.environ},
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert "Line 1" in result.stdout
        assert "Line 2" in result.stdout
        assert "Line 3" in result.stdout
        assert result.stderr == ""

    finally:
        # Clean up test file
        if test_file.exists():
            test_file.unlink()


def test_cli_output_file_functionality() -> None:
    """Test CLI output file option saves text correctly (Task 2.6)."""
    output_file = PROJECT_ROOT / "test_output_functionality.txt"
    test_content = "Hello output file test"

    try:
        # Test 1: Basic output file functionality
        cmd = [
            "uv",
            "run",
            "python",
            "-m",
            "lspeak",
            test_content,
            "-o",
            str(output_file),
        ]

        result = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            env={"PYTHONPATH": "src", **os.environ},
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert f"Output saved to {output_file}" in result.stdout
        assert result.stderr == ""

        # Verify file was created with correct content
        assert output_file.exists()
        assert output_file.read_text() == test_content

        # Test 2: Output file with stdin input
        stdin_content = "Content from stdin"
        cmd = ["uv", "run", "python", "-m", "lspeak", "-o", str(output_file)]

        result = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            env={"PYTHONPATH": "src", **os.environ},
            input=stdin_content,
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert f"Output saved to {output_file}" in result.stdout
        assert output_file.read_text() == stdin_content

        # Test 3: Output file with file input
        input_file = PROJECT_ROOT / "test_input_for_output.txt"
        file_content = "Content from input file"
        input_file.write_text(file_content)

        try:
            cmd = [
                "uv",
                "run",
                "python",
                "-m",
                "lspeak",
                "-f",
                str(input_file),
                "-o",
                str(output_file),
            ]

            result = subprocess.run(
                cmd,
                cwd=PROJECT_ROOT,
                env={"PYTHONPATH": "src", **os.environ},
                capture_output=True,
                text=True,
            )

            assert result.returncode == 0
            assert f"Output saved to {output_file}" in result.stdout
            assert output_file.read_text() == file_content

        finally:
            if input_file.exists():
                input_file.unlink()

    finally:
        if output_file.exists():
            output_file.unlink()


def test_cli_debug_error_handling() -> None:
    """Test CLI debug option shows enhanced error messages (Task 2.6)."""
    # Test 1: Debug with file not found error
    cmd = [
        "uv",
        "run",
        "python",
        "-m",
        "lspeak",
        "--debug",
        "-f",
        "absolutely_nonexistent_file.txt",
    ]

    result = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        env={"PYTHONPATH": "src", **os.environ},
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "Debug - File not found" in result.stderr
    assert "FileNotFoundError" in result.stderr
    assert "absolutely_nonexistent_file.txt" in result.stderr

    # Test 2: Compare debug vs non-debug error messages
    cmd_normal = ["uv", "run", "python", "-m", "lspeak", "-f", "missing.txt"]
    cmd_debug = ["uv", "run", "python", "-m", "lspeak", "--debug", "-f", "missing.txt"]

    result_normal = subprocess.run(
        cmd_normal,
        cwd=PROJECT_ROOT,
        env={"PYTHONPATH": "src", **os.environ},
        capture_output=True,
        text=True,
    )

    result_debug = subprocess.run(
        cmd_debug,
        cwd=PROJECT_ROOT,
        env={"PYTHONPATH": "src", **os.environ},
        capture_output=True,
        text=True,
    )

    # Both should fail, but debug should have more detailed error info
    assert result_normal.returncode == 1
    assert result_debug.returncode == 1
    assert "Error: File not found" in result_normal.stderr
    assert "Debug - File not found" in result_debug.stderr
    assert len(result_debug.stderr) > len(
        result_normal.stderr
    )  # Debug message is longer


def test_cli_output_and_debug_combined() -> None:
    """Test CLI output and debug options work together (Task 2.6)."""
    output_file = PROJECT_ROOT / "test_combined_output.txt"
    test_content = "Testing combined options"

    try:
        # Test 1: Both options with successful operation
        cmd = [
            "uv",
            "run",
            "python",
            "-m",
            "lspeak",
            "--debug",
            test_content,
            "-o",
            str(output_file),
        ]

        result = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            env={"PYTHONPATH": "src", **os.environ},
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert f"Output saved to {output_file}" in result.stdout
        assert result.stderr == ""
        assert output_file.exists()
        assert output_file.read_text() == test_content

        # Test 2: Both options with error condition (file input error)
        cmd = [
            "uv",
            "run",
            "python",
            "-m",
            "lspeak",
            "--debug",
            "-f",
            "missing.txt",
            "-o",
            str(output_file),
        ]

        result = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            env={"PYTHONPATH": "src", **os.environ},
            capture_output=True,
            text=True,
        )

        assert result.returncode == 1
        assert "Debug - File not found" in result.stderr
        assert "FileNotFoundError" in result.stderr

    finally:
        if output_file.exists():
            output_file.unlink()


def test_cli_list_voices_flag() -> None:
    """Test that --list-voices flag lists available voices and exits."""
    cmd = ["uv", "run", "python", "-m", "lspeak", "--list-voices"]

    result = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        env={"PYTHONPATH": "src", **os.environ},
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    # Check for some expected voice names that should appear
    assert "Aria:" in result.stdout
    assert "Sarah:" in result.stdout
    # Check for voice ID pattern (long alphanumeric strings)
    assert (
        "9BWtsMINqrJLrRacOk9x" in result.stdout
        or len([line for line in result.stdout.split("\n") if ":" in line]) > 10
    )
    assert (
        result.stderr == "" or "pygame" in result.stderr
    )  # pygame init messages are ok


def test_cli_list_voices_with_missing_api_key() -> None:
    """Test that --list-voices shows appropriate error when API key missing."""
    # Create env without API key
    env = {"PYTHONPATH": "src", **os.environ}
    env.pop("ELEVENLABS_API_KEY", None)

    cmd = ["uv", "run", "python", "-m", "lspeak", "--list-voices"]

    result = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "Error: Failed to list voices:" in result.stderr
    assert "ElevenLabs API key not found" in result.stderr


def test_cli_list_voices_with_debug_flag() -> None:
    """Test that --list-voices with --debug shows detailed error on failure."""
    # Create env without API key
    env = {"PYTHONPATH": "src", **os.environ}
    env.pop("ELEVENLABS_API_KEY", None)

    cmd = ["uv", "run", "python", "-m", "lspeak", "--list-voices", "--debug"]

    result = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "Debug - Failed to list voices:" in result.stderr
    assert "TTSAuthError" in result.stderr
    assert "ElevenLabs API key not found" in result.stderr


def test_cli_help_includes_list_voices() -> None:
    """Test that --help output includes the --list-voices option."""
    cmd = ["uv", "run", "python", "-m", "lspeak", "--help"]

    result = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        env={"PYTHONPATH": "src", **os.environ},
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--list-voices" in result.stdout
    assert "List available voices and exit" in result.stdout


def test_cli_speak_functionality_with_text() -> None:
    """Test that speak command actually calls TTS and plays audio (Task 5.4)."""
    # Skip if no API key available
    if "ELEVENLABS_API_KEY" not in os.environ:
        pytest.skip("ELEVENLABS_API_KEY not set, skipping TTS test")

    cmd = ["uv", "run", "python", "-m", "lspeak", "Test speech synthesis"]

    result = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        env={"PYTHONPATH": "src", **os.environ},
        capture_output=True,
        text=True,
    )

    # Should succeed without any output (audio plays through speakers)
    assert result.returncode == 0
    # pygame init messages may appear in stdout
    assert "pygame" in result.stdout or result.stdout == ""
    assert result.stderr == "" or "pygame" in result.stderr


def test_cli_speak_with_voice_selection() -> None:
    """Test that voice selection option works correctly (Task 5.4)."""
    # Skip if no API key available
    if "ELEVENLABS_API_KEY" not in os.environ:
        pytest.skip("ELEVENLABS_API_KEY not set, skipping TTS test")

    # Use a known voice ID (Rachel)
    cmd = [
        "uv",
        "run",
        "python",
        "-m",
        "lspeak",
        "-v",
        "21m00Tcm4TlvDq8ikWAM",
        "Hello with Rachel voice",
    ]

    result = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        env={"PYTHONPATH": "src", **os.environ},
        capture_output=True,
        text=True,
    )

    # Should succeed
    assert result.returncode == 0
    # pygame init messages may appear in stdout
    assert "pygame" in result.stdout or result.stdout == ""
    assert result.stderr == "" or "pygame" in result.stderr


def test_cli_speak_saves_audio_file() -> None:
    """Test that output option saves audio file instead of playing (Task 5.4)."""
    # Skip if no API key available
    if "ELEVENLABS_API_KEY" not in os.environ:
        pytest.skip("ELEVENLABS_API_KEY not set, skipping TTS test")

    output_file = PROJECT_ROOT / "test_audio_output.mp3"

    try:
        cmd = [
            "uv",
            "run",
            "python",
            "-m",
            "lspeak",
            "Save this as audio",
            "-o",
            str(output_file),
        ]

        result = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            env={"PYTHONPATH": "src", **os.environ},
            capture_output=True,
            text=True,
        )

        # Should succeed and confirm save
        assert result.returncode == 0
        assert f"Audio saved to {output_file}" in result.stdout
        assert result.stderr == "" or "pygame" in result.stderr

        # File should exist and contain audio data (not text)
        assert output_file.exists()
        file_size = output_file.stat().st_size
        assert file_size > 1000  # Audio files are typically several KB

        # Verify it's not text by checking first few bytes
        with open(output_file, "rb") as f:
            header = f.read(4)
            # MP3 files often start with ID3 tag or FF FB/FF FA
            assert (
                header[:3] == b"ID3"
                or header[:2] in [b"\xff\xfb", b"\xff\xfa"]
                or header == b"RIFF"
            )

    finally:
        if output_file.exists():
            output_file.unlink()


def test_cli_speak_with_missing_api_key() -> None:
    """Test that speak command shows appropriate error when API key missing (Task 5.4)."""
    # Create env without API key
    env = {"PYTHONPATH": "src", **os.environ}
    env.pop("ELEVENLABS_API_KEY", None)

    cmd = ["uv", "run", "python", "-m", "lspeak", "Test without API key"]

    result = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    # pygame init messages may appear in stdout
    assert "pygame" in result.stdout or result.stdout == ""
    assert "Error: ElevenLabs API key not found" in result.stderr


def test_cli_speak_error_handling_with_debug() -> None:
    """Test that TTS errors show enhanced messages with debug flag (Task 5.4)."""
    # Test with missing API key to trigger auth error
    env = {"PYTHONPATH": "src", **os.environ}
    env.pop("ELEVENLABS_API_KEY", None)

    # Test without debug
    cmd_normal = ["uv", "run", "python", "-m", "lspeak", "Test error"]
    result_normal = subprocess.run(
        cmd_normal,
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )

    # Test with debug
    cmd_debug = ["uv", "run", "python", "-m", "lspeak", "--debug", "Test error"]
    result_debug = subprocess.run(
        cmd_debug,
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )

    # Both should fail
    assert result_normal.returncode == 1
    assert result_debug.returncode == 1

    # Normal error should be concise
    assert "Error: ElevenLabs API key not found" in result_normal.stderr
    assert "TTSAuthError" not in result_normal.stderr

    # Debug error should have more detail
    assert "Debug - Authentication error:" in result_debug.stderr
    assert "TTSAuthError" in result_debug.stderr
    assert len(result_debug.stderr) > len(result_normal.stderr)


def test_cli_speak_with_stdin_input() -> None:
    """Test that speak command works with stdin input (Task 5.4)."""
    # Skip if no API key available
    if "ELEVENLABS_API_KEY" not in os.environ:
        pytest.skip("ELEVENLABS_API_KEY not set, skipping TTS test")

    stdin_text = "Hello from stdin"
    cmd = ["uv", "run", "python", "-m", "lspeak"]

    result = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        env={"PYTHONPATH": "src", **os.environ},
        input=stdin_text,
        capture_output=True,
        text=True,
    )

    # Should succeed
    assert result.returncode == 0
    # pygame init messages may appear in stdout
    assert "pygame" in result.stdout or result.stdout == ""
    assert result.stderr == "" or "pygame" in result.stderr


def test_cli_speak_with_file_input() -> None:
    """Test that speak command works with file input (Task 5.4)."""
    # Skip if no API key available
    if "ELEVENLABS_API_KEY" not in os.environ:
        pytest.skip("ELEVENLABS_API_KEY not set, skipping TTS test")

    # Create test file
    test_file = PROJECT_ROOT / "test_speak_input.txt"
    test_content = "Hello from file input"
    test_file.write_text(test_content)

    try:
        cmd = ["uv", "run", "python", "-m", "lspeak", "-f", str(test_file)]

        result = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            env={"PYTHONPATH": "src", **os.environ},
            capture_output=True,
            text=True,
        )

        # Should succeed
        assert result.returncode == 0
        # pygame init messages may appear in stdout
        assert "pygame" in result.stdout or result.stdout == ""
        assert result.stderr == "" or "pygame" in result.stderr

    finally:
        if test_file.exists():
            test_file.unlink()


def test_cli_help_includes_voice_option() -> None:
    """Test that --help output includes the voice option (Task 5.4)."""
    cmd = ["uv", "run", "python", "-m", "lspeak", "--help"]

    result = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        env={"PYTHONPATH": "src", **os.environ},
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--voice" in result.stdout
    assert "-v" in result.stdout
    assert "Voice ID to use for speech synthesis" in result.stdout


def test_cli_empty_text_error_handling() -> None:
    """Test that CLI handles empty text input with specific error message (Task 6.2)."""
    # Test with empty stdin
    cmd = ["uv", "run", "python", "-m", "lspeak"]

    result = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        env={"PYTHONPATH": "src", **os.environ},
        input="",  # Empty stdin
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "Error: Text cannot be empty" in result.stderr
    # pygame may output initialization messages to stdout
    assert result.stdout == "" or "pygame" in result.stdout


def test_cli_empty_text_error_handling_with_debug() -> None:
    """Test that CLI shows debug details for empty text error (Task 6.2)."""
    # Test with empty stdin and debug flag
    cmd = ["uv", "run", "python", "-m", "lspeak", "--debug"]

    result = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        env={"PYTHONPATH": "src", **os.environ},
        input="",  # Empty stdin
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert (
        "Debug - Text processing error: ValueError('Text cannot be empty')"
        in result.stderr
    )
    # pygame may output initialization messages to stdout
    assert result.stdout == "" or "pygame" in result.stdout
