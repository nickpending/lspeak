"""High-level API for lspeak library usage."""

from pathlib import Path

from .core import speak_text


async def speak(
    text: str,
    provider: str = "elevenlabs",
    voice: str | None = None,
    output: str | Path | None = None,
    cache: bool = True,
    cache_threshold: float = 0.95,
    use_daemon: bool = True,
    queue: bool = True,
    debug: bool = False,
) -> bytes | None:
    """Synthesize speech from text.

    Args:
        text: Text to speak
        provider: TTS provider name
        voice: Voice ID/name (provider-specific)
        output: File path to save audio (if None, plays audio)
        cache: Whether to use semantic caching
        cache_threshold: Similarity threshold for cache hits
        use_daemon: Whether to use daemon for faster response (default True)
        queue: Whether to queue speech in daemon for serial playback (default True)
        debug: Enable debug logging

    Returns:
        Audio bytes if output specified, None if played

    Raises:
        TTSAuthError: If API key is not configured
        TTSAPIError: If TTS conversion fails
        RuntimeError: If audio playback fails
        OSError: If file save fails
        ValueError: If text is empty
        KeyError: If provider not found
    """
    # Convert output to string if Path
    output_str = str(output) if output else None

    # Call core speak_text function with all parameters
    await speak_text(
        text=text,
        provider=provider,
        voice_id=voice,
        output_file=output_str,
        cache=cache,
        cache_threshold=cache_threshold,
        use_daemon=use_daemon,
        queue=queue,
        debug=debug,
    )

    # If output was specified, read the file and return bytes
    if output:
        output_path = Path(output)
        if output_path.exists():
            return output_path.read_bytes()

    return None
