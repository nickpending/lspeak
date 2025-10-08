"""Core functionality for lspeak - orchestrates TTS and audio operations."""

import logging

from .audio.player import AudioPlayer
from .cache.manager import SemanticCacheManager
from .providers import ProviderRegistry
from .tts.errors import TTSAPIError, TTSAuthError

logger = logging.getLogger(__name__)


async def list_available_voices(provider: str = "elevenlabs") -> None:
    """List all available voices from specified provider.

    Prints voices in "Name: voice_id" format to stdout.

    Args:
        provider: Provider name to list voices from

    Raises:
        TTSAuthError: If API key is not configured
        TTSAPIError: If API call fails
        KeyError: If provider not found
    """
    try:
        provider_class = ProviderRegistry.get(provider)
        provider_instance = provider_class()
        voices = await provider_instance.list_voices()

        for voice in voices:
            print(f"{voice['name']}: {voice['id']}")

    except TTSAuthError:
        # Re-raise auth errors as-is for clear error messages
        raise
    except TTSAPIError:
        # Re-raise API errors as-is
        raise
    except KeyError:
        # Re-raise provider not found errors
        raise
    except Exception as e:
        # Catch any unexpected errors
        raise TTSAPIError(f"Failed to list voices: {e}", None, e) from e


async def speak_text(
    text: str,
    provider: str = "elevenlabs",
    voice_id: str | None = None,
    output_file: str | None = None,
    cache: bool = True,
    cache_threshold: float = 0.95,
    debug: bool = False,
    use_daemon: bool = True,
    queue: bool = True,
) -> None:
    """Convert text to speech and play or save the audio.

    Args:
        text: Text to convert to speech
        provider: Provider name to use for synthesis
        voice_id: Optional voice ID to use. If not provided, uses first available voice
        output_file: Optional file path to save audio. If not provided, plays through speakers
        cache: Whether to use semantic caching
        cache_threshold: Similarity threshold for cache hits (0.0-1.0)
        debug: Enable debug logging
        use_daemon: Whether to use daemon for faster response (default True)
        queue: Whether to queue speech in daemon for serial playback (default True)

    Raises:
        TTSAuthError: If API key is not configured
        TTSAPIError: If TTS conversion fails
        RuntimeError: If audio playback fails
        OSError: If file save fails
        ValueError: If text is empty
        KeyError: If provider not found
    """
    # Auto-disable cache for long text to prevent embedding model segfaults
    CACHE_TEXT_LIMIT = (
        500  # Characters - sentence transformers struggle with very long text
    )
    if cache and len(text) > CACHE_TEXT_LIMIT:
        cache = False
        if debug:
            logger.debug(
                f"Auto-disabled cache: text length {len(text)} > {CACHE_TEXT_LIMIT} chars"
            )

    # Task 5.2: Check if daemon should be used
    if use_daemon:
        try:
            # Task 5.3: Initialize daemon client
            from .daemon.client import DaemonClient
            from .daemon.spawn import ensure_daemon_running_async

            if debug:
                logger.debug("Attempting to use daemon for faster response")

            # Task 5.4: Ensure daemon is running and send request
            if await ensure_daemon_running_async():
                client = DaemonClient()
                response = await client.speak(
                    text=text,
                    provider=provider,
                    voice=voice_id,
                    output=output_file,
                    cache=cache,
                    cache_threshold=cache_threshold,
                    debug=debug,
                    queue=queue,
                )

                if response["status"] == "success":
                    if debug:
                        logger.debug(
                            f"Daemon handled request successfully: {response.get('result')}"
                        )
                    return  # Daemon handled it successfully

                # Task 5.6: If daemon returned error, try restart once
                if response["status"] == "error":
                    if debug:
                        logger.debug(
                            f"Daemon returned error: {response.get('error')}, attempting restart"
                        )

                    # Force restart and retry once
                    if await ensure_daemon_running_async(force_restart=True):
                        response = await client.speak(
                            text=text,
                            provider=provider,
                            voice=voice_id,
                            output=output_file,
                            cache=cache,
                            cache_threshold=cache_threshold,
                            debug=debug,
                            queue=queue,
                        )

                        if response["status"] == "success":
                            if debug:
                                logger.debug("Daemon handled request after restart")
                            return

                    if debug:
                        logger.debug(
                            "Daemon restart failed, falling back to direct execution"
                        )
            else:
                if debug:
                    logger.debug(
                        "Failed to start daemon, falling back to direct execution"
                    )

        except Exception as e:
            # Task 5.5: Fall back to direct execution on any daemon error
            if debug:
                logger.debug(f"Daemon failed ({e}), falling back to direct execution")

    # Task 5.5: Fallback to direct execution (existing code)
    if debug and use_daemon:
        logger.debug("Using direct execution (fallback path)")

    # Get provider and create instance
    provider_class = ProviderRegistry.get(provider)
    provider_instance = provider_class()

    audio_data = None
    cache_manager = None

    # Initialize cache if enabled
    if cache:
        try:
            cache_manager = SemanticCacheManager(similarity_threshold=cache_threshold)
            if debug:
                logger.debug(f"Cache enabled with threshold {cache_threshold}")

            # Check cache for similar text
            if debug:
                logger.debug(f"Checking cache for text: '{text[:50]}...'")

            cached_audio_path = await cache_manager.get_cached_audio(
                text, provider, voice_id or "default"
            )

            if cached_audio_path:
                if debug:
                    logger.debug(
                        f"Cache hit! Using cached audio from: {cached_audio_path}"
                    )
                # Read cached audio
                audio_data = cached_audio_path.read_bytes()
            else:
                if debug:
                    logger.debug("Cache miss - will generate new audio")
        except Exception as e:
            if debug:
                logger.debug(
                    f"Cache initialization/lookup failed: {e}. Proceeding without cache."
                )
            # Continue without cache on error
            cache_manager = None

    # Generate new audio if not found in cache
    if audio_data is None:
        if debug:
            logger.debug(f"Calling {provider} TTS API for synthesis")
        audio_data = await provider_instance.synthesize(text, voice=voice_id or "")

        # Cache the generated audio if cache is enabled
        if cache_manager:
            try:
                if debug:
                    logger.debug("Caching generated audio for future use")
                await cache_manager.cache_audio(
                    text, provider, voice_id or "default", audio_data
                )
                if debug:
                    logger.debug("Audio successfully cached")
            except Exception as e:
                if debug:
                    logger.debug(
                        f"Failed to cache audio: {e}. Continuing without caching."
                    )
                # Continue even if caching fails

    # Create audio player
    player = AudioPlayer()

    # Either save to file or play through speakers
    if output_file:
        player.save_to_file(audio_data, output_file)
    else:
        player.play_bytes(audio_data)
