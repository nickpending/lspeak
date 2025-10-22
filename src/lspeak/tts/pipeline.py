"""TTS pipeline orchestrator for lspeak.

Coordinates TTSProvider, SemanticCacheManager, and AudioPlayer to provide
a unified TTS workflow that eliminates code duplication between daemon and
direct execution paths.
"""

import logging
from typing import Any

from ..audio.player import AudioPlayer
from ..cache.manager import SemanticCacheManager
from ..providers import ProviderRegistry

logger = logging.getLogger(__name__)


class TTSPipeline:
    """Orchestrates complete TTS workflow from text to audio output.

    Handles cache lookup, TTS synthesis, audio caching, and playback/saving
    in a single coordinated workflow. Supports both pre-loaded instances
    (daemon path) and on-demand instance creation (direct path).

    Example (daemon path with pre-loaded instances):
        cache_manager = SemanticCacheManager()
        audio_player = AudioPlayer()
        pipeline = TTSPipeline(cache_manager=cache_manager, audio_player=audio_player)

        result = await pipeline.process(
            text="Deploy complete",
            provider="elevenlabs",
            voice="Rachel",
            output=None,
            cache=True,
            debug=False
        )
        # Returns: {"played": True, "saved": None, "cached": False}

    Example (direct path with on-demand instances):
        pipeline = TTSPipeline()

        result = await pipeline.process(
            text="Build finished",
            provider="system",
            voice=None,
            output="/tmp/output.mp3",
            cache=True,
            debug=False
        )
        # Returns: {"played": False, "saved": "/tmp/output.mp3", "cached": True}
    """

    def __init__(
        self,
        cache_manager: SemanticCacheManager | None = None,
        audio_player: AudioPlayer | None = None,
    ) -> None:
        """Initialize TTS pipeline with optional pre-loaded components.

        Args:
            cache_manager: Optional pre-loaded cache manager (daemon uses this)
            audio_player: Optional pre-loaded audio player (daemon uses this)

        Note:
            If components are not provided, they will be created on-demand
            during process() call. Daemon path provides pre-loaded instances
            for sub-second response times.
        """
        self.cache_manager = cache_manager
        self.audio_player = audio_player

        logger.debug(
            f"TTSPipeline initialized with "
            f"cache_manager={'pre-loaded' if cache_manager else 'on-demand'}, "
            f"audio_player={'pre-loaded' if audio_player else 'on-demand'}"
        )

    async def process(
        self,
        text: str,
        provider: str,
        voice: str | None,
        output: str | None,
        cache: bool,
        cache_threshold: float = 0.95,
        debug: bool = False,
    ) -> dict[str, Any]:
        """Process complete TTS workflow from text to audio output.

        Args:
            text: Text to convert to speech
            provider: Provider name (e.g., "elevenlabs", "system")
            voice: Voice ID to use (None uses provider's default)
            output: Output file path (None plays through speakers)
            cache: Whether to use semantic caching
            cache_threshold: Similarity threshold for cache hits (0.0-1.0)
            debug: Enable debug logging

        Returns:
            Dictionary with workflow results:
                {
                    "played": bool,      # True if played through speakers
                    "saved": str | None, # File path if saved to disk
                    "cached": bool       # True if audio came from cache
                }

        Raises:
            TTSAuthError: If provider authentication fails
            TTSAPIError: If TTS API call fails
            ValueError: If text is empty
            KeyError: If provider not found
        """
        # Validate required parameters
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        # Get provider and create instance
        try:
            provider_class = ProviderRegistry.get(provider)
            provider_instance = provider_class()
        except KeyError:
            raise KeyError(f"Provider '{provider}' not found") from None

        audio_data: bytes | None = None
        cache_hit = False

        # === CACHE LOOKUP PHASE ===
        if cache:
            # Create cache manager on-demand if not pre-loaded
            if self.cache_manager is None:
                try:
                    self.cache_manager = SemanticCacheManager(
                        similarity_threshold=cache_threshold
                    )
                    if debug:
                        logger.debug("Created cache manager on-demand")
                except Exception as e:
                    logger.error(f"Failed to create cache manager: {e}")
                    # Continue without cache
                    cache = False

            # Check cache for similar text
            if cache and self.cache_manager:
                try:
                    if debug:
                        logger.debug(
                            f"Checking cache for: '{text[:50]}{'...' if len(text) > 50 else ''}'"
                        )

                    cached_audio_path = await self.cache_manager.get_cached_audio(
                        text, provider, voice or ""
                    )

                    if cached_audio_path:
                        if debug:
                            logger.debug(
                                f"Cache hit! Using cached audio from: {cached_audio_path}"
                            )
                        # Read cached audio
                        audio_data = cached_audio_path.read_bytes()
                        cache_hit = True
                    else:
                        if debug:
                            logger.debug("Cache miss - will generate new audio")
                except Exception as e:
                    logger.error(f"Cache lookup failed: {e}")
                    # Continue without cache

        # === SYNTHESIS PHASE ===
        if audio_data is None:
            try:
                if debug:
                    logger.debug(f"Calling {provider} TTS API for synthesis")
                audio_data = await provider_instance.synthesize(text, voice=voice or "")
            except Exception:
                # Let provider errors bubble up
                raise

            # Cache the generated audio
            if cache and self.cache_manager:
                try:
                    if debug:
                        logger.debug("Caching generated audio for future use")
                    await self.cache_manager.cache_audio(
                        text, provider, voice or "", audio_data
                    )
                    if debug:
                        logger.debug("Audio successfully cached")
                except Exception as e:
                    logger.error(f"Failed to cache audio: {e}")
                    # Continue even if caching fails

        # === OUTPUT PHASE ===
        # Create audio player on-demand if not pre-loaded
        if self.audio_player is None:
            self.audio_player = AudioPlayer()
            if debug:
                logger.debug("Created audio player on-demand")

        # Either save to file or play through speakers
        if output:
            self.audio_player.save_to_file(audio_data, output)
            return {
                "played": False,
                "saved": output,
                "cached": cache_hit,
            }
        else:
            await self.audio_player.play_bytes_async(audio_data)
            return {
                "played": True,
                "saved": None,
                "cached": cache_hit,
            }
