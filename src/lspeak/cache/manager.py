"""Semantic cache manager for intelligent TTS caching.

Orchestrates CacheStorage, EmbeddingGenerator, and SimilarityIndex to provide
semantic caching where similar phrases (by meaning, not exact match) can reuse
cached audio files, dramatically reducing API costs.
"""

import hashlib
import logging
from datetime import datetime
from pathlib import Path

from ..embeddings.generator import EmbeddingGenerator
from . import get_cache_dir
from .models import CacheEntry
from .similarity import SimilarityIndex
from .storage import CacheStorage

logger = logging.getLogger(__name__)


class SemanticCacheManager:
    """High-level semantic cache manager for TTS audio caching.

    Coordinates CacheStorage (SQLite metadata), EmbeddingGenerator (text embeddings),
    and SimilarityIndex (FAISS similarity search) to enable intelligent caching
    where similar phrases by semantic meaning can reuse cached audio files.

    Example:
        cache = SemanticCacheManager()

        # First call - cache miss, generates and stores audio
        audio_path = await cache.get_cached_audio("Deploy complete", "elevenlabs", "Rachel")
        if not audio_path:
            # Generate audio via TTS API
            audio_bytes = await tts_provider.synthesize("Deploy complete", "Rachel")
            await cache.cache_audio("Deploy complete", "elevenlabs", "Rachel", audio_bytes)

        # Second call with similar phrase - cache hit!
        audio_path = await cache.get_cached_audio("Deployment complete", "elevenlabs", "Rachel")
        # Returns cached audio from first call due to semantic similarity
    """

    def __init__(
        self, cache_dir: Path | None = None, similarity_threshold: float = 0.95
    ):
        """Initialize semantic cache manager with component orchestration.

        Args:
            cache_dir: Directory for cache storage (defaults to ~/.cache/lspeak)
            similarity_threshold: Cosine similarity threshold for cache hits (0.0-1.0)

        Raises:
            ValueError: If similarity_threshold not in valid range
            RuntimeError: If component initialization fails
        """
        # Validate similarity threshold
        if not 0.0 <= similarity_threshold <= 1.0:
            raise ValueError(
                f"similarity_threshold must be between 0.0 and 1.0, got {similarity_threshold}"
            )

        # Set up cache directory structure
        self.cache_dir = cache_dir or get_cache_dir()
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Create audio subdirectory
        self.audio_dir = self.cache_dir / "audio"
        self.audio_dir.mkdir(exist_ok=True)

        self.threshold = similarity_threshold

        # Initialize components in proper order
        try:
            # Task 6.1: Initialize CacheStorage (creates SQLite DB)
            self.storage = CacheStorage(self.cache_dir)
            logger.debug(f"Initialized cache storage at {self.cache_dir}")

            # Initialize EmbeddingGenerator (loads sentence-transformer model)
            self.generator = EmbeddingGenerator()
            logger.debug(
                f"Initialized embedding generator with model {self.generator.model_name}"
            )

            # Initialize SimilarityIndex (loads/creates FAISS index)
            self.index = SimilarityIndex(self.cache_dir)
            logger.debug(
                f"Initialized similarity index with {self.index.index.ntotal} entries"
            )

        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize semantic cache manager: {e}"
            ) from e

        logger.info(
            f"SemanticCacheManager initialized at {self.cache_dir} "
            f"with threshold {similarity_threshold}"
        )

    async def get_cached_audio(
        self, text: str, provider: str, voice: str
    ) -> Path | None:
        """Check cache for semantically similar text with same provider/voice.

        Generates embedding for input text and searches FAISS index for similar
        cached embeddings above the similarity threshold. Checks multiple candidates
        to find one that matches the requested provider and voice.

        Args:
            text: Input text to search for in cache
            provider: TTS provider name (must match cached entry)
            voice: Voice identifier (must match cached entry)

        Returns:
            Path to cached audio file if semantic match found, None for cache miss

        Raises:
            RuntimeError: If embedding generation or search fails
        """
        try:
            # Task 6.2: Generate embedding for input text
            embedding = self.generator.generate([text])[0]
            logger.debug(f"Generated embedding for text: '{text[:50]}...'")

            # Search for multiple similar embeddings to handle same text with different voices
            candidates = self.index.search_multiple(
                embedding, self.threshold, max_results=10
            )

            if not candidates:
                logger.debug(
                    f"Cache miss: no similar text found above threshold {self.threshold}"
                )
                return None

            logger.debug(
                f"Found {len(candidates)} potential cache hits above threshold"
            )

            # Check each candidate for provider/voice match
            for embedding_idx, similarity in candidates:
                # Retrieve cache entry metadata from SQLite
                entry = self.storage.get_by_embedding_idx(embedding_idx)
                if entry is None:
                    logger.warning(
                        f"Index/storage mismatch: embedding_idx {embedding_idx} "
                        "found in FAISS but not in SQLite"
                    )
                    continue

                # Check if provider and voice match
                if entry.provider == provider and entry.voice == voice:
                    # Verify audio file still exists on filesystem
                    if not entry.audio_path.exists():
                        logger.warning(
                            f"Cache corruption: metadata exists but audio file missing: {entry.audio_path}"
                        )
                        continue

                    logger.info(
                        f"Cache hit! Similar text '{entry.text}' found for '{text}' "
                        f"with similarity {similarity:.3f} (provider: {provider}, voice: {voice})"
                    )
                    return entry.audio_path
                else:
                    logger.debug(
                        f"Skipping candidate: provider/voice mismatch. "
                        f"Cached: {entry.provider}/{entry.voice}, "
                        f"Requested: {provider}/{voice}"
                    )

            # No matching provider/voice found among similar texts
            logger.debug(
                f"Cache miss: no matching provider/voice among {len(candidates)} similar texts"
            )
            return None

        except Exception as e:
            logger.error(f"Error during cache lookup: {e}")
            # Graceful degradation - return cache miss rather than breaking TTS
            return None

    async def cache_audio(
        self, text: str, provider: str, voice: str, audio_bytes: bytes
    ) -> None:
        """Store audio with metadata and embedding for future semantic matching.

        Generates unique filename from text/provider/voice hash, saves audio file,
        creates text embedding, adds to FAISS index, and stores metadata in SQLite.
        All operations are atomic - if any step fails, partial state is cleaned up.

        Args:
            text: Original text that generated this audio
            provider: TTS provider name used
            voice: Voice identifier used
            audio_bytes: Audio data to cache

        Raises:
            RuntimeError: If caching operation fails
        """
        audio_path = None
        try:
            # Task 6.3 & 6.4: Generate unique hash for audio filename
            audio_hash = self._generate_audio_hash(text, provider, voice)
            filename = f"{audio_hash}.mp3"
            audio_path = self.audio_dir / filename

            logger.debug(f"Caching audio as {filename} for text: '{text[:50]}...'")

            # Save audio bytes to filesystem with atomic write
            audio_path.write_bytes(audio_bytes)
            logger.debug(f"Saved audio file: {audio_path}")

            # Generate text embedding
            embedding = self.generator.generate([text])[0]

            # Add embedding to FAISS index and get index position
            embedding_idx = self.index.add(embedding)
            logger.debug(f"Added embedding to index at position {embedding_idx}")

            # Create cache entry with all metadata
            entry = CacheEntry(
                text=text,
                provider=provider,
                voice=voice,
                audio_path=audio_path,
                embedding_idx=embedding_idx,
                timestamp=datetime.now(),
            )

            # Save metadata to SQLite
            self.storage.save(entry)

            logger.info(
                f"Successfully cached audio for '{text}' "
                f"({provider}/{voice}) at embedding index {embedding_idx}"
            )

        except Exception as e:
            # Clean up partial state if caching failed
            if audio_path and audio_path.exists():
                try:
                    audio_path.unlink()
                    logger.debug(f"Cleaned up partial audio file: {audio_path}")
                except Exception as cleanup_error:
                    logger.warning(
                        f"Failed to clean up partial audio file: {cleanup_error}"
                    )

            raise RuntimeError(f"Failed to cache audio: {e}") from e

    def _generate_audio_hash(self, text: str, provider: str, voice: str) -> str:
        """Generate unique hash for audio filename from text/provider/voice.

        Creates deterministic hash from the combination of text, provider, and voice
        to ensure the same input always generates the same filename while avoiding
        collisions between different combinations.

        Args:
            text: Input text string
            provider: TTS provider name
            voice: Voice identifier

        Returns:
            8-character hex hash string for use as filename prefix

        Raises:
            ValueError: If any input parameter is None
        """
        # Task 6.4: Validate inputs
        if text is None or provider is None or voice is None:
            raise ValueError("All parameters (text, provider, voice) must be non-None")

        # Create deterministic input string
        input_string = f"{text}:{provider}:{voice}"

        # Generate SHA-256 hash and take first 8 characters
        hash_bytes = input_string.encode("utf-8")
        hash_hex = hashlib.sha256(hash_bytes).hexdigest()

        return hash_hex[:8]
