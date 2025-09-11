"""Unit tests for SemanticCacheManager logic and validation."""

import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from lspeak.cache.manager import SemanticCacheManager


class TestSemanticCacheManagerValidation:
    """Test validation logic in SemanticCacheManager."""

    def test_similarity_threshold_validation_valid_range(self) -> None:
        """Test that valid similarity thresholds are accepted."""
        with TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir)

            # Test boundary values
            manager1 = SemanticCacheManager(cache_dir, similarity_threshold=0.0)
            assert manager1.threshold == 0.0

            manager2 = SemanticCacheManager(cache_dir, similarity_threshold=1.0)
            assert manager2.threshold == 1.0

            manager3 = SemanticCacheManager(cache_dir, similarity_threshold=0.95)
            assert manager3.threshold == 0.95

    def test_similarity_threshold_validation_below_minimum(self) -> None:
        """Test that threshold below 0.0 is rejected."""
        with TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir)

            with pytest.raises(
                ValueError,
                match="similarity_threshold must be between 0.0 and 1.0, got -0.1",
            ):
                SemanticCacheManager(cache_dir, similarity_threshold=-0.1)

    def test_similarity_threshold_validation_above_maximum(self) -> None:
        """Test that threshold above 1.0 is rejected."""
        with TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir)

            with pytest.raises(
                ValueError,
                match="similarity_threshold must be between 0.0 and 1.0, got 1.5",
            ):
                SemanticCacheManager(cache_dir, similarity_threshold=1.5)


class TestHashGeneration:
    """Test audio filename hash generation logic."""

    def test_hash_generation_deterministic(self) -> None:
        """Test that same inputs always produce same hash."""
        with TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir)
            manager = SemanticCacheManager(cache_dir)

            # Generate hash multiple times with same input
            hash1 = manager._generate_audio_hash("Hello world", "elevenlabs", "Rachel")
            hash2 = manager._generate_audio_hash("Hello world", "elevenlabs", "Rachel")
            hash3 = manager._generate_audio_hash("Hello world", "elevenlabs", "Rachel")

            # All should be identical
            assert hash1 == hash2 == hash3
            assert len(hash1) == 8

    def test_hash_generation_different_inputs(self) -> None:
        """Test that different inputs produce different hashes."""
        with TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir)
            manager = SemanticCacheManager(cache_dir)

            # Different text
            hash1 = manager._generate_audio_hash("Hello world", "elevenlabs", "Rachel")
            hash2 = manager._generate_audio_hash(
                "Goodbye world", "elevenlabs", "Rachel"
            )
            assert hash1 != hash2

            # Different provider
            hash3 = manager._generate_audio_hash("Hello world", "system", "Rachel")
            assert hash1 != hash3

            # Different voice
            hash4 = manager._generate_audio_hash("Hello world", "elevenlabs", "Adam")
            assert hash1 != hash4

    def test_hash_generation_length(self) -> None:
        """Test that generated hashes are always 8 characters."""
        with TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir)
            manager = SemanticCacheManager(cache_dir)

            test_cases = [
                ("short", "p1", "v1"),
                ("very long text " * 100, "provider2", "voice2"),
                ("", "p3", "v3"),  # Empty text
                ("special chars: !@#$%^&*()", "p4", "v4"),
            ]

            for text, provider, voice in test_cases:
                hash_val = manager._generate_audio_hash(text, provider, voice)
                assert len(hash_val) == 8
                # Verify it's valid hex
                assert all(c in "0123456789abcdef" for c in hash_val)

    def test_hash_generation_none_inputs_validation(self) -> None:
        """Test that None inputs are rejected."""
        with TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir)
            manager = SemanticCacheManager(cache_dir)

            with pytest.raises(
                ValueError,
                match="All parameters \\(text, provider, voice\\) must be non-None",
            ):
                manager._generate_audio_hash(None, "provider", "voice")

            with pytest.raises(
                ValueError,
                match="All parameters \\(text, provider, voice\\) must be non-None",
            ):
                manager._generate_audio_hash("text", None, "voice")

            with pytest.raises(
                ValueError,
                match="All parameters \\(text, provider, voice\\) must be non-None",
            ):
                manager._generate_audio_hash("text", "provider", None)


class TestErrorHandling:
    """Test error handling and graceful degradation."""

    @pytest.mark.asyncio
    async def test_get_cached_audio_handles_embedding_generation_error(self) -> None:
        """Test that embedding generation errors return cache miss."""
        with TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir)
            manager = SemanticCacheManager(cache_dir)

            # Mock embedding generator to raise error
            with patch.object(
                manager.generator, "generate", side_effect=Exception("Embedding error")
            ):
                result = await manager.get_cached_audio("test", "provider", "voice")

                # Should return None (cache miss) instead of crashing
                assert result is None

    @pytest.mark.asyncio
    async def test_get_cached_audio_handles_search_error(self) -> None:
        """Test that FAISS search errors return cache miss."""
        with TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir)
            manager = SemanticCacheManager(cache_dir)

            # Mock successful embedding generation
            mock_embedding = Mock()
            with patch.object(
                manager.generator, "generate", return_value=[mock_embedding]
            ):
                # Mock search to raise error
                with patch.object(
                    manager.index, "search", side_effect=Exception("Search error")
                ):
                    result = await manager.get_cached_audio("test", "provider", "voice")

                    # Should return None (cache miss) instead of crashing
                    assert result is None

    @pytest.mark.asyncio
    async def test_cache_audio_cleans_up_on_failure(self) -> None:
        """Test that partial audio files are cleaned up on cache failure."""
        with TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir)
            manager = SemanticCacheManager(cache_dir)

            # Create test data
            test_audio = b"test audio data"
            audio_hash = manager._generate_audio_hash("test", "provider", "voice")
            expected_path = manager.audio_dir / f"{audio_hash}.mp3"

            # Mock storage.save to fail after audio is written
            with patch.object(
                manager.storage, "save", side_effect=Exception("Database error")
            ):
                # Attempt to cache audio
                with pytest.raises(RuntimeError, match="Failed to cache audio"):
                    await manager.cache_audio("test", "provider", "voice", test_audio)

                # Verify audio file was cleaned up
                assert not expected_path.exists()

    def test_initialization_handles_component_failures(self) -> None:
        """Test that component initialization failures are properly reported."""
        with TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir)

            # Mock CacheStorage to fail initialization
            with patch(
                "lspeak.cache.manager.CacheStorage",
                side_effect=Exception("Storage init failed"),
            ):
                with pytest.raises(
                    RuntimeError,
                    match="Failed to initialize semantic cache manager: Storage init failed",
                ):
                    SemanticCacheManager(cache_dir)
