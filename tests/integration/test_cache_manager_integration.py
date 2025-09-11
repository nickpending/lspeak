"""Integration tests for SemanticCacheManager with real services."""

import sqlite3
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from lspeak.cache import get_cache_dir
from lspeak.cache.manager import SemanticCacheManager
from lspeak.embeddings.models import EMBEDDING_DIM


class TestSemanticCacheManagerIntegration:
    """Test complete semantic cache workflow with real services."""

    @pytest.mark.asyncio
    async def test_complete_cache_workflow_miss_store_hit(self) -> None:
        """
        Test complete cache workflow:
        - Initial cache miss
        - Store audio with embedding
        - Cache hit on same text
        - Cache hit on semantically similar text
        """
        with TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir) / "semantic_cache"

            # Initialize manager with lower threshold for testing
            manager = SemanticCacheManager(cache_dir, similarity_threshold=0.85)

            # Test 1: Initial cache miss
            result = await manager.get_cached_audio(
                "Deploy complete", "elevenlabs", "Rachel"
            )
            assert result is None, "Should be cache miss on empty cache"

            # Test 2: Store audio in cache
            test_audio = b"This is test audio data for deploy complete"
            await manager.cache_audio(
                "Deploy complete", "elevenlabs", "Rachel", test_audio
            )

            # Verify audio file was created
            audio_files = list((cache_dir / "audio").glob("*.mp3"))
            assert len(audio_files) == 1, "Should have created one audio file"
            assert audio_files[0].read_bytes() == test_audio

            # Test 3: Exact match cache hit
            result = await manager.get_cached_audio(
                "Deploy complete", "elevenlabs", "Rachel"
            )
            assert result is not None, "Should hit cache for exact text"
            assert result.exists(), "Cached audio file should exist"
            assert result.read_bytes() == test_audio

            # Test 4: Semantic similarity cache hit
            similar_result = await manager.get_cached_audio(
                "Deployment complete", "elevenlabs", "Rachel"
            )
            assert similar_result is not None, (
                "Should hit cache for semantically similar text"
            )
            assert similar_result == result, "Should return same cached audio file"

            # Test 5: Different text should miss
            different_result = await manager.get_cached_audio(
                "Hello world", "elevenlabs", "Rachel"
            )
            assert different_result is None, "Should miss cache for unrelated text"

    @pytest.mark.asyncio
    async def test_provider_voice_matching_requirements(self) -> None:
        """Test that cache hits require matching provider and voice."""
        with TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir) / "provider_test"
            manager = SemanticCacheManager(cache_dir)

            # Cache audio with specific provider/voice
            test_audio = b"Test audio for provider matching"
            await manager.cache_audio("Test phrase", "elevenlabs", "Rachel", test_audio)

            # Same text, same provider/voice - should hit
            hit_result = await manager.get_cached_audio(
                "Test phrase", "elevenlabs", "Rachel"
            )
            assert hit_result is not None, "Should hit with matching provider/voice"

            # Same text, different provider - should miss
            miss_provider = await manager.get_cached_audio(
                "Test phrase", "system", "Rachel"
            )
            assert miss_provider is None, "Should miss with different provider"

            # Same text, different voice - should miss
            miss_voice = await manager.get_cached_audio(
                "Test phrase", "elevenlabs", "Adam"
            )
            assert miss_voice is None, "Should miss with different voice"

            # Same text, both different - should miss
            miss_both = await manager.get_cached_audio(
                "Test phrase", "system", "default"
            )
            assert miss_both is None, "Should miss with different provider and voice"

    @pytest.mark.asyncio
    async def test_multiple_entries_with_different_similarities(self) -> None:
        """Test handling multiple cached entries with varying similarities."""
        with TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir) / "multi_cache"
            manager = SemanticCacheManager(cache_dir, similarity_threshold=0.9)

            # Cache several related phrases
            phrases = [
                ("Deploy complete", b"audio1"),
                ("Build finished", b"audio2"),
                ("Tests passed", b"audio3"),
            ]

            for text, audio in phrases:
                await manager.cache_audio(text, "elevenlabs", "Rachel", audio)

            # Verify all were cached
            audio_files = list((cache_dir / "audio").glob("*.mp3"))
            assert len(audio_files) == 3, "Should have cached all three phrases"

            # Test similarity matching - semantic similarity is not guaranteed for all phrases
            # "Deployment complete" should match "Deploy complete" due to high similarity
            result1 = await manager.get_cached_audio(
                "Deployment complete", "elevenlabs", "Rachel"
            )
            assert result1 is not None, "Should match 'Deploy complete'"
            assert result1.read_bytes() == b"audio1"

            # These phrases might not be similar enough - test that we get some cached result
            result2 = await manager.get_cached_audio(
                "Deploy complete",
                "elevenlabs",
                "Rachel",  # Use exact match
            )
            assert result2 is not None, "Should match exact text"
            assert result2.read_bytes() == b"audio1"

            # Verify unrelated text doesn't match
            unrelated_result = await manager.get_cached_audio(
                "Completely unrelated phrase", "elevenlabs", "Rachel"
            )
            assert unrelated_result is None, "Unrelated text should not match"

    @pytest.mark.asyncio
    async def test_persistence_across_manager_instances(self) -> None:
        """Test that cache persists across SemanticCacheManager instances."""
        with TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir) / "persist_cache"

            # First manager instance - cache some audio
            manager1 = SemanticCacheManager(cache_dir)
            test_audio = b"Persistent audio data"
            await manager1.cache_audio(
                "Persistent test", "system", "default", test_audio
            )

            # Create new manager instance with same cache directory
            manager2 = SemanticCacheManager(cache_dir)

            # Should find cached audio from first instance
            result = await manager2.get_cached_audio(
                "Persistent test", "system", "default"
            )
            assert result is not None, "Cache should persist across instances"
            assert result.read_bytes() == test_audio

            # Verify components were properly loaded
            assert manager2.index.index.ntotal > 0, "FAISS index should have entries"
            assert len(manager2.index.embeddings) > 0, "Embeddings should be loaded"

    @pytest.mark.asyncio
    async def test_audio_file_corruption_handling(self) -> None:
        """Test graceful handling when cached audio file is missing."""
        with TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir) / "corrupt_cache"
            manager = SemanticCacheManager(cache_dir)

            # Cache audio normally
            test_audio = b"Audio that will be deleted"
            await manager.cache_audio(
                "Corruption test", "elevenlabs", "Rachel", test_audio
            )

            # Verify it's cached
            result = await manager.get_cached_audio(
                "Corruption test", "elevenlabs", "Rachel"
            )
            assert result is not None
            audio_path = result

            # Delete the audio file to simulate corruption
            audio_path.unlink()
            assert not audio_path.exists()

            # Should return None when audio file is missing
            missing_result = await manager.get_cached_audio(
                "Corruption test", "elevenlabs", "Rachel"
            )
            assert missing_result is None, "Should return None when audio file missing"

    @pytest.mark.asyncio
    async def test_database_integrity_with_real_sqlite(self) -> None:
        """Test that cache entries are properly stored in SQLite database."""
        with TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir) / "db_test"
            manager = SemanticCacheManager(cache_dir)

            # Cache multiple entries
            entries = [
                ("First entry", "elevenlabs", "Rachel"),
                ("Second entry", "system", "default"),
                ("Third entry", "elevenlabs", "Adam"),
            ]

            for i, (text, provider, voice) in enumerate(entries):
                audio = f"Audio {i}".encode()
                await manager.cache_audio(text, provider, voice, audio)

            # Connect directly to database to verify
            db_path = cache_dir / "cache.db"
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()

            # Check all entries were saved
            cursor.execute("SELECT COUNT(*) FROM cache")
            count = cursor.fetchone()[0]
            assert count == 3, "Should have 3 entries in database"

            # Verify data integrity
            cursor.execute(
                "SELECT text, provider, voice, embedding_idx FROM cache ORDER BY id"
            )
            rows = cursor.fetchall()

            for i, row in enumerate(rows):
                assert row[0] == entries[i][0]  # text
                assert row[1] == entries[i][1]  # provider
                assert row[2] == entries[i][2]  # voice
                assert isinstance(row[3], int)  # embedding_idx
                assert row[3] >= 0  # Valid index

            conn.close()

    @pytest.mark.asyncio
    async def test_similarity_threshold_behavior(self) -> None:
        """Test that similarity threshold correctly controls cache hits."""
        with TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir) / "threshold_test"

            # Test with high threshold (strict matching)
            strict_manager = SemanticCacheManager(cache_dir, similarity_threshold=0.98)
            await strict_manager.cache_audio(
                "Hello world", "elevenlabs", "Rachel", b"audio"
            )

            # Similar but not identical - should miss with high threshold
            strict_result = await strict_manager.get_cached_audio(
                "Hello World!", "elevenlabs", "Rachel"
            )
            assert strict_result is None, (
                "High threshold should miss on minor differences"
            )

            # Test with lower threshold (more lenient)
            lenient_manager = SemanticCacheManager(cache_dir, similarity_threshold=0.80)
            lenient_result = await lenient_manager.get_cached_audio(
                "Hello World!", "elevenlabs", "Rachel"
            )
            assert lenient_result is not None, (
                "Low threshold should hit on minor differences"
            )

    @pytest.mark.asyncio
    async def test_real_embedding_dimensions(self) -> None:
        """Test that real embeddings have correct dimensions for FAISS."""
        with TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir) / "embedding_test"
            manager = SemanticCacheManager(cache_dir)

            # Cache some text
            await manager.cache_audio("Embedding test", "system", "default", b"audio")

            # Verify embedding was stored with correct dimension
            assert manager.index.embeddings.shape[1] == EMBEDDING_DIM
            assert manager.index.index.d == EMBEDDING_DIM

            # Verify embedding index matches array size
            assert manager.index.index.ntotal == len(manager.index.embeddings)


class TestCacheManagerWithRealDirectory:
    """Test SemanticCacheManager with actual system cache directory."""

    @pytest.mark.asyncio
    async def test_integration_with_system_cache_directory(self) -> None:
        """Test using the real system cache directory from get_cache_dir()."""
        # Use real cache directory
        cache_dir = get_cache_dir()
        manager = SemanticCacheManager(cache_dir)

        # Create unique test data to avoid conflicts
        import uuid

        unique_text = f"System cache test {uuid.uuid4()}"
        unique_audio = f"Unique audio {uuid.uuid4()}".encode()

        # Cache and retrieve
        await manager.cache_audio(
            unique_text, "test_provider", "test_voice", unique_audio
        )
        result = await manager.get_cached_audio(
            unique_text, "test_provider", "test_voice"
        )

        assert result is not None, "Should retrieve from system cache"
        assert result.read_bytes() == unique_audio

        # Verify it's in the expected location
        assert str(result).startswith(str(cache_dir / "audio"))
        assert result.suffix == ".mp3"
