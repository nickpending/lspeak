"""Integration tests for cache storage system with real SQLite database."""

import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from lspeak.cache import get_cache_dir
from lspeak.cache.models import CacheEntry
from lspeak.cache.storage import CacheStorage


class TestCacheStorageIntegration:
    """Test complete cache storage workflow with real database."""

    def test_complete_cache_storage_workflow(self) -> None:
        """
        Test complete cache storage workflow:
        - Initialize cache directory and database
        - Create cache entry
        - Save to database
        - Retrieve by embedding index
        - Verify data integrity
        """
        with TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir) / "lspeak_cache"

            # Initialize storage (creates database)
            storage = CacheStorage(cache_dir)

            # Create test entry
            test_audio_path = Path("/tmp/test_audio.mp3")
            test_timestamp = datetime(2023, 11, 15, 10, 30, 45)

            entry = CacheEntry(
                text="Integration test text",
                provider="elevenlabs",
                voice="Rachel",
                audio_path=test_audio_path,
                embedding_idx=123,
                timestamp=test_timestamp,
            )

            # Save entry to database
            storage.save(entry)

            # Retrieve entry by embedding index
            retrieved = storage.get_by_embedding_idx(123)

            # Verify complete data integrity
            assert retrieved is not None
            assert retrieved.text == "Integration test text"
            assert retrieved.provider == "elevenlabs"
            assert retrieved.voice == "Rachel"
            assert retrieved.audio_path == test_audio_path
            assert retrieved.embedding_idx == 123
            assert retrieved.timestamp == test_timestamp

            storage.close()

    def test_cache_storage_with_multiple_entries(self) -> None:
        """Test storage and retrieval of multiple cache entries."""
        with TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir) / "multi_cache"
            storage = CacheStorage(cache_dir)

            # Create multiple entries with different embedding indices
            entries = [
                CacheEntry(
                    text=f"Test text {i}",
                    provider="elevenlabs" if i % 2 == 0 else "system",
                    voice=f"voice_{i}",
                    audio_path=Path(f"/tmp/audio_{i}.mp3"),
                    embedding_idx=i * 10,
                    timestamp=datetime.now(),
                )
                for i in range(5)
            ]

            # Save all entries
            for entry in entries:
                storage.save(entry)

            # Retrieve and verify each entry
            for i, original_entry in enumerate(entries):
                retrieved = storage.get_by_embedding_idx(i * 10)

                assert retrieved is not None
                assert retrieved.text == original_entry.text
                assert retrieved.provider == original_entry.provider
                assert retrieved.voice == original_entry.voice
                assert retrieved.embedding_idx == original_entry.embedding_idx

            storage.close()

    def test_cache_storage_unique_constraint_behavior(self) -> None:
        """Test unique constraint on (text, provider, voice) combination."""
        with TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir) / "constraint_cache"
            storage = CacheStorage(cache_dir)

            # Create first entry
            entry1 = CacheEntry(
                text="Same text",
                provider="elevenlabs",
                voice="Rachel",
                audio_path=Path("/tmp/first.mp3"),
                embedding_idx=100,
                timestamp=datetime.now(),
            )

            # Save first entry
            storage.save(entry1)

            # Create second entry with same text, provider, voice
            entry2 = CacheEntry(
                text="Same text",  # Same
                provider="elevenlabs",  # Same
                voice="Rachel",  # Same
                audio_path=Path("/tmp/second.mp3"),  # Different
                embedding_idx=200,  # Different
                timestamp=datetime.now(),
            )

            # Attempting to save should raise IntegrityError due to unique constraint
            with pytest.raises(sqlite3.IntegrityError):
                storage.save(entry2)

            # Verify original entry is still retrievable
            retrieved = storage.get_by_embedding_idx(100)
            assert retrieved is not None
            assert retrieved.audio_path == Path("/tmp/first.mp3")

            storage.close()

    def test_cache_storage_handles_missing_entries(self) -> None:
        """Test that retrieving non-existent entries returns None."""
        with TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir) / "missing_cache"
            storage = CacheStorage(cache_dir)

            # Try to retrieve non-existent entry
            result = storage.get_by_embedding_idx(999)
            assert result is None

            # Add one entry
            entry = CacheEntry(
                text="Only entry",
                provider="system",
                voice="default",
                audio_path=Path("/tmp/only.mp3"),
                embedding_idx=50,
                timestamp=datetime.now(),
            )
            storage.save(entry)

            # Verify existing entry works
            retrieved = storage.get_by_embedding_idx(50)
            assert retrieved is not None
            assert retrieved.text == "Only entry"

            # Verify non-existent still returns None
            missing = storage.get_by_embedding_idx(999)
            assert missing is None

            storage.close()


class TestCacheDirectoryAndDatabaseInitialization:
    """Test cache directory creation and database schema initialization."""

    def test_cache_directory_creation_with_get_cache_dir(self) -> None:
        """Test that get_cache_dir creates proper directory structure."""
        # Use system cache directory (will be cleaned up)
        cache_dir = get_cache_dir()

        # Verify main cache directory exists
        assert cache_dir.exists()
        assert cache_dir.is_dir()

        # Verify audio subdirectory exists
        audio_dir = cache_dir / "audio"
        assert audio_dir.exists()
        assert audio_dir.is_dir()

        # Verify path structure
        assert cache_dir.name == "lspeak"
        assert cache_dir.parent.name == ".cache"

    def test_database_schema_initialization(self) -> None:
        """Test that CacheStorage creates proper SQLite schema."""
        with TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir) / "schema_test"

            # Initialize storage (should create database and schema)
            storage = CacheStorage(cache_dir)

            # Verify database file was created
            db_path = cache_dir / "cache.db"
            assert db_path.exists()

            # Connect directly to database to verify schema
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()

            # Check that cache table exists with correct columns
            cursor.execute("PRAGMA table_info(cache)")
            columns = cursor.fetchall()

            # Verify expected columns exist
            column_names = [col[1] for col in columns]
            expected_columns = [
                "id",
                "text",
                "provider",
                "voice",
                "audio_path",
                "embedding_idx",
                "timestamp",
            ]

            for expected_col in expected_columns:
                assert expected_col in column_names, f"Column {expected_col} missing"

            # Check for unique constraint
            cursor.execute("PRAGMA index_list(cache)")
            indexes = cursor.fetchall()

            # Should have at least the unique constraint index and embedding index
            assert len(indexes) >= 1, "Missing expected indexes"

            # Verify embedding index exists
            cursor.execute("PRAGMA index_info(idx_embedding)")
            embedding_index_info = cursor.fetchall()
            assert len(embedding_index_info) > 0, "Embedding index not created"

            conn.close()
            storage.close()

    def test_storage_initialization_creates_directory_if_missing(self) -> None:
        """Test that CacheStorage creates cache directory if it doesn't exist."""
        with TemporaryDirectory() as temp_dir:
            # Use non-existent subdirectory
            cache_dir = Path(temp_dir) / "does_not_exist" / "lspeak_cache"

            # Verify directory doesn't exist initially
            assert not cache_dir.exists()

            # Initialize storage
            storage = CacheStorage(cache_dir)

            # Verify directory was created
            assert cache_dir.exists()
            assert cache_dir.is_dir()

            # Verify database was created
            db_path = cache_dir / "cache.db"
            assert db_path.exists()

            storage.close()

    def test_integration_with_real_cache_directory(self) -> None:
        """Test integration using actual cache directory structure."""
        # Get real cache directory
        cache_dir = get_cache_dir()

        # Initialize storage with real directory
        storage = CacheStorage(cache_dir)

        # Create unique test entry to avoid constraint violations
        import uuid

        unique_text = f"Real directory test {uuid.uuid4()}"
        unique_embedding_idx = abs(hash(unique_text)) % 100000

        entry = CacheEntry(
            text=unique_text,
            provider="elevenlabs",
            voice="Adam",
            audio_path=cache_dir / "audio" / "test_real.mp3",
            embedding_idx=unique_embedding_idx,
            timestamp=datetime.now(),
        )

        storage.save(entry)

        # Retrieve and verify
        retrieved = storage.get_by_embedding_idx(unique_embedding_idx)
        assert retrieved is not None
        assert retrieved.text == unique_text

        # Verify database exists in expected location
        db_path = cache_dir / "cache.db"
        assert db_path.exists()

        storage.close()
