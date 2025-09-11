"""Unit tests for cache models and storage type conversion logic."""

import sys
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from lspeak.cache.models import CacheEntry
from lspeak.cache.storage import CacheStorage


class TestCacheEntry:
    """Test CacheEntry dataclass functionality."""

    def test_cache_entry_creation_with_all_fields(self) -> None:
        """Test creating CacheEntry with all required fields."""
        timestamp = datetime.now()
        audio_path = Path("/tmp/test.mp3")

        entry = CacheEntry(
            text="Hello world",
            provider="elevenlabs",
            voice="Rachel",
            audio_path=audio_path,
            embedding_idx=42,
            timestamp=timestamp,
        )

        assert entry.text == "Hello world"
        assert entry.provider == "elevenlabs"
        assert entry.voice == "Rachel"
        assert entry.audio_path == audio_path
        assert entry.embedding_idx == 42
        assert entry.timestamp == timestamp


class TestCacheStorageTypeConversion:
    """Test CacheStorage type conversion logic."""

    def test_datetime_iso_string_conversion_roundtrip(self) -> None:
        """Test datetime to ISO string conversion and back preserves value."""
        with TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir)
            storage = CacheStorage(cache_dir)

            # Create entry with specific datetime
            original_time = datetime(2023, 12, 25, 14, 30, 45, 123456)
            entry = CacheEntry(
                text="conversion test",
                provider="system",
                voice="default",
                audio_path=Path("/tmp/test.mp3"),
                embedding_idx=1,
                timestamp=original_time,
            )

            # Save and retrieve
            storage.save(entry)
            retrieved = storage.get_by_embedding_idx(1)

            # Verify datetime conversion preserved value
            assert retrieved is not None
            assert retrieved.timestamp == original_time

            storage.close()

    def test_path_string_conversion_roundtrip(self) -> None:
        """Test Path to string conversion and back preserves value."""
        with TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir)
            storage = CacheStorage(cache_dir)

            # Create entry with complex path
            original_path = Path("/complex/path/with spaces/audio.mp3")
            entry = CacheEntry(
                text="path test",
                provider="elevenlabs",
                voice="Adam",
                audio_path=original_path,
                embedding_idx=2,
                timestamp=datetime.now(),
            )

            # Save and retrieve
            storage.save(entry)
            retrieved = storage.get_by_embedding_idx(2)

            # Verify path conversion preserved value
            assert retrieved is not None
            assert retrieved.audio_path == original_path
            assert isinstance(retrieved.audio_path, Path)

            storage.close()

    def test_datetime_microseconds_preserved(self) -> None:
        """Test that datetime microseconds are preserved through conversion."""
        with TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir)
            storage = CacheStorage(cache_dir)

            # Create datetime with microseconds
            precise_time = datetime(2023, 1, 1, 12, 0, 0, 999999)
            entry = CacheEntry(
                text="microseconds test",
                provider="elevenlabs",
                voice="Bella",
                audio_path=Path("/tmp/precise.mp3"),
                embedding_idx=3,
                timestamp=precise_time,
            )

            # Save and retrieve
            storage.save(entry)
            retrieved = storage.get_by_embedding_idx(3)

            # Verify microseconds preserved
            assert retrieved is not None
            assert retrieved.timestamp.microsecond == 999999
            assert retrieved.timestamp == precise_time

            storage.close()

    def test_absolute_vs_relative_path_handling(self) -> None:
        """Test that both absolute and relative paths are handled correctly."""
        with TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir)
            storage = CacheStorage(cache_dir)

            # Test absolute path
            abs_path = Path("/absolute/path/audio.mp3")
            abs_entry = CacheEntry(
                text="absolute path",
                provider="system",
                voice="voice1",
                audio_path=abs_path,
                embedding_idx=4,
                timestamp=datetime.now(),
            )

            # Test relative path
            rel_path = Path("relative/audio.mp3")
            rel_entry = CacheEntry(
                text="relative path",
                provider="system",
                voice="voice2",
                audio_path=rel_path,
                embedding_idx=5,
                timestamp=datetime.now(),
            )

            # Save both
            storage.save(abs_entry)
            storage.save(rel_entry)

            # Retrieve and verify
            retrieved_abs = storage.get_by_embedding_idx(4)
            retrieved_rel = storage.get_by_embedding_idx(5)

            assert retrieved_abs is not None
            assert retrieved_abs.audio_path == abs_path
            assert retrieved_abs.audio_path.is_absolute() == abs_path.is_absolute()

            assert retrieved_rel is not None
            assert retrieved_rel.audio_path == rel_path
            assert retrieved_rel.audio_path.is_absolute() == rel_path.is_absolute()

            storage.close()
