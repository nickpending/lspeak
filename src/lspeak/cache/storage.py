"""SQLite cache storage implementation."""

import sqlite3
from datetime import datetime
from pathlib import Path

from .models import CacheEntry


class CacheStorage:
    """SQLite-based cache storage for TTS metadata.

    Stores cache entry metadata in SQLite database while audio files
    are stored separately on the filesystem.
    """

    def __init__(self, cache_dir: Path):
        """Initialize cache storage with database in given directory.

        Args:
            cache_dir: Directory containing cache database
        """
        self.cache_dir = cache_dir

        # Create cache directory if it doesn't exist
        cache_dir.mkdir(parents=True, exist_ok=True)

        self.db_path = cache_dir / "cache.db"
        
        # Initialize DB with WAL mode for concurrency
        self._init_db_with_wal()
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get a new database connection with WAL mode for concurrency."""
        conn = sqlite3.connect(
            str(self.db_path),
            timeout=30.0,  # 30 second timeout if locked
            check_same_thread=False  # Allow use across threads
        )
        conn.row_factory = sqlite3.Row
        # Enable WAL mode for much better concurrent access
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")  # 30s retry on lock
        conn.execute("PRAGMA synchronous=NORMAL")  # Faster writes
        return conn
    
    def _init_db_with_wal(self) -> None:
        """Initialize database with WAL mode and schema."""
        # Get connection for initialization
        self.connection = self._get_connection()
        self._init_db()
        self.connection.close()
        # Don't keep connection - each operation gets its own

    def _init_db(self) -> None:
        """Initialize database schema with tables and indexes."""
        cursor = self.connection.cursor()

        # Create cache table with unique constraint on (text, provider, voice)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                provider TEXT NOT NULL,
                voice TEXT NOT NULL,
                audio_path TEXT NOT NULL,
                embedding_idx INTEGER NOT NULL,
                timestamp TEXT NOT NULL,
                UNIQUE(text, provider, voice)
            )
        """)

        # Create index on embedding_idx for fast FAISS integration
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_embedding 
            ON cache(embedding_idx)
        """)

        self.connection.commit()

    def save(self, entry: CacheEntry) -> None:
        """Save cache entry to database.

        Args:
            entry: Cache entry to save

        Raises:
            sqlite3.IntegrityError: If entry with same (text, provider, voice) exists
        """
        # Get fresh connection for this operation
        conn = self._get_connection()
        try:
            cursor = conn.cursor()

            # Convert datetime to ISO string and Path to string for storage
            timestamp_str = entry.timestamp.isoformat()
            audio_path_str = str(entry.audio_path)

            cursor.execute(
                """
                INSERT INTO cache (text, provider, voice, audio_path, embedding_idx, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                (
                    entry.text,
                    entry.provider,
                    entry.voice,
                    audio_path_str,
                    entry.embedding_idx,
                    timestamp_str,
                ),
            )

            conn.commit()
        finally:
            conn.close()

    def get_by_embedding_idx(self, embedding_idx: int) -> CacheEntry | None:
        """Retrieve cache entry by embedding index.

        Args:
            embedding_idx: FAISS embedding index to look up

        Returns:
            Cache entry if found, None otherwise
        """
        # Get fresh connection for this operation
        conn = self._get_connection()
        try:
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT text, provider, voice, audio_path, embedding_idx, timestamp
                FROM cache
                WHERE embedding_idx = ?
            """,
                (embedding_idx,),
            )

            row = cursor.fetchone()
        finally:
            conn.close()
        if row is None:
            return None

        # Convert stored strings back to proper types
        timestamp = datetime.fromisoformat(row["timestamp"])
        audio_path = Path(row["audio_path"])

        return CacheEntry(
            text=row["text"],
            provider=row["provider"],
            voice=row["voice"],
            audio_path=audio_path,
            embedding_idx=row["embedding_idx"],
            timestamp=timestamp,
        )

    def close(self) -> None:
        """Close database connection (no-op now since we use per-request connections)."""
        # No persistent connection to close anymore
        pass
