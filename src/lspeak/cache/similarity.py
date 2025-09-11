"""FAISS-based similarity search for semantic caching."""

import logging
from pathlib import Path

import faiss
import numpy as np

from ..embeddings.models import EMBEDDING_DIM

logger = logging.getLogger(__name__)


class SimilarityIndex:
    """FAISS-based similarity search for cached embeddings.

    Manages a FAISS IndexFlatL2 and corresponding numpy embeddings array
    for fast similarity search in semantic caching. Provides methods to
    search for similar embeddings above a configurable threshold and
    append new embeddings with persistent storage.
    """

    def __init__(self, cache_dir: Path):
        """Initialize similarity index with cache directory.

        Args:
            cache_dir: Directory for storing FAISS index and embeddings files
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.index_path = self.cache_dir / "faiss.index"
        self.embeddings_path = self.cache_dir / "embeddings.npy"

        # Task 5.1: Initialize FAISS index
        self.index = self._load_or_create_index()

        # Task 5.2: Load existing embeddings and sync with index
        self.embeddings = self._load_embeddings()
        self._sync_index_with_embeddings()

    def _load_or_create_index(self) -> faiss.IndexFlatL2:
        """Load existing FAISS index or create new one.

        Returns:
            FAISS IndexFlatL2 with EMBEDDING_DIM dimensions
        """
        if self.index_path.exists():
            try:
                index = faiss.read_index(str(self.index_path))
                logger.debug(f"Loaded existing FAISS index with {index.ntotal} entries")
                return index
            except Exception as e:
                logger.warning(f"Failed to load FAISS index, creating new one: {e}")

        # Create new index with correct dimensions
        index = faiss.IndexFlatL2(EMBEDDING_DIM)
        logger.debug(f"Created new FAISS index with dimension {EMBEDDING_DIM}")
        return index

    def _load_embeddings(self) -> np.ndarray:
        """Load embeddings numpy array from file.

        Returns:
            Numpy array of embeddings, shape (n, EMBEDDING_DIM)
        """
        if self.embeddings_path.exists():
            try:
                embeddings = np.load(self.embeddings_path, allow_pickle=False)
                logger.debug(
                    f"Loaded {len(embeddings)} embeddings from {self.embeddings_path}"
                )
                return embeddings
            except Exception as e:
                logger.warning(
                    f"Failed to load embeddings, starting with empty array: {e}"
                )

        # Return empty array with correct shape and dtype for FAISS
        return np.empty((0, EMBEDDING_DIM), dtype=np.float32)

    def _save_embeddings(self, embeddings: np.ndarray) -> None:
        """Save embeddings numpy array to file.

        Args:
            embeddings: Numpy array to save
        """
        try:
            np.save(self.embeddings_path, embeddings, allow_pickle=False)
            logger.debug(
                f"Saved {len(embeddings)} embeddings to {self.embeddings_path}"
            )
        except Exception as e:
            logger.error(f"Failed to save embeddings: {e}")

    def _sync_index_with_embeddings(self) -> None:
        """Synchronize FAISS index with loaded embeddings array."""
        if len(self.embeddings) > 0:
            # Clear index and re-add all embeddings
            self.index.reset()
            self.index.add(self.embeddings)
            logger.debug(
                f"Synchronized FAISS index with {len(self.embeddings)} embeddings"
            )

    def search(
        self, query_embedding: np.ndarray, threshold: float = 0.95, k: int = 10
    ) -> tuple[bool, int | None]:
        """Search for similar embedding above threshold.

        Args:
            query_embedding: Embedding to search for, shape (EMBEDDING_DIM,)
            threshold: Similarity threshold (0.0 to 1.0)
            k: Number of nearest neighbors to consider (for multiple candidates)

        Returns:
            Tuple of (hit: bool, index: Optional[int])
            - hit: True if similar embedding found above threshold
            - index: Index position in embeddings array if hit, None otherwise
        """
        # Validate input
        if query_embedding.shape != (EMBEDDING_DIM,):
            raise ValueError(
                f"Query embedding must have shape ({EMBEDDING_DIM},), got {query_embedding.shape}"
            )

        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"Threshold must be between 0.0 and 1.0, got {threshold}")

        # Handle empty index
        if self.index.ntotal == 0:
            return False, None

        # Prepare query for FAISS (needs shape (1, EMBEDDING_DIM))
        query = query_embedding.astype(np.float32).reshape(1, -1)

        # Search for k nearest neighbors (in case multiple cache entries exist)
        # Use min(k, total entries) to avoid errors when index has fewer entries
        search_k = min(k, self.index.ntotal)
        distances, indices = self.index.search(query, search_k)

        # Return first match above threshold (caller will check provider/voice)
        for i in range(search_k):
            # Convert L2 distance to similarity
            # For normalized embeddings, max L2 distance is approximately 2.0
            distance = distances[0][i]
            similarity = 1.0 - (distance / 2.0)

            # Return first match above threshold
            if similarity >= threshold:
                return True, int(indices[0][i])

        return False, None

    def search_multiple(
        self,
        query_embedding: np.ndarray,
        threshold: float = 0.95,
        max_results: int = 10,
    ) -> list[tuple[int, float]]:
        """Search for multiple similar embeddings above threshold.

        Args:
            query_embedding: Embedding to search for, shape (EMBEDDING_DIM,)
            threshold: Similarity threshold (0.0 to 1.0)
            max_results: Maximum number of results to return

        Returns:
            List of (index, similarity) tuples for all matches above threshold,
            ordered by similarity (highest first)
        """
        # Validate input
        if query_embedding.shape != (EMBEDDING_DIM,):
            raise ValueError(
                f"Query embedding must have shape ({EMBEDDING_DIM},), got {query_embedding.shape}"
            )

        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"Threshold must be between 0.0 and 1.0, got {threshold}")

        # Handle empty index
        if self.index.ntotal == 0:
            return []

        # Prepare query for FAISS (needs shape (1, EMBEDDING_DIM))
        query = query_embedding.astype(np.float32).reshape(1, -1)

        # Search for max_results nearest neighbors
        search_k = min(max_results, self.index.ntotal)
        distances, indices = self.index.search(query, search_k)

        # Collect all results above threshold
        results = []
        for i in range(search_k):
            # Convert L2 distance to similarity
            distance = distances[0][i]
            similarity = 1.0 - (distance / 2.0)

            # Add if above threshold
            if similarity >= threshold:
                results.append((int(indices[0][i]), float(similarity)))
            else:
                # Since results are ordered by similarity, we can stop here
                break

        return results

    def add(self, embedding: np.ndarray) -> int:
        """Add new embedding to index and return position.

        Args:
            embedding: Embedding to add, shape (EMBEDDING_DIM,)

        Returns:
            Index position of added embedding
        """
        # Validate input
        if embedding.shape != (EMBEDDING_DIM,):
            raise ValueError(
                f"Embedding must have shape ({EMBEDDING_DIM},), got {embedding.shape}"
            )

        # Get position for new embedding
        new_idx = len(self.embeddings)

        # Convert to float32 for FAISS compatibility
        embedding_f32 = embedding.astype(np.float32)

        # Append to embeddings array
        if len(self.embeddings) == 0:
            # First embedding - create new array
            self.embeddings = embedding_f32.reshape(1, -1)
        else:
            # Append to existing array
            self.embeddings = np.vstack([self.embeddings, embedding_f32.reshape(1, -1)])

        # Add to FAISS index
        self.index.add(embedding_f32.reshape(1, -1))

        # Persist both files
        self._save_embeddings(self.embeddings)
        self._save_index()

        logger.debug(
            f"Added embedding at index {new_idx}, total embeddings: {len(self.embeddings)}"
        )
        return new_idx

    def _save_index(self) -> None:
        """Save FAISS index to file."""
        try:
            faiss.write_index(self.index, str(self.index_path))
            logger.debug(f"Saved FAISS index with {self.index.ntotal} entries")
        except Exception as e:
            logger.error(f"Failed to save FAISS index: {e}")
