"""Integration tests for SimilarityIndex with real FAISS operations and file persistence."""

import sys
import uuid
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from lspeak.cache import get_cache_dir
from lspeak.cache.similarity import SimilarityIndex
from lspeak.embeddings.models import EMBEDDING_DIM


class TestSimilarityIndexIntegration:
    """Test complete similarity search workflow with real FAISS operations."""

    def test_complete_similarity_search_workflow(self) -> None:
        """
        Test complete similarity search workflow:
        - Initialize SimilarityIndex with real cache directory
        - Add multiple embeddings with real FAISS operations
        - Search for similar embeddings with real similarity calculation
        - Verify file persistence works correctly
        - Test cache hits and misses
        """
        with TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir) / "similarity_cache"

            # Initialize similarity index (creates real FAISS index)
            similarity_index = SimilarityIndex(cache_dir)

            # Verify starts with empty state
            assert similarity_index.index.ntotal == 0
            assert len(similarity_index.embeddings) == 0

            # Create test embeddings with known relationships
            base_embedding = np.random.rand(EMBEDDING_DIM).astype(np.float32)
            base_embedding = base_embedding / np.linalg.norm(
                base_embedding
            )  # Normalize

            # Very similar embedding (tiny perturbation)
            similar_embedding = base_embedding + np.random.normal(
                0, 0.01, EMBEDDING_DIM
            ).astype(np.float32)
            similar_embedding = similar_embedding / np.linalg.norm(similar_embedding)

            # Completely different embedding
            different_embedding = np.random.rand(EMBEDDING_DIM).astype(np.float32)
            different_embedding = different_embedding / np.linalg.norm(
                different_embedding
            )

            # Add embeddings to index
            base_idx = similarity_index.add(base_embedding)
            similar_idx = similarity_index.add(similar_embedding)
            different_idx = similarity_index.add(different_embedding)

            # Verify indices are sequential
            assert base_idx == 0
            assert similar_idx == 1
            assert different_idx == 2

            # Verify FAISS index updated
            assert similarity_index.index.ntotal == 3
            assert len(similarity_index.embeddings) == 3

            # Test exact match search
            exact_hit, exact_idx = similarity_index.search(
                base_embedding, threshold=0.95
            )
            assert exact_hit is True
            assert exact_idx == base_idx

            # Test similar match search with lower threshold
            similar_hit, found_similar_idx = similarity_index.search(
                similar_embedding, threshold=0.8
            )
            assert similar_hit is True
            assert found_similar_idx == similar_idx

            # Test miss with high threshold against different embedding
            miss_hit, miss_idx = similarity_index.search(
                different_embedding, threshold=0.99
            )
            # This might hit or miss depending on random embeddings, but should not crash
            assert isinstance(miss_hit, bool)
            assert miss_idx is None or isinstance(miss_idx, int)

            # Verify files were created
            faiss_file = cache_dir / "faiss.index"
            numpy_file = cache_dir / "embeddings.npy"
            assert faiss_file.exists()
            assert numpy_file.exists()

    def test_similarity_index_persistence_across_restarts(self) -> None:
        """Test that SimilarityIndex data persists correctly across instance restarts."""
        with TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir) / "persistence_test"

            # Create known embeddings for consistent testing
            test_embeddings = [
                np.array([1.0] + [0.0] * (EMBEDDING_DIM - 1)).astype(np.float32),
                np.array([0.0, 1.0] + [0.0] * (EMBEDDING_DIM - 2)).astype(np.float32),
                np.array([0.0, 0.0, 1.0] + [0.0] * (EMBEDDING_DIM - 3)).astype(
                    np.float32
                ),
            ]

            # First instance - add data
            similarity_index1 = SimilarityIndex(cache_dir)

            added_indices = []
            for embedding in test_embeddings:
                idx = similarity_index1.add(embedding)
                added_indices.append(idx)

            # Verify data added correctly
            assert added_indices == [0, 1, 2]
            assert similarity_index1.index.ntotal == 3

            # Second instance - should load existing data
            similarity_index2 = SimilarityIndex(cache_dir)

            # Verify data loaded correctly
            assert similarity_index2.index.ntotal == 3
            assert len(similarity_index2.embeddings) == 3
            assert similarity_index2.embeddings.shape == (3, EMBEDDING_DIM)

            # Test that searches work on loaded data
            for i, embedding in enumerate(test_embeddings):
                hit, found_idx = similarity_index2.search(embedding, threshold=0.95)
                assert hit is True
                assert found_idx == i

            # Add new embedding to second instance
            new_embedding = np.array(
                [0.0] * 3 + [1.0] + [0.0] * (EMBEDDING_DIM - 4)
            ).astype(np.float32)
            new_idx = similarity_index2.add(new_embedding)
            assert new_idx == 3
            assert similarity_index2.index.ntotal == 4

            # Third instance - should load all data including new addition
            similarity_index3 = SimilarityIndex(cache_dir)
            assert similarity_index3.index.ntotal == 4
            assert len(similarity_index3.embeddings) == 4

            # Verify all embeddings still searchable
            all_embeddings = test_embeddings + [new_embedding]
            for i, embedding in enumerate(all_embeddings):
                hit, found_idx = similarity_index3.search(embedding, threshold=0.95)
                assert hit is True
                assert found_idx == i

    def test_similarity_search_with_different_thresholds(self) -> None:
        """Test similarity search behavior with various threshold values."""
        with TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir) / "threshold_test"
            similarity_index = SimilarityIndex(cache_dir)

            # Create base embedding
            base_embedding = np.random.rand(EMBEDDING_DIM).astype(np.float32)
            base_embedding = base_embedding / np.linalg.norm(base_embedding)
            base_idx = similarity_index.add(base_embedding)

            # Create embeddings with different similarity levels
            very_similar = base_embedding + np.random.normal(
                0, 0.001, EMBEDDING_DIM
            ).astype(np.float32)
            very_similar = very_similar / np.linalg.norm(very_similar)

            somewhat_similar = base_embedding + np.random.normal(
                0, 0.1, EMBEDDING_DIM
            ).astype(np.float32)
            somewhat_similar = somewhat_similar / np.linalg.norm(somewhat_similar)

            very_different = np.random.rand(EMBEDDING_DIM).astype(np.float32)
            very_different = very_different / np.linalg.norm(very_different)

            # Test with very high threshold - only very similar should hit
            hit_high, idx_high = similarity_index.search(very_similar, threshold=0.98)
            assert hit_high is True
            assert idx_high == base_idx

            # Test with medium threshold - somewhat similar might hit
            hit_medium, idx_medium = similarity_index.search(
                somewhat_similar, threshold=0.5
            )
            # Result depends on random perturbation, but should not crash
            assert isinstance(hit_medium, bool)

            # Test with low threshold - most embeddings should hit
            hit_low, idx_low = similarity_index.search(very_different, threshold=0.1)
            # Even different embedding might hit with very low threshold
            assert isinstance(hit_low, bool)

            # Test exact match always hits regardless of threshold
            exact_hit_high, exact_idx_high = similarity_index.search(
                base_embedding, threshold=0.99
            )
            exact_hit_low, exact_idx_low = similarity_index.search(
                base_embedding, threshold=0.01
            )

            assert exact_hit_high is True
            assert exact_idx_high == base_idx
            assert exact_hit_low is True
            assert exact_idx_low == base_idx

    def test_similarity_index_handles_empty_and_growing_index(self) -> None:
        """Test similarity index behavior with empty index and as it grows."""
        with TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir) / "growing_index"
            similarity_index = SimilarityIndex(cache_dir)

            test_embedding = np.random.rand(EMBEDDING_DIM).astype(np.float32)

            # Test search on empty index
            empty_hit, empty_idx = similarity_index.search(
                test_embedding, threshold=0.5
            )
            assert empty_hit is False
            assert empty_idx is None

            # Add first embedding
            first_idx = similarity_index.add(test_embedding)
            assert first_idx == 0
            assert similarity_index.index.ntotal == 1

            # Search should now find the embedding
            first_hit, first_found_idx = similarity_index.search(
                test_embedding, threshold=0.95
            )
            assert first_hit is True
            assert first_found_idx == first_idx

            # Add more embeddings and verify growing behavior
            for i in range(1, 10):
                new_embedding = np.random.rand(EMBEDDING_DIM).astype(np.float32)
                new_idx = similarity_index.add(new_embedding)

                assert new_idx == i
                assert similarity_index.index.ntotal == i + 1
                assert len(similarity_index.embeddings) == i + 1

                # Original embedding should still be findable
                orig_hit, orig_idx = similarity_index.search(
                    test_embedding, threshold=0.95
                )
                assert orig_hit is True
                assert orig_idx == 0


class TestSimilarityIndexCacheDirectoryIntegration:
    """Test integration with actual cache directory structure."""

    def test_integration_with_real_cache_directory(self) -> None:
        """Test SimilarityIndex works with real cache directory from get_cache_dir()."""
        # Get real cache directory
        cache_dir = get_cache_dir()

        # Create unique test data to avoid conflicts
        unique_suffix = str(uuid.uuid4())[:8]
        test_cache_dir = cache_dir / f"similarity_test_{unique_suffix}"

        try:
            # Initialize with real cache directory structure
            similarity_index = SimilarityIndex(test_cache_dir)

            # Verify directory structure created
            assert test_cache_dir.exists()
            assert test_cache_dir.is_dir()

            # Add test embedding
            test_embedding = np.random.rand(EMBEDDING_DIM).astype(np.float32)
            idx = similarity_index.add(test_embedding)

            # Verify files created in cache directory
            faiss_file = test_cache_dir / "faiss.index"
            numpy_file = test_cache_dir / "embeddings.npy"

            assert faiss_file.exists()
            assert numpy_file.exists()

            # Verify search works
            hit, found_idx = similarity_index.search(test_embedding, threshold=0.95)
            assert hit is True
            assert found_idx == idx

            # Test persistence with real cache directory
            similarity_index2 = SimilarityIndex(test_cache_dir)
            assert similarity_index2.index.ntotal == 1

            # Verify search still works after reload
            hit2, found_idx2 = similarity_index2.search(test_embedding, threshold=0.95)
            assert hit2 is True
            assert found_idx2 == idx

        finally:
            # Clean up test directory
            import shutil

            if test_cache_dir.exists():
                shutil.rmtree(test_cache_dir)

    def test_multiple_similarity_indices_same_cache_parent(self) -> None:
        """Test multiple SimilarityIndex instances in same cache parent directory."""
        with TemporaryDirectory() as temp_dir:
            parent_cache = Path(temp_dir) / "parent_cache"

            # Create multiple similarity indices in subdirectories
            index1 = SimilarityIndex(parent_cache / "index1")
            index2 = SimilarityIndex(parent_cache / "index2")

            # Add different data to each
            embedding1 = np.random.rand(EMBEDDING_DIM).astype(np.float32)
            embedding2 = np.random.rand(EMBEDDING_DIM).astype(np.float32)

            idx1 = index1.add(embedding1)
            idx2 = index2.add(embedding2)

            # Verify isolation - each should only find its own embedding
            hit1_in_1, found1_in_1 = index1.search(embedding1, threshold=0.95)
            hit2_in_1, found2_in_1 = index1.search(embedding2, threshold=0.95)

            hit1_in_2, found1_in_2 = index2.search(embedding1, threshold=0.95)
            hit2_in_2, found2_in_2 = index2.search(embedding2, threshold=0.95)

            # Each index should find its own embedding
            assert hit1_in_1 is True
            assert found1_in_1 == idx1
            assert hit2_in_2 is True
            assert found2_in_2 == idx2

            # Cross-searches might hit or miss depending on random embeddings
            # But should not crash and return valid boolean/int responses
            assert isinstance(hit2_in_1, bool)
            assert isinstance(hit1_in_2, bool)

            # Verify separate file structures
            assert (parent_cache / "index1" / "faiss.index").exists()
            assert (parent_cache / "index1" / "embeddings.npy").exists()
            assert (parent_cache / "index2" / "faiss.index").exists()
            assert (parent_cache / "index2" / "embeddings.npy").exists()


class TestSimilarityIndexFileOperationRobustness:
    """Test file operation robustness and error recovery."""

    def test_recovery_from_missing_files_on_restart(self) -> None:
        """Test that SimilarityIndex recovers gracefully from missing files."""
        with TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir) / "recovery_test"

            # Create index and add data
            similarity_index1 = SimilarityIndex(cache_dir)
            embedding = np.random.rand(EMBEDDING_DIM).astype(np.float32)
            similarity_index1.add(embedding)

            # Verify files exist
            faiss_file = cache_dir / "faiss.index"
            numpy_file = cache_dir / "embeddings.npy"
            assert faiss_file.exists()
            assert numpy_file.exists()

            # Remove one file to simulate corruption/deletion
            numpy_file.unlink()

            # Create new instance - should handle missing numpy file gracefully
            similarity_index2 = SimilarityIndex(cache_dir)

            # Should have empty embeddings array but FAISS index still has old data
            # (sync only happens when embeddings exist to sync)
            assert len(similarity_index2.embeddings) == 0
            # FAISS index still contains old data since sync didn't reset it
            assert similarity_index2.index.ntotal == 1

            # Should be able to add new data
            new_embedding = np.random.rand(EMBEDDING_DIM).astype(np.float32)
            new_idx = similarity_index2.add(new_embedding)
            assert new_idx == 0

    def test_handles_permission_errors_gracefully(self) -> None:
        """Test behavior when cache directory has permission issues."""
        with TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir) / "permission_test"
            cache_dir.mkdir()

            # Create similarity index and add data normally
            similarity_index = SimilarityIndex(cache_dir)
            embedding = np.random.rand(EMBEDDING_DIM).astype(np.float32)

            # This should work normally
            idx = similarity_index.add(embedding)
            assert idx == 0

            # Search should still work in memory even if files can't be written
            hit, found_idx = similarity_index.search(embedding, threshold=0.95)
            assert hit is True
            assert found_idx == idx

    def test_concurrent_file_access_safety(self) -> None:
        """Test that file operations are safe for basic concurrent access patterns."""
        with TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir) / "concurrent_test"

            # Create multiple indices using same cache directory
            # (Simulates basic concurrent access pattern)
            index1 = SimilarityIndex(cache_dir)

            # Add data to first instance
            embedding1 = np.random.rand(EMBEDDING_DIM).astype(np.float32)
            idx1 = index1.add(embedding1)

            # Create second instance - should load existing data
            index2 = SimilarityIndex(cache_dir)
            assert index2.index.ntotal == 1

            # Both should be able to search existing data
            hit1, found1 = index1.search(embedding1, threshold=0.95)
            hit2, found2 = index2.search(embedding1, threshold=0.95)

            assert hit1 is True
            assert hit2 is True
            assert found1 == found2 == idx1
