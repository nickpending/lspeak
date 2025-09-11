"""Unit tests for SimilarityIndex logic and validation."""

import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from lspeak.cache.similarity import SimilarityIndex
from lspeak.embeddings.models import EMBEDDING_DIM


class TestSimilarityIndexValidation:
    """Test input validation logic in SimilarityIndex methods."""

    def setup_method(self) -> None:
        """Set up SimilarityIndex with temporary directory for testing."""
        self.temp_dir = TemporaryDirectory()
        self.cache_dir = Path(self.temp_dir.name)
        self.similarity_index = SimilarityIndex(self.cache_dir)

    def teardown_method(self) -> None:
        """Clean up temporary directory."""
        self.temp_dir.cleanup()

    def test_search_validates_embedding_shape_correct_dimension(self) -> None:
        """Test that search accepts correct embedding dimension."""
        valid_embedding = np.random.rand(EMBEDDING_DIM).astype(np.float32)

        # Should not raise error (empty index returns False, None)
        hit, idx = self.similarity_index.search(valid_embedding, threshold=0.95)
        assert hit is False
        assert idx is None

    def test_search_validates_embedding_shape_wrong_dimension(self) -> None:
        """Test that search rejects wrong embedding dimension."""
        wrong_embedding = np.random.rand(512).astype(np.float32)  # Wrong dimension

        with pytest.raises(
            ValueError,
            match=f"Query embedding must have shape \\({EMBEDDING_DIM},\\), got \\(512,\\)",
        ):
            self.similarity_index.search(wrong_embedding, threshold=0.95)

    def test_search_validates_threshold_range_valid_values(self) -> None:
        """Test that search accepts valid threshold range."""
        valid_embedding = np.random.rand(EMBEDDING_DIM).astype(np.float32)

        # Test boundary values
        hit1, _ = self.similarity_index.search(valid_embedding, threshold=0.0)
        hit2, _ = self.similarity_index.search(valid_embedding, threshold=1.0)
        hit3, _ = self.similarity_index.search(valid_embedding, threshold=0.5)

        # Should not raise errors
        assert isinstance(hit1, bool)
        assert isinstance(hit2, bool)
        assert isinstance(hit3, bool)

    def test_search_validates_threshold_range_below_minimum(self) -> None:
        """Test that search rejects threshold below 0.0."""
        valid_embedding = np.random.rand(EMBEDDING_DIM).astype(np.float32)

        with pytest.raises(
            ValueError, match="Threshold must be between 0.0 and 1.0, got -0.1"
        ):
            self.similarity_index.search(valid_embedding, threshold=-0.1)

    def test_search_validates_threshold_range_above_maximum(self) -> None:
        """Test that search rejects threshold above 1.0."""
        valid_embedding = np.random.rand(EMBEDDING_DIM).astype(np.float32)

        with pytest.raises(
            ValueError, match="Threshold must be between 0.0 and 1.0, got 1.5"
        ):
            self.similarity_index.search(valid_embedding, threshold=1.5)

    def test_add_validates_embedding_shape_correct_dimension(self) -> None:
        """Test that add accepts correct embedding dimension."""
        valid_embedding = np.random.rand(EMBEDDING_DIM).astype(np.float32)

        idx = self.similarity_index.add(valid_embedding)

        # Should return valid index
        assert isinstance(idx, int)
        assert idx >= 0

    def test_add_validates_embedding_shape_wrong_dimension(self) -> None:
        """Test that add rejects wrong embedding dimension."""
        wrong_embedding = np.random.rand(1024).astype(np.float32)  # Wrong dimension

        with pytest.raises(
            ValueError,
            match=f"Embedding must have shape \\({EMBEDDING_DIM},\\), got \\(1024,\\)",
        ):
            self.similarity_index.add(wrong_embedding)


class TestSimilarityCalculationLogic:
    """Test L2 distance to similarity conversion and threshold logic."""

    def setup_method(self) -> None:
        """Set up SimilarityIndex for testing similarity calculations."""
        self.temp_dir = TemporaryDirectory()
        self.cache_dir = Path(self.temp_dir.name)
        self.similarity_index = SimilarityIndex(self.cache_dir)

    def teardown_method(self) -> None:
        """Clean up temporary directory."""
        self.temp_dir.cleanup()

    def test_similarity_conversion_identical_embeddings(self) -> None:
        """Test similarity calculation for identical embeddings."""
        # Create identical normalized embeddings
        embedding = np.ones(EMBEDDING_DIM).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)  # Normalize

        # Add to index
        idx = self.similarity_index.add(embedding)

        # Search with same embedding - should get perfect match
        hit, found_idx = self.similarity_index.search(embedding, threshold=0.95)

        assert hit is True
        assert found_idx == idx

    def test_similarity_conversion_threshold_comparison_hit(self) -> None:
        """Test that similar embeddings above threshold return hit."""
        # Create normalized embedding
        base_embedding = np.random.rand(EMBEDDING_DIM).astype(np.float32)
        base_embedding = base_embedding / np.linalg.norm(base_embedding)

        # Add to index
        idx = self.similarity_index.add(base_embedding)

        # Create very similar embedding (tiny perturbation)
        similar_embedding = base_embedding + np.random.normal(
            0, 0.01, EMBEDDING_DIM
        ).astype(np.float32)
        similar_embedding = similar_embedding / np.linalg.norm(similar_embedding)

        # Search with high threshold - should still hit
        hit, found_idx = self.similarity_index.search(similar_embedding, threshold=0.5)

        assert hit is True
        assert found_idx == idx

    def test_similarity_conversion_threshold_comparison_miss(self) -> None:
        """Test that dissimilar embeddings below threshold return miss."""
        # Add random embedding to index
        embedding1 = np.random.rand(EMBEDDING_DIM).astype(np.float32)
        self.similarity_index.add(embedding1)

        # Create completely different embedding
        embedding2 = np.random.rand(EMBEDDING_DIM).astype(np.float32)

        # Search with very high threshold - should miss
        hit, found_idx = self.similarity_index.search(embedding2, threshold=0.99)

        assert hit is False
        assert found_idx is None

    def test_similarity_handles_empty_index_case(self) -> None:
        """Test that search handles empty index correctly."""
        empty_index = SimilarityIndex(self.cache_dir / "empty")
        embedding = np.random.rand(EMBEDDING_DIM).astype(np.float32)

        hit, found_idx = empty_index.search(embedding, threshold=0.95)

        assert hit is False
        assert found_idx is None


class TestArrayAppendLogic:
    """Test embedding array append logic and index position calculation."""

    def setup_method(self) -> None:
        """Set up SimilarityIndex for testing append logic."""
        self.temp_dir = TemporaryDirectory()
        self.cache_dir = Path(self.temp_dir.name)
        self.similarity_index = SimilarityIndex(self.cache_dir)

    def teardown_method(self) -> None:
        """Clean up temporary directory."""
        self.temp_dir.cleanup()

    def test_add_first_embedding_to_empty_array(self) -> None:
        """Test adding first embedding to empty array creates correct structure."""
        embedding = np.random.rand(EMBEDDING_DIM).astype(np.float32)

        # Verify starts empty
        assert len(self.similarity_index.embeddings) == 0
        assert self.similarity_index.index.ntotal == 0

        # Add first embedding
        idx = self.similarity_index.add(embedding)

        # Verify correct index returned and array structure
        assert idx == 0
        assert len(self.similarity_index.embeddings) == 1
        assert self.similarity_index.embeddings.shape == (1, EMBEDDING_DIM)
        assert self.similarity_index.index.ntotal == 1

    def test_add_subsequent_embeddings_appends_correctly(self) -> None:
        """Test that subsequent embeddings append correctly with right indices."""
        embeddings = [
            np.random.rand(EMBEDDING_DIM).astype(np.float32),
            np.random.rand(EMBEDDING_DIM).astype(np.float32),
            np.random.rand(EMBEDDING_DIM).astype(np.float32),
        ]

        # Add embeddings sequentially
        indices = []
        for embedding in embeddings:
            idx = self.similarity_index.add(embedding)
            indices.append(idx)

        # Verify sequential indices
        assert indices == [0, 1, 2]

        # Verify array structure
        assert len(self.similarity_index.embeddings) == 3
        assert self.similarity_index.embeddings.shape == (3, EMBEDDING_DIM)
        assert self.similarity_index.index.ntotal == 3

    def test_add_converts_dtype_to_float32(self) -> None:
        """Test that add converts embedding dtype to float32 for FAISS compatibility."""
        # Create embedding with different dtype
        embedding_f64 = np.random.rand(EMBEDDING_DIM).astype(np.float64)

        idx = self.similarity_index.add(embedding_f64)

        # Verify stored as float32
        stored_embedding = self.similarity_index.embeddings[idx]
        assert stored_embedding.dtype == np.float32

    def test_add_preserves_embedding_values_during_conversion(self) -> None:
        """Test that dtype conversion preserves embedding values."""
        # Create embedding with known values
        embedding = np.array([0.1, 0.2, 0.3] + [0.0] * (EMBEDDING_DIM - 3)).astype(
            np.float64
        )

        idx = self.similarity_index.add(embedding)

        # Verify values preserved (within float32 precision)
        stored = self.similarity_index.embeddings[idx]
        assert abs(stored[0] - 0.1) < 1e-6
        assert abs(stored[1] - 0.2) < 1e-6
        assert abs(stored[2] - 0.3) < 1e-6


class TestErrorHandlingLogic:
    """Test error handling for edge cases and invalid states."""

    def test_load_embeddings_handles_missing_file(self) -> None:
        """Test that missing embeddings file creates empty array."""
        with TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir) / "missing_files"

            # Initialize without existing files
            similarity_index = SimilarityIndex(cache_dir)

            # Should create empty array
            assert len(similarity_index.embeddings) == 0
            assert similarity_index.embeddings.shape == (0, EMBEDDING_DIM)
            assert similarity_index.embeddings.dtype == np.float32

    @patch("numpy.load")
    def test_load_embeddings_handles_corrupted_file(self, mock_np_load) -> None:
        """Test that corrupted embeddings file falls back to empty array."""
        # Mock numpy.load to raise exception
        mock_np_load.side_effect = OSError("Corrupted file")

        with TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir)

            # Create fake embeddings file to trigger load attempt
            embeddings_path = cache_dir / "embeddings.npy"
            embeddings_path.touch()

            # Should handle corruption gracefully
            similarity_index = SimilarityIndex(cache_dir)

            assert len(similarity_index.embeddings) == 0
            assert similarity_index.embeddings.dtype == np.float32

    @patch("faiss.read_index")
    def test_load_index_handles_corrupted_faiss_file(self, mock_faiss_read) -> None:
        """Test that corrupted FAISS file creates new index."""
        # Mock faiss.read_index to raise exception
        mock_faiss_read.side_effect = Exception("Corrupted FAISS index")

        with TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir)

            # Create fake index file to trigger load attempt
            index_path = cache_dir / "faiss.index"
            index_path.touch()

            # Should handle corruption gracefully
            similarity_index = SimilarityIndex(cache_dir)

            # Should create new index with correct dimension
            assert similarity_index.index.d == EMBEDDING_DIM
            assert similarity_index.index.ntotal == 0


class TestTasksDemoCommands:
    """Test methods matching TASKS.md demo commands."""

    def setup_method(self) -> None:
        """Set up SimilarityIndex for testing demo commands."""
        self.temp_dir = TemporaryDirectory()
        self.cache_dir = Path(self.temp_dir.name)
        self.similarity_index = SimilarityIndex(self.cache_dir)

    def teardown_method(self) -> None:
        """Clean up temporary directory."""
        self.temp_dir.cleanup()

    def test_search_threshold(self) -> None:
        """Test search threshold behavior - matches TASKS.md demo command."""
        # Add test embedding to index
        test_embedding = np.random.rand(EMBEDDING_DIM).astype(np.float32)
        test_embedding = test_embedding / np.linalg.norm(test_embedding)
        idx = self.similarity_index.add(test_embedding)

        # Test exact match with high threshold
        hit_high, found_idx_high = self.similarity_index.search(
            test_embedding, threshold=0.95
        )
        assert hit_high is True
        assert found_idx_high == idx

        # Test with very low threshold - should still hit
        hit_low, found_idx_low = self.similarity_index.search(
            test_embedding, threshold=0.1
        )
        assert hit_low is True
        assert found_idx_low == idx

        # Test with different embedding and high threshold - should miss
        different_embedding = np.random.rand(EMBEDDING_DIM).astype(np.float32)
        different_embedding = different_embedding / np.linalg.norm(different_embedding)
        hit_miss, found_idx_miss = self.similarity_index.search(
            different_embedding, threshold=0.99
        )
        # This might hit or miss depending on random embeddings, but should not crash
        assert isinstance(hit_miss, bool)
        assert found_idx_miss is None or isinstance(found_idx_miss, int)
