"""Unit tests for embedding generation logic and validation."""

import sys
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from lspeak.embeddings.generator import EmbeddingGenerator
from lspeak.embeddings.models import EMBEDDING_DIM, EMBEDDING_MODEL


class TestEmbeddingConstants:
    """Test embedding constants are correct values."""

    def test_embedding_model_constant(self) -> None:
        """Test EMBEDDING_MODEL is set to expected value."""
        assert EMBEDDING_MODEL == "all-mpnet-base-v2"

    def test_embedding_dimension_constant(self) -> None:
        """Test EMBEDDING_DIM is set to expected value."""
        assert EMBEDDING_DIM == 768


class TestEmbeddingGeneratorInitialization:
    """Test EmbeddingGenerator initialization validation logic."""

    @patch("lspeak.embeddings.generator.SentenceTransformer")
    def test_valid_initialization_with_default_model(self, mock_transformer) -> None:
        """Test creating EmbeddingGenerator with default model."""
        # Mock the sentence transformer to avoid loading real model
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 768
        mock_transformer.return_value = mock_model

        generator = EmbeddingGenerator()

        assert generator.model_name == EMBEDDING_MODEL
        assert generator.dimension == 768
        mock_transformer.assert_called_once_with(EMBEDDING_MODEL)

    @patch("lspeak.embeddings.generator.SentenceTransformer")
    def test_valid_initialization_with_custom_model(self, mock_transformer) -> None:
        """Test creating EmbeddingGenerator with custom model name."""
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 768
        mock_transformer.return_value = mock_model

        custom_model = "custom-model-name"
        generator = EmbeddingGenerator(model_name=custom_model)

        assert generator.model_name == custom_model
        assert generator.dimension == 768
        mock_transformer.assert_called_once_with(custom_model)

    @patch("lspeak.embeddings.generator.SentenceTransformer")
    def test_initialization_fails_with_wrong_dimension(self, mock_transformer) -> None:
        """Test that initialization fails if model has wrong dimension."""
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = (
            512  # Wrong dimension
        )
        mock_transformer.return_value = mock_model

        with pytest.raises(ValueError, match="has dimension 512, expected 768"):
            EmbeddingGenerator()


class TestEmbeddingGeneratorGenerateMethod:
    """Test generate() method input validation and logic."""

    def setup_method(self) -> None:
        """Set up mock generator for testing."""
        with patch(
            "lspeak.embeddings.generator.SentenceTransformer"
        ) as mock_transformer:
            mock_model = Mock()
            mock_model.get_sentence_embedding_dimension.return_value = 768
            mock_transformer.return_value = mock_model
            self.generator = EmbeddingGenerator()

    def test_generate_empty_list_raises_error(self) -> None:
        """Test that empty text list raises ValueError."""
        with pytest.raises(
            ValueError, match="Cannot generate embeddings for empty text list"
        ):
            self.generator.generate([])

    def test_generate_single_text_shape_handling(self) -> None:
        """Test that single text gets proper shape (1, 768)."""
        # Mock the model.encode to return 1D array for single text
        single_embedding = np.random.rand(768)
        self.generator.model.encode.return_value = single_embedding

        result = self.generator.generate(["single text"])

        # Should reshape to (1, 768)
        assert result.shape == (1, 768)
        assert np.array_equal(result[0], single_embedding)

    def test_generate_multiple_texts_preserves_batch_shape(self) -> None:
        """Test that multiple texts preserve batch shape."""
        # Mock the model.encode to return 2D array for batch
        batch_embeddings = np.random.rand(3, 768)
        self.generator.model.encode.return_value = batch_embeddings

        result = self.generator.generate(["text1", "text2", "text3"])

        # Should keep original shape
        assert result.shape == (3, 768)
        assert np.array_equal(result, batch_embeddings)

    def test_generate_calls_model_with_correct_parameters(self) -> None:
        """Test that generate() calls model.encode with correct parameters."""
        mock_embeddings = np.random.rand(2, 768)
        self.generator.model.encode.return_value = mock_embeddings

        texts = ["test1", "test2"]
        self.generator.generate(texts)

        # Verify model.encode called with correct parameters
        self.generator.model.encode.assert_called_once_with(
            texts, convert_to_numpy=True, normalize_embeddings=True
        )


class TestEmbeddingGeneratorSimilarityMethod:
    """Test similarity() method validation and calculation logic."""

    def setup_method(self) -> None:
        """Set up mock generator for testing."""
        with patch(
            "lspeak.embeddings.generator.SentenceTransformer"
        ) as mock_transformer:
            mock_model = Mock()
            mock_model.get_sentence_embedding_dimension.return_value = 768
            mock_transformer.return_value = mock_model
            self.generator = EmbeddingGenerator()

    def test_similarity_with_valid_embeddings(self) -> None:
        """Test similarity calculation with valid 768-dimensional embeddings."""
        # Create normalized embeddings for predictable similarity
        emb1 = np.ones(768) / np.sqrt(768)  # Normalized
        emb2 = np.ones(768) / np.sqrt(768)  # Identical, normalized

        similarity = self.generator.similarity(emb1, emb2)

        # Identical normalized vectors should have similarity 1.0
        assert abs(similarity - 1.0) < 1e-6

    def test_similarity_with_orthogonal_embeddings(self) -> None:
        """Test similarity calculation with orthogonal embeddings."""
        # Create orthogonal normalized embeddings
        emb1 = np.zeros(768)
        emb1[0] = 1.0  # Only first dimension

        emb2 = np.zeros(768)
        emb2[1] = 1.0  # Only second dimension

        similarity = self.generator.similarity(emb1, emb2)

        # Orthogonal vectors should have similarity 0.0
        assert abs(similarity - 0.0) < 1e-6

    def test_similarity_first_embedding_wrong_shape_raises_error(self) -> None:
        """Test that first embedding with wrong shape raises ValueError."""
        wrong_shape_emb = np.random.rand(512)  # Wrong dimension
        correct_emb = np.random.rand(768)

        with pytest.raises(
            ValueError, match="embedding1 has shape \\(512,\\), expected \\(768,\\)"
        ):
            self.generator.similarity(wrong_shape_emb, correct_emb)

    def test_similarity_second_embedding_wrong_shape_raises_error(self) -> None:
        """Test that second embedding with wrong shape raises ValueError."""
        correct_emb = np.random.rand(768)
        wrong_shape_emb = np.random.rand(1024)  # Wrong dimension

        with pytest.raises(
            ValueError, match="embedding2 has shape \\(1024,\\), expected \\(768,\\)"
        ):
            self.generator.similarity(correct_emb, wrong_shape_emb)

    def test_similarity_range_clamping_above_one(self) -> None:
        """Test that similarity values above 1.0 are clamped to 1.0."""
        with patch("numpy.dot") as mock_dot:
            # Mock dot product to return value > 1.0 (shouldn't happen with normalized embeddings)
            mock_dot.return_value = 1.1

            emb1 = np.random.rand(768)
            emb2 = np.random.rand(768)

            similarity = self.generator.similarity(emb1, emb2)

            # Should be clamped to 1.0
            assert similarity == 1.0

    def test_similarity_range_clamping_below_zero(self) -> None:
        """Test that similarity values below 0.0 are clamped to 0.0."""
        with patch("numpy.dot") as mock_dot:
            # Mock dot product to return negative value
            mock_dot.return_value = -0.1

            emb1 = np.random.rand(768)
            emb2 = np.random.rand(768)

            similarity = self.generator.similarity(emb1, emb2)

            # Should be clamped to 0.0
            assert similarity == 0.0

    def test_similarity_returns_float_type(self) -> None:
        """Test that similarity always returns Python float, not numpy float."""
        emb1 = np.random.rand(768)
        emb2 = np.random.rand(768)

        similarity = self.generator.similarity(emb1, emb2)

        # Should be Python float type, not numpy float
        assert isinstance(similarity, float)
        assert not isinstance(similarity, np.floating)
