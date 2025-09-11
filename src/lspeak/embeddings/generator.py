"""Text embedding generation using sentence-transformers."""

from typing import TYPE_CHECKING

import numpy as np

from .models import EMBEDDING_DIM, EMBEDDING_MODEL, Embedding, EmbeddingBatch

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


class EmbeddingGenerator:
    """Generate text embeddings for semantic similarity using sentence-transformers.

    Uses the all-mpnet-base-v2 model for high-quality embeddings with 768 dimensions.
    This model provides good balance of quality and speed for semantic similarity tasks.
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        """Initialize embedding generator with specified model.

        Args:
            model_name: Name of sentence-transformers model to use

        Raises:
            Exception: If model fails to load
        """
        self.model_name = model_name
        self._model: SentenceTransformer | None = None  # Lazy load the model
        self.dimension = EMBEDDING_DIM  # We know this for all-mpnet-base-v2

    @property
    def model(self) -> "SentenceTransformer":
        """Lazy-load the model only when actually needed."""
        if self._model is None:
            # Import here to avoid loading at module import time
            from sentence_transformers import SentenceTransformer

            # Note: Offline mode should already be configured by CLI
            # if models are cached (via ensure_models_available)
            self._model = SentenceTransformer(self.model_name)

            # Verify dimension after loading
            actual_dim = self._model.get_sentence_embedding_dimension()
            if actual_dim != EMBEDDING_DIM:
                raise ValueError(
                    f"Model {self.model_name} has dimension {actual_dim}, "
                    f"expected {EMBEDDING_DIM}"
                )
        return self._model

    def generate(self, texts: list[str]) -> EmbeddingBatch:
        """Generate embeddings for one or more texts.

        Args:
            texts: List of text strings to embed

        Returns:
            Numpy array of embeddings with shape (len(texts), 768)

        Raises:
            ValueError: If texts is empty
            Exception: If embedding generation fails
        """
        if not texts:
            raise ValueError("Cannot generate embeddings for empty text list")

        # Generate embeddings using sentence-transformers
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,  # Normalize for cosine similarity
        )

        # Ensure correct shape
        if len(texts) == 1:
            embeddings = embeddings.reshape(1, -1)

        return embeddings

    def similarity(self, embedding1: Embedding, embedding2: Embedding) -> float:
        """Calculate cosine similarity between two embeddings.

        Args:
            embedding1: First embedding vector (768,)
            embedding2: Second embedding vector (768,)

        Returns:
            Cosine similarity score between 0.0 and 1.0
            (1.0 = identical, 0.0 = completely different)

        Raises:
            ValueError: If embeddings have wrong shape or size
        """
        # Validate input shapes
        if embedding1.shape != (EMBEDDING_DIM,):
            raise ValueError(
                f"embedding1 has shape {embedding1.shape}, expected ({EMBEDDING_DIM},)"
            )
        if embedding2.shape != (EMBEDDING_DIM,):
            raise ValueError(
                f"embedding2 has shape {embedding2.shape}, expected ({EMBEDDING_DIM},)"
            )

        # Calculate cosine similarity
        # Since embeddings are normalized, dot product equals cosine similarity
        similarity = float(np.dot(embedding1, embedding2))

        # Clamp to valid range [0, 1] to handle floating point precision
        return max(0.0, min(1.0, similarity))
