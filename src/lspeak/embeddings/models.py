"""Embedding models and constants for semantic similarity."""

from typing import TypeAlias

import numpy as np

# Model configuration constants
EMBEDDING_MODEL = "all-mpnet-base-v2"
EMBEDDING_DIM = 768

# Type aliases for clarity
Embedding: TypeAlias = np.ndarray  # Shape: (768,)
EmbeddingBatch: TypeAlias = np.ndarray  # Shape: (n, 768)
