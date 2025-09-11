"""Integration tests for embedding generation with real sentence-transformers."""

import sys
from pathlib import Path

import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from lspeak.embeddings.generator import EmbeddingGenerator
from lspeak.embeddings.models import EMBEDDING_DIM, EMBEDDING_MODEL


class TestEmbeddingGeneratorIntegration:
    """Test complete embedding workflow with real sentence-transformers."""

    def test_complete_embedding_workflow(self) -> None:
        """
        Test complete embedding generation workflow:
        - Initialize generator with real model
        - Generate embeddings for text
        - Calculate similarity between embeddings
        - Verify semantic similarity works correctly
        """
        # Initialize with real sentence-transformers model
        generator = EmbeddingGenerator()

        # Verify model loaded correctly
        assert generator.model_name == EMBEDDING_MODEL
        assert generator.dimension == EMBEDDING_DIM

        # Generate embeddings for test texts
        test_texts = ["Deploy complete", "Hello world", "System ready"]
        embeddings = generator.generate(test_texts)

        # Verify embedding properties
        assert embeddings.shape == (3, 768)
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.dtype == np.float32 or embeddings.dtype == np.float64

        # Test individual embedding extraction
        deploy_emb = embeddings[0]
        hello_emb = embeddings[1]
        system_emb = embeddings[2]

        # All embeddings should be normalized (for cosine similarity)
        assert abs(np.linalg.norm(deploy_emb) - 1.0) < 1e-5
        assert abs(np.linalg.norm(hello_emb) - 1.0) < 1e-5
        assert abs(np.linalg.norm(system_emb) - 1.0) < 1e-5

        # Calculate similarity between embeddings
        deploy_hello_sim = generator.similarity(deploy_emb, hello_emb)
        deploy_system_sim = generator.similarity(deploy_emb, system_emb)
        hello_system_sim = generator.similarity(hello_emb, system_emb)

        # All similarities should be in valid range [0, 1]
        assert 0.0 <= deploy_hello_sim <= 1.0
        assert 0.0 <= deploy_system_sim <= 1.0
        assert 0.0 <= hello_system_sim <= 1.0

        # Similarities should be Python floats
        assert isinstance(deploy_hello_sim, float)
        assert isinstance(deploy_system_sim, float)
        assert isinstance(hello_system_sim, float)

    def test_semantic_similarity_with_real_text_pairs(self) -> None:
        """Test semantic similarity with real text pairs to verify meaningful scores."""
        generator = EmbeddingGenerator()

        # Test very similar phrases (should have high similarity)
        similar_texts = [
            "Deploy complete",
            "Deployment complete",
            "The deployment is complete",
        ]

        similar_embeddings = generator.generate(similar_texts)

        # Test similarity between very similar phrases
        sim_1_2 = generator.similarity(similar_embeddings[0], similar_embeddings[1])
        sim_1_3 = generator.similarity(similar_embeddings[0], similar_embeddings[2])

        # Very similar phrases should have high similarity (>0.8)
        assert sim_1_2 > 0.8, f"Similar phrases have low similarity: {sim_1_2}"
        assert sim_1_3 > 0.7, f"Similar phrases have low similarity: {sim_1_3}"

        # Test different phrases (should have lower similarity)
        different_texts = ["Deploy complete", "Hello world", "The weather is nice"]

        different_embeddings = generator.generate(different_texts)

        deploy_hello_sim = generator.similarity(
            different_embeddings[0], different_embeddings[1]
        )
        deploy_weather_sim = generator.similarity(
            different_embeddings[0], different_embeddings[2]
        )

        # Very different phrases should have low similarity (<0.5)
        assert deploy_hello_sim < 0.5, (
            f"Different phrases have high similarity: {deploy_hello_sim}"
        )
        assert deploy_weather_sim < 0.5, (
            f"Different phrases have high similarity: {deploy_weather_sim}"
        )

    def test_identical_text_similarity(self) -> None:
        """Test that identical texts have perfect similarity."""
        generator = EmbeddingGenerator()

        # Generate embeddings for identical text
        text = "This is a test sentence"
        embeddings = generator.generate([text, text])

        # Calculate similarity between identical embeddings
        similarity = generator.similarity(embeddings[0], embeddings[1])

        # Identical texts should have similarity 1.0 (or very close due to floating point)
        assert abs(similarity - 1.0) < 1e-5, f"Identical texts similarity: {similarity}"

    def test_batch_vs_single_embedding_consistency(self) -> None:
        """Test that batch and single embedding generation produce same results."""
        generator = EmbeddingGenerator()

        test_text = "Consistency test message"

        # Generate single embedding
        single_embedding = generator.generate([test_text])[0]

        # Generate as part of batch
        batch_embeddings = generator.generate([test_text, "Other text"])
        batch_embedding = batch_embeddings[0]

        # Calculate similarity between single and batch versions
        consistency_sim = generator.similarity(single_embedding, batch_embedding)

        # Should be identical (or very close)
        assert abs(consistency_sim - 1.0) < 1e-5, (
            f"Batch vs single inconsistency: {consistency_sim}"
        )

    def test_embedding_stability_across_calls(self) -> None:
        """Test that same text generates consistent embeddings across multiple calls."""
        generator = EmbeddingGenerator()

        text = "Stability test phrase"

        # Generate embedding multiple times
        emb1 = generator.generate([text])[0]
        emb2 = generator.generate([text])[0]
        emb3 = generator.generate([text])[0]

        # All should be identical
        sim_1_2 = generator.similarity(emb1, emb2)
        sim_1_3 = generator.similarity(emb1, emb3)
        sim_2_3 = generator.similarity(emb2, emb3)

        # Should all be perfect similarity
        assert abs(sim_1_2 - 1.0) < 1e-5, f"Embedding not stable: {sim_1_2}"
        assert abs(sim_1_3 - 1.0) < 1e-5, f"Embedding not stable: {sim_1_3}"
        assert abs(sim_2_3 - 1.0) < 1e-5, f"Embedding not stable: {sim_2_3}"

    def test_real_model_loading_and_properties(self) -> None:
        """Test that real sentence-transformers model loads correctly."""
        generator = EmbeddingGenerator()

        # Verify model properties
        assert hasattr(generator.model, "encode")
        assert hasattr(generator.model, "get_sentence_embedding_dimension")

        # Verify dimension is correct
        assert generator.dimension == 768
        assert generator.model.get_sentence_embedding_dimension() == 768

        # Test model can actually generate embeddings
        test_embedding = generator.generate(["Model test"])[0]

        # Verify embedding properties
        assert test_embedding.shape == (768,)
        assert isinstance(test_embedding, np.ndarray)

        # Verify embedding is normalized
        norm = np.linalg.norm(test_embedding)
        assert abs(norm - 1.0) < 1e-5, f"Embedding not normalized: {norm}"

    def test_large_batch_processing(self) -> None:
        """Test processing larger batches of text efficiently."""
        generator = EmbeddingGenerator()

        # Create batch of varied texts
        batch_texts = [f"Test message number {i}" for i in range(20)]

        # Generate embeddings for entire batch
        batch_embeddings = generator.generate(batch_texts)

        # Verify batch properties
        assert batch_embeddings.shape == (20, 768)
        assert isinstance(batch_embeddings, np.ndarray)

        # Verify all embeddings are normalized
        for i, embedding in enumerate(batch_embeddings):
            norm = np.linalg.norm(embedding)
            assert abs(norm - 1.0) < 1e-5, f"Embedding {i} not normalized: {norm}"

        # Test similarity calculations work across batch
        sim_0_1 = generator.similarity(batch_embeddings[0], batch_embeddings[1])
        sim_0_19 = generator.similarity(batch_embeddings[0], batch_embeddings[19])

        # Similarities should be in valid range
        assert 0.0 <= sim_0_1 <= 1.0
        assert 0.0 <= sim_0_19 <= 1.0

    def test_special_characters_and_unicode(self) -> None:
        """Test embedding generation with special characters and unicode."""
        generator = EmbeddingGenerator()

        # Test texts with special characters
        special_texts = [
            "Hello, world! 🌍",
            "Café résumé naïve",
            "测试中文文本",
            "Тестовый русский текст",
            "Special chars: @#$%^&*()",
        ]

        # Should not raise errors
        embeddings = generator.generate(special_texts)

        # Verify all embeddings generated
        assert embeddings.shape == (5, 768)

        # All should be properly normalized
        for i, embedding in enumerate(embeddings):
            norm = np.linalg.norm(embedding)
            assert abs(norm - 1.0) < 1e-5, (
                f"Special text {i} embedding not normalized: {norm}"
            )

        # Test similarity calculations work
        sim_0_1 = generator.similarity(embeddings[0], embeddings[1])
        assert 0.0 <= sim_0_1 <= 1.0
