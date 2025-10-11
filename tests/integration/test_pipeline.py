"""Integration tests for TTSPipeline with real services."""

import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from lspeak.audio.player import AudioPlayer
from lspeak.cache.manager import SemanticCacheManager
from lspeak.tts.pipeline import TTSPipeline


class TestCacheKeyConsistency:
    """Test that cache keys are consistent for same inputs."""

    @pytest.mark.asyncio
    async def test_cache_key_consistency_with_preloaded(self) -> None:
        """
        INVARIANT: Same (text, provider, voice) must return same cached audio
        BREAKS: Wrong audio played to user = broken experience
        SOURCE: Tester-identified
        """
        with TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir) / "cache"
            output_dir = Path(temp_dir) / "output"
            output_dir.mkdir()

            # Create real pre-loaded instances
            cache_manager = SemanticCacheManager(cache_dir, similarity_threshold=0.95)
            audio_player = AudioPlayer()

            pipeline = TTSPipeline(
                cache_manager=cache_manager, audio_player=audio_player
            )

            # First call - cache miss, generates audio
            result1 = await pipeline.process(
                text="Cache consistency test",
                provider="system",
                voice=None,
                output=str(output_dir / "test1.mp3"),
                cache=True,
                debug=False,
            )

            assert result1["cached"] is False, "First call should be cache miss"
            assert result1["saved"] == str(output_dir / "test1.mp3")

            # Second call - SAME inputs, should hit cache
            result2 = await pipeline.process(
                text="Cache consistency test",
                provider="system",
                voice=None,
                output=str(output_dir / "test2.mp3"),
                cache=True,
                debug=False,
            )

            assert result2["cached"] is True, (
                "Second call with identical inputs should hit cache"
            )

            # CRITICAL: Audio files should be identical
            audio1 = Path(result1["saved"]).read_bytes()
            audio2 = Path(result2["saved"]).read_bytes()
            assert audio1 == audio2, "Cache hit must return identical audio data"

    @pytest.mark.asyncio
    async def test_cache_key_consistency_voice_variations(self) -> None:
        """
        INVARIANT: voice=None vs voice="default" should be treated consistently
        BREAKS: Same audio cached under different keys = wasted API calls
        SOURCE: Tester-identified
        """
        with TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir) / "cache"
            output_dir = Path(temp_dir) / "output"
            output_dir.mkdir()

            cache_manager = SemanticCacheManager(cache_dir)
            audio_player = AudioPlayer()
            pipeline = TTSPipeline(
                cache_manager=cache_manager, audio_player=audio_player
            )

            # Call with voice=None (first call to populate cache)
            await pipeline.process(
                text="Voice variation test",
                provider="system",
                voice=None,
                output=str(output_dir / "none.mp3"),
                cache=True,
                debug=False,
            )

            # Call with voice="" (empty string, what synthesize gets)
            result_empty = await pipeline.process(
                text="Voice variation test",
                provider="system",
                voice="",
                output=str(output_dir / "empty.mp3"),
                cache=True,
                debug=False,
            )

            # Both should map to same cache entry
            # (pipeline normalizes None -> "default" for caching)
            assert result_empty["cached"] is True, (
                "Empty voice should hit cache from None voice"
            )


class TestComponentReuse:
    """Test that pre-loaded instances are preserved for performance."""

    @pytest.mark.asyncio
    async def test_component_reuse_preserves_instances(self) -> None:
        """
        INVARIANT: Pre-loaded instances must be used when provided
        BREAKS: Daemon performance degrades, sub-second promise broken
        SOURCE: Developer-identified
        """
        with TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir) / "cache"
            output_file = Path(temp_dir) / "output.mp3"

            # Create real pre-loaded instances
            cache_manager = SemanticCacheManager(cache_dir)
            audio_player = AudioPlayer()

            # Initialize pipeline with pre-loaded instances
            pipeline = TTSPipeline(
                cache_manager=cache_manager, audio_player=audio_player
            )

            # Process audio
            await pipeline.process(
                text="Component reuse test",
                provider="system",
                voice=None,
                output=str(output_file),
                cache=True,
                debug=False,
            )

            # CRITICAL: Verify pre-loaded instances are still the same objects
            assert pipeline.cache_manager is cache_manager, (
                "Pipeline must preserve pre-loaded cache_manager instance"
            )
            assert pipeline.audio_player is audio_player, (
                "Pipeline must preserve pre-loaded audio_player instance"
            )


class TestWorkflowResilience:
    """Test that cache failures don't prevent synthesis."""

    @pytest.mark.asyncio
    async def test_cache_lookup_failure_continues_synthesis(self) -> None:
        """
        FAILURE: Cache lookup exceptions (index corruption, disk issues)
        GRACEFUL: Synthesis continues, audio delivered
        SOURCE: Developer-identified
        """
        with TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir) / "cache"
            output_file = Path(temp_dir) / "output.mp3"

            # Create cache manager, then corrupt the FAISS index
            cache_manager = SemanticCacheManager(cache_dir)

            # Corrupt the FAISS index file to cause lookup failures
            index_file = cache_dir / "faiss.index"
            if index_file.exists():
                index_file.write_bytes(b"CORRUPTED_DATA")

            audio_player = AudioPlayer()
            pipeline = TTSPipeline(
                cache_manager=cache_manager, audio_player=audio_player
            )

            # Process should continue despite cache corruption
            result = await pipeline.process(
                text="Resilience test",
                provider="system",
                voice=None,
                output=str(output_file),
                cache=True,
                debug=False,
            )

            # Audio should still be generated and saved
            assert result["saved"] == str(output_file)
            assert output_file.exists(), "Audio must be saved despite cache failure"
            assert len(output_file.read_bytes()) > 0, "Audio file must not be empty"

    @pytest.mark.asyncio
    async def test_cache_write_failure_still_outputs_audio(self) -> None:
        """
        FAILURE: Cache write exceptions (disk full, permissions)
        GRACEFUL: Audio still played/saved, next request may succeed
        SOURCE: Developer-identified
        """
        with TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir) / "cache"
            output_file = Path(temp_dir) / "output.mp3"

            # Create cache manager normally
            cache_manager = SemanticCacheManager(cache_dir)
            audio_player = AudioPlayer()
            pipeline = TTSPipeline(
                cache_manager=cache_manager, audio_player=audio_player
            )

            # Make cache audio directory read-only to trigger write failures
            # Must chmod subdirectory BEFORE parent, otherwise we can't access it
            (cache_dir / "audio").chmod(0o444)
            cache_dir.chmod(0o555)  # Read+execute so we can list, but not write

            try:
                # Process should succeed despite inability to cache
                result = await pipeline.process(
                    text="Write failure test",
                    provider="system",
                    voice=None,
                    output=str(output_file),
                    cache=True,
                    debug=False,
                )

                # Audio should still be generated and saved
                assert result["saved"] == str(output_file)
                assert output_file.exists(), (
                    "Audio must be saved despite cache write failure"
                )
                assert len(output_file.read_bytes()) > 0

            finally:
                # Restore permissions for cleanup (parent first, then children)
                cache_dir.chmod(0o755)
                if (cache_dir / "audio").exists():
                    (cache_dir / "audio").chmod(0o755)

    @pytest.mark.asyncio
    async def test_provider_error_propagates_clearly(self) -> None:
        """
        FAILURE: Provider synthesis errors (invalid voice, API failure)
        GRACEFUL: Clear error message bubbles to caller
        SOURCE: Developer-identified
        """
        with TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir) / "cache"

            cache_manager = SemanticCacheManager(cache_dir)
            audio_player = AudioPlayer()
            pipeline = TTSPipeline(
                cache_manager=cache_manager, audio_player=audio_player
            )

            # Try to use non-existent provider
            with pytest.raises(KeyError, match="Provider 'nonexistent' not found"):
                await pipeline.process(
                    text="Error test",
                    provider="nonexistent",
                    voice=None,
                    output=None,
                    cache=True,
                    debug=False,
                )


class TestOnDemandInstanceCreation:
    """Test that pipeline creates instances on-demand when not pre-loaded."""

    @pytest.mark.asyncio
    async def test_on_demand_instance_creation(self) -> None:
        """
        CONFIDENCE: Pipeline creates instances on-demand for direct path
        ENSURES: Direct execution path works without pre-loaded instances
        """
        with TemporaryDirectory() as temp_dir:
            output_file = Path(temp_dir) / "output.mp3"

            # Initialize pipeline WITHOUT pre-loaded instances
            pipeline = TTSPipeline()

            # Process should work, creating instances on-demand
            result = await pipeline.process(
                text="On-demand test",
                provider="system",
                voice=None,
                output=str(output_file),
                cache=True,
                cache_threshold=0.95,
                debug=False,
            )

            # Verify instances were created
            assert pipeline.cache_manager is not None, (
                "Cache manager should be created on-demand"
            )
            assert pipeline.audio_player is not None, (
                "Audio player should be created on-demand"
            )

            # Verify audio was generated
            assert result["saved"] == str(output_file)
            assert output_file.exists()
            assert len(output_file.read_bytes()) > 0
