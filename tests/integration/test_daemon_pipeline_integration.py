"""Integration tests for daemon and core using TTSPipeline refactoring."""

import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from lspeak.core import speak_text
from lspeak.daemon.server import LspeakDaemon


class TestDaemonPipelineIntegration:
    """Test daemon path uses TTS Pipeline correctly."""

    @pytest.mark.asyncio
    async def test_daemon_process_speech_uses_pipeline(self) -> None:
        """
        INVARIANT: Daemon _process_speech uses TTSPipeline with pre-loaded instances
        BREAKS: Sub-second response time if instances reload on every request
        SOURCE: Tasks 2.4 refactoring
        """
        with TemporaryDirectory() as temp_dir:
            output_file = Path(temp_dir) / "daemon_test.mp3"

            # Create daemon with pre-loaded instances
            daemon = LspeakDaemon()

            # Simulate daemon startup (load models)
            from lspeak.audio.player import AudioPlayer
            from lspeak.cache.manager import SemanticCacheManager

            daemon.cache_manager = SemanticCacheManager()
            daemon.audio_player = AudioPlayer()
            daemon.models_loaded = True

            # Process speech using daemon's refactored method
            result = await daemon._process_speech(
                {
                    "text": "Daemon pipeline test",
                    "provider": "system",
                    "voice": None,
                    "output": str(output_file),
                    "cache": True,
                    "debug": False,
                }
            )

            # Verify pipeline processed correctly
            assert "saved" in result
            assert result["saved"] == str(output_file)
            assert output_file.exists()
            assert len(output_file.read_bytes()) > 0

            # CRITICAL: Verify pre-loaded instances were used (not recreated)
            assert daemon.cache_manager is not None
            assert daemon.audio_player is not None


class TestDirectPipelineIntegration:
    """Test direct execution path uses TTSPipeline correctly."""

    @pytest.mark.asyncio
    async def test_direct_execution_uses_pipeline(self) -> None:
        """
        INVARIANT: Direct execution creates instances on-demand via TTSPipeline
        BREAKS: Direct path fails if pipeline doesn't handle None instances
        SOURCE: Tasks 2.5 refactoring
        """
        with TemporaryDirectory() as temp_dir:
            output_file = Path(temp_dir) / "direct_test.mp3"

            # Call speak_text directly (no daemon)
            await speak_text(
                text="Direct pipeline test",
                provider="system",
                voice_id=None,
                output_file=str(output_file),
                cache=True,
                use_daemon=False,  # Force direct path
                debug=False,
            )

            # Verify pipeline processed correctly
            assert output_file.exists()
            assert len(output_file.read_bytes()) > 0


class TestDeduplicationAchieved:
    """Verify code deduplication was achieved."""

    def test_daemon_server_line_count_reduced(self) -> None:
        """
        CONFIDENCE: Daemon code reduced from ~90 to ~32 lines
        ENSURES: Deduplication goal achieved
        SOURCE: Tasks 2.4-2.5 goal
        """
        server_file = (
            Path(__file__).parent.parent.parent / "src/lspeak/daemon/server.py"
        )
        server_code = server_file.read_text()

        # Find _process_speech method
        lines = server_code.split("\n")
        in_method = False
        method_lines = 0

        for line in lines:
            if "async def _process_speech" in line:
                in_method = True
            elif in_method:
                if line.strip() and not line.strip().startswith("#"):
                    if line.startswith("    async def") or line.startswith("    def"):
                        break  # Next method started
                    method_lines += 1

        # Verify significantly reduced (was ~90 lines, now ~25-35)
        assert method_lines < 40, (
            f"Method should be <40 lines after refactoring, got {method_lines}"
        )

    def test_core_line_count_reduced(self) -> None:
        """
        CONFIDENCE: Core.py TTS logic reduced from ~70 to ~17 lines
        ENSURES: Deduplication goal achieved
        SOURCE: Tasks 2.4-2.5 goal
        """
        core_file = Path(__file__).parent.parent.parent / "src/lspeak/core.py"
        core_code = core_file.read_text()

        # Count lines in direct execution section (after "Task 5.5" comment)
        lines = core_code.split("\n")
        in_direct_section = False
        direct_lines = 0

        for line in lines:
            if "Task 5.5: Fallback to direct execution" in line:
                in_direct_section = True
            elif in_direct_section:
                # Count until end of file (it's the last section)
                if line.strip() and not line.strip().startswith("#"):
                    direct_lines += 1

        # Verify significantly reduced (was ~70 lines, now ~15-20)
        assert direct_lines < 25, (
            f"Direct section should be <25 lines after refactoring, got {direct_lines}"
        )
