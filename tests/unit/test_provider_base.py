"""Unit tests for TTSProvider abstract base class."""

import sys
from pathlib import Path

import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from lspeak.providers.base import TTSProvider


class TestTTSProviderAbstractClass:
    """Test TTSProvider abstract base class behavior."""

    def test_cannot_instantiate_abstract_class(self) -> None:
        """Test that TTSProvider cannot be instantiated directly."""
        with pytest.raises(TypeError) as exc_info:
            TTSProvider()

        # Verify error message mentions the abstract methods
        error_msg = str(exc_info.value)
        assert "abstract" in error_msg.lower()
        assert "synthesize" in error_msg
        assert "list_voices" in error_msg

    def test_abstract_methods_defined(self) -> None:
        """Test that expected abstract methods are defined."""
        abstract_methods = TTSProvider.__abstractmethods__

        # Should have exactly these two methods
        assert len(abstract_methods) == 2
        assert "synthesize" in abstract_methods
        assert "list_voices" in abstract_methods

    def test_partial_implementation_fails(self) -> None:
        """Test that implementing only one abstract method still fails."""

        # Create a class that only implements synthesize
        class PartialProvider(TTSProvider):
            async def synthesize(self, text: str, voice: str) -> bytes:
                return b"audio"

        # Should still fail to instantiate
        with pytest.raises(TypeError) as exc_info:
            PartialProvider()

        # Error should mention the missing method
        error_msg = str(exc_info.value)
        assert "list_voices" in error_msg
        assert "abstract" in error_msg.lower()

    def test_complete_implementation_succeeds(self) -> None:
        """Test that implementing all abstract methods allows instantiation."""

        # Create a complete implementation
        class ConcreteProvider(TTSProvider):
            async def synthesize(self, text: str, voice: str) -> bytes:
                return b"audio data"

            async def list_voices(self) -> list[dict]:
                return [{"id": "test", "name": "Test Voice", "provider": "test"}]

        # Should be able to instantiate
        provider = ConcreteProvider()
        assert provider is not None
        assert isinstance(provider, TTSProvider)

    def test_method_signatures_enforced(self) -> None:
        """Test that method signatures match the abstract definition."""
        import inspect

        # Get abstract method signatures
        synthesize_sig = inspect.signature(TTSProvider.synthesize)
        list_voices_sig = inspect.signature(TTSProvider.list_voices)

        # Verify synthesize signature
        params = list(synthesize_sig.parameters.keys())
        assert params == ["self", "text", "voice"]
        assert synthesize_sig.parameters["text"].annotation == str
        assert synthesize_sig.parameters["voice"].annotation == str
        assert synthesize_sig.return_annotation == bytes

        # Verify list_voices signature
        params = list(list_voices_sig.parameters.keys())
        assert params == ["self"]
        assert str(list_voices_sig.return_annotation) == "typing.List[dict]"

    def test_inheritance_chain(self) -> None:
        """Test that concrete providers properly inherit from TTSProvider."""

        class TestProvider(TTSProvider):
            async def synthesize(self, text: str, voice: str) -> bytes:
                return b"test"

            async def list_voices(self) -> list[dict]:
                return []

        provider = TestProvider()

        # Verify inheritance
        assert isinstance(provider, TTSProvider)
        assert issubclass(TestProvider, TTSProvider)

        # Verify MRO (Method Resolution Order)
        mro = TestProvider.__mro__
        assert TTSProvider in mro
        assert object in mro
