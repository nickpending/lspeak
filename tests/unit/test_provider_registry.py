"""Unit tests for provider registry functionality."""

import sys
from pathlib import Path

import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from lspeak.providers import ProviderRegistry


class MockProvider:
    """Mock provider class for testing registration."""

    pass


class AnotherMockProvider:
    """Another mock provider class for testing multiple registrations."""

    pass


class TestProviderRegistry:
    """Test ProviderRegistry functionality."""

    def test_registry_starts_empty(self) -> None:
        """Test that registry starts with no providers."""
        # Clear any existing providers first
        ProviderRegistry._providers.clear()

        assert len(ProviderRegistry._providers) == 0
        assert ProviderRegistry._providers == {}

    def test_register_and_get_provider(self) -> None:
        """Test registering and retrieving a provider."""
        # Clear registry for clean test
        ProviderRegistry._providers.clear()

        # Register a provider
        ProviderRegistry.register("test", MockProvider)

        # Verify it was registered
        assert "test" in ProviderRegistry._providers
        assert ProviderRegistry._providers["test"] is MockProvider

        # Retrieve the provider
        provider_class = ProviderRegistry.get("test")
        assert provider_class is MockProvider

    def test_register_multiple_providers(self) -> None:
        """Test registering multiple providers."""
        # Clear registry for clean test
        ProviderRegistry._providers.clear()

        # Register multiple providers
        ProviderRegistry.register("provider1", MockProvider)
        ProviderRegistry.register("provider2", AnotherMockProvider)

        # Verify both were registered
        assert len(ProviderRegistry._providers) == 2
        assert ProviderRegistry.get("provider1") is MockProvider
        assert ProviderRegistry.get("provider2") is AnotherMockProvider

    def test_register_overwrites_existing(self) -> None:
        """Test that registering with same name overwrites existing provider."""
        # Clear registry for clean test
        ProviderRegistry._providers.clear()

        # Register initial provider
        ProviderRegistry.register("test", MockProvider)
        assert ProviderRegistry.get("test") is MockProvider

        # Register different provider with same name
        ProviderRegistry.register("test", AnotherMockProvider)
        assert ProviderRegistry.get("test") is AnotherMockProvider
        assert len(ProviderRegistry._providers) == 1

    def test_get_nonexistent_provider_error(self) -> None:
        """Test that getting non-existent provider raises KeyError with helpful message."""
        # Clear registry for clean test
        ProviderRegistry._providers.clear()

        # Test with empty registry
        with pytest.raises(
            KeyError, match="Provider 'missing' not found. Available providers: none"
        ):
            ProviderRegistry.get("missing")

    def test_get_nonexistent_provider_with_available(self) -> None:
        """Test error message includes available providers when some exist."""
        # Clear registry for clean test
        ProviderRegistry._providers.clear()

        # Register some providers
        ProviderRegistry.register("elevenlabs", MockProvider)
        ProviderRegistry.register("system", AnotherMockProvider)

        # Test error message includes available providers
        with pytest.raises(
            KeyError,
            match="Provider 'missing' not found. Available providers: elevenlabs, system",
        ):
            ProviderRegistry.get("missing")

    def test_registry_persistence_across_calls(self) -> None:
        """Test that registry maintains state across multiple calls."""
        # Clear registry for clean test
        ProviderRegistry._providers.clear()

        # Register a provider
        ProviderRegistry.register("persistent", MockProvider)

        # Verify it persists across multiple get calls
        provider1 = ProviderRegistry.get("persistent")
        provider2 = ProviderRegistry.get("persistent")

        assert provider1 is provider2
        assert provider1 is MockProvider
