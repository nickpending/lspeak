"""Provider abstraction for text-to-speech services.

This module provides a registry pattern for managing TTS providers,
allowing runtime selection of different TTS backends.
"""

from typing import TYPE_CHECKING, Dict, Type

if TYPE_CHECKING:
    from .base import TTSProvider

from .elevenlabs import ElevenLabsProvider
from .system import SystemTTSProvider

__all__ = ["ProviderRegistry"]


class ProviderRegistry:
    """Registry for managing TTS providers.

    This class maintains a registry of available TTS providers,
    allowing registration and retrieval by name.
    """

    _providers: dict[str, type["TTSProvider"]] = {}

    @classmethod
    def register(cls, name: str, provider_class: type["TTSProvider"]) -> None:
        """Register a TTS provider.

        Args:
            name: Name to register the provider under
            provider_class: Provider class that implements TTSProvider
        """
        cls._providers[name] = provider_class

    @classmethod
    def get(cls, name: str) -> type["TTSProvider"]:
        """Get a provider class by name.

        Args:
            name: Name of the provider to retrieve

        Returns:
            Provider class

        Raises:
            KeyError: If provider name not found
        """
        if name not in cls._providers:
            available = ", ".join(cls._providers.keys()) if cls._providers else "none"
            raise KeyError(
                f"Provider '{name}' not found. Available providers: {available}"
            )
        return cls._providers[name]


# Register providers
ProviderRegistry.register("elevenlabs", ElevenLabsProvider)
ProviderRegistry.register("system", SystemTTSProvider)
