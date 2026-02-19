"""Provider abstraction for text-to-speech services.

This module provides a registry pattern for managing TTS providers,
allowing runtime selection of different TTS backends.
"""

from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from .base import TTSProvider

from .elevenlabs import ElevenLabsProvider
from .kokoro import KokoroProvider
from .system import SystemTTSProvider

__all__ = ["ProviderRegistry"]


class ProviderRegistry:
    """Registry for managing TTS providers.

    This class maintains a registry of available TTS providers,
    allowing registration and retrieval by name.
    """

    _providers: ClassVar[dict[str, type["TTSProvider"]]] = {}
    _instances: ClassVar[dict[str, "TTSProvider"]] = {}

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

    @classmethod
    def get_instance(cls, name: str) -> "TTSProvider":
        """Get a cached provider instance by name.

        Creates the instance on first call, returns cached instance after.
        Keeps models warm in memory across requests.

        Args:
            name: Name of the provider

        Returns:
            Cached provider instance

        Raises:
            KeyError: If provider name not found
        """
        if name not in cls._instances:
            provider_class = cls.get(name)
            cls._instances[name] = provider_class()
        return cls._instances[name]


# Register providers
ProviderRegistry.register("elevenlabs", ElevenLabsProvider)
ProviderRegistry.register("kokoro", KokoroProvider)
ProviderRegistry.register("system", SystemTTSProvider)
