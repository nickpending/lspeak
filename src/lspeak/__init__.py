"""lspeak - Unix-first text-to-speech CLI using AI voices."""

__version__ = "0.1.0"
__all__ = ["speak"]


def __getattr__(name: str):  # type: ignore[no-untyped-def]
    if name == "speak":
        from .api import speak

        return speak
    raise AttributeError(f"module 'lspeak' has no attribute {name!r}")
