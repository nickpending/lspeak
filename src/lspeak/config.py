"""Configuration management for lspeak.

Loads configuration from ~/.config/lspeak/config.toml.
Priority chain: CLI flags > env vars > config file.
"""

import os
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

CONFIG_DIR = Path.home() / ".config" / "lspeak"
CONFIG_PATH = CONFIG_DIR / "config.toml"

DEFAULT_CONFIG = """\
# lspeak configuration
# https://github.com/nickpending/lspeak

[tts]
# Provider: "kokoro" (local, free), "elevenlabs" (cloud), "system" (OS built-in)
provider = "kokoro"

# Voice ID for speech synthesis
# Kokoro: af_heart, af_alloy, af_bella, af_jessica, af_nova, af_sarah,
#          am_adam, am_echo, am_eric, am_liam, am_michael, am_onyx, am_puck
# ElevenLabs: use `lspeak --list-voices --provider elevenlabs`
voice = "af_heart"

# Compute device for local models: "auto", "mps" (Apple Silicon), "cuda", "cpu"
device = "auto"

[http]
# Bind address: "127.0.0.1" = localhost only, "0.0.0.0" = allow LAN access
host = "127.0.0.1"

# HTTP API port (uncomment to enable)
# port = 8888

[cache]
# Semantic caching of TTS results
enabled = true

# Similarity threshold for cache hits (0.0-1.0)
threshold = 0.95

# API keys are read from environment variables, not this file:
#   ELEVENLABS_API_KEY  - ElevenLabs provider
#   LSPEAK_API_KEY      - HTTP API authentication
"""


@dataclass(frozen=True)
class TTSConfig:
    """TTS provider configuration."""

    provider: str
    voice: str
    device: str


@dataclass(frozen=True)
class HTTPConfig:
    """HTTP API configuration."""

    host: str
    port: int | None


@dataclass(frozen=True)
class CacheConfig:
    """Semantic cache configuration."""

    enabled: bool
    threshold: float


@dataclass(frozen=True)
class LspeakConfig:
    """Top-level lspeak configuration."""

    tts: TTSConfig
    http: HTTPConfig
    cache: CacheConfig


_cached_config: LspeakConfig | None = None


def generate_config() -> Path:
    """Generate default config file at ~/.config/lspeak/config.toml."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(DEFAULT_CONFIG)
    return CONFIG_PATH


def load_config() -> LspeakConfig:
    """Load configuration from config file with env var overrides.

    On first run, generates the config file and exits so the user
    can review it before proceeding.

    Returns:
        Loaded and validated LspeakConfig.

    Raises:
        SystemExit: If config is missing (after generating) or invalid.
    """
    global _cached_config
    if _cached_config is not None:
        return _cached_config

    if not CONFIG_PATH.exists():
        path = generate_config()
        print(
            f"No config found. Generated {path} — review and run again.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    with open(CONFIG_PATH, "rb") as f:
        data = tomllib.load(f)

    tts = data.get("tts", {})
    http_cfg = data.get("http", {})
    cache = data.get("cache", {})

    # Validate required fields
    missing = []
    if "provider" not in tts:
        missing.append("tts.provider")
    if "voice" not in tts:
        missing.append("tts.voice")
    if "host" not in http_cfg:
        missing.append("http.host")
    if "enabled" not in cache:
        missing.append("cache.enabled")
    if "threshold" not in cache:
        missing.append("cache.threshold")

    if missing:
        print(
            f"Missing required config values: {', '.join(missing)}",
            file=sys.stderr,
        )
        print(f"Edit {CONFIG_PATH} or delete it to regenerate.", file=sys.stderr)
        raise SystemExit(1)

    # Env vars override config file values
    port_str = os.getenv("LSPEAK_HTTP_PORT", str(http_cfg.get("port", "")))

    _cached_config = LspeakConfig(
        tts=TTSConfig(
            provider=os.getenv("LSPEAK_PROVIDER", tts["provider"]),
            voice=os.getenv("LSPEAK_VOICE", tts["voice"]),
            device=os.getenv("LSPEAK_DEVICE", tts.get("device", "auto")),
        ),
        http=HTTPConfig(
            host=os.getenv("LSPEAK_HTTP_HOST", http_cfg["host"]),
            port=int(port_str) if port_str else None,
        ),
        cache=CacheConfig(
            enabled=cache["enabled"],
            threshold=cache["threshold"],
        ),
    )

    return _cached_config
