"""Model management for lspeak - handles downloading and caching of ML models."""

import os
import sys
from pathlib import Path

# Model we use for embeddings
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"


def get_model_cache_dir() -> Path:
    """Get the huggingface cache directory for models."""
    cache_home = os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")
    return Path(cache_home) / "hub"


def check_model_cached(model_name: str = EMBEDDING_MODEL) -> bool:
    """Check if a model is already cached locally.

    Args:
        model_name: Name of the model to check

    Returns:
        True if model is cached, False otherwise
    """
    cache_dir = get_model_cache_dir()
    # Convert model name to cache directory format
    model_dir_name = f"models--{model_name.replace('/', '--')}"
    model_path = cache_dir / model_dir_name

    # Check if directory exists and has files
    if model_path.exists() and model_path.is_dir():
        # Check for key files that indicate a complete download
        required_files = ["pytorch_model.bin", "config.json"]
        for root, dirs, files in os.walk(model_path):
            # If we find any of the required files, model is likely cached
            if any(f in files for f in required_files):
                return True
            # Also check for safetensors format
            if any(f.endswith(".safetensors") for f in files):
                return True

    return False


def download_models() -> None:
    """Download all required models for lspeak.

    This downloads the embedding model used for semantic caching.
    Shows progress and handles errors gracefully.
    """
    print("Downloading models for lspeak...")
    print(f"Model: {EMBEDDING_MODEL}")

    try:
        # Import here to avoid loading at module import time
        from sentence_transformers import SentenceTransformer

        # Force online mode for downloading
        if "HF_HUB_OFFLINE" in os.environ:
            del os.environ["HF_HUB_OFFLINE"]
        if "TRANSFORMERS_OFFLINE" in os.environ:
            del os.environ["TRANSFORMERS_OFFLINE"]

        print("Downloading... (this may take a few minutes)")
        model = SentenceTransformer(EMBEDDING_MODEL)

        # Verify it loaded
        dim = model.get_sentence_embedding_dimension()
        print(f"✓ Model downloaded successfully (dimension: {dim})")
        print(f"✓ Models cached at: {get_model_cache_dir()}")

    except Exception as e:
        print(f"✗ Failed to download models: {e}", file=sys.stderr)
        print("\nTroubleshooting:", file=sys.stderr)
        print("1. Check your internet connection", file=sys.stderr)
        print("2. Try setting HF_HUB_DISABLE_SYMLINKS=1", file=sys.stderr)
        print("3. Check disk space in ~/.cache/huggingface/", file=sys.stderr)
        sys.exit(1)


def configure_offline_mode() -> tuple[bool, str | None]:
    """Configure offline mode if models are cached.

    Returns:
        Tuple of (models_available, error_message)
        - (True, None) if models are cached and offline mode set
        - (False, error_msg) if models missing
    """
    if check_model_cached():
        # Models are cached, use offline mode to avoid network checks
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        return True, None
    else:
        error_msg = (
            "Required models not found. Please run:\n"
            "  lspeak --download-models\n"
            "\n"
            "This will download the embedding model (~400MB) used for semantic caching.\n"
            "You only need to do this once."
        )
        return False, error_msg


def ensure_models_available() -> None:
    """Ensure models are available, error if not.

    Raises:
        SystemExit: If models are not available
    """
    available, error_msg = configure_offline_mode()
    if not available:
        print(error_msg, file=sys.stderr)
        sys.exit(1)
