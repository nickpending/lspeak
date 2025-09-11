"""Test package structure and imports."""

import sys
from pathlib import Path

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_package_imports() -> None:
    """Test that lspeak package can be imported."""
    import lspeak

    assert lspeak.__version__ == "0.1.0"


def test_main_module_imports() -> None:
    """Test that main module can be imported without error."""
    from lspeak.__main__ import main

    # Should be able to import the main function
    assert callable(main)
