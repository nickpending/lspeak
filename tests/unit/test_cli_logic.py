"""Unit tests for CLI logic functions."""

import sys
from pathlib import Path

import pytest

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from lspeak.cli import process_text_input


def test_process_text_input_with_valid_text() -> None:
    """Test that process_text_input returns text when provided."""
    result = process_text_input("Hello world")
    assert result == "Hello world"


def test_process_text_input_with_empty_string() -> None:
    """Test that process_text_input returns empty string when provided."""
    result = process_text_input("")
    assert result == ""


def test_process_text_input_with_whitespace() -> None:
    """Test that process_text_input preserves whitespace."""
    result = process_text_input("  Hello   world  ")
    assert result == "  Hello   world  "


def test_process_text_input_with_multiline() -> None:
    """Test that process_text_input handles multiline text."""
    text = "Line 1\nLine 2\nLine 3"
    result = process_text_input(text)
    assert result == text


def test_process_text_input_with_none_raises_value_error() -> None:
    """Test that process_text_input raises ValueError when text is None."""
    with pytest.raises(ValueError, match="No text provided"):
        process_text_input(None)
