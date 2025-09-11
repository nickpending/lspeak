"""Entry point for running lspeak as a module."""

from .cli import app


def main() -> None:
    """Main entry point for the lspeak CLI application."""
    app()


if __name__ == "__main__":
    main()
