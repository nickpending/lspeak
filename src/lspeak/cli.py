"""Typer CLI definition for lspeak."""

import asyncio
import logging
import sys
from pathlib import Path

import typer

from .config import load_config
from .core import list_available_voices, speak_text
from .models_manager import download_models, ensure_models_available
from .tts.errors import TTSAPIError, TTSAuthError

app = typer.Typer(help="Convert text to speech using AI voices")


async def get_queue_status() -> dict:
    """Get current queue status from daemon.

    Returns:
        Dictionary with queue status information
    """
    from .daemon.client import DaemonClient

    client = DaemonClient()
    return await client.send_request("queue_status", {})


def process_text_input(text: str | None) -> str:
    """Process text input and return the text to output.

    Args:
        text: Optional text input from CLI argument

    Returns:
        The text to output

    Raises:
        ValueError: If no text is provided
    """
    if text is None:
        raise ValueError("No text provided")

    return text


@app.command()
def speak(
    text: str | None = typer.Argument(None, help="Text to convert to speech"),
    file: Path | None = typer.Option(None, "-f", "--file", help="Read text from file"),
    output: Path | None = typer.Option(
        None, "-o", "--output", help="Save output to file instead of playing"
    ),
    voice: str | None = typer.Option(
        None, "-v", "--voice", help="Voice ID (from config if omitted)"
    ),
    provider: str | None = typer.Option(
        None, "-p", "--provider", help="TTS provider (from config if omitted)"
    ),
    no_cache: bool = typer.Option(False, "--no-cache", help="Disable semantic caching"),
    cache_threshold: float | None = typer.Option(
        None, "--cache-threshold", help="Similarity threshold for cache hits (0.0-1.0)"
    ),
    debug: bool = typer.Option(
        False, "--debug", help="Show verbose error messages and cache activity"
    ),
    list_voices: bool = typer.Option(
        False, "--list-voices", help="List available voices and exit"
    ),
    download_models_flag: bool = typer.Option(
        False, "--download-models", help="Download required ML models and exit"
    ),
    daemon_status: bool = typer.Option(
        False, "--daemon-status", help="Show daemon status and exit"
    ),
    daemon_stop: bool = typer.Option(
        False, "--daemon-stop", help="Stop the daemon and exit"
    ),
    daemon_restart: bool = typer.Option(
        False, "--daemon-restart", help="Restart the daemon and exit"
    ),
    no_queue: bool = typer.Option(
        False, "--no-queue", help="Skip queue for immediate playback"
    ),
    queue_status: bool = typer.Option(
        False, "--queue-status", help="Show queue status and exit"
    ),
    model: str | None = typer.Option(
        None,
        "-m",
        "--model",
        help="Model ID (e.g., eleven_turbo_v2_5 for ElevenLabs)",
    ),
) -> None:
    """Convert text to speech using AI voices."""
    # Configure logging for debug mode
    if debug:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

    # Handle queue status first (needs daemon running)
    if queue_status:
        try:
            status = asyncio.run(get_queue_status())
            if status.get("status") == "error":
                typer.echo(f"✗ Daemon not running: {status.get('error')}")
            else:
                typer.echo("=== Queue Status ===")
                typer.echo(f"Queue size: {status.get('queue_size', 0)}")

                current = status.get("current")
                if current:
                    typer.echo("\nCurrently playing:")
                    typer.echo(f"  Text: {current.get('text', '')[:50]}...")
                    typer.echo(f"  ID: {current.get('id', '')}")
                    typer.echo(f"  Started: {current.get('timestamp', '')}")

                waiting = status.get("waiting", [])
                if waiting:
                    typer.echo(f"\nWaiting in queue ({len(waiting)} items):")
                    for i, item in enumerate(waiting[:5], 1):  # Show first 5
                        typer.echo(f"  {i}. {item.get('text', '')[:50]}...")
                    if len(waiting) > 5:
                        typer.echo(f"  ... and {len(waiting) - 5} more")
                else:
                    typer.echo("\nNo items waiting in queue")
        except Exception as e:
            if debug:
                typer.echo(f"Debug - Queue status failed: {e!r}", err=True)
            else:
                typer.echo("Error: Failed to get queue status", err=True)
            raise typer.Exit(1) from e
        raise typer.Exit(0)

    # Handle daemon control commands
    if daemon_status:
        from .daemon.control import daemon_status as get_daemon_status

        try:
            status = asyncio.run(get_daemon_status())
            if status["running"]:
                typer.echo(
                    f"✓ Daemon running (PID: {status['pid']}, Models: {'loaded' if status['models_loaded'] else 'loading'})"
                )
            else:
                typer.echo("✗ Daemon not running")
                if "error" in status and debug:
                    typer.echo(f"Debug - Error: {status['error']}", err=True)
        except Exception as e:
            if debug:
                typer.echo(f"Debug - Status check failed: {e!r}", err=True)
            else:
                typer.echo("Error: Failed to check daemon status", err=True)
            raise typer.Exit(1) from None
        raise typer.Exit(0)

    if daemon_stop:
        from .daemon.control import daemon_stop

        try:
            if daemon_stop():
                typer.echo("Daemon stopped")
            else:
                typer.echo("Failed to stop daemon (may not be running)")
        except Exception as e:
            if debug:
                typer.echo(f"Debug - Stop failed: {e!r}", err=True)
            else:
                typer.echo("Error: Failed to stop daemon", err=True)
            raise typer.Exit(1) from None
        raise typer.Exit(0)

    if daemon_restart:
        from .daemon.control import daemon_restart

        try:
            if daemon_restart():
                typer.echo("Daemon restarted")
            else:
                typer.echo("Failed to restart daemon")
        except Exception as e:
            if debug:
                typer.echo(f"Debug - Restart failed: {e!r}", err=True)
            else:
                typer.echo("Error: Failed to restart daemon", err=True)
            raise typer.Exit(1) from None
        raise typer.Exit(0)

    # Handle --download-models flag first
    if download_models_flag:
        download_models()
        raise typer.Exit(0)

    # Resolve config values for flags not provided
    config = load_config()
    cache = not no_cache if no_cache else config.cache.enabled

    # Ensure models are available before continuing (sets offline mode if cached)
    if cache:  # Only check if cache is enabled
        ensure_models_available()

    # Handle --list-voices flag
    if list_voices:
        try:
            asyncio.run(list_available_voices(provider))
        except Exception as e:
            if debug:
                typer.echo(f"Debug - Failed to list voices: {e!r}", err=True)
            else:
                typer.echo(f"Error: Failed to list voices: {e}", err=True)
            raise typer.Exit(1) from None
        raise typer.Exit(0)
    # Get text from argument, file, or stdin (in priority order)
    if text is None:
        if file:
            try:
                text = file.read_text()
            except FileNotFoundError as e:
                if debug:
                    typer.echo(f"Debug - File not found: {file} ({e!r})", err=True)
                else:
                    typer.echo(f"Error: File not found: {file}", err=True)
                raise typer.Exit(1) from None
            except PermissionError as e:
                if debug:
                    typer.echo(f"Debug - Permission denied: {file} ({e!r})", err=True)
                else:
                    typer.echo(
                        f"Error: Permission denied reading file: {file}", err=True
                    )
                raise typer.Exit(1) from None
            except UnicodeDecodeError as e:
                if debug:
                    typer.echo(f"Debug - Decode error: {file} ({e!r})", err=True)
                else:
                    typer.echo(
                        f"Error: Unable to decode file as text: {file}", err=True
                    )
                raise typer.Exit(1) from None
        elif not sys.stdin.isatty():
            text = sys.stdin.read().strip()

    # Process text input
    try:
        output_text = process_text_input(text)
    except ValueError as e:
        if debug:
            typer.echo(f"Debug - Text processing error: {e!r}", err=True)
        else:
            typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1) from None

    # Convert text to speech
    try:
        # Convert Path to str for output_file parameter
        output_file = str(output) if output else None
        asyncio.run(
            speak_text(
                output_text,
                provider=provider,
                voice_id=voice,
                output_file=output_file,
                cache=cache,
                cache_threshold=cache_threshold,
                debug=debug,
                queue=not no_queue,  # Invert no_queue flag for queue parameter
                model=model,
            )
        )

        # If output was specified, confirm save
        if output:
            typer.echo(f"Audio saved to {output}")

    except TTSAuthError as e:
        if debug:
            typer.echo(f"Debug - Authentication error: {e!r}", err=True)
        else:
            typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1) from None
    except TTSAPIError as e:
        if debug:
            typer.echo(f"Debug - TTS API error: {e!r}", err=True)
        else:
            typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1) from None
    except OSError as e:
        if debug:
            typer.echo(f"Debug - File system error: {e!r}", err=True)
        else:
            typer.echo(f"Error: Failed to save audio file: {e}", err=True)
        raise typer.Exit(1) from None
    except RuntimeError as e:
        if debug:
            typer.echo(f"Debug - Audio playback error: {e!r}", err=True)
        else:
            typer.echo(f"Error: Failed to play audio: {e}", err=True)
        raise typer.Exit(1) from None
    except ValueError as e:
        if debug:
            typer.echo(f"Debug - Text processing error: {e!r}", err=True)
        else:
            typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1) from None
    except Exception as e:
        if debug:
            typer.echo(f"Debug - Unexpected error: {e!r}", err=True)
        else:
            typer.echo("Error: An unexpected error occurred", err=True)
        raise typer.Exit(1) from None
