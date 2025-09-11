"""Entry point for lspeak daemon server."""

import asyncio
import logging
import signal
import sys

from .server import LspeakDaemon

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def signal_handler(daemon_task: asyncio.Task) -> None:
    """Handle SIGTERM for graceful shutdown."""
    logger.info("Received shutdown signal")
    daemon_task.cancel()


async def main() -> None:
    """Run the daemon server."""
    daemon = LspeakDaemon()

    # Set up signal handling for graceful shutdown
    daemon_task = asyncio.create_task(daemon.start())

    # Handle SIGTERM
    def handle_sigterm() -> None:
        signal_handler(daemon_task)

    signal.signal(signal.SIGTERM, lambda s, f: handle_sigterm())

    try:
        await daemon_task
    except asyncio.CancelledError:
        logger.info("Daemon shutdown complete")
    except Exception as e:
        logger.error(f"Daemon error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Daemon stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)
