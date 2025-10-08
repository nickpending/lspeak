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


async def main() -> None:
    """Run the daemon server."""
    daemon = LspeakDaemon()

    # Set up signal handling for graceful shutdown
    daemon_task = asyncio.create_task(daemon.start())

    # Handle SIGTERM using asyncio's proper signal handling
    loop = asyncio.get_running_loop()

    def handle_sigterm() -> None:
        logger.info("Received shutdown signal")
        daemon_task.cancel()

    # Use asyncio's signal handler instead of signal.signal()
    loop.add_signal_handler(signal.SIGTERM, handle_sigterm)

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
