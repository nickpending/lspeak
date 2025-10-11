"""Unix socket daemon server for lspeak."""

import asyncio
import contextlib
import fcntl
import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from lspeak.daemon.paths import get_runtime_dir, get_socket_path
from lspeak.tts.pipeline import TTSPipeline

logger = logging.getLogger(__name__)


@dataclass
class QueueItem:
    """Queue item with metadata for speech processing."""

    params: dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))


class LspeakDaemon:
    """Unix socket daemon server for fast TTS response times."""

    def __init__(self) -> None:
        """Initialize daemon with socket path and model state."""
        self.socket_path = get_socket_path()
        self.lock_path = get_runtime_dir() / "daemon.lock"
        self.lock_fd: int | None = None
        self.models_loaded = False
        self.cache_manager: Any | None = None
        self.audio_player: Any | None = None  # Initialize on first use

        # Queue infrastructure
        self.speech_queue: asyncio.Queue[QueueItem] = asyncio.Queue()
        self.queue_processor_task: asyncio.Task[None] | None = None
        self.current_item: QueueItem | None = None
        self.queue_items: list[QueueItem] = []

    async def handle_client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        """Handle incoming client connection and JSON request."""
        try:
            # Read JSON request
            data = await reader.read(65536)
            if not data:
                return

            request = json.loads(data.decode())
            logger.debug(f"Received request: {request.get('method', 'unknown')}")

            # Process based on method
            if request["method"] == "speak":
                result = await self.handle_speak(request["params"])
            elif request["method"] == "status":
                result = {"pid": os.getpid(), "models_loaded": self.models_loaded}
            elif request["method"] == "queue_status":
                result = await self.handle_queue_status()
            else:
                result = {"error": f"Unknown method: {request['method']}"}

            # Send JSON response
            response = {
                "id": request.get("id"),
                "status": "success" if "error" not in result else "error",
                "result": result,
            }

            writer.write(json.dumps(response).encode())
            await writer.drain()

        except json.JSONDecodeError as e:
            error_response = {
                "id": None,
                "status": "error",
                "result": {"error": f"Invalid JSON: {e}"},
            }
            writer.write(json.dumps(error_response).encode())
            await writer.drain()

        except Exception as e:
            logger.error(f"Error handling client: {e}")
            error_response = {
                "id": None,
                "status": "error",
                "result": {"error": str(e)},
            }
            writer.write(json.dumps(error_response).encode())
            await writer.drain()

        finally:
            writer.close()
            await writer.wait_closed()

    async def process_queue(self) -> None:
        """Process queue items serially to prevent audio overlap."""
        while True:
            try:
                # Get next item from queue
                item = await self.speech_queue.get()
                self.current_item = item

                # Process speech synthesis and audio playback serially
                await self._process_speech_async(item.params)

                # Remove from queue_items list and clear current
                if item in self.queue_items:
                    self.queue_items.remove(item)
                self.current_item = None

            except Exception as e:
                logger.error(f"Error processing queue item: {e}")
                self.current_item = None

    async def _process_speech_async(self, params: dict[str, Any]) -> dict[str, Any]:
        """Process speech synthesis asynchronously."""
        try:
            # Process the speech (this handles TTS and audio playback)
            result = await self._process_speech(params)
            return result

        except Exception as e:
            logger.error(f"Error in speech processing: {e}")
            return {"error": str(e)}

    async def _process_speech(self, params: dict[str, Any]) -> dict[str, Any]:
        """Process speech synthesis using TTSPipeline orchestrator."""
        try:
            # Extract parameters with defaults
            text = params.get("text", "")
            provider = params.get("provider", "elevenlabs")
            voice = params.get("voice")
            output = params.get("output")
            cache = params.get("cache", True)
            cache_threshold = params.get("cache_threshold", 0.95)
            debug = params.get("debug", False)

            # Create pipeline with pre-loaded instances for sub-second response
            pipeline = TTSPipeline(
                cache_manager=self.cache_manager,
                audio_player=self.audio_player,
            )

            # Process through unified pipeline
            return await pipeline.process(
                text=text,
                provider=provider,
                voice=voice,
                output=output,
                cache=cache,
                cache_threshold=cache_threshold,
                debug=debug,
            )

        except Exception as e:
            logger.error(f"Error in _process_speech: {e}")
            return {"error": str(e)}

    async def handle_speak(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle speak request - queue or process immediately based on queue parameter."""
        # Check if we should queue (default True)
        queue = params.get("queue", True)

        if queue:
            # Create queue item
            item = QueueItem(params=params)

            # Add to queue and tracking list
            await self.speech_queue.put(item)
            self.queue_items.append(item)

            # Return queued response
            return {
                "queued": True,
                "queue_id": item.id,
                "queue_position": len(self.queue_items),
                "timestamp": item.timestamp.isoformat(),
            }
        else:
            # Process immediately (bypass queue)
            return await self._process_speech(params)

    async def handle_queue_status(self) -> dict[str, Any]:
        """Return current queue status information."""
        return {
            "queue_size": len(self.queue_items),
            "current": self._format_queue_item(self.current_item)
            if self.current_item
            else None,
            "waiting": [
                self._format_queue_item(item)
                for item in self.queue_items
                if item != self.current_item
            ],
        }

    def _format_queue_item(self, item: QueueItem) -> dict[str, Any]:
        """Format queue item for status response."""
        if not item:
            return {}

        text = item.params.get("text", "")
        truncated = text[:50] + "..." if len(text) > 50 else text

        return {
            "id": item.id,
            "text": truncated,
            "timestamp": item.timestamp.isoformat(),
            "provider": item.params.get("provider", "elevenlabs"),
            "voice": item.params.get("voice", "default"),
        }

    async def start(self) -> None:
        """Start the daemon server with model loading."""
        try:
            logger.info("Starting lspeak daemon...")

            # Acquire exclusive daemon lock FIRST
            # This prevents multiple daemons from running simultaneously
            self.lock_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            try:
                self.lock_fd = os.open(
                    str(self.lock_path), os.O_CREAT | os.O_RDWR, 0o600
                )
                fcntl.flock(self.lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                logger.info("✓ Acquired exclusive daemon lock")
            except BlockingIOError:
                logger.error("Another daemon is already running (lock held)")
                if self.lock_fd is not None:
                    os.close(self.lock_fd)
                raise RuntimeError("Another daemon process is already running")

            # Load models once at startup for fast response
            logger.info("Loading models (this may take a moment)...")
            from ..cache.manager import SemanticCacheManager

            # Initialize cache manager and force model loading
            self.cache_manager = SemanticCacheManager()

            # Warm up the cache to force embedding model loading
            logger.info("Warming up embedding models...")
            with contextlib.suppress(Exception):
                await self.cache_manager.get_cached_audio("warmup", "system", "default")

            # Initialize audio player during startup (not during first request)
            logger.info("Initializing audio system...")
            from ..audio.player import AudioPlayer

            self.audio_player = AudioPlayer()
            logger.info("✓ Audio system initialized")

            self.models_loaded = True
            logger.info("✓ Models loaded and ready")

            # Start queue processor task
            self.queue_processor_task = asyncio.create_task(self.process_queue())
            logger.info("✓ Queue processor started")

            # Check if socket exists and is in use
            if self.socket_path.exists():
                # Try to connect to see if another daemon is using it
                import socket as sock

                try:
                    test_sock = sock.socket(sock.AF_UNIX, sock.SOCK_STREAM)
                    test_sock.settimeout(0.5)
                    test_sock.connect(str(self.socket_path))
                    test_sock.close()
                    # Another daemon is running, we should exit
                    logger.error(
                        f"Another daemon is already running on {self.socket_path}"
                    )
                    raise RuntimeError(
                        f"Socket {self.socket_path} is already in use by another daemon"
                    )
                except (OSError, ConnectionError, TimeoutError):
                    # Socket exists but nobody listening, safe to remove
                    self.socket_path.unlink()
                    logger.debug(f"Removed stale socket: {self.socket_path}")

            # Start Unix socket server
            server = await asyncio.start_unix_server(
                self.handle_client, str(self.socket_path)
            )

            logger.info(f"✓ Daemon listening on {self.socket_path}")
            logger.info(f"✓ PID: {os.getpid()}")

            # Serve forever
            async with server:
                await server.serve_forever()

        except Exception as e:
            logger.error(f"Failed to start daemon: {e}")
            # Release lock if we acquired it
            if self.lock_fd is not None:
                try:
                    os.close(self.lock_fd)
                except Exception:
                    pass
            raise
        finally:
            # Always clean up lock on exit
            if self.lock_fd is not None:
                try:
                    os.close(self.lock_fd)
                    logger.debug("Released daemon lock")
                except Exception:
                    pass
