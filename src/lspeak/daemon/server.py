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
from ipaddress import ip_address
from typing import Any
from urllib.parse import urlparse

from fastapi import Depends, FastAPI, HTTPException, Request, Security
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

from lspeak.config import load_config
from lspeak.daemon.paths import get_runtime_dir, get_socket_path
from lspeak.tts.pipeline import TTSPipeline

logger = logging.getLogger(__name__)


# Custom CORS middleware for private network validation
class PrivateNetworkCORS(CORSMiddleware):
    """CORS middleware that only allows private network origins.

    Uses Python's ipaddress module to validate that request origins come from:
    - localhost (127.0.0.1, ::1, 'localhost')
    - RFC1918 private networks (10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16)
    - Link-local (169.254.0.0/16)

    This prevents stolen API key attacks from external sites while allowing
    LAN and localhost access for development and home network use.
    """

    def is_allowed_origin(self, origin: str) -> bool:
        """Check if origin is from a private network."""
        try:
            parsed = urlparse(origin)
            host = parsed.hostname

            if not host:
                return False

            # Allow localhost by name
            if host in ("localhost", "127.0.0.1", "::1"):
                return True

            # Check if IP is private using Python stdlib
            ip = ip_address(host)
            return ip.is_private

        except (ValueError, AttributeError):
            # Invalid origin format or IP address
            return False


# API Error handling (prismis pattern)
class APIError(HTTPException):
    """Base API error with consistent format."""

    def __init__(self, status_code: int, message: str):
        super().__init__(status_code=status_code, detail=message)
        self.message = message


class ValidationError(APIError):
    """422 - Request validation failed."""

    def __init__(self, message: str):
        super().__init__(422, message)


class AuthenticationError(APIError):
    """401 - Authentication failed (correct HTTP semantics, not prismis's 403)."""

    def __init__(self, message: str = "Invalid API key"):
        super().__init__(401, message)


class ServerError(APIError):
    """500 - Internal server error."""

    def __init__(self, message: str):
        super().__init__(500, message)


# Pydantic models for HTTP API
class SpeakRequest(BaseModel):
    """Request model for /speak endpoint."""

    text: str = Field(..., description="Text to synthesize and speak")
    provider: str | None = Field(
        default=None, description="TTS provider (from config if omitted)"
    )
    voice: str | None = Field(
        default=None, description="Voice ID (from config if omitted)"
    )
    output: str | None = Field(default=None, description="Output file path")
    cache: bool | None = Field(
        default=None, description="Enable semantic caching (from config if omitted)"
    )
    cache_threshold: float | None = Field(
        default=None,
        description="Similarity threshold for cache hits (from config if omitted)",
    )
    queue: bool = Field(
        default=True, description="Queue request or process immediately"
    )
    debug: bool = Field(default=False, description="Enable debug logging")
    model: str | None = Field(
        default=None,
        description="Model ID (e.g., eleven_turbo_v2_5 for ElevenLabs)",
    )


class APIResponse(BaseModel):
    """Standard API response envelope (prismis pattern)."""

    success: bool = Field(..., description="Whether the operation succeeded")
    message: str = Field(..., description="Human-readable message")
    data: dict[str, Any] | None = Field(None, description="Response data if any")


# API key authentication (prismis pattern with X-API-Key header)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: str | None = Security(api_key_header)) -> str | None:
    """Verify API key from X-API-Key header (optional authentication).

    Args:
        api_key: API key from X-API-Key header

    Returns:
        The validated API key if auth is enabled, None if auth disabled

    Raises:
        AuthenticationError: 401 if API key is invalid or missing when required
    """
    # Check if authentication is enabled via environment variable
    expected_key = os.getenv("LSPEAK_API_KEY")

    # If no API key configured, authentication is disabled (backward compatible)
    if not expected_key:
        return None

    # Authentication is enabled - validate the provided key
    if not api_key:
        raise AuthenticationError("Missing API key. Please provide X-API-Key header")

    if api_key != expected_key:
        raise AuthenticationError("Invalid API key")

    return api_key


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
        self.start_time = datetime.now()

        # Queue infrastructure
        self.speech_queue: asyncio.Queue[QueueItem] = asyncio.Queue()
        self.queue_processor_task: asyncio.Task[None] | None = None
        self.current_item: QueueItem | None = None
        self.queue_items: list[QueueItem] = []

        # HTTP API infrastructure
        self.http_host = self._get_http_host()
        self.http_port = self._get_http_port()
        self.app: FastAPI | None = None
        self.http_server_task: asyncio.Task[None] | None = None

        # Initialize FastAPI app if HTTP port configured
        if self.http_port:
            self.app = FastAPI(
                title="lspeak HTTP API",
                description="Text-to-speech API with semantic caching",
                version="0.1.0",
            )
            self._setup_cors()
            self._setup_exception_handlers()
            self._setup_http_routes()

    def _get_http_host(self) -> str:
        """Get HTTP bind address from config (env var override supported)."""
        config = load_config()
        return config.http.host

    def _get_http_port(self) -> int | None:
        """Get HTTP port from config (env var override supported)."""
        config = load_config()
        return config.http.port

    def _setup_cors(self) -> None:
        """Configure CORS middleware for private network access.

        Uses PrivateNetworkCORS to allow requests from:
        - localhost (127.0.0.1, ::1, 'localhost')
        - RFC1918 private networks (10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16)
        - Link-local addresses (169.254.0.0/16)

        This prevents stolen API key attacks from external sites.
        """
        if not self.app:
            return

        # Use custom middleware with IP-based validation (more robust than regex)
        self.app.add_middleware(
            PrivateNetworkCORS,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    def _setup_exception_handlers(self) -> None:
        """Setup FastAPI exception handlers for consistent error formatting."""
        if not self.app:
            return

        @self.app.exception_handler(APIError)
        async def api_error_handler(request: Request, exc: APIError) -> JSONResponse:
            """Handle custom APIError exceptions with consistent format."""
            return JSONResponse(
                status_code=exc.status_code,
                content={"success": False, "message": exc.message, "data": None},
            )

        @self.app.exception_handler(RequestValidationError)
        async def validation_error_handler(
            request: Request, exc: RequestValidationError
        ) -> JSONResponse:
            """Handle FastAPI validation errors with consistent format."""
            first_error = exc.errors()[0]
            field = " -> ".join(str(loc) for loc in first_error["loc"])
            return JSONResponse(
                status_code=422,
                content={
                    "success": False,
                    "message": f"Validation error in {field}: {first_error['msg']}",
                    "data": None,
                },
            )

    def _setup_http_routes(self) -> None:
        """Setup FastAPI routes for HTTP API."""
        if not self.app:
            return

        @self.app.post(
            "/speak",
            response_model=APIResponse,
            dependencies=[Depends(verify_api_key)],
        )
        async def speak_endpoint(request: SpeakRequest) -> APIResponse:
            """Queue or process speech synthesis request."""
            try:
                result = await self.handle_speak(request.model_dump())

                # Check if error occurred
                if result.get("error"):
                    raise ServerError(result["error"])

                # Format response based on whether queued or immediate
                if result.get("queued"):
                    return APIResponse(
                        success=True,
                        message="Speech queued successfully",
                        data={
                            "queue_id": result["queue_id"],
                            "queue_position": result["queue_position"],
                            "timestamp": result["timestamp"],
                        },
                    )
                else:
                    return APIResponse(
                        success=True,
                        message="Speech processed successfully",
                        data={
                            "played": result.get("played"),
                            "saved": result.get("saved"),
                            "cached": result.get("cached"),
                        },
                    )
            except APIError:
                raise
            except Exception as e:
                raise ServerError(f"Failed to process speech: {e!s}") from e

        @self.app.get(
            "/status",
            response_model=APIResponse,
            dependencies=[Depends(verify_api_key)],
        )
        async def status_endpoint() -> APIResponse:
            """Get daemon status including uptime and model state."""
            try:
                uptime = (datetime.now() - self.start_time).total_seconds()
                return APIResponse(
                    success=True,
                    message="Daemon running",
                    data={
                        "pid": os.getpid(),
                        "models_loaded": self.models_loaded,
                        "uptime": uptime,
                    },
                )
            except Exception as e:
                raise ServerError(f"Failed to get status: {e!s}") from e

        @self.app.get(
            "/queue",
            response_model=APIResponse,
            dependencies=[Depends(verify_api_key)],
        )
        async def queue_endpoint() -> APIResponse:
            """Get current queue status."""
            try:
                result = await self.handle_queue_status()
                return APIResponse(
                    success=True,
                    message=f"Queue has {result['queue_size']} items",
                    data=result,
                )
            except Exception as e:
                raise ServerError(f"Failed to get queue status: {e!s}") from e

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
            # Extract parameters, resolve from config when not provided
            config = load_config()
            text = params.get("text", "")
            provider = params.get("provider") or config.tts.provider
            voice = params.get("voice") or config.tts.voice
            output = params.get("output")
            cache = (
                params.get("cache")
                if params.get("cache") is not None
                else config.cache.enabled
            )
            cache_threshold = (
                params.get("cache_threshold")
                if params.get("cache_threshold") is not None
                else config.cache.threshold
            )
            debug = params.get("debug", False)
            model = params.get("model")

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
                model=model,
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

    async def _start_http_server(self) -> None:
        """Start Uvicorn HTTP server in same event loop."""
        if not self.app or not self.http_port:
            return

        try:
            import uvicorn

            config = uvicorn.Config(
                app=self.app,
                host=self.http_host,
                port=self.http_port,
                log_level="warning",  # Reduce noise in logs
                access_log=False,  # Disable access logs
            )

            server = uvicorn.Server(config)
            await server.serve()
        except Exception as e:
            logger.error(f"HTTP server failed to start: {e}")
            logger.error("Unix socket daemon will continue running without HTTP API")
            # Don't propagate - let Unix socket continue operating

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
            except BlockingIOError as e:
                logger.error("Another daemon is already running (lock held)")
                if self.lock_fd is not None:
                    os.close(self.lock_fd)
                raise RuntimeError("Another daemon process is already running") from e

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

            # Start HTTP server if configured
            if self.http_port and self.app:
                self.http_server_task = asyncio.create_task(self._start_http_server())
                logger.info(
                    f"✓ HTTP API listening on http://localhost:{self.http_port}"
                )
                logger.info(
                    f"✓ API docs available at http://localhost:{self.http_port}/docs"
                )
            else:
                logger.info("HTTP API disabled (set LSPEAK_HTTP_PORT to enable)")

            # Serve forever
            async with server:
                await server.serve_forever()

        except Exception as e:
            logger.error(f"Failed to start daemon: {e}")
            # Release lock if we acquired it
            if self.lock_fd is not None:
                with contextlib.suppress(Exception):
                    os.close(self.lock_fd)
            raise
        finally:
            # Always clean up lock on exit
            if self.lock_fd is not None:
                with contextlib.suppress(Exception):
                    os.close(self.lock_fd)
                    logger.debug("Released daemon lock")
