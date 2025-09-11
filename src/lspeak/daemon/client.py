"""Unix socket client for communicating with lspeak daemon."""

import asyncio
import json
import uuid
from typing import Any

from lspeak.daemon.paths import get_socket_path


class DaemonClient:
    """Client for communicating with lspeak daemon via Unix socket."""

    def __init__(self) -> None:
        """Initialize client with socket path matching daemon server."""
        self.socket_path = get_socket_path()

    async def send_request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        """Send JSON request to daemon and return parsed response.

        Args:
            method: RPC method name (e.g., "speak", "status")
            params: Parameters for the method

        Returns:
            Parsed JSON response from daemon

        Raises:
            FileNotFoundError: When daemon socket doesn't exist
        """
        try:
            # Connect to Unix socket
            reader, writer = await asyncio.open_unix_connection(str(self.socket_path))

            # Build request with unique ID
            request = {"id": str(uuid.uuid4()), "method": method, "params": params}

            # Send JSON request
            writer.write(json.dumps(request).encode())
            await writer.drain()

            # Read response (up to 64KB)
            data = await reader.read(65536)
            response = json.loads(data.decode())

            # Close connection
            writer.close()
            await writer.wait_closed()

            return response

        except FileNotFoundError:
            # Daemon not running - return error dict
            return {"status": "error", "error": "Daemon not running"}
        except ConnectionRefusedError:
            # Socket exists but connection refused
            return {"status": "error", "error": "Daemon not responding"}
        except json.JSONDecodeError as e:
            # Invalid response from daemon
            return {"status": "error", "error": f"Invalid response from daemon: {e}"}
        except Exception as e:
            # Unexpected error
            return {"status": "error", "error": f"Communication error: {e}"}

    async def speak(self, **kwargs) -> dict[str, Any]:
        """Send speak request to daemon with provided parameters.

        Args:
            **kwargs: Parameters to pass to speak method
                - text: Text to speak
                - provider: TTS provider (default: "elevenlabs")
                - voice: Voice ID to use
                - output: Output file path
                - cache: Whether to use cache (default: True)
                - cache_threshold: Similarity threshold (default: 0.95)
                - debug: Debug mode flag
                - queue: Whether to queue for serial playback (default: True)

        Returns:
            Response from daemon with result or error
        """
        return await self.send_request("speak", kwargs)
