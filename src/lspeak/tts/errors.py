"""Custom TTS exceptions."""



class TTSError(Exception):
    """Base exception for TTS-related errors."""

    def __init__(
        self, message: str, original_error: Exception | None = None
    ) -> None:
        super().__init__(message)
        self.original_error = original_error


class TTSAuthError(TTSError):
    """Exception raised for authentication failures.

    This typically occurs when:
    - API key is missing or invalid
    - Account has insufficient credits
    - API key permissions are insufficient
    """

    pass


class TTSAPIError(TTSError):
    """Exception raised for API communication errors.

    This typically occurs when:
    - API server is unavailable (5xx errors)
    - Rate limits are exceeded (429 error)
    - Request format is invalid (4xx errors)
    - Network connectivity issues
    """

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        original_error: Exception | None = None,
    ) -> None:
        super().__init__(message, original_error)
        self.status_code = status_code
