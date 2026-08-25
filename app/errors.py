"""Typed errors and the single JSON envelope they are rendered into.

Design note -- the split that matters:

  * *Unusable audio* is NOT an error. A driver answering in a loud cab and
    saying "yeah?" is an ordinary, expected outcome on a logistics call. That
    returns 200 with audio_quality="insufficient" and unknown/unknown, so the
    voice agent can fall back to a neutral persona without special-casing an
    exception path.

  * *Malformed input* -- no body, undecodable bytes, oversized upload -- is a
    4xx, because the caller has a bug to fix.

Conflating the two is the most common way this kind of service ends up either
throwing on normal traffic or silently emitting confident nonsense.
"""

from __future__ import annotations

from fastapi import Request, status
from fastapi.responses import JSONResponse

from app.schemas import ErrorResponse


class VoiceAttributeError(Exception):
    """Base class. `code` is stable API surface; `message` is human-facing."""

    code: str = "INTERNAL_ERROR"
    http_status: int = status.HTTP_500_INTERNAL_SERVER_ERROR

    def __init__(self, message: str, *, detail: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail


class EmptyAudioError(VoiceAttributeError):
    code = "EMPTY_AUDIO"
    http_status = status.HTTP_400_BAD_REQUEST


class AudioTooLargeError(VoiceAttributeError):
    code = "AUDIO_TOO_LARGE"
    http_status = status.HTTP_413_CONTENT_TOO_LARGE


class AudioTooShortError(VoiceAttributeError):
    code = "AUDIO_TOO_SHORT"
    http_status = status.HTTP_400_BAD_REQUEST


class DecodeError(VoiceAttributeError):
    code = "DECODE_FAILED"
    http_status = status.HTTP_415_UNSUPPORTED_MEDIA_TYPE


class DecodeTimeoutError(VoiceAttributeError):
    code = "DECODE_TIMEOUT"
    http_status = status.HTTP_504_GATEWAY_TIMEOUT


class InferenceTimeoutError(VoiceAttributeError):
    code = "INFERENCE_TIMEOUT"
    http_status = status.HTTP_504_GATEWAY_TIMEOUT


class OverloadedError(VoiceAttributeError):
    """All inference slots busy. 429 + Retry-After beats an unbounded queue:
    a caller that waits 8 s for an age guess has already lost the call."""

    code = "OVERLOADED"
    http_status = status.HTTP_429_TOO_MANY_REQUESTS


class ModelNotReadyError(VoiceAttributeError):
    code = "MODEL_NOT_READY"
    http_status = status.HTTP_503_SERVICE_UNAVAILABLE


def install_exception_handlers(app) -> None:
    import structlog

    log = structlog.get_logger(__name__)

    @app.exception_handler(VoiceAttributeError)
    async def _handle_known(request: Request, exc: VoiceAttributeError):
        request_id = getattr(request.state, "request_id", None)
        log.warning(
            "request_failed",
            error_code=exc.code,
            message=exc.message,
            detail=exc.detail,
            status=exc.http_status,
        )
        headers = {"Retry-After": "1"} if isinstance(exc, OverloadedError) else None
        return JSONResponse(
            status_code=exc.http_status,
            headers=headers,
            content=ErrorResponse(
                error=exc.code,
                message=exc.message,
                detail=exc.detail,
                request_id=request_id,
            ).model_dump(),
        )

    @app.exception_handler(Exception)
    async def _handle_unknown(request: Request, exc: Exception):
        request_id = getattr(request.state, "request_id", None)
        # exc_info, not str(exc), in the message field: an unexpected exception
        # could carry fragments of input in its repr, and input here is PII.
        log.exception("unhandled_error", error_type=type(exc).__name__)
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error="INTERNAL_ERROR",
                message="Internal server error.",
                request_id=request_id,
            ).model_dump(),
        )
