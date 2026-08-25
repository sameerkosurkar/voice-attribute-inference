"""POST /analyze -- the required REST contract.

Accepts either a multipart upload (field `audio`, or `file`) or a raw request
body. Both are in the assignment ("multipart audio upload or raw stream") and
both show up in practice: multipart from a batch/backfill job, raw bytes from a
telephony bridge that is already streaming and does not want to build a MIME
envelope.

Content-Type is used only to pick which reader to run, never to decide the
codec -- ffmpeg probes the actual container. Telephony gateways label audio
wrongly often enough that trusting the header would be a bug.
"""

from __future__ import annotations

from fastapi import APIRouter, File, Query, Request, UploadFile
from fastapi.responses import JSONResponse

from app.errors import AudioTooLargeError, EmptyAudioError
from app.observability import REQUEST_SECONDS, REQUESTS, StageTimer
from app.schemas import AnalyzeResponse, ErrorResponse

router = APIRouter(tags=["inference"])


@router.post(
    "/analyze",
    response_model=AnalyzeResponse,
    response_model_exclude_none=True,
    summary="Estimate gender and age bracket from an audio sample",
    responses={
        400: {"model": ErrorResponse, "description": "Empty or too-short audio"},
        413: {"model": ErrorResponse, "description": "Upload exceeds the size limit"},
        415: {"model": ErrorResponse, "description": "Undecodable audio"},
        429: {"model": ErrorResponse, "description": "All inference slots busy"},
        503: {"model": ErrorResponse, "description": "Model still loading"},
        504: {"model": ErrorResponse, "description": "Decode or inference deadline exceeded"},
    },
)
async def analyze(
    request: Request,
    audio: UploadFile | None = File(
        default=None, description="Audio file (wav, mp3, flac, ogg/opus, webm, m4a)"
    ),
    contact_id: str | None = Query(
        default=None,
        description="Optional caller-supplied correlation id. If omitted the "
                    "service generates a random uuid4. Never derived from the audio.",
    ),
    debug: bool = Query(
        default=False,
        description="Include quality_detail and per-stage timings in the response.",
    ),
):
    service = request.app.state.service
    settings = service.settings
    timer = StageTimer()
    request_id = getattr(request.state, "request_id", None)

    raw = await _read_body(request, audio, settings.max_upload_bytes)

    outcome = "error"
    try:
        result = await service.analyze_bytes(
            raw,
            timer=timer,
            contact_id=contact_id,
            request_id=request_id,
            debug=debug,
        )
        outcome = result.response.audio_quality
        if hasattr(outcome, "value"):
            outcome = outcome.value
        return JSONResponse(
            content=result.response.model_dump(mode="json", exclude_none=True)
        )
    finally:
        # Drop our reference to the caller's audio as early as possible; see
        # PRIVACY.md. The bytes object is immutable so it cannot be wiped, but
        # it can be made unreachable before the response is serialised.
        del raw
        REQUESTS.labels(endpoint="analyze", outcome=outcome).inc()
        REQUEST_SECONDS.labels(endpoint="analyze", outcome=outcome).observe(
            timer.total_ms / 1000.0
        )


async def _read_body(request: Request, audio: UploadFile | None, limit: int) -> bytes:
    if audio is not None:
        raw = await audio.read()
        await audio.close()
        if len(raw) > limit:
            raise AudioTooLargeError("Uploaded audio exceeds the size limit.")
        if not raw:
            raise EmptyAudioError("The uploaded file was empty.")
        return raw

    # Raw-body path. Read incrementally and abort the moment the limit is
    # crossed, rather than buffering an arbitrarily large body first -- a
    # 2 GB POST should cost us a few kilobytes, not 2 GB of RSS.
    chunks: list[bytes] = []
    total = 0
    async for chunk in request.stream():
        total += len(chunk)
        if total > limit:
            raise AudioTooLargeError("Request body exceeds the size limit.")
        chunks.append(chunk)

    raw = b"".join(chunks)
    if not raw:
        raise EmptyAudioError(
            "No audio supplied. Send multipart/form-data with an 'audio' field, "
            "or POST the encoded bytes as the raw request body."
        )
    return raw
