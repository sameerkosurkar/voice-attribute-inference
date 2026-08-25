"""WS /ws/analyze -- progressive predictions over a live call (bonus task).

PROTOCOL
    client -> {"type":"start","format":"pcm_s16le","sample_rate":16000,
               "contact_id":"..."}          (format defaults to pcm_s16le)
    server -> {"type":"ready", ...}
    client -> <binary audio frames>          repeated
    server -> {"type":"partial", ...}        roughly every ws_emit_interval_ms
    client -> {"type":"end"}                 (or just close)
    server -> {"type":"final", "is_final":true, ...}

Two input paths, because the two realistic clients differ:

  * `pcm_s16le` -- raw 16-bit PCM. No decoder, no subprocess, sub-millisecond
    conversion. This is what a telephony bridge already has in hand and it is
    the path that actually matters for real-time.
  * anything else (webm/opus from a browser, mp3, ...) -- a long-lived ffmpeg
    subprocess per connection, fed incrementally over stdin. One process for
    the whole session, not one per chunk, so we pay process setup once.

DESIGN NOTES

  * Emits are time-triggered, not chunk-triggered. A client sending 20 ms
    frames must not cause 50 forward passes a second; a client sending 2 s
    frames must still get an answer promptly.

  * `_emitting` guard: if inference is still running when the next emit is due,
    we skip that emit rather than queue it. Under load the correct behaviour is
    fewer, current predictions -- not a growing backlog of stale ones.

  * The window is a bounded ring, so a 40-minute call uses the same memory as a
    40-second one.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import shutil
import time
import uuid

import numpy as np
import structlog
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.audio.decode import pcm16_to_float32
from app.audio.ring import PredictionAggregator, SlidingWindow
from app.audio import quality as quality_mod
from app.inference.calibration import calibrate, unknown_prediction
from app.observability import WS_SESSIONS, StageTimer
from app.schemas import (
    AudioQuality,
    QualityDetail,
    StreamEventType,
    StreamPrediction,
)

log = structlog.get_logger(__name__)
router = APIRouter(tags=["streaming"])

_RAW_FORMATS = {"pcm_s16le", "pcm", "raw", "s16le"}


@router.websocket("/ws/analyze")
async def analyze_stream(websocket: WebSocket) -> None:
    await websocket.accept()
    service = websocket.app.state.service
    settings = service.settings
    session_id = str(uuid.uuid4())

    if not service.ready:
        await websocket.send_json(
            {"type": "error", "error": "MODEL_NOT_READY", "message": "Model is still loading."}
        )
        await websocket.close(code=1013)
        return

    WS_SESSIONS.inc()
    session = _Session(websocket, service, settings, session_id)
    try:
        await session.run()
    except WebSocketDisconnect:
        log.info("ws_client_disconnected", session_id=session_id,
                 seconds=round(session.window.total_seconds, 2))
    except Exception:
        log.exception("ws_session_failed", session_id=session_id)
        with contextlib.suppress(Exception):
            await websocket.send_json(
                {"type": "error", "error": "INTERNAL_ERROR",
                 "message": "Streaming session failed."}
            )
    finally:
        WS_SESSIONS.dec()
        await session.close()


class _Session:
    def __init__(self, websocket: WebSocket, service, settings, session_id: str) -> None:
        self.ws = websocket
        self.service = service
        self.settings = settings
        self.session_id = session_id
        self.contact_id = str(uuid.uuid4())

        self.window = SlidingWindow(settings.target_sample_rate, settings.ws_window_seconds)
        self.aggregator = PredictionAggregator(alpha=settings.ws_ema_alpha)
        self.decoder: asyncio.subprocess.Process | None = None
        self._pump: asyncio.Task | None = None
        self._chunks = 0
        self._bytes = 0
        self._last_emit = 0.0
        self._emitting = False
        self._started = time.monotonic()
        self._last_quality = AudioQuality.INSUFFICIENT
        self._last_reasons: list[str] = []

    # ------------------------------------------------------------------ run
    async def run(self) -> None:
        fmt, sample_rate = await self._handshake()
        if fmt not in _RAW_FORMATS:
            await self._start_decoder(fmt)

        while True:
            if time.monotonic() - self._started > self.settings.ws_max_session_seconds:
                log.info("ws_session_expired", session_id=self.session_id)
                break

            message = await self.ws.receive()
            if message["type"] == "websocket.disconnect":
                break

            if (data := message.get("bytes")) is not None:
                await self._ingest(data, sample_rate, fmt)
                if self._bytes > self.settings.ws_max_bytes:
                    log.warning("ws_byte_cap_reached", session_id=self.session_id)
                    break
            elif (text := message.get("text")) is not None:
                if await self._handle_text(text):
                    break

            await self._maybe_emit(final=False)

        # Drain the decoder before the final verdict. Without this, a client
        # that sends its audio and immediately says "end" -- which any client
        # replaying a buffered recording does -- races ffmpeg and gets an empty
        # window back, i.e. a confident "insufficient" for perfectly good audio.
        await self._flush_decoder()
        await self._emit(final=True)

    async def _handshake(self) -> tuple[str, int]:
        """Read an optional `start` frame. Defaults let a client just open the
        socket and start sending 16 kHz PCM with no preamble."""
        fmt, sample_rate = "pcm_s16le", self.settings.target_sample_rate
        try:
            message = await asyncio.wait_for(self.ws.receive(), timeout=10.0)
        except asyncio.TimeoutError:
            message = {}

        if (text := message.get("text")) is not None:
            with contextlib.suppress(json.JSONDecodeError, TypeError):
                payload = json.loads(text)
                fmt = str(payload.get("format", fmt)).lower()
                sample_rate = int(payload.get("sample_rate", sample_rate))
                if payload.get("contact_id"):
                    self.contact_id = str(payload["contact_id"])
        elif (data := message.get("bytes")) is not None:
            # Client skipped the handshake and sent audio; keep it.
            await self._ingest(data, sample_rate, fmt)

        await self.ws.send_json(
            {
                "type": StreamEventType.READY.value,
                "contact_id": self.contact_id,
                "session_id": self.session_id,
                "accepted_format": fmt,
                "sample_rate": sample_rate,
                "emit_interval_ms": self.settings.ws_emit_interval_ms,
            }
        )
        self._last_emit = time.monotonic()
        return fmt, sample_rate

    async def _handle_text(self, text: str) -> bool:
        """Returns True when the client has signalled end-of-stream."""
        with contextlib.suppress(json.JSONDecodeError, TypeError):
            payload = json.loads(text)
            if str(payload.get("type", "")).lower() in {"end", "stop", "eof", "close"}:
                return True
        return False

    # --------------------------------------------------------------- ingest
    async def _ingest(self, data: bytes, sample_rate: int, fmt: str) -> None:
        self._chunks += 1
        self._bytes += len(data)

        if fmt in _RAW_FORMATS:
            samples = pcm16_to_float32(data)
            if sample_rate != self.settings.target_sample_rate and samples.size:
                samples = _resample_linear(
                    samples, sample_rate, self.settings.target_sample_rate
                )
            self.window.append(samples)
        elif self.decoder is not None and self.decoder.stdin is not None:
            try:
                self.decoder.stdin.write(data)
                await self.decoder.stdin.drain()
            except (BrokenPipeError, ConnectionResetError):
                log.warning("ws_decoder_pipe_broken", session_id=self.session_id)
                self.decoder = None

    async def _start_decoder(self, fmt: str) -> None:
        """One long-lived ffmpeg per connection, streaming in and out."""
        ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
        self.decoder = await asyncio.create_subprocess_exec(
            ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin",
            "-i", "pipe:0",
            "-map", "0:a:0", "-ac", "1",
            "-ar", str(self.settings.target_sample_rate),
            "-f", "f32le", "-acodec", "pcm_f32le", "pipe:1",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        self._pump = asyncio.create_task(self._drain_decoder())
        log.info("ws_decoder_started", session_id=self.session_id, format=fmt)

    async def _drain_decoder(self) -> None:
        assert self.decoder is not None and self.decoder.stdout is not None
        # 4 bytes/sample; read ~0.25 s at a time.
        block = self.settings.target_sample_rate  # bytes, not samples
        try:
            while True:
                data = await self.decoder.stdout.read(block)
                if not data:
                    break
                self.window.append(np.frombuffer(data, dtype="<f4").astype(np.float32))
        except asyncio.CancelledError:
            raise
        except Exception:
            log.warning("ws_decoder_drain_failed", session_id=self.session_id, exc_info=True)

    async def _flush_decoder(self, timeout: float = 5.0) -> None:
        """Close ffmpeg's stdin and wait for the last decoded samples.

        Closing stdin is what tells ffmpeg the stream has ended; it then flushes
        its remaining output and closes stdout, which ends the drain task. We
        wait for that task rather than for the process, because it is the drain
        task that actually moves samples into the window.
        """
        if self.decoder is None:
            return
        with contextlib.suppress(Exception):
            if self.decoder.stdin is not None and not self.decoder.stdin.is_closing():
                self.decoder.stdin.close()

        if self._pump is not None and not self._pump.done():
            # shield: on timeout we stop waiting, but cancelling mid-read could
            # drop samples we already paid to decode.
            with contextlib.suppress(asyncio.TimeoutError, Exception):
                await asyncio.wait_for(asyncio.shield(self._pump), timeout=timeout)

    # ----------------------------------------------------------------- emit
    async def _maybe_emit(self, *, final: bool) -> None:
        due = (time.monotonic() - self._last_emit) * 1000.0 >= self.settings.ws_emit_interval_ms
        if not due or self._emitting:
            return
        await self._emit(final=final)

    async def _emit(self, *, final: bool) -> None:
        if self._emitting:
            return
        self._emitting = True
        timer = StageTimer()
        try:
            samples = self.window.snapshot()
            if samples.size == 0:
                if final:
                    await self._send(unknown_prediction("no audio received"),
                                     None, timer, final=True)
                return

            loop = asyncio.get_running_loop()
            with timer.stage("quality"):
                report = await loop.run_in_executor(
                    self.service._executor, quality_mod.assess, samples,
                    self.settings.target_sample_rate, self.settings,
                )
            self._last_quality = report.quality
            self._last_reasons = report.reasons

            enough = report.speech_seconds >= self.settings.ws_min_speech_seconds
            if not enough or not report.usable:
                # Not yet worth a forward pass. On a `final` we still answer --
                # with unknown if we never accumulated enough, or with whatever
                # the EMA holds if earlier windows were good.
                if not final:
                    return
                if self.aggregator.updates == 0:
                    await self._send(
                        unknown_prediction(
                            report.reasons[0] if report.reasons
                            else "insufficient speech in the stream"
                        ),
                        report, timer, final=True,
                    )
                    return
            else:
                with timer.stage("inference"):
                    raw = await loop.run_in_executor(
                        self.service._executor,
                        self.service.backend.predict,
                        samples,
                        self.settings.target_sample_rate,
                    )
                self.aggregator.update(
                    raw,
                    speech_seconds=report.speech_seconds,
                    quality_factor=report.confidence_factor,
                )

            if self.aggregator.updates == 0:
                return

            prediction = calibrate(
                self.aggregator.snapshot(),
                self.settings,
                # Confidence rises with accumulated evidence, so an early
                # partial is visibly less certain than the final.
                confidence_factor=report.confidence_factor * self.aggregator.confidence_scale,
            )
            await self._send(prediction, report, timer, final=final)
        finally:
            self._emitting = False
            self._last_emit = time.monotonic()

    async def _send(self, prediction, report, timer: StageTimer, *, final: bool) -> None:
        quality = report.quality if report is not None else self._last_quality
        event = StreamPrediction(
            type=StreamEventType.FINAL if final else StreamEventType.PARTIAL,
            is_final=final,
            contact_id=self.contact_id,
            gender=prediction.gender,
            age_bracket=prediction.age_bracket,
            processing_ms=int(round(timer.total_ms)),
            audio_quality=quality,
            chunks_seen=self._chunks,
            audio_seconds=round(self.window.total_seconds, 2),
            speech_seconds=round(report.speech_seconds, 2) if report else 0.0,
            stable=self.aggregator.stable,
            request_id=self.session_id,
            quality_detail=(
                QualityDetail(
                    speech_seconds=report.speech_seconds,
                    total_seconds=report.total_seconds,
                    snr_db=report.snr_db,
                    clipping_ratio=report.clipping_ratio,
                    high_band_ratio=report.high_band_ratio,
                    reasons=report.reasons + prediction.notes,
                )
                if report is not None
                else None
            ),
        )
        with contextlib.suppress(RuntimeError, WebSocketDisconnect):
            await self.ws.send_json(event.model_dump(mode="json", exclude_none=True))

    # ---------------------------------------------------------------- close
    async def close(self) -> None:
        if self._pump is not None:
            self._pump.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._pump
        if self.decoder is not None:
            with contextlib.suppress(Exception):
                if self.decoder.stdin is not None:
                    self.decoder.stdin.close()
                self.decoder.kill()
                await self.decoder.wait()
        # Wipe the ring before the session object is collected.
        self.window.clear()
        with contextlib.suppress(Exception):
            await self.ws.close()


def _resample_linear(samples: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Linear resample for the raw-PCM path only.

    Deliberately cheap and deliberately not used for file uploads, which go
    through ffmpeg's soxr. Linear interpolation aliases, and aliasing shifts
    the high-frequency content that a gender model reads -- so the honest
    guidance, documented in the README, is to send 16 kHz on the raw path.
    """
    if src_rate == dst_rate or samples.size == 0:
        return samples
    duration = samples.size / float(src_rate)
    target_n = int(duration * dst_rate)
    if target_n <= 0:
        return np.zeros(0, dtype=np.float32)
    src_idx = np.linspace(0, samples.size - 1, target_n, dtype=np.float64)
    return np.interp(src_idx, np.arange(samples.size), samples).astype(np.float32)
