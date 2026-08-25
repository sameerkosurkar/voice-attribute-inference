"""Pipeline orchestration and process-wide model lifecycle.

decode -> quality gate -> (inference || language id) -> calibrate -> respond

Concurrency model, stated explicitly because it is the thing that decides
whether the latency target survives contact with load:

  * The event loop never runs torch. Every forward pass goes to a small,
    fixed-size ThreadPoolExecutor. torch releases the GIL inside its kernels,
    so threads (not processes) are the right tool and we avoid paying to
    serialise audio across a process boundary.

  * Thread pool size and torch intra-op threads are BOTH pinned and small
    (2 x 2 by default). The instinct to set both high is wrong: oversubscribing
    cores makes every request slower under concurrency. Real horizontal scale
    comes from more replicas, not more threads. See README "Scaling".

  * A semaphore caps in-flight inferences and we return 429 past it. An
    unbounded queue would keep accepting work and quietly turn a latency
    problem into a timeout problem for every caller at once.
"""

from __future__ import annotations

import asyncio
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import structlog

from app.audio import quality as quality_mod
from app.audio.decode import DecodedAudio, decode
from app.config import Settings
from app.errors import InferenceTimeoutError, ModelNotReadyError, OverloadedError
from app.inference import language_registry, registry
from app.inference.base import AttributeBackend
from app.inference.calibration import CalibratedPrediction, calibrate, unknown_prediction
from app.observability import (
    INFLIGHT,
    MODEL_READY,
    PREDICTIONS,
    QUALITY,
    StageTimer,
)
from app.schemas import (
    AnalyzeResponse,
    AudioQuality,
    LanguagePrediction,
    QualityDetail,
    Timings,
)

log = structlog.get_logger(__name__)


@dataclass(slots=True)
class AnalysisResult:
    response: AnalyzeResponse
    prediction: CalibratedPrediction
    report: quality_mod.QualityReport


class AnalyzerService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.backend: AttributeBackend | None = None
        self.language: object | None = None
        self._executor: ThreadPoolExecutor | None = None
        self._semaphore: asyncio.Semaphore | None = None
        self._ready = False

    # ----------------------------------------------------------- lifecycle --
    async def startup(self) -> None:
        self._executor = ThreadPoolExecutor(
            max_workers=self.settings.inference_threads,
            thread_name_prefix="infer",
        )
        self._semaphore = asyncio.Semaphore(self.settings.max_concurrent_inferences)

        self.backend = build_backend(self.settings)

        loop = asyncio.get_running_loop()
        # Load and warm off the loop: model load is seconds of blocking I/O and
        # compute, and holding the loop would make /health hang during rollout.
        await loop.run_in_executor(self._executor, self._blocking_startup)
        self._ready = True
        MODEL_READY.set(1)
        log.info("service_ready", backend=self.backend.name,
                 language_id=self.settings.enable_language_id)

    def _blocking_startup(self) -> None:
        assert self.backend is not None
        self.backend.load()
        self.backend.warmup()
        try:
            quality_mod.warmup_vad()
        except Exception:
            log.warning("vad_warmup_failed", exc_info=True)

        if self.settings.enable_language_id:
            try:
                identifier = language_registry.create(
                    self.settings.language_backend, self.settings
                )
                identifier.load()
                identifier.warmup()
                self.language = identifier
                log.info("language_backend_ready",
                         backend=self.settings.language_backend)
            except Exception:
                # A bonus field must never stop the service from starting.
                log.warning("language_id_unavailable",
                            backend=self.settings.language_backend, exc_info=True)
                self.language = None

    async def shutdown(self) -> None:
        MODEL_READY.set(0)
        self._ready = False
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=False)
            self._executor = None

    @property
    def ready(self) -> bool:
        return self._ready and self.backend is not None and self.backend.ready

    # -------------------------------------------------------------- analyze --
    async def analyze_bytes(
        self,
        raw: bytes,
        *,
        timer: StageTimer,
        contact_id: str | None = None,
        request_id: str | None = None,
        debug: bool = False,
    ) -> AnalysisResult:
        audio: DecodedAudio | None = None
        try:
            with timer.stage("decode"):
                audio = await decode(raw, self.settings)
            return await self.analyze_samples(
                audio.samples,
                timer=timer,
                contact_id=contact_id,
                request_id=request_id,
                debug=debug,
            )
        finally:
            if audio is not None:
                audio.wipe()

    async def analyze_samples(
        self,
        samples: np.ndarray,
        *,
        timer: StageTimer,
        contact_id: str | None = None,
        request_id: str | None = None,
        debug: bool = False,
    ) -> AnalysisResult:
        if not self.ready:
            raise ModelNotReadyError("Model is still loading.")

        # A fresh uuid4 per call, never derived from the audio. A hash of the
        # waveform would be a stable biometric identifier and would let two
        # calls from the same person be linked -- see PRIVACY.md.
        contact_id = contact_id or str(uuid.uuid4())

        loop = asyncio.get_running_loop()
        with timer.stage("quality"):
            report = await loop.run_in_executor(
                self._executor, quality_mod.assess, samples, self.settings.target_sample_rate,
                self.settings,
            )
        QUALITY.labels(quality=report.quality.value).inc()

        if not report.usable:
            # Short-circuit. Deliberately a 200, not an error: on a real call
            # this is a normal outcome and the agent just uses a neutral
            # persona. Also saves the forward pass, which is the whole latency
            # budget spent on audio we already know is unusable.
            prediction = unknown_prediction(
                report.reasons[0] if report.reasons else "insufficient audio"
            )
            return self._assemble(
                prediction, report, timer, contact_id, request_id, None, debug
            )

        prediction, language = await self._infer(samples, report, timer)
        return self._assemble(
            prediction, report, timer, contact_id, request_id, language, debug
        )

    async def _infer(
        self,
        samples: np.ndarray,
        report: quality_mod.QualityReport,
        timer: StageTimer,
    ) -> tuple[CalibratedPrediction, LanguagePrediction | None]:
        assert self._semaphore is not None and self.backend is not None
        loop = asyncio.get_running_loop()

        # Bounded queue, not "reject on any contention" and not "queue forever".
        # Waiting briefly absorbs bursts that would otherwise be shed even
        # though they could have completed in budget; the timeout guarantees we
        # never trade a latency problem for a timeout problem.
        wait_s = max(self.settings.max_queue_wait_ms, 0.0) / 1000.0
        if wait_s <= 0.0:
            # asyncio.wait_for(..., timeout=0) cancels before the coroutine gets
            # a single scheduling step, so it would reject even a completely
            # idle service. Check-then-acquire instead. There is no race: the
            # loop is single-threaded and Semaphore.acquire() returns without
            # awaiting when the semaphore is free.
            if self._semaphore.locked():
                raise OverloadedError("All inference slots are busy; retry shortly.")
            await self._semaphore.acquire()
        else:
            try:
                await asyncio.wait_for(self._semaphore.acquire(), timeout=wait_s)
            except asyncio.TimeoutError as exc:
                raise OverloadedError(
                    "All inference slots are busy; retry shortly."
                ) from exc

        try:
            INFLIGHT.inc()
            try:
                attribute_task = loop.run_in_executor(
                    self._executor,
                    self.backend.predict,
                    samples,
                    self.settings.target_sample_rate,
                )
                # Kick language ID off in parallel. It has its own deadline and
                # its own failure handling: it can never delay or fail the
                # required part of the response.
                language_task = None
                if self.settings.enable_language_id and self.language is not None:
                    language_task = loop.run_in_executor(
                        self._executor,
                        self.language.identify,  # type: ignore[union-attr]
                        samples,
                        self.settings.target_sample_rate,
                    )

                try:
                    with timer.stage("inference"):
                        raw = await asyncio.wait_for(
                            attribute_task, timeout=self.settings.inference_timeout_s
                        )
                except asyncio.TimeoutError as exc:
                    raise InferenceTimeoutError("Inference exceeded its deadline.") from exc

                language = await self._collect_language(language_task, timer)
            finally:
                INFLIGHT.dec()
        finally:
            self._semaphore.release()

        prediction = calibrate(
            raw, self.settings, confidence_factor=report.confidence_factor
        )
        return prediction, language

    async def _collect_language(self, task, timer: StageTimer) -> LanguagePrediction | None:
        if task is None:
            return None
        budget_s = max(self.settings.language_budget_ms, 0.0) / 1000.0
        try:
            with timer.stage("language"):
                # shield: on timeout we abandon the *result*, but we must not
                # cancel the thread mid-forward-pass and leave the pool wedged.
                return await asyncio.wait_for(asyncio.shield(task), timeout=budget_s)
        except asyncio.TimeoutError:
            log.info("language_id_over_budget", budget_ms=self.settings.language_budget_ms)
            return None
        except Exception:
            log.warning("language_id_failed", exc_info=True)
            return None

    # ------------------------------------------------------------- assemble --
    def _assemble(
        self,
        prediction: CalibratedPrediction,
        report: quality_mod.QualityReport,
        timer: StageTimer,
        contact_id: str,
        request_id: str | None,
        language: LanguagePrediction | None,
        debug: bool,
    ) -> AnalysisResult:
        PREDICTIONS.labels(attribute="gender", label=prediction.gender.prediction.value).inc()
        PREDICTIONS.labels(
            attribute="age_bracket", label=prediction.age_bracket.prediction.value
        ).inc()

        total_ms = timer.total_ms
        response = AnalyzeResponse(
            contact_id=contact_id,
            gender=prediction.gender,
            age_bracket=prediction.age_bracket,
            processing_ms=int(round(total_ms)),
            audio_quality=report.quality,
            language=language,
            request_id=request_id,
            quality_detail=(
                QualityDetail(
                    speech_seconds=report.speech_seconds,
                    total_seconds=report.total_seconds,
                    snr_db=report.snr_db,
                    clipping_ratio=report.clipping_ratio,
                    high_band_ratio=report.high_band_ratio,
                    reasons=report.reasons + prediction.notes,
                )
                if debug
                else None
            ),
            timings=(
                Timings(
                    decode_ms=timer.ms("decode"),
                    quality_ms=timer.ms("quality"),
                    inference_ms=timer.ms("inference"),
                    language_ms=timer.ms("language") or None,
                    total_ms=round(total_ms, 2),
                )
                if debug
                else None
            ),
        )
        return AnalysisResult(response=response, prediction=prediction, report=report)


def build_backend(settings: Settings) -> AttributeBackend:
    """Resolve `VA_BACKEND` to a concrete backend via the registry.

    This function deliberately knows the name of NO specific backend. Each one
    advertises its own availability and its own `auto` priority through
    `@register_backend`, so adding a model -- including one that should win on
    some platform -- requires no edit here. See app/inference/registry.py.

    On `auto` the registry picks by platform, because the measurements invert:
    PyTorch is 2.3x faster than ONNX Runtime on macOS (where torch links Apple
    Accelerate) and 2.1x slower in the Linux container (where it does not).
    "ONNX is faster" is a fact about which BLAS your PyTorch linked, not about
    ONNX, so the choice is measured per target rather than assumed.
    """
    choice = (settings.backend or "auto").strip().lower()

    if choice == "auto":
        try:
            choice = registry.select_auto(settings)
        except KeyError as exc:
            raise ModelNotReadyError(str(exc)) from exc
        log.info("backend_auto_selected", backend=choice, platform=sys.platform)

    spec = registry.get_spec(choice)
    if spec is None:
        raise ModelNotReadyError(
            f"unknown VA_BACKEND={choice!r}. Registered backends: "
            f"{', '.join(registry.available_backends())}"
        )
    if spec.is_available is not None and not spec.is_available(settings):
        raise ModelNotReadyError(
            f"backend {choice!r} is registered but not usable here "
            f"(missing artefacts or unsupported platform)."
        )

    return registry.create(choice, settings)


def quality_of(value: str) -> AudioQuality:
    return AudioQuality(value)
