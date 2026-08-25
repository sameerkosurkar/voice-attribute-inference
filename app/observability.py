"""Prometheus metrics and a stopwatch.

The histogram buckets are deliberately dense between 50 ms and 500 ms: the SLO
is "under 500 ms on a 5 s chunk", so that is the region where a p95 needs real
resolution. Buckets copied from a default HTTP histogram would put the entire
interesting range into two buckets.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

REGISTRY = CollectorRegistry(auto_describe=True)

_LATENCY_BUCKETS = (
    0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5,
    0.75, 1.0, 2.0, 5.0, float("inf"),
)

STAGE_SECONDS = Histogram(
    "va_stage_duration_seconds",
    "Wall-clock duration of one pipeline stage.",
    labelnames=("stage",),
    buckets=_LATENCY_BUCKETS,
    registry=REGISTRY,
)

REQUEST_SECONDS = Histogram(
    "va_request_duration_seconds",
    "End-to-end duration of an analyze request.",
    labelnames=("endpoint", "outcome"),
    buckets=_LATENCY_BUCKETS,
    registry=REGISTRY,
)

REQUESTS = Counter(
    "va_requests_total",
    "Analyze requests by endpoint and outcome.",
    labelnames=("endpoint", "outcome"),
    registry=REGISTRY,
)

QUALITY = Counter(
    "va_audio_quality_total",
    "Audio-quality verdicts. A rising 'insufficient' share is the leading "
    "indicator that upstream telephony or codec settings have regressed.",
    labelnames=("quality",),
    registry=REGISTRY,
)

PREDICTIONS = Counter(
    "va_predictions_total",
    "Emitted predictions by attribute and label, including 'unknown'.",
    labelnames=("attribute", "label"),
    registry=REGISTRY,
)

ERRORS = Counter(
    "va_errors_total",
    "Failed requests by typed error code.",
    labelnames=("code",),
    registry=REGISTRY,
)

INFLIGHT = Gauge(
    "va_inflight_inferences",
    "Inference slots currently occupied.",
    registry=REGISTRY,
)

WS_SESSIONS = Gauge(
    "va_websocket_sessions",
    "Open streaming sessions.",
    registry=REGISTRY,
)

MODEL_READY = Gauge(
    "va_model_ready",
    "1 once weights are loaded and a warmup pass has completed.",
    registry=REGISTRY,
)


@dataclass
class StageTimer:
    """Accumulates per-stage milliseconds for one request.

    Recorded into Prometheus and echoed back in the optional `timings` field,
    so a slow call can be attributed to decode vs. model without a profiler.
    """

    started: float = field(default_factory=time.perf_counter)
    stages: dict[str, float] = field(default_factory=dict)

    @contextmanager
    def stage(self, name: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self.stages[name] = self.stages.get(name, 0.0) + elapsed * 1000.0
            STAGE_SECONDS.labels(stage=name).observe(elapsed)

    def record(self, name: str, milliseconds: float) -> None:
        self.stages[name] = self.stages.get(name, 0.0) + milliseconds
        STAGE_SECONDS.labels(stage=name).observe(milliseconds / 1000.0)

    def ms(self, name: str) -> float:
        return round(self.stages.get(name, 0.0), 2)

    @property
    def total_ms(self) -> float:
        return round((time.perf_counter() - self.started) * 1000.0, 2)
