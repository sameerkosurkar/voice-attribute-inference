"""A deterministic stand-in for the real weights.

Exists so the API-contract tests can run in ~1 second in CI without pulling a
gigabyte of weights, and so the service can boot in a constrained environment.
It derives its "prediction" from cheap acoustic statistics (mean f0 proxy via
zero-crossing rate) -- not accurate, but not random either, which keeps the
contract tests meaningful rather than tautological.
"""

from __future__ import annotations

import numpy as np

from app.inference.registry import register_backend
from app.inference.types import RawPrediction


@register_backend("mock", description="Deterministic stand-in for fast tests")
class MockBackend:
    name = "mock"

    # Accepts settings for a uniform factory signature, ignores them.
    def __init__(self, settings=None) -> None:
        self._settings = settings
        self._ready = False

    def load(self) -> None:
        self._ready = True

    def warmup(self) -> None:
        self.predict(np.zeros(16_000, dtype=np.float32), 16_000)

    @property
    def ready(self) -> bool:
        return self._ready

    def predict(self, samples: np.ndarray, sample_rate: int) -> RawPrediction:
        if samples.size == 0:
            return RawPrediction(age_years=35.0, p_child=0.0, p_female=0.5, p_male=0.5)

        # Zero-crossing rate is a crude proxy for pitch; higher ZCR skews the
        # mock towards "female". Deterministic and dependency-free.
        zcr = float(np.mean(np.abs(np.diff(np.sign(samples))) > 0)) if samples.size > 1 else 0.0
        centred = np.clip((zcr - 0.06) * 12.0, -1.0, 1.0)
        p_female = float(0.5 + 0.45 * centred)
        p_male = 1.0 - p_female - 0.02

        rms = float(np.sqrt(np.mean(samples**2)))
        age = float(np.clip(24.0 + rms * 60.0, 18.0, 85.0))
        return RawPrediction(
            age_years=age,
            p_child=0.02,
            p_female=max(p_female, 0.0),
            p_male=max(p_male, 0.0),
        )
