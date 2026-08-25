"""The backend seam.

Everything above this line (routing, quality gating, calibration, streaming)
is model-agnostic. Everything below it is one specific set of weights.

That seam is not architecture astronautics -- it is the mitigation for a
concrete, known problem. The default model is licensed CC-BY-NC-SA-4.0, which
forbids commercial use. Shipping it in a product means swapping the weights.
With this interface that is one new file and one env var; without it, it is a
rewrite of the request path. See README "Model licence".
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from app.schemas import LanguagePrediction
from app.inference.types import RawPrediction


@runtime_checkable
class AttributeBackend(Protocol):
    """Synchronous and CPU-bound by contract.

    Implementations are called from a thread pool, never from the event loop.
    `predict` must be safe to call concurrently from multiple threads.
    """

    name: str

    def load(self) -> None:
        """Materialise weights. Called once, during app startup."""

    def warmup(self) -> None:
        """Run a throwaway forward pass.

        Not optional. The first torch forward pass on a fresh process pays for
        lazy kernel selection and allocator growth -- measured at several
        hundred ms on this model. Without a warmup the first real caller eats
        that, which is precisely the caller you cannot afford to keep waiting.
        """

    def predict(self, samples: np.ndarray, sample_rate: int) -> RawPrediction:
        """Raw, uncalibrated age and gender for one mono float32 buffer."""

    @property
    def ready(self) -> bool:
        ...


@runtime_checkable
class LanguageBackend(Protocol):
    """Best-effort spoken-language identification.

    A separate Protocol from `AttributeBackend`, not a method on it, because the
    two have genuinely different lifecycles: language ID is optional, runs under
    its own deadline, and is allowed to fail or be absent without affecting the
    required part of the response. Folding it into the attribute interface would
    force every attribute backend to implement or stub something it has no
    opinion about.

    It previously had no interface at all -- `LanguageIdentifier` was referenced
    concretely from `service.py`, so swapping Whisper for a purpose-built LID
    model (SpeechBrain's VoxLingua107 ECAPA is the obvious candidate, and is
    more accurate for this job) meant editing the orchestration layer. Now it is
    a drop-in, the same as an attribute backend.

    Like `AttributeBackend`, implementations are called from a thread pool and
    `identify` must be safe to call concurrently.
    """

    name: str

    def load(self) -> None:
        ...

    def warmup(self) -> None:
        ...

    def identify(
        self, samples: np.ndarray, sample_rate: int
    ) -> LanguagePrediction | None:
        """Return None when below the confidence threshold, or when the model
        has no opinion. Never raise for ordinary "don't know" cases."""

    @property
    def ready(self) -> bool:
        ...
