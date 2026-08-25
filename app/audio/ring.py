"""Sliding audio window and cross-chunk prediction aggregation for streaming.

TWO PROBLEMS THE STREAMING PATH HAS THAT THE REST PATH DOES NOT.

1. Unbounded input. A call can run for minutes. Re-running inference over the
   whole call each second is quadratic work and blows the latency budget, so
   `SlidingWindow` keeps only the most recent `window_seconds` and drops the
   oldest samples. Bounded memory per session, bounded compute per emit.

2. Jitter. Independent per-window predictions flicker -- window N says male
   0.7, window N+1 says female 0.6 -- which is useless to a voice agent that
   has to pick a persona and stick to it. `PredictionAggregator` fixes this by
   accumulating *probabilities*, not labels, in an exponential moving average
   weighted by how much usable speech each window actually contained.

   Weighting by speech duration and quality is the important detail. A window
   that was mostly engine noise should barely move the estimate; a window with
   3 s of clean speech should move it a lot. Averaging raw labels, or an
   unweighted EMA, throws that information away and lets one noisy second undo
   ten good ones.

   The aggregator also reports `stable`: once consecutive estimates stop moving
   materially, the agent can commit to a persona early instead of waiting for
   the call to end. That is the entire point of streaming for this use case.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from app.inference.calibration import RawPrediction


class SlidingWindow:
    """Fixed-duration mono float32 ring buffer."""

    def __init__(self, sample_rate: int, window_seconds: float) -> None:
        self.sample_rate = sample_rate
        self.capacity = max(1, int(window_seconds * sample_rate))
        self._chunks: deque[np.ndarray] = deque()
        self._size = 0
        self.total_samples_seen = 0

    def append(self, samples: np.ndarray) -> None:
        if samples.size == 0:
            return
        block = np.asarray(samples, dtype=np.float32)
        self._chunks.append(block)
        self._size += block.size
        self.total_samples_seen += block.size
        # Drop-oldest, not reject-newest: for attribute inference the most
        # recent speech is the most useful, and a slow consumer must never
        # apply backpressure to a live call.
        while self._size > self.capacity and self._chunks:
            head = self._chunks[0]
            excess = self._size - self.capacity
            if head.size <= excess:
                self._chunks.popleft()
                self._size -= head.size
            else:
                self._chunks[0] = head[excess:]
                self._size -= excess

    def snapshot(self) -> np.ndarray:
        if not self._chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(list(self._chunks))

    def clear(self) -> None:
        for chunk in self._chunks:
            try:
                chunk.fill(0.0)   # wipe before release; see PRIVACY.md
            except (ValueError, AttributeError):
                pass
        self._chunks.clear()
        self._size = 0

    @property
    def seconds(self) -> float:
        return self._size / float(self.sample_rate)

    @property
    def total_seconds(self) -> float:
        return self.total_samples_seen / float(self.sample_rate)

    def __len__(self) -> int:
        return self._size


@dataclass(slots=True)
class PredictionAggregator:
    """Speech-weighted EMA over per-window raw predictions."""

    alpha: float = 0.55
    age_years: float = 0.0
    p_child: float = 0.0
    p_female: float = 0.0
    p_male: float = 0.0
    updates: int = 0
    total_weight: float = 0.0
    _history: list[tuple[float, float]] = field(default_factory=list)

    def update(self, raw: RawPrediction, *, speech_seconds: float,
               quality_factor: float) -> None:
        # Two seconds of clean speech is the reference unit of evidence; a
        # window with less, or with a poor quality factor, gets proportionally
        # less say. Capped at 1.0 so one long window cannot dominate.
        weight = float(np.clip(speech_seconds / 2.0, 0.0, 1.0)) * float(quality_factor)
        if weight <= 0.0:
            return

        effective = self.alpha * weight
        if self.updates == 0:
            self.age_years = raw.age_years
            self.p_child, self.p_female, self.p_male = (
                raw.p_child, raw.p_female, raw.p_male,
            )
        else:
            self.age_years += effective * (raw.age_years - self.age_years)
            self.p_child += effective * (raw.p_child - self.p_child)
            self.p_female += effective * (raw.p_female - self.p_female)
            self.p_male += effective * (raw.p_male - self.p_male)

        self.updates += 1
        self.total_weight += weight
        self._history.append((self.age_years, self.p_male - self.p_female))
        if len(self._history) > 5:
            self._history.pop(0)

    def snapshot(self) -> RawPrediction:
        return RawPrediction(
            age_years=self.age_years,
            p_child=self.p_child,
            p_female=self.p_female,
            p_male=self.p_male,
        )

    @property
    def confidence_scale(self) -> float:
        """Evidence accumulated so far, as a multiplier in [0.55, 1.0].

        A prediction from a single 1.5 s window genuinely deserves less
        confidence than the same prediction after 10 s of consistent speech.
        Without this, the first partial would come out just as confident as the
        final -- which would defeat the purpose of streaming progressively.
        """
        return float(np.clip(0.55 + 0.15 * self.total_weight, 0.55, 1.0))

    @property
    def stable(self) -> bool:
        """True once the last three estimates agree closely.

        Thresholds: age within 2 years and the male-minus-female margin within
        0.1 -- both comfortably below the resolution the API actually reports,
        so "stable" means "further audio will not change the answer".
        """
        if len(self._history) < 3:
            return False
        ages = [h[0] for h in self._history[-3:]]
        margins = [h[1] for h in self._history[-3:]]
        return (max(ages) - min(ages) < 2.0) and (max(margins) - min(margins) < 0.1)
