"""Runtime configuration.

Every tunable is an environment variable so the same image can be re-tuned per
environment without a rebuild. Defaults are chosen for the assignment's target:
end-to-end inference under 500 ms on a 5-second chunk, on CPU.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="VA_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=(),
    )

    # ---------------------------------------------------------------- service
    service_name: str = "voice-attributes"
    log_level: str = "INFO"
    log_json: bool = True
    host: str = "0.0.0.0"
    port: int = 8000

    # ----------------------------------------------------------------- models
    # The default is the 6-transformer-layer variant. The 24-layer variant is
    # more accurate but roughly 3x the compute; switch with VA_AGE_GENDER_MODEL.
    age_gender_model: str = "audeering/wav2vec2-large-robust-6-ft-age-gender"
    language_model: str = "openai/whisper-tiny"
    # Any name registered via @register_backend, or "auto" to let the registry
    # pick. Deliberately an open string, not a Literal: a closed enum would mean
    # every new model requires editing this file, which is exactly the friction
    # the registry exists to remove. Unknown names fail at startup with the list
    # of what IS registered -- see app/inference/registry.py.
    backend: str = "auto"
    model_cache_dir: str | None = None

    # Bounded thread pool for the (GIL-releasing) torch forward passes, plus a
    # semaphore so we shed load instead of queueing unboundedly and blowing the
    # latency SLO for everyone.
    inference_threads: int = 2
    torch_threads: int = 2
    # 0 => derive from inference_threads (see the validator below). Set an
    # explicit value only if you have measured your own target.
    max_concurrent_inferences: int = 0
    inference_timeout_s: float = 5.0
    # How long a request may wait for a free inference slot before we shed it
    # with 429. Zero would reject on the slightest overlap, which throws away
    # requests that could still have been served inside the latency budget; an
    # unbounded queue would turn a load problem into a timeout problem for
    # everyone. 150 ms is a bounded queue: it absorbs bursts while leaving most
    # of the 500 ms budget for the work itself.
    max_queue_wait_ms: float = 150.0

    # ------------------------------------------------------------------ audio
    target_sample_rate: int = 16_000
    max_upload_bytes: int = 25 * 1024 * 1024
    ffmpeg_timeout_s: float = 10.0
    # Longer uploads are windowed down to the most energetic segment rather than
    # rejected -- a 60 s voicemail should still get an answer.
    max_analysis_seconds: float = 10.0
    min_audio_seconds: float = 0.5

    # ---------------------------------------------------------- quality gate
    min_speech_seconds: float = 1.0       # below this -> "insufficient"
    good_speech_seconds: float = 2.0      # below this -> at best "degraded"
    vad_threshold: float = 0.5
    good_snr_db: float = 15.0             # >= this and clean -> "good"
    degraded_snr_db: float = 3.0          # below this -> "insufficient"
    max_clipping_ratio: float = 0.01      # >1% clipped samples -> "degraded"
    # Fraction of spectral energy above 4 kHz. Real wideband speech measures
    # 0.05-0.75%; G.711 narrowband upsampled to 16 kHz measures ~0.005%.
    # 0.02% sits an order of magnitude clear of both.
    min_high_band_ratio: float = 2e-4

    # ------------------------------------------------------------ calibration
    # Std-dev, in years, of the age regressor's error. The source paper reports
    # MAE 7.1-10.8 y depending on corpus; 8.0 is a mid-range prior. `eval/` fits
    # this empirically and prints a suggested replacement.
    age_sigma_years: float = 8.0
    age_min_confidence: float = 0.34      # below -> "unknown" (4 brackets)
    gender_min_confidence: float = 0.60   # below -> "unknown"
    # Multiplicative shrinkage applied to confidences on imperfect audio, so a
    # noisy truck cab cannot produce a 0.99.
    degraded_confidence_factor: float = 0.80
    child_age_bracket_unknown: bool = True

    # -------------------------------------------------------------- language
    enable_language_id: bool = True
    # Any name registered via @register_language_backend. Swapping Whisper for
    # VoxLingua107 is a new file plus this env var -- no core edits.
    language_backend: str = "whisper"
    # Best-effort: if LID has not finished within this budget the field comes
    # back null rather than pushing the response past the latency target.
    language_budget_ms: float = 250.0
    language_min_confidence: float = 0.50

    # ------------------------------------------------------------- streaming
    ws_emit_interval_ms: float = 1_000.0
    ws_min_speech_seconds: float = 1.5
    ws_window_seconds: float = 8.0
    ws_ema_alpha: float = 0.55
    ws_max_session_seconds: float = 600.0
    ws_max_bytes: int = 64 * 1024 * 1024

    @model_validator(mode="after")
    def _derive_admission_limit(self) -> "Settings":
        """Couple the admission semaphore to actual parallelism.

        These two numbers must be set together, and getting it wrong is subtle
        rather than loud. The semaphore bounds how many requests are ADMITTED;
        the thread pool bounds how many are actually RUNNING. If the first is
        much larger than the second, the surplus does not fail fast -- it queues
        invisibly inside the executor, where nothing measures it and no timeout
        applies. Measured with an 8-slot semaphore over a 2-thread pool: 40
        concurrent requests gave a p95 of 2.1 s, four times the SLO, while the
        service reported itself healthy and shed only the requests it never
        admitted.

        Admitting a small multiple of the thread count keeps the invisible queue
        to roughly one extra round of work, so an admitted request still has a
        realistic chance of finishing inside its budget. Anything beyond that
        should be shed with 429, which is visible, or absorbed by another
        replica.
        """
        if self.max_concurrent_inferences <= 0:
            object.__setattr__(
                self,
                "max_concurrent_inferences",
                max(1, self.inference_threads * ADMISSION_MULTIPLIER),
            )
        return self


ADMISSION_MULTIPLIER = 2


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
