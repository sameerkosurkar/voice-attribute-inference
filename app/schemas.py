"""Wire contract.

The shape of `AnalyzeResponse` is fixed by the assignment. Optional fields
(`language`, `request_id`) are additive and never replace a required one, so a
client coded against the specified contract keeps working.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class Gender(str, Enum):
    MALE = "male"
    FEMALE = "female"
    UNKNOWN = "unknown"


class AgeBracket(str, Enum):
    A18_30 = "18-30"
    A31_45 = "31-45"
    A46_60 = "46-60"
    A60_PLUS = "60+"
    UNKNOWN = "unknown"


class AudioQuality(str, Enum):
    GOOD = "good"
    DEGRADED = "degraded"
    INSUFFICIENT = "insufficient"


class GenderPrediction(BaseModel):
    prediction: Gender
    confidence: float = Field(ge=0.0, le=1.0)


class AgePrediction(BaseModel):
    prediction: AgeBracket
    confidence: float = Field(ge=0.0, le=1.0)


class LanguagePrediction(BaseModel):
    """Best-effort. `prediction` is an ISO 639-1 code."""

    prediction: str
    confidence: float = Field(ge=0.0, le=1.0)


class QualityDetail(BaseModel):
    """Why the quality verdict came out the way it did.

    Returned only when `?debug=true`. Kept out of the default response so the
    documented contract stays exactly as specified, but invaluable when a
    customer asks "why did you say unknown for this call?".
    """

    speech_seconds: float
    total_seconds: float
    snr_db: float
    clipping_ratio: float
    high_band_ratio: float  # energy fraction above 4 kHz
    reasons: list[str] = Field(default_factory=list)


class Timings(BaseModel):
    decode_ms: float
    quality_ms: float
    inference_ms: float
    language_ms: float | None = None
    total_ms: float


class AnalyzeResponse(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    contact_id: str
    gender: GenderPrediction
    age_bracket: AgePrediction
    processing_ms: int
    audio_quality: AudioQuality

    # --- additive, optional -------------------------------------------------
    language: LanguagePrediction | None = None
    request_id: str | None = None
    quality_detail: QualityDetail | None = None
    timings: Timings | None = None


class StreamEventType(str, Enum):
    READY = "ready"
    PARTIAL = "partial"
    FINAL = "final"
    ERROR = "error"


class StreamPrediction(AnalyzeResponse):
    """A progressive prediction over an open WebSocket.

    Extends the REST contract so a client can reuse one parser for both, and
    adds the fields that only make sense mid-stream.
    """

    type: StreamEventType = StreamEventType.PARTIAL
    is_final: bool = False
    chunks_seen: int = 0
    audio_seconds: float = 0.0
    speech_seconds: float = 0.0
    # True once consecutive partials stop moving materially -- lets a voice
    # agent commit to a persona early instead of waiting for the call to end.
    stable: bool = False


class ErrorResponse(BaseModel):
    error: str
    message: str
    request_id: str | None = None
    detail: str | None = None
