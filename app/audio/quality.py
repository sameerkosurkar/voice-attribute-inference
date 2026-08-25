"""The audio-quality gate.

This runs BEFORE inference, and it is the most important safety property in the
service. wav2vec2 will happily return a confident age and gender for two
seconds of diesel engine; nothing in the model tells you it was nonsense. The
assignment asks for graceful degradation, and the only honest way to get it is
to decide independently whether the audio could possibly support a prediction.

Four signals, each catching a different real failure on logistics calls:

  speech_seconds  Silero VAD. Catches "driver picked up and said nothing", hold
                  music, IVR tones, and dead air. This is the dominant signal:
                  the model needs roughly 2 s of voiced speech before its
                  outputs stabilise.

  snr_db          Energy in VAD-positive frames vs. VAD-negative frames. Catches
                  the truck-cab and warehouse case -- there IS speech, it is
                  just buried. Note this is a *speech-to-background* ratio, not
                  a true SNR: it needs no clean reference, which we never have.

  clipping_ratio  Fraction of samples at full scale. Catches wind buffeting a
                  handset mic and over-driven bluetooth headsets. Clipping adds
                  broadband harmonics that shift formants, which is precisely
                  what an age/gender model reads.

  high_band_ratio Fraction of spectral energy in 4-8 kHz. Catches narrowband
                  telephony and aggressive codecs. G.711 cuts everything above
                  ~3.4 kHz, so 8 kHz-sampled audio upsampled to 16 kHz has
                  essentially nothing above 4 kHz, while genuine wideband speech
                  always does -- fricatives (/s/, /sh/) live up there. The model
                  was trained on wideband speech, so a narrowband leg is a real
                  domain shift, not just "quieter".

                  Note this deliberately is NOT spectral roll-off, which is the
                  obvious choice and is wrong here. Speech energy is so heavily
                  low-frequency that the 95% roll-off of perfectly clean
                  wideband speech sits around 1.3 kHz -- indistinguishable from
                  a narrowband clip, which measures 1.28 kHz. Measured on real
                  speech, roll-off separates the two by 1%; the 4-8 kHz band
                  ratio separates them by a factor of ~17 (0.085% vs 0.005%).

Thresholds are heuristics, tuned on the synthetic fixtures and sanity-checked
against Common Voice with injected noise (see eval/). They are all env-tunable
because the right cut-point depends on the telephony vendor.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

import numpy as np
import structlog

from app.config import Settings
from app.schemas import AudioQuality

log = structlog.get_logger(__name__)

_VAD_SR = 16_000
_VAD_FRAME = 512          # Silero v5+ requires exactly 512 samples @ 16 kHz
_EPS = 1e-10


@dataclass(slots=True)
class QualityReport:
    quality: AudioQuality
    speech_seconds: float
    total_seconds: float
    snr_db: float
    clipping_ratio: float
    high_band_ratio: float
    speech_mask: np.ndarray | None = None
    reasons: list[str] = field(default_factory=list)
    # Multiplicative shrinkage applied to model confidences; filled in by
    # assess() via _confidence_factor().
    confidence_factor: float = 1.0

    @property
    def usable(self) -> bool:
        return self.quality is not AudioQuality.INSUFFICIENT


# One Silero instance PER THREAD, not one shared instance.
#
# This is not premature caution -- a shared instance segfaults. Silero VAD is a
# recurrent TorchScript module carrying mutable hidden state, and `assess()` runs
# in a thread pool, so under concurrent requests several threads call forward()
# and reset_states() on the same object at once. That crashed the process
# (SIGSEGV inside torch's Module._call_impl) within a handful of parallel
# requests.
#
# Even without the crash it would be a correctness and privacy bug: interleaved
# recurrent state means one caller's audio influences another caller's speech
# detection. PRIVACY.md promises that cannot happen.
#
# Thread-local instances remove all three problems by construction, and the cost
# is trivial -- the model is ~2 MB and the pool is 2-4 threads. The alternative,
# a global lock, would serialise the VAD across the whole process.
_vad_local = threading.local()


def _load_vad():
    """The calling thread's Silero VAD. Weights ship inside the pip wheel, so
    this never touches the network -- which is what lets the container run
    fully offline."""
    model = getattr(_vad_local, "model", None)
    if model is None:
        from silero_vad import load_silero_vad

        model = load_silero_vad()
        model.eval()
        _vad_local.model = model
    return model


def warmup_vad() -> None:
    import torch

    model = _load_vad()
    with torch.inference_mode():
        model(torch.zeros(_VAD_FRAME), _VAD_SR)
    model.reset_states()


def _vad_frame_probs(samples: np.ndarray) -> np.ndarray:
    """Per-frame speech probability. Falls back to an energy heuristic if
    Silero is unavailable, so the service still starts in a stripped env."""
    import torch

    try:
        model = _load_vad()
    except Exception:  # pragma: no cover - only hit in a broken install
        log.warning("vad_unavailable_using_energy_fallback")
        return _energy_fallback_probs(samples)

    n_frames = len(samples) // _VAD_FRAME
    if n_frames == 0:
        return np.zeros(0, dtype=np.float32)

    frames = samples[: n_frames * _VAD_FRAME].reshape(n_frames, _VAD_FRAME)
    tensor = torch.from_numpy(np.ascontiguousarray(frames, dtype=np.float32))

    model.reset_states()  # VAD is stateful/recurrent; stale state across
                          # requests would leak one caller's audio into the
                          # next caller's decision. Both a bug and a privacy leak.
    probs = np.empty(n_frames, dtype=np.float32)
    with torch.inference_mode():
        for i in range(n_frames):
            probs[i] = float(model(tensor[i], _VAD_SR).item())
    model.reset_states()
    return probs


def _energy_fallback_probs(samples: np.ndarray) -> np.ndarray:
    n_frames = len(samples) // _VAD_FRAME
    if n_frames == 0:
        return np.zeros(0, dtype=np.float32)
    frames = samples[: n_frames * _VAD_FRAME].reshape(n_frames, _VAD_FRAME)
    rms = np.sqrt(np.mean(frames**2, axis=1) + _EPS)
    # Speech is the energy above the noise floor; the 20th percentile is a
    # reasonable stand-in for "background" in a short clip.
    floor = np.percentile(rms, 20)
    return (rms > max(floor * 3.0, 0.01)).astype(np.float32)


def _snr_db(samples: np.ndarray, speech_mask: np.ndarray) -> float:
    """Speech-frame energy over background-frame energy, in dB."""
    n_frames = len(speech_mask)
    if n_frames == 0:
        return 0.0
    frames = samples[: n_frames * _VAD_FRAME].reshape(n_frames, _VAD_FRAME)
    power = np.mean(frames**2, axis=1)

    speech_power = power[speech_mask]
    noise_power = power[~speech_mask]
    if speech_power.size == 0:
        return 0.0
    if noise_power.size == 0:
        # Wall-to-wall speech with no gap to measure against. Estimate the floor
        # from the quietest decile of the speech itself -- conservative, and it
        # avoids reporting an absurd +60 dB for a clip that simply never pauses.
        noise_power = np.percentile(power, 10, keepdims=True)

    ratio = (np.mean(speech_power) + _EPS) / (np.mean(noise_power) + _EPS)
    return float(np.clip(10.0 * np.log10(ratio), -20.0, 60.0))


def _clipping_ratio(samples: np.ndarray) -> float:
    if samples.size == 0:
        return 0.0
    return float(np.mean(np.abs(samples) >= 0.985))


def _band_energy_ratio(
    samples: np.ndarray, sample_rate: int, lo_hz: float, hi_hz: float
) -> float:
    """Fraction of average spectral power falling in [lo_hz, hi_hz)."""
    if samples.size < 512:
        return 0.0
    window = min(4096, 1 << int(np.floor(np.log2(samples.size))))
    if window < 256:
        return 0.0

    # Average several windows so one transient cannot set the verdict.
    hop = max(1, (samples.size - window) // 8) if samples.size > window else 1
    spectra = []
    for start in range(0, max(1, samples.size - window + 1), hop):
        segment = samples[start : start + window]
        if segment.size < window:
            break
        spectra.append(np.abs(np.fft.rfft(segment * np.hanning(window))) ** 2)
        if len(spectra) >= 8:
            break
    if not spectra:
        return 0.0

    power = np.mean(spectra, axis=0)
    total = float(power.sum())
    if total <= _EPS:
        return 0.0
    freqs = np.fft.rfftfreq(window, d=1.0 / sample_rate)
    band = power[(freqs >= lo_hz) & (freqs < hi_hz)]
    return float(band.sum() / total)


def high_band_ratio(samples: np.ndarray, sample_rate: int) -> float:
    """Energy fraction above 4 kHz -- the narrowband/codec detector."""
    nyquist = sample_rate / 2.0
    if nyquist <= 4000.0:
        # Already sampled at or below 8 kHz: definitionally narrowband, and
        # there is no 4-8 kHz band to measure.
        return 0.0
    return _band_energy_ratio(samples, sample_rate, 4000.0, nyquist)


def assess(samples: np.ndarray, sample_rate: int, settings: Settings) -> QualityReport:
    total_seconds = len(samples) / float(sample_rate)
    probs = _vad_frame_probs(samples)
    speech_mask = probs >= settings.vad_threshold
    speech_seconds = float(speech_mask.sum() * _VAD_FRAME / sample_rate)

    snr = _snr_db(samples, speech_mask)
    clipping = _clipping_ratio(samples)
    hi_band = high_band_ratio(samples, sample_rate)

    reasons: list[str] = []
    quality = AudioQuality.GOOD

    def demote(to: AudioQuality, reason: str) -> None:
        nonlocal quality
        reasons.append(reason)
        order = {AudioQuality.GOOD: 0, AudioQuality.DEGRADED: 1, AudioQuality.INSUFFICIENT: 2}
        if order[to] > order[quality]:
            quality = to

    # --- hard stops: no amount of model confidence can rescue these ---------
    if speech_seconds < settings.min_speech_seconds:
        demote(
            AudioQuality.INSUFFICIENT,
            f"only {speech_seconds:.2f}s of detected speech "
            f"(need {settings.min_speech_seconds:.2f}s)",
        )
    if snr < settings.degraded_snr_db and speech_seconds > 0:
        demote(
            AudioQuality.INSUFFICIENT,
            f"speech-to-background ratio {snr:.1f} dB is below "
            f"{settings.degraded_snr_db:.1f} dB",
        )

    # --- soft demotions: predict, but shrink the confidence -----------------
    # Skipped once a hard stop has fired. Listing "moderate background noise"
    # underneath "there is no speech at all" is noise in the diagnostics: the
    # reasons list should say why we refused, not everything else that was also
    # true of a silent buffer.
    if quality is not AudioQuality.INSUFFICIENT:
        _apply_soft_demotions(
            demote, settings, speech_seconds, snr, clipping, hi_band
        )

    report = QualityReport(
        quality=quality,
        speech_seconds=round(speech_seconds, 3),
        total_seconds=round(total_seconds, 3),
        snr_db=round(snr, 2),
        clipping_ratio=round(clipping, 5),
        high_band_ratio=round(hi_band, 6),
        speech_mask=speech_mask,
        reasons=reasons,
    )
    report.confidence_factor = _confidence_factor(report, settings)
    return report


def _confidence_factor(report: QualityReport, settings: Settings) -> float:
    """How much to shrink model confidence given the audio.

    This is deliberately a blunt multiplicative prior rather than a learned
    correction: with no labelled noisy-call data we cannot fit anything
    better, and a blunt shrink that is directionally right beats a precise
    number that is wrong. `eval/run_eval.py --noise-snr` measures whether it
    actually improves calibration (ECE) instead of just lowering confidence.
    """
    if report.quality is AudioQuality.INSUFFICIENT:
        return 0.0
    factor = 1.0
    if report.quality is AudioQuality.DEGRADED:
        factor *= settings.degraded_confidence_factor
    # Extra taper for very short speech: 1.2 s of audio genuinely tells you
    # less than 5 s, even when it is clean.
    if report.speech_seconds < settings.good_speech_seconds:
        span = max(settings.good_speech_seconds - settings.min_speech_seconds, _EPS)
        shortfall = (settings.good_speech_seconds - report.speech_seconds) / span
        factor *= float(np.clip(1.0 - 0.25 * shortfall, 0.7, 1.0))
    return float(np.clip(factor, 0.0, 1.0))


def _apply_soft_demotions(
    demote,
    settings: Settings,
    speech_seconds: float,
    snr: float,
    clipping: float,
    hi_band: float,
) -> None:
    """Conditions that shrink confidence but still permit a prediction."""
    if speech_seconds < settings.good_speech_seconds:
        demote(AudioQuality.DEGRADED, f"short speech segment ({speech_seconds:.2f}s)")
    if snr < settings.good_snr_db:
        demote(AudioQuality.DEGRADED, f"moderate background noise ({snr:.1f} dB)")
    if clipping > settings.max_clipping_ratio:
        demote(AudioQuality.DEGRADED, f"{clipping * 100:.1f}% of samples clipped")
    if hi_band < settings.min_high_band_ratio:
        demote(
            AudioQuality.DEGRADED,
            f"band-limited audio (only {hi_band * 100:.3f}% of energy above "
            "4 kHz) -- likely narrowband telephony or a lossy codec",
        )
