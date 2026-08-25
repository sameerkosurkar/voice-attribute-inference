"""The audio-quality gate.

Split deliberately into two kinds of test:

  * Estimator tests, where ground truth is constructed rather than assumed --
    a signal mixed at a known SNR, a known fraction of clipped samples, a known
    brick-wall band limit. These assert the measurement is right.

  * Decision tests, which assert the good/degraded/insufficient policy and the
    confidence shrinkage that follows from it.

The negative cases matter more than the positive ones here. A gate that says
"good" for clean speech but also "good" for a hold tone has not gated anything.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.audio.quality import assess, high_band_ratio
from app.config import Settings
from app.schemas import AudioQuality

SAMPLE_RATE = 16_000


def _settings(**overrides) -> Settings:
    base = dict(backend="mock", enable_language_id=False)
    base.update(overrides)
    return Settings(**base)


# ------------------------------------------------------------------ estimators
def test_high_band_ratio_separates_wideband_from_narrowband(speech, narrowband_speech):
    """Real wideband speech keeps energy above 4 kHz (fricatives live there);
    an 8 kHz telephony leg has essentially none."""
    wide = high_band_ratio(speech, SAMPLE_RATE)
    narrow = high_band_ratio(narrowband_speech, SAMPLE_RATE)
    assert narrow < wide
    assert narrow < 1e-4, "brick-walled audio should have ~no energy above 4 kHz"


def test_high_band_ratio_of_white_noise_is_large():
    noise = np.random.default_rng(0).standard_normal(SAMPLE_RATE * 2).astype(np.float32)
    assert high_band_ratio(noise, SAMPLE_RATE) > 0.2


def test_high_band_ratio_handles_degenerate_input():
    assert high_band_ratio(np.zeros(0, dtype=np.float32), SAMPLE_RATE) == 0.0
    assert high_band_ratio(np.zeros(64, dtype=np.float32), SAMPLE_RATE) == 0.0
    assert high_band_ratio(np.zeros(SAMPLE_RATE, dtype=np.float32), SAMPLE_RATE) == 0.0


def test_clipping_ratio_tracks_actual_clipping(speech, clipped_speech):
    clean = assess(speech, SAMPLE_RATE, _settings())
    clipped = assess(clipped_speech, SAMPLE_RATE, _settings())
    assert clean.clipping_ratio < 0.001
    assert clipped.clipping_ratio > 0.01


def test_snr_estimate_falls_as_noise_rises(speech):
    """Monotonicity is the property that matters -- the absolute dB value is a
    speech-to-background ratio, not a true SNR against a clean reference."""
    from scripts.make_sample_audio import add_noise

    measured = [
        assess(add_noise(speech, snr), SAMPLE_RATE, _settings()).snr_db
        for snr in (20.0, 10.0, 0.0)
    ]
    assert measured == sorted(measured, reverse=True), measured


# ------------------------------------------------------------------- decisions
def test_silence_is_insufficient(silence):
    report = assess(silence, SAMPLE_RATE, _settings())
    assert report.quality is AudioQuality.INSUFFICIENT
    assert report.speech_seconds == 0.0
    assert report.usable is False
    assert report.confidence_factor == 0.0


def test_pure_tone_is_not_speech(tone):
    """A 440 Hz hold tone is loud and highly periodic. Any gate driven by
    energy alone would pass it; the VAD must not."""
    report = assess(tone, SAMPLE_RATE, _settings())
    assert report.quality is AudioQuality.INSUFFICIENT
    assert report.speech_seconds < 1.0


def test_insufficient_audio_reports_only_the_blocking_reason(silence):
    """Diagnostics should say why we refused, not list every other property
    that also happened to be true of a silent buffer."""
    report = assess(silence, SAMPLE_RATE, _settings())
    assert len(report.reasons) == 1
    assert "speech" in report.reasons[0]


def test_narrowband_audio_is_degraded_not_rejected(narrowband_speech):
    """A G.711 leg is a domain shift worth flagging, but the call still
    deserves an answer."""
    report = assess(narrowband_speech, SAMPLE_RATE, _settings())
    assert report.quality is AudioQuality.DEGRADED
    assert report.usable is True
    assert any("band-limited" in reason for reason in report.reasons)


def test_clipped_audio_is_flagged(clipped_speech):
    report = assess(clipped_speech, SAMPLE_RATE, _settings())
    assert report.quality is AudioQuality.DEGRADED
    assert any("clipped" in reason for reason in report.reasons)


def test_degraded_audio_shrinks_the_confidence_factor(speech, noisy_speech):
    clean = assess(speech, SAMPLE_RATE, _settings())
    noisy = assess(noisy_speech, SAMPLE_RATE, _settings())
    assert noisy.confidence_factor < clean.confidence_factor
    assert 0.0 < noisy.confidence_factor < 1.0


def test_quality_degrades_monotonically_with_noise(speech):
    """The core graceful-degradation guarantee: as a truck cab gets louder the
    verdict only ever moves in one direction, and eventually refuses."""
    from scripts.make_sample_audio import add_noise

    rank = {AudioQuality.GOOD: 0, AudioQuality.DEGRADED: 1, AudioQuality.INSUFFICIENT: 2}
    verdicts = [
        rank[assess(add_noise(speech, snr), SAMPLE_RATE, _settings()).quality]
        for snr in (20.0, 10.0, 0.0, -10.0)
    ]
    assert verdicts == sorted(verdicts), verdicts
    assert verdicts[-1] == rank[AudioQuality.INSUFFICIENT], "-10 dB must be refused"


def test_thresholds_are_configurable(speech):
    """Ops needs to retune per telephony vendor without a code change."""
    impossible = _settings(min_speech_seconds=999.0)
    assert assess(speech, SAMPLE_RATE, impossible).quality is AudioQuality.INSUFFICIENT


def test_short_but_clean_audio_is_penalised(speech):
    """1.2 s genuinely tells you less than 5 s, even with no noise at all."""
    short = assess(speech[: int(1.2 * SAMPLE_RATE)], SAMPLE_RATE, _settings())
    full = assess(speech, SAMPLE_RATE, _settings())
    if short.usable:
        assert short.confidence_factor < full.confidence_factor


def test_report_fields_are_json_safe(speech):
    report = assess(speech, SAMPLE_RATE, _settings())
    for value in (report.speech_seconds, report.total_seconds, report.snr_db,
                  report.clipping_ratio, report.high_band_ratio):
        assert isinstance(value, float)
        assert not np.isnan(value) and not np.isinf(value)


@pytest.mark.skipif(
    "not __import__('shutil').which('espeak-ng')",
    reason="needs espeak-ng for speech the VAD will accept",
)
def test_clean_speech_is_good(speech):
    report = assess(speech, SAMPLE_RATE, _settings())
    assert report.quality is AudioQuality.GOOD
    assert report.reasons == []
    assert report.confidence_factor == 1.0
    assert report.speech_seconds > 2.0


def test_numpy_synthesis_is_correctly_rejected_as_non_speech(numpy_speech):
    """Documents a real finding rather than hiding it.

    The numpy source-filter synthesiser produces something that *looks* like
    speech on a spectrogram, and Silero VAD refuses to certify it. That is the
    VAD working: a detector loose enough to accept a three-formant pulse train
    would also accept engine noise. See scripts/make_sample_audio.py.
    """
    report = assess(numpy_speech, SAMPLE_RATE, _settings())
    assert report.speech_seconds < 2.0
