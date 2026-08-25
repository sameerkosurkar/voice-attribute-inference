"""Real-weights integration tests. `pytest -m slow`.

Kept out of the default run because they download ~1 GB on first use. They
cover the three things the mock backend cannot:

  1. That the gender labels are not permuted. This is THE bug this codebase
     already hit once -- the model card documents the head as
     `child, female, male` while the checkpoint's config.json declares
     `{0: female, 1: male, 2: child}`. Trusting the card inverts every
     prediction while still returning well-formed, high-confidence JSON. No
     schema or shape assertion catches that; only running known-gender audio
     through the model does.

  2. That the latency target actually holds on real weights.

  3. That the model's raw outputs land in a plausible range.
"""

from __future__ import annotations

import shutil
import statistics
import time

import numpy as np
import pytest

from app.config import Settings
from app.inference.calibration import calibrate
from scripts.make_sample_audio import ESPEAK_VOICES, add_noise, espeak_available, synth_espeak

pytestmark = pytest.mark.slow

SAMPLE_RATE = 16_000
# Stated in the assignment: end-to-end inference under 500 ms on a 5 s chunk.
LATENCY_BUDGET_MS = 500.0


@pytest.fixture(scope="module")
def backend():
    from app.inference.audeering import AudeeringBackend

    instance = AudeeringBackend(Settings())
    instance.load()
    instance.warmup()
    return instance


@pytest.fixture(scope="module")
def five_seconds():
    speech = synth_espeak("adult_male", 5.0)
    if speech is None:
        pytest.skip("espeak-ng not installed")
    return speech


def test_model_loads_and_resolves_its_label_order(backend):
    assert backend.ready
    assert set(backend.gender_index()) == {"female", "male", "child"}


@pytest.mark.skipif(not espeak_available(), reason="needs espeak-ng")
@pytest.mark.parametrize(
    "preset, expected",
    [("adult_male", "male"), ("older_male", "male"), ("adult_female", "female")],
)
def test_gender_labels_are_not_permuted(backend, preset, expected):
    """Guards against the model-card/config.json label discrepancy.

    Synthetic voices, so this is not an accuracy measurement -- it is a
    permutation check, and a permutation is a gross error that synthetic
    speech detects perfectly well.
    """
    samples = synth_espeak(preset, 5.0)
    raw = backend.predict(samples, SAMPLE_RATE)
    winner = max(
        (("female", raw.p_female), ("male", raw.p_male), ("child", raw.p_child)),
        key=lambda kv: kv[1],
    )[0]
    assert winner == expected, (
        f"{preset} read as {winner}. Check config.id2label against how "
        f"RawPrediction is populated in app/inference/audeering.py."
    )


def test_raw_outputs_are_in_range(backend, five_seconds):
    raw = backend.predict(five_seconds, SAMPLE_RATE)
    assert 0.0 <= raw.age_years <= 120.0
    assert all(0.0 <= p <= 1.0 for p in (raw.p_child, raw.p_female, raw.p_male))
    assert raw.p_child + raw.p_female + raw.p_male == pytest.approx(1.0, abs=1e-3)


def test_inference_meets_the_latency_budget(backend, five_seconds):
    """The forward pass alone, measured over repeats after warmup."""
    timings = []
    for _ in range(10):
        started = time.perf_counter()
        backend.predict(five_seconds, SAMPLE_RATE)
        timings.append((time.perf_counter() - started) * 1000.0)

    p50 = statistics.median(timings)
    p95 = sorted(timings)[int(0.95 * len(timings)) - 1]
    print(f"\ninference p50={p50:.1f}ms p95={p95:.1f}ms")
    assert p95 < LATENCY_BUDGET_MS, f"p95 {p95:.1f}ms exceeds {LATENCY_BUDGET_MS}ms"


def test_end_to_end_meets_the_latency_budget(five_seconds):
    """Decode + quality gate + inference + calibration, through the HTTP layer."""
    import asyncio

    import httpx

    from app.config import get_settings
    from app.main import create_app
    from scripts.make_sample_audio import wav_bytes

    get_settings.cache_clear()
    payload = wav_bytes(five_seconds)

    async def run():
        application = create_app()
        async with application.router.lifespan_context(application):
            transport = httpx.ASGITransport(app=application)
            async with httpx.AsyncClient(transport=transport,
                                         base_url="http://test") as client:
                for _ in range(3):  # warm
                    await client.post("/analyze", content=payload,
                                      headers={"content-type": "audio/wav"})
                timings = []
                for _ in range(10):
                    response = await client.post(
                        "/analyze", content=payload,
                        headers={"content-type": "audio/wav"},
                    )
                    assert response.status_code == 200
                    timings.append(response.json()["processing_ms"])
                return timings

    timings = asyncio.run(run())
    get_settings.cache_clear()

    p95 = sorted(timings)[int(0.95 * len(timings)) - 1]
    print(f"\nend-to-end p50={statistics.median(timings):.0f}ms p95={p95:.0f}ms")
    assert p95 < LATENCY_BUDGET_MS, f"p95 {p95}ms exceeds {LATENCY_BUDGET_MS}ms"


@pytest.mark.skipif(not espeak_available(), reason="needs espeak-ng")
def test_noise_lowers_confidence_rather_than_flipping_the_answer(backend, five_seconds):
    """Graceful degradation, end to end on real weights.

    The requirement is not that noisy audio still be correct -- it is that the
    service become less certain rather than confidently wrong.
    """
    settings = Settings()
    from app.audio.quality import assess

    clean_report = assess(five_seconds, SAMPLE_RATE, settings)
    clean = calibrate(backend.predict(five_seconds, SAMPLE_RATE), settings,
                      confidence_factor=clean_report.confidence_factor)

    noisy_audio = add_noise(five_seconds, 3.0)
    noisy_report = assess(noisy_audio, SAMPLE_RATE, settings)
    noisy = calibrate(backend.predict(noisy_audio, SAMPLE_RATE), settings,
                      confidence_factor=noisy_report.confidence_factor)

    assert noisy.gender.confidence <= clean.gender.confidence
    if noisy.gender.prediction.value != "unknown":
        assert noisy.gender.prediction == clean.gender.prediction, (
            "noise should reduce confidence or yield unknown, not flip the label"
        )


@pytest.mark.skipif(not shutil.which("espeak-ng"), reason="needs espeak-ng")
def test_language_identification_reads_english(five_seconds):
    """Constructed through the REGISTRY, not directly.

    Production creates this via `language_registry.create(...)`. An earlier
    version of this test instantiated `LanguageIdentifier` directly, which meant
    the registry path -- factory signature, registration name, import order --
    was never exercised with the real model. Building it the way the service
    does is the whole point of an integration test.
    """
    from app.inference import language_registry
    from app.inference.base import LanguageBackend

    settings = Settings()
    identifier = language_registry.create(settings.language_backend, settings)
    assert isinstance(identifier, LanguageBackend)

    identifier.load()
    identifier.warmup()
    assert identifier.ready

    result = identifier.identify(five_seconds, SAMPLE_RATE)
    assert result is None or result.prediction == "en"


def test_shipped_language_backend_is_registered_under_its_configured_name():
    """`VA_LANGUAGE_BACKEND`'s default must actually resolve."""
    from app.inference import language_registry

    default = Settings().language_backend
    assert default in language_registry.available_language_backends(), (
        f"default VA_LANGUAGE_BACKEND={default!r} is not registered; "
        f"available: {language_registry.available_language_backends()}"
    )


def test_shipped_attribute_backends_are_registered_and_constructible():
    """Same for every shipped attribute backend: the name in the docs must be
    a name the registry actually knows."""
    from app.inference import registry

    settings = Settings()
    for name in ("mock", "audeering", "onnx"):
        spec = registry.get_spec(name)
        assert spec is not None, f"{name} is not registered"
        if spec.is_available is not None and not spec.is_available(settings):
            continue          # e.g. onnx with no export on this machine
        instance = registry.create(name, settings)
        assert hasattr(instance, "predict") and hasattr(instance, "load")


def test_repeated_calls_are_deterministic(backend, five_seconds):
    """No dropout leaking through, no state carried between requests."""
    first = backend.predict(five_seconds, SAMPLE_RATE)
    second = backend.predict(five_seconds, SAMPLE_RATE)
    assert first.age_years == pytest.approx(second.age_years, abs=1e-4)
    assert first.p_male == pytest.approx(second.p_male, abs=1e-5)


def test_vad_state_does_not_leak_between_calls(five_seconds, silence=None):
    """Silero VAD is recurrent. Stale state across requests would leak one
    caller's audio into the next caller's quality verdict -- a bug and a
    privacy problem.
    """
    from app.audio.quality import assess

    settings = Settings()
    quiet = np.zeros(5 * SAMPLE_RATE, dtype=np.float32)

    baseline = assess(quiet, SAMPLE_RATE, settings).speech_seconds
    assess(five_seconds, SAMPLE_RATE, settings)          # loud speech in between
    after = assess(quiet, SAMPLE_RATE, settings).speech_seconds
    assert baseline == after == 0.0
