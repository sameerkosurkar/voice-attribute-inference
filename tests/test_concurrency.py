"""Concurrency safety.

REGRESSION TEST WITH A STORY. The first version of the quality gate held one
`lru_cache`d Silero VAD instance for the whole process. Silero is a recurrent
TorchScript module with mutable hidden state, and `assess()` runs in a thread
pool -- so under concurrent requests several threads called forward() and
reset_states() on the same object simultaneously. The process **segfaulted**
(SIGSEGV inside torch's `Module._call_impl`) after a handful of parallel
requests.

The pre-existing VAD test did not catch it because it only made *sequential*
calls. Nothing about a single-threaded test can reveal a data race.

Worse than the crash: interleaved recurrent state means one caller's audio
influences another caller's speech detection, which is exactly the cross-request
leak PRIVACY.md promises does not happen.

Fixed with thread-local VAD instances. These tests fail (by crashing the
interpreter) if that ever regresses.
"""

from __future__ import annotations

import asyncio
import threading

import httpx
import numpy as np
import pytest

from app.audio.quality import assess
from app.config import Settings, get_settings
from app.schemas import AudioQuality

SAMPLE_RATE = 16_000


def test_quality_gate_is_thread_safe(speech, silence):
    """Hammer the VAD from many threads at once.

    Before the fix this segfaulted the interpreter rather than failing an
    assertion -- which is why the assertion below is about *consistency*, not
    just about not raising.
    """
    settings = Settings(backend="mock", enable_language_id=False)
    expected_speech = assess(speech, SAMPLE_RATE, settings).speech_seconds
    expected_silence = assess(silence, SAMPLE_RATE, settings).speech_seconds

    results: list[tuple[str, float]] = []
    errors: list[BaseException] = []
    lock = threading.Lock()

    def worker(index: int) -> None:
        try:
            for _ in range(3):
                # Alternate loud and silent buffers: if recurrent state leaked
                # between threads, silence would start "detecting" speech.
                if index % 2:
                    value = assess(speech, SAMPLE_RATE, settings).speech_seconds
                    tag = "speech"
                else:
                    value = assess(silence, SAMPLE_RATE, settings).speech_seconds
                    tag = "silence"
                with lock:
                    results.append((tag, value))
        except BaseException as exc:  # noqa: BLE001
            with lock:
                errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=120)

    assert not errors, f"concurrent VAD raised: {errors[:3]}"
    assert len(results) == 24

    for tag, value in results:
        expected = expected_speech if tag == "speech" else expected_silence
        assert value == pytest.approx(expected, abs=0.05), (
            f"{tag} measured {value}s under concurrency but {expected}s alone -- "
            "VAD state is leaking between threads"
        )


def test_silence_never_detects_speech_under_load(silence, speech):
    """The privacy-relevant half, stated directly."""
    settings = Settings(backend="mock", enable_language_id=False)
    verdicts: list[AudioQuality] = []
    lock = threading.Lock()

    def noisy_neighbour() -> None:
        for _ in range(4):
            assess(speech, SAMPLE_RATE, settings)

    def quiet_caller() -> None:
        for _ in range(4):
            report = assess(silence, SAMPLE_RATE, settings)
            with lock:
                verdicts.append(report.quality)

    threads = [threading.Thread(target=noisy_neighbour) for _ in range(4)]
    threads += [threading.Thread(target=quiet_caller) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=120)

    assert verdicts, "quiet callers produced no verdicts"
    assert all(v is AudioQuality.INSUFFICIENT for v in verdicts), (
        "silence was judged usable while loud audio was processed concurrently"
    )


async def test_concurrent_requests_are_independent(app_client, speech_wav, silence):
    """End to end: interleaved good and silent uploads must not affect each
    other's verdicts."""
    from scripts.make_sample_audio import wav_bytes

    application, transport = app_client
    silent_wav = wav_bytes(silence)

    async with application.router.lifespan_context(application):
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
            async def post(payload: bytes):
                return await client.post(
                    "/analyze", content=payload, headers={"content-type": "audio/wav"}
                )

            payloads = [speech_wav if i % 2 else silent_wav for i in range(12)]
            responses = await asyncio.gather(*[post(p) for p in payloads])

    for index, response in enumerate(responses):
        assert response.status_code == 200
        quality = response.json()["audio_quality"]
        if index % 2:
            assert quality != "insufficient", "speech misjudged under concurrency"
        else:
            assert quality == "insufficient", "silence misjudged under concurrency"


def test_repeated_assess_is_deterministic(speech):
    """Sequential determinism -- the property the original (insufficient) test
    checked. Kept, because it is still worth guarding."""
    settings = Settings(backend="mock", enable_language_id=False)
    first = assess(speech, SAMPLE_RATE, settings)
    for _ in range(5):
        again = assess(speech, SAMPLE_RATE, settings)
        assert again.speech_seconds == pytest.approx(first.speech_seconds, abs=1e-6)
        assert again.quality is first.quality


# --------------------------------------------------------- publish ordering
#
# A backend's "am I loaded?" flag must be the LAST thing load() assigns.
# predict() does its cold-path check outside the load lock, so if the flag is
# published before the state it implies, a concurrent caller sees a usable model
# alongside half-initialised metadata.
#
# Concretely: publishing `self._model` before `self._gender_index` let a racing
# predict() read an empty index and silently fall back to a hardcoded label
# order. On this checkpoint the fallback happens to be right, so the bug emits
# CORRECT output here and silently inverted labels on a differently-ordered
# checkpoint -- undetectable without this test.
#
# The window is microseconds wide, so a plain concurrency test never hits it.
# These tests widen it deliberately. That is the only honest way to test a
# narrow race: make it wide, prove the ordering holds, keep the test.

def _assert_publishes_flag_last(backend, flag_attr: str, *implied_attrs: str) -> None:
    """Record the ORDER in which load() assigns attributes.

    Racing for this would be futile -- the window is microseconds wide and a
    plain concurrency test will pass against the buggy ordering essentially
    always. (I confirmed that: an earlier version of this test raced for the
    window and passed happily with the bug reintroduced.)

    So the invariant is checked directly instead of probabilistically. Wrapping
    __setattr__ records the assignment sequence, and the assertion is simply
    that the completion flag comes last. That fails deterministically the moment
    someone reorders load(), which is exactly what a regression test should do.
    """
    order: list[str] = []
    original_setattr = type(backend).__setattr__

    def recording_setattr(self, name, value):
        if self is backend:
            order.append(name)
        original_setattr(self, name, value)

    type(backend).__setattr__ = recording_setattr
    try:
        backend.load()
    finally:
        type(backend).__setattr__ = original_setattr

    assert flag_attr in order, f"load() never assigned {flag_attr}"
    flag_position = len(order) - 1 - order[::-1].index(flag_attr)

    for attr in implied_attrs:
        assert attr in order, f"load() never assigned {attr}"
        attr_position = len(order) - 1 - order[::-1].index(attr)
        assert attr_position < flag_position, (
            f"load() assigned the completion flag `{flag_attr}` before `{attr}`.\n"
            f"predict() checks `{flag_attr}` OUTSIDE the load lock, so a "
            f"concurrent caller can see the backend as loaded while `{attr}` is "
            f"still empty.\n"
            f"For `_gender_index` specifically that means silently falling back "
            f"to a hardcoded label order -- which is correct for this checkpoint "
            f"and silently inverted for a differently-ordered one.\n"
            f"Assignment order was: {order}"
        )


@pytest.mark.slow
def test_torch_backend_publishes_state_before_its_flag():
    from app.inference.audeering import AudeeringBackend

    backend = AudeeringBackend(Settings(enable_language_id=False))
    _assert_publishes_flag_last(backend, "_model", "_gender_index", "_extractor")


@pytest.mark.slow
def test_onnx_backend_publishes_state_before_its_flag():
    from app.inference import onnx_backend

    settings = Settings(enable_language_id=False)
    if not onnx_backend.export_available(settings):
        pytest.skip("no ONNX export; run make export-onnx")

    backend = onnx_backend.OnnxBackend(settings)
    _assert_publishes_flag_last(
        backend, "_session", "_gender_index", "_extractor", "_input_name"
    )


@pytest.mark.slow
def test_language_backend_publishes_state_before_its_flag():
    from app.inference.language import LanguageIdentifier

    backend = LanguageIdentifier(Settings())
    _assert_publishes_flag_last(backend, "_model", "_lang_token_ids", "_processor")


@pytest.mark.slow
def test_backend_predictions_are_stable_across_threads():
    """Distinct inputs on many threads must each get their single-threaded answer.

    Different inputs per thread is the point: if any state leaked between
    threads, one speaker's result would drift toward another's.
    """
    import threading

    from app.inference.audeering import AudeeringBackend
    from scripts.make_sample_audio import espeak_available, synth_espeak

    if not espeak_available():
        pytest.skip("needs espeak-ng")

    settings = Settings(enable_language_id=False)
    backend = AudeeringBackend(settings)
    backend.load()
    backend.warmup()

    clips = {
        preset: synth_espeak(preset, 3.0)
        for preset in ("adult_male", "adult_female", "older_male")
    }
    baseline = {
        preset: backend.predict(clip, SAMPLE_RATE).p_male
        for preset, clip in clips.items()
    }

    results: list[tuple[str, float]] = []
    errors: list[BaseException] = []
    lock = threading.Lock()

    def worker(index: int) -> None:
        preset = list(clips)[index % len(clips)]
        try:
            for _ in range(3):
                value = backend.predict(clips[preset], SAMPLE_RATE).p_male
                with lock:
                    results.append((preset, value))
        except BaseException as exc:  # noqa: BLE001
            with lock:
                errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(9)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=180)

    assert not errors, f"concurrent inference raised: {errors[:2]}"
    for preset, value in results:
        assert value == pytest.approx(baseline[preset], abs=1e-4), (
            f"{preset} drifted under concurrency"
        )
