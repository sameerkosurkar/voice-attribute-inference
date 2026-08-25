"""Shared fixtures.

Default test runs use the MOCK backend: the full suite finishes in seconds
without downloading a gigabyte of weights, which is what makes it usable as a
pre-commit gate. The real model is exercised separately by tests marked `slow`
(`pytest -m slow`), which is where latency and label-mapping are checked.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import Settings, get_settings  # noqa: E402
from scripts.make_sample_audio import (  # noqa: E402
    add_noise,
    clip_signal,
    narrowband,
    synth_numpy,
    synth_voice,
    wav_bytes,
)

SAMPLE_RATE = 16_000


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: exercises the real model weights (downloads ~1 GB on first run)"
    )


@pytest.fixture(scope="session")
def settings() -> Settings:
    return Settings(backend="mock", enable_language_id=False, log_json=False)


@pytest.fixture(scope="session")
def speech() -> np.ndarray:
    """5 s of clean synthesised speech (espeak if available, else numpy)."""
    return synth_voice("adult_male", 5.0)


@pytest.fixture(scope="session")
def speech_wav(speech) -> bytes:
    return wav_bytes(speech)


@pytest.fixture(scope="session")
def noisy_speech(speech) -> np.ndarray:
    return add_noise(speech, 8.0)


@pytest.fixture(scope="session")
def clipped_speech(speech) -> np.ndarray:
    return clip_signal(speech)


@pytest.fixture(scope="session")
def narrowband_speech(speech) -> np.ndarray:
    return narrowband(speech)


@pytest.fixture(scope="session")
def silence() -> np.ndarray:
    return np.zeros(5 * SAMPLE_RATE, dtype=np.float32)


@pytest.fixture(scope="session")
def tone() -> np.ndarray:
    """A pure 440 Hz tone: energetic, but categorically not speech.

    The single most important negative fixture in the suite. A service that
    reports a confident gender for a hold tone is the exact failure this
    design is built to prevent.
    """
    t = np.arange(5 * SAMPLE_RATE) / SAMPLE_RATE
    return (0.6 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)


@pytest.fixture(scope="session")
def numpy_speech() -> np.ndarray:
    """Deterministic numpy synthesis, independent of whether espeak exists."""
    return synth_numpy("adult_male", 5.0)


@pytest.fixture
def app_client(settings, monkeypatch):
    """ASGI client against the app wired to the mock backend."""
    import httpx

    get_settings.cache_clear()
    monkeypatch.setenv("VA_BACKEND", "mock")
    monkeypatch.setenv("VA_ENABLE_LANGUAGE_ID", "false")
    monkeypatch.setenv("VA_LOG_JSON", "false")
    get_settings.cache_clear()

    from app.main import create_app

    application = create_app()
    transport = httpx.ASGITransport(app=application)
    yield application, transport
    get_settings.cache_clear()
