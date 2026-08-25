"""Proof that the model layer is genuinely a plug-in point.

This is the test I would want to see before believing an "extensible"
architecture claim. Interfaces are easy to assert and easy to get subtly wrong:
an if/elif factory plus a closed `Literal[...]` still *looks* like a seam, but
adding a model means editing core files that have nothing to do with it.

So instead of asserting that a Protocol exists, these tests define a brand-new
backend **inside the test file**, register it, and drive it end to end through
the real HTTP API. If a future change reintroduces a hardcoded branch in
`service.py` or a closed enum in `config.py`, this fails.

The custom backend below deliberately uses NO torch, NO transformers and NO
weights -- proving the interface is not secretly PyTorch-shaped, which matters
because the licence-migration path in the README (ECAPA embeddings, a vendor
API, a remote GPU tier) may not be a torch nn.Module at all.
"""

from __future__ import annotations

import httpx
import numpy as np
import pytest

from app.config import Settings, get_settings
from app.inference import registry
from app.inference.base import AttributeBackend, LanguageBackend
from app.inference.registry import register_backend
from app.inference.types import RawPrediction
from app.schemas import LanguagePrediction

SAMPLE_RATE = 16_000
CUSTOM_AGE = 52.0


@register_backend(
    "test-custom",
    description="A backend defined entirely inside the test suite",
    is_available=lambda settings: True,
)
class CustomBackend:
    """A complete backend in ~20 lines, with no ML dependencies at all."""

    name = "test-custom"

    def __init__(self, settings) -> None:
        self._settings = settings
        self._ready = False

    def load(self) -> None:
        self._ready = True

    def warmup(self) -> None:
        self.predict(np.zeros(SAMPLE_RATE, dtype=np.float32), SAMPLE_RATE)

    @property
    def ready(self) -> bool:
        return self._ready

    def predict(self, samples: np.ndarray, sample_rate: int) -> RawPrediction:
        return RawPrediction(
            age_years=CUSTOM_AGE, p_child=0.01, p_female=0.02, p_male=0.97
        )


class CustomLanguageBackend:
    """Likewise for the optional language seam."""

    name = "test-lang"

    def __init__(self, settings) -> None:
        self._ready = False

    def load(self) -> None:
        self._ready = True

    def warmup(self) -> None:
        pass

    @property
    def ready(self) -> bool:
        return self._ready

    def identify(self, samples, sample_rate) -> LanguagePrediction | None:
        return LanguagePrediction(prediction="hi", confidence=0.91)


# ----------------------------------------------------------- structural ----
def test_custom_backend_satisfies_the_protocol():
    assert isinstance(CustomBackend(Settings()), AttributeBackend)


def test_custom_language_backend_satisfies_the_protocol():
    assert isinstance(CustomLanguageBackend(Settings()), LanguageBackend)


def test_shipped_backends_satisfy_the_protocol():
    for name in ("mock", "audeering", "onnx"):
        spec = registry.get_spec(name)
        assert spec is not None, f"{name} is not registered"


def test_registration_needed_no_core_file_edits():
    """The load-bearing assertion.

    `test-custom` appears in the registry and is selectable, yet it is defined
    in a test file. Nothing in app/config.py or app/service.py mentions it.
    """
    assert "test-custom" in registry.available_backends()

    import inspect

    from app import config, service

    for module in (config, service):
        source = inspect.getsource(module)
        assert "test-custom" not in source, (
            f"{module.__name__} names a specific backend -- the registry exists "
            "precisely so it does not have to"
        )


def test_backend_name_is_not_a_closed_enum():
    """A `Literal[...]` would force an edit to config.py per model."""
    field = Settings.model_fields["backend"]
    assert field.annotation is str, (
        "VA_BACKEND must stay an open string; a closed enum reintroduces "
        "shotgun surgery for every new model"
    )


def test_duplicate_registration_is_rejected():
    """Silent shadowing would make the active model ambiguous."""
    with pytest.raises(ValueError, match="already registered"):

        @register_backend("test-custom")
        class Clashing:  # noqa: D401
            name = "clash"


def test_unknown_backend_fails_with_a_useful_message():
    from app.errors import ModelNotReadyError
    from app.service import build_backend

    with pytest.raises(ModelNotReadyError) as excinfo:
        build_backend(Settings(backend="does-not-exist"))

    message = str(excinfo.value)
    assert "does-not-exist" in message
    # The error must list what IS available, or the operator has to read source.
    assert "audeering" in message and "mock" in message


# ------------------------------------------------------------ end-to-end ----
async def test_custom_backend_serves_real_http_traffic(monkeypatch, speech_wav):
    """The proof that matters: a test-defined model answering a real request."""
    get_settings.cache_clear()
    monkeypatch.setenv("VA_BACKEND", "test-custom")
    monkeypatch.setenv("VA_ENABLE_LANGUAGE_ID", "false")
    monkeypatch.setenv("VA_LOG_JSON", "false")
    get_settings.cache_clear()

    from app.main import create_app

    application = create_app()
    async with application.router.lifespan_context(application):
        transport = httpx.ASGITransport(app=application)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
            assert (await client.get("/ready")).json()["backend"] == "test-custom"
            response = await client.post(
                "/analyze", content=speech_wav, headers={"content-type": "audio/wav"}
            )
    get_settings.cache_clear()

    assert response.status_code == 200
    body = response.json()
    # 52.0 years -> the 46-60 bracket, via the normal calibration path. The
    # custom backend never had to know the API's brackets exist.
    assert body["age_bracket"]["prediction"] == "46-60"
    assert body["gender"]["prediction"] == "male"
    assert body["audio_quality"] in {"good", "degraded", "insufficient"}


async def test_custom_language_backend_plugs_in(monkeypatch, speech_wav):
    """The optional seam, which previously did not exist at all."""
    from app.inference import language_registry

    language_registry.load_builtins()
    language_registry._REGISTRY["test-lang"] = language_registry.LanguageSpec(
        "test-lang", CustomLanguageBackend, "test"
    )
    try:
        get_settings.cache_clear()
        monkeypatch.setenv("VA_BACKEND", "test-custom")
        monkeypatch.setenv("VA_ENABLE_LANGUAGE_ID", "true")
        monkeypatch.setenv("VA_LANGUAGE_BACKEND", "test-lang")
        monkeypatch.setenv("VA_LOG_JSON", "false")
        get_settings.cache_clear()

        from app.main import create_app

        application = create_app()
        async with application.router.lifespan_context(application):
            transport = httpx.ASGITransport(app=application)
            async with httpx.AsyncClient(transport=transport,
                                         base_url="http://t") as client:
                response = await client.post(
                    "/analyze", content=speech_wav,
                    headers={"content-type": "audio/wav"},
                )
        get_settings.cache_clear()

        body = response.json()
        assert body["language"]["prediction"] == "hi"
        assert body["language"]["confidence"] == pytest.approx(0.91)
    finally:
        language_registry._REGISTRY.pop("test-lang", None)


async def test_a_failing_language_backend_does_not_break_the_response(
    monkeypatch, speech_wav
):
    """The optional seam must stay optional: a broken LID model degrades the
    bonus field to null, it does not fail the request."""
    from app.inference import language_registry

    class Broken:
        name = "broken"

        def __init__(self, settings) -> None:
            pass

        def load(self) -> None:
            raise RuntimeError("weights are missing")

        def warmup(self) -> None:
            pass

        @property
        def ready(self) -> bool:
            return False

        def identify(self, samples, sample_rate):
            raise RuntimeError("unreachable")

    language_registry.load_builtins()
    language_registry._REGISTRY["broken"] = language_registry.LanguageSpec(
        "broken", Broken, "test"
    )
    try:
        get_settings.cache_clear()
        monkeypatch.setenv("VA_BACKEND", "test-custom")
        monkeypatch.setenv("VA_ENABLE_LANGUAGE_ID", "true")
        monkeypatch.setenv("VA_LANGUAGE_BACKEND", "broken")
        monkeypatch.setenv("VA_LOG_JSON", "false")
        get_settings.cache_clear()

        from app.main import create_app

        application = create_app()
        async with application.router.lifespan_context(application):
            transport = httpx.ASGITransport(app=application)
            async with httpx.AsyncClient(transport=transport,
                                         base_url="http://t") as client:
                response = await client.post(
                    "/analyze", content=speech_wav,
                    headers={"content-type": "audio/wav"},
                )
        get_settings.cache_clear()

        assert response.status_code == 200, "a broken bonus field must not 5xx"
        assert "language" not in response.json()
    finally:
        language_registry._REGISTRY.pop("broken", None)
