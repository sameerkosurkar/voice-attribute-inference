"""The ONNX backend must be numerically identical to the PyTorch one.

Running two backends is only defensible if they are interchangeable. If ONNX
drifted, latency would be measured on one path while accuracy was measured on
the other, and the eval numbers would describe a model that never serves
traffic. So this asserts agreement directly, at several input lengths --
a dynamic axis that only works at the length it was exported with is the
classic ONNX export bug.

Skipped when no export is present (a plain checkout); it runs in the image,
where the export is baked in, and after `python scripts/export_onnx.py`.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.config import Settings
from app.inference import onnx_backend

pytestmark = pytest.mark.slow

SAMPLE_RATE = 16_000


@pytest.fixture(scope="module")
def settings() -> Settings:
    return Settings(enable_language_id=False)


@pytest.fixture(scope="module")
def onnx(settings):
    if not onnx_backend.export_available(settings):
        pytest.skip(
            f"no ONNX export at {onnx_backend.export_path(settings)}; "
            "run scripts/export_onnx.py"
        )
    backend = onnx_backend.OnnxBackend(settings)
    backend.load()
    backend.warmup()
    return backend


@pytest.fixture(scope="module")
def torch_backend(settings):
    from app.inference.audeering import AudeeringBackend

    backend = AudeeringBackend(settings)
    backend.load()
    backend.warmup()
    return backend


@pytest.mark.parametrize("seconds", [1.0, 2.5, 5.0, 10.0])
def test_backends_agree(onnx, torch_backend, seconds):
    rng = np.random.default_rng(int(seconds * 10))
    samples = (rng.standard_normal(int(seconds * SAMPLE_RATE)) * 0.1).astype(np.float32)

    a = torch_backend.predict(samples, SAMPLE_RATE)
    b = onnx.predict(samples, SAMPLE_RATE)

    assert a.age_years == pytest.approx(b.age_years, abs=0.1)
    assert a.p_male == pytest.approx(b.p_male, abs=1e-3)
    assert a.p_female == pytest.approx(b.p_female, abs=1e-3)
    assert a.p_child == pytest.approx(b.p_child, abs=1e-3)


def test_backends_resolve_the_same_label_order(onnx, torch_backend):
    assert onnx.gender_index() == torch_backend.gender_index()


def test_onnx_backend_reports_ready(onnx):
    assert onnx.ready is True
    assert onnx.name == "audeering-onnx"


def test_missing_export_is_a_clear_error(settings, monkeypatch, tmp_path):
    monkeypatch.setenv("VA_ONNX_PATH", str(tmp_path / "nope.onnx"))
    backend = onnx_backend.OnnxBackend(settings)
    with pytest.raises(FileNotFoundError, match="export_onnx"):
        backend.load()
