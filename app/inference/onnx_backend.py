"""ONNX Runtime backend -- same weights, faster kernels on Linux.

WHY THIS EXISTS, WITH THE MEASUREMENTS THAT JUSTIFY IT.

Same checkpoint, same 5 s input, 4 threads, measured on this machine:

    platform                         PyTorch    ONNX Runtime
    macOS arm64 (host)                  89 ms         209 ms
    Linux arm64 (container)            453 ms         213 ms

ONNX Runtime is 2.1x FASTER in the container and 2.3x SLOWER on the macOS
host. That inversion is not noise -- macOS PyTorch links Apple's Accelerate
framework, whose GEMM kernels are excellent on Apple silicon, while the
manylinux aarch64 PyTorch wheel falls back to generic kernels. ONNX Runtime is
roughly platform-independent, so it wins wherever PyTorch's BLAS is weak.

The lesson worth stating: "ONNX is faster" is not a fact about ONNX, it is a
fact about which BLAS your PyTorch happened to link. It has to be measured per
deployment target, which is why both backends ship and `VA_BACKEND=auto` picks
by what is actually present rather than by folklore.

The image bakes an ONNX export at build time, so containers get this path by
default; a local checkout without the export uses PyTorch. Numerical parity is
asserted in tests/test_onnx_parity.py, not assumed.
"""

from __future__ import annotations

import os
import pathlib
import sys
import threading
import time

import numpy as np
import structlog

from app.config import Settings
from app.inference.registry import register_backend
from app.inference.types import RawPrediction, resolve_gender_index

log = structlog.get_logger(__name__)

DEFAULT_EXPORT_PATH = "/opt/models/onnx/age_gender.onnx"


def export_path(settings: Settings) -> pathlib.Path:
    return pathlib.Path(os.environ.get("VA_ONNX_PATH", DEFAULT_EXPORT_PATH))


def export_available(settings: Settings) -> bool:
    return export_path(settings).is_file()


@register_backend(
    "onnx",
    description="wav2vec2 age+gender, ONNX Runtime (same weights, faster on Linux)",
    is_available=lambda settings: export_available(settings),
    # Wins everywhere except macOS: 2.1x faster than torch in the Linux
    # container, 2.3x slower on a macOS host. See the module docstring.
    auto_priority=lambda settings: 5 if sys.platform.startswith("darwin") else 30,
)
class OnnxBackend:
    name = "audeering-onnx"

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._session = None
        self._extractor = None
        self._ready = False
        self._gender_index: dict[str, int] = {}
        self._input_name = "input_values"
        self._lock = threading.Lock()

    def load(self) -> None:
        with self._lock:
            if self._session is not None:
                return
            import onnxruntime as ort
            from transformers import Wav2Vec2FeatureExtractor

            path = export_path(self._settings)
            if not path.is_file():
                raise FileNotFoundError(
                    f"No ONNX export at {path}. Build the image (which exports it) "
                    f"or run: python scripts/export_onnx.py --out {path}"
                )

            options = ort.SessionOptions()
            # Same pinning rationale as the torch backend: oversubscribing cores
            # under concurrency makes p95 worse, not better.
            options.intra_op_num_threads = self._settings.torch_threads
            options.inter_op_num_threads = 1
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            started = time.perf_counter()
            session = ort.InferenceSession(
                str(path), options, providers=["CPUExecutionProvider"]
            )
            input_name = session.get_inputs()[0].name

            name = self._settings.age_gender_model
            kwargs = {}
            if self._settings.model_cache_dir:
                kwargs["cache_dir"] = self._settings.model_cache_dir
            extractor = Wav2Vec2FeatureExtractor.from_pretrained(name, **kwargs)

            # Label order is resolved from the same config.json the torch
            # backend reads -- see the comment there about the model card
            # disagreeing with the checkpoint.
            from transformers import AutoConfig

            gender_index = resolve_gender_index(
                AutoConfig.from_pretrained(name, **kwargs)
            )

            # Publish-last, same reasoning as the torch backend: predict()
            # checks self._session outside this lock, so it must be the final
            # assignment.
            self._input_name = input_name
            self._extractor = extractor
            self._gender_index = gender_index
            self._session = session

            log.info(
                "onnx_model_loaded",
                path=str(path),
                load_ms=round((time.perf_counter() - started) * 1000, 1),
                intra_op_threads=self._settings.torch_threads,
                gender_index=gender_index,
            )

    def warmup(self) -> None:
        if self._session is None:
            self.load()
        for seconds in (1.0, 5.0):
            n = int(seconds * self._settings.target_sample_rate)
            self.predict(np.zeros(n, dtype=np.float32), self._settings.target_sample_rate)
        self._ready = True

    @property
    def ready(self) -> bool:
        return self._ready

    def gender_index(self) -> dict[str, int]:
        if self._session is None:
            self.load()
        return dict(self._gender_index)

    def predict(self, samples: np.ndarray, sample_rate: int) -> RawPrediction:
        if self._session is None or self._extractor is None:
            self.load()

        # One internally-consistent snapshot; see the torch backend.
        session, extractor = self._session, self._extractor
        idx, input_name = self._gender_index, self._input_name

        started = time.perf_counter()
        features = extractor(
            samples, sampling_rate=sample_rate, return_tensors="np"
        )["input_values"][0]
        batch = np.ascontiguousarray(features, dtype=np.float32).reshape(1, -1)

        # Outputs are (pooled_hidden, age_logits, gender_probs) -- the gender
        # head has its softmax inside the graph, matching the torch module.
        # ort.InferenceSession.run() is documented thread-safe, so one shared
        # session serves every worker thread.
        _hidden, age_logits, gender_probs = session.run(None, {input_name: batch})

        age_years = float(np.clip(age_logits[0][0], 0.0, 1.2) * 100.0)
        probs = gender_probs[0]

        return RawPrediction(
            age_years=age_years,
            p_child=float(probs[idx["child"]]),
            p_female=float(probs[idx["female"]]),
            p_male=float(probs[idx["male"]]),
            inference_ms=(time.perf_counter() - started) * 1000.0,
        )
