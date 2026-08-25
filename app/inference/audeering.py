"""wav2vec2 age/gender backend.

MODEL CHOICE -- why audeering/wav2vec2-large-robust-*-ft-age-gender:

  * One model, both attributes. A shared wav2vec2 trunk with two small heads
    means one forward pass for age and gender instead of two. On a CPU-bound
    latency budget that halves the dominant cost, and the attributes are
    correlated anyway, so a shared representation is the right factorisation.

  * Trained on the right kind of variation. aGender + Common Voice + TIMIT +
    VoxCeleb2 spans read speech, telephone speech, and in-the-wild YouTube
    audio. VoxCeleb2 in particular is noisy and far-field, which is closer to a
    truck cab than a clean read corpus would be.

  * `wav2vec2-large-robust` is pretrained on noisy/telephony data specifically
    to survive domain shift -- exactly the failure mode of this application.

  * Age as regression, not classification. This is the property the calibration
    module depends on: a continuous estimate can be integrated over arbitrary
    bracket edges, so if the product later wants 25-35 / 36-50 we re-cut the
    brackets without retraining. A model that classified into someone else's
    buckets could not do that.

  * The 6-layer variant. The 24-layer variant is more accurate but roughly 3x
    the transformer compute. The assignment's constraint is a hard 500 ms on a
    5 s chunk, and 6 layers meets it with room to spare on CPU. Both are
    supported; see the measured table in the README.

LICENCE -- CC-BY-NC-SA-4.0, non-commercial. This is fine for an evaluation
exercise and NOT fine for production. Flagged loudly here, in the README, and
at startup, and mitigated by the AttributeBackend seam. See README.
"""

from __future__ import annotations

import sys
import threading
import time

import numpy as np
import structlog

from app.config import Settings
from app.inference.registry import register_backend
from app.inference.types import RawPrediction, fallback_gender_index, resolve_gender_index

log = structlog.get_logger(__name__)

_NONCOMMERCIAL_MODELS = ("audeering/wav2vec2-large-robust",)



def _build_classes():
    """Defined lazily so importing this module does not import torch.

    Keeps `pytest tests/test_calibration.py` fast and lets the app start with
    VA_BACKEND=mock in an environment with no torch at all.

    The two classes below are the architecture published on the model card --
    the checkpoint has two custom heads on a wav2vec2 trunk, which no stock
    transformers AutoModel class knows how to instantiate. Reproduced here (and
    only here) so the weights load.
    """
    import torch
    import torch.nn as nn
    from transformers.models.wav2vec2.modeling_wav2vec2 import (
        Wav2Vec2Model,
        Wav2Vec2PreTrainedModel,
    )

    class ModelHead(nn.Module):
        def __init__(self, config, num_labels):
            super().__init__()
            self.dense = nn.Linear(config.hidden_size, config.hidden_size)
            self.dropout = nn.Dropout(config.final_dropout)
            self.out_proj = nn.Linear(config.hidden_size, num_labels)

        def forward(self, features, **kwargs):
            x = self.dropout(features)
            x = self.dense(x)
            x = torch.tanh(x)
            x = self.dropout(x)
            return self.out_proj(x)

    class AgeGenderModel(Wav2Vec2PreTrainedModel):
        def __init__(self, config):
            super().__init__(config)
            self.config = config
            self.wav2vec2 = Wav2Vec2Model(config)
            self.age = ModelHead(config, 1)
            self.gender = ModelHead(config, 3)
            self.init_weights()

        def forward(self, input_values):
            hidden = self.wav2vec2(input_values)[0]
            pooled = torch.mean(hidden, dim=1)
            return pooled, self.age(pooled), torch.softmax(self.gender(pooled), dim=1)

    return AgeGenderModel


@register_backend(
    "audeering",
    description="wav2vec2 age+gender, PyTorch (CC-BY-NC-SA-4.0)",
    is_available=lambda settings: True,
    # Wins on macOS, where torch links Apple Accelerate and beats ONNX ~2.3x.
    # Expressed here rather than in the selector, so the selector needs no
    # knowledge of any particular backend.
    auto_priority=lambda settings: 20 if sys.platform.startswith("darwin") else 10,
)
class AudeeringBackend:
    name = "audeering-wav2vec2"

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model = None
        self._extractor = None
        self._ready = False
        self._gender_index: dict[str, int] = {}
        # torch modules are thread-safe for inference, but from_pretrained is
        # not; guard load/warmup rather than every predict call.
        self._load_lock = threading.Lock()

    # ------------------------------------------------------------------ load
    def load(self) -> None:
        with self._load_lock:
            if self._model is not None:
                return
            import torch
            from transformers import Wav2Vec2FeatureExtractor

            # Pin intra-op threads. Left unset, torch grabs every core, and
            # under concurrency the resulting oversubscription makes p95 worse,
            # not better -- threads spend their time fighting each other.
            torch.set_num_threads(self._settings.torch_threads)
            torch.set_grad_enabled(False)

            name = self._settings.age_gender_model
            if any(name.startswith(p) for p in _NONCOMMERCIAL_MODELS):
                log.warning(
                    "noncommercial_model_licence",
                    model=name,
                    licence="cc-by-nc-sa-4.0",
                    note="Not licensed for commercial use. See README 'Model licence'.",
                )

            kwargs = {}
            if self._settings.model_cache_dir:
                kwargs["cache_dir"] = self._settings.model_cache_dir

            started = time.perf_counter()
            # FeatureExtractor, not the full Processor: we never decode text, so
            # the tokenizer is dead weight. All we need is the zero-mean /
            # unit-variance normalisation the model was trained with.
            extractor = Wav2Vec2FeatureExtractor.from_pretrained(name, **kwargs)
            model_cls = _build_classes()
            model = model_cls.from_pretrained(name, **kwargs)
            model.eval()
            gender_index = resolve_gender_index(model.config)

            # PUBLISH-LAST ORDERING. Everything is built into locals first and
            # the "is it loaded?" flag is assigned LAST, because predict() does
            # its cold-path check outside this lock. Assigning self._model
            # before self._gender_index opened a real window in which a
            # concurrent predict() saw a usable model but an empty index and
            # silently fell back to a hardcoded label order -- verified by
            # widening the window. That fallback happens to match this
            # checkpoint, so the bug would produce correct output here and
            # silently inverted labels on a differently-ordered one. Exactly the
            # failure mode this codebase already hit once (see the module
            # docstring), so it gets structural prevention, not a comment.
            self._extractor = extractor
            self._gender_index = gender_index
            self._model = model          # <-- the completion flag; assign last

            log.info(
                "model_loaded",
                model=name,
                load_ms=round((time.perf_counter() - started) * 1000, 1),
                torch_threads=self._settings.torch_threads,
                gender_index=gender_index,
            )

    def warmup(self) -> None:
        if self._model is None:
            self.load()
        started = time.perf_counter()
        # Warm at the size we actually expect, so the allocator reaches its
        # steady-state arena before real traffic rather than during it.
        for seconds in (1.0, 5.0):
            n = int(seconds * self._settings.target_sample_rate)
            self.predict(np.zeros(n, dtype=np.float32), self._settings.target_sample_rate)
        self._ready = True
        log.info("model_warmed", warmup_ms=round((time.perf_counter() - started) * 1000, 1))

    @property
    def ready(self) -> bool:
        return self._ready

    def torch_module(self):
        """The underlying nn.Module, for tooling that must have it.

        Exists so `scripts/export_onnx.py` does not reach into `_model`. A
        private attribute that external code depends on is not private, it is
        just undocumented -- and it silently breaks when the backend changes.
        """
        if self._model is None:
            self.load()
        return self._model

    def gender_index(self) -> dict[str, int]:
        """Resolved label positions. Public because tests and the
        verification script legitimately need to assert on it."""
        if self._model is None:
            self.load()
        return dict(self._gender_index)

    # --------------------------------------------------------------- predict
    def predict(self, samples: np.ndarray, sample_rate: int) -> RawPrediction:
        import torch

        if self._model is None or self._extractor is None:
            self.load()

        # Snapshot the published state once. Reading self._* repeatedly could
        # observe a reload mid-request; a local snapshot is internally
        # consistent by construction.
        model, extractor, idx = self._model, self._extractor, self._gender_index

        started = time.perf_counter()
        features = extractor(
            samples, sampling_rate=sample_rate, return_tensors="np"
        )["input_values"][0]
        tensor = torch.from_numpy(np.ascontiguousarray(features, dtype=np.float32)).reshape(1, -1)

        with torch.inference_mode():
            _pooled, age_logits, gender_probs = model(tensor)

        # The age head is a linear regression onto ~[0, 1] representing 0-100
        # years. It is unbounded, so clamp: an out-of-domain clip can push it
        # past 1.0, and a 130-year-old caller is not a prediction we want to
        # emit downstream.
        age_years = float(np.clip(age_logits[0][0].item(), 0.0, 1.2) * 100.0)
        probs = gender_probs[0].detach().cpu().numpy()
        idx = idx or fallback_gender_index()

        try:
            return RawPrediction(
                age_years=age_years,
                p_child=float(probs[idx["child"]]),
                p_female=float(probs[idx["female"]]),
                p_male=float(probs[idx["male"]]),
                inference_ms=(time.perf_counter() - started) * 1000.0,
            )
        finally:
            # Do not let a caller's activations sit in memory any longer than
            # the request. See PRIVACY.md.
            del tensor, features
