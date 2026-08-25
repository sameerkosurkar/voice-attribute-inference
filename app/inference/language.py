"""Best-effort spoken-language identification (bonus task).

WHY WHISPER-TINY AND NOT A DEDICATED LID MODEL. SpeechBrain's VoxLingua107
ECAPA classifier is the better pure-LID model and covers 107 languages. It is
not what is used here, for two engineering reasons:

  1. Dependency surface. It would add speechbrain + a second model-loading
     idiom to a service that otherwise has exactly one inference stack
     (torch + transformers). One stack is materially easier to pin, patch, and
     reason about than two, and this field is a bonus, not a requirement.
  2. Licence. Whisper is MIT / Apache-clean. Given the main model already has a
     non-commercial licence problem, adding a second encumbered dependency for
     an optional field would be a poor trade.

The cost is accuracy: Whisper's language token is a by-product of an ASR
objective, so it is weaker than a purpose-built LID head, especially on short
or accented audio. That is why the field is documented as best-effort, is
thresholded, and is nullable. See README "Known limitations".

LATENCY. Whisper pads every input to 30 s of mel, so the encoder cost is
constant regardless of clip length -- it does not scale down for a 2 s chunk.
That makes it unsuitable for the critical path, so it runs concurrently with
the age/gender pass under a hard deadline (`VA_LANGUAGE_BUDGET_MS`). If it
misses the deadline the response carries `language: null` and ships on time.
Degrading the optional field to protect the required one is the right call for
a real-time voice agent.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import structlog

from app.config import Settings
from app.inference.language_registry import register_language_backend
from app.schemas import LanguagePrediction

log = structlog.get_logger(__name__)


# Registered so an alternative (VoxLingua107, MMS-LID, a vendor API) can be
# dropped in via VA_LANGUAGE_BACKEND without touching service.py.
@register_language_backend(
    "whisper",
    description="Whisper language token; MIT, reuses the existing torch stack",
)
class LanguageIdentifier:
    """Implements app.inference.base.LanguageBackend."""

    name = "whisper-lid"

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model = None
        self._processor = None
        self._lang_token_ids: dict[int, str] = {}
        self._ready = False
        self._lock = threading.Lock()

    def load(self) -> None:
        with self._lock:
            if self._model is not None:
                return
            import torch
            from transformers import WhisperForConditionalGeneration, WhisperProcessor

            kwargs = {}
            if self._settings.model_cache_dir:
                kwargs["cache_dir"] = self._settings.model_cache_dir

            name = self._settings.language_model
            started = time.perf_counter()
            processor = WhisperProcessor.from_pretrained(name, **kwargs)
            model = WhisperForConditionalGeneration.from_pretrained(name, **kwargs)
            model.eval()

            # Whisper encodes the language as a special decoder token
            # "<|en|>". Build id -> code once so detection is a single decoder
            # step rather than a generate() call.
            lang_token_ids: dict[int, str] = {}
            for token, tid in processor.tokenizer.get_vocab().items():
                if len(token) > 4 and token.startswith("<|") and token.endswith("|>"):
                    code = token[2:-2]
                    if 2 <= len(code) <= 3 and code.isalpha() and code.islower():
                        lang_token_ids[tid] = code

            # Publish-last: identify() checks self._model outside this lock and
            # then reads _lang_token_ids, so the token table must be in place
            # before the model becomes visible.
            self._processor = processor
            self._lang_token_ids = lang_token_ids
            self._ready = bool(lang_token_ids)
            self._model = model

            log.info(
                "language_model_loaded",
                model=name,
                languages=len(lang_token_ids),
                load_ms=round((time.perf_counter() - started) * 1000, 1),
            )
            del torch

    def warmup(self) -> None:
        if self._model is None:
            self.load()
        self.identify(np.zeros(self._settings.target_sample_rate, dtype=np.float32),
                      self._settings.target_sample_rate)

    @property
    def ready(self) -> bool:
        return self._ready

    def identify(self, samples: np.ndarray, sample_rate: int) -> LanguagePrediction | None:
        import torch

        if self._model is None or self._processor is None:
            self.load()

        model, processor, lang_tokens = (
            self._model, self._processor, self._lang_token_ids,
        )
        if not lang_tokens:
            return None

        features = processor.feature_extractor(
            samples, sampling_rate=sample_rate, return_tensors="pt"
        ).input_features

        with torch.inference_mode():
            encoder_out = model.model.encoder(features)
            # One decoder step from the <|startoftranscript|> token: the next
            # token Whisper wants to emit IS the language tag.
            sot = model.config.decoder_start_token_id
            decoder_input = torch.tensor([[sot]], dtype=torch.long)
            logits = model(
                encoder_outputs=encoder_out, decoder_input_ids=decoder_input
            ).logits[0, -1]

            # Restrict the softmax to language tokens. Normalising over the full
            # 51k vocabulary would give an artificially small probability that
            # is not comparable to the other confidences in the response.
            ids = torch.tensor(sorted(lang_tokens), dtype=torch.long)
            probs = torch.softmax(logits[ids], dim=-1)
            best = int(torch.argmax(probs).item())
            code = lang_tokens[int(ids[best].item())]
            confidence = float(probs[best].item())

        del features, encoder_out, logits

        if confidence < self._settings.language_min_confidence:
            return None
        return LanguagePrediction(prediction=code, confidence=round(confidence, 4))
