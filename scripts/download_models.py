#!/usr/bin/env python3
"""Pre-fetch model weights into the image at BUILD time.

The container must not download a gigabyte of weights on first request. Baking
them in buys three things that matter operationally:

  * Cold start is bounded and predictable. A pod that has to fetch weights on
    boot has a startup time set by someone else's CDN.
  * The runtime needs no network at all (HF_HUB_OFFLINE=1), which is both an
    availability property and a security one -- a container that cannot make
    outbound calls cannot exfiltrate caller audio.
  * The image is a reproducible artefact. The weights are pinned by whatever
    the hub served at build time, not by whatever it serves at deploy time.

Run explicitly during the build so a fetch failure fails the BUILD loudly,
rather than surfacing as a mysterious 30-second first request in production.
"""

from __future__ import annotations

import os
import sys


def main() -> int:
    age_gender = os.environ.get(
        "VA_AGE_GENDER_MODEL", "audeering/wav2vec2-large-robust-6-ft-age-gender"
    )
    language = os.environ.get("VA_LANGUAGE_MODEL", "openai/whisper-tiny")
    want_language = os.environ.get("VA_ENABLE_LANGUAGE_ID", "true").lower() != "false"

    from transformers import (
        Wav2Vec2FeatureExtractor,
        WhisperForConditionalGeneration,
        WhisperProcessor,
    )

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from app.inference.audeering import _build_classes

    print(f"[models] fetching age/gender model: {age_gender}")
    Wav2Vec2FeatureExtractor.from_pretrained(age_gender)
    _build_classes().from_pretrained(age_gender)
    print("[models]   ok")

    if want_language:
        print(f"[models] fetching language-id model: {language}")
        WhisperProcessor.from_pretrained(language)
        WhisperForConditionalGeneration.from_pretrained(language)
        print("[models]   ok")
    else:
        print("[models] language id disabled; skipping")

    # Silero VAD ships inside its wheel, but touch it here so a packaging
    # change that reintroduced a download would break the build, not production.
    print("[models] checking Silero VAD (bundled in the wheel)")
    from silero_vad import load_silero_vad

    load_silero_vad()
    print("[models]   ok")

    print("[models] all weights cached in the image")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
