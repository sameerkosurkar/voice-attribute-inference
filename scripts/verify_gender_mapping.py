#!/usr/bin/env python3
"""Verify that the gender head's output positions are interpreted correctly.

WHY THIS SCRIPT EXISTS. The audeering model card annotates its gender head as
`child, female, male`, while the checkpoint's own config.json declares
`id2label = {0: female, 1: male, 2: child}`. Those disagree. Trusting the
comment inverts every prediction -- and it does so *silently*: the service still
returns well-formed JSON with high confidence, just with the labels swapped.
No unit test on shapes or schemas would catch it.

The only way to catch a label-permutation bug is to run known-gender audio
through the model and check the labels come back right. That is this script.

Run it whenever VA_AGE_GENDER_MODEL changes.

    python scripts/verify_gender_mapping.py                 # macOS 'say' voices
    python scripts/verify_gender_mapping.py --dir ./clips   # your own, named
                                                            # <label>_<name>.wav
"""

from __future__ import annotations

import argparse
import pathlib
import shutil
import subprocess
import sys
import tempfile
import wave

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

SENTENCE = (
    "Hi, this is the driver calling about the delivery for order four seven two. "
    "I am running about twenty minutes late because of traffic on the highway."
)

# macOS system voices with unambiguous intended gender. Not real speakers, but
# real speech signals -- good enough to detect a permuted label, which is a
# gross error, not a subtle one.
SAY_VOICES = {
    "Daniel": "male", "Alex": "male", "Fred": "male",
    "Samantha": "female", "Karen": "female", "Moira": "female",
}


def _read_wav(path: pathlib.Path) -> np.ndarray:
    with wave.open(str(path)) as handle:
        frames = handle.readframes(handle.getnframes())
        data = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    return data


def _build_say_corpus(workdir: pathlib.Path) -> list[tuple[str, str, pathlib.Path]]:
    if not shutil.which("say") or not shutil.which("ffmpeg"):
        return []
    corpus = []
    for voice, label in SAY_VOICES.items():
        aiff, wav = workdir / f"{voice}.aiff", workdir / f"{voice}.wav"
        try:
            subprocess.run(["say", "-v", voice, "-o", str(aiff), SENTENCE],
                           check=True, capture_output=True, timeout=60)
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(aiff),
                            "-ac", "1", "-ar", "16000", str(wav)],
                           check=True, capture_output=True, timeout=60)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            continue
        corpus.append((voice, label, wav))
    return corpus


def _build_dir_corpus(directory: pathlib.Path) -> list[tuple[str, str, pathlib.Path]]:
    corpus = []
    for path in sorted(directory.glob("*.wav")):
        label = path.stem.split("_", 1)[0].lower()
        if label in ("male", "female", "child"):
            corpus.append((path.stem, label, path))
    return corpus


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", type=pathlib.Path,
                        help="Directory of <label>_<name>.wav files (16 kHz mono).")
    parser.add_argument("--min-accuracy", type=float, default=0.8)
    args = parser.parse_args()

    from app.config import get_settings
    from app.inference.audeering import AudeeringBackend

    settings = get_settings()
    backend = AudeeringBackend(settings)
    backend.load()

    with tempfile.TemporaryDirectory() as tmp:
        corpus = (
            _build_dir_corpus(args.dir) if args.dir
            else _build_say_corpus(pathlib.Path(tmp))
        )
        if not corpus:
            print("No verification audio available.")
            print("On macOS this uses 'say' + ffmpeg; elsewhere pass --dir with "
                  "files named <label>_<name>.wav.")
            return 2

        print(f"model        : {settings.age_gender_model}")
        print(f"gender index : {backend.gender_index()}")
        print(f"{'sample':12s} {'expect':8s} {'got':8s} {'female':>7s} {'male':>7s} "
              f"{'child':>7s} {'age':>6s}")
        print("-" * 62)

        correct = 0
        for name, expected, path in corpus:
            samples = _read_wav(path)[: 16_000 * 5]
            raw = backend.predict(samples, 16_000)
            got = max(
                (("female", raw.p_female), ("male", raw.p_male), ("child", raw.p_child)),
                key=lambda kv: kv[1],
            )[0]
            ok = got == expected
            correct += ok
            print(f"{name:12s} {expected:8s} {got:8s} {raw.p_female:7.3f} "
                  f"{raw.p_male:7.3f} {raw.p_child:7.3f} {raw.age_years:6.1f} "
                  f"{'OK' if ok else '<-- MISMATCH'}")

    accuracy = correct / len(corpus)
    print("-" * 62)
    print(f"{correct}/{len(corpus)} correct ({accuracy:.0%})")

    if accuracy < args.min_accuracy:
        print("\nFAIL: the gender labels look permuted or the checkpoint is wrong.")
        print("Check config.id2label against how RawPrediction is populated in")
        print("app/inference/audeering.py.")
        return 1
    print("\nPASS: gender label mapping is correct.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
