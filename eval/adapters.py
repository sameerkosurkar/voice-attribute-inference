"""Dataset adapters for the eval harness.

WHY THERE ARE SEVERAL. Mozilla moved Common Voice off Hugging Face to the
Mozilla Data Collective in October 2025; the `mozilla-foundation/common_voice_*`
dataset repos are now empty. A harness hardcoded to `load_dataset(
"mozilla-foundation/common_voice_17_0")` -- which is what most published
examples still do -- simply fails today. So:

  local  A directory extracted from a Common Voice download (`validated.tsv`
         + `clips/`). This is the canonical path and the one to use for
         numbers you intend to quote: you know exactly which release it is.

  hf     A community mirror on Hugging Face, streamed. Zero-friction, but
         unofficial -- treat it as a smoke test of the harness, not as a
         citable benchmark.

  dir    Any directory of audio named `<gender>_<age>_*.wav`. For evaluating
         on your own call recordings, which is what actually matters: Common
         Voice is read speech from volunteers, and a logistics call is neither.

Each adapter yields the same `Sample`, so `run_eval.py` does not care which
one produced it.
"""

from __future__ import annotations

import csv
import pathlib
import sys
from dataclasses import dataclass
from typing import Iterator

# Common Voice records age as a decade bucket, not a number. We take the
# midpoint of each decade as the point estimate for MAE, which is the best
# available and introduces a known +/-5 y quantisation floor -- reported by the
# harness so it is not mistaken for model error.
CV_AGE_MIDPOINT: dict[str, float] = {
    "teens": 16.0, "twenties": 25.0, "thirties": 35.0, "fourties": 45.0,
    "forties": 45.0, "fifties": 55.0, "sixties": 65.0, "seventies": 75.0,
    "eighties": 85.0, "nineties": 95.0,
}

CV_AGE_BRACKET: dict[str, str] = {
    "teens": "unknown",        # spans 13-19; straddles the 18 boundary
    "twenties": "18-30",
    "thirties": "31-45",
    "fourties": "31-45", "forties": "31-45",
    "fifties": "46-60",
    "sixties": "60+", "seventies": "60+", "eighties": "60+", "nineties": "60+",
}

CV_GENDER: dict[str, str] = {
    "male": "male", "male_masculine": "male",
    "female": "female", "female_feminine": "female",
}


@dataclass(slots=True)
class Sample:
    audio_bytes: bytes
    gender: str | None          # "male" | "female" | None
    age_bracket: str | None     # one of the API brackets, or None
    age_years: float | None     # decade midpoint, for MAE
    source: str


def _map_row(gender_raw: str, age_raw: str) -> tuple[str | None, str | None, float | None]:
    gender = CV_GENDER.get((gender_raw or "").strip().lower())
    age_key = (age_raw or "").strip().lower()
    bracket = CV_AGE_BRACKET.get(age_key)
    years = CV_AGE_MIDPOINT.get(age_key)
    if bracket == "unknown":
        bracket = None          # unusable as ground truth; drop rather than guess
    return gender, bracket, years


# ------------------------------------------------------------- local CV extract
def local_common_voice(
    root: pathlib.Path, limit: int, tsv_name: str = "validated.tsv",
    require_both_labels: bool = True,
) -> Iterator[Sample]:
    tsv = root / tsv_name
    clips = root / "clips"
    if not tsv.exists():
        raise FileNotFoundError(
            f"{tsv} not found.\n"
            "Expected a Common Voice extract: a directory containing "
            "validated.tsv and clips/.\n"
            "Download one from https://commonvoice.mozilla.org/datasets "
            "(now served via Mozilla Data Collective)."
        )

    yielded = 0
    with tsv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if yielded >= limit:
                return
            gender, bracket, years = _map_row(row.get("gender", ""), row.get("age", ""))
            if require_both_labels and (gender is None or bracket is None):
                continue
            if gender is None and bracket is None:
                continue

            path = clips / row.get("path", "")
            if not path.exists():
                continue
            yield Sample(path.read_bytes(), gender, bracket, years, f"cv:{path.name}")
            yielded += 1


# ------------------------------------------------------------------- HF mirror
def hf_common_voice(
    repo: str, config: str, split: str, limit: int,
    require_both_labels: bool = True,
) -> Iterator[Sample]:
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "The hf adapter needs the datasets library:\n"
            "    pip install datasets soundfile\n"
            "Or use --adapter local against a Common Voice extract."
        ) from exc

    print(
        f"[adapter] streaming {repo} ({config}/{split}).\n"
        "[adapter] NOTE: this is an unofficial community mirror. Mozilla moved\n"
        "[adapter] Common Voice to the Mozilla Data Collective in Oct 2025, so\n"
        "[adapter] the official HF repos are empty. Fine for a smoke test; use\n"
        "[adapter] --adapter local for numbers you intend to quote.",
        file=sys.stderr,
    )
    dataset = load_dataset(repo, config, split=split, streaming=True)

    yielded = 0
    for row in dataset:
        if yielded >= limit:
            return
        gender, bracket, years = _map_row(row.get("gender", "") or "",
                                          row.get("age", "") or "")
        if require_both_labels and (gender is None or bracket is None):
            continue
        if gender is None and bracket is None:
            continue

        audio = row.get("audio") or {}
        raw = audio.get("bytes")
        if raw is None:
            # Decoded form: re-encode the array to a WAV so the adapter's
            # output is always encoded bytes, exactly like a real upload.
            array, rate = audio.get("array"), audio.get("sampling_rate")
            if array is None:
                continue
            import numpy as np

            sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
            from scripts.make_sample_audio import wav_bytes

            raw = wav_bytes(np.asarray(array, dtype="float32"), int(rate))

        yield Sample(raw, gender, bracket, years,
                     f"hf:{row.get('path', yielded)}")
        yielded += 1


# ------------------------------------------------------- your own recordings
def labelled_directory(root: pathlib.Path, limit: int) -> Iterator[Sample]:
    """Files named `<gender>_<age>_<anything>.<ext>`, e.g. `male_34_call01.wav`.

    The adapter to use for real evaluation. Common Voice is read speech from
    volunteers in quiet rooms; the model's behaviour on your actual telephony
    traffic is a different question, and this is how you answer it.
    """
    from app.inference.calibration import BRACKETS

    def bracket_of(years: float) -> str | None:
        for label, lo, hi in BRACKETS:
            if lo <= years < hi:
                return label.value
        return None

    yielded = 0
    for path in sorted(root.rglob("*")):
        if yielded >= limit:
            return
        if path.suffix.lower() not in {".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a"}:
            continue
        parts = path.stem.split("_")
        if len(parts) < 2:
            continue
        gender = parts[0].lower()
        if gender not in ("male", "female"):
            continue
        try:
            years = float(parts[1])
        except ValueError:
            continue

        yield Sample(path.read_bytes(), gender, bracket_of(years), years,
                     f"dir:{path.name}")
        yielded += 1
