"""Types and helpers shared by every inference backend.

This module exists to fix a layering inversion. `RawPrediction` used to live in
`calibration.py`, so `base.py` -- the interface every backend implements --
imported from a module that *consumes* backends. And `onnx_backend.py` imported
`_resolve_gender_index`, a private function, from its sibling `audeering.py`.

Both meant the "seam" only looked like a seam: a new backend still had to reach
into an existing implementation to work. Everything a backend needs now lives
here, so implementations depend on this module and on nothing else in the
package.

Dependency direction, deliberately one-way:

    types.py  <-- base.py (Protocol)
        ^
        +------ audeering.py / onnx_backend.py / mock.py / <yours>
        ^
        +------ calibration.py (consumes RawPrediction)
"""

from __future__ import annotations

from dataclasses import dataclass

import structlog

log = structlog.get_logger(__name__)

# Fallback label order, used only when a checkpoint declares no usable
# id2label. See resolve_gender_index() for why this must not be trusted.
FALLBACK_GENDER_ORDER: tuple[str, ...] = ("female", "male", "child")

GENDER_LABELS = frozenset({"female", "male", "child"})


@dataclass(slots=True)
class RawPrediction:
    """What a backend returns, before any calibration.

    Deliberately the *raw* model view -- a continuous age in years and three
    unnormalised class probabilities -- not the API's brackets. Bracketing,
    thresholding and quality shrinkage all happen in `calibration.py`, so a new
    backend never has to know what the HTTP contract looks like, and changing
    the brackets never touches a backend.
    """

    age_years: float
    p_child: float
    p_female: float
    p_male: float
    inference_ms: float = 0.0


def resolve_gender_index(config) -> dict[str, int]:
    """Map "female"/"male"/"child" onto their output positions via id2label.

    Shared by every backend that wraps this checkpoint family, because getting
    it wrong is silent: the audeering model card documents the head as
    `child, female, male` while the checkpoint's config.json declares
    `{0: female, 1: male, 2: child}`. Following the card inverts every
    prediction while still returning well-formed, high-confidence JSON.

    Reading the order from the checkpoint rather than hardcoding a position is
    what makes a differently-ordered checkpoint safe to drop in.
    """
    id2label = getattr(config, "id2label", None) or {}
    resolved: dict[str, int] = {}
    for raw_id, label in id2label.items():
        key = str(label).strip().lower()
        if key in GENDER_LABELS:
            resolved[key] = int(raw_id)

    if set(resolved) == set(GENDER_LABELS):
        return resolved

    log.warning(
        "gender_index_unresolved_using_fallback",
        id2label=id2label,
        fallback=FALLBACK_GENDER_ORDER,
        note="Checkpoint declared no usable id2label. Verify with "
             "scripts/verify_gender_mapping.py before trusting predictions.",
    )
    return fallback_gender_index()


def fallback_gender_index() -> dict[str, int]:
    return {label: i for i, label in enumerate(FALLBACK_GENDER_ORDER)}
