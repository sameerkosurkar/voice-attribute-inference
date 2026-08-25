"""Turning raw model outputs into calibrated, bracketed predictions.

This module is where most of the actual thinking in the service lives, so the
reasoning is written down rather than implied.

THE PROBLEM. The model gives a *continuous* age estimate. The API asks for one
of four *brackets*. The naive implementation is `bracket_of(age)` with the
softmax'd gender probability as the confidence and some made-up number for age.
That is wrong in a way that matters: a predicted age of 45.4 and a predicted
age of 22.0 both land in a bracket, but only one of them is a real answer. The
first is a coin flip between "31-45" and "46-60" that the caller is entitled to
know about.

THE APPROACH. Treat the regressor's output as the mean of a predictive
distribution rather than a point. The source paper reports MAE 7.1-10.8 years,
so model the error as N(0, sigma^2) with sigma ~= 8 years, and integrate that
Gaussian over each bracket:

    P(bracket) = Phi((hi - age)/sigma) - Phi((lo - age)/sigma)

renormalised over the four brackets. Confidence is the winning bracket's mass.
This gives, for free, exactly the behaviour you want:

    age 24.0  -> 18-30 at 0.72   (comfortably inside a bracket)
    age 45.4  -> 31-45 at 0.38   (sitting on a boundary -- honest coin flip)
    age 70.0  -> 60+   at 0.86   (open-ended bracket, high mass)

and 45.4 then falls below `age_min_confidence` and reports "unknown", which is
the correct answer.

Gaussian error is an assumption, and a slightly wrong one -- real age-estimation
error is heteroscedastic (worse for the young and the old) and left-skewed at
the ends. It is defensible as a first approximation and, critically, it is
*measurable*: eval/run_eval.py fits sigma empirically and reports ECE, so the
assumption gets checked rather than trusted. See README "Known limitations".

THE CHILD CASE. The model's gender head is 3-way: child / female / male, not
male / female. A child is not a gender, and a child's age is under 18, which is
outside every bracket the API defines. Silently folding "child" into 18-30, or
into whichever of male/female scores higher, would produce a confident wrong
answer for exactly the caller you most want to handle carefully. So a dominant
child probability yields unknown/unknown with the reason recorded.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from app.config import Settings
from app.inference.types import RawPrediction
from app.schemas import AgeBracket, AgePrediction, Gender, GenderPrediction

__all__ = [
    "BRACKETS", "CalibratedPrediction", "RawPrediction",
    "bracket_probabilities", "calibrate", "unknown_prediction",
]

# (label, low_inclusive, high_exclusive). The API's brackets are not contiguous
# as written (18-30 then 31-45), so we use 30.5 / 45.5 / 60.5 as the real
# decision boundaries: age is continuous, and a 30.6-year-old has to go
# somewhere.
BRACKETS: tuple[tuple[AgeBracket, float, float], ...] = (
    (AgeBracket.A18_30, 18.0, 30.5),
    (AgeBracket.A31_45, 30.5, 45.5),
    (AgeBracket.A46_60, 45.5, 60.5),
    (AgeBracket.A60_PLUS, 60.5, 120.0),
)

_SQRT2 = math.sqrt(2.0)


@dataclass(slots=True)
class CalibratedPrediction:
    gender: GenderPrediction
    age_bracket: AgePrediction
    age_years: float
    bracket_probabilities: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / _SQRT2))


def bracket_probabilities(age_years: float, sigma_years: float) -> dict[AgeBracket, float]:
    """Integrate N(age_years, sigma^2) over each bracket, renormalised.

    Renormalisation matters: mass below 18 is real (the model does see children
    and teenagers) but the API has no bracket for it, so we condition on the
    speaker being an adult. The child head is what actually catches under-18s;
    this is just bookkeeping so the four probabilities sum to 1.
    """
    sigma = max(float(sigma_years), 0.5)
    masses: dict[AgeBracket, float] = {}
    for label, lo, hi in BRACKETS:
        masses[label] = _norm_cdf((hi - age_years) / sigma) - _norm_cdf((lo - age_years) / sigma)

    total = sum(masses.values())
    if total <= 1e-9:
        # Degenerate: predicted age far outside [18, 120]. Put the mass on the
        # nearest bracket rather than dividing by ~0.
        nearest = BRACKETS[0][0] if age_years < 18.0 else BRACKETS[-1][0]
        return {label: (1.0 if label is nearest else 0.0) for label, _, _ in BRACKETS}
    return {label: mass / total for label, mass in masses.items()}


def calibrate(
    raw: RawPrediction,
    settings: Settings,
    *,
    confidence_factor: float = 1.0,
) -> CalibratedPrediction:
    """Apply bracket integration, the child rule, quality shrinkage, and the
    unknown thresholds. `confidence_factor` comes from the quality gate."""
    notes: list[str] = []

    # ------------------------------------------------------------- gender ---
    total = raw.p_child + raw.p_female + raw.p_male
    p_child = raw.p_child / total if total > 0 else 0.0
    p_female = raw.p_female / total if total > 0 else 0.0
    p_male = raw.p_male / total if total > 0 else 0.0

    child_dominant = p_child >= max(p_female, p_male)

    # Confidence in the binary decision, conditioned on the speaker being an
    # adult. The child mass is then charged against it separately below --
    # otherwise a 0.6-child / 0.3-male / 0.1-female clip would report male at
    # 0.75, which overstates the case badly.
    adult_mass = p_female + p_male
    if adult_mass > 0:
        gender_label = Gender.MALE if p_male >= p_female else Gender.FEMALE
        gender_conf = max(p_male, p_female) / adult_mass
    else:
        gender_label, gender_conf = Gender.UNKNOWN, 0.0

    gender_conf *= adult_mass          # charge the child mass
    gender_conf *= confidence_factor   # charge the audio quality

    if child_dominant:
        notes.append(
            "child voice detected; gender and age bracket are not reported for "
            "speakers the model places under 18"
        )
        gender_label, gender_conf = Gender.UNKNOWN, 0.0

    if gender_conf < settings.gender_min_confidence:
        if gender_label is not Gender.UNKNOWN:
            notes.append(
                f"gender confidence {gender_conf:.2f} below threshold "
                f"{settings.gender_min_confidence:.2f}"
            )
        gender_label = Gender.UNKNOWN

    # ---------------------------------------------------------------- age ---
    probs = bracket_probabilities(raw.age_years, settings.age_sigma_years)
    age_label = max(probs, key=probs.__getitem__)
    age_conf = probs[age_label] * confidence_factor

    if child_dominant and settings.child_age_bracket_unknown:
        age_label, age_conf = AgeBracket.UNKNOWN, 0.0
    elif age_conf < settings.age_min_confidence:
        notes.append(
            f"predicted age {raw.age_years:.1f}y sits near a bracket boundary "
            f"(best bracket only {age_conf:.2f})"
        )
        age_label = AgeBracket.UNKNOWN

    return CalibratedPrediction(
        gender=GenderPrediction(
            prediction=gender_label,
            confidence=round(_clamp(gender_conf), 4),
        ),
        age_bracket=AgePrediction(
            prediction=age_label,
            confidence=round(_clamp(age_conf), 4),
        ),
        age_years=round(raw.age_years, 2),
        bracket_probabilities={k.value: round(v, 4) for k, v in probs.items()},
        notes=notes,
    )


def unknown_prediction(reason: str | None = None) -> CalibratedPrediction:
    """The response for audio we refuse to guess from."""
    return CalibratedPrediction(
        gender=GenderPrediction(prediction=Gender.UNKNOWN, confidence=0.0),
        age_bracket=AgePrediction(prediction=AgeBracket.UNKNOWN, confidence=0.0),
        age_years=0.0,
        bracket_probabilities={},
        notes=[reason] if reason else [],
    )


def _clamp(value: float) -> float:
    if value != value:  # NaN
        return 0.0
    return max(0.0, min(1.0, float(value)))
