"""Calibration: turning a continuous age estimate into a bracket + confidence.

These are pure-function tests with no model and no audio, so they encode the
*intended reasoning* directly. If someone later replaces the Gaussian bracket
integration with `bracket_of(age)`, these fail loudly.
"""

from __future__ import annotations

import math

import pytest

from app.config import Settings
from app.inference.calibration import (
    BRACKETS,
    RawPrediction,
    bracket_probabilities,
    calibrate,
    unknown_prediction,
)
from app.schemas import AgeBracket, Gender


def _settings(**overrides) -> Settings:
    base = dict(backend="mock", enable_language_id=False)
    base.update(overrides)
    return Settings(**base)


# --------------------------------------------------------- bracket integration
def test_bracket_probabilities_form_a_distribution():
    for age in (18.0, 22.0, 30.5, 38.0, 45.5, 52.0, 61.0, 75.0):
        probs = bracket_probabilities(age, 8.0)
        assert set(probs) == {label for label, _, _ in BRACKETS}
        assert math.isclose(sum(probs.values()), 1.0, rel_tol=1e-9)
        assert all(0.0 <= p <= 1.0 for p in probs.values())


def test_age_well_inside_a_bracket_is_confident():
    probs = bracket_probabilities(24.0, 8.0)
    assert max(probs, key=probs.__getitem__) is AgeBracket.A18_30
    assert probs[AgeBracket.A18_30] > 0.6


def test_age_on_a_boundary_splits_between_neighbours():
    """The whole point of integrating rather than thresholding.

    A predicted 45.5 sits exactly on the 31-45 / 46-60 edge. An honest system
    reports roughly a coin flip; a thresholding one would report a confident
    bracket and be wrong half the time.
    """
    probs = bracket_probabilities(45.5, 8.0)
    near = probs[AgeBracket.A31_45] + probs[AgeBracket.A46_60]
    assert near > 0.85, "boundary mass should sit in the two adjacent brackets"
    assert abs(probs[AgeBracket.A31_45] - probs[AgeBracket.A46_60]) < 0.05
    assert max(probs.values()) < 0.5, "no bracket deserves confidence on a boundary"


def test_wider_sigma_lowers_peak_confidence():
    """Confidence must track the model's actual error, not be a constant."""
    tight = max(bracket_probabilities(24.0, 4.0).values())
    loose = max(bracket_probabilities(24.0, 16.0).values())
    assert tight > loose


def test_open_ended_bracket_accumulates_upper_mass():
    probs = bracket_probabilities(75.0, 8.0)
    assert probs[AgeBracket.A60_PLUS] > 0.9


def test_age_outside_the_supported_range_degrades_gracefully():
    for age in (-5.0, 0.0, 200.0):
        probs = bracket_probabilities(age, 8.0)
        assert math.isclose(sum(probs.values()), 1.0, rel_tol=1e-6)


# --------------------------------------------------------------------- gender
def test_confident_male_is_reported():
    result = calibrate(
        RawPrediction(age_years=35.0, p_child=0.01, p_female=0.04, p_male=0.95),
        _settings(),
    )
    assert result.gender.prediction is Gender.MALE
    assert result.gender.confidence > 0.9


def test_ambiguous_gender_becomes_unknown():
    """A 52/48 split is not a prediction, and must not be dressed up as one."""
    result = calibrate(
        RawPrediction(age_years=35.0, p_child=0.0, p_female=0.48, p_male=0.52),
        _settings(),
    )
    assert result.gender.prediction is Gender.UNKNOWN


def test_child_yields_unknown_for_both_attributes():
    """A child has no bracket in this API and 'child' is not a gender.

    Folding them into 18-30, or into whichever of male/female scores higher,
    would produce a confident wrong answer for a caller who deserves the
    opposite.
    """
    result = calibrate(
        RawPrediction(age_years=12.0, p_child=0.93, p_female=0.05, p_male=0.02),
        _settings(),
    )
    assert result.gender.prediction is Gender.UNKNOWN
    assert result.age_bracket.prediction is AgeBracket.UNKNOWN
    assert result.gender.confidence == 0.0
    assert any("child" in note for note in result.notes)


def test_child_mass_is_charged_against_gender_confidence():
    """0.6 child / 0.3 male / 0.1 female must not report male at 0.75."""
    result = calibrate(
        RawPrediction(age_years=20.0, p_child=0.45, p_female=0.10, p_male=0.45),
        _settings(gender_min_confidence=0.0),
    )
    assert result.gender.confidence < 0.6


# ------------------------------------------------------------------- quality
def test_degraded_audio_shrinks_confidence():
    raw = RawPrediction(age_years=24.0, p_child=0.01, p_female=0.02, p_male=0.97)
    clean = calibrate(raw, _settings(), confidence_factor=1.0)
    noisy = calibrate(raw, _settings(), confidence_factor=0.8)
    assert noisy.gender.confidence < clean.gender.confidence
    assert noisy.age_bracket.confidence < clean.age_bracket.confidence


def test_zero_confidence_factor_forces_unknown():
    result = calibrate(
        RawPrediction(age_years=24.0, p_child=0.0, p_female=0.02, p_male=0.98),
        _settings(),
        confidence_factor=0.0,
    )
    assert result.gender.prediction is Gender.UNKNOWN
    assert result.age_bracket.prediction is AgeBracket.UNKNOWN


def test_confidences_are_always_within_range():
    for p_male in (0.0, 0.5, 1.0):
        for age in (5.0, 30.0, 99.0):
            result = calibrate(
                RawPrediction(age_years=age, p_child=0.0,
                              p_female=1.0 - p_male, p_male=p_male),
                _settings(),
            )
            assert 0.0 <= result.gender.confidence <= 1.0
            assert 0.0 <= result.age_bracket.confidence <= 1.0


def test_degenerate_all_zero_probabilities_do_not_crash():
    result = calibrate(
        RawPrediction(age_years=40.0, p_child=0.0, p_female=0.0, p_male=0.0),
        _settings(),
    )
    assert result.gender.prediction is Gender.UNKNOWN


def test_unknown_prediction_helper():
    result = unknown_prediction("dead air")
    assert result.gender.prediction is Gender.UNKNOWN
    assert result.age_bracket.prediction is AgeBracket.UNKNOWN
    assert result.gender.confidence == 0.0
    assert result.notes == ["dead air"]
