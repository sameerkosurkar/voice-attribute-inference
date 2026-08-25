"""Scoring, including the calibration metrics.

Accuracy alone is the wrong scorecard for this service. The API promises a
*confidence*, and a downstream voice agent will branch on it -- "if gender
confidence > 0.8, use a gendered greeting". If a reported 0.8 is right only 55%
of the time, that branch is broken, and no accuracy number would have told you.

So the harness reports:

  accuracy / macro-F1    Was the label right?
  confusion matrix       Which way is it wrong? (systematically confusing
                         adjacent age brackets is a very different problem from
                         confusing 18-30 with 60+)
  MAE (years)            Regression error before bracketing, which is the
                         quantity the calibration's sigma models.
  ECE + reliability      Does a reported confidence of p mean "right p of the
                         time"? This is the metric that validates the whole
                         calibration design.
  fitted sigma           The empirical age error, to feed back into
                         VA_AGE_SIGMA_YEARS instead of the paper's prior.
  coverage               What fraction did we answer at all? Trivially
                         maximised by never saying "unknown", so it is only
                         meaningful read together with accuracy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class ClassificationScore:
    labels: list[str]
    confusion: dict[str, dict[str, int]]
    accuracy: float
    macro_f1: float
    per_class: dict[str, dict[str, float]]
    n: int
    n_answered: int

    @property
    def coverage(self) -> float:
        return self.n_answered / self.n if self.n else 0.0


def score_classification(
    truths: list[str], predictions: list[str], labels: list[str],
    abstain: str = "unknown",
) -> ClassificationScore:
    """Accuracy and macro-F1 over the ANSWERED subset.

    Abstentions are excluded from accuracy and reported separately as coverage.
    Counting "unknown" as a miss would punish the service for the behaviour the
    assignment explicitly asks for; counting it as a hit would reward silence.
    Reporting the two numbers side by side is the only honest option -- neither
    can be improved without showing up in the other.
    """
    confusion = {t: {p: 0 for p in labels + [abstain]} for t in labels}
    answered = 0
    correct = 0

    for truth, prediction in zip(truths, predictions):
        if truth not in confusion:
            continue
        confusion[truth][prediction] = confusion[truth].get(prediction, 0) + 1
        if prediction != abstain:
            answered += 1
            correct += prediction == truth

    per_class: dict[str, dict[str, float]] = {}
    f1s = []
    for label in labels:
        tp = confusion.get(label, {}).get(label, 0)
        fn = sum(v for k, v in confusion.get(label, {}).items()
                 if k != label and k != abstain)
        fp = sum(confusion[t].get(label, 0) for t in labels if t != label)

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1,
                            "support": float(sum(confusion.get(label, {}).values()))}
        f1s.append(f1)

    total = sum(sum(row.values()) for row in confusion.values())
    return ClassificationScore(
        labels=labels,
        confusion=confusion,
        accuracy=correct / answered if answered else 0.0,
        macro_f1=sum(f1s) / len(f1s) if f1s else 0.0,
        per_class=per_class,
        n=total,
        n_answered=answered,
    )


@dataclass
class CalibrationScore:
    ece: float
    mce: float
    bins: list[dict] = field(default_factory=list)
    n: int = 0

    def reliability_table(self) -> str:
        lines = [
            f"{'confidence bin':>16s} {'n':>6s} {'mean conf':>10s} "
            f"{'accuracy':>9s} {'gap':>7s}",
            "-" * 52,
        ]
        for b in self.bins:
            if not b["n"]:
                continue
            lines.append(
                f"{b['lo']:.2f}-{b['hi']:.2f}".rjust(16)
                + f"{b['n']:>7d} {b['mean_confidence']:>10.3f} "
                f"{b['accuracy']:>9.3f} {b['accuracy'] - b['mean_confidence']:>+7.3f}"
            )
        return "\n".join(lines)


def expected_calibration_error(
    confidences: list[float], correct: list[bool], n_bins: int = 10
) -> CalibrationScore:
    """Standard binned ECE.

    ECE = sum over bins of (bin weight) * |accuracy - mean confidence|.
    0 is perfect. A positive gap in the table means UNDER-confident (safe); a
    negative gap means OVER-confident, which is the dangerous direction for
    this service -- it means a downstream agent trusts a prediction it should
    not.
    """
    if not confidences:
        return CalibrationScore(ece=0.0, mce=0.0, bins=[], n=0)

    edges = [i / n_bins for i in range(n_bins + 1)]
    bins = []
    ece = 0.0
    mce = 0.0
    total = len(confidences)

    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        # Upper-inclusive on the last bin so confidence == 1.0 is counted.
        members = [
            (c, ok) for c, ok in zip(confidences, correct)
            if (lo <= c < hi) or (i == n_bins - 1 and c == hi)
        ]
        if members:
            mean_conf = sum(c for c, _ in members) / len(members)
            accuracy = sum(ok for _, ok in members) / len(members)
            gap = abs(accuracy - mean_conf)
            ece += (len(members) / total) * gap
            mce = max(mce, gap)
        else:
            mean_conf = accuracy = 0.0
        bins.append({"lo": lo, "hi": hi, "n": len(members),
                     "mean_confidence": mean_conf, "accuracy": accuracy})

    return CalibrationScore(ece=ece, mce=mce, bins=bins, n=total)


@dataclass
class RegressionScore:
    mae: float
    rmse: float
    bias: float
    fitted_sigma: float
    n: int


def score_regression(truths: list[float], predictions: list[float]) -> RegressionScore:
    """Age error, plus the sigma to feed back into calibration.

    `fitted_sigma` is the RMSE of the residuals after removing the mean bias,
    which is exactly the sigma parameter the Gaussian bracket integration in
    app/inference/calibration.py assumes. Setting VA_AGE_SIGMA_YEARS to this
    replaces the paper's prior with a measurement on your own traffic -- which
    is the difference between a confidence that is calibrated and one that is
    merely plausible.

    `bias` is worth reading separately: a systematic +6 y offset is fixable with
    a constant, while the same magnitude as random scatter is not.
    """
    pairs = [(t, p) for t, p in zip(truths, predictions)
             if t is not None and p is not None]
    if not pairs:
        return RegressionScore(0.0, 0.0, 0.0, 0.0, 0)

    errors = [p - t for t, p in pairs]
    n = len(errors)
    mae = sum(abs(e) for e in errors) / n
    rmse = math.sqrt(sum(e * e for e in errors) / n)
    bias = sum(errors) / n
    variance = sum((e - bias) ** 2 for e in errors) / n
    return RegressionScore(mae=mae, rmse=rmse, bias=bias,
                           fitted_sigma=math.sqrt(variance), n=n)


def format_confusion(score: ClassificationScore, abstain: str = "unknown") -> str:
    columns = score.labels + [abstain]
    width = max(len(c) for c in columns + score.labels) + 2
    header = "truth \\ pred".ljust(width) + "".join(c.rjust(width) for c in columns)
    lines = [header, "-" * len(header)]
    for truth in score.labels:
        row = truth.ljust(width)
        for prediction in columns:
            row += str(score.confusion.get(truth, {}).get(prediction, 0)).rjust(width)
        lines.append(row)
    return "\n".join(lines)
