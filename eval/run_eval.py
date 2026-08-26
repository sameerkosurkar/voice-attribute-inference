#!/usr/bin/env python3
"""Evaluate the service against labelled speech and report calibration.

    # Common Voice extract you downloaded (the citable path)
    .venv/bin/python eval/run_eval.py --adapter local --path ~/cv-corpus-17.0-en --limit 500

    # Community HF mirror, streamed (zero-setup smoke test)
    .venv/bin/python eval/run_eval.py --adapter hf --limit 200

    # Your own call recordings, named <gender>_<age>_*.wav
    .venv/bin/python eval/run_eval.py --adapter dir --path ./recordings

    # Noise-robustness curve: how the service degrades in a truck cab
    .venv/bin/python eval/run_eval.py --adapter local --path ~/cv --noise-snr 20 10 5 0

The harness runs the SAME pipeline the API runs -- decode, quality gate,
calibration, thresholds -- rather than calling the model directly. Evaluating
the bare model would measure something the service never actually emits: it
would miss the abstentions, the quality shrinkage, and the bracket integration,
which are most of what distinguishes this service from a raw forward pass.
"""

from __future__ import annotations

import argparse
import asyncio
import pathlib
import statistics
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from app.audio import quality as quality_mod          # noqa: E402
from app.audio.decode import decode                    # noqa: E402
from app.config import Settings                        # noqa: E402
from app.errors import VoiceAttributeError             # noqa: E402
from app.inference.calibration import calibrate, unknown_prediction  # noqa: E402
from eval import adapters                              # noqa: E402
from eval.metrics import (                             # noqa: E402
    expected_calibration_error,
    format_confusion,
    score_classification,
    score_regression,
)

GENDER_LABELS = ["male", "female"]
BRACKET_LABELS = ["18-30", "31-45", "46-60", "60+"]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--adapter", choices=["local", "hf", "dir"], default="local")
    parser.add_argument("--path", type=pathlib.Path,
                        help="Corpus root (local / dir adapters).")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--hf-repo", default="fsicoli/common_voice_17_0")
    parser.add_argument("--hf-config", default="en")
    parser.add_argument("--hf-split", default="validated")
    parser.add_argument("--model", default=None,
                        help="Override VA_AGE_GENDER_MODEL for this run.")
    parser.add_argument("--sigma", type=float, default=None,
                        help="Override the age sigma used for bracket confidence.")
    parser.add_argument("--noise-snr", type=float, nargs="*", default=None,
                        metavar="DB",
                        help="Also evaluate with truck noise injected at these "
                             "SNRs, producing a degradation curve.")
    parser.add_argument("--bins", type=int, default=10,
                        help="Reliability bins for ECE.")
    return parser


def load_samples(args) -> list[adapters.Sample]:
    if args.adapter == "local":
        if not args.path:
            raise SystemExit("--adapter local needs --path <common voice extract>")
        return list(adapters.local_common_voice(args.path, args.limit))
    if args.adapter == "dir":
        if not args.path:
            raise SystemExit("--adapter dir needs --path <directory>")
        return list(adapters.labelled_directory(args.path, args.limit))
    return list(
        adapters.hf_common_voice(args.hf_repo, args.hf_config, args.hf_split, args.limit)
    )


async def evaluate(samples, settings: Settings, backend, noise_snr: float | None):
    """Run the production pipeline over every sample."""
    from scripts.make_sample_audio import add_noise, wav_bytes

    rows = []
    for index, sample in enumerate(samples, 1):
        if index % 25 == 0:
            print(f"  ... {index}/{len(samples)}", file=sys.stderr)

        payload = sample.audio_bytes
        started = time.perf_counter()
        try:
            audio = await decode(payload, settings)
        except VoiceAttributeError as exc:
            rows.append({"error": exc.code, "sample": sample})
            continue

        try:
            samples_array = audio.samples
            if noise_snr is not None:
                # Re-encode after mixing so the noisy run goes through exactly
                # the same decode path as the clean one.
                noisy = add_noise(samples_array, noise_snr)
                audio.wipe()
                audio = await decode(wav_bytes(noisy), settings)
                samples_array = audio.samples

            report = quality_mod.assess(samples_array, settings.target_sample_rate,
                                        settings)
            if not report.usable:
                prediction = unknown_prediction("insufficient")
                age_years = None
            else:
                raw = backend.predict(samples_array, settings.target_sample_rate)
                prediction = calibrate(raw, settings,
                                       confidence_factor=report.confidence_factor)
                age_years = raw.age_years

            rows.append({
                "sample": sample,
                "quality": report.quality.value,
                "gender": prediction.gender.prediction.value,
                "gender_confidence": prediction.gender.confidence,
                "bracket": prediction.age_bracket.prediction.value,
                "bracket_confidence": prediction.age_bracket.confidence,
                "age_years": age_years,
                "latency_ms": (time.perf_counter() - started) * 1000.0,
            })
        finally:
            audio.wipe()
    return rows


def report(rows, label: str, n_bins: int) -> dict:
    ok = [r for r in rows if "error" not in r]
    errors = [r for r in rows if "error" in r]

    print(f"\n{'=' * 68}\n{label}\n{'=' * 68}")
    print(f"samples: {len(rows)}   decoded: {len(ok)}   failed: {len(errors)}")
    if errors:
        codes: dict[str, int] = {}
        for row in errors:
            codes[row["error"]] = codes.get(row["error"], 0) + 1
        print(f"decode failures: {codes}")
    if not ok:
        return {}

    quality_counts: dict[str, int] = {}
    for row in ok:
        quality_counts[row["quality"]] = quality_counts.get(row["quality"], 0) + 1
    print(f"audio quality: {quality_counts}")

    latencies = sorted(r["latency_ms"] for r in ok)
    print(
        f"latency ms: p50={statistics.median(latencies):.0f} "
        f"p95={latencies[int(0.95 * len(latencies)) - 1]:.0f} "
        f"max={latencies[-1]:.0f}"
    )

    summary: dict = {}

    # ------------------------------------------------------------- gender ---
    gender_rows = [r for r in ok if r["sample"].gender]
    if gender_rows:
        truths = [r["sample"].gender for r in gender_rows]
        predictions = [r["gender"] for r in gender_rows]
        score = score_classification(truths, predictions, GENDER_LABELS)
        print(f"\n--- gender ---")
        print(f"accuracy (answered): {score.accuracy:.3f}   "
              f"macro-F1: {score.macro_f1:.3f}   "
              f"coverage: {score.coverage:.3f} "
              f"({score.n_answered}/{score.n} answered)")
        print(format_confusion(score))

        answered = [(r, t) for r, t in zip(gender_rows, truths)
                    if r["gender"] != "unknown"]
        if answered:
            calibration = expected_calibration_error(
                [r["gender_confidence"] for r, _ in answered],
                [r["gender"] == t for r, t in answered],
                n_bins=n_bins,
            )
            print(f"\ncalibration: ECE={calibration.ece:.4f}  MCE={calibration.mce:.4f}")
            print(calibration.reliability_table())
            summary["gender_ece"] = calibration.ece
        summary["gender_accuracy"] = score.accuracy
        summary["gender_coverage"] = score.coverage

    # ---------------------------------------------------------------- age ---
    bracket_rows = [r for r in ok if r["sample"].age_bracket]
    if bracket_rows:
        truths = [r["sample"].age_bracket for r in bracket_rows]
        predictions = [r["bracket"] for r in bracket_rows]
        score = score_classification(truths, predictions, BRACKET_LABELS)
        print(f"\n--- age bracket ---")
        print(f"accuracy (answered): {score.accuracy:.3f}   "
              f"macro-F1: {score.macro_f1:.3f}   "
              f"coverage: {score.coverage:.3f} "
              f"({score.n_answered}/{score.n} answered)")
        print(format_confusion(score))

        answered = [(r, t) for r, t in zip(bracket_rows, truths)
                    if r["bracket"] != "unknown"]
        if answered:
            calibration = expected_calibration_error(
                [r["bracket_confidence"] for r, _ in answered],
                [r["bracket"] == t for r, t in answered],
                n_bins=n_bins,
            )
            print(f"\ncalibration: ECE={calibration.ece:.4f}  MCE={calibration.mce:.4f}")
            print(calibration.reliability_table())
            summary["bracket_ece"] = calibration.ece
        summary["bracket_accuracy"] = score.accuracy
        summary["bracket_coverage"] = score.coverage

    # --------------------------------------------------- age regression ----
    age_rows = [r for r in ok
                if r["sample"].age_years is not None and r["age_years"] is not None]
    if age_rows:
        regression = score_regression(
            [r["sample"].age_years for r in age_rows],
            [r["age_years"] for r in age_rows],
        )
        print(f"\n--- age regression (n={regression.n}) ---")
        print(f"MAE  : {regression.mae:.2f} years")
        print(f"RMSE : {regression.rmse:.2f} years")
        print(f"bias : {regression.bias:+.2f} years "
              f"({'over' if regression.bias > 0 else 'under'}-estimates)")
        print(f"\nfitted sigma: {regression.fitted_sigma:.2f} years")
        print("  -> set VA_AGE_SIGMA_YEARS to this to replace the paper's prior")
        print("     with a measurement on this corpus, which is what makes the")
        print("     bracket confidences calibrated rather than merely plausible.")
        if any(s.source.startswith(("cv:", "hf:")) for s in
               (r["sample"] for r in age_rows)):
            print("\n  NOTE: Common Voice labels age by decade, so ground truth is a")
            print("  decade midpoint. That adds a +/-5 y quantisation floor to MAE")
            print("  which is NOT model error.")
        summary["age_mae"] = regression.mae
        summary["fitted_sigma"] = regression.fitted_sigma

    return summary


def main() -> int:
    args = build_arg_parser().parse_args()

    overrides: dict = {"enable_language_id": False}
    if args.model:
        overrides["age_gender_model"] = args.model
    if args.sigma is not None:
        overrides["age_sigma_years"] = args.sigma
    settings = Settings(**overrides)

    print(f"model: {settings.age_gender_model}")
    print(f"sigma: {settings.age_sigma_years} years")
    print(f"loading samples via '{args.adapter}' adapter ...", file=sys.stderr)

    samples = load_samples(args)
    if not samples:
        raise SystemExit(
            "No labelled samples found. Common Voice rows without BOTH an age "
            "and a gender label are skipped -- most rows have neither."
        )
    print(f"loaded {len(samples)} labelled samples")

    from app.inference.audeering import AudeeringBackend

    backend = AudeeringBackend(settings)
    backend.load()
    backend.warmup()

    rows = asyncio.run(evaluate(samples, settings, backend, None))
    clean = report(rows, "CLEAN AUDIO", args.bins)

    curve = [("clean", clean)]
    for snr in args.noise_snr or []:
        noisy_rows = asyncio.run(evaluate(samples, settings, backend, snr))
        curve.append((f"{snr:g} dB",
                      report(noisy_rows, f"TRUCK NOISE AT {snr:g} dB SNR", args.bins)))

    if len(curve) > 1:
        print(f"\n{'=' * 68}\nDEGRADATION CURVE\n{'=' * 68}")
        print(f"{'condition':>12s} {'gender acc':>11s} {'gender cov':>11s} "
              f"{'age acc':>9s} {'age cov':>9s} {'MAE':>7s}")
        print("-" * 64)
        for name, s in curve:
            if not s:
                continue
            print(f"{name:>12s} {s.get('gender_accuracy', 0):>11.3f} "
                  f"{s.get('gender_coverage', 0):>11.3f} "
                  f"{s.get('bracket_accuracy', 0):>9.3f} "
                  f"{s.get('bracket_coverage', 0):>9.3f} "
                  f"{s.get('age_mae', 0):>7.2f}")
        print(
            "\nRead COVERAGE alongside ACCURACY. The service is designed to abstain\n"
            "as audio degrades, so falling coverage with roughly held accuracy is\n"
            "the intended behaviour -- it is refusing rather than guessing. Accuracy\n"
            "collapsing while coverage stays high would be the real failure."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
