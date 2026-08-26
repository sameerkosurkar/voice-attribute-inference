#!/usr/bin/env python3
"""Generate audio fixtures offline -- no network, no dataset download.

WHY THIS EXISTS. A smoke test that first requires a Common Voice download is
not a smoke test, it is a second setup step that gets skipped. These fixtures
let `pytest` and `scripts/smoke_test.sh` run on a fresh clone, offline.

TWO SOURCES, AND THE REASON THERE ARE TWO
-----------------------------------------
1. espeak-ng (preferred). A ~5 MB formant TTS, apt-installable, in the image.
   Its output is real synthesised *speech*: Silero VAD accepts it (~90% of the
   clip is voiced) and the age/gender model reads its intended gender correctly
   (verified 7/7 across espeak voices). So it can exercise the GOOD-quality
   path end to end.

2. numpy source-filter synthesis (fallback). No dependency at all, but Silero
   VAD *rejects* it -- roughly 1 s of "speech" detected in a 5 s clip.

That second fact is worth stating plainly rather than papering over, because it
was the surprise in building this: I first wrote the numpy synthesiser as the
only fixture source, and every clip came back `insufficient`. That is not a bug
in the VAD. A three-formant pulse train is not speech, and a voice-activity
detector that accepts it would be broken. The VAD refusing to certify audio
that merely *resembles* speech is precisely the behaviour this service depends
on to avoid confident predictions about engine noise.

So the numpy path is used for what it is genuinely good at -- constructing
signals with *known, controlled* degradation (an exact SNR, an exact clipping
ratio) so the quality gate can be tested against ground truth -- and espeak is
used for anything that needs to look like a real caller.

Neither source measures ACCURACY. Synthetic voices are not real speakers.
Accuracy is measured in eval/run_eval.py against real labelled speech.

Usage:
    .venv/bin/python scripts/make_sample_audio.py --outdir samples   # or: make sample
"""

from __future__ import annotations

import argparse
import math
import pathlib
import shutil
import struct
import subprocess
import tempfile
import wave

import numpy as np

SAMPLE_RATE = 16_000

SENTENCE = (
    "Hi, this is the driver calling about the delivery for order four seven two. "
    "I am running about twenty minutes late because of heavy traffic on the highway. "
    "Please let the warehouse know I will be there shortly."
)

# espeak voice -> intended speaker. Verified against the model in
# scripts/verify_gender_mapping.py.
ESPEAK_VOICES = {
    "adult_male": "en-us+m3",
    "adult_female": "en-us+f3",
    "older_male": "en-us+m7",
    "child": "en-us+f4",
}

# Rough population averages for the numpy fallback. Formants scale inversely
# with vocal-tract length, which is the actual acoustic reason adult male
# voices read the way they do.
VOICE_PRESETS: dict[str, dict] = {
    "adult_male":   {"f0": 118.0, "formants": (570, 1100, 2500), "jitter": 0.02},
    "adult_female": {"f0": 210.0, "formants": (650, 1300, 2900), "jitter": 0.02},
    "older_male":   {"f0": 132.0, "formants": (600, 1150, 2450), "jitter": 0.05},
    "child":        {"f0": 300.0, "formants": (800, 1700, 3400), "jitter": 0.03},
}


# --------------------------------------------------------------------- espeak
def espeak_available() -> bool:
    return bool(shutil.which("espeak-ng") and shutil.which("ffmpeg"))


def synth_espeak(preset: str = "adult_male", seconds: float = 5.0) -> np.ndarray | None:
    """Real synthesised speech via espeak-ng. Returns None if unavailable."""
    voice = ESPEAK_VOICES.get(preset, "en-us")
    if not espeak_available():
        return None
    with tempfile.TemporaryDirectory() as tmp:
        raw = pathlib.Path(tmp) / "raw.wav"
        out = pathlib.Path(tmp) / "out.wav"
        try:
            subprocess.run(
                ["espeak-ng", "-v", voice, "-s", "150", "-w", str(raw), SENTENCE],
                check=True, capture_output=True, timeout=60,
            )
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw),
                 "-ar", str(SAMPLE_RATE), "-ac", "1", str(out)],
                check=True, capture_output=True, timeout=60,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
            return None
        samples = read_wav(out)

    want = int(seconds * SAMPLE_RATE)
    if samples.size >= want:
        return samples[:want]
    # Loop rather than pad with silence: padding would drag the measured SNR
    # and speech-ratio down and make the fixture lie about its own quality.
    reps = int(np.ceil(want / max(samples.size, 1)))
    return np.tile(samples, reps)[:want]


# ---------------------------------------------------------------- numpy synth
def _glottal_source(seconds: float, f0: float, jitter: float, rng) -> np.ndarray:
    n = int(seconds * SAMPLE_RATE)
    t = np.arange(n) / SAMPLE_RATE
    contour = f0 * (1.0 - 0.12 * t / max(seconds, 1e-6))
    contour *= 1.0 + jitter * rng.standard_normal(n).cumsum() / max(math.sqrt(n), 1.0)
    phase = 2.0 * np.pi * np.cumsum(contour) / SAMPLE_RATE
    source = 2.0 * (phase / (2.0 * np.pi) % 1.0) - 1.0
    return source.astype(np.float32)


def _formant_filter(x: np.ndarray, freq: float, bandwidth: float = 90.0) -> np.ndarray:
    """Two-pole resonator -- the standard Klatt formant section."""
    r = math.exp(-math.pi * bandwidth / SAMPLE_RATE)
    theta = 2.0 * math.pi * freq / SAMPLE_RATE
    a1, a2 = -2.0 * r * math.cos(theta), r * r
    gain = (1.0 - r) * math.sqrt(1.0 - 2.0 * r * math.cos(2 * theta) + r * r)
    y = np.zeros_like(x)
    y1 = y2 = 0.0
    for i in range(x.size):
        out = gain * x[i] - a1 * y1 - a2 * y2
        y[i] = out
        y2, y1 = y1, out
    return y


def synth_numpy(preset: str = "adult_male", seconds: float = 5.0, seed: int = 7) -> np.ndarray:
    cfg = VOICE_PRESETS[preset]
    rng = np.random.default_rng(seed)
    source = _glottal_source(seconds, cfg["f0"], cfg["jitter"], rng)

    voiced = np.zeros_like(source)
    for i, freq in enumerate(cfg["formants"]):
        voiced += _formant_filter(source, freq) * (1.0 / (i + 1))

    t = np.arange(voiced.size) / SAMPLE_RATE
    envelope = np.clip(0.5 + 0.5 * np.sin(2 * np.pi * 3.5 * t), 0.0, 1.0) ** 1.5
    for start in np.arange(0.9, seconds, 1.7):
        lo, hi = int(start * SAMPLE_RATE), int((start + 0.28) * SAMPLE_RATE)
        envelope[lo:hi] *= 0.02

    out = voiced * envelope
    peak = float(np.max(np.abs(out))) or 1.0
    return (out / peak * 0.7).astype(np.float32)


def synth_voice(preset: str = "adult_male", seconds: float = 5.0, seed: int = 7) -> np.ndarray:
    """espeak if present, else the numpy fallback."""
    speech = synth_espeak(preset, seconds)
    if speech is not None:
        return speech
    return synth_numpy(preset, seconds, seed)


# ------------------------------------------------------------- degradations
def add_noise(clean: np.ndarray, snr_db: float, kind: str = "truck", seed: int = 11) -> np.ndarray:
    """Mix broadband noise in at a target SNR (measured over the whole clip)."""
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(clean.size).astype(np.float32)

    if kind == "truck":
        # Diesel cab noise is strongly low-frequency. A one-pole lowpass plus a
        # ~31 Hz engine rumble is crude but directionally honest, and it matters:
        # white noise would overstate how much high-frequency (formant-carrying)
        # content actually survives, and the model reads formants.
        alpha = 0.92
        filtered = np.zeros_like(noise)
        acc = 0.0
        for i, sample in enumerate(noise):
            acc = alpha * acc + (1 - alpha) * sample
            filtered[i] = acc
        t = np.arange(clean.size) / SAMPLE_RATE
        noise = filtered * 6.0 + 0.35 * np.sin(2 * np.pi * 31.0 * t).astype(np.float32)

    speech_power = float(np.mean(clean**2)) + 1e-12
    noise_power = float(np.mean(noise**2)) + 1e-12
    scale = math.sqrt(speech_power / (noise_power * (10.0 ** (snr_db / 10.0))))
    mixed = clean + noise * scale
    peak = float(np.max(np.abs(mixed))) or 1.0
    return (mixed / max(peak, 1.0)).astype(np.float32)


def clip_signal(x: np.ndarray, drive: float = 3.0) -> np.ndarray:
    """Hard-clip, as an over-driven bluetooth headset does."""
    return np.clip(x * drive, -1.0, 1.0).astype(np.float32)


def narrowband(x: np.ndarray) -> np.ndarray:
    """Simulate a G.711 narrowband leg: band-limit to ~3.4 kHz.

    Done in numpy (brick-wall in the frequency domain) so the fixture exists
    even without ffmpeg, and so the cutoff is exact and testable.
    """
    spectrum = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(x.size, d=1.0 / SAMPLE_RATE)
    spectrum[freqs > 3400.0] = 0.0
    out = np.fft.irfft(spectrum, n=x.size).astype(np.float32)
    peak = float(np.max(np.abs(out))) or 1.0
    return (out / max(peak, 1.0) * 0.7).astype(np.float32)


# ------------------------------------------------------------------- wav i/o
def read_wav(path: pathlib.Path) -> np.ndarray:
    with wave.open(str(path)) as handle:
        frames = handle.readframes(handle.getnframes())
    return np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0


def write_wav(path: pathlib.Path, samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> None:
    """16-bit PCM WAV via the stdlib -- no soundfile/libsndfile dependency."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ints = np.clip(samples * 32767.0, -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(ints.tobytes())


def wav_bytes(samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> bytes:
    """In-memory WAV, for tests that should not touch the filesystem."""
    ints = np.clip(samples * 32767.0, -32768, 32767).astype("<i2")
    data = ints.tobytes()
    header = b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVEfmt "
    header += struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16)
    header += b"data" + struct.pack("<I", len(data))
    return header + data


# ---------------------------------------------------------------------- main
def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--outdir", default="samples", type=pathlib.Path)
    parser.add_argument("--seconds", default=5.0, type=float)
    parser.add_argument("--force-numpy", action="store_true",
                        help="Ignore espeak-ng even if installed.")
    args = parser.parse_args()

    synth = synth_numpy if args.force_numpy else synth_voice
    source = "numpy source-filter synthesis" if args.force_numpy or not espeak_available() \
        else "espeak-ng"
    written: list[tuple[str, str]] = []

    for preset in VOICE_PRESETS:
        write_wav(args.outdir / f"{preset}_clean.wav", synth(preset, args.seconds))
        written.append((f"{preset}_clean.wav", "clean wideband speech"))

    base = synth("adult_male", args.seconds)
    for snr in (20, 10, 5, 0, -5, -10):
        write_wav(args.outdir / f"adult_male_truck_{snr}db.wav".replace("-", "minus"), add_noise(base, float(snr)))
        written.append((f"adult_male_truck_{snr}db.wav".replace("-", "minus"), f"truck cab noise at {snr} dB SNR"))

    write_wav(args.outdir / "adult_male_clipped.wav", clip_signal(base))
    written.append(("adult_male_clipped.wav", "over-driven / clipped headset"))

    write_wav(args.outdir / "adult_male_narrowband.wav", narrowband(base))
    written.append(("adult_male_narrowband.wav", "G.711-style 3.4 kHz band limit"))

    write_wav(args.outdir / "silence.wav",
              np.zeros(int(args.seconds * SAMPLE_RATE), dtype=np.float32))
    written.append(("silence.wav", "dead air -> expect audio_quality=insufficient"))

    write_wav(args.outdir / "too_short.wav", synth("adult_male", 0.3))
    written.append(("too_short.wav", "0.3 s -> expect HTTP 400 AUDIO_TOO_SHORT"))

    print(f"Wrote {len(written)} fixtures to {args.outdir}/  (source: {source})\n")
    for name, description in written:
        print(f"  {name:32s} {description}")

    if source != "espeak-ng":
        print(
            "\nNOTE: espeak-ng was not found, so these came from the numpy fallback.\n"
            "Silero VAD does not accept that output as speech, so the clean fixtures\n"
            "will report audio_quality=insufficient. That is correct VAD behaviour,\n"
            "not a service bug. Install espeak-ng (apt-get install espeak-ng /\n"
            "brew install espeak-ng) for fixtures that exercise the good-quality path."
        )
    print(
        "\nThese are SYNTHETIC voices. They exercise the pipeline and the quality\n"
        "gate; they do not measure accuracy. Use eval/run_eval.py against real\n"
        "labelled speech for that."
    )


if __name__ == "__main__":
    main()
