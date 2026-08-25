# Sample audio

Three committed fixtures, one per outcome the service is designed to produce:

| File | Expect |
|---|---|
| `adult_male_clean.wav` | `audio_quality: good` + a confident prediction |
| `adult_male_truck_5db.wav` | `audio_quality: degraded` — same answer, lower confidence |
| `silence.wav` | `audio_quality: insufficient` + `unknown` — the service abstains |

```bash
curl -X POST localhost:8000/analyze -F "audio=@samples/adult_male_clean.wav"
```

## Provenance, and what these are not

Generated offline by `scripts/make_sample_audio.py` using **espeak-ng** (a
formant speech synthesiser) for the voice, plus a low-pass-filtered noise model
approximating a diesel cab for the noisy one. No dataset download, no network.

They are **synthetic voices**. They exercise the pipeline and the quality gate
end to end, and they are good enough to catch a permuted gender label — but they
are not real speakers, so they measure nothing about accuracy. For accuracy, run
`eval/run_eval.py` against real labelled speech (see the README).

## Regenerating, and the rest of the set

```bash
make sample          # or: python scripts/make_sample_audio.py --outdir samples
```

That writes the full set — a truck-noise ladder from +20 dB down to −10 dB, a
clipped/over-driven headset, a G.711-style narrowband clip, and a too-short clip
for the 400 path. Only the three above are committed, to keep the repo small.

`scripts/smoke_test.sh` generates them automatically if they are missing, and
will borrow the container's toolchain when the host has no numpy — so it needs
nothing installed to run.
