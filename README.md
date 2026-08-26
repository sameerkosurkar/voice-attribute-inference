# Voice Attribute Inference Service

Estimates a caller's **gender** and **age bracket** from a short audio sample,
with calibrated confidences and an explicit audio-quality verdict. Built for
voice AI agents on logistics calls, where the audio is a driver on a mobile in
a truck cab and the honest answer is often "I don't know".

```bash
docker compose up --build -d       # ~5 min first build (weights are baked in)
./scripts/smoke_test.sh            # generates its own fixtures; no downloads
```

The smoke test needs **nothing installed on the host** — not even numpy. If the
host cannot generate fixtures it borrows the container's numpy and espeak-ng.

```json
{
  "contact_id": "0e3a...-...",
  "gender":      { "prediction": "male",  "confidence": 0.9985 },
  "age_bracket": { "prediction": "31-45", "confidence": 0.5613 },
  "processing_ms": 187,
  "audio_quality": "good",
  "language":    { "prediction": "en", "confidence": 0.9812 }
}
```

---

## Submission map

| Asked for | Where |
|---|---|
| README with setup, design decisions, model rationale, known limitations | this file — [Quick start](#quick-start) · [Design decisions](#design-decisions) · [Why this model](#design-decisions) · [Known limitations](#known-limitations) |
| 200-word design write-up | **[DESIGN.md](DESIGN.md)** — 199 words, then an optional appendix |
| At least one working test | `make test` — 111 fast tests, no weights needed (135 with `make test-slow`) |
| A sample audio file to smoke-test with | **`samples/`** — three committed WAVs, one per outcome. Also generated on demand by `scripts/make_sample_audio.py` |

```bash
docker compose up --build -d && ./scripts/smoke_test.sh
```

Runs on a fresh clone with **nothing installed on the host** — not even numpy.

---

## Contents

- [The one idea](#the-one-idea)
- [Quick start](#quick-start)
- [API](#api)
- [How it works](#how-it-works)
- [Design decisions](#design-decisions)
- [Measured performance](#measured-performance)
- [Two things I got wrong first](#two-things-i-got-wrong-first)
- [Swapping the model](#swapping-the-model)
- [Thread safety](#thread-safety)
- [Testing](#testing)
- [Evaluation harness](#evaluation-harness)
- [Configuration](#configuration)
- [Known limitations](#known-limitations)
- [Model licence](#model-licence)

---

## The one idea

A softmax always sums to one. Feed wav2vec2 two seconds of diesel engine and it
returns a confident gender, and nothing in the output tells you it was noise.

So the interesting part of this service is not the model — it is the **quality
gate that runs before the model** and is allowed to refuse. Everything else
(bracket calibration, streaming aggregation, the abstention thresholds) follows
from taking that seriously.

Concretely, on a truck-cab noise ladder:

| Condition | `audio_quality` | Prediction |
|---|---|---|
| clean speech | `good` | `male` @ 0.998 |
| truck cab, 5 dB SNR | `degraded` | `male` @ 0.799 — same answer, less certain |
| truck cab, −5 dB SNR | `insufficient` | `unknown` — refuses |
| dead air / hold tone | `insufficient` | `unknown` — refuses |

Confidence falls before the answer flips, and then the service stops answering.

---

## Quick start

### Docker (recommended)

```bash
docker compose up --build -d
curl -s localhost:8000/ready

# `samples/` is generated, not committed -- create it first (either works):
./scripts/smoke_test.sh                                  # generates + exercises
make sample                                              # or, if you have a venv

curl -s -X POST localhost:8000/analyze -F "audio=@samples/adult_male_clean.wav"
```

The image bakes in all model weights, so the container needs **no network at
runtime** (`HF_HUB_OFFLINE=1`). Measured on a cold `--no-cache` build: the
resulting image is **~4.0 GB** — roughly half dependencies (torch, ONNX Runtime,
transformers) and half baked-in weights (wav2vec2 ~1.2 GB, its ONNX export
346 MB, whisper-tiny). Startup to `/ready` is then **~2.2 s**.

### Local

```bash
make install        # venv + CPU-only torch + deps
make sample         # generate audio fixtures (offline)
make run            # serve on :8000
make test           # fast suite, no weights needed
```

Needs Python 3.11+, `ffmpeg`, and optionally `espeak-ng` for speech fixtures
(`brew install ffmpeg espeak-ng` / `apt-get install ffmpeg espeak-ng`).

### Smoke test

```bash
./scripts/smoke_test.sh
```

Generates its own fixtures and asserts the four outcomes that matter: a
good-quality prediction, graceful degradation under noise, abstention on
unusable audio, and a typed 415 on garbage bytes.

### Streaming demo

```bash
python scripts/ws_client.py
```

Streams a WAV at 1× wall clock and prints progressive predictions. Watch
`conf` climb and `stable` flip to `True` — that is the argument for streaming
here: a voice agent can commit to a persona mid-call instead of waiting for it
to end.

---

## API

### `POST /analyze`

Accepts **either** a multipart upload (field `audio`) **or** raw bytes as the
request body. Both appear in practice — multipart from a batch job, raw bytes
from a telephony bridge that is already streaming.

```bash
curl -X POST localhost:8000/analyze -F "audio=@call.wav"
curl -X POST localhost:8000/analyze -H 'content-type: audio/mpeg' --data-binary @call.mp3
```

| Query param | Default | Meaning |
|---|---|---|
| `contact_id` | generated `uuid4` | Your correlation id. If omitted, a fresh random one — never derived from the audio ([why](PRIVACY.md)). |
| `debug` | `false` | Adds `quality_detail` and per-stage `timings`. |

Formats: wav, mp3, flac, ogg/opus, webm, m4a, G.711 µ-law/A-law — anything
ffmpeg decodes. `Content-Type` is *not* trusted; ffmpeg probes the container,
because telephony gateways mislabel audio often enough to matter.

**Status codes.** Note the deliberate split:

| | |
|---|---|
| **200** + `insufficient` / `unknown` | Audio was unusable. This is a *normal outcome*, not an error — a driver answering in a loud cab is ordinary traffic, and the agent should handle it on the normal code path. |
| 400 `EMPTY_AUDIO` / `AUDIO_TOO_SHORT` | Client bug |
| 413 `AUDIO_TOO_LARGE` | Over 25 MB |
| 415 `DECODE_FAILED` | Not decodable audio |
| 429 `OVERLOADED` | All inference slots busy; `Retry-After: 1` |
| 503 `MODEL_NOT_READY` | Still loading |
| 504 `DECODE_TIMEOUT` / `INFERENCE_TIMEOUT` | Deadline exceeded |

Conflating "bad audio" with "bad request" is the most common way this kind of
service ends up either throwing on normal traffic or silently emitting
confident nonsense.

### `WS /ws/analyze` (bonus)

```jsonc
// ->  {"type":"start","format":"pcm_s16le","sample_rate":16000}
// <-  {"type":"ready","contact_id":"...","session_id":"..."}
// ->  <binary audio frames>
// <-  {"type":"partial","gender":{...},"stable":false,"chunks_seen":5, ...}
// ->  {"type":"end"}
// <-  {"type":"final","is_final":true,"stable":true, ...}
```

`format: "pcm_s16le"` takes a zero-copy path with no decoder. Anything else
(webm/opus from a browser, mp3) gets one long-lived ffmpeg per connection.
Partial events extend the REST schema, so one client parser serves both.

### Ops

`GET /health` liveness · `GET /ready` readiness (weights loaded **and** warmed)
· `GET /metrics` Prometheus · `GET /docs` OpenAPI.

The health/ready split matters on Kubernetes: `/health` must answer while the
model loads or the orchestrator kills the pod mid-startup, and `/ready` must not
pass until a warmup pass has run or the first callers of every new pod get a
slow response.

---

## How it works

```
bytes ──▶ decode ──▶ QUALITY GATE ──┬── insufficient ──▶ unknown/unknown (200)
                                    │                    ~45 ms, no model run
                                    └── good/degraded
                                             │
                              ┌──────────────┴──────────────┐
                              ▼                             ▼
                     wav2vec2 age+gender          whisper-tiny language id
                     (one forward pass)           (budgeted, best-effort)
                              │                             │
                              ▼                             │
                     calibration ◀── quality shrinkage      │
                              │                             │
                              └──────────┬──────────────────┘
                                         ▼
                                    JSON response
```

**Quality gate** — four independent signals, each catching a different real
failure:

| Signal | Catches |
|---|---|
| Silero VAD speech seconds | dead air, hold music, IVR tones |
| speech-to-background ratio (dB) | speech buried under a diesel engine |
| clipping ratio | wind on a mic, over-driven bluetooth |
| 4–8 kHz band energy | narrowband G.711, aggressive codecs |

`insufficient` short-circuits before inference, which is why unusable audio
costs 45 ms rather than the whole budget.

**Calibration** — the model gives a continuous age; the API asks for a bracket.
Rather than `bracket_of(age)`, we place a Gaussian `N(age, σ²)` (σ ≈ 8 y, from
the source paper's reported MAE 7.1–10.8 y) on the estimate and integrate it
over each bracket:

```
age 24.0  ->  18-30  @ 0.72     comfortably inside a bracket
age 45.4  ->  31-45  @ 0.38     on a boundary: an honest coin flip
age 70.0  ->  60+    @ 0.86     open-ended bracket
```

45.4 then falls below the confidence threshold and reports `unknown`, which is
the correct answer. This also means the product can re-cut the brackets
(25-35 / 36-50 / …) without retraining — the reason a *regression* head mattered
when choosing the model.

**The child case** — the gender head is 3-way: child / female / male. A child is
not a gender, and a child's age is under 18, outside every bracket the API
defines. So a dominant child probability yields `unknown` for both, rather than
being folded into `18-30` or into whichever of male/female scores higher.

---

## Design decisions

**Why this model.** `audeering/wav2vec2-large-robust-*-ft-age-gender`: one
shared trunk with two heads gives both attributes in a single forward pass;
`wav2vec2-large-robust` is pretrained specifically to survive telephony/noise
domain shift; training spans aGender + Common Voice + TIMIT + VoxCeleb2 (read,
telephone, and in-the-wild audio); and **age is a regression**, which is what
the bracket calibration above depends on. The 6-layer variant is the default —
the 24-layer is more accurate at ~3× the transformer compute
(`VA_AGE_GENDER_MODEL` switches it).

**Why ffmpeg over pipes.** Coverage (one code path for every codec a telephony
vendor might send), resampling quality (soxr; wav2vec2 is sensitive to the
16 kHz assumption), and privacy — stdin→stdout means audio never becomes a file,
which is the single most important property in the decode path.

**Why a WAV fast path.** Spawning ffmpeg costs ~80 ms p50 — 16% of the budget
spent on process creation for a format needing no decoding. Uncompressed 16 kHz
PCM WAV is exactly what telephony recorders emit. The fast path parses the RIFF
header in numpy (**80 ms → 0.09 ms**) and is *bit-identical* to ffmpeg's output
(asserted in tests). It is conservative by construction: wrong sample rate,
µ-law, 24-bit, anything unusual → falls back, because it never resamples and a
hand-rolled resampler would alias exactly the high-frequency content the gender
model reads.

**Why a backend interface, and why `auto` is platform-aware.** The default
model is non-commercially licensed (see below), so replacing it must be a swap
rather than a rewrite. The second reason is the table above: ONNX Runtime is
2.1× *faster* than PyTorch in the container and 2.3× *slower* on the macOS host,
because macOS PyTorch links Apple's Accelerate framework and the manylinux
aarch64 wheel does not. "ONNX is faster" is not a fact about ONNX — it is a fact
about which BLAS your PyTorch happened to link. So `auto` picks by platform, and
`VA_BACKEND` overrides it once you have measured your own target.

**Why threads are pinned small.** Both the thread pool and torch intra-op
threads default to 2. The instinct to set both high is wrong: under a 2-core
limit, 2 threads gave 621 ms and 8 threads gave 905 ms. Scale replicas, not
threads.

**Why load is shed, not queued.** A semaphore caps in-flight inferences and
returns 429 past it. A caller who waits 8 s for an age guess has already lost
the call.

---

## Measured performance

5-second chunk, warm process, measured on this machine (M-series Mac, 14 cores).

**End-to-end, host (`make run`)** — includes decode, quality gate, inference,
calibration, HTTP:

| Input | p50 | p95 |
|---|---|---|
| WAV 16 kHz (fast path) | **149 ms** | 158 ms |
| mp3 64 kbps (ffmpeg) | 224 ms | 245 ms |
| unusable audio (short-circuit) | 45 ms | — |

Stage breakdown for the WAV case: decode 0.1 ms · quality gate 11 ms ·
inference 133 ms · language ID 0 ms (runs concurrently).

**Backends** — same checkpoint, same input, 4 threads:

| | PyTorch | ONNX Runtime |
|---|---|---|
| macOS arm64 (host) | **89 ms** | 209 ms |
| Linux arm64 (container) | 453 ms | **213 ms** |

**In the container** (`docker compose up`, ONNX backend, 5 s WAV, p50) — sized
from these measurements rather than from habit:

| CPUs / torch threads | language ID off | language ID on |
|---|---|---|
| 2 / 2 | ~450 ms | ~740 ms |
| **4 / 4  (compose default)** | **~245 ms** | ~430 ms |
| 6 / 6 | ~195 ms | ~370 ms |

Two things fall out of that table:

- **More threads than cores makes p95 worse, not better.** Under a 2-core limit,
  2 threads gave 621 ms and 8 threads gave 905 ms. Scale replicas, not threads.
- **Language ID costs ~180 ms** and does not scale down with clip length, because
  Whisper pads every input to 30 s of mel. It is therefore **off by default in
  `docker-compose.yml`** — an optional field must not put the required
  contract's SLO at risk. Set `VA_ENABLE_LANGUAGE_ID=true` (and ideally
  `cpus: 6.0`) to see it; it is fully implemented and tested.

**Honest caveat about the container numbers.** Docker Desktop on Apple silicon
is a Linux VM with virtualised CPU, and the manylinux aarch64 PyTorch wheel has
no Accelerate-class BLAS — which is why the container is ~4× slower than the
host on the PyTorch path and why the ONNX backend exists. Production targets
`linux/amd64` on real hardware, where PyTorch links oneDNN/MKL and the numbers
differ again. **Measure on your deployment target; do not read either column
here as a prediction for it.**

**Streaming**, 5 s clip fed at 1× wall clock in 200 ms frames:

```
     t  chunks  speech   gender   conf      age   conf      quality  stable
   2.4      12     2.1  unknown   0.54  unknown   0.29     degraded   False
   3.6      18     3.3     male   0.63    31-45   0.39     degraded   False
   4.9      24     4.3     male   0.94    31-45   0.60         good   False
   5.3      25     4.3     male   1.00    31-45   0.64         good    True  <- FINAL
```

Confidence rises monotonically as evidence accumulates, the service abstains
until it has enough, and `stable` flips once further audio stops changing the
answer — which is the point of streaming here.

**Capacity, and what happens past it.** The 500 ms target is a per-request SLO
at designed load, not a promise under arbitrary concurrency. One 4-core pod
sustains roughly **4 requests/sec**. Driving 40 concurrent requests at a single
pod gives 22 served (p95 1.1 s) and 18 shed with 429 — degraded but bounded and
visible. Past ~4 req/s the answer is more replicas, not bigger thread counts;
see [DESIGN.md](DESIGN.md) for the arithmetic.

Two settings are coupled and must not be tuned independently:

- `VA_INFERENCE_THREADS` bounds how many inferences actually **run**.
- `VA_MAX_CONCURRENT_INFERENCES` bounds how many are **admitted**, and is
  *derived* from the first (× 2) unless you override it.

Setting the second much higher than the first is the subtle failure: the
surplus does not fail fast, it queues invisibly inside the thread pool where
nothing measures it and no timeout applies. Measured with an 8-slot semaphore
over a 2-thread pool, 40 concurrent requests produced a **p95 of 2.1 s** — four
times the SLO — while the service reported itself healthy and shed only the
requests it never admitted. Coupling the two halved it.

---

## Two things I got wrong first

Both are in the code as comments and both are guarded by tests, because they
are the kind of bug that does not announce itself.

**1. The model card's label order is wrong.** It documents the gender head as
`child, female, male`. The checkpoint's own `config.json` says
`{0: female, 1: male, 2: child}`. Following the card inverts every prediction —
and does so *silently*: the service still returns well-formed JSON with 0.99
confidence, just with the labels swapped. No schema assertion catches a
permutation; only running known-gender audio does (0/6 correct with the card's
order, 6/6 with the config's). Fixed by resolving indices from
`config.id2label` at load time, guarded by `scripts/verify_gender_mapping.py`
and a slow test.

```bash
make verify     # 6/6 correct
```

**2. Spectral roll-off does not detect narrowband audio.** The obvious
bandwidth measure, and it does not work: speech energy is so heavily
low-frequency that the 95% roll-off of *clean wideband* speech sits at ~1297 Hz
— versus ~1281 Hz for a G.711 narrowband leg. A 1% difference. The 4–8 kHz band
energy ratio separates the same pair by **17×** (0.085% vs 0.005%), because
fricatives put real energy up there and an 8 kHz-sampled signal has none.
Replaced accordingly.

**3. A shared VAD instance segfaulted under concurrency.** The quality gate
originally held one `lru_cache`d Silero VAD for the whole process. Silero is a
*recurrent* TorchScript module with mutable hidden state, and the gate runs in a
thread pool — so parallel requests called `forward()` and `reset_states()` on the
same object at once and the process died with SIGSEGV inside torch's
`Module._call_impl`. It survived every test I had, because they were all
sequential, and no single-threaded test can reveal a data race. Beyond the
crash it was the exact cross-request leak PRIVACY.md promises does not happen:
interleaved recurrent state means one caller's audio affects another caller's
speech detection. Fixed with thread-local instances; `tests/test_concurrency.py`
crashes the interpreter if it regresses (verified by reverting the fix).

A fourth, smaller one worth mentioning: my first fixture generator was a pure
numpy formant synthesiser, and every clip came back `insufficient`. That was not
a bug — Silero VAD correctly refuses to certify a three-formant pulse train as
speech, and a VAD loose enough to accept it would also accept engine noise. The
fixtures now use `espeak-ng` for anything that must look like a real caller, and
keep numpy synthesis for what it is genuinely good at: constructing signals with
*exactly known* SNR and clipping so the quality gate can be tested against
ground truth.

---

## Swapping the model

The model layer is a plug-in point, not just an interface. **Adding a model is
one self-contained file plus an env var** — no edit to `config.py`, `service.py`,
or any sibling backend:

```python
# app/inference/my_backend.py
from app.inference.registry import register_backend
from app.inference.types import RawPrediction

@register_backend("my-model", description="ECAPA embeddings + a trained head")
class MyBackend:
    name = "my-model"
    def __init__(self, settings): ...
    def load(self): ...          # materialise weights
    def warmup(self): ...        # one throwaway pass; not optional
    @property
    def ready(self) -> bool: ...
    def predict(self, samples, sample_rate) -> RawPrediction: ...
```

```bash
VA_BACKEND=my-model
```

That is not an aspiration — `tests/test_extensibility.py` defines a backend
**inside the test file**, registers it, and drives it through the real HTTP API.
It also asserts that `config.py` and `service.py` do not name it, so a future
hardcoded branch fails the build.

This matters more than usual here: the default checkpoint is
**non-commercially licensed**, so replacing it is on the critical path to
production, not a hypothetical.

### Two seams, deliberately separate

| | `AttributeBackend` | `LanguageBackend` |
|---|---|---|
| Contract | `predict() -> RawPrediction` | `identify() -> LanguagePrediction \| None` |
| Failure | fatal — the service cannot serve | the field goes `null`, request still 200 |
| Selected by | `VA_BACKEND` | `VA_LANGUAGE_BACKEND` |
| Ships | `audeering` (torch), `onnx`, `mock` | `whisper` |

They are separate Protocols rather than one interface with an optional method,
because the failure semantics genuinely differ. Folding LID into the attribute
interface would force every attribute backend to stub something it has no
opinion about — and a broken LID model would then be able to fail a request it
should only have degraded. There is a test for exactly that
(`test_a_failing_language_backend_does_not_break_the_response`).

### What the seam deliberately hides

A backend returns `RawPrediction` — a continuous age in years and three raw
class probabilities. It never sees the API's brackets, the confidence
thresholds, or the quality shrinkage; all of that lives in `calibration.py`.
So a new backend cannot accidentally couple itself to the HTTP contract, and
re-cutting the brackets (`25-35`/`36-50`/…) touches no backend at all.

The custom backend in the test suite uses **no torch, no transformers and no
weights**, which is the point: the interface is not secretly PyTorch-shaped.
The migration paths in [Model licence](#model-licence) — ECAPA embeddings, a
vendor API, a remote GPU tier — are not all `nn.Module`s, and the seam has to
survive that.

### `VA_BACKEND=auto`

Each backend advertises its own availability and priority through
`@register_backend`, so `auto` selection contains no knowledge of any specific
backend. That is how the platform-dependent choice (ONNX on Linux, PyTorch on
macOS — see [Measured performance](#measured-performance)) stays out of the
orchestration layer.

---

## Thread safety

The event loop runs on one thread; every model forward pass runs on a small
`ThreadPoolExecutor`. So each shared object is reachable from several threads at
once, and each one needs a stated reason it is safe. Audited component by
component and stress-tested rather than assumed:

| Shared object | Why it is safe | Verified by |
|---|---|---|
| **Silero VAD** | **Thread-local instance per worker.** It is a *recurrent* TorchScript module with mutable hidden state — a shared one segfaulted. | `test_quality_gate_is_thread_safe` (crashes the interpreter if reverted) |
| wav2vec2 (PyTorch) | `eval()` + `torch.inference_mode()`; forward mutates no module state | `test_backend_predictions_are_stable_across_threads` |
| ONNX session | `InferenceSession.run()` is thread-safe by design; one session serves all workers | 48-call concurrent stress, 0 mismatches |
| Whisper (language ID) | Same as wav2vec2; `past_key_values` are returned, not stored | 16-call concurrent stress, 0 mismatches |
| Backend `load()` | `threading.Lock` + **publish-last ordering** (below) | `test_*_publishes_state_before_its_flag` |
| Feature extractors | Stateless transforms | covered by the backend stress tests |
| WebSocket sessions | Per-connection ring buffer, aggregator and ffmpeg; nothing shared but the executor | `test_concurrent_requests_are_independent`, 16 simultaneous live sessions |
| Prometheus metrics | `prometheus_client` collectors are thread-safe | — |
| `asyncio.Semaphore` | Not thread-safe, but only ever touched from the event loop | — |

**Publish-last ordering.** `predict()` does its cold-path `if self._model is
None: self.load()` check *outside* the load lock. So `load()` must assign the
"am I loaded?" flag **last**, after everything that flag implies. Getting this
backwards let a concurrent caller see a usable model beside an empty
`_gender_index` and silently fall back to a hardcoded label order — which is
correct for this checkpoint and silently *inverted* for a differently-ordered
one. All three backends now build into locals and publish the flag last.

That invariant is asserted **deterministically**, not by racing: the test wraps
`__setattr__` and checks the assignment sequence. Racing for it is useless — the
window is microseconds wide, and an earlier version of that test which did race
passed happily with the bug reintroduced.

**Evidence.** 100 mixed requests at 20-way concurrency produced exactly **one
distinct outcome per input class** — bit-identical results regardless of
concurrency — with no crashes or restarts.

**What is *not* thread-safe, by design:** `AnalyzerService` is single-instance
per process and its `startup()`/`shutdown()` are not reentrant; the app calls
them once from the lifespan hook.

---

## Testing

```bash
make test          # 111 tests, ~7 s, mock backend — no weights needed
make export-onnx   # once, so the ONNX parity tests have a graph to compare
make test-slow     # 24 tests, real weights: label mapping, latency, ONNX parity
```

Without `make export-onnx` the 7 parity tests skip (they need an exported graph;
the container has one baked in, a fresh checkout does not). The remaining 17
slow tests run either way.

The split is deliberate: the fast suite is usable as a pre-commit gate because
it never downloads a gigabyte of weights, and it asserts *contract and policy*,
never model accuracy — a contract test that depended on model output would start
failing when weights changed, for reasons unrelated to the contract.

What the slow suite covers that the mock cannot: the gender label mapping is not
permuted, the latency budget actually holds on real weights, the two backends
agree numerically, and Silero's recurrent state does not leak between calls.

Highlights worth reading: `test_age_on_a_boundary_splits_between_neighbours`
(the calibration idea in one assertion), `test_quality_degrades_monotonically_with_noise`
(the graceful-degradation guarantee), `test_pure_tone_is_not_speech`,
`test_contact_id_is_a_fresh_uuid_per_request`, and
`test_fast_path_is_bit_identical_to_ffmpeg`.

---

## Evaluation harness

```bash
# Common Voice extract you downloaded (the citable path)
python eval/run_eval.py --adapter local --path ~/cv-corpus-17.0-en --limit 500

# Community HF mirror, streamed (zero-setup smoke test)
python eval/run_eval.py --adapter hf --limit 200

# Your own call recordings, named <gender>_<age>_*.wav  <- the one that matters
python eval/run_eval.py --adapter dir --path ./recordings

# Noise-robustness curve
python eval/run_eval.py --adapter local --path ~/cv --noise-snr 20 10 5 0
```

> **Note.** Mozilla moved Common Voice off Hugging Face to the Mozilla Data
> Collective in October 2025 — the `mozilla-foundation/common_voice_*` repos are
> now empty, so the `load_dataset("mozilla-foundation/...")` line in most
> published examples fails today. Hence the three adapters, and hence the `hf`
> one pointing at a community mirror with a warning attached.

Reports gender accuracy / macro-F1 / confusion, age-bracket accuracy, age MAE
and bias, **ECE with a reliability table**, latency percentiles, and a **fitted
σ** to feed back into `VA_AGE_SIGMA_YEARS`.

Two things it does deliberately:

- **It runs the whole pipeline**, not the bare model — including the quality
  gate, the abstentions, and the thresholds. Evaluating the raw model would
  measure something the service never emits.
- **It reports coverage next to accuracy.** Coverage is trivially maximised by
  never saying `unknown`, and accuracy is trivially maximised by saying it
  constantly. Neither number can be gamed without showing up in the other, so
  they are only meaningful side by side. On the degradation curve, *falling
  coverage with held accuracy* is the intended behaviour — it means the service
  is refusing rather than guessing.

ECE is the metric that validates the whole calibration design: if a reported
0.8 is right only 55% of the time, a downstream `if confidence > 0.8` branch is
broken, and no accuracy number would have told you.

---

## Configuration

All via `VA_`-prefixed environment variables. Full list with rationale in
[`app/config.py`](app/config.py).

| Variable | Default | Notes |
|---|---|---|
| `VA_BACKEND` | `auto` | `auto` (platform-aware: ONNX on Linux, torch on macOS) \| `audeering` \| `onnx` \| `mock` |
| `VA_AGE_GENDER_MODEL` | `...6-ft-age-gender` | `...24-ft-...` for accuracy at ~3× compute |
| `VA_TORCH_THREADS` | `2` | Keep ≤ cores; more makes p95 worse |
| `VA_MAX_CONCURRENT_INFERENCES` | *derived* | `inference_threads × 2`. Beyond this → 429. Don't set it independently — see below. |
| `VA_MAX_QUEUE_WAIT_MS` | `150` | Bounded queue before shedding |
| `VA_INFERENCE_THREADS` | `2` | Concurrent inferences. `1` starves language ID. |
| `VA_AGE_SIGMA_YEARS` | `8.0` | Age-error σ; fit it with `eval/run_eval.py` |
| `VA_GENDER_MIN_CONFIDENCE` | `0.60` | Below → `unknown` |
| `VA_AGE_MIN_CONFIDENCE` | `0.34` | Below → `unknown` (4 brackets) |
| `VA_MIN_SPEECH_SECONDS` | `1.0` | Below → `insufficient` |
| `VA_DEGRADED_SNR_DB` | `3.0` | Below → `insufficient` |
| `VA_ENABLE_LANGUAGE_ID` | `true` | Bonus field; never blocks the response |
| `VA_LANGUAGE_BUDGET_MS` | `250` | Over budget → `language: null` |

---

## Known limitations

Ordered by how much they should worry you.

1. **Out of domain for the actual use case.** Training corpora are read and
   wideband speech (Common Voice, TIMIT) plus YouTube (VoxCeleb2). A driver on a
   mobile in a truck cab is none of those, and accented non-Western English is
   underrepresented in all of them. The noise-degradation curve in `eval/`
   measures the cost; fine-tuning on real telephony is the fix and the single
   highest-value next step.

2. **Age from voice is weakly identifiable, inherently.** The source paper
   reports MAE 7.1–10.8 years. Against 15-year brackets that means boundary
   cases are genuinely ambiguous — which is why confidence is *calibrated*
   rather than asserted, and why `unknown` is a frequent and correct answer.
   No amount of engineering fixes this; it is a property of the signal.

3. **Gender is modelled as binary + child.** That is what the model outputs; it
   does not reflect how gender works. The API returns `unknown` rather than
   forcing a call when the model is unsure, but a non-binary or transgender
   caller may be misgendered by a model that has no category for them. Worth an
   explicit product decision about whether to use this field at all.

4. **The Gaussian error assumption is a simplification.** Real age-estimation
   error is heteroscedastic (worse at the extremes) and skewed at the ends. A
   single σ cannot express that. It is defensible as a first approximation,
   and — importantly — it is *measurable*: the harness reports ECE, so the
   assumption is checked rather than trusted.

5. **Quality thresholds are hand-tuned heuristics.** Validated against
   constructed signals with known SNR and clipping, and against a noise ladder,
   but not fitted to labelled call data. They are all env-tunable because the
   right cut-point depends on the telephony vendor.

6. **Language ID is best-effort and deliberately weak.** Whisper's language
   token is a by-product of an ASR objective, so it is worse than a purpose-built
   LID head (SpeechBrain's VoxLingua107 ECAPA would be more accurate). It was
   chosen to avoid a second inference stack and a second encumbered licence for
   an optional field. It is thresholded, budgeted, and nullable.

7. **No accent detection.** The bonus mentions "language / accent"; only
   language is implemented. Accent ID needs a model with accent labels, and the
   honest answer is that a bad accent classifier is worse than none.

8. **Linear resampling on the raw-PCM WebSocket path.** File uploads get
   ffmpeg's soxr. The raw path uses linear interpolation for off-rate input,
   which aliases — send 16 kHz on that path.

9. **Single-speaker assumption.** No diarisation. On a call with two voices the
   prediction is over whoever dominates the window.

---

## Model licence

**The default model is licensed CC-BY-NC-SA-4.0 — non-commercial use only.**

That is fine for an evaluation exercise and **not fine for a product**. It is
flagged here, in `app/inference/audeering.py`, and as a warning log line at
startup, because it is the kind of thing that is easy to miss and expensive to
discover late.

Mitigation: inference sits behind the `AttributeBackend` protocol
(`app/inference/base.py`), so replacing it is one new file and one env var
rather than a rewrite of the request path. A commercially-clean route:

1. Extract speaker embeddings with an Apache-2.0 model (e.g. SpeechBrain
   ECAPA-TDNN, `spkrec-ecapa-voxceleb`).
2. Train small age-regression and gender-classification heads on licence-clean,
   consented data — ideally your own labelled calls, which also fixes limitation
   #1 above.
3. Implement `AttributeBackend` in one new file and set `VA_BACKEND`. The
   quality gate, calibration, streaming, ops and API layers are untouched —
   see [Swapping the model](#swapping-the-model), which is proven by a test
   rather than asserted here.

Everything else in the stack is permissive: Silero VAD (MIT), Whisper (MIT),
espeak-ng (GPLv3, build-time fixture tool only, not linked), ffmpeg (LGPL),
FastAPI/PyTorch/ONNX Runtime (MIT/BSD/Apache-2.0).

---

## Repository layout

```
app/
  main.py              FastAPI app, middleware, lifespan
  config.py            every tunable, env-driven
  schemas.py           the wire contract
  service.py           pipeline orchestration + concurrency model
  errors.py            typed errors; the 200-vs-4xx policy
  audio/
    decode.py          ffmpeg pipes + the WAV fast path
    quality.py         the quality gate  <- the interesting one
    ring.py            streaming window + evidence-weighted aggregator
  inference/
    base.py            the two Protocols (attributes, language)
    registry.py        @register_backend -- the plug-in point
    types.py           RawPrediction + shared label resolution
    calibration.py     age regression -> brackets  <- the other interesting one
    audeering.py       PyTorch backend
    onnx_backend.py    ONNX Runtime backend
    language.py        best-effort language ID
    mock.py            deterministic stand-in for fast tests
  routers/             analyze.py, stream.py, health.py
eval/                  adapters.py, metrics.py, run_eval.py
scripts/               fixtures, ONNX export, label-mapping check, smoke test, WS demo
tests/                 91 fast + 18 slow
```

Further reading: [DESIGN.md](DESIGN.md) (the write-up and scaling arithmetic) ·
[PRIVACY.md](PRIVACY.md) (the PII data path, end to end).
