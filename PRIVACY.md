# Privacy: how caller audio is handled

The assignment requires that no audio be stored beyond the duration of a
request and that caller audio be treated as PII. This document states exactly
what the service does, so the claim can be checked against the code rather than
taken on trust.

Voice is biometric data. Under GDPR Art. 9 it is a special category when
processed to identify a person, and under India's DPDP Act it is personal data
requiring a lawful basis. Inferred attributes — someone's estimated age and
gender — are themselves personal data about that person, whether or not they
are correct.

---

## The data path, end to end

```
   caller audio bytes
        │
        ▼
   [1] HTTP request body / WebSocket frame        in memory (bytes)
        │
        ▼
   [2] ffmpeg subprocess  ── stdin pipe ──▶ stdout pipe    NO FILE
        │                    (or: WAV fast path, pure numpy, no subprocess)
        ▼
   [3] float32 numpy array                        in memory
        │
        ├──▶ Silero VAD ─────▶ scalars (speech seconds, SNR, ...)
        ├──▶ wav2vec2 / ONNX ─▶ scalars (age float, 3 probabilities)
        └──▶ whisper-tiny ────▶ scalar  (language code, confidence)
        │
        ▼
   [4] buffer zero-filled and dereferenced        `finally:` block
        │
        ▼
   [5] JSON response: predictions + confidences only
```

Nothing crosses from step 3 to any durable medium. What leaves the process is
the JSON in step 5 and the log/metric scalars described below.

---

## Guarantees, and where each is implemented

**Audio never touches disk.**
ffmpeg is invoked with `-i pipe:0 ... pipe:1` and fed over stdin
([`app/audio/decode.py`](app/audio/decode.py)). There is no `tempfile`, no
`NamedTemporaryFile`, no upload directory. This is the main reason the service
shells out to ffmpeg over pipes rather than using a library that wants a path.
The uncompressed-WAV fast path skips the subprocess entirely and parses the
RIFF header in numpy, so that case never leaves the process at all.

**No volumes, and a read-only root filesystem.**
`docker-compose.yml` mounts no volumes, sets `read_only: true`, and provides
only a 64 MB `noexec,nosuid` tmpfs at `/tmp`. Even a future code change that
tried to write audio would fail.

**Buffers are wiped, not merely dropped.**
`DecodedAudio.wipe()` zero-fills the array and is called from a `finally`, as
is `SlidingWindow.clear()` for streaming sessions. Python cannot force
collection, but the bytes are gone before the allocation returns to the
allocator and can be handed to another request.

**No caching or cross-request state keyed on audio.**
There is no memoisation of predictions, no content-addressed store, no request
replay buffer. Two identical uploads are two independent computations.

**Silero VAD state cannot cross between callers.**
The VAD is a recurrent model with mutable hidden state, and quality assessment
runs in a thread pool — so this needs two mechanisms, not one:

- `reset_states()` before and after each buffer, so state cannot carry from one
  request to the next *on the same thread*.
- **A separate VAD instance per thread** (`threading.local()`), so concurrent
  requests on different threads cannot interleave state at all.

The second one was added after the first proved insufficient. A single shared
instance did not merely risk a subtle leak — it **segfaulted the process**
(SIGSEGV inside torch's `Module._call_impl`) under a handful of parallel
requests, because a recurrent TorchScript module is not safe to call from
several threads at once. The original test only made *sequential* calls, so it
never exercised the race.

Both properties are now asserted:
`tests/test_concurrency.py::test_silence_never_detects_speech_under_load` runs
silent and loud buffers through the gate from eight threads simultaneously and
requires the silent ones to still be judged `insufficient`, and
`tests/test_model_integration.py::test_vad_state_does_not_leak_between_calls`
covers the sequential case. Reverting to a shared instance crashes the test
suite rather than passing it quietly.

**`contact_id` is not derived from the audio.**
It is a fresh `uuid4` per request. A hash of the waveform would be a stable
biometric identifier: it would let two calls from the same person be linked
across time without ever storing audio, which is exactly the harm this section
exists to prevent. `tests/test_api_contract.py::test_contact_id_is_a_fresh_uuid_per_request`
asserts that identical audio yields different ids. A caller may supply their
own `contact_id` for correlation, in which case the linkage is their choice and
their responsibility.

**The container makes no outbound network calls.**
Weights are baked in at build time and the runtime sets `HF_HUB_OFFLINE=1` and
`TRANSFORMERS_OFFLINE=1`. There is no telemetry and no model-hub call at
request time. A service that cannot reach the network cannot exfiltrate audio.

---

## What is logged

Only scalars derived from the audio — never the audio, never a transcript,
never a filename.

| Logged | Not logged |
|---|---|
| `request_id`, `contact_id` | audio bytes, in any encoding |
| byte count, duration, sample rate | file names, upload paths |
| `speech_seconds`, `snr_db`, `clipping_ratio`, `high_band_ratio` | the transcript (never produced) |
| quality verdict, per-stage timings | the raw feature vectors or embeddings |
| predicted labels and confidences | request query strings |

Two mechanisms enforce this rather than leaving it to review discipline:

- `scrub_processor` in [`app/logging_setup.py`](app/logging_setup.py) redacts
  any `bytes`-valued log field and any key on a denylist (`audio`, `pcm`,
  `waveform`, `transcript`, `filename`, …).
- uvicorn's access log is **disabled** and replaced with our own event. The
  built-in one records the full request line; harmless today, but it is exactly
  the thing that later grows a query string with a phone number in it.

The unhandled-exception handler logs `exc_info` and the exception *type*, never
`str(exc)` — an arbitrary exception's repr can quote fragments of its input.

Note that the language-ID model is a Whisper encoder used for its language
token only. The service takes one decoder step and reads the language tag; it
never runs generation, so a transcript is never produced and there is nothing
to leak.

---

## What this design does *not* give you

Stated plainly, because these are the questions a privacy review would ask.

- **The predictions are personal data.** "We don't store audio" is not the same
  as "we don't process personal data". Estimated age and gender are inferences
  about an identifiable person and need a lawful basis, a retention policy, and
  a place in the privacy notice — on the *caller's* side, wherever the response
  is consumed.
- **Nothing here covers the caller's consent.** This service is a component. In
  most jurisdictions the disclosure obligation sits with whoever runs the call.
- **In-flight memory is not encrypted.** Audio is plaintext in process memory
  for the request's lifetime, and would appear in a core dump. Disable core
  dumps (`RLIMIT_CORE=0`) and disable swap on the host for a production deploy.
- **TLS is assumed to terminate upstream.** The service speaks plain HTTP and
  must not be exposed directly to the internet.
- **`?debug=true` widens the response** with quality diagnostics and timings.
  Still no audio, but treat it as an internal-only flag.
- **CORS is wide open** (`allow_origins=["*"]`) for the demo. Restrict it before
  any real deployment; it is flagged in `app/main.py` rather than left as an
  accidental default.

## For a production deployment

1. Restrict CORS; terminate TLS at the edge; keep the service on a private network.
2. Set `RLIMIT_CORE=0` and disable swap on the node.
3. Ship logs to a store with a retention policy — the metadata is not audio, but
   `contact_id` plus a timestamp plus an inferred gender is still a record about
   a person.
4. Decide and document a retention policy for the *responses*, which is where
   the personal data actually lives.
5. Consider whether inferring these attributes is appropriate at all for your
   use case, and whether callers are told. That is a product question, not an
   engineering one, and it is the most important item on this list.
