# Design write-up

## The 200 words

I chose `audeering/wav2vec2-large-robust-6-ft-age-gender`: one trunk with two
heads gives age *and* gender in a single forward pass, its training data spans
telephone and in-the-wild audio, and it predicts age as a **regression**, not a
class. That last property carries the design: a continuous estimate integrates
over arbitrary bracket edges, so
confidence reflects real boundary uncertainty (45.5 years honestly splits
between `31-45` and `46-60`) and brackets can be re-cut without retraining.

The service's actual contribution is refusing to use the model when it
shouldn't. A quality gate — VAD, speech-to-background ratio, clipping, 4–8 kHz
band energy — runs *before* inference and can short-circuit to `unknown`,
because wav2vec2 will confidently classify diesel engine noise and nothing in a
softmax says otherwise.

With more time: fine-tune on real logistics telephony (by far the biggest win),
replace the Gaussian error prior with a learned heteroscedastic head, and fit
the thresholds against labelled calls instead of reasoning about them.

To 1,000 concurrent calls: it's CPU-bound and stateless, so scale replicas, not
threads. A measured ~4 requests/sec per four-core pod puts steady state near six
pods and a synchronised peak near twenty-five, autoscaled on in-flight
inferences, shedding with 429 rather than queueing.

*(200 words.)*

---

## Appendix: the reasoning in more detail

### Why the quality gate is the centrepiece

The brief asks for a service that "degrades gracefully and surfaces an
`audio_quality` flag rather than silently returning bad predictions". A
classifier has no way to express "this input was not the kind of thing I was
trained on" — softmax always sums to one. So the abstention decision has to be
made *outside* the model, from properties of the signal.

Four signals, each catching a different real failure on a logistics call:

| Signal | Catches |
|---|---|
| Silero VAD speech seconds | dead air, hold music, IVR tones, "yeah?" and nothing else |
| speech-to-background ratio | there *is* speech, it is buried under a diesel engine |
| clipping ratio | wind on a handset mic, over-driven bluetooth headsets |
| 4–8 kHz band energy | narrowband G.711 legs and aggressive codecs — a genuine domain shift |

The gate runs first, so unusable audio costs ~45 ms instead of the full budget.

### Two measured findings that changed the implementation

**The model card's label order is wrong.** It annotates the gender head as
`child, female, male`; the checkpoint's `config.json` declares
`{0: female, 1: male, 2: child}`. Following the card inverts every prediction —
and does so *silently*, still returning well-formed, high-confidence JSON. No
schema or shape assertion catches a permutation. Found by running known-gender
audio through the model (0/6 correct with the card's order, 6/6 with the
config's), fixed by resolving indices from `config.id2label` at load time, and
guarded by `scripts/verify_gender_mapping.py` plus a slow test.

**"ONNX is faster" is a claim about your BLAS, not about ONNX.** Same
checkpoint, same input, 4 threads:

| | PyTorch | ONNX Runtime |
|---|---|---|
| macOS arm64 (host) | **89 ms** | 209 ms |
| Linux arm64 (container) | 453 ms | **213 ms** |

macOS PyTorch links Apple's Accelerate framework; the manylinux aarch64 wheel
does not. Both backends ship behind one interface and `VA_BACKEND=auto` picks by
what is present, because the right answer differs per deployment target and the
only way to know is to measure it there.

### What I would do with more time, in priority order

1. **Fine-tune on real logistics telephony.** Every other item is second-order.
   The corpora behind this model are read and wideband speech; a driver on a
   mobile in a cab is out of domain, and the honest evidence for that is the
   noise-degradation curve in `eval/`, not a hope.
2. **Replace the Gaussian error prior with a learned uncertainty head.** Age
   error is heteroscedastic — worse for the young and the old — and a single σ
   cannot express that. `eval/run_eval.py` already fits σ empirically as a
   stopgap and prints the value to feed back.
3. **Fit the quality thresholds rather than reason about them.** They are
   currently defensible heuristics validated on constructed signals. With
   labelled calls they become a small calibrated model, and ECE (already
   measured) becomes the objective.
4. **Replace the non-commercially-licensed model.** See the README; the backend
   seam makes this a swap rather than a rewrite.
5. **Per-speaker aggregation across a call**, not just across chunks. The
   streaming aggregator already accumulates evidence; with diarisation it could
   attribute that evidence to the right speaker on a multi-party call.

### Scaling to 1,000 concurrent calls

The arithmetic, stated so it can be argued with:

- One 5 s chunk costs ~215 ms of CPU (container ONNX path, 2 threads).
- A voice agent needs the attributes **once**, early — not continuously. Assume
  one inference per ~10 s of call for the first 30 s, then none.
- 1,000 concurrent calls ⇒ roughly 1000/10 = **100 inferences/sec at peak**, but
  only during the opening of each call; steady state with staggered arrivals is
  closer to **20–25/sec**.
- A 4-core pod sustains ~4 inferences/sec — measured, not extrapolated:
  240 ms per request with 4 torch threads, and the cores are already
  saturated by a single inference.
- ⇒ **5–6 pods** steady state, ~25 for a synchronised peak.

The properties that make this work:

- **Stateless.** No session affinity for `/analyze`. WebSocket sessions are
  sticky by nature but hold only a bounded ring buffer, so a dropped pod costs
  one call's partial results, not correctness.
- **Scale replicas, not workers or threads.** Each uvicorn worker would hold its
  own full copy of the weights in RSS, and more torch threads than cores makes
  p95 *worse* — measured: 2 threads 621 ms, 8 threads 905 ms under a 2-core
  limit.
- **Bound the queue, then shed.** A request waits up to 150 ms for a slot,
  then gets 429. Zero wait throws away requests that would have finished in
  budget; an unbounded queue hides the backlog inside the thread pool, where
  no timeout applies — measured p95 2.1 s against a 500 ms SLO. A semaphore
  sized to actual parallelism caps in-flight inferences and returns 429
  with `Retry-After`. A caller who waits 8 seconds for an age guess has already
  lost the call; failing fast lets the agent fall back to a neutral persona on
  time.
- **Autoscale on `va_inflight_inferences`**, not CPU. It is the direct measure
  of saturation and it leads CPU by a few seconds.

Beyond ~50 pods the next step is a GPU inference tier with dynamic batching
(Triton or vLLM-style), since wav2vec2 batches well and 1,000 calls generate
plenty of concurrent requests to fill a batch. That trades latency for
throughput, so it is worth it only past the point where CPU replica count
becomes the dominant cost.
