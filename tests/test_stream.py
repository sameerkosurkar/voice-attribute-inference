"""Streaming: the sliding window, the aggregator, and the WebSocket protocol.

The aggregator tests matter most. Progressive prediction is only useful if the
answer *converges* -- an endpoint that emits a different guess every second is
worse than one that waits, because a voice agent has to pick a persona and
commit. These assert convergence and stability explicitly.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.audio.ring import PredictionAggregator, SlidingWindow
from app.inference.calibration import RawPrediction

SAMPLE_RATE = 16_000


# ------------------------------------------------------------ sliding window
def test_window_is_bounded():
    """A 40-minute call must cost the same memory as a 40-second one."""
    window = SlidingWindow(SAMPLE_RATE, window_seconds=2.0)
    for _ in range(50):
        window.append(np.ones(SAMPLE_RATE, dtype=np.float32))
    assert len(window) == 2 * SAMPLE_RATE
    assert window.seconds == pytest.approx(2.0)
    assert window.total_seconds == pytest.approx(50.0)


def test_window_drops_oldest_not_newest():
    """Recent speech is the useful speech, and a slow consumer must never
    apply backpressure to a live call."""
    window = SlidingWindow(SAMPLE_RATE, window_seconds=1.0)
    window.append(np.full(SAMPLE_RATE, 0.1, dtype=np.float32))
    window.append(np.full(SAMPLE_RATE, 0.9, dtype=np.float32))
    assert np.allclose(window.snapshot(), 0.9)


def test_window_handles_partial_eviction():
    window = SlidingWindow(SAMPLE_RATE, window_seconds=1.0)
    window.append(np.full(SAMPLE_RATE, 0.1, dtype=np.float32))
    window.append(np.full(SAMPLE_RATE // 2, 0.9, dtype=np.float32))
    snapshot = window.snapshot()
    assert snapshot.size == SAMPLE_RATE
    assert np.allclose(snapshot[-SAMPLE_RATE // 2 :], 0.9)
    assert np.allclose(snapshot[: SAMPLE_RATE // 2], 0.1)


def test_window_clear_wipes_the_buffer():
    window = SlidingWindow(SAMPLE_RATE, window_seconds=1.0)
    window.append(np.full(SAMPLE_RATE, 0.5, dtype=np.float32))
    window.clear()
    assert len(window) == 0
    assert window.snapshot().size == 0


def test_empty_window_snapshot_is_safe():
    assert SlidingWindow(SAMPLE_RATE, 1.0).snapshot().size == 0


# ---------------------------------------------------------------- aggregator
def _raw(age=40.0, child=0.0, female=0.1, male=0.9) -> RawPrediction:
    return RawPrediction(age_years=age, p_child=child, p_female=female, p_male=male)


def test_aggregator_converges_on_consistent_evidence():
    aggregator = PredictionAggregator(alpha=0.55)
    for _ in range(6):
        aggregator.update(_raw(age=42.0), speech_seconds=2.0, quality_factor=1.0)
    snapshot = aggregator.snapshot()
    assert snapshot.age_years == pytest.approx(42.0, abs=0.5)
    assert snapshot.p_male > 0.85


def test_noisy_window_barely_moves_the_estimate():
    """The reason the EMA is weighted by speech duration and quality.

    A window that was mostly engine noise should not be able to undo several
    seconds of clean speech.
    """
    aggregator = PredictionAggregator(alpha=0.55)
    for _ in range(4):
        aggregator.update(_raw(age=40.0, female=0.05, male=0.95),
                          speech_seconds=2.0, quality_factor=1.0)
    before = aggregator.snapshot().p_male

    aggregator.update(_raw(age=70.0, female=0.9, male=0.1),
                      speech_seconds=0.2, quality_factor=0.3)
    after = aggregator.snapshot().p_male

    assert before - after < 0.1, "a low-evidence window must not dominate"


def test_zero_evidence_window_is_ignored_entirely():
    aggregator = PredictionAggregator()
    aggregator.update(_raw(), speech_seconds=0.0, quality_factor=1.0)
    assert aggregator.updates == 0
    aggregator.update(_raw(), speech_seconds=2.0, quality_factor=0.0)
    assert aggregator.updates == 0


def test_confidence_grows_with_accumulated_evidence():
    """An early partial must be visibly less certain than the final, or
    streaming progressively would be pointless."""
    aggregator = PredictionAggregator()
    aggregator.update(_raw(), speech_seconds=1.5, quality_factor=1.0)
    first = aggregator.confidence_scale
    for _ in range(6):
        aggregator.update(_raw(), speech_seconds=2.0, quality_factor=1.0)
    assert aggregator.confidence_scale > first
    assert aggregator.confidence_scale <= 1.0


def test_stability_requires_agreement_over_several_windows():
    aggregator = PredictionAggregator(alpha=0.55)
    assert aggregator.stable is False
    for _ in range(6):
        aggregator.update(_raw(age=40.0), speech_seconds=2.0, quality_factor=1.0)
    assert aggregator.stable is True


def test_disagreeing_windows_are_not_stable():
    aggregator = PredictionAggregator(alpha=0.9)
    for age in (25.0, 65.0, 25.0, 65.0):
        aggregator.update(_raw(age=age), speech_seconds=2.0, quality_factor=1.0)
    assert aggregator.stable is False


# ------------------------------------------------------------- ws protocol
@pytest.fixture
def ws_client(app_client):
    application, _ = app_client
    with TestClient(application) as client:
        yield client


def _pcm16(samples: np.ndarray) -> bytes:
    return np.clip(samples * 32767, -32768, 32767).astype("<i2").tobytes()


def test_stream_emits_ready_then_final(ws_client, speech):
    with ws_client.websocket_connect("/ws/analyze") as ws:
        ws.send_json({"type": "start", "format": "pcm_s16le", "sample_rate": 16000})
        assert ws.receive_json()["type"] == "ready"

        for offset in range(0, speech.size, SAMPLE_RATE):
            ws.send_bytes(_pcm16(speech[offset : offset + SAMPLE_RATE]))
        ws.send_text(json.dumps({"type": "end"}))

        events = []
        while True:
            event = ws.receive_json()
            events.append(event)
            if event.get("is_final"):
                break

    final = events[-1]
    assert final["type"] == "final" and final["is_final"] is True
    assert final["gender"]["prediction"] in {"male", "female", "unknown"}
    assert final["age_bracket"]["prediction"] in {"18-30", "31-45", "46-60", "60+", "unknown"}
    assert final["audio_quality"] in {"good", "degraded", "insufficient"}
    assert final["chunks_seen"] >= 1


def test_stream_reuses_the_rest_contract(ws_client, speech):
    """One parser should serve both endpoints."""
    with ws_client.websocket_connect("/ws/analyze") as ws:
        ws.send_json({"type": "start", "format": "pcm_s16le", "sample_rate": 16000})
        ws.receive_json()
        ws.send_bytes(_pcm16(speech))
        ws.send_text(json.dumps({"type": "end"}))
        while not (event := ws.receive_json()).get("is_final"):
            pass

    assert set(event) >= {"contact_id", "gender", "age_bracket",
                          "processing_ms", "audio_quality"}


def test_stream_accepts_a_caller_supplied_contact_id(ws_client, speech):
    with ws_client.websocket_connect("/ws/analyze") as ws:
        ws.send_json({"type": "start", "format": "pcm_s16le",
                      "sample_rate": 16000, "contact_id": "call-99"})
        assert ws.receive_json()["contact_id"] == "call-99"
        ws.send_bytes(_pcm16(speech))
        ws.send_text(json.dumps({"type": "end"}))
        while not (event := ws.receive_json()).get("is_final"):
            pass
    assert event["contact_id"] == "call-99"


def test_stream_without_a_handshake_still_works(ws_client, speech):
    """A client may just open the socket and start sending 16 kHz PCM."""
    with ws_client.websocket_connect("/ws/analyze") as ws:
        ws.send_bytes(_pcm16(speech[:SAMPLE_RATE]))
        assert ws.receive_json()["type"] == "ready"
        ws.send_bytes(_pcm16(speech[SAMPLE_RATE:]))
        ws.send_text(json.dumps({"type": "end"}))
        while not (event := ws.receive_json()).get("is_final"):
            pass
    assert event["is_final"] is True


def test_stream_on_silence_ends_unknown(ws_client, silence):
    with ws_client.websocket_connect("/ws/analyze") as ws:
        ws.send_json({"type": "start", "format": "pcm_s16le", "sample_rate": 16000})
        ws.receive_json()
        ws.send_bytes(_pcm16(silence))
        ws.send_text(json.dumps({"type": "end"}))
        while not (event := ws.receive_json()).get("is_final"):
            pass

    assert event["gender"]["prediction"] == "unknown"
    assert event["audio_quality"] == "insufficient"


def test_stream_survives_an_abrupt_disconnect(ws_client, speech):
    """A dropped call must not leave a wedged ffmpeg or a leaked session."""
    with ws_client.websocket_connect("/ws/analyze") as ws:
        ws.send_json({"type": "start", "format": "pcm_s16le", "sample_rate": 16000})
        ws.receive_json()
        ws.send_bytes(_pcm16(speech[:SAMPLE_RATE]))
    # Exiting the context manager closes without an "end" frame. The next
    # session must still work.
    with ws_client.websocket_connect("/ws/analyze") as ws:
        ws.send_json({"type": "start", "format": "pcm_s16le", "sample_rate": 16000})
        assert ws.receive_json()["type"] == "ready"


# ------------------------------------------------- compressed codec path ---
@pytest.mark.skipif(
    "not __import__('shutil').which('ffmpeg')", reason="needs ffmpeg"
)
def test_stream_accepts_a_compressed_codec(ws_client, speech):
    """The browser / WebRTC path: opus in a WebM container, fed incrementally.

    This exercises a genuinely different code path from raw PCM -- one
    long-lived ffmpeg subprocess per connection, written to over stdin and
    drained from stdout by a background task -- so it needs its own test rather
    than riding on the PCM one.
    """
    import subprocess

    from scripts.make_sample_audio import wav_bytes

    encoded = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", "pipe:0",
         "-c:a", "libopus", "-b:a", "32k", "-f", "webm", "pipe:1"],
        input=wav_bytes(speech), capture_output=True, check=True,
    ).stdout
    assert encoded, "ffmpeg produced no webm"

    with ws_client.websocket_connect("/ws/analyze") as ws:
        ws.send_json({"type": "start", "format": "webm"})
        assert ws.receive_json()["type"] == "ready"
        for offset in range(0, len(encoded), 8192):
            ws.send_bytes(encoded[offset : offset + 8192])
        ws.send_text(json.dumps({"type": "end"}))
        while not (event := ws.receive_json()).get("is_final"):
            pass

    assert event["is_final"] is True
    assert event["audio_quality"] in {"good", "degraded", "insufficient"}
    # The decoder must actually have produced audio -- a broken ffmpeg pipe
    # would silently yield an empty window and an "insufficient" verdict.
    assert event["speech_seconds"] > 1.0, "compressed stream decoded to nothing"


def test_stream_rejects_an_undecodable_stream_without_hanging(ws_client):
    """Garbage on a compressed stream must terminate, not wedge the session."""
    with ws_client.websocket_connect("/ws/analyze") as ws:
        ws.send_json({"type": "start", "format": "webm"})
        assert ws.receive_json()["type"] == "ready"
        ws.send_bytes(b"this is not a webm container" * 100)
        ws.send_text(json.dumps({"type": "end"}))
        event = ws.receive_json()

    assert event.get("is_final") is True
    assert event["gender"]["prediction"] == "unknown"
    assert event["audio_quality"] == "insufficient"
