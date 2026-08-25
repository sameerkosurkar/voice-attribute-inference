"""The wire contract, exactly as specified in the assignment.

These run against the mock backend, so they are fast and assert *shape and
policy*, never model accuracy. That separation is deliberate: a contract test
that also depended on model output would start failing when weights changed,
for reasons that have nothing to do with the contract.
"""

from __future__ import annotations

import httpx
import numpy as np
import pytest

from scripts.make_sample_audio import wav_bytes

GENDERS = {"male", "female", "unknown"}
BRACKETS = {"18-30", "31-45", "46-60", "60+", "unknown"}
QUALITIES = {"good", "degraded", "insufficient"}


@pytest.fixture
async def client(app_client):
    application, transport = app_client
    async with application.router.lifespan_context(application):
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            yield c


# ------------------------------------------------------------------- contract
async def test_response_matches_the_specified_schema(client, speech_wav):
    response = await client.post(
        "/analyze", files={"audio": ("call.wav", speech_wav, "audio/wav")}
    )
    assert response.status_code == 200
    body = response.json()

    assert set(body) >= {"contact_id", "gender", "age_bracket",
                         "processing_ms", "audio_quality"}
    assert body["gender"]["prediction"] in GENDERS
    assert body["age_bracket"]["prediction"] in BRACKETS
    assert body["audio_quality"] in QUALITIES
    assert isinstance(body["processing_ms"], int)
    assert 0.0 <= body["gender"]["confidence"] <= 1.0
    assert 0.0 <= body["age_bracket"]["confidence"] <= 1.0


async def test_contact_id_is_a_fresh_uuid_per_request(client, speech_wav):
    """It must not be derived from the audio.

    A hash of the waveform would be a stable biometric identifier and would let
    two calls from the same person be linked -- exactly what PRIVACY.md
    promises the service does not do. Identical audio, different ids.
    """
    import uuid

    ids = set()
    for _ in range(3):
        body = (await client.post(
            "/analyze", files={"audio": ("c.wav", speech_wav, "audio/wav")}
        )).json()
        uuid.UUID(body["contact_id"])          # well-formed
        ids.add(body["contact_id"])
    assert len(ids) == 3, "contact_id must not be a function of the audio"


async def test_caller_supplied_contact_id_is_echoed(client, speech_wav):
    body = (await client.post(
        "/analyze?contact_id=order-4472",
        files={"audio": ("c.wav", speech_wav, "audio/wav")},
    )).json()
    assert body["contact_id"] == "order-4472"


async def test_optional_fields_are_omitted_not_null(client, speech_wav):
    """A client coded against the documented contract must not have to handle
    keys it was never told about."""
    body = (await client.post(
        "/analyze", files={"audio": ("c.wav", speech_wav, "audio/wav")}
    )).json()
    assert "quality_detail" not in body
    assert "timings" not in body


async def test_debug_adds_diagnostics(client, speech_wav):
    body = (await client.post(
        "/analyze?debug=true", files={"audio": ("c.wav", speech_wav, "audio/wav")}
    )).json()
    assert set(body["timings"]) >= {"decode_ms", "quality_ms", "inference_ms", "total_ms"}
    assert set(body["quality_detail"]) >= {"speech_seconds", "snr_db",
                                           "clipping_ratio", "high_band_ratio"}


# ----------------------------------------------------------------- transports
async def test_accepts_a_raw_body(client, speech_wav):
    """Telephony bridges stream bytes and will not build a MIME envelope."""
    response = await client.post(
        "/analyze", content=speech_wav, headers={"content-type": "audio/wav"}
    )
    assert response.status_code == 200


async def test_multipart_and_raw_body_agree(client, speech_wav):
    multipart = (await client.post(
        "/analyze", files={"audio": ("c.wav", speech_wav, "audio/wav")}
    )).json()
    raw = (await client.post(
        "/analyze", content=speech_wav, headers={"content-type": "audio/wav"}
    )).json()
    assert multipart["gender"] == raw["gender"]
    assert multipart["age_bracket"] == raw["age_bracket"]
    assert multipart["audio_quality"] == raw["audio_quality"]


# --------------------------------------------------------------- error policy
async def test_unusable_audio_is_a_200_not_an_error(client, silence):
    """The policy decision at the heart of the service.

    A driver answering in a loud cab and saying nothing is ORDINARY traffic,
    not a client bug. It returns 200 with insufficient/unknown so the voice
    agent falls back to a neutral persona on the normal code path.
    """
    response = await client.post(
        "/analyze", content=wav_bytes(silence), headers={"content-type": "audio/wav"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["audio_quality"] == "insufficient"
    assert body["gender"]["prediction"] == "unknown"
    assert body["age_bracket"]["prediction"] == "unknown"
    assert body["gender"]["confidence"] == 0.0
    assert body["age_bracket"]["confidence"] == 0.0


async def test_unusable_audio_skips_inference(client, silence):
    """No point spending the entire latency budget on audio already known to
    be unusable."""
    body = (await client.post(
        "/analyze?debug=true", content=wav_bytes(silence),
        headers={"content-type": "audio/wav"},
    )).json()
    assert body["timings"]["inference_ms"] == 0.0


@pytest.mark.parametrize(
    "payload, expected_status, expected_code",
    [
        (b"", 400, "EMPTY_AUDIO"),
        (b"not audio at all" * 200, 415, "DECODE_FAILED"),
    ],
)
async def test_malformed_input_is_a_typed_4xx(client, payload, expected_status,
                                              expected_code):
    """Malformed input IS a client bug and gets a 4xx with a stable code."""
    response = await client.post(
        "/analyze", content=payload, headers={"content-type": "audio/wav"}
    )
    assert response.status_code == expected_status
    assert response.json()["error"] == expected_code


async def test_too_short_audio_is_rejected(client):
    response = await client.post(
        "/analyze",
        content=wav_bytes(np.zeros(1_600, dtype=np.float32)),
        headers={"content-type": "audio/wav"},
    )
    assert response.status_code == 400
    assert response.json()["error"] == "AUDIO_TOO_SHORT"


async def test_oversized_upload_is_rejected(client):
    response = await client.post(
        "/analyze",
        content=b"\x00" * (26 * 1024 * 1024),
        headers={"content-type": "audio/wav"},
    )
    assert response.status_code == 413
    assert response.json()["error"] == "AUDIO_TOO_LARGE"


async def test_errors_share_one_envelope(client):
    body = (await client.post(
        "/analyze", content=b"", headers={"content-type": "audio/wav"}
    )).json()
    assert set(body) >= {"error", "message"}
    assert isinstance(body["error"], str) and body["error"].isupper()


# ---------------------------------------------------------------------- ops
async def test_health_is_up_and_ready_is_ready(client):
    assert (await client.get("/health")).json()["status"] == "ok"
    assert (await client.get("/ready")).json()["status"] == "ready"


async def test_metrics_are_exposed(client, speech_wav):
    await client.post("/analyze", content=speech_wav,
                      headers={"content-type": "audio/wav"})
    text = (await client.get("/metrics")).text
    assert "va_request_duration_seconds" in text
    assert "va_audio_quality_total" in text
    assert "va_stage_duration_seconds" in text


async def test_request_id_round_trips(client, speech_wav):
    response = await client.post(
        "/analyze", content=speech_wav,
        headers={"content-type": "audio/wav", "x-request-id": "trace-123"},
    )
    assert response.headers["x-request-id"] == "trace-123"
    assert response.json()["request_id"] == "trace-123"


async def test_openapi_documents_the_endpoint(client):
    schema = (await client.get("/openapi.json")).json()
    assert "/analyze" in schema["paths"]
    assert "/ws/analyze" not in schema["paths"]  # websockets are not in OpenAPI
