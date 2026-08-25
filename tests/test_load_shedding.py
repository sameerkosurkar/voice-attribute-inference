"""Overload behaviour.

Under saturation the service sheds with 429 rather than queueing without
bound. The reasoning: a caller who waits 8 seconds for an age guess has already
lost the call, so an unbounded queue converts a load problem into a timeout
problem for *everyone* instead of a clean failure for *some*.

But shedding on the slightest contention is equally wrong -- it throws away
requests that would have completed comfortably inside the 500 ms budget. So
there is a bounded queue in between, and these tests pin both edges of it:
work is absorbed up to `max_queue_wait_ms`, and shed past it.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest

from app.config import get_settings


@pytest.fixture
async def slow_client(monkeypatch):
    """One inference slot, no queue wait: the strictest possible shedding."""
    get_settings.cache_clear()
    monkeypatch.setenv("VA_BACKEND", "mock")
    monkeypatch.setenv("VA_ENABLE_LANGUAGE_ID", "false")
    monkeypatch.setenv("VA_MAX_CONCURRENT_INFERENCES", "1")
    monkeypatch.setenv("VA_MAX_QUEUE_WAIT_MS", "0")
    monkeypatch.setenv("VA_LOG_JSON", "false")
    get_settings.cache_clear()

    from app.main import create_app

    application = create_app()
    async with application.router.lifespan_context(application):
        transport = httpx.ASGITransport(app=application)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            yield application, c
    get_settings.cache_clear()


async def test_saturation_sheds_with_429(slow_client, speech_wav, monkeypatch):
    """With one slot and no queue, concurrent load must shed -- not pile up."""
    application, client = slow_client
    service = application.state.service

    # Make each inference slow enough that requests genuinely overlap.
    original = service.backend.predict

    def slow_predict(samples, sample_rate):
        import time

        time.sleep(0.25)
        return original(samples, sample_rate)

    service.backend.predict = slow_predict

    responses = await asyncio.gather(*[
        client.post("/analyze", content=speech_wav,
                    headers={"content-type": "audio/wav"})
        for _ in range(8)
    ])
    codes = [r.status_code for r in responses]

    assert 429 in codes, "saturation must shed, not queue without bound"
    assert 200 in codes, "at least one request must still be served"

    shed = next(r for r in responses if r.status_code == 429)
    assert shed.json()["error"] == "OVERLOADED"
    # Retry-After lets a caller back off intelligently instead of hot-looping.
    assert shed.headers.get("retry-after") == "1"


async def test_shedding_is_fast(slow_client, speech_wav):
    """A shed request must fail immediately.

    If shedding were slow it would consume the very capacity it is protecting.
    """
    application, client = slow_client
    service = application.state.service
    original = service.backend.predict

    def slow_predict(samples, sample_rate):
        import time

        time.sleep(0.3)
        return original(samples, sample_rate)

    service.backend.predict = slow_predict

    responses = await asyncio.gather(*[
        client.post("/analyze?debug=true", content=speech_wav,
                    headers={"content-type": "audio/wav"})
        for _ in range(6)
    ])
    for response in responses:
        if response.status_code == 429:
            assert response.elapsed.total_seconds() < 1.0


async def test_a_bounded_queue_absorbs_bursts(monkeypatch, speech_wav):
    """The other edge: with a queue wait configured, a short burst that fits
    inside the budget is served rather than shed."""
    get_settings.cache_clear()
    monkeypatch.setenv("VA_BACKEND", "mock")
    monkeypatch.setenv("VA_ENABLE_LANGUAGE_ID", "false")
    monkeypatch.setenv("VA_MAX_CONCURRENT_INFERENCES", "2")
    monkeypatch.setenv("VA_MAX_QUEUE_WAIT_MS", "2000")
    monkeypatch.setenv("VA_LOG_JSON", "false")
    get_settings.cache_clear()

    from app.main import create_app

    application = create_app()
    async with application.router.lifespan_context(application):
        transport = httpx.ASGITransport(app=application)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
            responses = await asyncio.gather(*[
                client.post("/analyze", content=speech_wav,
                            headers={"content-type": "audio/wav"})
                for _ in range(6)
            ])
    get_settings.cache_clear()

    assert all(r.status_code == 200 for r in responses), (
        "a burst that fits within the queue wait must be absorbed, not shed"
    )


async def test_inflight_gauge_returns_to_zero(slow_client, speech_wav):
    """The semaphore and the gauge must be released on every path, including
    the shed path -- otherwise capacity leaks away one request at a time."""
    application, client = slow_client
    from app.observability import INFLIGHT

    await asyncio.gather(*[
        client.post("/analyze", content=speech_wav,
                    headers={"content-type": "audio/wav"})
        for _ in range(6)
    ])
    await asyncio.sleep(0.1)
    assert INFLIGHT._value.get() == 0
