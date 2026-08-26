#!/usr/bin/env python3
"""Minimal WebSocket smoke check, for scripts/smoke_test.sh.

Separate from ws_client.py (which is a human-facing demo) because this one is
a pass/fail assertion with an exit code, and because smoke_test.sh runs it
either on the host or inside the container depending on what has `websockets`.

Why it exists: the REST smoke test passed happily while the streaming path was
never exercised against a real container. A refactor that broke only the
WebSocket route would have shipped green.

    .venv/bin/python scripts/ws_smoke.py --url ws://localhost:8000/ws/analyze --samples samples
"""

from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
import sys
import wave


def read_pcm(path: pathlib.Path) -> bytes:
    with wave.open(str(path)) as handle:
        if handle.getsampwidth() != 2 or handle.getnchannels() != 1:
            raise SystemExit(f"{path}: need 16-bit mono PCM")
        return handle.readframes(handle.getnframes())


async def one(url: str, payload: bytes, label: str) -> dict:
    import websockets

    async with websockets.connect(url, max_size=None) as ws:
        await ws.send(json.dumps(
            {"type": "start", "format": "pcm_s16le", "sample_rate": 16000}
        ))
        ready = json.loads(await ws.recv())
        if ready.get("type") != "ready":
            raise AssertionError(f"{label}: expected 'ready', got {ready}")

        for offset in range(0, len(payload), 6400):
            await ws.send(payload[offset : offset + 6400])
            await asyncio.sleep(0.03)
        await ws.send(json.dumps({"type": "end"}))

        events = []
        while True:
            event = json.loads(await asyncio.wait_for(ws.recv(), timeout=60))
            events.append(event)
            if event.get("is_final"):
                break
    return {"final": events[-1], "count": len(events)}


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="ws://127.0.0.1:8000/ws/analyze")
    parser.add_argument("--samples", type=pathlib.Path, default=pathlib.Path("samples"))
    args = parser.parse_args()

    speech = args.samples / "adult_male_clean.wav"
    silence = args.samples / "silence.wav"
    for path in (speech, silence):
        if not path.exists():
            print(f"missing fixture {path}", file=sys.stderr)
            return 2

    failures = 0

    try:
        result = await one(args.url, read_pcm(speech), "speech")
    except Exception as exc:
        # A dead or renamed route must fail the smoke test with a readable
        # message and a non-zero exit -- not a raw traceback that a CI log
        # reader has to decode, and certainly not a silent pass.
        print(f"  FAIL: could not complete a streaming session: "
              f"{type(exc).__name__}: {exc}")
        print(f"        url = {args.url}")
        return 1

    final = result["final"]
    print(f"  speech : {result['count']} events, final="
          f"{final['gender']['prediction']}({final['gender']['confidence']:.2f}) "
          f"{final['age_bracket']['prediction']} q={final['audio_quality']} "
          f"speech={final['speech_seconds']}s")
    if not final.get("is_final"):
        print("    FAIL: no final event"); failures += 1
    if final["audio_quality"] == "insufficient":
        print("    FAIL: clean speech judged insufficient over the stream"); failures += 1
    if final["gender"]["prediction"] == "unknown":
        print("    FAIL: no gender from clean streamed speech"); failures += 1
    required = {"contact_id", "gender", "age_bracket", "processing_ms", "audio_quality"}
    if not required <= set(final):
        print(f"    FAIL: stream event missing REST contract keys: "
              f"{sorted(required - set(final))}"); failures += 1

    try:
        result = await one(args.url, read_pcm(silence), "silence")
    except Exception as exc:
        print(f"  FAIL: silence session errored: {type(exc).__name__}: {exc}")
        return 1
    final = result["final"]
    print(f"  silence: final={final['gender']['prediction']} q={final['audio_quality']}")
    if final["audio_quality"] != "insufficient" or final["gender"]["prediction"] != "unknown":
        print("    FAIL: streamed silence must abstain"); failures += 1

    if failures:
        print(f"  WEBSOCKET SMOKE FAILED ({failures} checks)")
        return 1
    print("  websocket ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
