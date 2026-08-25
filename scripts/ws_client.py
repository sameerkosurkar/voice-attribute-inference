#!/usr/bin/env python3
"""Demo client for the streaming endpoint.

Streams a WAV in real time -- honouring wall-clock pacing rather than blasting
the file -- so the progressive output looks like it would on a live call.

    python scripts/ws_client.py                                  # uses a fixture
    python scripts/ws_client.py --file call.wav --url ws://localhost:8000/ws/analyze

Watch two columns as it runs: `confidence` should climb as evidence accumulates,
and `stable` should flip to True once further audio stops changing the answer.
That is the entire argument for streaming here -- a voice agent can commit to a
persona part-way through the call instead of waiting for it to end.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
import sys
import time
import wave

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))


def read_wav(path: pathlib.Path) -> tuple[bytes, int]:
    with wave.open(str(path)) as handle:
        if handle.getsampwidth() != 2 or handle.getnchannels() != 1:
            raise SystemExit(
                f"{path}: need 16-bit mono PCM for the raw path.\n"
                f"Convert with: ffmpeg -i {path} -ac 1 -ar 16000 out.wav"
            )
        return handle.readframes(handle.getnframes()), handle.getframerate()


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="ws://localhost:8000/ws/analyze")
    parser.add_argument("--file", type=pathlib.Path,
                        default=pathlib.Path("samples/adult_male_clean.wav"))
    parser.add_argument("--chunk-ms", type=int, default=200,
                        help="Frame size, mimicking a telephony bridge.")
    parser.add_argument("--realtime", action="store_true", default=True)
    parser.add_argument("--fast", dest="realtime", action="store_false",
                        help="Send as fast as possible instead of at 1x.")
    args = parser.parse_args()

    try:
        import websockets
    except ImportError:
        raise SystemExit("pip install websockets")

    if not args.file.exists():
        raise SystemExit(
            f"{args.file} not found. Generate fixtures first:\n"
            "    python scripts/make_sample_audio.py --outdir samples"
        )

    pcm, rate = read_wav(args.file)
    bytes_per_chunk = int(rate * 2 * args.chunk_ms / 1000)
    chunk_s = args.chunk_ms / 1000.0

    print(f"streaming {args.file.name}  ({len(pcm) / (rate * 2):.1f}s @ {rate} Hz) "
          f"in {args.chunk_ms} ms frames\n")

    async with websockets.connect(args.url, max_size=None) as ws:
        await ws.send(json.dumps({
            "type": "start", "format": "pcm_s16le", "sample_rate": rate,
        }))
        ready = json.loads(await ws.recv())
        print(f"session {ready.get('session_id', '?')[:8]}  "
              f"contact {ready.get('contact_id', '?')[:8]}\n")

        header = (f"{'t':>6s} {'chunks':>7s} {'speech':>7s} {'gender':>8s} "
                  f"{'conf':>6s} {'age':>8s} {'conf':>6s} {'quality':>12s} {'stable':>7s}")
        print(header)
        print("-" * len(header))

        started = time.monotonic()
        stop = asyncio.Event()

        async def receive():
            try:
                async for message in ws:
                    event = json.loads(message)
                    if event.get("type") == "error":
                        print(f"\nERROR {event.get('error')}: {event.get('message')}")
                        stop.set()
                        return
                    print(
                        f"{time.monotonic() - started:6.1f} "
                        f"{event.get('chunks_seen', 0):>7d} "
                        f"{event.get('speech_seconds', 0):>7.1f} "
                        f"{event['gender']['prediction']:>8s} "
                        f"{event['gender']['confidence']:>6.2f} "
                        f"{event['age_bracket']['prediction']:>8s} "
                        f"{event['age_bracket']['confidence']:>6.2f} "
                        f"{event['audio_quality']:>12s} "
                        f"{str(event.get('stable', False)):>7s}"
                        + ("   <- FINAL" if event.get("is_final") else "")
                    )
                    if event.get("is_final"):
                        stop.set()
                        return
            except Exception:
                stop.set()

        receiver = asyncio.create_task(receive())

        for offset in range(0, len(pcm), bytes_per_chunk):
            if stop.is_set():
                break
            await ws.send(pcm[offset : offset + bytes_per_chunk])
            if args.realtime:
                await asyncio.sleep(chunk_s)

        if not stop.is_set():
            await ws.send(json.dumps({"type": "end"}))
        try:
            await asyncio.wait_for(receiver, timeout=15.0)
        except asyncio.TimeoutError:
            receiver.cancel()

    print("\nWatch `conf` rise and `stable` flip to True: the endpoint gets more")
    print("certain as evidence accumulates, which is what lets a voice agent")
    print("commit to a persona mid-call instead of waiting for it to end.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
