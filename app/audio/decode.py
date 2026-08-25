"""Container/codec-agnostic decoding to mono float32 PCM at 16 kHz.

Why shell out to ffmpeg instead of using a Python audio library:

  1. Coverage. Logistics traffic arrives as whatever the telephony vendor
     hands us -- mu-law/A-law G.711 from a PSTN leg, Opus from a WebRTC leg,
     mp3 from a recorded voicemail, WebM from a browser. ffmpeg decodes all of
     them with one code path; soundfile/librosa do not.
  2. Privacy. stdin -> stdout pipes mean the audio never becomes a file. No
     tempfile, so nothing to leak via /tmp, a crash dump, or a container layer.
     This is the single most important property in this module.
  3. Resampling quality. ffmpeg's soxr resampler is better than a naive
     decimation, and wav2vec2 is sensitive to the 16 kHz assumption.

The cost is a subprocess per request (~5-15 ms). Measured and accepted; see
README "Latency budget".
"""

from __future__ import annotations

import asyncio
import shutil
import struct
from dataclasses import dataclass

import numpy as np
import structlog

from app.config import Settings
from app.errors import (
    AudioTooLargeError,
    AudioTooShortError,
    DecodeError,
    DecodeTimeoutError,
    EmptyAudioError,
)

log = structlog.get_logger(__name__)

FFMPEG = shutil.which("ffmpeg") or "ffmpeg"


@dataclass(slots=True)
class DecodedAudio:
    samples: np.ndarray          # float32, mono, in [-1, 1]
    sample_rate: int
    source_bytes: int
    original_seconds: float      # before any windowing
    windowed: bool               # True if we analysed a sub-segment

    @property
    def duration_s(self) -> float:
        return len(self.samples) / float(self.sample_rate)

    def wipe(self) -> None:
        """Zero the buffer in place. See PRIVACY.md -- we cannot force the GC
        to run, but we can guarantee the bytes are gone before the array is
        released back to the allocator and possibly handed to another request."""
        try:
            self.samples.fill(0.0)
        except (ValueError, AttributeError):  # non-writeable view
            pass


# --------------------------------------------------------------------------
# Uncompressed-WAV fast path
#
# Measured: spawning ffmpeg costs ~80 ms p50 on a 156 KB file -- 16% of the
# 500 ms budget spent entirely on process creation, for a format that needs no
# decoding at all. Uncompressed 16 kHz PCM WAV is exactly what telephony
# recorders and media servers hand you, so this is the common case, not an
# exotic one.
#
# The rule for this path is conservative by construction: it handles only the
# cases it can handle EXACTLY (right sample rate, plain PCM or IEEE float), and
# returns None for anything else so ffmpeg takes over. It never resamples --
# ffmpeg's soxr does that properly, and a hand-rolled resampler here would
# alias, shifting the high-frequency content that both the gender model and the
# bandwidth check read. Being fast is not worth being subtly wrong.
# --------------------------------------------------------------------------

_WAV_PCM = 1
_WAV_FLOAT = 3
_WAV_EXTENSIBLE = 0xFFFE


def _try_fast_wav(raw: bytes, target_rate: int) -> np.ndarray | None:
    """Decode a plain PCM/float WAV without spawning ffmpeg. None => fall back."""
    if len(raw) < 44 or raw[:4] != b"RIFF" or raw[8:12] != b"WAVE":
        return None

    fmt_tag = channels = bits = 0
    sample_rate = 0
    data: bytes | None = None
    pos = 12

    while pos + 8 <= len(raw):
        chunk_id = raw[pos : pos + 4]
        (chunk_size,) = struct.unpack_from("<I", raw, pos + 4)
        body = pos + 8
        if chunk_size < 0 or body + chunk_size > len(raw):
            # Truncated or streaming WAV with a placeholder size. ffmpeg is
            # more forgiving about these than we should be.
            if chunk_id == b"data":
                chunk_size = len(raw) - body
            else:
                return None

        if chunk_id == b"fmt ":
            if chunk_size < 16:
                return None
            fmt_tag, channels, sample_rate, _byte_rate, _align, bits = struct.unpack_from(
                "<HHIIHH", raw, body
            )
            if fmt_tag == _WAV_EXTENSIBLE:
                # The real format lives in the GUID's first two bytes.
                if chunk_size < 40:
                    return None
                (fmt_tag,) = struct.unpack_from("<H", raw, body + 24)
        elif chunk_id == b"data":
            data = raw[body : body + chunk_size]

        pos = body + chunk_size + (chunk_size & 1)  # chunks are word-aligned

    if data is None or not channels or sample_rate != target_rate:
        return None  # wrong rate -> ffmpeg resamples it properly
    if fmt_tag == _WAV_PCM and bits in (16, 32):
        dtype = "<i2" if bits == 16 else "<i4"
        scale = 32768.0 if bits == 16 else 2147483648.0
    elif fmt_tag == _WAV_FLOAT and bits == 32:
        dtype, scale = "<f4", 1.0
    else:
        return None  # 8-bit, 24-bit, mu-law, A-law, ADPCM -> ffmpeg

    frame_bytes = channels * (bits // 8)
    usable = (len(data) // frame_bytes) * frame_bytes
    if usable == 0:
        return None

    samples = np.frombuffer(data, dtype=dtype, count=usable // (bits // 8))
    samples = samples.astype(np.float32) / scale
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    return np.ascontiguousarray(samples, dtype=np.float32)


def _ffmpeg_argv(sample_rate: int) -> list[str]:
    return [
        FFMPEG,
        "-hide_banner",
        "-loglevel", "error",
        "-nostdin",
        # Let ffmpeg probe the container; we deliberately do not trust the
        # client's Content-Type, which is wrong often enough to matter.
        "-i", "pipe:0",
        "-map", "0:a:0",          # first audio stream only; ignore video
        "-ac", "1",               # downmix to mono
        "-ar", str(sample_rate),
        "-f", "f32le",            # raw float32 little-endian on stdout
        "-acodec", "pcm_f32le",
        "pipe:1",
    ]


async def decode(raw: bytes, settings: Settings) -> DecodedAudio:
    """Decode arbitrary encoded audio to normalised PCM. Raises VoiceAttributeError."""
    if not raw:
        raise EmptyAudioError("Request contained no audio bytes.")
    if len(raw) > settings.max_upload_bytes:
        raise AudioTooLargeError(
            f"Audio exceeds the {settings.max_upload_bytes // (1024 * 1024)} MB limit.",
            detail=f"received {len(raw)} bytes",
        )

    fast = _try_fast_wav(raw, settings.target_sample_rate)
    if fast is not None:
        return _finalise(fast, raw, settings, decoder="wav-fast-path")

    proc = await asyncio.create_subprocess_exec(
        *_ffmpeg_argv(settings.target_sample_rate),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(input=raw), timeout=settings.ffmpeg_timeout_s
        )
    except asyncio.TimeoutError as exc:
        # Kill hard: a wedged ffmpeg holding a copy of caller audio in memory is
        # both a leak and a privacy problem.
        proc.kill()
        await proc.wait()
        raise DecodeTimeoutError("Decoding timed out.") from exc

    if proc.returncode != 0 or not stdout:
        # stderr may quote a container header; truncate hard and never log the
        # input itself.
        tail = stderr.decode("utf-8", "replace").strip().splitlines()
        detail = tail[-1][:200] if tail else None
        raise DecodeError(
            "Could not decode the supplied audio. Supported: wav, mp3, "
            "flac, ogg/opus, webm, m4a, and raw G.711.",
            detail=detail,
        )

    samples = np.frombuffer(stdout, dtype="<f4")
    return _finalise(samples, raw, settings, decoder="ffmpeg")


def _finalise(
    samples: np.ndarray, raw: bytes, settings: Settings, *, decoder: str
) -> DecodedAudio:
    """Length guard, windowing, and normalisation, shared by both decode paths."""
    # frombuffer may hand back a read-only view over a foreign buffer; copy so
    # we own the memory and can wipe it later.
    samples = np.array(samples, dtype=np.float32, copy=True)
    original_seconds = len(samples) / float(settings.target_sample_rate)

    if original_seconds < settings.min_audio_seconds:
        raise AudioTooShortError(
            f"Audio is {original_seconds:.2f}s; need at least "
            f"{settings.min_audio_seconds:.2f}s to attempt inference."
        )

    samples, windowed = _limit_duration(samples, settings)
    samples = _normalise(samples)

    log.debug(
        "decoded",
        decoder=decoder,
        source_bytes=len(raw),
        original_seconds=round(original_seconds, 3),
        analysed_seconds=round(len(samples) / settings.target_sample_rate, 3),
        windowed=windowed,
    )
    return DecodedAudio(
        samples=samples,
        sample_rate=settings.target_sample_rate,
        source_bytes=len(raw),
        original_seconds=original_seconds,
        windowed=windowed,
    )


def _limit_duration(samples: np.ndarray, settings: Settings) -> tuple[np.ndarray, bool]:
    """Cap analysis length by selecting the most energetic window.

    wav2vec2 self-attention is quadratic in sequence length, so an unbounded
    clip is an unbounded latency. Rather than truncating to the first N seconds
    -- which on a real call is usually ring tone and "hello?" -- we slide a
    window and keep the loudest one. That biases towards continuous speech,
    which is exactly what the model wants.
    """
    limit = int(settings.max_analysis_seconds * settings.target_sample_rate)
    if len(samples) <= limit:
        return samples, False

    hop = max(1, limit // 4)
    best_start, best_energy = 0, -1.0
    for start in range(0, len(samples) - limit + 1, hop):
        energy = float(np.sum(samples[start : start + limit] ** 2))
        if energy > best_energy:
            best_energy, best_start = energy, start
    return samples[best_start : best_start + limit], True


def _normalise(samples: np.ndarray) -> np.ndarray:
    """Peak-normalise quiet audio; leave already-loud audio alone.

    Deliberately NOT full AGC. Compressing dynamic range would erase the
    loudness cues the quality gate uses to distinguish "far from the handset in
    a noisy warehouse" from "clean close-mic", and would let genuinely bad audio
    masquerade as good. We only rescue clips so quiet that float precision and
    the model's input normalisation would otherwise suffer.
    """
    if samples.size == 0:
        return samples
    peak = float(np.max(np.abs(samples)))
    if 0.0 < peak < 0.1:
        samples = samples * (0.5 / peak)
    return np.clip(samples, -1.0, 1.0).astype(np.float32, copy=False)


def pcm16_to_float32(raw: bytes) -> np.ndarray:
    """Fast path for the streaming endpoint when the client already sends
    16-bit PCM at the target rate -- skips the ffmpeg hop entirely."""
    if len(raw) % 2:
        raw = raw[:-1]
    ints = np.frombuffer(raw, dtype="<i2")
    return (ints.astype(np.float32) / 32768.0).clip(-1.0, 1.0)
