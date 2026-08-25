"""Audio ingestion: codec coverage, the WAV fast path, and input guards."""

from __future__ import annotations

import shutil
import subprocess

import numpy as np
import pytest

from app.audio.decode import _try_fast_wav, decode, pcm16_to_float32
from app.errors import (
    AudioTooLargeError,
    AudioTooShortError,
    DecodeError,
    EmptyAudioError,
)
from scripts.make_sample_audio import wav_bytes

HAS_FFMPEG = shutil.which("ffmpeg") is not None
needs_ffmpeg = pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not installed")


def _transcode(raw: bytes, *args: str) -> bytes:
    return subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", "pipe:0", *args, "pipe:1"],
        input=raw, capture_output=True, check=True,
    ).stdout


# --------------------------------------------------------------- happy paths
async def test_decodes_wav(settings, speech_wav, speech):
    audio = await decode(speech_wav, settings)
    assert audio.sample_rate == settings.target_sample_rate
    assert audio.samples.dtype == np.float32
    assert audio.duration_s == pytest.approx(len(speech) / 16_000, abs=0.05)


@needs_ffmpeg
@pytest.mark.parametrize(
    "args, label",
    [
        (["-c:a", "libmp3lame", "-b:a", "64k", "-f", "mp3"], "mp3"),
        (["-c:a", "libopus", "-b:a", "24k", "-f", "ogg"], "opus"),
        (["-c:a", "flac", "-f", "flac"], "flac"),
        (["-ar", "8000", "-c:a", "pcm_mulaw", "-f", "wav"], "g711-ulaw"),
        (["-ar", "8000", "-c:a", "pcm_alaw", "-f", "wav"], "g711-alaw"),
        (["-ar", "44100", "-f", "wav"], "wav-44k1"),
        (["-ac", "2", "-f", "wav"], "wav-stereo"),
    ],
)
async def test_decodes_common_call_codecs(settings, speech_wav, args, label):
    """Logistics audio arrives as whatever the telephony vendor emits."""
    audio = await decode(_transcode(speech_wav, *args), settings)
    assert audio.sample_rate == 16_000
    assert audio.duration_s > 1.0, f"{label} decoded to nothing"


async def test_content_type_is_not_trusted(settings, speech_wav):
    """Gateways mislabel audio; ffmpeg probes the real container.

    decode() takes bytes and no mime type at all, which is the point -- there is
    no header for a caller to get wrong.
    """
    mp3 = _transcode(speech_wav, "-c:a", "libmp3lame", "-f", "mp3") if HAS_FFMPEG else None
    if mp3 is None:
        pytest.skip("ffmpeg not installed")
    assert (await decode(mp3, settings)).duration_s > 1.0


# ------------------------------------------------------------- the fast path
def test_fast_path_takes_plain_16k_wav(speech_wav):
    assert _try_fast_wav(speech_wav, 16_000) is not None


def test_fast_path_declines_what_it_cannot_do_exactly(speech_wav):
    """Conservative by construction: wrong rate -> ffmpeg, which resamples
    properly instead of us aliasing."""
    assert _try_fast_wav(speech_wav, 8_000) is None            # rate mismatch
    assert _try_fast_wav(b"", 16_000) is None                  # empty
    assert _try_fast_wav(b"RIFF" + b"\x00" * 60, 16_000) is None   # not WAVE
    assert _try_fast_wav(b"OggS" + b"\x00" * 60, 16_000) is None   # not WAV


@needs_ffmpeg
def test_fast_path_declines_compressed_wav_payloads(speech_wav):
    """mu-law is *inside* a RIFF container -- the header check alone is not
    enough, the format tag has to be inspected too."""
    ulaw = _transcode(speech_wav, "-ar", "16000", "-c:a", "pcm_mulaw", "-f", "wav")
    assert _try_fast_wav(ulaw, 16_000) is None


@needs_ffmpeg
def test_fast_path_is_bit_identical_to_ffmpeg(speech_wav):
    """The optimisation is only worth having if it changes nothing.

    Anything else would mean latency measured on one path and accuracy on
    another -- so this asserts exact equality, not approximate.
    """
    from app.audio.decode import _ffmpeg_argv

    fast = _try_fast_wav(speech_wav, 16_000)
    reference = np.frombuffer(
        subprocess.run(_ffmpeg_argv(16_000), input=speech_wav,
                       capture_output=True, check=True).stdout,
        dtype="<f4",
    )
    assert fast is not None and fast.size == reference.size
    np.testing.assert_array_equal(fast, reference)


def test_fast_path_downmixes_stereo():
    left = np.full(16_000, 0.5, dtype=np.float32)
    right = np.full(16_000, -0.1, dtype=np.float32)
    interleaved = np.empty(32_000, dtype=np.float32)
    interleaved[0::2], interleaved[1::2] = left, right

    ints = np.clip(interleaved * 32767, -32768, 32767).astype("<i2")
    import struct
    data = ints.tobytes()
    header = (b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVEfmt "
              + struct.pack("<IHHIIHH", 16, 1, 2, 16_000, 16_000 * 4, 4, 16)
              + b"data" + struct.pack("<I", len(data)))

    mono = _try_fast_wav(header + data, 16_000)
    assert mono is not None and mono.size == 16_000
    assert mono[0] == pytest.approx(0.2, abs=0.01)


# ------------------------------------------------------------------- guards
async def test_empty_body_rejected(settings):
    with pytest.raises(EmptyAudioError):
        await decode(b"", settings)


async def test_oversized_upload_rejected(settings, speech_wav):
    small = settings.model_copy(update={"max_upload_bytes": 1024})
    with pytest.raises(AudioTooLargeError):
        await decode(speech_wav, small)


async def test_too_short_rejected(settings):
    with pytest.raises(AudioTooShortError):
        await decode(wav_bytes(np.zeros(1_600, dtype=np.float32)), settings)


async def test_garbage_rejected(settings):
    with pytest.raises(DecodeError):
        await decode(b"this is definitely not audio" * 100, settings)


async def test_truncated_file_rejected(settings, speech_wav):
    """A half-uploaded mp3 must fail cleanly, not hang or half-decode."""
    with pytest.raises((DecodeError, AudioTooShortError)):
        await decode(speech_wav[:20], settings)


# ------------------------------------------------------------- long uploads
async def test_long_audio_is_windowed_not_rejected(settings, speech):
    """A 60 s voicemail should still get an answer, at bounded cost."""
    long_audio = np.tile(speech, 12)[: 60 * 16_000]
    audio = await decode(wav_bytes(long_audio), settings)
    assert audio.windowed is True
    assert audio.original_seconds == pytest.approx(60.0, abs=0.5)
    assert audio.duration_s <= settings.max_analysis_seconds + 0.01


async def test_windowing_picks_the_loudest_segment(settings, speech):
    """Truncating to the first N seconds would usually capture ring tone and
    'hello?'. We keep the most energetic window instead."""
    quiet = np.zeros(20 * 16_000, dtype=np.float32)
    loud = np.concatenate([quiet, speech, quiet])
    audio = await decode(wav_bytes(loud), settings)
    assert float(np.sqrt(np.mean(audio.samples**2))) > 0.01


# --------------------------------------------------------------- misc units
def test_pcm16_conversion_round_trips():
    original = np.array([0.0, 0.5, -0.5, 0.999], dtype=np.float32)
    ints = np.clip(original * 32767, -32768, 32767).astype("<i2")
    np.testing.assert_allclose(pcm16_to_float32(ints.tobytes()), original, atol=1e-3)


def test_pcm16_tolerates_an_odd_trailing_byte():
    """Frame boundaries do not align with WebSocket message boundaries."""
    assert pcm16_to_float32(b"\x00\x01\x02").size == 1


async def test_wipe_zeroes_the_buffer(settings, speech_wav):
    audio = await decode(speech_wav, settings)
    assert float(np.max(np.abs(audio.samples))) > 0.0
    audio.wipe()
    assert float(np.max(np.abs(audio.samples))) == 0.0
