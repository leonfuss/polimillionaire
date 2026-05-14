"""Unit tests for the offline parts of WhisperTranscriber.

The model loader is not exercised here -- pulling whisper-large-v3-turbo
into CI would be ~1.6 GB. We cover the WAV decode, the resample, and the
output normaliser, which are the pieces most likely to silently corrupt
input or output. The transcribe-end-to-end path is validated live against
the assignment server.
"""

from __future__ import annotations

import io
import wave

import numpy as np
import pytest

from polimillionaire.asr.whisper import (
    TARGET_SAMPLE_RATE,
    _decode_wav,
    _normalize_text,
    _resample_to_16k,
)


def _make_wav(samples: np.ndarray, sample_rate: int, channels: int = 1) -> bytes:
    """Pack float samples in [-1, 1] as a 16-bit PCM WAV blob."""
    pcm = np.clip(samples, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype(np.int16)
    if channels > 1:
        pcm = np.repeat(pcm[:, None], channels, axis=1)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


def _sine(duration_s: float, freq: float, sample_rate: int) -> np.ndarray:
    t = np.arange(int(sample_rate * duration_s), dtype=np.float32) / sample_rate
    return 0.5 * np.sin(2 * np.pi * freq * t).astype(np.float32)


def test_decode_wav_mono_24k_matches_input():
    sr = 24_000
    samples = _sine(1.0, freq=440.0, sample_rate=sr)
    blob = _make_wav(samples, sr)

    decoded, decoded_sr = _decode_wav(blob)

    assert decoded_sr == sr
    assert decoded.dtype == np.float32
    assert decoded.shape == samples.shape
    # int16 round-trip is lossy at ~1/32768; allow a small tolerance.
    np.testing.assert_allclose(decoded, samples, atol=1e-4)


def test_decode_wav_stereo_is_averaged_to_mono():
    sr = 16_000
    left = _sine(0.5, freq=440.0, sample_rate=sr)
    right = -left  # opposite phase -> mean is zero everywhere
    pcm = np.stack([left, right], axis=1)
    pcm_int = (pcm * 32767).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm_int.tobytes())

    decoded, decoded_sr = _decode_wav(buf.getvalue())

    assert decoded_sr == sr
    assert decoded.ndim == 1
    np.testing.assert_allclose(decoded, np.zeros_like(decoded), atol=1e-4)


def test_decode_wav_rejects_unsupported_bit_depth():
    # Synthesise a 32-bit WAV. _decode_wav is only spec'd for 16-bit; we want
    # a loud failure rather than silent mis-scaling.
    sr = 16_000
    samples = _sine(0.1, freq=440.0, sample_rate=sr)
    pcm32 = (samples * 2_147_483_647).astype(np.int32)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(4)
        wf.setframerate(sr)
        wf.writeframes(pcm32.tobytes())

    with pytest.raises(ValueError, match="sample width"):
        _decode_wav(buf.getvalue())


def test_resample_no_op_at_16k():
    audio = _sine(0.5, freq=440.0, sample_rate=TARGET_SAMPLE_RATE)
    out = _resample_to_16k(audio, TARGET_SAMPLE_RATE)
    # Same object isn't required, but the values must be identical.
    np.testing.assert_array_equal(out, audio)


def test_resample_24k_to_16k_changes_length_by_ratio():
    sr = 24_000
    audio = _sine(1.0, freq=440.0, sample_rate=sr)
    out = _resample_to_16k(audio, sr)
    # 24k -> 16k is a 2/3 ratio; length should drop accordingly.
    expected = int(audio.shape[0] * TARGET_SAMPLE_RATE / sr)
    assert abs(out.shape[0] - expected) <= 1  # off-by-one is fine for polyphase
    assert out.dtype == np.float32


def test_normalize_text_collapses_whitespace_and_strips():
    assert _normalize_text("  hello   world\n") == "hello world"
    assert _normalize_text("hello") == "hello"
    assert _normalize_text("") == ""
    assert _normalize_text("\tfoo\n  bar\r\nbaz   ") == "foo bar baz"
