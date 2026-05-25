"""Whisper-large-v3-turbo wrapper for the speech-mode game API.

The assignment API serves WAV blobs (probed: 24 kHz mono 16-bit PCM, ~5 s
per clip). Whisper wants 16 kHz mono float32. This wrapper handles both
the resample and the model lifecycle.

Defaults:
- model: ``openai/whisper-large-v3-turbo`` -- ~1.6 GB fp16 on CUDA, same
  multilingual encoder as ``large-v3`` with a 4-layer distilled decoder
  (5-8x faster, negligible WER cost on short clips).
- device: ``cuda:1`` (overridable via ``POLIMILLIONAIRE_ASR_DEVICE``).
  Hard-defaulting to cuda:1 keeps us off the same device as the main LLM,
  which is typically on cuda:0 in our Colab setup.
- language: ``"en"`` -- pinned, skips Whisper's slow language-id pass.
  Flip to ``None`` if non-English audio ever shows up.
- decoding: greedy (``num_beams=1``). Beam search is ~3x slower for
  questionable gain on short, clean clips like these.

Loading is lazy: constructing a ``WhisperTranscriber`` doesn't import
torch or pull weights. The first ``transcribe()`` call triggers both.
"""

from __future__ import annotations

import io
import os
import wave
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

DEFAULT_MODEL = "openai/whisper-large-v3-turbo"
DEFAULT_DEVICE = "cuda:1"
DEFAULT_LANGUAGE = "en"
TARGET_SAMPLE_RATE = 16_000


def _normalize_text(text: str) -> str:
    """Whitespace-normalise so the same transcript becomes the same DB key
    even when the decoder emits stray leading/trailing spaces."""
    return " ".join(text.split()).strip()


def _decode_wav(wav_bytes: bytes) -> tuple[np.ndarray, int]:
    """Decode a 16-bit PCM WAV blob to float32 mono in [-1, 1].

    Returns ``(audio, sample_rate)``. Stereo is averaged to mono. Other
    bit-depths (24/32-bit) raise -- the assignment API serves 16-bit, and
    we'd rather fail loudly than silently mis-scale.
    """
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        nchannels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())

    if sampwidth != 2:
        raise ValueError(f"unsupported sample width: {sampwidth} bytes (expected 2 / 16-bit PCM)")

    # int16 -> float32 in [-1, 1]
    pcm = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    if nchannels > 1:
        pcm = pcm.reshape(-1, nchannels).mean(axis=1)
    return pcm, framerate


def _resample_to_16k(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Polyphase resample to 16 kHz. No-op when already at 16 kHz."""
    if sample_rate == TARGET_SAMPLE_RATE:
        return audio
    # scipy.signal.resample_poly is FFT-free and fast for integer ratios
    # like 24k -> 16k (up=2, down=3). Imported lazily so the asr package
    # doesn't pull scipy at module import time.
    from math import gcd

    from scipy.signal import resample_poly

    g = gcd(sample_rate, TARGET_SAMPLE_RATE)
    up = TARGET_SAMPLE_RATE // g
    down = sample_rate // g
    return resample_poly(audio, up=up, down=down).astype(np.float32)


class WhisperTranscriber:
    """Lazy wrapper around a Whisper model.

    Construction is free. The first ``transcribe()`` call loads the
    processor and model onto the chosen device.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        *,
        device: str | None = None,
        language: str | None = DEFAULT_LANGUAGE,
    ) -> None:
        self.name = model_name
        self.device = device or os.environ.get("POLIMILLIONAIRE_ASR_DEVICE") or DEFAULT_DEVICE
        self.language = language
        self._processor: WhisperProcessor | None = None
        self._model: WhisperForConditionalGeneration | None = None
        # prompt_ids are tokenized once per unique prompt string. Speech mode
        # calls transcribe() 5x per question; caching avoids re-tokenizing
        # the same math prompt 4 extra times.
        self._prompt_ids_cache: dict[str, torch.Tensor] = {}

    def preload(self) -> None:
        """Eagerly load the processor + model so the first transcribe() doesn't
        pay the HF download + load cost. Idempotent."""
        self._ensure_loaded()

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import (
            WhisperForConditionalGeneration,
            WhisperProcessor,
        )

        dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        self._processor = WhisperProcessor.from_pretrained(self.name)
        model = WhisperForConditionalGeneration.from_pretrained(self.name, torch_dtype=dtype)
        # whisper-large-v3-turbo ships generation_config.max_length=448, which
        # collides with our max_new_tokens=256 and prints a verbose
        # "both set, max_new_tokens will take precedence" warning on every
        # transcribe() call (5x per question in speech mode). Clear it so
        # only our bound applies.
        model.generation_config.max_length = None
        model.to(self.device)
        model.eval()
        self._model = model

    def transcribe(self, wav_bytes: bytes, *, initial_prompt: str | None = None) -> str:
        """Transcribe a WAV blob to whitespace-normalised text.

        `initial_prompt` biases the decoder's language model with a
        short in-domain string (Whisper treats it as if it were the
        previous transcribed segment). Useful for vocabulary the model
        otherwise mishears -- e.g. math symbols/variables. Keep it
        natural-sounding rather than instruction-shaped; the model was
        trained on transcribed speech, not prose commands.
        """
        self._ensure_loaded()
        assert self._model is not None and self._processor is not None

        audio, sample_rate = _decode_wav(wav_bytes)
        audio = _resample_to_16k(audio, sample_rate)

        import torch

        inputs = self._processor(
            audio,
            sampling_rate=TARGET_SAMPLE_RATE,
            return_tensors="pt",
        )
        features = inputs.input_features.to(self.device, dtype=self._model.dtype)

        # `language` and `task` are forwarded to the generation config; passing
        # them as kwargs avoids touching the model's forced_decoder_ids state.
        gen_kwargs: dict = {"num_beams": 1, "max_new_tokens": 256}
        if self.language:
            gen_kwargs["language"] = self.language
            gen_kwargs["task"] = "transcribe"
        if initial_prompt:
            gen_kwargs["prompt_ids"] = self._encode_prompt(initial_prompt)

        with torch.inference_mode():
            ids = self._model.generate(features, **gen_kwargs)

        text = self._processor.batch_decode(ids, skip_special_tokens=True)[0]
        return _normalize_text(text)

    def _encode_prompt(self, prompt: str) -> torch.Tensor:
        """Return cached prompt_ids tensor on the model's device."""
        assert self._processor is not None
        cached = self._prompt_ids_cache.get(prompt)
        if cached is not None:
            return cached
        ids = self._processor.get_prompt_ids(prompt, return_tensors="pt").to(self.device)
        self._prompt_ids_cache[prompt] = ids
        return ids


__all__ = ["WhisperTranscriber"]
