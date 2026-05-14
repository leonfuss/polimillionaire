"""Tests for the speech-mode orchestration in play.py.

Focuses on `_speech_question_builder`: confirms the audio endpoints are
called in the right order (question, then options 0..3 sequentially) and
that the synthesised Question uses the server-provided question id with
ASR transcripts in the text fields.
"""

from __future__ import annotations

from polimillionaire._vendor.millionaire_client.models import Option, Question
from polimillionaire.play import _speech_question_builder


class _MockTranscriber:
    """Maps WAV blobs to known transcripts so we can pin expected output."""

    def __init__(self, mapping: dict[bytes, str]) -> None:
        self._mapping = mapping
        self.calls: list[bytes] = []

    def transcribe(self, wav_bytes: bytes) -> str:
        self.calls.append(wav_bytes)
        return self._mapping[wav_bytes]


class _MockGameSession:
    """Tracks fetch order and returns canned bytes; emulates the speech-mode
    contract: option fetches must be sequential, current_question has id but
    text=None."""

    def __init__(self, qid: int, level: int, q_wav: bytes, option_wavs: list[bytes]) -> None:
        # Server-side: only id+level are populated in speech mode.
        self.current_question = Question(
            id=qid,
            text=None,  # speech mode wipes text
            options=[Option(id=i, text=None) for i in range(4)],
            level=level,
        )
        self.current_level = level
        self._q_wav = q_wav
        self._option_wavs = option_wavs
        self._next_option = 0
        self.fetch_log: list[str] = []

    def fetch_audio_question(self) -> bytes:
        self.fetch_log.append("Q")
        return self._q_wav

    def fetch_audio_option_next(self) -> bytes:
        idx = self._next_option
        if idx >= 4:
            raise RuntimeError("server would reject: all options already delivered")
        self._next_option += 1
        self.fetch_log.append(f"opt{idx}")
        return self._option_wavs[idx]


def test_speech_builder_fetches_in_order_and_synthesises_question():
    q_wav = b"<wav-q>"
    option_wavs = [b"<wav-a>", b"<wav-b>", b"<wav-c>", b"<wav-d>"]
    transcripts = {
        q_wav: "What is the capital of France?",
        option_wavs[0]: "London",
        option_wavs[1]: "Paris",
        option_wavs[2]: "Rome",
        option_wavs[3]: "Berlin",
    }
    transcriber = _MockTranscriber(transcripts)
    game = _MockGameSession(qid=42, level=3, q_wav=q_wav, option_wavs=option_wavs)

    build = _speech_question_builder(transcriber, verbose=False)
    q = build(game, level=3)

    assert q is not None
    # Server's question id survives; level survives.
    assert q.id == 42
    assert q.level == 3
    # Text comes from the transcriber.
    assert q.text == "What is the capital of France?"
    # Options 0..3 keep their ids with ASR text.
    assert [o.id for o in q.options] == [0, 1, 2, 3]
    assert [o.text for o in q.options] == ["London", "Paris", "Rome", "Berlin"]
    # Question audio fetched first, then options sequentially -- the option-3
    # fetch is the one that starts the 30s clock server-side, so its position
    # in the order is load-bearing.
    assert game.fetch_log == ["Q", "opt0", "opt1", "opt2", "opt3"]
    # Five transcriptions total; same blobs we fetched, in the same order.
    assert transcriber.calls == [q_wav, *option_wavs]


def test_speech_builder_returns_none_when_no_question_available():
    transcriber = _MockTranscriber({})
    game = _MockGameSession(qid=0, level=0, q_wav=b"", option_wavs=[b"", b"", b"", b""])
    game.current_question = None  # game has ended

    build = _speech_question_builder(transcriber, verbose=False)
    assert build(game, level=1) is None
    assert game.fetch_log == []  # no audio fetched
    assert transcriber.calls == []
