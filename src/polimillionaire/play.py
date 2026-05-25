"""Live game runners.

`manual_play_loop` is the human-in-the-loop tool that bootstrapped the corpus.
`auto_play_loop` is its strategy-driven sibling: hand it a `Strategy` (e.g.
`ZeroShotStrategy(load_llm("qwen3-8b"))`) and it plays games end-to-end,
logging the same `predictions` schema so replay/eval works uniformly.

`speech_auto_play_loop` is the speech-mode variant: starts a `mode="speech"`
session, fetches Q+4 option WAVs via the audio endpoints, transcribes them
with the provided `WhisperTranscriber`, then hands a synthetic Question
(with ASR-derived text) to the strategy. The server's 30s clock starts on
the fourth option fetch, so the helper deliberately defers that fetch to
the last possible moment.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from polimillionaire._vendor.millionaire_client import MillionaireClient
from polimillionaire._vendor.millionaire_client.models import Option, Question
from polimillionaire.recording import PredictionRecord, QuestionLog
from polimillionaire.strategies.base import Context, Strategy

if TYPE_CHECKING:
    from polimillionaire._vendor.millionaire_client.game import GameSession
    from polimillionaire.asr.whisper import WhisperTranscriber

# Project root: <repo>/src/polimillionaire/play.py -> three parents up.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _resolve_db_path(db_path: str | None) -> str:
    """Pick a DB path and anchor it to the project root if relative.

    Without this, calling auto_play_loop from a subdirectory (e.g.
    `scripts/`) wrote `data/questions.sqlite` next to the cwd, splitting
    the corpus across one log per launch directory. Now relative paths --
    including the default -- always land at <project_root>/data/...
    """
    raw = db_path or os.environ.get("POLIMILLIONAIRE_DB_PATH") or "data/questions.sqlite"
    p = Path(raw)
    return str(p if p.is_absolute() else _PROJECT_ROOT / p)


@dataclass(frozen=True)
class _GameAnswer:
    """What the answer provider returns for each question."""

    option_id: int
    confidence: float | None
    rationale: str | None
    model_name: str
    strategy_name: str
    prompt_version: str
    latency_ms: int


# (question, level, competition_id) -> _GameAnswer | None
# None signals "quit cleanly" (used by manual_play_loop's 'q' option).
AnswerProvider = Callable[[Question, int, int], "_GameAnswer | None"]

# (game, level) -> Question | None
# Speech mode swaps in a builder that fetches the audio endpoints and
# transcribes them; text mode just reads game.current_question. None means
# the server has no question (game ended).
QuestionBuilder = Callable[["GameSession", int], "Question | None"]


def _text_question_builder(game: GameSession, _level: int) -> Question | None:
    """Default builder: read the server-provided question text directly."""
    return game.current_question


def _play_one_game(
    client: MillionaireClient,
    competition_id: int,
    log: QuestionLog,
    answer_provider: AnswerProvider,
    *,
    time_label: str = "",
    mode: str = "text",
    question_builder: QuestionBuilder = _text_question_builder,
) -> dict[str, int] | None:
    """Play one game, calling `answer_provider` for each question.

    `time_label` is appended to the level header (e.g. "left on the wire").
    `mode` is "text" or "speech"; passed straight to `client.game.start` and
    recorded on each PredictionRecord. Speech mode requires that callers
    supplied a provider that knows how to fetch+transcribe audio (the
    `Question` from `current_question` has `text=None` in speech mode).
    Returns a counts dict {"correct", "wrong", "timeouts"}, or None if the
    provider requested a clean quit.
    """
    game = client.game.start(competition_id=competition_id, mode=mode)
    print(f"=== session {game.session_id} ({mode}) ===")

    counts: dict[str, int] = {"correct": 0, "wrong": 0, "timeouts": 0}
    builder = question_builder

    while game.in_progress:
        current_level = game.current_question.level if game.current_question else game.current_level
        q = builder(game, current_level)
        if not q:
            break

        time_left = game.time_remaining or 0
        level = q.level or game.current_level
        time_str = f"{time_left:.0f}s {time_label}".rstrip() if time_label else f"{time_left:.0f}s"
        print(f"\n--- level {level} ({time_str}) ---")
        print(f"Q: {q.text}")
        for opt in q.options:
            print(f"  [{opt.id}] {opt.text}")

        answer = answer_provider(q, level, competition_id)
        if answer is None:
            return None

        if answer.strategy_name != "manual_human":
            if answer.confidence is not None:
                print(
                    f"-> chose [{answer.option_id}] "
                    f"(conf={answer.confidence:.2f}, {answer.latency_ms} ms)"
                )
            else:
                print(f"-> chose [{answer.option_id}] ({answer.latency_ms} ms)")
            if answer.rationale:
                print(f"   reason: {answer.rationale}")

        result = game.answer(answer.option_id)

        log.record(
            PredictionRecord(
                account_username=client.user.username,
                session_id=game.session_id,
                competition_id=competition_id,
                level=level,
                question_id=q.id,
                question_text=q.text,
                options=[{"id": o.id, "text": o.text} for o in q.options],
                predicted_option_id=answer.option_id,
                correct_option_id_if_known=answer.option_id if result.correct else None,
                strategy_name=answer.strategy_name,
                model_name=answer.model_name,
                prompt_version=answer.prompt_version,
                confidence=answer.confidence,
                rationale=answer.rationale,
                latency_ms=answer.latency_ms,
                mode=mode,
            )
        )

        if result.correct:
            counts["correct"] += 1
            print(f"correct — earned ${result.earned_amount:,.0f}")
        elif result.timed_out:
            counts["timeouts"] += 1
            print(f"timed out — earned ${result.earned_amount:,.0f}")
            break
        else:
            counts["wrong"] += 1
            print(f"wrong — earned ${result.earned_amount:,.0f}")

        if result.game_over:
            print(f"\ngame over: level {game.current_level}, ${result.earned_amount:,.0f}")
            break

    return counts


def manual_play_loop(
    client: MillionaireClient,
    competition_id: int,
    db_path: str | None = None,
    max_games: int = 1,
) -> None:
    """Play `max_games` consecutive games, logging every question to the DB.

    At each prompt, type the option id (or `q` to abort cleanly).
    Picks up the DB path from `POLIMILLIONAIRE_DB_PATH` if `db_path` is None.
    Relative paths anchor to the project root (not cwd).
    """
    log = QuestionLog(_resolve_db_path(db_path))

    def _manual_provider(q: Question, level: int, _comp_id: int) -> _GameAnswer | None:
        while True:
            try:
                raw = input("answer id (or 'q' to quit): ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\naborting.")
                return None
            if raw.lower() == "q":
                print("quitting (current question will time out on the server)")
                return None
            try:
                chosen = int(raw)
            except ValueError:
                print("not a number, try again")
                continue
            return _GameAnswer(
                option_id=chosen,
                confidence=None,
                rationale=None,
                model_name=client.user.username,
                strategy_name="manual_human",
                prompt_version="n/a",
                latency_ms=0,
            )

    for game_num in range(max_games):
        print(f"=== game {game_num + 1}/{max_games} ===")
        result = _play_one_game(client, competition_id, log, _manual_provider)
        if result is None:
            return
        print()


def auto_play_loop(
    client: MillionaireClient,
    competition_id: int,
    strategy: Strategy,
    db_path: str | None = None,
    max_games: int = 1,
) -> dict[str, int]:
    """Play `max_games` consecutive games using `strategy`, logging every question.

    Returns a small summary dict (`{"correct": int, "wrong": int, "timeouts": int}`)
    so callers can sanity-check accuracy without opening the DB.
    Picks up the DB path from `POLIMILLIONAIRE_DB_PATH` if `db_path` is None.
    Relative paths anchor to the project root (not cwd).
    """
    log = QuestionLog(_resolve_db_path(db_path))
    summary: dict[str, int] = {"correct": 0, "wrong": 0, "timeouts": 0}

    def _auto_provider(q: Question, level: int, comp_id: int) -> _GameAnswer:
        ctx = Context(competition_id=comp_id, level=level)
        decision = strategy(q, ctx)
        return _GameAnswer(
            option_id=decision.option_id,
            confidence=decision.confidence,
            rationale=decision.rationale,
            model_name=decision.model_name,
            strategy_name=decision.strategy_name,
            prompt_version=decision.prompt_version,
            latency_ms=decision.latency_ms,
        )

    for game_num in range(max_games):
        print(f"=== game {game_num + 1}/{max_games} ===")
        counts = _play_one_game(
            client, competition_id, log, _auto_provider, time_label="left on the wire"
        )
        if counts is not None:
            for k in summary:
                summary[k] += counts[k]
        print()

    print(f"summary: {summary}")
    return summary


# The server's TTS reads "Option A.", "Option B,", etc. at the start of
# each option clip, and Whisper transcribes that literally. Leaving the
# prefix in the option text confuses the strategy ("Option A. Two." vs
# the cached "Two" -- the cross-mode DB fuzzy matcher gets ~0.4 ratio
# instead of an exact hit) and bloats the prompt. The question clip has
# no such prefix, so we strip only on option transcripts.
# Match the letter, then any run of whitespace / punctuation up to the
# first content character. Covers the four forms Whisper emits in practice:
# "Option A. Two.", "Option B, three.", "Option C. One...", "Option D - four".
_OPTION_PREFIX_RE = re.compile(
    r"^\s*option\s+[a-d](?:[\s.,:\-]+|$)",
    re.IGNORECASE,
)


def _strip_option_prefix(text: str) -> str:
    """Drop the spoken option-letter announcement at the start of an option
    transcript. No-op when the prefix isn't present."""
    return _OPTION_PREFIX_RE.sub("", text, count=1).strip()


# Whisper biases its decoder toward whatever vocabulary appears in the
# initial prompt -- the model treats it as the previous transcribed
# segment. We feed an exemplar-style math snippet so single letters
# ("x", "n"), digits, and operator words ("squared", "divided by") get
# weighted up. Only applied for the math competition; other categories
# get a clean decode with no bias.
#
# Kept natural-sounding rather than instruction-shaped: Whisper was
# trained on transcribed speech and prefers prompts that look like real
# audio. ~25 tokens -- well under the 224-token cap.
_MATH_ASR_PROMPT = (
    "Math problem: solve for x squared plus three times x minus two, where x equals five. "
    "Numbers, variables x y n k, plus minus times divided by, squared cubed, square root, equals."
)

# Maps competition_id to the initial_prompt forwarded to Whisper. Add an
# entry here to bias other categories' vocabulary.
_ASR_PROMPT_BY_CID: dict[int, str] = {3: _MATH_ASR_PROMPT}


def _speech_question_builder(
    transcriber: WhisperTranscriber,
    *,
    competition_id: int,
    verbose: bool = True,
) -> QuestionBuilder:
    """Build a question_builder that fetches WAV audio and transcribes it.

    Fetch order matters: question, then options 0..2, then option 3 *last*.
    The server starts the 30 s clock on the option-3 fetch, so transcribing
    everything before then is free time. Whisper-large-v3-turbo on a 5 s
    clip runs in ~0.3-0.8 s on a warm GPU, so all four transcriptions
    typically complete well inside the pre-clock window.
    """
    initial_prompt = _ASR_PROMPT_BY_CID.get(competition_id)
    if verbose and initial_prompt is not None:
        print(f"  [asr] using competition-{competition_id} prompt bias for math vocabulary")

    def _build(game: GameSession, level: int) -> Question | None:
        if game.current_question is None:
            return None
        # Server-side question id is real even when text is None.
        qid = game.current_question.id

        if verbose:
            print(f"  level {level}: fetching question audio...", end="", flush=True)
        wav_q = game.fetch_audio_question()
        text_q = transcriber.transcribe(wav_q, initial_prompt=initial_prompt)
        if verbose:
            print(f" {len(wav_q)} B -> {text_q!r}")

        option_texts: list[str] = []
        for i in range(4):
            letter = chr(ord("A") + i)
            if verbose:
                print(f"  level {level}: fetching option {letter} audio...", end="", flush=True)
            wav_o = game.fetch_audio_option_next()
            text_o = _strip_option_prefix(
                transcriber.transcribe(wav_o, initial_prompt=initial_prompt)
            )
            if verbose:
                print(f" {len(wav_o)} B -> {text_o!r}")
            option_texts.append(text_o)

        return Question(
            id=qid,
            text=text_q,
            options=[Option(id=i, text=t) for i, t in enumerate(option_texts)],
            level=level,
        )

    return _build


def speech_auto_play_loop(
    client: MillionaireClient,
    competition_id: int,
    strategy: Strategy,
    transcriber: WhisperTranscriber,
    db_path: str | None = None,
    max_games: int = 1,
    *,
    verbose: bool = True,
) -> dict[str, int]:
    """Play `max_games` speech-mode games using `strategy` + `transcriber`.

    Mirrors `auto_play_loop`, but starts each session with `mode="speech"`
    and replaces the server-provided question text (which is None in speech
    mode) with the Whisper transcript of the question + option audios.

    Logged rows carry `mode='speech'`. Use `make_strategy(..., mode="speech",
    use_text_mode_retrieval=True)` to also consult prior text-mode rows on
    a cold start.
    """
    log = QuestionLog(_resolve_db_path(db_path))
    summary: dict[str, int] = {"correct": 0, "wrong": 0, "timeouts": 0}

    def _provider(q: Question, level: int, comp_id: int) -> _GameAnswer:
        ctx = Context(competition_id=comp_id, level=level)
        decision = strategy(q, ctx)
        return _GameAnswer(
            option_id=decision.option_id,
            confidence=decision.confidence,
            rationale=decision.rationale,
            model_name=decision.model_name,
            strategy_name=decision.strategy_name,
            prompt_version=decision.prompt_version,
            latency_ms=decision.latency_ms,
        )

    builder = _speech_question_builder(transcriber, competition_id=competition_id, verbose=verbose)
    for game_num in range(max_games):
        print(f"=== game {game_num + 1}/{max_games} (speech) ===")
        counts = _play_one_game(
            client,
            competition_id,
            log,
            _provider,
            time_label="left on the wire",
            mode="speech",
            question_builder=builder,
        )
        if counts is not None:
            for k in summary:
                summary[k] += counts[k]
        print()

    print(f"summary: {summary}")
    return summary
