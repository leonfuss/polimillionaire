"""Live game runners.

`manual_play_loop` is the human-in-the-loop tool that bootstrapped the corpus.
`auto_play_loop` is its strategy-driven sibling: hand it a `Strategy` (e.g.
`ZeroShotStrategy(load_llm("qwen3-8b"))`) and it plays games end-to-end,
logging the same `predictions` schema so replay/eval works uniformly.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from polimillionaire._vendor.millionaire_client import MillionaireClient
from polimillionaire._vendor.millionaire_client.models import Question
from polimillionaire.recording import PredictionRecord, QuestionLog
from polimillionaire.strategies.base import Context, Strategy

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


def _play_one_game(
    client: MillionaireClient,
    competition_id: int,
    log: QuestionLog,
    answer_provider: AnswerProvider,
    *,
    time_label: str = "",
) -> dict[str, int] | None:
    """Play one game, calling `answer_provider` for each question.

    `time_label` is appended to the level header (e.g. "left on the wire").
    Returns a counts dict {"correct", "wrong", "timeouts"}, or None if the
    provider requested a clean quit.
    """
    game = client.game.start(competition_id=competition_id)
    print(f"=== session {game.session_id} ===")

    counts: dict[str, int] = {"correct": 0, "wrong": 0, "timeouts": 0}

    while game.in_progress:
        q = game.current_question
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
