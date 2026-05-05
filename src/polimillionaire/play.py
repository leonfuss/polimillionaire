"""Live game runners.

`manual_play_loop` is the human-in-the-loop tool that bootstrapped the corpus.
`auto_play_loop` is its strategy-driven sibling: hand it a `Strategy` (e.g.
`ZeroShotStrategy(load_llm("qwen3-8b"))`) and it plays games end-to-end,
logging the same `predictions` schema so replay/eval works uniformly.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from polimillionaire._vendor.millionaire_client import MillionaireClient
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

    for game_num in range(max_games):
        game = client.game.start(competition_id=competition_id)
        print(f"=== game {game_num + 1}/{max_games}, session {game.session_id} ===")

        while game.in_progress:
            q = game.current_question
            if not q:
                break

            time_left = game.time_remaining or 0
            print(f"\n--- level {game.current_level} ({time_left:.0f}s) ---")
            print(f"Q: {q.text}")
            for opt in q.options:
                print(f"  [{opt.id}] {opt.text}")

            try:
                raw = input("answer id (or 'q' to quit): ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\naborting.")
                return
            if raw.lower() == "q":
                print("quitting (current question will time out on the server)")
                return
            try:
                chosen = int(raw)
            except ValueError:
                print("not a number, try again")
                continue

            start = time.perf_counter()
            result = game.answer(chosen)
            elapsed_ms = int((time.perf_counter() - start) * 1000)

            log.record(
                PredictionRecord(
                    account_username=client.user.username,
                    session_id=game.session_id,
                    competition_id=competition_id,
                    level=q.level or game.current_level,
                    question_id=q.id,
                    question_text=q.text,
                    options=[{"id": o.id, "text": o.text} for o in q.options],
                    predicted_option_id=chosen,
                    correct_option_id_if_known=chosen if result.correct else None,
                    strategy_name="manual_human",
                    model_name=client.user.username,
                    prompt_version="n/a",
                    confidence=None,
                    rationale=None,
                    latency_ms=elapsed_ms,
                )
            )

            if result.correct:
                print(f"correct — earned ${result.earned_amount:,.0f}")
            elif result.timed_out:
                print(f"timed out — earned ${result.earned_amount:,.0f}")
                break
            else:
                print(f"wrong — earned ${result.earned_amount:,.0f}")

            if result.game_over:
                print(f"\ngame over: level {game.current_level}, ${result.earned_amount:,.0f}")
                break

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
    summary = {"correct": 0, "wrong": 0, "timeouts": 0}

    for game_num in range(max_games):
        game = client.game.start(competition_id=competition_id)
        print(f"=== game {game_num + 1}/{max_games}, session {game.session_id} ===")

        while game.in_progress:
            q = game.current_question
            if not q:
                break

            time_left = game.time_remaining or 0
            print(f"\n--- level {game.current_level} ({time_left:.0f}s left on the wire) ---")
            print(f"Q: {q.text}")
            for opt in q.options:
                print(f"  [{opt.id}] {opt.text}")

            ctx = Context(competition_id=competition_id, level=q.level or game.current_level)
            decision = strategy(q, ctx)
            print(
                f"-> chose [{decision.option_id}] "
                f"(conf={decision.confidence:.2f}, {decision.latency_ms} ms)"
                if decision.confidence is not None
                else f"-> chose [{decision.option_id}] ({decision.latency_ms} ms)"
            )
            if decision.rationale:
                print(f"   reason: {decision.rationale}")

            result = game.answer(decision.option_id)

            log.record(
                PredictionRecord(
                    account_username=client.user.username,
                    session_id=game.session_id,
                    competition_id=competition_id,
                    level=q.level or game.current_level,
                    question_id=q.id,
                    question_text=q.text,
                    options=[{"id": o.id, "text": o.text} for o in q.options],
                    predicted_option_id=decision.option_id,
                    correct_option_id_if_known=decision.option_id if result.correct else None,
                    strategy_name=decision.strategy_name,
                    model_name=decision.model_name,
                    prompt_version=decision.prompt_version,
                    confidence=decision.confidence,
                    rationale=decision.rationale,
                    latency_ms=decision.latency_ms,
                )
            )

            if result.correct:
                summary["correct"] += 1
                print(f"correct — earned ${result.earned_amount:,.0f}")
            elif result.timed_out:
                summary["timeouts"] += 1
                print(f"timed out — earned ${result.earned_amount:,.0f}")
                break
            else:
                summary["wrong"] += 1
                print(f"wrong — earned ${result.earned_amount:,.0f}")

            if result.game_over:
                print(f"\ngame over: level {game.current_level}, ${result.earned_amount:,.0f}")
                break

        print()

    print(f"summary: {summary}")
    return summary
