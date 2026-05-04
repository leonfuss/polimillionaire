"""Manual play helper for corpus bootstrap.

Drives a human-in-the-loop game session, logging every question + your
answer to the SQLite log. Each teammate runs this on their assigned
competition until we have ~50 labelled questions per competition.

Usage in a Colab cell:

    from polimillionaire import make_client
    from polimillionaire.play import manual_play_loop

    client = make_client()
    manual_play_loop(client, competition_id=0, max_games=3)
"""

from __future__ import annotations

import time

from polimillionaire._vendor.millionaire_client import MillionaireClient
from polimillionaire.recording import PredictionRecord, QuestionLog


def manual_play_loop(
    client: MillionaireClient,
    competition_id: int,
    db_path: str | None = None,
    max_games: int = 1,
) -> None:
    """Play `max_games` consecutive games, logging every question to the DB.

    At each prompt, type the option id (or `q` to abort cleanly).
    Picks up the DB path from `POLIMILLIONAIRE_DB_PATH` if `db_path` is None.
    """
    import os

    log = QuestionLog(db_path or os.environ.get("POLIMILLIONAIRE_DB_PATH", "data/questions.sqlite"))

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
