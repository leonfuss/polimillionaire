"""Replay a strategy over every labelled question in the log."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

from polimillionaire._vendor.millionaire_client.models import Option, Question
from polimillionaire.strategies.base import AnswerDecision, Context, Strategy


def replay(strategy: Strategy, log_path: str | Path) -> dict:
    """Run `strategy` over every question in the log that has a known answer."""
    path = Path(log_path)
    if not path.exists():
        raise FileNotFoundError(f"No question log at {path}")

    with sqlite3.connect(path) as con:
        con.row_factory = sqlite3.Row
        cur = con.execute(
            """
            SELECT DISTINCT question_id, question_text, options_json, level,
                            competition_id, correct_option_id_if_known
            FROM predictions
            WHERE correct_option_id_if_known IS NOT NULL
            """
        )
        rows = cur.fetchall()

    correct = 0
    for row in rows:
        options = [Option(**o) for o in json.loads(row["options_json"])]
        question = Question(
            id=row["question_id"],
            text=row["question_text"],
            options=options,
            level=row["level"],
        )
        ctx = Context(competition_id=row["competition_id"], level=row["level"])
        if strategy(question, ctx).option_id == row["correct_option_id_if_known"]:
            correct += 1

    total = len(rows)
    return {"correct": correct, "total": total, "accuracy": correct / total if total else 0.0}


def _smoke() -> None:
    """Trivial baseline that always picks option 0."""

    def always_zero(_question: Question, _ctx: Context) -> AnswerDecision:
        return AnswerDecision(option_id=0, strategy_name="always_zero")

    log_path = Path("data/questions.sqlite")
    if not log_path.exists():
        print(f"No log at {log_path} yet. Run a live game first to populate it.")
        return
    print(replay(always_zero, log_path))


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay a strategy over the question log.")
    parser.add_argument("--smoke", action="store_true", help="Run a trivial baseline.")
    args = parser.parse_args()
    if args.smoke:
        _smoke()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
