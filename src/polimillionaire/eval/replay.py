"""Replay a strategy over every labelled question in the log.

`replay_records()` is the primary entry point: it returns one `ReplayResult` per
question, which the notebook composes into polars tables. `replay()` is a thin
sugar layer that aggregates the records into the old `combined/hand_labeled/
server_confirmed` dict shape so existing CLI callers keep working.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any

from polimillionaire._vendor.millionaire_client.models import Option, Question
from polimillionaire.eval.results import ReplayResult
from polimillionaire.strategies.base import AnswerDecision, Context, Strategy

# Stable enough to inline rather than fetch from the API (which would require
# auth just to render a CLI summary). Sourced from the README.
COMPETITION_NAMES: dict[int, str] = {
    0: "Entertainment",
    1: "Ancient History and Politics",
    2: "Science and Nature",
    3: "Maths",
}


def replay_records(
    strategy: Strategy,
    log_path: str | Path,
    *,
    competition_id: int | None = None,
    show_progress: bool = False,
) -> list[ReplayResult]:
    """Run `strategy` over every labelled question; return one ReplayResult per question."""
    path = Path(log_path)
    if not path.exists():
        raise FileNotFoundError(f"No question log at {path}")

    with sqlite3.connect(path) as con:
        con.row_factory = sqlite3.Row
        # GROUP BY collapses the multiple prediction rows per question to one
        # eval row. MAX(generated_answer) means: if a question was ever
        # hand-labelled, treat it as hand-labelled (in practice the flag is
        # consistent per question; this is just defence in depth).
        sql = """
            SELECT question_id,
                   MAX(question_text)              AS question_text,
                   MAX(options_json)               AS options_json,
                   MAX(level)                      AS level,
                   competition_id,
                   MAX(correct_option_id_if_known) AS correct_option_id_if_known,
                   MAX(generated_answer)           AS generated_answer
            FROM predictions
            WHERE correct_option_id_if_known IS NOT NULL
            """
        params: list = []
        if competition_id is not None:
            sql += " AND competition_id = ? "
            params.append(competition_id)
        sql += " GROUP BY question_id, competition_id"
        cur = con.execute(sql, params)
        rows = cur.fetchall()

    results: list[ReplayResult] = []
    correct_so_far = 0

    iterable: Any = rows
    pbar = None
    if show_progress:
        # Imported lazily so the eval module stays importable on a base
        # install (tqdm is pulled in transitively by the `[rag]` group).
        from tqdm import tqdm

        pbar = tqdm(rows, desc="replay", unit="q")
        iterable = pbar

    for row in iterable:
        options = [Option(**o) for o in json.loads(row["options_json"])]
        question = Question(
            id=row["question_id"],
            text=row["question_text"],
            options=options,
            level=row["level"],
        )
        ctx = Context(competition_id=row["competition_id"], level=row["level"])
        decision = strategy(question, ctx)
        # `correct_option_id_if_known` is NOT NULL-filtered in the WHERE clause,
        # but sqlite3.Row returns Any -- cast so the dataclass typing stays honest.
        correct_id = int(row["correct_option_id_if_known"])
        is_right = decision.option_id == correct_id
        if is_right:
            correct_so_far += 1

        results.append(
            ReplayResult(
                strategy_name=decision.strategy_name,
                model_name=decision.model_name,
                prompt_version=decision.prompt_version,
                question_id=row["question_id"],
                competition_id=row["competition_id"],
                level=row["level"],
                predicted_option_id=decision.option_id,
                correct_option_id=correct_id,
                correct=is_right,
                confidence=decision.confidence,
                latency_ms=decision.latency_ms,
                generated_answer=bool(row["generated_answer"]),
            )
        )

        if pbar is not None:
            pbar.set_postfix(
                comp=row["competition_id"],
                lvl=row["level"],
                acc=f"{correct_so_far / len(results):.0%}",
                refresh=False,
            )

    if pbar is not None:
        pbar.close()

    return results


def replay(
    strategy: Strategy,
    log_path: str | Path,
    *,
    competition_id: int | None = None,
    show_progress: bool = False,
) -> dict:
    """Run `strategy` over every question in the log that has a known answer.

    `competition_id`, if set, restricts the replay to that single competition
    (0-3). When `None`, every labelled question is replayed regardless of
    category.

    `show_progress=True` renders a tqdm progress bar with running accuracy
    in the postfix. Off by default so library callers (and the smoke test)
    stay silent.

    Returns combined accuracy with a `by_competition` breakdown, plus two
    sub-views over the same shape:
      - `hand_labeled`: rows where `generated_answer = 1` (we reasoned the
        answer; ground truth is opinion, not server-confirmed).
      - `server_confirmed`: rows where `generated_answer = 0` (the server
        said "correct" at play time; ground truth is gold).
    Splitting these lets us see whether accuracy is driven by easy
    server-confirmed questions or holds up on the harder hand-labelled set.
    """
    records = replay_records(
        strategy, log_path, competition_id=competition_id, show_progress=show_progress
    )
    return _aggregate(records)


def _aggregate(records: list[ReplayResult]) -> dict:
    """Build the combined/hand_labeled/server_confirmed summary dict from flat records."""
    combined = _empty_bucket()
    hand_labeled = _empty_bucket()
    server_confirmed = _empty_bucket()

    for r in records:
        _record(combined, r.competition_id, r.correct)
        if r.generated_answer:
            _record(hand_labeled, r.competition_id, r.correct)
        else:
            _record(server_confirmed, r.competition_id, r.correct)

    result = _finalize(combined)
    result["hand_labeled"] = _finalize(hand_labeled)
    result["server_confirmed"] = _finalize(server_confirmed)
    return result


def print_summary(results: dict) -> None:
    """Pretty-print replay results: combined, hand-labelled, server-confirmed."""
    _print_table("combined", results)
    _print_table("hand-labelled", results["hand_labeled"])
    _print_table("server-confirmed", results["server_confirmed"])


def _empty_bucket() -> dict[str, Any]:
    return {"correct": 0, "total": 0, "by_comp": {}}


def _record(bucket: dict[str, Any], comp_id: int, is_right: bool) -> None:
    bucket["total"] += 1
    if is_right:
        bucket["correct"] += 1
    sub = bucket["by_comp"].setdefault(comp_id, {"correct": 0, "total": 0})
    sub["total"] += 1
    if is_right:
        sub["correct"] += 1


def _finalize(bucket: dict[str, Any]) -> dict[str, Any]:
    total = bucket["total"]
    return {
        "correct": bucket["correct"],
        "total": total,
        "accuracy": bucket["correct"] / total if total else 0.0,
        "by_competition": {
            cid: {
                "name": COMPETITION_NAMES.get(cid, f"competition_{cid}"),
                "correct": s["correct"],
                "total": s["total"],
                "accuracy": s["correct"] / s["total"] if s["total"] else 0.0,
            }
            for cid, s in sorted(bucket["by_comp"].items())
        },
    }


def _print_table(label: str, totals: dict) -> None:
    print(f"=== {label} ({totals['total']} rows) ===")
    rows = list(totals["by_competition"].values())
    if not rows:
        print("(no rows)\n")
        return
    name_width = max(max(len(r["name"]) for r in rows), len("competition"), len("overall"))
    header = f"{'competition':<{name_width}}  {'correct':>7} / {'total':<5}  {'accuracy':>8}"
    print(header)
    print("-" * len(header))
    for stats in rows:
        print(
            f"{stats['name']:<{name_width}}  "
            f"{stats['correct']:>7} / {stats['total']:<5}  "
            f"{stats['accuracy']:>7.1%}"
        )
    print("-" * len(header))
    print(
        f"{'overall':<{name_width}}  "
        f"{totals['correct']:>7} / {totals['total']:<5}  "
        f"{totals['accuracy']:>7.1%}"
    )
    print()


def _smoke() -> None:
    """Trivial baseline that always picks option 0."""

    def always_zero(_question: Question, _ctx: Context) -> AnswerDecision:
        return AnswerDecision(option_id=0, strategy_name="always_zero")

    log_path = Path("data/questions.sqlite")
    if not log_path.exists():
        print(f"No log at {log_path} yet. Run a live game first to populate it.")
        return
    print_summary(replay(always_zero, log_path))


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay a strategy over the question log.")
    parser.add_argument("--smoke", action="store_true", help="Run a trivial baseline.")
    args = parser.parse_args()
    if args.smoke:
        _smoke()
    else:
        parser.print_help()


def nb_replay2(strategy, log_path, competition_id=None, limit=None):
    import sqlite3
    import json
    from tqdm import tqdm
    from polimillionaire._vendor.millionaire_client.models import Option, Question
    from polimillionaire.eval.results import ReplayResult
    from polimillionaire.strategies.base import Context
    
    # 1. Fetch and close immediately. Do not hold DB locks during inference.
    with sqlite3.connect(log_path) as con:
        con.row_factory = sqlite3.Row
        
        sql = """
            WITH RankedPredictions AS (
                SELECT question_id, question_text, options_json, level, 
                       competition_id, correct_option_id_if_known, generated_answer,
                       ROW_NUMBER() OVER(PARTITION BY question_id ORDER BY id DESC) as rn
                FROM predictions
                WHERE correct_option_id_if_known IS NOT NULL
        """
        params = []
        
        if competition_id is not None:
            sql += " AND competition_id = ?"
            params.append(competition_id)
            
        sql += """
            )
            SELECT * FROM RankedPredictions WHERE rn = 1
            ORDER BY question_id  -- CRITICAL: Deterministic sorting for reliable ablation
        """
        
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
            
        cur = con.execute(sql, params)
        rows = cur.fetchall()

    results = []
    
    for row in tqdm(rows, desc=f"Evaluating Domain {competition_id}"):
        options = [Option(**o) for o in json.loads(row["options_json"])]
        question = Question(id=row["question_id"], text=row["question_text"], options=options, level=row["level"])
        ctx = Context(competition_id=row["competition_id"], level=row["level"])
        
        decision = strategy(question, ctx)
        correct_id = int(row["correct_option_id_if_known"])
        
        # Safely parse SQLite mixed-type booleans
        raw_gen = row["generated_answer"]
        is_generated = raw_gen in (1, '1', 'true', 'True', True)
        
        results.append(ReplayResult(
            strategy_name=decision.strategy_name, 
            model_name=decision.model_name, 
            prompt_version=decision.prompt_version, 
            question_id=row["question_id"], 
            competition_id=row["competition_id"], 
            level=row["level"],
            predicted_option_id=decision.option_id, 
            correct_option_id=correct_id, 
            correct=(decision.option_id == correct_id), 
            confidence=decision.confidence, 
            latency_ms=decision.latency_ms, 
            generated_answer=is_generated
        ))

    return results


if __name__ == "__main__":
    main()
