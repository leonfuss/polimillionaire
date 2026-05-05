"""Historical progress report from the predictions DB.

Distinct from `eval/replay.py` -- which re-asks the questions to a
strategy *now* and reports offline accuracy. This script reads what was
actually played: each row in `predictions` records `strategy_name`,
`model_name`, `prompt_version`, `level`, `predicted_option_id`,
`correct_option_id_if_known`, and `session_id`, so we can group by
configuration and ask both "how far did this combination historically
reach in each competition?" and "how often did it get the answer
right?". Useful for tracking real-game improvements across prompt
bumps and model swaps.

The script prints two reports:

  1. **Accuracy summary by (model, competition)** -- one accuracy
     number per (model, competition), collapsing across strategy and
     prompt. The shape that matches `replay.py`'s by-competition view,
     so live and offline numbers can be eyeballed side by side.

  2. **Detailed progress by configuration** -- the existing per-
     (competition, strategy, model, prompt_version) table with peak /
     mean / median level reached, latency stats, and now an `acc`
     column (correct over total questions answered).

Accuracy treats a NULL `correct_option_id_if_known` as a wrong answer:
that NULL only appears when the server rejected the bot's answer at
play time, and rows that were hand-labelled have the column populated.
So `correct = (predicted == correct_id)` (which is `False` for NULL
correct_ids) is the right truth check across both labelling modes.

Manual play (`strategy_name = manual_human`) is included for reference.

Run:

    uv run python scripts/progress_report.py
    uv run python scripts/progress_report.py --db /path/to/questions.sqlite
"""

from __future__ import annotations

import argparse
import sqlite3
from collections import defaultdict
from pathlib import Path

# Mirror eval/replay.py — same source (README) so output stays consistent.
COMPETITION_NAMES: dict[int, str] = {
    0: "Entertainment",
    1: "Ancient History and Politics",
    2: "Science and Nature",
    3: "Maths",
}

# scripts/ -> repo root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_DB = _PROJECT_ROOT / "data" / "questions.sqlite"


def _percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile. `pct` in [0, 100]."""
    if not values:
        return 0.0
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round(pct / 100.0 * (len(s) - 1)))))
    return s[k]


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2


def collect(db_path: Path) -> list[dict]:
    """Group predictions by session, then by configuration.

    Returns one dict per (competition, strategy, model, prompt_version)
    with aggregate stats over the games that configuration played.
    """
    with sqlite3.connect(db_path) as con:
        con.row_factory = sqlite3.Row
        rows = con.execute(
            """
            SELECT session_id, competition_id, strategy_name, model_name,
                   prompt_version, level, latency_ms,
                   predicted_option_id, correct_option_id_if_known
            FROM predictions
            """
        ).fetchall()

    # First pass: per session, per configuration -> max level + latencies +
    # correct/total. Tracking correctness per session (rather than only at
    # rollup) is overkill today but keeps the door open for "how many
    # sessions cleared X correct" follow-ups later.
    per_session: dict[tuple, dict] = {}
    for r in rows:
        # Configuration is (competition, strategy, model, prompt_version);
        # session_id distinguishes individual games inside that config.
        key = (
            r["session_id"],
            r["competition_id"],
            r["strategy_name"],
            r["model_name"],
            r["prompt_version"],
        )
        s = per_session.setdefault(key, {"max_level": 0, "latencies": [], "correct": 0, "total": 0})
        if r["level"] > s["max_level"]:
            s["max_level"] = r["level"]
        s["latencies"].append(r["latency_ms"])
        s["total"] += 1
        # `predicted_option_id == None` is False, so a NULL correct_id
        # (server-rejected, not yet hand-labelled) counts as a wrong
        # answer -- which it is, by definition of how the column gets set.
        if r["predicted_option_id"] == r["correct_option_id_if_known"]:
            s["correct"] += 1

    # Second pass: roll sessions up by configuration.
    by_config: dict[tuple, dict] = defaultdict(
        lambda: {"max_levels": [], "latencies": [], "correct": 0, "total": 0}
    )
    for (_session, comp, strat, model, prompt), s in per_session.items():
        cfg = (comp, strat, model, prompt)
        by_config[cfg]["max_levels"].append(s["max_level"])
        by_config[cfg]["latencies"].extend(s["latencies"])
        by_config[cfg]["correct"] += s["correct"]
        by_config[cfg]["total"] += s["total"]

    out: list[dict] = []
    for (comp, strat, model, prompt), agg in by_config.items():
        levels = agg["max_levels"]
        latencies = agg["latencies"]
        out.append(
            {
                "competition_id": comp,
                "strategy_name": strat,
                "model_name": model,
                "prompt_version": prompt,
                "sessions": len(levels),
                "best_level": max(levels),
                "mean_level": sum(levels) / len(levels),
                "median_level": _median(levels),
                "mean_ms": int(sum(latencies) / len(latencies)) if latencies else 0,
                "p95_ms": int(_percentile(latencies, 95)) if latencies else 0,
                "correct": agg["correct"],
                "total_answered": agg["total"],
                "accuracy": agg["correct"] / agg["total"] if agg["total"] else 0.0,
            }
        )
    return out


def summarize_by_model_competition(report: list[dict]) -> list[dict]:
    """Collapse the per-config rows down to one row per (model, competition).

    Same shape that `eval/replay.py` reports per competition, so a live
    progress accuracy can be compared directly against an offline
    replay accuracy. Strategy and prompt are merged together; if you
    care which strategy drove a number, look at the detailed table.
    """
    by_pair: dict[tuple[str, int], dict] = defaultdict(
        lambda: {"correct": 0, "total": 0, "sessions": 0, "best_level": 0, "levels_sum": 0}
    )
    for r in report:
        key = (r["model_name"], r["competition_id"])
        b = by_pair[key]
        b["correct"] += r["correct"]
        b["total"] += r["total_answered"]
        b["sessions"] += r["sessions"]
        # mean is a session-weighted mean of per-config means; fine for a summary.
        b["levels_sum"] += r["mean_level"] * r["sessions"]
        b["best_level"] = max(b["best_level"], r["best_level"])

    return [
        {
            "model_name": model,
            "competition_id": comp,
            "competition_name": COMPETITION_NAMES.get(comp, f"competition_{comp}"),
            "sessions": b["sessions"],
            "best_level": b["best_level"],
            "mean_level": b["levels_sum"] / b["sessions"] if b["sessions"] else 0.0,
            "correct": b["correct"],
            "total_answered": b["total"],
            "accuracy": b["correct"] / b["total"] if b["total"] else 0.0,
        }
        for (model, comp), b in by_pair.items()
    ]


def print_accuracy_summary(rows: list[dict]) -> None:
    """Top-level table: one row per (model, competition), sorted for compare."""
    print("\n=== Accuracy by model and competition ===")
    if not rows:
        print("(no rows in predictions table)")
        return

    # Sort: by competition first (so all rows for the same category cluster),
    # then by accuracy descending so the best model in each category is on top.
    rows = sorted(rows, key=lambda r: (r["competition_id"], -r["accuracy"]))

    model_w = max((len(r["model_name"]) for r in rows), default=5)
    comp_w = max((len(r["competition_name"]) for r in rows), default=11)
    model_w = max(model_w, len("model"))
    comp_w = max(comp_w, len("competition"))

    header = (
        f"{'model':<{model_w}}  {'competition':<{comp_w}}  "
        f"{'sessions':>8}  {'best':>4}  {'mean':>5}  "
        f"{'correct':>7}/{'total':<5}  {'acc':>6}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['model_name']:<{model_w}}  "
            f"{r['competition_name']:<{comp_w}}  "
            f"{r['sessions']:>8}  "
            f"{r['best_level']:>4}  "
            f"{r['mean_level']:>5.1f}  "
            f"{r['correct']:>7}/{r['total_answered']:<5}  "
            f"{r['accuracy']:>6.1%}"
        )


def print_detailed_report(report: list[dict]) -> None:
    by_comp: dict[int, list[dict]] = defaultdict(list)
    for r in report:
        by_comp[r["competition_id"]].append(r)
    if not by_comp:
        return

    for comp_id in sorted(by_comp):
        comp_name = COMPETITION_NAMES.get(comp_id, f"competition_{comp_id}")
        print(f"\n=== {comp_name} (competition {comp_id}) ===")
        # Sort within competition: best mean level first, then best peak.
        rows = sorted(
            by_comp[comp_id],
            key=lambda r: (r["mean_level"], r["best_level"]),
            reverse=True,
        )

        strat_w = max((len(r["strategy_name"]) for r in rows), default=8)
        model_w = max((len(r["model_name"]) for r in rows), default=5)
        prompt_w = max((len(r["prompt_version"]) for r in rows), default=6)
        strat_w = max(strat_w, len("strategy"))
        model_w = max(model_w, len("model"))
        prompt_w = max(prompt_w, len("prompt"))

        header = (
            f"{'strategy':<{strat_w}}  {'model':<{model_w}}  {'prompt':<{prompt_w}}  "
            f"{'sessions':>8}  {'best':>4}  {'median':>6}  {'mean':>5}  "
            f"{'acc':>6}  {'mean_ms':>7}  {'p95_ms':>7}"
        )
        print(header)
        print("-" * len(header))
        for r in rows:
            print(
                f"{r['strategy_name']:<{strat_w}}  "
                f"{r['model_name']:<{model_w}}  "
                f"{r['prompt_version']:<{prompt_w}}  "
                f"{r['sessions']:>8}  "
                f"{r['best_level']:>4}  "
                f"{r['median_level']:>6.1f}  "
                f"{r['mean_level']:>5.1f}  "
                f"{r['accuracy']:>6.1%}  "
                f"{r['mean_ms']:>7}  "
                f"{r['p95_ms']:>7}"
            )


def print_report(report: list[dict]) -> None:
    """Print both the model-level summary and the per-config detail."""
    print_accuracy_summary(summarize_by_model_competition(report))
    print_detailed_report(report)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Show how far each (strategy, model, prompt) configuration "
        "has historically reached in each competition, plus an accuracy "
        "rollup per (model, competition) for replay comparisons."
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=_DEFAULT_DB,
        help=f"Path to predictions DB (default: {_DEFAULT_DB})",
    )
    args = parser.parse_args()
    if not args.db.exists():
        raise SystemExit(f"No DB at {args.db}. Play some games first.")
    print_report(collect(args.db))


if __name__ == "__main__":
    main()
