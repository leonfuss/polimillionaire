"""Historical progress report from the predictions DB.

Distinct from `eval/replay.py` -- which re-asks the questions to a strategy
*now* and reports offline accuracy. This script reads what was actually
played: each row in `predictions` records `strategy_name`, `model_name`,
`prompt_version`, `level`, and `session_id`, so we can group by
configuration and ask "how far did this combination historically reach in
each competition?". Useful for tracking real-game improvements across
prompt bumps and model swaps.

Per (competition, strategy, model, prompt_version) we report:
  - sessions:     number of distinct games played with this configuration
  - best:         highest level reached in any single game
  - mean / median:level reached, averaged across this config's games
  - p95_ms:       95th-percentile per-question latency (live time-budget proxy:
                  if this is approaching 30000 ms, you'll start timing out)

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
                   prompt_version, level, latency_ms
            FROM predictions
            """
        ).fetchall()

    # First pass: per session, per configuration -> max level + per-question latencies.
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
        s = per_session.setdefault(key, {"max_level": 0, "latencies": []})
        if r["level"] > s["max_level"]:
            s["max_level"] = r["level"]
        s["latencies"].append(r["latency_ms"])

    # Second pass: roll sessions up by configuration.
    by_config: dict[tuple, dict] = defaultdict(lambda: {"max_levels": [], "latencies": []})
    for (_session, comp, strat, model, prompt), s in per_session.items():
        cfg = (comp, strat, model, prompt)
        by_config[cfg]["max_levels"].append(s["max_level"])
        by_config[cfg]["latencies"].extend(s["latencies"])

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
            }
        )
    return out


def print_report(report: list[dict]) -> None:
    by_comp: dict[int, list[dict]] = defaultdict(list)
    for r in report:
        by_comp[r["competition_id"]].append(r)
    if not by_comp:
        print("(no rows in predictions table)")
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
            f"{'mean_ms':>7}  {'p95_ms':>7}"
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
                f"{r['mean_ms']:>7}  "
                f"{r['p95_ms']:>7}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Show how far each (strategy, model, prompt) configuration "
        "has historically reached in each competition."
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
