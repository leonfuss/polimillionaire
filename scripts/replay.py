"""Run `eval.replay` with a real strategy from the command line.

Replays whichever strategy you choose against every labelled question in
the SQLite log and prints the combined / hand-labelled / server-confirmed
breakdown. Edit the constants below to swap routing or model. The script
intentionally has no CLI flags -- this is a debug/eval surface, re-run
with edits.

`STRATEGY_KIND = "auto"` routes per competition (calc-react on Maths,
zero-shot elsewhere) -- the same routing `continuous_play.py` uses for
live games, so offline numbers reflect what would actually run.

Run:

    uv sync --group llm
    uv run python scripts/replay.py
"""

from __future__ import annotations

from pathlib import Path

from polimillionaire import load_llm
from polimillionaire.eval.replay import print_summary, replay
from polimillionaire.llm import LLM
from polimillionaire.strategies import CalcReactStrategy, ZeroShotStrategy
from polimillionaire.strategies.base import AnswerDecision, Context, Strategy

# --- config -----------------------------------------------------------------

MODEL_NAME = "qwen3-8b"
STRATEGY_KIND = "auto"  # "auto" (per-competition), "zero_shot", or "calc_react"
MAX_STEPS = 3  # calc_react only
MATH_COMPETITION_ID = 3

# scripts/ -> repo root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = _PROJECT_ROOT / "data" / "questions.sqlite"

# ----------------------------------------------------------------------------


class RoutedStrategy:
    """Per-competition strategy router.

    Matches `continuous_play.make_strategy` so replay numbers reflect
    what would actually run during live play. Calc-react for Maths,
    zero-shot for everything else.
    """

    strategy_name = "routed"

    def __init__(self, llm: LLM) -> None:
        self._calc = CalcReactStrategy(llm, max_steps=MAX_STEPS)
        self._zero = ZeroShotStrategy(llm)

    def __call__(self, question, ctx: Context) -> AnswerDecision:
        if ctx.competition_id == MATH_COMPETITION_ID:
            return self._calc(question, ctx)
        return self._zero(question, ctx)


def make_strategy(llm: LLM) -> Strategy:
    if STRATEGY_KIND == "auto":
        return RoutedStrategy(llm)
    if STRATEGY_KIND == "calc_react":
        return CalcReactStrategy(llm, max_steps=MAX_STEPS)
    if STRATEGY_KIND == "zero_shot":
        return ZeroShotStrategy(llm)
    raise ValueError(f"unknown STRATEGY_KIND: {STRATEGY_KIND!r}")


def main() -> None:
    if not DB_PATH.exists():
        raise SystemExit(f"No DB at {DB_PATH}. Play some games first.")

    print(f"loading {MODEL_NAME}...")
    llm = load_llm(MODEL_NAME)
    strategy = make_strategy(llm)
    print(f"strategy={STRATEGY_KIND}, model={MODEL_NAME}, db={DB_PATH}\n")

    print_summary(replay(strategy, DB_PATH))


if __name__ == "__main__":
    main()
