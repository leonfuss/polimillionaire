"""Run `eval.replay` with a real strategy from the command line.

Replays whichever strategy you choose against every labelled question in
the SQLite log and prints the combined / hand-labelled / server-confirmed
breakdown. Edit the constants below to swap routing or model.

`STRATEGY_KIND = "auto"` routes per competition (rag-calc-react or
calc-react on Maths, zero-shot elsewhere) -- the same routing
`continuous_play.py` uses for live games, so offline numbers reflect
what would actually run.

Run:

    uv sync --group llm --group rag
    uv run python scripts/replay.py                  # all categories
    uv run python scripts/replay.py -c 3             # just Maths
    uv run python scripts/replay.py --competition 0  # just Entertainment
"""

from __future__ import annotations

import argparse
from pathlib import Path

from polimillionaire import load_llm
from polimillionaire.eval.replay import print_summary, replay
from polimillionaire.llm import LLM
from polimillionaire.strategies import (
    CalcReactStrategy,
    RagCalcReactStrategy,
    ZeroShotStrategy,
)
from polimillionaire.strategies.base import AnswerDecision, Context, Strategy

# --- config -----------------------------------------------------------------

MODEL_NAME = "qwen3-8b"
# "auto" (per-competition routing), "zero_shot", "calc_react", "rag_calc_react".
STRATEGY_KIND = "auto"
MAX_STEPS = 3  # calc_react / rag_calc_react only
RAG_K = 3  # retrieved exemplars per math question (rag_calc_react only)
MATH_COMPETITION_ID = 3

# scripts/ -> repo root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = _PROJECT_ROOT / "data" / "questions.sqlite"
MATH_INDEX_DIR = _PROJECT_ROOT / "data" / "index" / "math"

# ----------------------------------------------------------------------------


def _load_math_retriever():
    """Return a `Retriever` over the math index, or `None` if not built yet."""
    if not MATH_INDEX_DIR.exists():
        return None
    try:
        from polimillionaire.retrieval import Retriever

        return Retriever(MATH_INDEX_DIR)
    except Exception as exc:  # noqa: BLE001 -- fall back to plain calc-react
        print(f"!! math retriever unavailable ({type(exc).__name__}: {exc}); using calc-react")
        return None


class RoutedStrategy:
    """Per-competition strategy router.

    Matches `continuous_play.make_strategy` so replay numbers reflect
    what would actually run during live play. Maths uses RAG-augmented
    calc-react when an index is loaded, plain calc-react otherwise.
    Everything else is zero-shot until Wikipedia RAG lands.
    """

    strategy_name = "routed"

    def __init__(self, llm: LLM, retriever=None) -> None:
        if retriever is not None:
            self._math: Strategy = RagCalcReactStrategy(
                llm, retriever, k=RAG_K, max_steps=MAX_STEPS
            )
        else:
            self._math = CalcReactStrategy(llm, max_steps=MAX_STEPS)
        self._zero = ZeroShotStrategy(llm)

    def __call__(self, question, ctx: Context) -> AnswerDecision:
        if ctx.competition_id == MATH_COMPETITION_ID:
            return self._math(question, ctx)
        return self._zero(question, ctx)


def make_strategy(llm: LLM, retriever=None) -> Strategy:
    if STRATEGY_KIND == "auto":
        return RoutedStrategy(llm, retriever=retriever)
    if STRATEGY_KIND == "rag_calc_react":
        if retriever is None:
            raise SystemExit(
                f"rag_calc_react needs a math index at {MATH_INDEX_DIR}. "
                "Run `uv run python scripts/build_math_index.py` first."
            )
        return RagCalcReactStrategy(llm, retriever, k=RAG_K, max_steps=MAX_STEPS)
    if STRATEGY_KIND == "calc_react":
        return CalcReactStrategy(llm, max_steps=MAX_STEPS)
    if STRATEGY_KIND == "zero_shot":
        return ZeroShotStrategy(llm)
    raise ValueError(f"unknown STRATEGY_KIND: {STRATEGY_KIND!r}")


_COMPETITION_NAMES = {
    0: "Entertainment",
    1: "Ancient History and Politics",
    2: "Science and Nature",
    3: "Maths",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "-c",
        "--competition",
        type=int,
        choices=sorted(_COMPETITION_NAMES),
        default=None,
        help="Replay only this competition id. Default: all four.",
    )
    args = parser.parse_args()

    if not DB_PATH.exists():
        raise SystemExit(f"No DB at {DB_PATH}. Play some games first.")

    print(f"loading {MODEL_NAME}...")
    llm = load_llm(MODEL_NAME)
    retriever = _load_math_retriever()
    if retriever is not None:
        print(f"math retriever loaded ({len(retriever)} problems)")
    else:
        print(f"no math index at {MATH_INDEX_DIR} — math falls back to plain calc-react")
    strategy = make_strategy(llm, retriever=retriever)
    if args.competition is None:
        scope = "all competitions"
    else:
        scope = f"competition={args.competition} ({_COMPETITION_NAMES[args.competition]})"
    print(f"strategy={STRATEGY_KIND}, model={MODEL_NAME}, db={DB_PATH}, scope={scope}\n")

    print_summary(
        replay(
            strategy,
            DB_PATH,
            competition_id=args.competition,
            show_progress=True,
        )
    )


if __name__ == "__main__":
    main()
