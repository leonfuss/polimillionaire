"""Play a live game against the PoliMillionaire server, locally.

Reads credentials from `.env` (POLIMILLIONAIRE_API_URL / USER / PASSWORD).
Predictions land in the SQLite log at `POLIMILLIONAIRE_DB_PATH` if set,
else `data/questions.sqlite` relative to wherever you run from.

Run on a Mac:

    uv sync --group llm --group rag
    uv run python scripts/live_game.py

Edit the constants below to swap competitions, strategies, or budgets.
The script intentionally does no CLI parsing -- this is a debug surface,
not a tool. Re-run with edits.

Strategy options:
  - "auto"           per-competition routing (matches continuous_play
                     and replay): rag_calc_react for Maths when an
                     index exists, calc_react otherwise; zero_shot for
                     the other three categories.
  - "rag_calc_react" calc-react with retrieved MATH exemplars; needs
                     `data/index/math/` (build via build_math_index.py).
  - "calc_react"     calc-react with the four hand-crafted exemplars.
  - "zero_shot"      single prompted JSON completion.
"""

from __future__ import annotations

from pathlib import Path

from polimillionaire import load_llm, make_client
from polimillionaire.llm import LLM
from polimillionaire.play import auto_play_loop
from polimillionaire.strategies import (
    CalcReactStrategy,
    RagCalcReactStrategy,
    ZeroShotStrategy,
)
from polimillionaire.strategies.base import Strategy

# --- config -----------------------------------------------------------------

COMPETITION_ID = 3  # set to whichever competition you want to play
MODEL_NAME = "qwen3-8b"
STRATEGY_KIND = "auto"  # "auto", "rag_calc_react", "calc_react", "zero_shot"
MAX_GAMES = 1
MAX_STEPS = 3  # calc_react / rag_calc_react only
RAG_K = 3  # retrieved exemplars per math question (rag_calc_react only)
MATH_COMPETITION_ID = 3

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
MATH_INDEX_DIR = _PROJECT_ROOT / "data" / "index" / "math"

# ----------------------------------------------------------------------------


def _load_math_retriever():
    """Return a `Retriever` over the math index, or `None` if unavailable.

    Lazy import keeps the optional `[rag]` deps off the critical path
    when this script is run for a non-math competition.
    """
    if not MATH_INDEX_DIR.exists():
        return None
    try:
        from polimillionaire.retrieval import Retriever

        return Retriever(MATH_INDEX_DIR)
    except Exception as exc:  # noqa: BLE001 -- fall back to plain calc-react
        print(f"!! math retriever unavailable ({type(exc).__name__}: {exc}); using calc-react")
        return None


def make_strategy(llm: LLM, competition_id: int, retriever=None) -> Strategy:
    if STRATEGY_KIND == "auto":
        if competition_id == MATH_COMPETITION_ID:
            if retriever is not None:
                return RagCalcReactStrategy(
                    llm, retriever, k=RAG_K, max_steps=MAX_STEPS, verbose=True
                )
            return CalcReactStrategy(llm, max_steps=MAX_STEPS, verbose=True)
        return ZeroShotStrategy(llm)
    if STRATEGY_KIND == "rag_calc_react":
        if retriever is None:
            raise SystemExit(
                f"rag_calc_react needs a math index at {MATH_INDEX_DIR}. "
                "Run `uv run python scripts/build_math_index.py` first."
            )
        return RagCalcReactStrategy(llm, retriever, k=RAG_K, max_steps=MAX_STEPS, verbose=True)
    if STRATEGY_KIND == "calc_react":
        return CalcReactStrategy(llm, max_steps=MAX_STEPS, verbose=True)
    if STRATEGY_KIND == "zero_shot":
        return ZeroShotStrategy(llm)
    raise ValueError(f"unknown STRATEGY_KIND: {STRATEGY_KIND!r}")


def main() -> None:
    print(f"loading {MODEL_NAME}...")
    llm = load_llm(MODEL_NAME)

    # Only load the retriever when there's a chance the chosen strategy
    # will use it; saves a sentence-transformers + faiss load for the
    # zero-shot/calc-react debug paths on a non-math competition.
    needs_retriever = STRATEGY_KIND == "rag_calc_react" or (
        STRATEGY_KIND == "auto" and COMPETITION_ID == MATH_COMPETITION_ID
    )
    retriever = _load_math_retriever() if needs_retriever else None
    if retriever is not None:
        print(f"math retriever loaded ({len(retriever)} problems)")
    elif needs_retriever:
        print(f"no math index at {MATH_INDEX_DIR} — math falls back to plain calc-react")

    client = make_client()
    strategy = make_strategy(llm, COMPETITION_ID, retriever=retriever)

    print(
        f"strategy={STRATEGY_KIND} (effective: {strategy.strategy_name}), "
        f"model={MODEL_NAME}, competition={COMPETITION_ID}, max_games={MAX_GAMES}"
    )
    auto_play_loop(
        client,
        competition_id=COMPETITION_ID,
        strategy=strategy,
        max_games=MAX_GAMES,
    )


if __name__ == "__main__":
    main()
