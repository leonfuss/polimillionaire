"""Continuously play games to fill the SQLite log for hand-labelling.

Plays one game at a time, rotating across competitions, sleeping
`BREAK_SECONDS` between games. Any exception inside a game is caught,
logged, and the loop continues -- so a flaky network, a server hiccup,
or a one-off LLM/grammar parse error doesn't kill the run.

Goal isn't to score well; it's to log as many distinct questions as
possible. Strategy is routed per competition: calc-react for Maths
(competition 3), zero-shot for everything else. The math exemplars in
calc-react actively bias the model toward calling `calculate` on
non-numeric questions, so it's the wrong tool for the other three
categories; zero-shot is also faster, which means more questions per
unit time on the easier categories. Wrong answers end the current game;
the wrapper just starts the next one. Everything is recorded -- including
incorrect predictions -- so afterwards we can hand-label
`correct_option_id_if_known` for the rows that need it.

Run on a Mac:

    uv sync --group llm
    uv run python scripts/continuous_play.py

Stop with Ctrl-C.
"""

from __future__ import annotations

import time
import traceback
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

MODEL_NAME = "qwen3-8b"
COMPETITION_IDS = [0, 1, 2, 3]  # rotate to fill all four categories
MATH_COMPETITION_ID = 3  # the only category that gets calc-react
BREAK_SECONDS = 20  # pause between games
MAX_STEPS = 3  # calc_react only
RAG_K = 3  # retrieved exemplars per math question (rag_calc_react only)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
MATH_INDEX_DIR = _PROJECT_ROOT / "data" / "index" / "math"

# ----------------------------------------------------------------------------


def _load_math_retriever():
    """Return a `Retriever` over the math index, or `None` if not built yet.

    Lazily imports the retrieval module so missing `[rag]` deps don't
    break this script when math RAG isn't being used.
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
    """RAG-calc-react for Maths if an index is loaded, else plain calc-react.

    Zero-shot for the other competitions until the Wikipedia index lands.
    """
    if competition_id == MATH_COMPETITION_ID:
        if retriever is not None:
            return RagCalcReactStrategy(llm, retriever, k=RAG_K, max_steps=MAX_STEPS)
        return CalcReactStrategy(llm, max_steps=MAX_STEPS)
    return ZeroShotStrategy(llm)


def main() -> None:
    print(f"loading {MODEL_NAME}...")
    llm = load_llm(MODEL_NAME)
    retriever = _load_math_retriever()
    if retriever is not None:
        print(f"math retriever loaded ({len(retriever)} problems)")
    else:
        print(f"no math index at {MATH_INDEX_DIR} — math falls back to plain calc-react")
    print(f"model={MODEL_NAME}, rotating competitions {COMPETITION_IDS}")
    print("Ctrl-C to stop.\n")

    games = 0
    errors = 0
    while True:
        comp_id = COMPETITION_IDS[games % len(COMPETITION_IDS)]
        strategy = make_strategy(llm, comp_id, retriever=retriever)
        print(
            f">>> game #{games + 1}, competition={comp_id}, "
            f"strategy={strategy.strategy_name} (errors so far: {errors})"
        )
        try:
            # Re-create the client each iteration so an expired auth cookie
            # or a dropped session heals on the next loop without manual
            # intervention.
            client = make_client()
            auto_play_loop(
                client,
                competition_id=comp_id,
                strategy=strategy,
                max_games=1,
            )
        except KeyboardInterrupt:
            print("\nstopping.")
            return
        except Exception as exc:  # noqa: BLE001 -- this loop must not die
            errors += 1
            print(f"!! game errored ({type(exc).__name__}): {exc}")
            traceback.print_exc()

        games += 1
        try:
            time.sleep(BREAK_SECONDS)
        except KeyboardInterrupt:
            print("\nstopping.")
            return


if __name__ == "__main__":
    main()
