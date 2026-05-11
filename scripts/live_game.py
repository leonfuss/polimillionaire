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
  - "auto"           per-competition routing: rag_calc_react for Maths
                     when a math index exists (calc_react otherwise);
                     wiki_rag for Entertainment/History/Science when a
                     wiki index exists (zero_shot otherwise).
  - "wiki_rag"       Wikipedia-RAG for the current competition; needs
                     `data/index/wiki_<competition>/` built via the
                     corpus pipeline.
  - "rag_calc_react" calc-react with retrieved MATH exemplars; needs
                     `data/index/math/` (build via build_math_index.py).
  - "calc_react"     calc-react with the four hand-crafted exemplars.
  - "zero_shot"      single prompted JSON completion.
"""

from __future__ import annotations

from polimillionaire import load_llm, make_client
from polimillionaire.play import auto_play_loop
from polimillionaire.strategies import make_strategy

# --- config -----------------------------------------------------------------

COMPETITION_ID = 3  # set to whichever competition you want to play
MODEL_NAME = "qwen3-8b"
STRATEGY_KIND = "auto"  # "auto", "wiki_rag", "rag_calc_react", "calc_react", "zero_shot"
MAX_GAMES = 1
MAX_STEPS = 3  # calc_react / rag_calc_react only
RAG_K = 3  # retrieved exemplars per math question (rag_calc_react only)

# ----------------------------------------------------------------------------


def main() -> None:
    print(f"loading {MODEL_NAME}...")
    llm = load_llm(MODEL_NAME)

    strategy = make_strategy(
        STRATEGY_KIND,
        llm,
        competition_id=COMPETITION_ID,
        verbose=True,
        max_steps=MAX_STEPS,
        k=RAG_K,
    )

    print(
        f"strategy={STRATEGY_KIND} (effective: {strategy.strategy_name}), "
        f"model={MODEL_NAME}, competition={COMPETITION_ID}, max_games={MAX_GAMES}"
    )

    client = make_client()
    auto_play_loop(
        client,
        competition_id=COMPETITION_ID,
        strategy=strategy,
        max_games=MAX_GAMES,
    )


if __name__ == "__main__":
    main()
