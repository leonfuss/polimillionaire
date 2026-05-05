"""Play a live game against the PoliMillionaire server, locally.

Reads credentials from `.env` (POLIMILLIONAIRE_API_URL / USER / PASSWORD).
Predictions land in the SQLite log at `POLIMILLIONAIRE_DB_PATH` if set,
else `data/questions.sqlite` relative to wherever you run from.

Run on a Mac:

    uv sync --group llm
    uv run python scripts/live_game.py

Edit the constants below to swap competitions, strategies, or budgets.
The script intentionally does no CLI parsing -- this is a debug surface,
not a tool. Re-run with edits.
"""

from __future__ import annotations

from polimillionaire import load_llm, make_client
from polimillionaire.play import auto_play_loop
from polimillionaire.strategies import CalcReactStrategy, ZeroShotStrategy

# --- config -----------------------------------------------------------------

COMPETITION_ID = 3  # set to whichever competition you want to play
MODEL_NAME = "qwen3-8b"
STRATEGY_KIND = "calc_react"  # "calc_react" or "zero_shot"
MAX_GAMES = 2
MAX_STEPS = 3  # calc_react only: tool calls before forced answer

# ----------------------------------------------------------------------------


def make_strategy(llm):
    if STRATEGY_KIND == "calc_react":
        return CalcReactStrategy(llm, max_steps=MAX_STEPS, verbose=True)
    if STRATEGY_KIND == "zero_shot":
        return ZeroShotStrategy(llm)
    raise ValueError(f"unknown STRATEGY_KIND: {STRATEGY_KIND!r}")


def main() -> None:
    print(f"loading {MODEL_NAME}...")
    llm = load_llm(MODEL_NAME)

    client = make_client()
    strategy = make_strategy(llm)

    print(
        f"strategy={STRATEGY_KIND}, model={MODEL_NAME}, "
        f"competition={COMPETITION_ID}, max_games={MAX_GAMES}"
    )
    auto_play_loop(
        client,
        competition_id=COMPETITION_ID,
        strategy=strategy,
        max_games=MAX_GAMES,
    )


if __name__ == "__main__":
    main()
