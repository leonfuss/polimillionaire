"""Run `eval.replay` with a real strategy from the command line.

Replays whichever strategy you choose against every labelled question in
the SQLite log and prints the combined / hand-labelled / server-confirmed
breakdown. Edit the constants below to swap routing or model.

`STRATEGY_KIND = "auto"` routes per competition (wiki_rag for
Entertainment/History/Science, rag_calc_react or calc_react for Maths)
-- the same routing `continuous_play.py` uses for live games, so offline
numbers reflect what would actually run.

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
from polimillionaire.strategies import make_strategy

# --- config -----------------------------------------------------------------

MODEL_NAME = "qwen3-8b"
# "auto" (per-competition routing), "zero_shot", "calc_react", "rag_calc_react".
STRATEGY_KIND = "auto"
MAX_STEPS = 3  # calc_react / rag_calc_react only
RAG_K = 3  # retrieved exemplars per math question (rag_calc_react only)

# scripts/ -> repo root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = _PROJECT_ROOT / "data" / "questions.sqlite"

# ----------------------------------------------------------------------------

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

    # no competition_id -> build all routes (replay path)
    strategy = make_strategy(
        STRATEGY_KIND,
        llm,
        max_steps=MAX_STEPS,
        k=RAG_K,
    )

    if args.competition is None:
        scope = "all competitions"
    else:
        scope = f"competition={args.competition} ({_COMPETITION_NAMES[args.competition]})"
    print(
        f"strategy={STRATEGY_KIND} (effective: {strategy.strategy_name}), model={MODEL_NAME}, db={DB_PATH}, scope={scope}\n"
    )

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
