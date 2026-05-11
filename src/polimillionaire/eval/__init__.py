"""Offline evaluation harness.

Replays strategies over the SQLite question log so we can iterate on
prompts/RAG/ensembles without burning the live API or the 30-second timer.
"""

from polimillionaire.eval.replay import print_summary, replay, replay_records
from polimillionaire.eval.results import ReplayResult, to_polars

__all__ = ["ReplayResult", "print_summary", "replay", "replay_records", "to_polars"]
