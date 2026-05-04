"""Offline evaluation harness.

Replays strategies over the SQLite question log so we can iterate on
prompts/RAG/ensembles without burning the live API or the 30-second timer.
"""

from polimillionaire.eval.replay import replay

__all__ = ["replay"]
