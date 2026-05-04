"""Answer strategies.

A strategy is the swappable unit during the project: zero-shot, CoT, RAG,
ensemble, agent-with-tool. Every strategy implements the same callable
interface so `eval.replay` can iterate over (strategy x question) trivially.

Add new strategies as `<name>.py` in this package, and re-export the class
from this `__init__.py` once it's stable.
"""

from polimillionaire.strategies.base import AnswerDecision, Context, Strategy

__all__ = ["AnswerDecision", "Context", "Strategy"]
