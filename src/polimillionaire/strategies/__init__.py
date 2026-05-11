"""Answer strategies.

A strategy is the swappable unit during the project: zero-shot, CoT, RAG,
ensemble, agent-with-tool. Every strategy implements the same callable
interface so `eval.replay` can iterate over (strategy x question) trivially.

Add new strategies as `<name>.py` in this package, and re-export the class
from this `__init__.py` once it's stable.
"""

from polimillionaire.strategies.base import AnswerDecision, Context, Strategy
from polimillionaire.strategies.calc_react import CalcReactStrategy
from polimillionaire.strategies.rag_calc_react import RagCalcReactStrategy
from polimillionaire.strategies.wiki_rag import WikiRagStrategy
from polimillionaire.strategies.zero_shot import ZeroShotStrategy

__all__ = [
    "AnswerDecision",
    "CalcReactStrategy",
    "Context",
    "RagCalcReactStrategy",
    "Strategy",
    "WikiRagStrategy",
    "ZeroShotStrategy",
]
