"""Strategy abstraction.

Single shape every strategy returns; lets the recording layer and the
eval harness iterate over heterogeneous strategies uniformly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from polimillionaire._vendor.millionaire_client.models import Question


@dataclass(frozen=True)
class AnswerDecision:
    option_id: int
    confidence: float | None = None
    rationale: str | None = None
    model_name: str = ""
    strategy_name: str = ""
    prompt_version: str = ""
    latency_ms: int = 0


@dataclass(frozen=True)
class Context:
    """Whatever a strategy needs at decision time beyond the question itself.

    Kept open by design — strategies pull what they need (LLM handle, RAG
    retriever, tool registry) from `extras` and ignore the rest.
    """

    competition_id: int
    level: int
    extras: dict[str, Any] = field(default_factory=dict)


class Strategy(Protocol):
    """Callable that decides which option to answer."""

    def __call__(self, question: Question, ctx: Context) -> AnswerDecision: ...
