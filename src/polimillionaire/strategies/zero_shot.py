"""Zero-shot baseline: hand the question + options to a single LLM, parse choice."""

from __future__ import annotations

from polimillionaire._vendor.millionaire_client.models import Question
from polimillionaire.llm import LLM
from polimillionaire.strategies.base import AnswerDecision, Context


class ZeroShotStrategy:
    """Single-LLM, no examples, no retrieval."""

    strategy_name = "zero_shot"
    prompt_version = "v1"

    def __init__(self, llm: LLM, model_name: str) -> None:
        self._llm = llm
        self.model_name = model_name

    def __call__(self, question: Question, ctx: Context) -> AnswerDecision:
        # TODO: render prompt from prompts/zero_shot.txt, call self._llm,
        # parse the chosen option id, return AnswerDecision with timing.
        raise NotImplementedError("Wire up zero-shot when prompts/zero_shot.txt exists.")
