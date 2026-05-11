"""Zero-shot baseline: hand the question + options to a single LLM, parse choice."""

from __future__ import annotations

import time

from polimillionaire._vendor.millionaire_client.models import Question
from polimillionaire.llm import LLM
from polimillionaire.prompts import zero_shot as prompt
from polimillionaire.strategies._common import build_decision, make_schema
from polimillionaire.strategies.base import AnswerDecision, Context


class ZeroShotStrategy:
    """Single-LLM, no examples, no retrieval.

    One `complete_json` call per question. The schema is built per-question
    so `answer_id` is constrained to the actual option ids. `rationale` is
    listed first in the schema so the model produces its reasoning before
    committing to the answer.
    """

    strategy_name = "zero_shot"
    prompt_version = prompt.PROMPT_VERSION

    def __init__(self, llm: LLM) -> None:
        self._llm = llm

    @property
    def model_name(self) -> str:
        return self._llm.name

    def __call__(self, question: Question, ctx: Context) -> AnswerDecision:  # noqa: ARG002
        messages = prompt.render(question)
        schema = make_schema(question)
        start = time.perf_counter()
        out = self._llm.complete_json(messages, schema)
        return build_decision(
            out,
            start,
            model_name=self.model_name,
            strategy_name=self.strategy_name,
            prompt_version=self.prompt_version,
        )
