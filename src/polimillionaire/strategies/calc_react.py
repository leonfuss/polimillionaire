"""ReAct-style strategy with a calculator tool.

Each step the model emits either `{"action":"calculate", ...}` or
`{"action":"answer", ...}`. Calculate results are appended to the message
list and the loop continues. At `max_steps` the action schema is swapped for
the answer-only schema, forcing a commit.
"""

from __future__ import annotations

import json
import time

from polimillionaire._vendor.millionaire_client.models import Question
from polimillionaire.llm import LLM, Message
from polimillionaire.prompts import calc_react as prompt
from polimillionaire.strategies._common import make_action_schema, make_schema
from polimillionaire.strategies.base import AnswerDecision, Context
from polimillionaire.tools import calc


class CalcReactStrategy:
    """Single LLM, calculator tool, bounded ReAct loop.

    `max_steps` caps the total number of `complete_json` calls *before* the
    forced-answer fallback, so worst case is `max_steps + 1` calls per question.
    """

    strategy_name = "calc_react"
    prompt_version = prompt.PROMPT_VERSION

    def __init__(self, llm: LLM, *, max_steps: int = 3) -> None:
        if max_steps < 1:
            raise ValueError(f"max_steps must be >= 1, got {max_steps}")
        self._llm = llm
        self._max_steps = max_steps

    @property
    def model_name(self) -> str:
        return self._llm.name

    def __call__(self, question: Question, ctx: Context) -> AnswerDecision:  # noqa: ARG002
        messages: list[Message] = list(prompt.render(question))
        action_schema = make_action_schema(question)
        start = time.perf_counter()

        for _ in range(self._max_steps):
            out = self._llm.complete_json(messages, action_schema)
            if out["action"] == "answer":
                return self._decision(out, start)
            expression = out["expression"]
            result = calc(expression)
            messages.append({"role": "assistant", "content": json.dumps(out)})
            messages.append({"role": "user", "content": f"Calculator: `{expression}` = {result}"})

        # Step cap hit — force a commit by swapping in the answer-only schema.
        messages.append(
            {"role": "user", "content": "Step limit reached. Answer now using the answer schema."}
        )
        out = self._llm.complete_json(messages, make_schema(question))
        return self._decision({"action": "answer", **out}, start)

    def _decision(self, out: dict, start: float) -> AnswerDecision:
        latency_ms = int((time.perf_counter() - start) * 1000)
        return AnswerDecision(
            option_id=int(out["answer_id"]),
            confidence=float(out["confidence"]),
            rationale=out.get("rationale"),
            model_name=self.model_name,
            strategy_name=self.strategy_name,
            prompt_version=self.prompt_version,
            latency_ms=latency_ms,
        )
