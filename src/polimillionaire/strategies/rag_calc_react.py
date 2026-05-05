"""Calc-react with retrieved MATH problems as reference solutions.

Same bounded ReAct loop as `CalcReactStrategy` (one calculate-or-answer
JSON action per step, force-answer at `max_steps`). Difference is the
prompt: before the loop starts, the strategy retrieves the top-k
similar problems from a MATH index and prepends them to the system
message as natural-language reference solutions. The hand-crafted ReAct
exemplars stay -- those teach the action format; the retrieved ones
teach the math pattern.

If retrieval fails (no index, embedder load error, FAISS hiccup) the
strategy degrades silently to plain calc-react: the prompt is rendered
with an empty reference list, the loop runs as before, the game keeps
playing. Retrieval failures are logged when `verbose=True` for debug.
"""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING

from polimillionaire._vendor.millionaire_client.models import Question
from polimillionaire.llm import LLM, Message
from polimillionaire.prompts import rag_calc_react as prompt
from polimillionaire.strategies._common import make_action_schema, make_schema
from polimillionaire.strategies.base import AnswerDecision, Context
from polimillionaire.tools import calc

if TYPE_CHECKING:
    # Importing the retriever pulls numpy + faiss + sentence-transformers,
    # which live behind the optional `[rag]` group. Deferred so this
    # strategy module can be imported (and re-exported from the package
    # __init__) on a base install -- only instantiation needs `[rag]`.
    from polimillionaire.retrieval.retriever import Retriever


class RagCalcReactStrategy:
    """Calc-react augmented with retrieved math reference solutions."""

    strategy_name = "rag_calc_react"
    prompt_version = prompt.PROMPT_VERSION

    def __init__(
        self,
        llm: LLM,
        retriever: Retriever,
        *,
        k: int = 3,
        max_steps: int = 3,
        verbose: bool = False,
    ) -> None:
        if max_steps < 1:
            raise ValueError(f"max_steps must be >= 1, got {max_steps}")
        if k < 0:
            raise ValueError(f"k must be >= 0, got {k}")
        self._llm = llm
        self._retriever = retriever
        self._k = k
        self._max_steps = max_steps
        self._verbose = verbose

    @property
    def model_name(self) -> str:
        return self._llm.name

    def __call__(self, question: Question, ctx: Context) -> AnswerDecision:  # noqa: ARG002
        start = time.perf_counter()

        try:
            references = self._retriever.search(question.text, k=self._k) if self._k else []
        except Exception as e:  # noqa: BLE001 -- never block answering on retrieval
            if self._verbose:
                print(f"   [rag-calc-react] retrieval failed ({type(e).__name__}): {e}")
            references = []

        if self._verbose and references:
            print(
                "   [rag-calc-react] retrieved "
                + ", ".join(f"{r.id} ({r.score:.2f})" for r in references)
            )

        messages: list[Message] = list(prompt.render(question, references))
        action_schema = make_action_schema(question)

        for _ in range(self._max_steps):
            try:
                out = self._llm.complete_json(messages, action_schema)
            except ValueError:
                if self._verbose:
                    print("   [rag-calc-react] action step failed to parse — forcing answer")
                break
            if out["action"] == "answer":
                return self._decision(out, start)
            expression = out["expression"]
            result = calc(expression)
            if self._verbose:
                print(f'   [rag-calc-react] calc("{expression}") -> {result}')
            messages.append({"role": "assistant", "content": json.dumps(out)})
            messages.append({"role": "user", "content": f"Calculator: `{expression}` = {result}"})

        # Step cap or parse failure -- force a commit on the answer-only schema.
        messages.append(
            {"role": "user", "content": "Step limit reached. Answer now using the answer schema."}
        )
        try:
            out = self._llm.complete_json(messages, make_schema(question))
        except ValueError:
            if self._verbose:
                print("   [rag-calc-react] forced answer also failed — defaulting to option 0")
            return self._decision(
                {
                    "action": "answer",
                    "rationale": "Model output failed to parse; defaulting to first option.",
                    "answer_id": question.options[0].id,
                    "confidence": 0.0,
                },
                start,
            )
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
