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

import time
from typing import TYPE_CHECKING

from polimillionaire._vendor.millionaire_client.models import Question
from polimillionaire.llm import LLM
from polimillionaire.prompts import rag_calc_react as prompt
from polimillionaire.strategies._common import run_react_loop
from polimillionaire.strategies.base import AnswerDecision, Context

if TYPE_CHECKING:
    # Importing the retriever pulls numpy + faiss + sentence-transformers,
    # which live behind the optional `[rag]` group. Deferred so this
    # strategy module can be imported (and re-exported from the package
    # __init__) on a base install -- only instantiation needs `[rag]`.
    from polimillionaire.retrieval.retriever import Retriever


class RagCalcReactStrategy:
    """Calc-react augmented with retrieved math reference solutions."""

    strategy_name = "rag_calc_react"

    def __init__(
        self,
        llm: LLM,
        retriever: Retriever,
        *,
        k: int = 3,
        max_steps: int = 1,
        verbose: bool = False,
        prompt_version: str = prompt.LATEST,
    ) -> None:
        if max_steps < 1:
            raise ValueError(f"max_steps must be >= 1, got {max_steps}")
        if k < 0:
            raise ValueError(f"k must be >= 0, got {k}")
        if prompt_version not in prompt.PROMPTS:
            raise ValueError(
                f"unknown prompt version {prompt_version!r}; "
                f"available: {sorted(prompt.PROMPTS)}"
            )
        self._llm = llm
        self._retriever = retriever
        self._k = k
        self._max_steps = max_steps
        self._verbose = verbose
        self._variant = prompt.PROMPTS[prompt_version]

    @property
    def model_name(self) -> str:
        return self._llm.name

    @property
    def prompt_version(self) -> str:
        return self._variant.version

    def __call__(self, question: Question, ctx: Context) -> AnswerDecision:  # noqa: ARG002
        # start before retrieval so latency_ms reflects the full wall-clock
        # cost of answering, not just LLM time
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

        messages = list(self._variant.render(question, references))
        return run_react_loop(
            self._llm,
            messages,
            question,
            max_steps=self._max_steps,
            model_name=self.model_name,
            strategy_name=self.strategy_name,
            prompt_version=self.prompt_version,
            start=start,
            verbose=self._verbose,
            log_prefix="rag-calc-react",
        )
