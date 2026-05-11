"""Wikipedia-RAG strategy for non-math competitions.

Hybrid retrieval (dense + BM25) fused via RRF, reranked by a cross-encoder,
then a single zero-shot JSON completion. No ReAct loop or calculator.

If any retrieval step raises (missing index, embedder error, etc.) the
strategy degrades to a bare-LLM answer: the prompt is rendered with an
empty passage list and the model answers from its own knowledge.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from polimillionaire._vendor.millionaire_client.models import Question
from polimillionaire.llm import LLM
from polimillionaire.prompts import wiki_rag as prompt
from polimillionaire.retrieval.fusion import reciprocal_rank_fusion
from polimillionaire.strategies._common import build_decision, make_schema
from polimillionaire.strategies.base import AnswerDecision, Context

if TYPE_CHECKING:
    # Heavy deps live behind the optional [rag] group; deferred so this module
    # can be imported on a base install -- only instantiation needs [rag].
    from polimillionaire.retrieval.bm25 import BM25Index
    from polimillionaire.retrieval.reranker import Reranker
    from polimillionaire.retrieval.retriever import Retriever


class WikiRagStrategy:
    """Retrieve Wikipedia passages, rerank, then answer in one shot."""

    strategy_name = "wiki_rag"

    def __init__(
        self,
        llm: LLM,
        retriever: Retriever,
        bm25: BM25Index,
        reranker: Reranker,
        *,
        dense_k: int = 50,
        sparse_k: int = 50,
        fused_k: int = 25,
        top_k: int = 5,
        verbose: bool = False,
        prompt_version: str = prompt.LATEST,
    ) -> None:
        if prompt_version not in prompt.PROMPTS:
            raise ValueError(
                f"unknown prompt version {prompt_version!r}; "
                f"available: {sorted(prompt.PROMPTS)}"
            )
        self._llm = llm
        self._retriever = retriever
        self._bm25 = bm25
        self._reranker = reranker
        self._dense_k = dense_k
        self._sparse_k = sparse_k
        self._fused_k = fused_k
        self._top_k = top_k
        self._verbose = verbose
        self._variant = prompt.PROMPTS[prompt_version]

    @property
    def model_name(self) -> str:
        return self._llm.name

    @property
    def prompt_version(self) -> str:
        return self._variant.version

    def __call__(self, question: Question, ctx: Context) -> AnswerDecision:  # noqa: ARG002
        # short trivia questions need the options as anchor tokens in the query;
        # the question text alone often lacks enough signal for entity matching.
        query = question.text + " | " + " | ".join(o.text for o in question.options)

        try:
            dense_hits = self._retriever.search(query, k=self._dense_k)
            sparse_hits = self._bm25.search(query, k=self._sparse_k)
            fused = reciprocal_rank_fusion([dense_hits, sparse_hits], top_n=self._fused_k)
            top_passages = self._reranker.rerank(query, fused, top_k=self._top_k)
        except Exception as e:  # noqa: BLE001 -- never block answering on retrieval
            if self._verbose:
                print(
                    f"   [wiki_rag] retrieval failed ({type(e).__name__}: {e}); "
                    "answering without context"
                )
            top_passages = []

        if self._verbose and top_passages:
            print(
                "   [wiki_rag] retrieved "
                + ", ".join(f"{p.id} ({p.score:.2f})" for p in top_passages)
            )

        # latency_ms reflects LLM time only (consistent with calc_react and
        # rag_calc_react); retrieval cost is observable as the wall-clock gap.
        start = time.perf_counter()
        messages = self._variant.render(question, top_passages)
        schema = make_schema(question)

        try:
            out = self._llm.complete_json(messages, schema)
        except ValueError:
            if self._verbose:
                print("   [wiki_rag] answer JSON failed to parse — defaulting to option 0")
            out = {
                "rationale": "Model output failed to parse; defaulting to first option.",
                "answer_id": question.options[0].id,
                "confidence": 0.0,
            }
        return build_decision(
            out,
            start,
            model_name=self.model_name,
            strategy_name=self.strategy_name,
            prompt_version=self.prompt_version,
        )
