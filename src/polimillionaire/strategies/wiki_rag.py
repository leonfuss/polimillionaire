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
        use_dense: bool = True,
        use_sparse: bool = True,
        use_reranker: bool = True,
        prompt_version: str = prompt.LATEST,
        verbose: bool = False,
    ) -> None:
        if not (use_dense or use_sparse):
            raise ValueError("at least one of use_dense, use_sparse must be True")
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
        self._use_dense = use_dense
        self._use_sparse = use_sparse
        self._use_reranker = use_reranker
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
        # cost of answering (retrieval + LLM), matching the 30s server timer.
        start = time.perf_counter()

        # short trivia questions need the options as anchor tokens in the query;
        # the question text alone often lacks enough signal for entity matching.
        query = question.text + " | " + " | ".join(o.text for o in question.options)

        try:
            rankings = []
            if self._use_dense:
                rankings.append(self._retriever.search(query, k=self._dense_k))
            if self._use_sparse:
                rankings.append(self._bm25.search(query, k=self._sparse_k))
            # single-list rrf is a rank-ordered passthrough — still correct
            fused = reciprocal_rank_fusion(rankings, top_n=self._fused_k)
            if self._use_reranker:
                top_passages = self._reranker.rerank(query, fused, top_k=self._top_k)
            else:
                top_passages = fused[: self._top_k]
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
