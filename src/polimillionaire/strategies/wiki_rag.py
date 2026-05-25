"""Wikipedia-RAG strategy for non-math competitions.

Hybrid retrieval (dense + BM25) fused via RRF, reranked by a cross-encoder,
then a single zero-shot JSON completion. No ReAct loop or calculator.

When `retriever` and `bm25` are None and `live` is provided, runs in
live-only mode: per-question MediaWiki lookup as the sole candidate
source. Used for competitions with no pre-built static index.

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
    from polimillionaire.retrieval.live_wiki import LiveWikiRetriever
    from polimillionaire.retrieval.reranker import Reranker
    from polimillionaire.retrieval.retriever import Retriever


class WikiRagStrategy:
    """Retrieve Wikipedia passages, rerank, then answer in one shot."""

    strategy_name = "wiki_rag"

    def __init__(
        self,
        llm: LLM,
        retriever: Retriever | None,
        bm25: BM25Index | None,
        reranker: Reranker | None,
        *,
        live: LiveWikiRetriever | None = None,
        live_k: int = 4,
        dense_k: int = 50,
        sparse_k: int = 50,
        fused_k: int = 25,
        top_k: int = 5,
        nprobe: int | None = None,
        min_rerank_score: float = 0.15,
        use_dense: bool = True,
        use_sparse: bool = True,
        use_reranker: bool = True,
        include_rationale: bool = True,
        prompt_version: str | None = None,
        verbose: bool = False,
    ) -> None:
        # Three candidate sources: dense + sparse from a static index, plus
        # live MediaWiki lookup. Need at least one. Live-only mode (cids
        # without a pre-built index) flips both static toggles off and
        # relies on `live`.
        has_static = use_dense or use_sparse
        has_live = live is not None and live_k > 0
        if not (has_static or has_live):
            raise ValueError(
                "need at least one source: use_dense, use_sparse, or live (with live_k>0)"
            )
        if use_dense and retriever is None:
            raise ValueError("use_dense=True requires a retriever")
        if use_sparse and bm25 is None:
            raise ValueError("use_sparse=True requires a bm25 index")
        if use_reranker and reranker is None:
            raise ValueError("use_reranker=True requires a reranker")
        if nprobe is not None and retriever is not None:
            retriever.set_nprobe(nprobe)
        # When the caller doesn't override the prompt, pick a variant whose
        # system message matches the schema: no rationale -> noreason prompt.
        if prompt_version is None:
            prompt_version = prompt.LATEST if include_rationale else prompt.NOREASON
        if prompt_version not in prompt.PROMPTS:
            raise ValueError(
                f"unknown prompt version {prompt_version!r}; available: {sorted(prompt.PROMPTS)}"
            )
        self._llm = llm
        self._retriever = retriever
        self._bm25 = bm25
        self._reranker = reranker
        self._live = live
        self._live_k = live_k
        self._dense_k = dense_k
        self._sparse_k = sparse_k
        self._fused_k = fused_k
        self._top_k = top_k
        self._use_dense = use_dense
        self._use_sparse = use_sparse
        self._use_reranker = use_reranker
        self._min_rerank_score = min_rerank_score
        self._include_rationale = include_rationale
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

            # Live lookup, when enabled, augments the rerank pool with
            # per-question hits. Static index covers high-traffic topics
            # but is frozen at crawl time; live lookup picks up the long
            # tail (recent films, obscure scientists, breaking news).
            # Dedup by article title against the static fused list so
            # the reranker doesn't see the same article twice.
            #
            # We pass option_texts so news-flavoured retrievers (GDELT)
            # can OR-anchor on entity names. MediaWiki ignores them --
            # piping options into keyword search adds distractors and
            # nulls out the result set.
            live_passages: list = []
            if self._live is not None and self._live_k > 0:
                live_passages = self._live.search(
                    question.text,
                    k=self._live_k,
                    option_texts=[o.text for o in question.options if o.text],
                )
                if live_passages:
                    static_titles = {
                        p.metadata.get("title", "").lower()
                        for p in fused
                        if p.metadata.get("title")
                    }
                    before = len(live_passages)
                    live_passages = [
                        p
                        for p in live_passages
                        if p.metadata.get("title", "").lower() not in static_titles
                    ]
                    if self._verbose and len(live_passages) < before:
                        print(
                            f"   [wiki_rag] live: dropped {before - len(live_passages)} "
                            "title(s) already in static pool"
                        )

            pool = fused + live_passages
            if self._verbose:
                print(
                    f"   [wiki_rag] pool: {len(fused)} static + {len(live_passages)} live "
                    f"= {len(pool)} candidates"
                )

            if self._use_reranker:
                top_passages = self._reranker.rerank(query, pool, top_k=self._top_k)
            else:
                top_passages = pool[: self._top_k]
        except Exception as e:  # noqa: BLE001 -- never block answering on retrieval
            if self._verbose:
                print(
                    f"   [wiki_rag] retrieval failed ({type(e).__name__}: {e}); "
                    "answering without context"
                )
            top_passages = []

        # Conditional RAG: when the reranker's best score is below the floor,
        # the retrieved passages are off-topic and tend to mislead the LLM
        # rather than help it. Drop everything below the threshold and let
        # the model fall back to its parametric knowledge -- the prompt
        # renders the same way as the retrieval-failed path above.
        if self._use_reranker and self._min_rerank_score > 0 and top_passages:
            kept = [p for p in top_passages if p.score >= self._min_rerank_score]
            if self._verbose and len(kept) != len(top_passages):
                dropped = len(top_passages) - len(kept)
                print(
                    f"   [wiki_rag] dropped {dropped} passage(s) below "
                    f"rerank floor {self._min_rerank_score:.2f}"
                )
            top_passages = kept

        if self._verbose and top_passages:
            n_live = sum(1 for p in top_passages if p.metadata.get("source") == "live_wiki")
            tag = f" ({n_live} live)" if n_live else ""
            print(
                f"   [wiki_rag] retrieved{tag} "
                + ", ".join(f"{p.id} ({p.score:.2f})" for p in top_passages)
            )
        elif self._verbose:
            print("   [wiki_rag] no passages above threshold; answering from parametric knowledge")

        messages = self._variant.render(question, top_passages)
        schema = make_schema(question, include_rationale=self._include_rationale)

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
