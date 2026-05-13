"""Ablation-flag tests for WikiRagStrategy.

Verifies that use_dense, use_sparse, and use_reranker flags correctly
gate the underlying retrieval calls without changing the public interface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

import pytest
from test_wiki_rag import _FakeLLM, _FakeReranker

from polimillionaire._vendor.millionaire_client.models import Option, Question
from polimillionaire.llm import LLM
from polimillionaire.retrieval.retriever import Passage
from polimillionaire.strategies.base import Context
from polimillionaire.strategies.wiki_rag import WikiRagStrategy


@dataclass
class _TrackingBM25:
    """BM25 fake that records whether it was called."""

    passages: list[Passage] = field(default_factory=list)
    called: bool = False

    def search(self, query: str, k: int) -> list[Passage]:  # noqa: ARG002
        self.called = True
        return self.passages[:k]


@dataclass
class _TrackingRetriever:
    """Dense retriever fake that records whether it was called."""

    passages: list[Passage] = field(default_factory=list)
    called: bool = False

    def search(self, query: str, k: int) -> list[Passage]:  # noqa: ARG002
        self.called = True
        return self.passages[:k]


@dataclass
class _TrackingReranker:
    called: bool = False
    # Mirrors production reranker: returned passages get a fresh score.
    # 1.0 keeps tests above WikiRagStrategy.min_rerank_score's default.
    score: float = 1.0

    def rerank(
        self, query: str, passages: list[Passage], *, top_k: int | None = None
    ) -> list[Passage]:  # noqa: ARG002
        self.called = True
        kept = passages[:top_k] if top_k is not None else passages
        return [Passage(id=p.id, text=p.text, metadata=p.metadata, score=self.score) for p in kept]


_PASSAGES = [
    Passage(
        id="p1", text="Paris is the capital of France.", metadata={"title": "France"}, score=0.9
    ),
    Passage(
        id="p2", text="Berlin is the capital of Germany.", metadata={"title": "Germany"}, score=0.8
    ),
]

_LLM_RESPONSE: dict[str, Any] = {"rationale": "x", "confidence": 0.5, "answer_id": 1}


def _make_question() -> Question:
    return Question(
        id=42,
        text="What is the capital of France?",
        options=[
            Option(id=1, text="Berlin"),
            Option(id=2, text="Madrid"),
            Option(id=3, text="Paris"),
            Option(id=4, text="Rome"),
        ],
        level=1,
    )


def _ctx() -> Context:
    return Context(competition_id=0, level=1)


def test_sparse_only_skips_dense_retriever() -> None:
    """use_dense=False must not call the dense retriever."""
    retriever = _TrackingRetriever(passages=_PASSAGES)
    bm25 = _TrackingBM25(passages=_PASSAGES)
    strategy = WikiRagStrategy(
        cast(LLM, _FakeLLM(_LLM_RESPONSE)),
        retriever,  # type: ignore[arg-type]
        bm25,  # type: ignore[arg-type]
        _FakeReranker(),  # type: ignore[arg-type]
        use_dense=False,
        use_sparse=True,
    )
    strategy(_make_question(), _ctx())

    assert not retriever.called
    assert bm25.called


def test_dense_only_skips_bm25() -> None:
    """use_sparse=False must not call the BM25 index."""
    retriever = _TrackingRetriever(passages=_PASSAGES)
    bm25 = _TrackingBM25(passages=_PASSAGES)
    strategy = WikiRagStrategy(
        cast(LLM, _FakeLLM(_LLM_RESPONSE)),
        retriever,  # type: ignore[arg-type]
        bm25,  # type: ignore[arg-type]
        _FakeReranker(),  # type: ignore[arg-type]
        use_dense=True,
        use_sparse=False,
    )
    strategy(_make_question(), _ctx())

    assert retriever.called
    assert not bm25.called


def test_no_reranker_skips_rerank_call() -> None:
    """use_reranker=False must not call reranker.rerank."""
    reranker = _TrackingReranker()
    strategy = WikiRagStrategy(
        cast(LLM, _FakeLLM(_LLM_RESPONSE)),
        _TrackingRetriever(passages=_PASSAGES),  # type: ignore[arg-type]
        _TrackingBM25(passages=_PASSAGES),  # type: ignore[arg-type]
        reranker,  # type: ignore[arg-type]
        use_reranker=False,
        top_k=1,
    )
    strategy(_make_question(), _ctx())

    assert not reranker.called


def test_no_reranker_slices_fused_to_top_k() -> None:
    """Without reranker the strategy passes at most top_k passages to the LLM."""
    llm = _FakeLLM(_LLM_RESPONSE)
    strategy = WikiRagStrategy(
        cast(LLM, llm),
        _TrackingRetriever(passages=_PASSAGES),  # type: ignore[arg-type]
        _TrackingBM25(passages=_PASSAGES),  # type: ignore[arg-type]
        _TrackingReranker(),  # type: ignore[arg-type]
        use_reranker=False,
        top_k=1,
    )
    strategy(_make_question(), _ctx())

    # the prompt should contain p1 but not p2 (only 1 passage allowed)
    user_content = llm.calls[0][0][1]["content"]
    assert "France" in user_content
    assert "Germany" not in user_content


def test_both_disabled_raises_at_construction() -> None:
    """Constructing with use_dense=False and use_sparse=False must raise ValueError."""
    with pytest.raises(ValueError, match="at least one of use_dense, use_sparse"):
        WikiRagStrategy(
            cast(LLM, _FakeLLM(_LLM_RESPONSE)),
            _TrackingRetriever(),  # type: ignore[arg-type]
            _TrackingBM25(),  # type: ignore[arg-type]
            _TrackingReranker(),  # type: ignore[arg-type]
            use_dense=False,
            use_sparse=False,
        )
