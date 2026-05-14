"""Unit tests for WikiRagStrategy.

All retrieval components are replaced with lightweight fakes so the test
suite runs without sentence-transformers, FAISS, or rank-bm25.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

from polimillionaire._vendor.millionaire_client.models import Option, Question
from polimillionaire.llm import LLM
from polimillionaire.retrieval.retriever import Passage
from polimillionaire.strategies.base import Context
from polimillionaire.strategies.wiki_rag import WikiRagStrategy


class _FakeLLM:
    name = "fake-model"

    def __init__(self, response: dict[str, Any] | ValueError) -> None:
        self._response = response
        self.calls: list[tuple[list[dict], dict]] = []

    def complete_json(
        self, messages: list[dict[str, str]], schema: dict[str, Any], **_: Any
    ) -> dict[str, Any]:
        self.calls.append(([dict(m) for m in messages], schema))
        if isinstance(self._response, ValueError):
            raise self._response
        return self._response


@dataclass
class _FakeRetriever:
    passages: list[Passage] = field(default_factory=list)
    queries: list[str] = field(default_factory=list)
    fail: bool = False

    def search(self, query: str, k: int) -> list[Passage]:  # noqa: ARG002
        self.queries.append(query)
        if self.fail:
            raise RuntimeError("index missing")
        return self.passages[:k]


@dataclass
class _FakeBM25:
    passages: list[Passage] = field(default_factory=list)

    def search(self, query: str, k: int) -> list[Passage]:  # noqa: ARG002
        return self.passages[:k]


@dataclass
class _FakeReranker:
    last_top_k: int | None = None
    # Score assigned to every returned passage; mirrors production
    # Reranker.rerank() which overwrites Passage.score with its logit.
    # 1.0 is above WikiRagStrategy.min_rerank_score so tests still see the
    # passages in the prompt unless they set the field explicitly.
    score: float = 1.0

    def rerank(
        self, query: str, passages: list[Passage], *, top_k: int | None = None
    ) -> list[Passage]:  # noqa: ARG002
        self.last_top_k = top_k
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


def _make_strategy(llm, retriever=None, bm25=None, reranker=None, **kwargs) -> WikiRagStrategy:
    return WikiRagStrategy(
        cast(LLM, llm),
        retriever or _FakeRetriever(passages=_PASSAGES),
        bm25 or _FakeBM25(passages=_PASSAGES),
        reranker or _FakeReranker(),
        **kwargs,
    )


def _ctx() -> Context:
    return Context(competition_id=0, level=1)


def test_strategy_returns_answer_decision_with_correct_option() -> None:
    fake = _FakeLLM({"rationale": "Paris is correct.", "confidence": 0.9, "answer_id": 3})
    strategy = _make_strategy(fake)
    decision = strategy(_make_question(), _ctx())

    assert decision.option_id == 3
    assert decision.confidence == 0.9
    assert decision.rationale == "Paris is correct."
    assert decision.model_name == "fake-model"
    assert decision.strategy_name == "wiki_rag"
    assert decision.prompt_version == "wiki_rag/v1"
    assert decision.latency_ms >= 0


def test_strategy_name_and_prompt_version() -> None:
    assert WikiRagStrategy.strategy_name == "wiki_rag"
    strategy = _make_strategy(_FakeLLM({"rationale": "x", "confidence": 0.5, "answer_id": 1}))
    assert strategy.prompt_version == "wiki_rag/v1"


def test_retrieval_failure_degrades_gracefully() -> None:
    """When the dense retriever raises, the strategy still answers from the LLM."""
    fake = _FakeLLM({"rationale": "best guess", "confidence": 0.5, "answer_id": 1})
    failing_retriever = _FakeRetriever(passages=_PASSAGES, fail=True)
    strategy = _make_strategy(fake, retriever=failing_retriever, verbose=False)
    decision = strategy(_make_question(), _ctx())

    assert decision.option_id == 1
    assert decision.strategy_name == "wiki_rag"
    # LLM was still called once (with empty passages)
    assert len(fake.calls) == 1
    # no passages in the user message because retrieval failed
    user_content = fake.calls[0][0][1]["content"]
    assert "Wikipedia excerpts" not in user_content


def test_llm_parse_failure_defaults_to_first_option() -> None:
    fake = _FakeLLM(ValueError("grammar-constrained output did not parse: ..."))
    strategy = _make_strategy(fake, verbose=False)
    question = _make_question()
    decision = strategy(question, _ctx())

    assert decision.option_id == question.options[0].id
    assert decision.confidence == 0.0
    assert decision.rationale is not None
    assert "fail" in decision.rationale.lower()


def test_retrieval_query_contains_question_and_options() -> None:
    """Query must include the question text and all option texts."""
    fake = _FakeLLM({"rationale": "x", "confidence": 0.5, "answer_id": 1})
    retriever = _FakeRetriever(passages=_PASSAGES)
    strategy = _make_strategy(fake, retriever=retriever)
    question = _make_question()
    strategy(question, _ctx())

    assert len(retriever.queries) == 1
    query = retriever.queries[0]
    assert question.text in query
    for opt in question.options:
        assert opt.text in query


def test_passages_appear_in_prompt_when_retrieval_succeeds() -> None:
    fake = _FakeLLM({"rationale": "found it", "confidence": 0.8, "answer_id": 3})
    strategy = _make_strategy(fake, top_k=2)
    strategy(_make_question(), _ctx())

    user_content = fake.calls[0][0][1]["content"]
    assert "Wikipedia excerpts" in user_content
    # at least the first passage title/text should appear
    assert "France" in user_content


def test_empty_passage_list_when_no_hits() -> None:
    """When retriever returns empty lists, no passage block is added."""
    fake = _FakeLLM({"rationale": "no context", "confidence": 0.4, "answer_id": 2})
    strategy = _make_strategy(
        fake,
        retriever=_FakeRetriever(passages=[]),
        bm25=_FakeBM25(passages=[]),
    )
    strategy(_make_question(), _ctx())

    user_content = fake.calls[0][0][1]["content"]
    assert "Wikipedia excerpts" not in user_content
    assert "Q: What is the capital" in user_content


def test_low_rerank_scores_are_dropped_from_prompt() -> None:
    """When the reranker returns scores below min_rerank_score, the passages
    must not appear in the prompt — the LLM should fall back to its
    parametric knowledge instead of being misled by off-topic context.
    """
    fake = _FakeLLM({"rationale": "no context", "confidence": 0.4, "answer_id": 3})
    # Score 0.05 < default min_rerank_score 0.15 → all passages filtered out.
    low_score_reranker = _FakeReranker(score=0.05)
    strategy = _make_strategy(fake, reranker=low_score_reranker)
    strategy(_make_question(), _ctx())

    user_content = fake.calls[0][0][1]["content"]
    assert "Wikipedia excerpts" not in user_content
    assert "Q: What is the capital" in user_content


def test_high_rerank_scores_pass_the_floor() -> None:
    """Passages above min_rerank_score are kept and appear in the prompt."""
    fake = _FakeLLM({"rationale": "got it", "confidence": 0.9, "answer_id": 3})
    high_score_reranker = _FakeReranker(score=0.8)
    strategy = _make_strategy(fake, reranker=high_score_reranker, min_rerank_score=0.5)
    strategy(_make_question(), _ctx())

    user_content = fake.calls[0][0][1]["content"]
    assert "Wikipedia excerpts" in user_content


def test_min_rerank_score_zero_disables_the_floor() -> None:
    """min_rerank_score=0 must keep every reranked passage even at tiny scores."""
    fake = _FakeLLM({"rationale": "got it", "confidence": 0.9, "answer_id": 3})
    low_score_reranker = _FakeReranker(score=0.001)
    strategy = _make_strategy(fake, reranker=low_score_reranker, min_rerank_score=0.0)
    strategy(_make_question(), _ctx())

    user_content = fake.calls[0][0][1]["content"]
    assert "Wikipedia excerpts" in user_content


def test_reranker_receives_configured_top_k() -> None:
    fake = _FakeLLM({"rationale": "r", "confidence": 0.5, "answer_id": 1})
    reranker = _FakeReranker()
    strategy = _make_strategy(fake, reranker=reranker, top_k=3)
    strategy(_make_question(), _ctx())

    assert reranker.last_top_k == 3


class _FakeLive:
    """Stand-in for LiveWikiRetriever -- same `search()` shape, no HTTP."""

    def __init__(self, passages: list[Passage]) -> None:
        self.passages = passages
        self.queries: list[str] = []

    def search(self, query: str, k: int | None = None) -> list[Passage]:
        self.queries.append(query)
        return self.passages if k is None else self.passages[:k]


def test_live_passages_join_rerank_pool() -> None:
    """When `live` is set, its hits are passed to the reranker alongside
    the static fused pool. The combined pool's size must reflect both."""
    fake = _FakeLLM({"rationale": "r", "confidence": 0.5, "answer_id": 1})

    class _CountingReranker(_FakeReranker):
        seen: list[Passage] = []

        def rerank(self, query: str, passages: list[Passage], *, top_k=None):  # noqa: ARG002
            self.seen = list(passages)
            return super().rerank(query, passages, top_k=top_k)

    live_hit = Passage(
        id="live/Inception",
        text="2010 Nolan film about dreams.",
        metadata={"source": "live_wiki", "title": "Inception"},
        score=1.0,
    )
    counting = _CountingReranker()
    strategy = _make_strategy(fake, reranker=counting, live=_FakeLive([live_hit]), live_k=1)
    strategy(_make_question(), _ctx())

    # static (_PASSAGES has 2) + live (1) = 3 passages reach the reranker
    pool_ids = {p.id for p in counting.seen}
    assert "live/Inception" in pool_ids
    assert "p1" in pool_ids and "p2" in pool_ids


def test_live_dedup_by_title_against_static_pool() -> None:
    """If a live hit's title is already in the static fused list, drop it
    so the reranker isn't reading the same article twice."""
    fake = _FakeLLM({"rationale": "r", "confidence": 0.5, "answer_id": 1})

    class _CountingReranker(_FakeReranker):
        seen: list[Passage] = []

        def rerank(self, query: str, passages: list[Passage], *, top_k=None):  # noqa: ARG002
            self.seen = list(passages)
            return super().rerank(query, passages, top_k=top_k)

    # Static `_PASSAGES[0]` has metadata title "France"; the live hit
    # shares that title with different casing -- dedup must be case-insensitive.
    dup = Passage(
        id="live/France",
        text="France is a country.",
        metadata={"source": "live_wiki", "title": "france"},
        score=1.0,
    )
    counting = _CountingReranker()
    strategy = _make_strategy(fake, reranker=counting, live=_FakeLive([dup]), live_k=1)
    strategy(_make_question(), _ctx())

    pool_ids = {p.id for p in counting.seen}
    assert "live/France" not in pool_ids


def test_live_disabled_when_no_retriever_supplied() -> None:
    """The strategy without a `live` retriever must behave exactly as before."""
    fake = _FakeLLM({"rationale": "r", "confidence": 0.5, "answer_id": 1})

    class _CountingReranker(_FakeReranker):
        seen: list[Passage] = []

        def rerank(self, query: str, passages: list[Passage], *, top_k=None):  # noqa: ARG002
            self.seen = list(passages)
            return super().rerank(query, passages, top_k=top_k)

    counting = _CountingReranker()
    strategy = _make_strategy(fake, reranker=counting)
    strategy(_make_question(), _ctx())
    assert all(p.id in {"p1", "p2"} for p in counting.seen)


def test_schema_constrains_answer_id_to_question_options() -> None:
    fake = _FakeLLM({"rationale": "r", "confidence": 0.5, "answer_id": 1})
    strategy = _make_strategy(fake)
    strategy(_make_question(), _ctx())

    schema = fake.calls[0][1]
    assert schema["properties"]["answer_id"]["enum"] == [1, 2, 3, 4]
