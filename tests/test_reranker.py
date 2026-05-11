"""Tests for the cross-encoder reranker.

Monkeypatches `Reranker._ensure_loaded` to avoid downloading any model weights.
The fake CrossEncoder assigns a score equal to the number of query tokens
found in the passage text, so we can predict which passage wins.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("sentence_transformers")

from polimillionaire.retrieval.reranker import Reranker  # noqa: E402
from polimillionaire.retrieval.retriever import Passage  # noqa: E402


def _make_passage(pid: str, text: str, score: float = 0.0) -> Passage:
    return Passage(id=pid, text=text, metadata={}, score=score)


class _FakeCrossEncoder:
    """predict() returns a score = number of query words found in each passage."""

    def predict(self, pairs: list[list[str]]) -> np.ndarray:
        scores = []
        for query, text in pairs:
            tokens = set(query.lower().split())
            hit_count = sum(1 for t in tokens if t in text.lower())
            scores.append(float(hit_count))
        return np.array(scores, dtype=np.float32)


def _patch_reranker(monkeypatch: pytest.MonkeyPatch, reranker: Reranker) -> None:
    """Install the fake model without loading anything from disk."""

    def _fake_ensure_loaded() -> None:
        reranker._model = _FakeCrossEncoder()  # type: ignore[assignment]

    monkeypatch.setattr(reranker, "_ensure_loaded", _fake_ensure_loaded)


def test_rerank_returns_sorted_by_score_desc(monkeypatch: pytest.MonkeyPatch) -> None:
    reranker = Reranker()
    _patch_reranker(monkeypatch, reranker)

    # token-hit counts: p1=3 ("capital", "france", "paris"), p2=2 ("france", "paris"),
    # p3=1 ("paris"). distinct so the ordering is deterministic.
    passages = [
        _make_passage("p1", "Paris is the capital of France."),
        _make_passage("p2", "France borders Paris on every side."),
        _make_passage("p3", "Paris Hilton is unrelated to this question."),
    ]
    results = reranker.rerank("capital France Paris", passages)

    assert [p.id for p in results] == ["p1", "p2", "p3"]
    assert results[0].score > results[1].score > results[2].score


def test_rerank_top_k_truncates(monkeypatch: pytest.MonkeyPatch) -> None:
    reranker = Reranker()
    _patch_reranker(monkeypatch, reranker)

    passages = [_make_passage(f"p{i}", f"word{i}") for i in range(5)]
    results = reranker.rerank("some query", passages, top_k=2)
    assert len(results) == 2


def test_rerank_empty_passages_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    reranker = Reranker()
    _patch_reranker(monkeypatch, reranker)

    assert reranker.rerank("anything", []) == []


def test_rerank_score_field_replaced(monkeypatch: pytest.MonkeyPatch) -> None:
    reranker = Reranker()
    _patch_reranker(monkeypatch, reranker)

    # original score is a sentinel value that shouldn't survive reranking
    passage = _make_passage("p1", "hello world", score=999.0)
    results = reranker.rerank("hello", [passage])

    assert len(results) == 1
    assert results[0].score != 999.0


def test_rerank_low_score_passage_rises_after_reranking(monkeypatch: pytest.MonkeyPatch) -> None:
    reranker = Reranker()
    _patch_reranker(monkeypatch, reranker)

    # p_relevant has a low retrieval score but its text matches the query well
    # p_irrelevant has a high retrieval score but no query tokens
    p_relevant = _make_passage("relevant", "capital France Paris city", score=0.1)
    p_irrelevant = _make_passage("irrelevant", "mitochondria ATP biology", score=0.9)

    results = reranker.rerank("capital France Paris", [p_irrelevant, p_relevant])

    # the reranker should promote p_relevant to the top
    assert results[0].id == "relevant"


def test_rerank_single_passage_still_calls_model(monkeypatch: pytest.MonkeyPatch) -> None:
    reranker = Reranker()
    called: list[bool] = []

    original_fake = _FakeCrossEncoder()

    class _InstrumentedCrossEncoder:
        def predict(self, pairs: list[list[str]]) -> np.ndarray:
            called.append(True)
            return original_fake.predict(pairs)

    def _fake_ensure_loaded() -> None:
        reranker._model = _InstrumentedCrossEncoder()  # type: ignore[assignment]

    monkeypatch.setattr(reranker, "_ensure_loaded", _fake_ensure_loaded)

    results = reranker.rerank("hello", [_make_passage("p1", "hello world")])
    assert len(results) == 1
    assert called, "predict() was not called for a single-passage input"
