"""Tests for reciprocal_rank_fusion."""

from __future__ import annotations

from polimillionaire.retrieval.fusion import reciprocal_rank_fusion
from polimillionaire.retrieval.retriever import Passage


def _p(id: str, score: float = 0.0) -> Passage:
    return Passage(id=id, text=f"text {id}", metadata={}, score=score)


def test_single_list_passes_through() -> None:
    ranking = [_p("a"), _p("b"), _p("c")]
    result = reciprocal_rank_fusion([ranking])
    assert [r.id for r in result] == ["a", "b", "c"]


def test_two_disjoint_lists_merge_all() -> None:
    r1 = [_p("a"), _p("b")]
    r2 = [_p("c"), _p("d")]
    result = reciprocal_rank_fusion([r1, r2])
    ids = {r.id for r in result}
    assert ids == {"a", "b", "c", "d"}


def test_overlapping_item_scores_higher() -> None:
    # "shared" appears in both lists; "unique" only in one
    r1 = [_p("shared"), _p("unique")]
    r2 = [_p("shared"), _p("other")]
    result = reciprocal_rank_fusion([r1, r2])
    shared_score = next(r.score for r in result if r.id == "shared")
    unique_score = next(r.score for r in result if r.id == "unique")
    assert shared_score > unique_score


def test_top_n_truncates() -> None:
    ranking = [_p("a"), _p("b"), _p("c"), _p("d")]
    result = reciprocal_rank_fusion([ranking], top_n=2)
    assert len(result) == 2


def test_top_n_none_returns_all() -> None:
    ranking = [_p("a"), _p("b"), _p("c")]
    result = reciprocal_rank_fusion([ranking], top_n=None)
    assert len(result) == 3


def test_dedupe_by_id_uses_first_seen_text() -> None:
    # "dup" appears in both lists with different text -- first-seen should win
    p_first = Passage(id="dup", text="first text", metadata={"src": 1}, score=0.9)
    p_second = Passage(id="dup", text="second text", metadata={"src": 2}, score=0.5)
    result = reciprocal_rank_fusion([[p_first], [p_second]])
    dup = next(r for r in result if r.id == "dup")
    assert dup.text == "first text"
    assert dup.metadata == {"src": 1}


def test_score_is_rrf_formula() -> None:
    # single passage at rank 1 in one list: expected score = 1 / (60 + 1)
    result = reciprocal_rank_fusion([[_p("only")]], k=60)
    assert abs(result[0].score - 1.0 / 61) < 1e-9


def test_empty_rankings_returns_empty() -> None:
    assert reciprocal_rank_fusion([]) == []


def test_sorted_descending() -> None:
    # a is rank-1 in list1, b is rank-1 in list2; combined they tie,
    # but a also appears at rank 2 in list2 -- verify stable descending order
    r1 = [_p("a"), _p("b")]
    r2 = [_p("b"), _p("a")]
    result = reciprocal_rank_fusion([r1, r2])
    scores = [r.score for r in result]
    assert scores == sorted(scores, reverse=True)
