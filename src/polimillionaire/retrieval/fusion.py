"""Reciprocal Rank Fusion for combining multiple ranked retrieval lists."""

from __future__ import annotations

from polimillionaire.retrieval.retriever import Passage


def reciprocal_rank_fusion(
    rankings: list[list[Passage]],
    *,
    k: int = 60,
    top_n: int | None = None,
) -> list[Passage]:
    """Fuse multiple ranked lists into one. RRF score = sum(1/(k+rank))."""
    # rank is 1-based
    rrf_scores: dict[str, float] = {}
    # first time we see a passage id, capture its text/metadata from that list
    seen: dict[str, Passage] = {}

    for ranking in rankings:
        for rank, passage in enumerate(ranking, start=1):
            rrf_scores[passage.id] = rrf_scores.get(passage.id, 0.0) + 1.0 / (k + rank)
            if passage.id not in seen:
                seen[passage.id] = passage

    fused = sorted(rrf_scores.keys(), key=lambda pid: rrf_scores[pid], reverse=True)
    if top_n is not None:
        fused = fused[:top_n]

    return [
        Passage(
            id=pid,
            text=seen[pid].text,
            metadata=seen[pid].metadata,
            score=rrf_scores[pid],
        )
        for pid in fused
    ]
