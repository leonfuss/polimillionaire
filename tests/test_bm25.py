"""Tests for BM25Index: build, save/load roundtrip, and search correctness.

Skips when `rank_bm25` isn't installed so a base-install test run still passes.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("rank_bm25")

from polimillionaire.retrieval.bm25 import BM25Index  # noqa: E402

_PASSAGES = [
    {"id": "p1", "text": "Roll Safe is a British comedy sketch series.", "metadata": {}},
    {"id": "p2", "text": "Thanos is a fictional supervillain in Marvel Comics.", "metadata": {}},
    {"id": "p3", "text": "Kallinikos was a Byzantine architect and engineer.", "metadata": {}},
    {"id": "p4", "text": "The Eiffel Tower is located in Paris, France.", "metadata": {}},
    {"id": "p5", "text": "Photosynthesis converts sunlight into chemical energy.", "metadata": {}},
]


def test_search_returns_exact_token_match() -> None:
    idx = BM25Index.build(_PASSAGES)
    hits = idx.search("Thanos Marvel supervillain", k=1)
    assert len(hits) == 1
    assert hits[0].id == "p2"


def test_search_top_k_is_respected() -> None:
    idx = BM25Index.build(_PASSAGES)
    hits = idx.search("series sketch comedy", k=3)
    assert len(hits) == 3


def test_search_k_zero_returns_empty() -> None:
    idx = BM25Index.build(_PASSAGES)
    assert idx.search("anything", k=0) == []


def test_search_score_is_float() -> None:
    idx = BM25Index.build(_PASSAGES)
    hits = idx.search("Kallinikos Byzantine", k=1)
    assert isinstance(hits[0].score, float)


def test_save_load_roundtrip(tmp_path: Path) -> None:
    idx = BM25Index.build(_PASSAGES)
    idx.save(tmp_path)

    assert (tmp_path / "bm25_tokens.jsonl").exists()
    assert (tmp_path / "bm25.json").exists()

    loaded = BM25Index.load(tmp_path, _PASSAGES)
    original_hits = idx.search("Eiffel Tower Paris", k=3)
    loaded_hits = loaded.search("Eiffel Tower Paris", k=3)

    # rank order is preserved
    assert [h.id for h in original_hits] == [h.id for h in loaded_hits]
    # scores are numerically identical (would fail if e.g. epsilon weren't persisted)
    for a, b in zip(original_hits, loaded_hits, strict=True):
        assert abs(a.score - b.score) < 1e-9


def test_load_wrong_passage_count_raises(tmp_path: Path) -> None:
    idx = BM25Index.build(_PASSAGES)
    idx.save(tmp_path)
    with pytest.raises(ValueError, match="passages list length"):
        BM25Index.load(tmp_path, _PASSAGES[:2])


def test_load_reads_passages_jsonl_by_default(tmp_path: Path) -> None:
    """When passages aren't supplied, load() reads them from the sibling passages.jsonl."""
    import json as _json

    idx = BM25Index.build(_PASSAGES)
    idx.save(tmp_path)
    (tmp_path / "passages.jsonl").write_text("\n".join(_json.dumps(p) for p in _PASSAGES))

    loaded = BM25Index.load(tmp_path)
    hits = loaded.search("Eiffel Tower Paris", k=1)
    assert hits[0].id == "p4"
