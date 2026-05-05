"""Smoke test for the dense retriever.

Builds a tiny on-disk index over three synthetic passages and verifies
that `Retriever.search` returns the topically-closest one. Skips when
the optional `[rag]` deps aren't installed -- the rest of the test
suite must still pass on a `uv sync --group dev` install.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("sentence_transformers")
pytest.importorskip("faiss")

# Imports must follow `importorskip` so a base install (no `[rag]` deps)
# skips this file cleanly instead of erroring at collection time.
import numpy as np  # noqa: E402

from polimillionaire.retrieval.embedder import Embedder  # noqa: E402
from polimillionaire.retrieval.retriever import Retriever  # noqa: E402


def _write_index(dir_path: Path, embedder: Embedder, passages: list[dict]) -> None:
    embeddings = embedder.encode([p["text"] for p in passages])
    np.save(dir_path / "embeddings.npy", embeddings)
    (dir_path / "passages.jsonl").write_text("\n".join(json.dumps(p) for p in passages))
    (dir_path / "manifest.json").write_text(
        json.dumps(
            {
                "model_name": embedder.name,
                "dim": int(embeddings.shape[1]),
                "count": len(passages),
                "dataset": "synthetic",
            }
        )
    )


def test_retriever_returns_topical_match(tmp_path: Path) -> None:
    embedder = Embedder()
    passages = [
        {"id": "p1", "text": "The capital of France is Paris.", "metadata": {}},
        {
            "id": "p2",
            "text": "Quadratic equations of the form ax^2 + bx + c = 0 "
            "have at most two real roots, found by the quadratic formula.",
            "metadata": {},
        },
        {"id": "p3", "text": "The mitochondrion produces ATP for the cell.", "metadata": {}},
    ]
    _write_index(tmp_path, embedder, passages)

    # Reuse the same embedder so we don't pay the model load cost twice.
    retriever = Retriever(tmp_path, embedder=embedder)
    hits = retriever.search("How do I solve a quadratic equation?", k=1)
    assert len(hits) == 1
    assert hits[0].id == "p2"
    assert hits[0].score > 0  # cosine similarity, normalised


def test_retriever_search_k_zero_returns_empty(tmp_path: Path) -> None:
    embedder = Embedder()
    _write_index(
        tmp_path,
        embedder,
        [{"id": "p1", "text": "anything", "metadata": {}}],
    )
    retriever = Retriever(tmp_path, embedder=embedder)
    assert retriever.search("anything", k=0) == []
