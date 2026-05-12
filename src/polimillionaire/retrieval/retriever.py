"""Dense retriever over a saved-on-disk index directory.

Index layout (everything under one directory):

    manifest.json   {"model_name": str, "dim": int, "count": int, "dataset": str, ...}
    embeddings.npy  (N, dim) float32, rows L2-normalised
    passages.jsonl  one JSON object per row: {id, text, metadata}

The FAISS index is rebuilt in memory at load time rather than serialised
to disk -- 12k vectors rebuild in <1s, and the `.npy` form is portable
across faiss versions and easy to inspect.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from polimillionaire.retrieval.embedder import Embedder


@dataclass(frozen=True)
class Passage:
    """One retrieval result. `score` is cosine similarity in [-1, 1]."""

    id: str
    text: str
    metadata: dict[str, Any]
    score: float


class Retriever:
    """Loads an on-disk index and serves nearest-neighbour search."""

    def __init__(self, index_dir: Path | str, *, embedder: Embedder | None = None) -> None:
        index_dir = Path(index_dir)
        if not index_dir.exists():
            raise FileNotFoundError(f"no index at {index_dir}")

        manifest_path = index_dir / "manifest.json"
        embeddings_path = index_dir / "embeddings.npy"
        passages_path = index_dir / "passages.jsonl"
        for p in (manifest_path, embeddings_path, passages_path):
            if not p.exists():
                raise FileNotFoundError(f"index at {index_dir} is missing {p.name}")

        self.dir = index_dir
        self.manifest = json.loads(manifest_path.read_text())
        # Either reuse a caller-supplied embedder (lets multiple retrievers
        # share one model handle) or instantiate one matching the manifest.
        self.embedder = embedder or Embedder(self.manifest["model_name"])

        # mmap the embeddings so the OS pages them in lazily. FAISS still
        # copies into its own internal storage during add(), but we never
        # also hold a long-lived numpy copy of the full array -- on a
        # 794k x 768 fp32 index that's a 2.3 GB CPU-RAM win.
        embeddings = np.load(embeddings_path, mmap_mode="r")
        if embeddings.ndim != 2:
            raise ValueError(f"embeddings.npy must be 2D, got shape {embeddings.shape}")
        if embeddings.shape[1] != self.manifest["dim"]:
            raise ValueError(
                f"manifest dim {self.manifest['dim']} != embeddings dim {embeddings.shape[1]}"
            )
        n, dim = int(embeddings.shape[0]), int(embeddings.shape[1])

        self._passages: list[dict[str, Any]] = [
            json.loads(line) for line in passages_path.read_text().splitlines() if line.strip()
        ]
        if len(self._passages) != n:
            raise ValueError(f"passages count {len(self._passages)} != embeddings count {n}")

        # Inner product on L2-normalised vectors == cosine similarity. FAISS
        # needs a contiguous fp32 view; cast lazily and drop the reference
        # right after add() so the mmap pages can be reclaimed.
        import faiss

        self._faiss = faiss.IndexFlatIP(dim)
        self._faiss.add(np.ascontiguousarray(embeddings, dtype=np.float32))
        del embeddings

    def __len__(self) -> int:
        return len(self._passages)

    def search(self, query: str, k: int = 5) -> list[Passage]:
        """Return up to `k` passages closest to `query` by cosine similarity."""
        if k <= 0:
            return []
        q = self.embedder.encode([query])
        scores, indices = self._faiss.search(q, k)
        out: list[Passage] = []
        for score, idx in zip(scores[0], indices[0], strict=True):
            if idx == -1:  # FAISS sentinel when fewer than k results exist.
                continue
            row = self._passages[int(idx)]
            out.append(
                Passage(
                    id=row["id"],
                    text=row["text"],
                    metadata=row.get("metadata", {}),
                    score=float(score),
                )
            )
        return out
