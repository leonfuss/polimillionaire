"""BM25 sparse retrieval over tokenised passage text.

Uses bm25s under the hood (scipy CSR sparse matrices). For an 800k-passage
corpus this fits in ~1.5GB instead of the 15+ GB rank-bm25 needed for the
same list-of-list-of-Python-strings layout, which matters on Colab's
12 GB CPU runtime.

Pairs with the dense FAISS retriever via `fusion.reciprocal_rank_fusion`
for hybrid retrieval.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from polimillionaire.retrieval.retriever import Passage

if TYPE_CHECKING:
    import bm25s as _bm25s

# bm25s writes this file alongside its three .npy sidecars; we use it as
# the cheap "is the index on disk?" marker without enumerating every file.
_PARAMS_FILE = "params.index.json"

# Single tokenisation rule shared between corpus indexing and query time.
# Plain word-char split keeps exact-token matching on entities like
# "Kallinikos" or "Thanos" -- stemming would smear those.
_TOKEN_PATTERN = r"\w+"


def _tokenize(text: str) -> list[str]:
    return re.findall(_TOKEN_PATTERN, text.lower())


class BM25Index:
    """BM25 index over a list of passages, backed by bm25s."""

    def __init__(self, bm25: _bm25s.BM25, passages: list[dict[str, Any]]) -> None:
        self._bm25 = bm25
        self._passages = passages

    @classmethod
    def build(cls, passages: list[dict[str, Any]]) -> BM25Index:
        """Build an in-memory index from a list of passage dicts.

        Goes straight from str to int-encoded tokens via bm25s.tokenize so
        we never hold the full corpus as list[list[str]] of Python strings.
        """
        import bm25s

        tokens = bm25s.tokenize(
            [p["text"] for p in passages],
            lower=True,
            token_pattern=_TOKEN_PATTERN,
            stopwords=None,
            show_progress=True,
        )
        bm25 = bm25s.BM25()
        bm25.index(tokens, show_progress=True)
        return cls(bm25, list(passages))

    def save(self, index_dir: Path) -> None:
        """Write bm25s sidecar files into `index_dir`.

        Writes: data.csc.index.npy, indices.csc.index.npy,
        indptr.csc.index.npy, params.index.json, vocab.index.json.
        """
        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)
        self._bm25.save(str(index_dir))

    @classmethod
    def load(cls, index_dir: Path, passages: list[dict[str, Any]] | None = None) -> BM25Index:
        """Reload a previously-saved bm25s index.

        Reads `passages.jsonl` from `index_dir` by default so the same
        on-disk layout serves both dense retrieval and BM25. Tests can
        pass an explicit `passages` list to skip the file read.
        """
        import bm25s

        index_dir = Path(index_dir)
        params_path = index_dir / _PARAMS_FILE
        if not params_path.exists():
            raise FileNotFoundError(f"bm25 index at {index_dir} is missing {_PARAMS_FILE}")

        if passages is None:
            passages_path = index_dir / "passages.jsonl"
            if not passages_path.exists():
                raise FileNotFoundError(
                    f"bm25 index at {index_dir} expects passages.jsonl alongside it"
                )
            passages = [
                json.loads(line) for line in passages_path.read_text().splitlines() if line.strip()
            ]

        bm25 = bm25s.BM25.load(str(index_dir), load_corpus=False)

        # Cross-check that the passage list lines up with the indexed corpus
        # so search() can't return stale ids if passages.jsonl was rewritten.
        expected = int(json.loads(params_path.read_text())["num_docs"])
        if len(passages) != expected:
            raise ValueError(f"passages list length {len(passages)} != index num_docs {expected}")
        return cls(bm25, list(passages))

    def search(self, query: str, k: int) -> list[Passage]:
        """Return top-k passages by BM25 score (raw, not normalised)."""
        if k <= 0:
            return []
        q_tokens = [_tokenize(query)]
        results, scores = self._bm25.retrieve(q_tokens, k=k, show_progress=False)
        out: list[Passage] = []
        for idx, score in zip(results[0], scores[0], strict=False):
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
